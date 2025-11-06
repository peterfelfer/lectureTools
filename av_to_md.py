#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mkv_to_markdown.py — batch-capable ASR + notes generator

Transcribes MKV/MP4/MP3/WAV into:
  1) Full transcript as Markdown (one segment per line, timestamps when available via whisper-1)
  2) Cleaned lecture notes / summary as Markdown (~ --summary-words, same language by default)

Batch modes:
  • --input-dir <DIR>  → process all *.mkv in that folder (non-recursive) (alias --folder-in)
  • --input-list <TXT> → process the list of files in a text file (one path per line) (alias --list-in)
       ⤷ also writes a results mapping file (--output-list, default: <input-list>.transcribed.txt; alias --list-out)
  • --input-file <FILE> → single file (preferred; positional input still accepted but deprecated) (alias --mkv-in)

Key safeguards for correct notes:
  • Summarization re-reads the just-written .full.md
  • Deterministic params (temperature=0), extractive instructions
  • Optional sanity check for domain terms (retry once if misaligned)

Other features:
  • ffmpeg auto-detection (system or imageio-ffmpeg fallback) + optional ffprobe duration
  • Robust retries on API errors (429/5xx) + helpful message for 413 size errors
  • Chunked transcription (default 20 min) with optional progress bar (--progress)
  • Handles models without timestamps (gpt-4o(-mini)-transcribe) by producing text only
  • Summary language control (same/auto/explicit), and --summary-model selector

Outputs per input:
  • <input>.full.md  – full transcript
  • <input>.md       – notes/summary
  • <debug-dir>/transcription.json (if --debug-dir)
If --input-list used:
  • <input-list>.transcribed.txt (or --output-list) with tab-separated: INPUT  NOTES_MD  FULL_MD
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from hashlib import sha256
from typing import List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".defaults.yaml")


def load_yaml_defaults(path: Path) -> dict:
    if yaml is None:
        return {}
    try:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - configuration errors should surface quickly
        raise RuntimeError(f"Failed to read YAML defaults from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML defaults at {path} must be a mapping, got {type(data).__name__}")
    return data

from openai import OpenAI
from openai._exceptions import APIStatusError, APIConnectionError, APITimeoutError

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ========== ANSI colors & logging ==========
def _supports_color():
    return sys.stderr.isatty()

COLOR = _supports_color()
def c(s, code): return f"\033[{code}m{s}\033[0m" if COLOR else s
def INF(s):  return c(s, "36")
def OK(s):   return c(s, "32")
def WARN(s): return c(s, "33")
def ERR(s):  return c(s, "31")
def BOLD(s): return c(s, "1")

class Log:
    def __init__(self, verbose=False, quiet=False, prefix=""):
        self.verbose = verbose
        self.quiet = quiet
        self._t0 = time.time()
        self.prefix = prefix  # allows per-file prefixes in batch
    def _stamp(self): return f"[{timedelta(seconds=int(time.time()-self._t0))}]"
    def _fmt(self, level, msg): return f"{self._stamp()} {level} {self.prefix}{msg}"
    def info(self, msg):  print(self._fmt(INF("[INFO]"),  msg), file=sys.stderr) if not self.quiet else None
    def step(self, msg):  print("\n" + self._fmt(BOLD("[STEP]"), msg), file=sys.stderr) if not self.quiet else None
    def ok(self, msg):    print(self._fmt(OK("[OK]"),    "  " + msg), file=sys.stderr) if not self.quiet else None
    def warn(self, msg):  print(self._fmt(WARN("[WARN]"), msg), file=sys.stderr) if not self.quiet else None
    def error(self, msg): print(self._fmt(ERR("[ERROR]"), msg), file=sys.stderr)
    def dbg(self, msg):   print(self._fmt("[DBG] ",       msg), file=sys.stderr) if self.verbose and not self.quiet else None
    def with_prefix(self, pfx):  # create a derived logger with a prefix
        return Log(self.verbose, self.quiet, prefix=pfx)

# ========== ffmpeg / ffprobe ==========
def find_ffmpeg_exe(log: Log):
    p = shutil.which("ffmpeg")
    if p:
        log.dbg(f"system ffmpeg: {p}")
        return p
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            log.dbg(f"imageio-ffmpeg: {p}")
            return p
    except Exception:
        pass
    return None

def find_ffprobe_exe():
    return shutil.which("ffprobe")  # optional

# ========== utilities ==========
def run(cmd, log: Log, show_stderr=False):
    log.dbg(f"exec: {cmd}")
    proc = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=(None if show_stderr else subprocess.PIPE),
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n\nSTDERR:\n{proc.stderr}")
    return proc.stdout

def ffprobe_duration_seconds(path: Path, log: Log) -> float:
    exe = find_ffprobe_exe()
    if not exe:
        log.warn("ffprobe not found; skipping duration probe.")
        return 0.0
    try:
        out = run(f'"{exe}" -v error -show_entries format=duration -of default=nw=1:nk=1 "{path}"', log)
        return float(out.strip())
    except Exception:
        return 0.0

def hhmmss(seconds: float) -> str:
    try: return str(timedelta(seconds=int(seconds)))
    except Exception: return "00:00:00"

def format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"

VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".m4v", ".webm"}

_LAST_RESULTS: List[Tuple[bool, Path, Optional[Path], Optional[Path]]] = []
_LAST_JOBS: List[List[Path]] = []


def get_last_results() -> List[Tuple[bool, Path, Optional[Path], Optional[Path]]]:
    return _LAST_RESULTS


def get_last_jobs() -> List[List[Path]]:
    return _LAST_JOBS


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".webm"}

def ensure_unique_path(path: Path) -> Path:
    """Return a path variant that does not overwrite existing files."""
    candidate = path
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({idx}){path.suffix}")
        idx += 1
    return candidate


def serialize_concat_list(audio_parts: Sequence[Path], concat_file: Path) -> None:
    lines = []
    for path in audio_parts:
        escaped = str(path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    concat_file.write_text("".join(lines), encoding="utf-8")


def build_concat_command(codec: str, bitrate: str) -> str:
    codec = codec.lower()
    if codec == "wav":
        return "-c:a pcm_s16le"
    if codec == "flac":
        return "-c:a flac"
    if codec == "aac":
        return f"-c:a aac -b:a {bitrate}"
    return f"-c:a libmp3lame -b:a {bitrate}"

def extract_audio(input_path: Path, out_dir: Path, codec: str, bitrate: str,
                  log: Log, show_ffmpeg=False) -> Path:
    ffmpeg_exe = find_ffmpeg_exe(log)
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg executable not found. Install Homebrew ffmpeg or `pip install imageio-ffmpeg`.")
    out_path = out_dir / f"{input_path.stem}.{codec}"
    if codec == "wav":
        cmd = f'"{ffmpeg_exe}" -y -i "{input_path}" -vn -ac 1 -ar 16000 "{out_path}"'
    elif codec == "flac":
        cmd = f'"{ffmpeg_exe}" -y -i "{input_path}" -vn -ac 1 -ar 16000 -c:a flac "{out_path}"'
    elif codec == "aac":
        cmd = f'"{ffmpeg_exe}" -y -i "{input_path}" -vn -c:a aac -b:a {bitrate} "{out_path}"'
    else:
        cmd = f'"{ffmpeg_exe}" -y -i "{input_path}" -vn -c:a libmp3lame -b:a {bitrate} "{out_path}"'
    log.info(f"Using ffmpeg at: {ffmpeg_exe}")
    run(cmd, log, show_stderr=show_ffmpeg)
    return out_path

def split_audio_chunks(audio_path: Path, chunk_secs: int, out_dir: Path, codec: str, bitrate: str, log: Log):
    """Split audio into ~chunk_secs segments using ffmpeg segment muxer (re-encode for predictable sizes)."""
    ffmpeg_exe = find_ffmpeg_exe(log)
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg executable not found for chunking.")
    stem = audio_path.stem
    pattern = out_dir / f"{stem}_part_%03d.{codec}"
    if codec == "wav":
        cmd = f'"{ffmpeg_exe}" -y -i "{audio_path}" -f segment -segment_time {chunk_secs} -ac 1 -ar 16000 "{pattern}"'
    elif codec == "flac":
        cmd = f'"{ffmpeg_exe}" -y -i "{audio_path}" -f segment -segment_time {chunk_secs} -ac 1 -ar 16000 -c:a flac "{pattern}"'
    elif codec == "aac":
        cmd = f'"{ffmpeg_exe}" -y -i "{audio_path}" -f segment -segment_time {chunk_secs} -c:a aac -b:a {bitrate} "{pattern}"'
    else:  # mp3
        cmd = f'"{ffmpeg_exe}" -y -i "{audio_path}" -f segment -segment_time {chunk_secs} -c:a libmp3lame -b:a {bitrate} "{pattern}"'
    run(cmd, log)
    parts = sorted(out_dir.glob(f"{stem}_part_*.{codec}"))
    if not parts:
        raise RuntimeError("No chunks were produced; check ffmpeg support.")
    return parts


def prepare_combined_audio(
    sources: Sequence[Path],
    args,
    base_log: Log,
) -> Tuple[Path, tempfile.TemporaryDirectory]:
    """Extract audio for each source and concatenate in chronological order."""
    if not sources:
        raise ValueError("No sources provided for audio combination.")

    combo_log = base_log.with_prefix("[COMBINE] ")
    combo_log.step("Prepare concatenated audio")

    temp_dir_ctx = tempfile.TemporaryDirectory()
    work_dir = Path(temp_dir_ctx.name)

    ffmpeg_exe = find_ffmpeg_exe(combo_log)
    if not ffmpeg_exe:
        temp_dir_ctx.cleanup()
        raise RuntimeError("ffmpeg executable not found. Cannot combine audio files.")

    audio_parts: List[Path] = []
    for idx, src in enumerate(sources, start=1):
        part_dir = work_dir / f"part_{idx:03d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_log = combo_log.with_prefix(f"[{src.name}] ")
        part_log.info(f"Extracting audio for concatenation.")
        try:
            audio_path = extract_audio(
                src,
                part_dir,
                args.audio_codec,
                args.bitrate,
                part_log,
                args.show_ffmpeg,
            )
            audio_parts.append(audio_path)
        except Exception as exc:
            temp_dir_ctx.cleanup()
            raise RuntimeError(f"Failed to extract audio from {src}: {exc}") from exc

    concat_file = work_dir / "concat.txt"
    serialize_concat_list(audio_parts, concat_file)
    combined_audio = work_dir / f"combined.{args.audio_codec}"
    encode_flags = build_concat_command(args.audio_codec, args.bitrate)
    cmd = f'"{ffmpeg_exe}" -y -f concat -safe 0 -i "{concat_file}" {encode_flags} "{combined_audio}"'
    combo_log.info("Concatenating audio segments.")
    try:
        run(cmd, combo_log, show_stderr=args.show_ffmpeg)
    except Exception as exc:
        temp_dir_ctx.cleanup()
        raise RuntimeError(f"Audio concatenation failed: {exc}") from exc

    combo_log.ok(f"Combined audio → {combined_audio}")
    return combined_audio, temp_dir_ctx

def sanitize_markdown(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def offset_segments(segments, offset_s):
    out = []
    for s in segments:
        ss = dict(s)
        if "start" in ss: ss["start"] = float(ss["start"]) + offset_s
        if "end" in ss:   ss["end"]   = float(ss["end"]) + offset_s
        out.append(ss)
    return out

def words_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def compute_fingerprint(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

# ========== language helpers ==========
def pick_summary_language(user_flag: str, detected: str, fallback_from_text: str) -> str:
    if user_flag and user_flag not in ("same", "auto"):
        return user_flag
    if user_flag == "auto":
        return "auto"
    if detected:
        return detected
    return fallback_from_text or "auto"

def guess_german(text: str) -> bool:
    if re.search(r"[äöüÄÖÜß]", text): return True
    if re.search(r"\b(und|oder|nicht|für|eine|einer|einem|ist|sind|werden|wird|Koks|Hochofen|Erz|Schlacke|Reduktionszone|Boudouard)\b", text, flags=re.I):
        return True
    return False

# ========== OpenAI: transcription ==========
def transcribe_raw(client: OpenAI, audio_path: Path, model: str, language: str, prompt: str):
    """
    Handle API differences: older Whisper supports 'verbose_json' + timestamps;
    newer 4o transcribe models may require 'json' and omit timestamps.
    """
    with open(audio_path, "rb") as f:
        use_verbose = not any(x in model for x in ["-api-", "transcribe-api", "transcribe-ev3"])
        kwargs = {
            "model": model,
            "file": f,
            "response_format": "verbose_json" if use_verbose else "json",
        }
        if language: kwargs["language"] = language
        if prompt:   kwargs["prompt"]   = prompt
        try:
            if use_verbose:
                kwargs["timestamp_granularities"] = ["segment"]
            return client.audio.transcriptions.create(**kwargs)
        except Exception as e:
            if "response_format" in str(e):
                kwargs["response_format"] = "json"
                kwargs.pop("timestamp_granularities", None)
                return client.audio.transcriptions.create(**kwargs)
            raise

def transcribe_with_retry(client: OpenAI, audio_path: Path, model: str, language: str, prompt: str,
                          log: Log, max_retries=5, backoff_start=3.0, backoff_factor=2.0):
    attempt = 0
    wait = backoff_start
    while True:
        attempt += 1
        try:
            log.info(f"Transcribing chunk: attempt {attempt}")
            resp = transcribe_raw(client, audio_path, model, language, prompt)
            return json.loads(resp.model_dump_json())
        except (APIStatusError, APIConnectionError, APITimeoutError) as e:
            code = getattr(e, "status_code", None)
            if isinstance(e, APIStatusError): code = e.status_code
            if code == 413 or "Maximum content size limit" in str(e):
                log.error(
                    "Upload chunk exceeded the server size limit.\n"
                    "Tip: re-run with smaller chunks and/or lower bitrate, e.g.:\n"
                    "  --chunk-mins 10 --bitrate 96k   (or 8 mins / 64k if needed)"
                )
                raise
            retriable = (code in (429, 500, 502, 503, 504)) or isinstance(e, (APIConnectionError, APITimeoutError))
            if attempt < max_retries and retriable:
                log.warn(f"API error (code={code}): {e}\nRetrying in {int(wait)}s …")
                time.sleep(wait); wait *= backoff_factor; continue
            log.error(f"Transcription failed after {attempt} attempt(s): {e}")
            raise

def transcribe_with_timestamps(client: OpenAI, audio_path: Path, model: str,
                               language: str, prompt: str, log: Log,
                               chunk_secs: int = None, max_retries=5,
                               backoff_start=3.0, backoff_factor=2.0,
                               show_progress=False):
    """
    Transcribe single file, optionally chunked. Returns merged dict:
      'text' (concatenated), 'segments' (may be empty), 'language' (if API gives it).
    """
    if not chunk_secs:
        return transcribe_with_retry(client, audio_path, model, language, prompt, log,
                                     max_retries, backoff_start, backoff_factor)

    with tempfile.TemporaryDirectory() as ctd:
        cdir = Path(ctd)
        codec = audio_path.suffix.lower().lstrip(".") or "mp3"
        if codec not in ("mp3", "wav", "flac", "aac"): codec = "mp3"
        parts = split_audio_chunks(audio_path, chunk_secs, cdir, codec, "96k", log)

        kbps = 96.0
        est_mb = (kbps * 1000 * chunk_secs) / 8 / 1e6
        log.info(f"Estimated per-chunk size at {int(kbps)} kbps and {chunk_secs//60} min: ~{est_mb:.1f} MB")

        merged_text, merged_segments, offset = [], [], 0.0
        reported_lang = None

        total_parts = len(parts)
        progress = None
        start_time = None
        if show_progress and tqdm is not None and total_parts:
            progress = tqdm(total=total_parts, unit="chunk", desc="Transcribing")
            start_time = time.time()

        for i, part in enumerate(parts, start=1):
            if progress is None and show_progress:
                log.info(f"Transcribing chunk {i}/{total_parts} …")
            data = transcribe_with_retry(client, part, model, language, prompt, log,
                                         max_retries, backoff_start, backoff_factor)
            segs = data.get("segments") or []
            txt  = data.get("text") or " ".join(s.get("text","") for s in segs)
            merged_text.append(txt)
            merged_segments.extend(offset_segments(segs, offset))
            if (not reported_lang) and isinstance(data, dict):
                reported_lang = data.get("language") or reported_lang
            part_dur = ffprobe_duration_seconds(part, log)
            offset += part_dur if part_dur > 0 else (len(txt.split())/2.5)

            if progress is not None:
                progress.update(1)
                elapsed = time.time() - start_time
                remaining = (elapsed / progress.n) * max(total_parts - progress.n, 0)
                progress.set_postfix_str(f"ETA {format_eta(remaining)}")

        out = {"text": " ".join(merged_text).strip(), "segments": merged_segments}
        if reported_lang:
            out["language"] = reported_lang
        if progress is not None:
            progress.set_postfix_str("ETA 0s")
            progress.close()
        return out

# ========== rendering ==========
def sanitize_markdown(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_full_markdown(segments: list, raw_text: str) -> str:
    if segments:
        lines = []
        for s in segments:
            t = s.get("text", "").strip()
            if not t: continue
            ts = hhmmss(float(s.get("start", 0.0))) if "start" in s else "00:00:00"
            lines.append(f"[{ts}] {t}")
        return "# Full Transcript\n\n" + "\n\n".join(lines) + "\n"
    return "# Full Transcript\n\n" + sanitize_markdown(raw_text) + "\n"

# ========== OpenAI: summarization ==========
def ask_model_for_notes(client: OpenAI, full_text: str, segments: list,
                        domain_hint: str, target_level: str, target_words: int,
                        summary_language: str, summary_model: str, log: Log):
    have_timestamps = bool(segments)
    compact = [{"start": s.get("start", 0.0), "text": s.get("text", "")}
               for s in segments if isinstance(s, dict)][:2000]

    if summary_language and summary_language not in ("same", "auto"):
        lang_directive = f"Write the notes in **{summary_language}**.\n"
    elif summary_language == "same":
        lang_directive = "Write the notes in the **exact same language as the transcript**; do not translate.\n"
    else:
        lang_directive = "Use the most likely language of the transcript; do not translate unless unavoidable.\n"

    system_prompt = (
        "You are a scientific editor. Produce accurate, well-structured lecture notes from the transcript ONLY. "
        "Do not invent or import content from outside the transcript; if key details are missing, explicitly note the gaps within that section rather than omitting the section.\n"
        + lang_directive +
        f"Aim for ~{target_words} words (±10%). Keep technical accuracy; remove filler; prefer short paragraphs and bullet lists. "
        + ("For each section heading, append a single [HH:MM:SS] from the earliest relevant segment."
           if have_timestamps else "Do not add timestamps if none are available.")
    )

    topic_hint = domain_hint if domain_hint else "the lecture topic"
    guide = (
        "Suggested outline: Overview; Key Concepts; Methods or Derivations; "
        f"Illustrative Examples or Case Studies related to {topic_hint}; "
        "Common Pitfalls or Limitations; Practical Implications; Summary & Key Takeaways."
    )

    fingerprint = sha256(full_text[:5000].encode("utf-8", errors="ignore")).hexdigest()[:16]
    user_prompt = (
        f"[TRANSCRIPT_FINGERPRINT:{fingerprint}]\n"
        f"DOMAIN HINT: {domain_hint}\n"
        f"TARGET LEVEL: {target_level}\n"
        f"OUTLINE HINT: {guide}\n\n"
        f"TRANSCRIPT TEXT (begin):\n{full_text[:120000]}\n\n"
        "TASK: Output ONLY the cleaned Markdown lecture notes. "
        "Use clear section headings (##). If timestamps are available, format them like [HH:MM:SS] at the end of the heading."
    )

    chat = client.chat.completions.create(
        model=summary_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=1 if summary_model and "gpt-5" in summary_model.lower() else 0,
        top_p=1,
    )
    return sanitize_markdown(chat.choices[0].message.content)

def sanity_check_alignment(transcript_text: str, notes_text: str) -> bool:
    probes = ["Hochofen", "Koks", "Erz", "Schlacke", "Boudouard", "Reduktionszone", "Blast Furnace", "Coke", "Basicity"]
    hits = sum(1 for w in probes if re.search(rf"\b{re.escape(w)}\b", notes_text, flags=re.I))
    return hits >= 2

# ========== single-file processing ==========
def process_one(
    in_path: Path,
    args,
    base_log: Log,
    client: OpenAI,
    pre_extracted_audio: Path | None = None,
):
    log = base_log.with_prefix(f"[{in_path.name}] ")

    if not in_path.exists():
        log.error(f"Input not found: {in_path}")
        return False, in_path, None, None

    # Output paths
    notes_md_path = Path(args.output_file).expanduser().resolve() if args.output_file else in_path.with_suffix(".md")
    full_md_path  = in_path.with_suffix(".full.md")

    # Report
    log.step(f"Input")
    log.info(f"Path: {in_path}")
    try:
        log.info(f"Size: {in_path.stat().st_size/1e6:.2f} MB")
    except Exception:
        pass
    dur = ffprobe_duration_seconds(in_path, log)
    if dur: log.info(f"Duration: {hhmmss(dur)}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Extract or use audio
        source_is_audio = is_audio_file(in_path)
        if pre_extracted_audio is not None:
            audio_path = pre_extracted_audio
            log.info("Using pre-combined audio input.")
        elif source_is_audio:
            audio_path = in_path
            log.info("Using existing audio file.")
        else:
            log.step("Extract audio with ffmpeg")
            try:
                audio_path = extract_audio(in_path, td, args.audio_codec, args.bitrate, log, args.show_ffmpeg)
                log.ok(f"Extracted audio → {audio_path.name}")
            except Exception as e:
                log.error(f"Audio extraction failed: {e}")
                return False, in_path, None, None

        # Transcribe
        chunk_secs = None if args.no_chunk else int(args.chunk_mins * 60)
        log.step("Transcription")
        t0 = time.time()
        try:
            tdata = transcribe_with_timestamps(
                client, audio_path, args.model, args.language, args.asr_prompt, log,
                chunk_secs=chunk_secs, max_retries=args.max_retries,
                backoff_start=args.retry_initial_wait, backoff_factor=args.retry_backoff,
                show_progress=args.progress
            )
            log.ok(f"Transcribed in {hhmmss(time.time()-t0)}")
        except Exception as e:
            log.error(f"Transcription error: {e}")
            return False, in_path, None, None

        if args.debug_dir:
            dbg_dir = Path(args.debug_dir).expanduser().resolve()
            dbg_dir.mkdir(parents=True, exist_ok=True)
            (dbg_dir / f"{in_path.stem}.transcription.json").write_text(json.dumps(tdata, indent=2), encoding="utf-8")

        segments = tdata.get("segments") or []
        raw_text = tdata.get("text") or " ".join(s.get("text","") for s in segments)
        if not raw_text.strip():
            log.error("Empty transcript.")
            return False, in_path, None, None

        # Determine language for notes
        detected_lang = tdata.get("language") or ("de" if guess_german(raw_text) else "")
        chosen_lang   = pick_summary_language(args.summary_language, detected_lang, ("de" if guess_german(raw_text) else ""))

        # Save FULL transcript
        log.step("Write FULL transcript")
        full_md = build_full_markdown(segments, raw_text)
        try:
            full_md_path.write_text(full_md, encoding="utf-8")
            log.ok(f"Full transcript → {full_md_path}")
        except Exception as e:
            log.error(f"Failed writing full transcript: {e}")
            return False, in_path, None, None

        # Re-read the exact text to summarize
        try:
            full_text_for_notes = full_md_path.read_text(encoding="utf-8")
        except Exception as e:
            log.error(f"Failed reading back full transcript: {e}")
            return False, in_path, None, None

        # Build NOTES / summary
        log.step("Build NOTES / summary")
        try:
            notes_md = ask_model_for_notes(
                client=client,
                full_text=full_text_for_notes,
                segments=segments,
                domain_hint=args.domain_hint,
                target_level=args.target_level,
                target_words=args.summary_words,
                summary_language=chosen_lang,
                summary_model=args.summary_model,
                log=log
            )
            if not sanity_check_alignment(full_text_for_notes, notes_md):
                log.warn("Notes may be misaligned with the transcript; retrying with stricter instructions.")
                notes_md = ask_model_for_notes(
                    client=client,
                    full_text=full_text_for_notes,
                    segments=segments,
                    domain_hint=args.domain_hint,
                    target_level=args.target_level,
                    target_words=args.summary_words,
                    summary_language=chosen_lang,
                    summary_model=args.summary_model,
                    log=log
                )
            notes_md_path.write_text(notes_md, encoding="utf-8")
            log.ok(f"Notes/summary → {notes_md_path}")
        except Exception as e:
            log.error(f"Failed to build/write notes: {e}")
            return False, in_path, None, None

        # Optionally keep audio
        if (pre_extracted_audio is None) and (not source_is_audio) and args.keep_audio:
            final_audio = in_path.with_suffix(f".{args.audio_codec}")
            try:
                shutil.move(str(audio_path), final_audio)
                log.ok(f"Kept audio → {final_audio}")
            except Exception as e:
                log.warn(f"Could not keep audio: {e}")

    return True, in_path, notes_md_path, full_md_path


def process_sequence(
    sources: Sequence[Path],
    args,
    base_log: Log,
    client: OpenAI,
) -> Tuple[bool, Path, Path | None, Path | None]:
    ordered = sorted(sources, key=lambda p: p.stat().st_mtime if p.exists() else float("inf"))
    primary = ordered[0]
    multi_log = base_log.with_prefix(f"[MULTI:{primary.stem}] ")
    multi_log.step(f"Combining {len(ordered)} recordings for {primary.name}")
    for src in ordered:
        multi_log.info(f" - {src}")
    try:
        combined_audio, temp_dir_ctx = prepare_combined_audio(ordered, args, base_log)
    except Exception as exc:
        multi_log.error(str(exc))
        return False, primary, None, None

    try:
        result = process_one(primary, args, base_log, client, pre_extracted_audio=combined_audio)
        ok = result[0]
        if ok and args.keep_audio:
            dest_audio = primary.with_suffix(f".{args.audio_codec}")
            try:
                shutil.move(str(combined_audio), dest_audio)
                multi_log.ok(f"Kept combined audio → {dest_audio}")
            except Exception as exc:
                multi_log.warn(f"Could not keep combined audio: {exc}")
        return result
    finally:
        temp_dir_ctx.cleanup()

# ========== main / batch orchestration ==========
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description="Transcribe MKV/MP4/MP3/WAV into full transcript + length-controlled notes (Markdown)."
    )
    p.add_argument("--config-yaml", dest="config_yaml", help="Path to YAML file providing default arguments.")

    # Batch inputs
    p.add_argument("--input-file", "--mkv-in", dest="input_file", help="Path to a single media file (.mkv/.mp4/.mp3/.wav/...) (--mkv-in is deprecated).")
    p.add_argument(
        "--input-files",
        dest="input_files",
        nargs="+",
        help="Treat the given media files as a single lecture recording (concatenated in chronological order).",
    )
    p.add_argument(
        "--in",
        dest="cli_inputs",
        nargs="+",
        help="One or more media files (oldest first when multiple); shorthand for --input-file/--input-files.",
    )
    p.add_argument("--input-dir", "--folder-in", dest="input_dir", help="Folder containing media files (non-recursive). (--folder-in is deprecated)")
    p.add_argument("--input-list", "--list-in", dest="input_list", help="Text file with one media path per line. (--list-in is deprecated)")
    p.add_argument("--output-list", "--list-out", dest="output_list", help="Where to write the results mapping (default: <input-list>.transcribed.txt). (--list-out is deprecated)")

    # Legacy positional (deprecated)
    p.add_argument("input", nargs="?", help="(Deprecated) Positional single input. Prefer --input-file/--input-dir/--input-list.")

    # Output (applies to single-file mode only; batch will ignore -o and write next to each input)
    p.add_argument("-O", "--output-file", "-o", "--output", "--out", dest="output_file", help="Output Markdown path for NOTES (single-file mode only). (-o/--output are deprecated)")

    # Transcription
    p.add_argument("--model", default="gpt-4o-mini-transcribe", help="ASR model (e.g., gpt-4o-mini-transcribe, gpt-4o-transcribe, or whisper-1 for timestamps).")
    p.add_argument("--language", help="ISO-639-1 language code hint (e.g. de, en). If omitted, auto-detect.")
    p.add_argument("--asr-prompt", "--prompt", dest="asr_prompt", default="Lecture on iron and steelmaking (blast furnace & iron ore).", help="Context prompt for ASR. (--prompt is deprecated)")

    # Chunking & retries
    p.add_argument("--chunk-mins", type=int, default=20, help="Chunk length in minutes (default 20). Set --no-chunk to disable.")
    p.add_argument("--no-chunk", action="store_true", help="Disable chunked transcription (single-shot).")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries per API call.")
    p.add_argument("--retry-initial-wait", type=float, default=3.0, help="Initial backoff wait seconds.")
    p.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier.")

    # Audio extraction
    p.add_argument("--audio-codec", default="mp3", choices=["mp3","wav","flac","aac"], help="Codec for extracted audio.")
    p.add_argument("--bitrate", default="96k", help="Bitrate for chunking/extraction (default 96k; keeps uploads small).")
    p.add_argument("--keep-audio", action="store_true", help="Keep extracted audio next to input(s).")

    # Notes / summary
    p.add_argument("--summary-words", type=int, default=2000, help="Target words for notes (~±10%).")
    p.add_argument("--summary-language", default="same", help="Notes language: 'same' (default), 'auto', or a code like 'de','en'.")
    p.add_argument("--summary-model", default="gpt-5", help="Model used for summarizing/cleanup (e.g., gpt-5).")
    p.add_argument("--domain-hint", default="Iron & steelmaking; Blast furnace; Iron ore chemistry.", help="Domain hint for sectioning.")
    p.add_argument("--target-level", default="Advanced undergraduate/early graduate lecture notes.", help="Target level for the notes.")

    # Visibility / debug
    p.add_argument("--progress", action="store_true", help="Show a progress bar for chunked transcription.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--show-ffmpeg", action="store_true")
    p.add_argument("--debug-dir", help="Save raw JSON/intermediates here")
    p.add_argument("--dry-run", action="store_true", help="Print planned steps then exit")
    p.add_argument(
        "--combine-transcriptions",
        action="store_true",
        help="After processing, merge all successful transcripts into one Markdown file and create a combined AI summary."
    )

    base_defaults = load_yaml_defaults(DEFAULT_CONFIG_PATH)
    if base_defaults:
        p.set_defaults(**base_defaults)

    known_args, _ = p.parse_known_args(argv)
    if known_args.config_yaml:
        if yaml is None:
            p.error("PyYAML is required to use --config-yaml.")
        config_path = Path(known_args.config_yaml).expanduser()
        if not config_path.exists():
            p.error(f"Config YAML not found: {config_path}")
        data = load_yaml_defaults(config_path)
        p.set_defaults(**data)

    args = p.parse_args(argv)
    base_log = Log(args.verbose, args.quiet)

    global _LAST_RESULTS, _LAST_JOBS
    _LAST_RESULTS = []
    _LAST_JOBS = []

    if getattr(args, "cli_inputs", None):
        cli_list = args.cli_inputs
        if len(cli_list) == 1:
            if not args.input_file:
                args.input_file = cli_list[0]
        else:
            merged = list(args.input_files) if args.input_files else []
            merged.extend(cli_list)
            args.input_files = merged
        args.cli_inputs = None

    if getattr(args, "title", None) and (not args.domain_hint or args.domain_hint == "materials science"):
        args.domain_hint = args.title

    # API key
    if not os.getenv("OPENAI_API_KEY"):
        base_log.error("Please set OPENAI_API_KEY environment variable.")
        sys.exit(2)
    client = OpenAI()

    # Resolve batch inputs
    inputs = []
    multi_groups: List[List[Path]] = []
    mode = None
    if args.input_dir:
        mode = "folder"
        folder = Path(args.input_dir).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            base_log.error(f"--input-dir not a directory: {folder}")
            sys.exit(1)
        inputs = sorted(
            [
                p for p in folder.iterdir()
                if p.is_file() and (p.suffix.lower() in VIDEO_EXTS or is_audio_file(p))
            ]
        )
        if not inputs:
            base_log.warn("No supported audio/video files found in the specified folder.")
    if args.input_list:
        if mode and mode != "list":
            base_log.warn("Both --input-dir and --input-list provided; proceeding with both (union).")
        mode = "list" if not mode else "mixed"
        listp = Path(args.input_list).expanduser().resolve()
        if not listp.exists() or not listp.is_file():
            base_log.error(f"--input-list not a file: {listp}")
            sys.exit(1)
        with open(listp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):  # allow comments / blank lines
                    continue
                pth = Path(s).expanduser().resolve()
                inputs.append(pth)
    # Single-file input (preferred flag or deprecated positional)
    single_arg = args.input_file or args.input
    if single_arg:
        pth = Path(single_arg).expanduser().resolve()
        inputs.append(pth)
        mode = "single" if not mode else "mixed"

    if args.input_files:
        if mode and mode not in ("multi", "mixed"):
            base_log.warn("Combining --input-files with other input modes; will process all.")
        mode = "multi" if not mode else "mixed"
        group: List[Path] = []
        for raw in args.input_files:
            pth = Path(raw).expanduser().resolve()
            if not pth.exists() or not pth.is_file():
                base_log.error(f"--input-files entry not found: {pth}")
                sys.exit(1)
            group.append(pth)
        if not group:
            base_log.error("--input-files provided but no valid files detected.")
            sys.exit(1)
        multi_groups.append(group)

    # If nothing specified, error
    if not inputs and not multi_groups:
        base_log.error("No inputs provided. Use one of: --input-file, --input-files, --input-dir, --input-list")
        sys.exit(1)

    # De-duplicate, keep order
    seen = set()
    unique_inputs = []
    for p in inputs:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique_inputs.append(p)
    multi_set = {str(p) for group in multi_groups for p in group}
    inputs = [p for p in unique_inputs if str(p) not in multi_set]

    jobs: List[List[Path]] = []
    jobs.extend(multi_groups)
    for p in inputs:
        jobs.append([p])

    # Dry run?
    if args.dry_run:
        base_log.step("Dry run")
        base_log.info(f"Planned jobs ({len(jobs)}):")
        for group in jobs:
            if len(group) == 1:
                base_log.info(f"  - {group[0]}")
            else:
                base_log.info("  - Combined sequence:")
                for path in sorted(group, key=lambda p: p.stat().st_mtime if p.exists() else float("inf")):
                    base_log.info(f"      * {path}")
        base_log.info("Would extract audio, transcribe, write FULL transcript and NOTES for each job.")
        base_log.ok("Exiting.")
        _LAST_JOBS = jobs
        return

    # Batch execution
    results = []  # list of tuples: (ok_bool, input_path, notes_md_path, full_md_path)
    base_log.step(f"Batch start ({len(jobs)} job(s))")
    for group in jobs:
        if len(group) == 1:
            ok, ipath, notes_p, full_p = process_one(group[0], args, base_log, client)
        else:
            ok, ipath, notes_p, full_p = process_sequence(group, args, base_log, client)
        results.append((ok, ipath, notes_p, full_p))

    # If --list-in provided, write mapping file
    if args.input_list:
        list_in_path = Path(args.input_list).expanduser().resolve()
        list_out_path = Path(args.output_list).expanduser().resolve() if args.output_list else list_in_path.with_suffix(list_in_path.suffix + "_transcribed.txt")
        try:
            with open(list_out_path, "w", encoding="utf-8") as f:
                for ok, ip, np, fp in results:
                    f.write(f"{ip}\t{np if np else ''}\t{fp if fp else ''}\n")
            base_log.ok(f"Wrote results mapping → {list_out_path}")
        except Exception as e:
            base_log.warn(f"Could not write output-list file: {e}")

    # Final summary
    success = sum(1 for r in results if r[0])
    base_log.step("Batch summary")
    base_log.info(f"Successful: {success}/{len(results)}")
    for ok, ip, np, fp in results:
        if ok:
            base_log.info(f"OK: {ip} → notes={np} | full={fp}")
        else:
            base_log.warn(f"FAIL: {ip}")

    if args.combine_transcriptions:
        combine_log = base_log.with_prefix("[COMBINED] ")
        successes = [
            (Path(ip), Path(np), Path(fp))
            for ok, ip, np, fp in results
            if ok and np and fp and Path(fp).exists()
        ]
        if not successes:
            combine_log.warn("No successful transcripts to combine; skipping combined outputs.")
        else:
            if args.input_list:
                list_path = Path(args.input_list).expanduser().resolve()
                out_dir = list_path.parent
                stem_base = list_path.stem
            elif args.input_dir and not args.input_list:
                folder_path = Path(args.input_dir).expanduser().resolve()
                out_dir = folder_path
                stem_base = folder_path.name
            elif args.output_file and len(successes) == 1 and args.input_file:
                out_path = Path(args.output_file).expanduser().resolve()
                out_dir = out_path.parent
                stem_base = out_path.stem
            else:
                out_dir = successes[0][0].parent
                stem_base = "combined"
            out_dir.mkdir(parents=True, exist_ok=True)

            combined_transcript_path = ensure_unique_path(out_dir / f"{stem_base}_combined_transcript.md")
            combined_summary_path = ensure_unique_path(out_dir / f"{stem_base}_combined_summary.md")

            combined_sections = ["# Combined Transcript"]
            summary_sources = []

            for idx, (ipath, notes_path, full_path) in enumerate(successes, start=1):
                try:
                    full_text = full_path.read_text(encoding="utf-8")
                except Exception as e:
                    combine_log.warn(f"Skipping {ipath} in combined transcript (read failure: {e})")
                    continue
                body = re.sub(r"^#\s*Full Transcript\s*\n?", "", full_text, flags=re.IGNORECASE).strip()
                if not body:
                    combine_log.warn(f"Skipping {ipath} in combined transcript (empty content).")
                    continue
                combined_sections.append("")
                combined_sections.append(f"## {idx}. {ipath.stem}")
                combined_sections.append(f"**Source:** {ipath}")
                combined_sections.append("")
                combined_sections.append(body)
                summary_sources.append(f"## {ipath.stem}\n{body}")

            if not summary_sources:
                combine_log.warn("Combined transcript would be empty; skipping combined outputs.")
            else:
                try:
                    combined_transcript_path.write_text(
                        "\n".join(combined_sections).rstrip() + "\n",
                        encoding="utf-8"
                    )
                    combine_log.ok(f"Combined transcript → {combined_transcript_path}")
                except Exception as e:
                    combine_log.warn(f"Failed to write combined transcript: {e}")

                combined_text = "\n\n".join(summary_sources)
                detected_lang = "de" if guess_german(combined_text) else ""
                chosen_lang = pick_summary_language(args.summary_language, detected_lang, detected_lang)
                try:
                    combined_summary_md = ask_model_for_notes(
                        client=client,
                        full_text=combined_text,
                        segments=[],
                        domain_hint=args.domain_hint,
                        target_level=args.target_level,
                        target_words=args.summary_words,
                        summary_language=chosen_lang,
                        summary_model=args.summary_model,
                        log=combine_log
                    )
                    combined_summary_path.write_text(combined_summary_md, encoding="utf-8")
                    combine_log.ok(f"Combined summary → {combined_summary_path}")
                except Exception as e:
                    combine_log.warn(f"Failed to build combined summary: {e}")

    _LAST_RESULTS = results
    _LAST_JOBS = jobs
    return 0

if __name__ == "__main__":
    sys.exit(main())

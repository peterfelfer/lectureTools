#!/usr/bin/env python3
"""
End-to-end lecture preparation pipeline.

This script orchestrates the slide image → PDF conversion and the audio/video
transcription workflow using configuration stored in YAML files. Defaults are
loaded from `lecture_pipeline.defaults.yaml` (next to this script) and can be
overridden via `--config`.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

import av_to_md
import image_series_to_pdf
from image_series_to_pdf import A4_DIMENSIONS, create_title_page, load_font

DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".defaults.yaml")

IMAGE_PATH_KEYS = {"directory", "output_dir", "config_yaml", "input_dir"}
IMAGE_BOOL_KEYS = {"overwrite", "verbose"}

AUDIO_PATH_KEYS = {
    "input_file",
    "input_files",
    "input_dir",
    "input_list",
    "output_list",
    "output_file",
    "debug_dir",
    "config_yaml",
}
AUDIO_BOOL_KEYS = {
    "no_chunk",
    "keep_audio",
    "progress",
    "verbose",
    "quiet",
    "show_ffmpeg",
    "dry_run",
    "combine_transcriptions",
}

FINAL_PATH_KEYS = {"output"}
AUDIO_VIDEO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".webm", ".mkv", ".mp4", ".mov", ".avi", ".m4v", ".webm"}
IMAGE_FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".kra"}


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise PipelineError(f"Failed to read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"Configuration at {path} must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def convert_paths(section: Dict[str, Any], path_keys: Iterable[str], base_dir: Path) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    path_keys_set = set(path_keys)
    for key, value in section.items():
        if isinstance(value, dict):
            converted[key] = convert_paths(value, path_keys_set, base_dir)
            continue
        if isinstance(value, list):
            if key in path_keys_set:
                converted_list = []
                for item in value:
                    if item in (None, ""):
                        continue
                    path_value = item if isinstance(item, Path) else Path(item)
                    if not path_value.is_absolute():
                        path_value = (base_dir / path_value).resolve()
                    converted_list.append(path_value)
                converted[key] = converted_list
            else:
                converted[key] = value
            continue
        if key in path_keys_set:
            if value in (None, ""):
                converted[key] = None
                continue
            path_value = Path(value) if not isinstance(value, Path) else value
            if not path_value.is_absolute():
                path_value = (base_dir / path_value).resolve()
            converted[key] = path_value
        else:
            converted[key] = value
    return converted


def measure(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=font)


def has_image_files(directory: Path) -> bool:
    try:
        for child in directory.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_FILE_SUFFIXES:
                return True
    except FileNotFoundError:
        return False
    return False


def locate_slide_directory(base_dir: Path) -> Optional[Path]:
    candidates = [
        base_dir / "slides",
        base_dir / "Slides",
        base_dir / "images",
        base_dir / "Images",
    ]
    for candidate in candidates:
        if candidate.is_dir() and has_image_files(candidate):
            return candidate
    if has_image_files(base_dir):
        return base_dir
    for child in base_dir.iterdir():
        if child.is_dir() and has_image_files(child):
            return child
    return None


def collect_media_files(base_dir: Path) -> List[Path]:
    media: List[Path] = []
    seen: set[str] = set()
    candidates = []
    audio_dir = base_dir / "audio"
    videos_dir = base_dir / "video"
    for candidate in (audio_dir, videos_dir, base_dir):
        if candidate.exists() and candidate.is_dir():
            candidates.append(candidate)
    for directory in candidates:
        for item in sorted(directory.iterdir()):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix in AUDIO_VIDEO_SUFFIXES or suffix in av_to_md.VIDEO_EXTS or av_to_md.is_audio_file(item):
                key = str(item.resolve())
                if key not in seen:
                    seen.add(key)
                    media.append(item.resolve())
    return sorted(media, key=lambda p: p.stat().st_mtime if p.exists() else float("inf"))


def build_auto_config(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    exports_dir = base_dir / "exports"
    slides_dir = locate_slide_directory(base_dir)
    media_files = collect_media_files(base_dir)

    image_conf: Dict[str, Any] = {}
    audio_conf: Dict[str, Any] = {}
    final_conf: Dict[str, Any] = {}

    bundle_name = base_dir.name.replace("_", " ").strip() or "Lecture"

    if slides_dir:
        image_conf.update(
            {
                "enabled": True,
                "input_dir": slides_dir,
                "output_dir": exports_dir / "slides",
                "orientation": "landscape",
                "title_text": bundle_name.title(),
                "title_font_size": 160,
                "title_date_format": "%d %B %Y",
            }
        )
    else:
        image_conf["enabled"] = False

    if media_files:
        audio_conf.update(
            {
                "enabled": True,
                "input_files": media_files,
                "summary_words": 1500,
            }
        )
    else:
        audio_conf["enabled"] = False

    bundle_path = exports_dir / f"{base_dir.name}_bundle.pdf"
    final_conf.update(
        {
            "enabled": True,
            "output": bundle_path,
            "include_slides": bool(slides_dir),
            "title_text": bundle_name.title(),
            "title_date_format": "%d %B %Y",
            "title_font_size": 160,
        }
    )

    presenter_name_default = image_series_to_pdf.FALLBACK_DEFAULTS.get("presenter_name")
    presenter_aff_default = image_series_to_pdf.FALLBACK_DEFAULTS.get("presenter_affiliation")
    if presenter_name_default:
        image_conf.setdefault("presenter_name", presenter_name_default)
        final_conf.setdefault("presenter_name", presenter_name_default)
    if presenter_aff_default:
        image_conf.setdefault("presenter_affiliation", presenter_aff_default)
        final_conf.setdefault("presenter_affiliation", presenter_aff_default)

    return {
        "image_to_pdf": image_conf,
        "audio_transcription": audio_conf,
        "final_document": final_conf,
    }


def markdown_to_plaintext(md: str) -> str:
    lines = []
    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        stripped = stripped.replace("**", "").replace("__", "")
        stripped = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", stripped)
        if stripped.startswith("#"):
            lines.append(stripped.lstrip("#").strip().upper())
        elif stripped.startswith(("-", "*")):
            content = stripped.lstrip("-* ").strip()
            lines.append(f"• {content}")
        else:
            lines.append(stripped)
    return "\n".join(lines)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width, _ = measure(draw, candidate, font)
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def parse_markdown_blocks(md: str) -> List[Any]:
    blocks: List[Any] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            blocks.append(("spacer",))
            i += 1
            continue
        if stripped.startswith("$$"):
            content = stripped.strip("$")
            if stripped == "$$":
                i += 1
                collected = []
                while i < len(lines) and lines[i].strip() != "$$":
                    collected.append(lines[i])
                    i += 1
                content = "\n".join(collected)
                if i < len(lines) and lines[i].strip() == "$$":
                    i += 1
            else:
                i += 1
            blocks.append(("latex", content.strip()))
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            blocks.append(("heading", level, text))
            i += 1
            continue
        if stripped[0] in ("-", "*"):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith(("-", "*")):
                items.append(lines[i].lstrip()[1:].strip())
                i += 1
            blocks.append(("list", items))
            continue
        if re.match(r"\d+\.", stripped):
            items = []
            while i < len(lines) and re.match(r"\d+\.", lines[i].strip()):
                items.append(lines[i].strip().split(".", 1)[1].strip())
                i += 1
            blocks.append(("olist", items))
            continue
        paragraph = []
        while i < len(lines):
            current = lines[i]
            cur_stripped = current.strip()
            if not cur_stripped:
                i += 1
                break
            if cur_stripped.startswith("#") or current.lstrip().startswith(("-", "*")) or re.match(r"\d+\.", cur_stripped) or cur_stripped.startswith("$$"):
                break
            paragraph.append(current)
            i += 1
        blocks.append(("paragraph", " ".join(paragraph).strip()))
    return blocks


def render_latex_to_image(expr: str, max_width: int, dpi: int = 300) -> Optional[Image.Image]:
    expr = expr.strip()
    if not expr:
        return None
    fig = plt.figure(dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    text = ax.text(0.5, 0.5, f"${expr}$", fontsize=48, ha="center", va="center", color="black")
    fig.canvas.draw()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    img = Image.open(buffer).convert("RGBA")
    if img.width > max_width:
        scale = max_width / img.width
        new_size = (int(img.width * scale), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def render_markdown_section(
    title: str,
    markdown: str,
    heading_font_size: int,
    body_font_size: int,
    page_size: Sequence[int],
) -> List[Image.Image]:
    width, height = page_size
    margin_x = int(width * 0.06)
    margin_y = int(height * 0.06)
    max_width = width - 2 * margin_x

    body_font = load_font(body_font_size)
    heading_fonts = {
        1: load_font(int(heading_font_size * 1.2)),
        2: load_font(int(heading_font_size)),
        3: load_font(max(int(heading_font_size * 0.8), body_font_size + 8)),
    }

    blocks = parse_markdown_blocks(markdown)
    if title:
        blocks.insert(0, ("heading", 1, title))

    pages: List[Image.Image] = []
    current_page = Image.new("RGB", (width, height), "white")
    current_draw = ImageDraw.Draw(current_page)
    y = margin_y

    def new_page():
        nonlocal current_page, current_draw, y
        pages.append(current_page)
        current_page = Image.new("RGB", (width, height), "white")
        current_draw = ImageDraw.Draw(current_page)
        y = margin_y

    def ensure_space(required: int):
        nonlocal y
        if y + required > height - margin_y:
            new_page()

    paragraph_spacing = max(body_font_size // 2, 18)

    for block in blocks:
        kind = block[0]
        if kind == "spacer":
            y += paragraph_spacing // 2
            continue
        if kind == "heading":
            level = block[1]
            text = block[2]
            font = heading_fonts.get(level, load_font(max(body_font_size + (3 - level) * 6, body_font_size + 6)))
            tw, th = measure(current_draw, text, font)
            ensure_space(th + paragraph_spacing)
            x = (width - tw) // 2 if level == 1 else margin_x
            current_draw.text((x, y), text, font=font, fill="black")
            y += th + paragraph_spacing
            continue
        if kind in ("paragraph",):
            text = block[1]
            lines = wrap_text(current_draw, text, body_font, max_width)
            line_height = measure(current_draw, "Ag", body_font)[1]
            for line in lines:
                ensure_space(line_height + 4)
                current_draw.text((margin_x, y), line, font=body_font, fill="black")
                y += line_height + 4
            y += paragraph_spacing // 2
            continue
        if kind == "list":
            items = block[1]
            line_height = measure(current_draw, "Ag", body_font)[1]
            bullet_indent = int(body_font_size * 1.5)
            for item in items:
                lines = wrap_text(current_draw, item, body_font, max_width - bullet_indent)
                if not lines:
                    continue
                ensure_space(line_height + 4)
                current_draw.text((margin_x, y), "•", font=body_font, fill="black")
                current_draw.text((margin_x + bullet_indent // 2, y), lines[0], font=body_font, fill="black")
                y += line_height + 4
                for cont in lines[1:]:
                    ensure_space(line_height + 4)
                    current_draw.text((margin_x + bullet_indent // 2, y), cont, font=body_font, fill="black")
                    y += line_height + 4
            y += paragraph_spacing // 2
            continue
        if kind == "olist":
            items = block[1]
            line_height = measure(current_draw, "Ag", body_font)[1]
            number_indent = int(body_font_size * 1.5)
            for idx, item in enumerate(items, 1):
                prefix = f"{idx}."
                lines = wrap_text(current_draw, item, body_font, max_width - number_indent)
                if not lines:
                    continue
                ensure_space(line_height + 4)
                current_draw.text((margin_x, y), prefix, font=body_font, fill="black")
                current_draw.text((margin_x + number_indent // 2, y), lines[0], font=body_font, fill="black")
                y += line_height + 4
                for cont in lines[1:]:
                    ensure_space(line_height + 4)
                    current_draw.text((margin_x + number_indent // 2, y), cont, font=body_font, fill="black")
                    y += line_height + 4
            y += paragraph_spacing // 2
            continue
        if kind == "latex":
            equation = block[1]
            img = render_latex_to_image(equation, max_width)
            if img:
                ensure_space(img.height + paragraph_spacing)
                x = (width - img.width) // 2
                current_page.paste(img, (x, y), img if img.mode == "RGBA" else None)
                y += img.height + paragraph_spacing
            continue

    pages.append(current_page)
    return pages


def determine_lecture_date(
    sources: Sequence[Path],
    date_format: str,
) -> Optional[str]:
    timestamps: List[float] = []
    for path in sources:
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    if not timestamps:
        return None
    dt = datetime.fromtimestamp(min(timestamps))
    try:
        return dt.strftime(date_format)
    except Exception:
        return dt.strftime("%Y-%m-%d")


def build_final_document(
    output_path: Path,
    title_text: Optional[str],
    presenter_name: Optional[str],
    presenter_affiliation: Optional[str],
    lecture_date: Optional[str],
    title_orientation: str,
    title_font_size: int,
    summary_title: Optional[str],
    summary_text: Optional[str],
    transcript_title: Optional[str],
    transcript_text: Optional[str],
    include_summary: bool,
    include_full_transcript: bool,
    section_heading_font_size: int,
    section_body_font_size: int,
    slides_pdfs: Sequence[Path],
) -> Path:
    orientation_key = title_orientation if title_orientation in A4_DIMENSIONS else "landscape"
    title_size = A4_DIMENSIONS[orientation_key]
    portrait_size = A4_DIMENSIONS["portrait"]

    title_pages: List[Image.Image] = []
    if title_text:
        title_page = create_title_page(
            title_text,
            title_size,
            title_font_size,
            presenter_name,
            presenter_affiliation,
            lecture_date,
        )
        title_pages.append(title_page.convert("RGB"))

    text_page_size = title_size

    summary_pages: List[Image.Image] = []
    if include_summary and summary_text and summary_text.strip():
        summary_pages = render_markdown_section(
            summary_title or "Lecture Summary",
            summary_text,
            section_heading_font_size,
            section_body_font_size,
            text_page_size,
        )

    transcript_pages: List[Image.Image] = []
    if include_full_transcript and transcript_text and transcript_text.strip():
        transcript_pages = render_markdown_section(
            transcript_title or "Full Transcript",
            transcript_text,
            section_heading_font_size,
            section_body_font_size,
            text_page_size,
        )

    if not title_pages and not summary_pages and not transcript_pages and not slides_pdfs:
        raise PipelineError("Final document has no content to render.")

    writer = PdfWriter()
    current_page_number = 1

    def add_image_page(image: Image.Image) -> None:
        buffer = BytesIO()
        image.save(buffer, format="PDF", resolution=300.0)
        buffer.seek(0)
        reader = PdfReader(buffer)
        for page in reader.pages:
            writer.add_page(page)

    def add_numbered_image_page(image: Image.Image) -> None:
        nonlocal current_page_number
        draw = ImageDraw.Draw(image)
        font = load_font(section_body_font_size)
        label = str(current_page_number)
        tw, th = measure(draw, label, font)
        margin = max(section_body_font_size // 2, 24)
        draw.text((image.width - tw - margin, image.height - th - margin), label, font=font, fill="black")
        add_image_page(image)
        current_page_number += 1

    for img in title_pages:
        add_numbered_image_page(img)

    for slide_pdf in slides_pdfs:
        if not slide_pdf.exists():
            continue
        reader = PdfReader(str(slide_pdf))
        for page in reader.pages:
            writer.add_page(page)
            current_page_number += 1

    for img in summary_pages:
        add_numbered_image_page(img)

    for img in transcript_pages:
        add_numbered_image_page(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)
    return output_path


def build_image_args(config: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if config.get("config_yaml"):
        args.extend(["--config-yaml", str(config["config_yaml"])])
    if config.get("orientation"):
        args.extend(["--orientation", str(config["orientation"])])
    if config.get("output_dir"):
        args.extend(["--out", str(config["output_dir"])])
    for flag in IMAGE_BOOL_KEYS:
        if config.get(flag):
            args.append(f"--{flag.replace('_', '-')}")
    if config.get("page_numbers") is False:
        args.append("--no-page-numbers")
    if config.get("page_number_font_size"):
        args.extend(["--page-number-font-size", str(config["page_number_font_size"])])
    if config.get("title_text"):
        args.extend(["--title", str(config["title_text"])])
    if config.get("title_font_size"):
        args.extend(["--title-font-size", str(config["title_font_size"])])
    if config.get("presenter_name"):
        args.extend(["--presenter-name", str(config["presenter_name"])])
    if config.get("presenter_affiliation"):
        args.extend(["--presenter-affiliation", str(config["presenter_affiliation"])])
    if config.get("title_date_format"):
        args.extend(["--title-date-format", str(config["title_date_format"])])
    directory = config.get("input_dir") or config.get("directory")
    if not directory:
        raise PipelineError("image_to_pdf.directory must be set in the pipeline configuration.")
    args.extend(["--in", str(directory)])
    extra = config.get("extra_args") or []
    args.extend(str(item) for item in extra)
    return args


def build_audio_args(config: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    positional: List[str] = []
    if config.get("config_yaml"):
        args.extend(["--config-yaml", str(config["config_yaml"])])

    for key, value in config.items():
        if key in {"enabled", "config_yaml", "extra_args"}:
            continue
        if value in (None, "", []):
            continue
        if key == "input":
            positional.append(str(value))
            continue
        if key == "input_files":
            files = value if isinstance(value, (list, tuple)) else [value]
            if files:
                args.append("--in")
                args.extend(str(Path(v)) for v in files)
            continue
        if key in AUDIO_BOOL_KEYS:
            if value:
                args.append(f"--{key.replace('_', '-')}")
            continue
        flag = f"--{key.replace('_', '-')}"
        args.extend([flag, str(value)])

    extra = config.get("extra_args") or []
    args.extend(str(item) for item in extra)
    args.extend(positional)
    return args


def describe_command(program: str, arguments: List[str]) -> str:
    parts = [program] + [shlex.quote(str(arg)) for arg in arguments]
    return " ".join(parts)


def main(argv: Iterable[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run the lecture preparation pipeline (slides → PDF + audio transcription)."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file overriding lecture_pipeline.defaults.yaml.",
    )
    parser.add_argument(
        "--run-image",
        dest="image_enabled",
        action="store_true",
        help="Force running the image-to-PDF step (overrides config).",
    )
    parser.add_argument(
        "--skip-image",
        dest="image_enabled",
        action="store_false",
        help="Skip the image-to-PDF step.",
    )
    parser.add_argument(
        "--run-audio",
        dest="audio_enabled",
        action="store_true",
        help="Force running the audio transcription step (overrides config).",
    )
    parser.add_argument(
        "--skip-audio",
        dest="audio_enabled",
        action="store_false",
        help="Skip the audio transcription step.",
    )
    parser.add_argument(
        "--in",
        dest="manual_input_dir",
        help="Explicit slide/image directory (overrides config and auto-detected paths).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the commands that would run without executing them.",
    )
    parser.add_argument(
        "--title",
        dest="override_title",
        help="Override the lecture title for the final PDF and title page.",
    )
    parser.add_argument(
        "lecture_dir",
        nargs="?",
        help="Directory containing lecture assets; enables auto-configuration mode.",
    )
    parser.set_defaults(image_enabled=None, audio_enabled=None)
    args = parser.parse_args(argv)

    default_config = load_yaml(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    user_config_path = Path(args.config).expanduser() if args.config else None
    if user_config_path and not user_config_path.exists():
        raise PipelineError(f"Config file not found: {user_config_path}")
    user_config = load_yaml(user_config_path) if user_config_path else {}

    default_image = convert_paths(default_config.get("image_to_pdf", {}), IMAGE_PATH_KEYS, DEFAULT_CONFIG_PATH.parent)
    default_audio = convert_paths(default_config.get("audio_transcription", {}), AUDIO_PATH_KEYS, DEFAULT_CONFIG_PATH.parent)
    default_final = convert_paths(default_config.get("final_document", {}), FINAL_PATH_KEYS, DEFAULT_CONFIG_PATH.parent)
    user_base = user_config_path.parent if user_config_path else Path.cwd()
    user_image = convert_paths(user_config.get("image_to_pdf", {}), IMAGE_PATH_KEYS, user_base)
    user_audio = convert_paths(user_config.get("audio_transcription", {}), AUDIO_PATH_KEYS, user_base)
    user_final = convert_paths(user_config.get("final_document", {}), FINAL_PATH_KEYS, user_base)

    image_config = deep_merge(default_image, user_image)
    audio_config = deep_merge(default_audio, user_audio)
    final_config = deep_merge(default_final, user_final)

    lecture_base: Optional[Path] = None

    if args.lecture_dir:
        lecture_dir = Path(args.lecture_dir).expanduser().resolve()
        if not lecture_dir.exists() or not lecture_dir.is_dir():
            raise PipelineError(f"Lecture directory not found: {lecture_dir}")
        lecture_base = lecture_dir
        auto_config = build_auto_config(lecture_dir)
        image_config = deep_merge(image_config, auto_config.get("image_to_pdf", {}))
        audio_config = deep_merge(audio_config, auto_config.get("audio_transcription", {}))
        final_config = deep_merge(final_config, auto_config.get("final_document", {}))
        print(f"[INFO] Auto-configured lecture from {lecture_dir}")
    else:
        lecture_base = None

    image_config.setdefault("enabled", True)
    audio_config.setdefault("enabled", True)
    final_config.setdefault("enabled", True)
    audio_config.setdefault("summary_words", 1500)

    if args.override_title:
        image_config["title_text"] = args.override_title
        final_config["title_text"] = args.override_title
    def audio_inputs_present(conf):
        if conf.get("input_file"):
            return True
        if conf.get("input_dir"):
            return True
        if conf.get("input_list"):
            return True
        files = conf.get("input_files")
        if isinstance(files, (list, tuple, set)) and files:
            return True
        if isinstance(files, Path):
            return True
        return False

    if args.manual_input_dir:
        manual_dir = Path(args.manual_input_dir).expanduser().resolve()
        if not manual_dir.exists() or not manual_dir.is_dir():
            raise PipelineError(f"--in directory not found: {manual_dir}")
        image_config["input_dir"] = manual_dir
        image_config["enabled"] = True
        if not audio_inputs_present(audio_config):
            detected_media = collect_media_files(manual_dir)
            if detected_media:
                audio_config["input_files"] = detected_media
                audio_config["enabled"] = True
                print(f"[INFO] Found {len(detected_media)} media file(s) for transcription under {manual_dir}")

    if args.image_enabled is not None:
        image_config["enabled"] = args.image_enabled
    if args.audio_enabled is not None:
        audio_config["enabled"] = args.audio_enabled

    def audio_inputs_present(conf: Dict[str, Any]) -> bool:
        if conf.get("input_file"):
            return True
        if conf.get("input_dir"):
            return True
        if conf.get("input_list"):
            return True
        files = conf.get("input_files")
        if isinstance(files, (list, tuple, set)) and files:
            return True
        if isinstance(files, Path):
            return True
        return False

    image_enabled = image_config.get("enabled", True)
    audio_enabled = audio_config.get("enabled", True) and audio_inputs_present(audio_config)

    if args.dry_run:
        if image_enabled:
            planned_image = build_image_args(image_config)
            print(describe_command("image_series_to_pdf.py", planned_image))
        if audio_enabled:
            planned_audio = build_audio_args(audio_config)
            print(describe_command("av_to_md.py", planned_audio))
        return 0

    slide_pdfs: List[Path] = []
    if image_enabled:
        slide_config = dict(image_config)
        for key in ("title_text", "presenter_name", "presenter_affiliation", "title_font_size", "title_date_format"):
            slide_config.pop(key, None)
        slide_config["title_text"] = None
        image_args = build_image_args(slide_config)
        slides_output_dir = image_config.get("output_dir") or slide_config.get("input_dir")
        before = set()
        if isinstance(slides_output_dir, Path):
            slides_output_dir.mkdir(parents=True, exist_ok=True)
            before = {p.resolve() for p in slides_output_dir.glob("*.pdf")}
        rc = image_series_to_pdf.main(image_args)
        if rc != 0:
            raise PipelineError("image_series_to_pdf step failed.")
        if isinstance(slides_output_dir, Path):
            after = {p.resolve() for p in slides_output_dir.glob("*.pdf")}
            slide_pdfs = sorted(after - before, key=lambda p: p.stat().st_mtime)
            if not slide_pdfs:
                slide_pdfs = sorted(after, key=lambda p: p.stat().st_mtime)

    audio_results = []
    audio_jobs: List[List[Path]] = []
    if audio_enabled:
        audio_args = build_audio_args(audio_config)
        try:
            result = av_to_md.main(audio_args)
        except SystemExit as exc:
            code = exc.code or 0
            if code != 0:
                raise PipelineError(f"av_to_md step exited with code {code}.")
            result = code
        if isinstance(result, int) and result != 0:
            raise PipelineError(f"av_to_md step returned non-zero code {result}.")
        audio_results = av_to_md.get_last_results()
        audio_jobs = av_to_md.get_last_jobs()

    if final_config.get("enabled", True):
        summary_text = None
        transcript_text = None
        job_sources: List[Path] = []
        selected_notes_path: Optional[Path] = None
        selected_transcript_path: Optional[Path] = None

        for group, outcome in zip(audio_jobs, audio_results):
            ok, ipath, notes_path, full_path = outcome
            if ok:
                job_sources = list(group)
                if notes_path and Path(notes_path).exists():
                    selected_notes_path = Path(notes_path)
                    try:
                        summary_text = selected_notes_path.read_text(encoding="utf-8")
                    except Exception:
                        summary_text = None
                if full_path and Path(full_path).exists():
                    selected_transcript_path = Path(full_path)
                    try:
                        transcript_text = selected_transcript_path.read_text(encoding="utf-8")
                    except Exception:
                        transcript_text = None
                break

        date_format = (
            final_config.get("title_date_format")
            or image_config.get("title_date_format")
            or "%Y-%m-%d"
        )
        lecture_date = determine_lecture_date(job_sources, date_format) if job_sources else None
        if lecture_date is None and isinstance(image_config.get("input_dir"), Path):
            slide_dir = image_config["input_dir"]
            slide_files = [p for p in slide_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_FILE_SUFFIXES]
            lecture_date = determine_lecture_date(slide_files, date_format)

        title_text = final_config.get("title_text") or image_config.get("title_text")
        presenter_name = final_config.get("presenter_name") or image_config.get("presenter_name")
        presenter_affiliation = final_config.get("presenter_affiliation") or image_config.get("presenter_affiliation")
        title_orientation = image_config.get("orientation") or "landscape"
        title_font_size = int(final_config.get("title_font_size") or image_config.get("title_font_size") or 160)

        summary_title = final_config.get("summary_title")
        transcript_title = final_config.get("transcript_title")
        section_heading_font_size = int(final_config.get("section_heading_font_size") or 96)
        section_body_font_size = int(final_config.get("section_body_font_size") or 64)
        include_summary = bool(final_config.get("include_summary", True))
        include_full_transcript = bool(final_config.get("include_full_transcript", False))
        include_slides = bool(final_config.get("include_slides", True))

        slides_to_append = list(slide_pdfs) if include_slides else []

        if final_config.get("output"):
            final_output = Path(final_config["output"]).expanduser().resolve()
        else:
            if selected_notes_path:
                base_dir = selected_notes_path.parent
                base_name = selected_notes_path.stem
            elif selected_transcript_path:
                base_dir = selected_transcript_path.parent
                base_name = selected_transcript_path.stem
            elif slide_pdfs:
                base_dir = slide_pdfs[0].parent
                base_name = slide_pdfs[0].stem
            else:
                base_dir = Path(lecture_base or Path.cwd())
                base_name = base_dir.name or "lecture"
            final_output = (base_dir / f"{base_name}_bundle.pdf").resolve()

        if not title_text:
            title_text = final_output.stem.replace("_", " ").title()

        try:
            build_final_document(
                final_output,
                title_text,
                presenter_name,
                presenter_affiliation,
                lecture_date,
                title_orientation,
                title_font_size,
                summary_title,
                summary_text,
                transcript_title,
                transcript_text,
                include_summary,
                include_full_transcript,
                section_heading_font_size,
                section_body_font_size,
                slides_to_append,
            )
            print(f"[INFO] Final lecture PDF → {final_output}")
        except PipelineError as exc:
            print(f"[WARN] Skipped final document: {exc}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

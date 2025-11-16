# Lecture Tools

Utilities for preparing lecture material:

- Convert ordered slide images (PNG/JPEG/KRA) into A4 PDFs.
- Transcribe lecture audio/video into full transcripts and cleaned-up notes.
- Orchestrate both steps via a single, YAML-configured pipeline.
- Run everything from a minimal desktop GUI (`lecture_gui.py`) that supports drag-and-drop folders/files and a password-style box for the OpenAI API key.

## Pipeline

- `lecture_pipeline.py` runs the full workflow (slides → PDF, audio → Markdown).
- Defaults live in `lecture_pipeline.defaults.yaml`. Override them with your own config:
  ```bash
  python3 lecture_pipeline.py --config my-lecture.yaml
  ```
- Auto mode: drop a directory that contains your lecture assets (e.g. `slides/` with images and `audio/` with recordings) and run:
  ```bash
  python3 lecture_pipeline.py ./lectures/VL3
  ```
  The tool detects slides, concatenates multiple recordings (oldest first), runs transcription, and emits a bundled PDF with title page, AI summary, and slides under `./lectures/VL3/exports/`.
  - Slides are discovered automatically from `slides/`, `Slides/`, `images/`, or the root directory (any folder containing PNG/JPG/KRA files).
  - Audio/video files are collected from `audio/`, `video/`, or the root directory (supported formats: MKV/MP4/MP3/WAV/etc.).
  - Use `--title "My Lecture Title"` to override the auto-generated title for the bundle and title page.
  - Use `--in ./some/other/slides` to point at a specific slide directory when you don’t want auto-detection.
  - Audio summaries default to ~1500 words; adjust with `summary_words` in the config if needed.
  - The bundled PDF name defaults to `<lecture-directory>_bundle.pdf` and is saved inside `exports/` within that directory.
- Config sections:
  - `image_to_pdf`: points to the slide directory and optional overrides for the PDF generator.
  - `audio_transcription`: passes inputs and overrides to the transcription script.
- `final_document`: controls the bundled lecture PDF (title page → summaries → slides).
- Dry run available with `--dry-run` to inspect the commands before execution.

Example excerpt:
```yaml
image_to_pdf:
  input_dir: ./slides/VL3
  output_dir: ./exports
  orientation: landscape

audio_transcription:
  input_file: ./audio/VL3.mkv
  combine_transcriptions: true

final_document:
  output: ./exports/VL3_lecture.pdf
  include_slides: true
  include_full_transcript: false
```

## Slide PDFs (`image_series_to_pdf.py`)

- Usage:
  ```bash
  python3 image_series_to_pdf.py --in <directory> [--out <output>] \
      [--orientation landscape|portrait] [--overwrite] [--verbose] \
      [--no-page-numbers] [--page-number-font-size N] \
      [--title "Lecture Title"] [--title-font-size N] \
      [--presenter-name "Name"] [--presenter-affiliation "Affiliation"] \
      [--title-date-format "%d %B %Y"] [--config-yaml file]
  ```
- Processes files named like `<prefix>-<index>.<ext>` (e.g. `VL3-0.png`, `VL3-1.kra`) and writes `<prefix>.pdf`.
- Krita `.kra` files are unpacked by reading their `mergedimage.png`.
- All pages are letterboxed onto A4 at 300 DPI, numbered in the bottom-right corner, and default to `landscape`.
- Optional title page is rendered when `--title` (or the YAML equivalent) is provided; it automatically includes the earliest slide creation date, presenter name, and affiliation.
- Edit `image_series_to_pdf.defaults.yaml` to change orientation, default input/output paths (if desired), toggle page numbers, tweak font sizes, set a default title/name/affiliation/date format, verbosity, or overwrite behaviour.

## Audio Transcription (`av_to_md.py`)

- Converts MKV/MP4/MP3/WAV (and more) into:
  - `<input>.full.md` – full transcript (timestamps when the model supports them).
  - `<input>.md` – lecture notes / summary with controlled length.
- Supports concatenating multiple recordings into one lecture (`--in file1 file2 ...` sorts by recording time and stitches them together), batch modes (`--input-dir`, `--input-list`), and optional combined summaries.
- Requires `OPENAI_API_KEY` in the environment and `ffmpeg` available in `PATH` (system install or `imageio-ffmpeg` fallback).
- Defaults are editable in `av_to_md.defaults.yaml`; pass a different file via `--config-yaml`.

## Final Lecture Bundle

- After transcription and slide rendering, `lecture_pipeline.py` can emit a single PDF that contains:
  1. The title page (date/name/affiliation pulled from the YAML defaults).
  2. The AI-generated lecture summary (and, optionally, the full transcript).
  3. The slide deck PDF.
- Configure this in `final_document` (`output`, `include_slides`, `include_full_transcript`, font sizes, section titles, optional overrides for title text/name/affiliation/date format).
- For purely audio/video lectures, the bundle will consist of the title page plus the AI summary pages.
- Markdown structure (headings, lists) is preserved in the rendered summary/transcript pages, and LaTeX equations are typeset into the PDF automatically. All generated pages (title, summary, transcript, slides) use landscape A4 layout so the content fills the page.

## Configuration Files

- `lecture_pipeline.defaults.yaml` – high-level pipeline switches and placeholders.
- `image_series_to_pdf.defaults.yaml` – default CLI options for the PDF builder.
- `av_to_md.defaults.yaml` – transcription, summarisation, and retry settings.

All scripts honour `--config-yaml` (or `--config` for the pipeline) so you can maintain per-lecture variants without editing code.

## Desktop GUI

- Launch: `python3 lecture_gui.py`
  - Drag a lecture folder (or a single audio/video file) onto the window or use the browse buttons.
  - Paste/drop your `OPENAI_API_KEY` into the password-style field; it is only injected into the environment for the current run and is not saved to disk.
  - Click **Run Lecture Pipeline** to execute the same workflow as the CLI. Logs stream into the lower pane; use **Stop** to terminate the background process.
- Drag-and-drop uses [`tkinterdnd2`](https://pypi.org/project/tkinterdnd2/) (installed via `pip install -r requirements.txt`). Without it, the GUI still works via the browse buttons.
- Create a macOS `.app`: `pyinstaller --windowed --name "Lecture Tools" lecture_gui.py` (ships a self-contained app bundle you can pin to the Dock).

## Setup & Checks

- Install dependencies: `pip install -r requirements.txt`
- Ensure `ffmpeg` is available if you plan to run transcriptions.
- Quick sanity check: `python3 -m compileall image_series_to_pdf.py av_to_md.py lecture_pipeline.py`
- Example:
  ```bash
  python3 av_to_md.py --in ./audio/lecture_part1.mkv ./audio/lecture_part2.mkv \
      --keep-audio --summary-words 1800 --progress
  ```

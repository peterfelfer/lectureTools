#!/usr/bin/env python3
"""
Convert ordered image sequences (including Krita .kra files) into A4 PDFs.

Given a directory containing files named <prefix>-<index>.<ext>, this script
groups files by prefix, sorts them by their numeric index, converts each entry
to an A4-sized image, and exports one PDF per prefix.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from PIL import Image, ImageDraw, ImageFont

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".kra"}

FILENAME_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<index>\d+)$")

A4_DIMENSIONS = {
    "portrait": (2480, 3508),  # width x height @ 300 DPI
    "landscape": (3508, 2480),
}

DEFAULT_CONFIG_PATH = Path(__file__).with_suffix(".defaults.yaml")
FALLBACK_DEFAULTS = {
    "orientation": "landscape",
    "output_dir": None,
    "overwrite": False,
    "verbose": False,
    "page_numbers": True,
    "page_number_font_size": 64,
    "title_text": None,
    "title_font_size": 160,
    "presenter_name": "Prof. Dr. Peter Felfer",
    "presenter_affiliation": "Institute for General Materials Properties, FAU Erlangen",
    "title_date_format": "%d %B %Y",
}


def load_yaml_defaults(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - misconfigured YAML should surface immediately
        raise RuntimeError(f"Failed to parse YAML defaults at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML defaults at {path} must be a mapping, got {type(data).__name__}")
    return data


def normalize_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if "orientation" in raw and raw["orientation"]:
        orientation = str(raw["orientation"]).lower()
        if orientation not in A4_DIMENSIONS:
            raise ValueError(f"Invalid orientation '{raw['orientation']}'. Must be one of {tuple(A4_DIMENSIONS)}.")
        normalized["orientation"] = orientation
    if "input_dir" in raw:
        value = raw["input_dir"]
        if value in ("", None):
            normalized["input_dir"] = None
        else:
            normalized["input_dir"] = Path(str(value)).expanduser()
    if "output_dir" in raw:
        value = raw["output_dir"]
        if value in ("", None):
            normalized["output_dir"] = None
        else:
            normalized["output_dir"] = Path(str(value)).expanduser()
    if "overwrite" in raw:
        normalized["overwrite"] = bool(raw["overwrite"])
    if "verbose" in raw:
        normalized["verbose"] = bool(raw["verbose"])
    if "page_numbers" in raw and raw["page_numbers"] is not None:
        normalized["page_numbers"] = bool(raw["page_numbers"])
    if "page_number_font_size" in raw and raw["page_number_font_size"]:
        size = int(raw["page_number_font_size"])
        if size <= 0:
            raise ValueError("page_number_font_size must be a positive integer.")
        normalized["page_number_font_size"] = size
    if "title_text" in raw:
        value = raw["title_text"]
        normalized["title_text"] = None if value in ("", None) else str(value)
    if "title_font_size" in raw and raw["title_font_size"]:
        size = int(raw["title_font_size"])
        if size <= 0:
            raise ValueError("title_font_size must be a positive integer.")
        normalized["title_font_size"] = size
    if "presenter_name" in raw:
        value = raw["presenter_name"]
        normalized["presenter_name"] = None if value in ("", None) else str(value)
    if "presenter_affiliation" in raw:
        value = raw["presenter_affiliation"]
        normalized["presenter_affiliation"] = None if value in ("", None) else str(value)
    if "title_date_format" in raw and raw["title_date_format"]:
        normalized["title_date_format"] = str(raw["title_date_format"])
    return normalized


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be processed."""


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    argv = list(argv)
    parser = argparse.ArgumentParser(
        description="Convert ordered image sequences (PNG/JPEG/KRA) into A4 PDFs."
    )
    parser.add_argument(
        "--config-yaml",
        dest="config_yaml",
        help="Path to a YAML file providing default arguments.",
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=Path,
        required=False,
        help="Directory containing files named <prefix>-<index>.<ext>.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="(Deprecated) positional directory argument; use --in instead.",
    )
    parser.add_argument(
        "--orientation",
        choices=tuple(A4_DIMENSIONS),
        help="Page orientation for the output PDF (default: landscape).",
    )
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help="Directory where PDFs will be written (default: input directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing PDF files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--no-page-numbers",
        dest="page_numbers",
        action="store_false",
        help="Disable page numbers on generated slides.",
    )
    parser.add_argument(
        "--page-number-font-size",
        dest="page_number_font_size",
        type=int,
        help="Font size for page numbers in pixels (default from config).",
    )
    parser.add_argument(
        "--title",
        dest="title_text",
        help="Optional title to render on a generated cover page.",
    )
    parser.add_argument(
        "--title-font-size",
        dest="title_font_size",
        type=int,
        help="Font size (pixels) for the title text (default from config).",
    )
    parser.add_argument(
        "--presenter-name",
        dest="presenter_name",
        help="Presenter name to print on the title page.",
    )
    parser.add_argument(
        "--presenter-affiliation",
        dest="presenter_affiliation",
        help="Presenter affiliation to print on the title page.",
    )
    parser.add_argument(
        "--title-date-format",
        dest="title_date_format",
        help="strftime() format string for the lecture date derived from slide files.",
    )
    try:
        parser.set_defaults(**normalize_defaults(FALLBACK_DEFAULTS))
        default_config = load_yaml_defaults(DEFAULT_CONFIG_PATH)
        if default_config:
            parser.set_defaults(**normalize_defaults(default_config))
    except ValueError as exc:
        parser.error(str(exc))

    known_args, _ = parser.parse_known_args(argv)
    if getattr(known_args, "config_yaml", None):
        config_path = Path(known_args.config_yaml).expanduser()
        try:
            config_defaults = load_yaml_defaults(config_path)
            parser.set_defaults(**normalize_defaults(config_defaults))
        except FileNotFoundError:
            parser.error(f"Config YAML not found: {config_path}")
        except ValueError as exc:
            parser.error(str(exc))
        except RuntimeError as exc:
            parser.error(str(exc))

    parsed = parser.parse_args(argv)
    if parsed.input_dir is None:
        if parsed.directory is None:
            parser.error("Missing required argument --in <directory>.")
        parsed.input_dir = parsed.directory
    return parsed


def configure_logging(verbose: bool) -> None:
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)


def collect_ordered_files(directory: Path) -> Dict[str, List[Tuple[int, Path]]]:
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        match = FILENAME_PATTERN.match(entry.stem)
        if not match:
            logging.info("Skipping %s: filename does not match <prefix>-<index> pattern.", entry.name)
            continue
        index = int(match.group("index"))
        prefix = match.group("prefix")
        groups.setdefault(prefix, []).append((index, entry))
    for files in groups.values():
        files.sort(key=lambda item: item[0])
    return groups


def load_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".kra":
        return load_kra_image(path)
    return Image.open(path)


def load_kra_image(path: Path) -> Image.Image:
    try:
        with ZipFile(path) as archive:
            with archive.open("mergedimage.png") as merged:
                data = merged.read()
    except KeyError as exc:
        raise ImageProcessingError(
            f"{path.name}: missing mergedimage.png inside .kra archive."
        ) from exc
    except BadZipFile as exc:
        raise ImageProcessingError(f"{path.name}: invalid .kra archive.") from exc
    return Image.open(BytesIO(data))


def flatten_transparency(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def fit_to_canvas(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    width, height = image.size
    ratio = min(target_width / width, target_height / height)
    new_size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    resized = image.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    offset = ((target_width - resized.width) // 2, (target_height - resized.height) // 2)
    canvas.paste(resized, offset)
    resized.close()
    return canvas


def load_font(font_size: int) -> ImageFont.ImageFont:
    preferred_fonts = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in preferred_fonts:
        try:
            return ImageFont.truetype(name, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def apply_page_numbers(pages: List[Image.Image], font_size: int) -> None:
    if not pages:
        return
    font = load_font(font_size)
    total = len(pages)
    for idx, page in enumerate(pages, start=1):
        draw = ImageDraw.Draw(page)
        label = f"{idx}/{total}"
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:  # pragma: no cover - fallback for older Pillow
            text_width, text_height = draw.textsize(label, font=font)
        margin = max(20, font_size // 2)
        x = page.width - text_width - margin
        y = page.height - text_height - margin
        draw.text(
            (x, y),
            label,
            font=font,
            fill=(0, 0, 0),
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=font)


def _split_long_segment(segment: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    pieces: List[str] = []
    current = ""
    for char in segment:
        candidate = current + char
        width, _ = measure_text(draw, candidate, font)
        if width <= max_width or not current:
            current = candidate
        else:
            pieces.append(current.rstrip())
            current = char
    if current:
        pieces.append(current.rstrip())
    return pieces


def wrap_title_line(line: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = line.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width, _ = measure_text(draw, candidate, font)
        if width <= max_width:
            current = candidate
        else:
            lines.append(current.rstrip())
            current = word
    lines.append(current.rstrip())

    wrapped: List[str] = []
    for segment in lines:
        width, _ = measure_text(draw, segment, font)
        if width <= max_width:
            wrapped.append(segment)
        else:
            wrapped.extend(_split_long_segment(segment, draw, font, max_width))
    return wrapped


def create_title_page(
    title: str,
    target_size: Tuple[int, int],
    font_size: int,
    presenter_name: Optional[str],
    presenter_affiliation: Optional[str],
    lecture_date: Optional[str],
) -> Image.Image:
    page = Image.new("RGB", target_size, (255, 255, 255))
    draw = ImageDraw.Draw(page)
    font = load_font(font_size)
    raw_lines = [line.strip() for line in title.replace("\r", "\n").split("\n")]
    raw_lines = [line for line in raw_lines if line]
    if not raw_lines:
        return page
    margin = max(target_size[0] // 10, 80)
    max_width = max(target_size[0] - 2 * margin, font_size)
    lines: List[str] = []
    for raw_line in raw_lines:
        lines.extend(wrap_title_line(raw_line, draw, font, max_width))
    if not lines:
        return page
    line_spacing = max(font_size // 3, 20)
    title_metrics = [(line, *measure_text(draw, line, font)) for line in lines]

    metadata_lines: List[str] = []
    if presenter_name:
        metadata_lines.append(presenter_name)
    if presenter_affiliation:
        metadata_lines.append(presenter_affiliation)
    if lecture_date:
        metadata_lines.append(lecture_date)

    subtitle_font_size = max(font_size // 2, 32)
    subtitle_font = load_font(subtitle_font_size)
    metadata_spacing = max(subtitle_font_size // 3, 16)
    wrapped_metadata: List[str] = []
    for meta_line in metadata_lines:
        wrapped_metadata.extend(wrap_title_line(meta_line, draw, subtitle_font, max_width))
    metadata_metrics = [
        (line, *measure_text(draw, line, subtitle_font)) for line in wrapped_metadata
    ]

    title_height = sum(h for _, _, h in title_metrics) + line_spacing * (len(title_metrics) - 1)
    metadata_height = (
        sum(h for _, _, h in metadata_metrics) + metadata_spacing * (len(metadata_metrics) - 1)
        if metadata_metrics
        else 0
    )
    gap = line_spacing * 2 if metadata_metrics else 0
    total_height = title_height + gap + metadata_height
    start_y = max((page.height - total_height) // 2, margin)

    for line, width, height in title_metrics:
        x = (page.width - width) // 2
        draw.text((x, start_y), line, font=font, fill=(0, 0, 0))
        start_y += height + line_spacing
    if metadata_metrics:
        start_y += gap - line_spacing  # align gap after last increment
        for line, width, height in metadata_metrics:
            x = (page.width - width) // 2
            draw.text((x, start_y), line, font=subtitle_font, fill=(0, 0, 0))
            start_y += height + metadata_spacing
    return page


def derive_lecture_date(files: List[Tuple[int, Path]], date_format: str) -> Optional[str]:
    timestamps: List[float] = []
    for _, path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        timestamps.append(stat.st_mtime)
    if not timestamps:
        return None
    dt = datetime.fromtimestamp(min(timestamps))
    try:
        return dt.strftime(date_format)
    except Exception:
        return dt.strftime("%Y-%m-%d")


def build_pdf_for_group(
    prefix: str,
    files: List[Tuple[int, Path]],
    target_size: Tuple[int, int],
    output_dir: Path,
    overwrite: bool,
    add_page_numbers: bool,
    page_number_font_size: int,
    title_text: str | None,
    title_font_size: int,
    presenter_name: Optional[str],
    presenter_affiliation: Optional[str],
    lecture_date: Optional[str],
) -> Path:
    processed_pages: List[Image.Image] = []
    if title_text:
        title_page = create_title_page(
            title_text,
            target_size,
            title_font_size,
            presenter_name,
            presenter_affiliation,
            lecture_date,
        )
        processed_pages.append(title_page)
        logging.info("Added title page for %s", prefix)
    for index, path in files:
        try:
            with load_image(path) as raw_image:  # type: ignore
                page = flatten_transparency(raw_image)
        except (OSError, ImageProcessingError) as exc:
            logging.error("Failed to load %s: %s", path.name, exc)
            continue
        prepared = fit_to_canvas(page, target_size)
        page.close()
        processed_pages.append(prepared)
        logging.info("Added %s (index %d) to %s", path.name, index, prefix)

    if not processed_pages:
        raise ImageProcessingError(f"No valid pages for prefix {prefix!r}.")

    if add_page_numbers:
        apply_page_numbers(processed_pages, page_number_font_size)

    output_path = output_dir / f"{prefix}.pdf"
    if output_path.exists() and not overwrite:
        raise ImageProcessingError(
            f"Refusing to overwrite existing file {output_path}. Use --overwrite to replace it."
        )

    first, *rest = processed_pages
    first.save(
        output_path,
        "PDF",
        save_all=True,
        append_images=rest,
        resolution=300,
    )
    for image in processed_pages:
        image.close()
    logging.info("Wrote %s", output_path)
    return output_path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    directory = args.input_dir.resolve()
    if not directory.exists():
        logging.error("Input directory %s does not exist.", directory)
        return 1
    if not directory.is_dir():
        logging.error("Input path %s is not a directory.", directory)
        return 1

    output_dir = args.output_dir.resolve() if args.output_dir else directory
    ensure_directory(output_dir)

    grouped_files = collect_ordered_files(directory)
    if not grouped_files:
        logging.error("No matching files found in %s.", directory)
        return 1

    target_size = A4_DIMENSIONS[args.orientation]
    generated: List[Path] = []
    for prefix, files in sorted(grouped_files.items()):
        lecture_date = None
        if args.title_text:
            date_format = args.title_date_format or "%Y-%m-%d"
            lecture_date = derive_lecture_date(files, date_format)
        try:
            pdf_path = build_pdf_for_group(
                prefix,
                files,
                target_size,
                output_dir,
                args.overwrite,
                args.page_numbers,
                args.page_number_font_size,
                args.title_text,
                args.title_font_size,
                args.presenter_name,
                args.presenter_affiliation,
                lecture_date,
            )
        except ImageProcessingError as exc:
            logging.error("Skipping %s: %s", prefix, exc)
            continue
        generated.append(pdf_path)

    if not generated:
        logging.error("No PDFs were generated.")
        return 1

    logging.info("Generated %d PDF(s).", len(generated))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

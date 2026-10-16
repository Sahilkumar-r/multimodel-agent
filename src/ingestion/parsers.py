"""File parsers: PDF, DOCX, PPTX, XLSX, HTML, plain text and images (with Tesseract OCR).

Only *files* are accepted. Video, audio and link/shortcut files are rejected on purpose.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from src.config import settings
from src.utils import normalize_whitespace

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".yaml", ".yml", ".log", ".ini", ".toml", ".tex",
}
HTML_EXTS = {".html", ".htm"}
OFFICE_EXTS = {".pdf", ".docx", ".pptx", ".xlsx"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus", ".aiff"}
LINK_EXTS = {".url", ".webloc", ".lnk", ".desktop"}
LEGACY_EXTS = {".doc", ".ppt", ".xls", ".rtf", ".odt", ".odp", ".ods"}


class ParseError(RuntimeError):
    """A file could not be turned into text (recorded in the manifest, not retried until it changes)."""


@dataclass
class Section:
    text: str
    location: str
    page: int | None = None
    ocr_used: bool = False


@dataclass
class ParsedDocument:
    rel_path: str
    file_type: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedImage:
    rel_path: str
    file_type: str
    width: int
    height: int
    file_size: int
    llm_data_url: str  # downscaled copy sent to the vision model
    thumb_b64: str  # small JPEG (base64) stored in Pinecone metadata
    ocr_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def classify(path: Path) -> tuple[str, str]:
    """Return (kind, reason). kind is one of document | image | blocked | unsupported."""
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "blocked", "video files are not supported"
    if ext in AUDIO_EXTS:
        return "blocked", "audio files are not supported"
    if ext in LINK_EXTS:
        return "blocked", "URL / website-link files are not supported"
    if ext in IMAGE_EXTS:
        return "image", ""
    if ext in OFFICE_EXTS or ext in TEXT_EXTS or ext in HTML_EXTS:
        return "document", ""
    if ext in LEGACY_EXTS:
        return "unsupported", f"legacy format {ext} (convert to PDF / DOCX / PPTX / XLSX first)"
    return "unsupported", f"unsupported file type {ext or '(no extension)'}"


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
_ocr_warned = False


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def ocr_image(img: Image.Image) -> str:
    """Run Tesseract on a PIL image. Returns '' (and warns once) if Tesseract is unavailable."""
    global _ocr_warned
    if settings.ocr_mode == "off":
        return ""
    try:
        import pytesseract
    except ImportError:
        if not _ocr_warned:
            log.warning("pytesseract is not installed: OCR disabled.")
            _ocr_warned = True
        return ""
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    gray = ImageOps.exif_transpose(img).convert("L")
    if min(gray.size) < 40:
        return ""
    if max(gray.size) < 1200:
        gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    try:
        text = pytesseract.image_to_string(gray, lang=settings.ocr_lang, config="--psm 3")
    except pytesseract.TesseractNotFoundError:
        if not _ocr_warned:
            log.warning(
                "Tesseract binary not found. Install it (see README) or set TESSERACT_CMD. OCR disabled."
            )
            _ocr_warned = True
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR failed: %s", exc)
        return ""
    return normalize_whitespace(text)


def _ocr_blob(blob: bytes) -> str:
    """OCR an embedded image (docx/pptx), skipping tiny icons."""
    try:
        img = Image.open(io.BytesIO(blob))
        if min(img.size) < settings.ocr_min_image_px:
            return ""
        return ocr_image(img)
    except Exception as exc:  # noqa: BLE001
        log.debug("Skipping embedded image: %s", exc)
        return ""


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def _parse_pdf(path: Path) -> tuple[list[Section], dict[str, Any]]:
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz  # type: ignore[no-redef]

    doc = fitz.open(str(path))
    try:
        if doc.needs_pass:
            raise ParseError("PDF is password protected")
        sections: list[Section] = []
        for number, page in enumerate(doc, start=1):
            native = page.get_text("text").strip()
            text, used_ocr = native, False
            wants_ocr = settings.ocr_mode == "always" or (
                settings.ocr_mode == "auto" and len(native) < settings.ocr_min_chars
            )
            if wants_ocr:
                pix = page.get_pixmap(dpi=settings.ocr_dpi)
                ocr_text = ocr_image(Image.open(io.BytesIO(pix.tobytes("png"))))
                if len(ocr_text) > len(native):
                    text, used_ocr = ocr_text, True
            if text.strip():
                sections.append(Section(text=text, location=f"page {number}", page=number, ocr_used=used_ocr))
        meta = {
            "title": (doc.metadata or {}).get("title") or None,
            "author": (doc.metadata or {}).get("author") or None,
            "pages": len(doc),
        }
        return sections, meta
    finally:
        doc.close()


def _parse_docx(path: Path) -> tuple[list[Section], dict[str, Any]]:
    from docx import Document

    doc = Document(str(path))
    lines: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "") if paragraph.style is not None else ""
        lines.append(f"# {text}" if style.startswith("Heading") or style == "Title" else text)
    sections = [Section(text="\n".join(lines), location="body")] if lines else []

    table_lines: list[str] = []
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                table_lines.append(" | ".join(cells))
        table_lines.append("")
    if any(line.strip() for line in table_lines):
        sections.append(Section(text="\n".join(table_lines).strip(), location="tables"))

    for index, rel in enumerate(doc.part.rels.values(), start=1):
        if "image" in rel.reltype and not rel.is_external:
            text = _ocr_blob(rel.target_part.blob)
            if text:
                sections.append(Section(text=text, location=f"embedded image {index}", ocr_used=True))

    props = doc.core_properties
    return sections, {"title": props.title or None, "author": props.author or None}


def _parse_pptx(path: Path) -> tuple[list[Section], dict[str, Any]]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    prs = Presentation(str(path))
    sections: list[Section] = []

    def walk(shapes, parts: list[str], ocr_used: list[bool]) -> None:
        for shape in shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                walk(shape.shapes, parts, ocr_used)
                continue
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text.strip())
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                text = _ocr_blob(shape.image.blob)
                if text:
                    parts.append(f"[image text] {text}")
                    ocr_used.append(True)

    for number, slide in enumerate(prs.slides, start=1):
        parts: list[str] = []
        ocr_used: list[bool] = []
        walk(slide.shapes, parts, ocr_used)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[speaker notes] {notes}")
        if parts:
            sections.append(
                Section(text="\n".join(parts), location=f"slide {number}", page=number, ocr_used=bool(ocr_used))
            )
    props = prs.core_properties
    return sections, {"title": props.title or None, "author": props.author or None, "slides": len(prs.slides)}


def _parse_xlsx(path: Path) -> tuple[list[Section], dict[str, Any]]:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    sections: list[Section] = []
    try:
        for sheet in workbook.worksheets:
            rows: list[str] = []
            for row in sheet.iter_rows(values_only=True, max_row=20_000):
                cells = ["" if value is None else str(value) for value in row]
                if any(cell.strip() for cell in cells):
                    rows.append(" | ".join(cells))
            if rows:
                sections.append(Section(text="\n".join(rows), location=f"sheet {sheet.title}"))
        return sections, {"sheets": len(workbook.worksheets)}
    finally:
        workbook.close()


def _parse_html(path: Path) -> tuple[list[Section], dict[str, Any]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    text = soup.get_text("\n")
    return ([Section(text=text, location="body")] if text.strip() else []), {"title": title}


def _parse_text(path: Path) -> tuple[list[Section], dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ([Section(text=text, location="body")] if text.strip() else []), {}


def parse_document(path: Path, rel_path: str) -> ParsedDocument:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            sections, meta = _parse_pdf(path)
        elif ext == ".docx":
            sections, meta = _parse_docx(path)
        elif ext == ".pptx":
            sections, meta = _parse_pptx(path)
        elif ext == ".xlsx":
            sections, meta = _parse_xlsx(path)
        elif ext in HTML_EXTS:
            sections, meta = _parse_html(path)
        else:
            sections, meta = _parse_text(path)
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"{type(exc).__name__}: {exc}") from exc

    sections = [s for s in sections if normalize_whitespace(s.text)]
    if not sections:
        hint = " (scanned file? install Tesseract and keep OCR_MODE=auto)" if ext in {".pdf", ".docx", ".pptx"} else ""
        raise ParseError(f"no extractable text{hint}")
    return ParsedDocument(rel_path=rel_path, file_type=ext.lstrip("."), sections=sections, metadata=meta)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def _jpeg_b64(img: Image.Image, quality: int) -> str:
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_thumbnail_b64(img: Image.Image, max_bytes: int) -> str:
    """Smallest-first search for a JPEG whose base64 fits the Pinecone metadata budget."""
    b64 = ""
    for side in (640, 512, 384, 256, 192, 128):
        for quality in (75, 60, 45, 30):
            thumb = img.copy()
            thumb.thumbnail((side, side))
            b64 = _jpeg_b64(thumb, quality)
            if len(b64) <= max_bytes:
                return b64
    return b64


def _exif_summary(img: Image.Image) -> dict[str, Any]:
    wanted = {271: "exif_make", 272: "exif_model", 306: "exif_datetime", 274: "exif_orientation"}
    out: dict[str, Any] = {}
    try:
        exif = img.getexif()
        for tag, name in wanted.items():
            value = exif.get(tag)
            if value is not None:
                out[name] = str(value)[:80] if not isinstance(value, (int, float)) else value
    except Exception:  # noqa: BLE001
        pass
    return out


def parse_image(path: Path, rel_path: str) -> ParsedImage:
    try:
        with Image.open(path) as opened:
            opened.load()
            original_format = opened.format or path.suffix.lstrip(".").upper()
            exif = _exif_summary(opened)
            img = ImageOps.exif_transpose(opened)
            rgb = _to_rgb(img)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"cannot open image: {exc}") from exc

    llm_copy = rgb.copy()
    llm_copy.thumbnail((settings.vision_max_side, settings.vision_max_side))
    llm_data_url = "data:image/jpeg;base64," + _jpeg_b64(llm_copy, 85)

    metadata = {
        "format": original_format,
        "mode": img.mode,
        **exif,
    }
    return ParsedImage(
        rel_path=rel_path,
        file_type=path.suffix.lower().lstrip("."),
        width=rgb.width,
        height=rgb.height,
        file_size=path.stat().st_size,
        llm_data_url=llm_data_url,
        thumb_b64=_make_thumbnail_b64(rgb, settings.max_image_b64_bytes),
        ocr_text=ocr_image(rgb),
        metadata=metadata,
    )

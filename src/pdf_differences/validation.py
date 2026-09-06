"""Strict PDF-only validation; there is intentionally no OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from pdf_differences.config import SETTINGS, ComparisonSettings
from pdf_differences.errors import PdfValidationError, RasterPdfError


@dataclass(frozen=True, slots=True)
class PageCapabilities:
    page_index: int
    drawing_count: int
    text_span_count: int
    image_count: int
    image_coverage_fraction: float = 0.0
    structured_coverage_fraction: float = 0.0
    effectively_scanned: bool = False

    @property
    def has_vector_or_text(self) -> bool:
        return self.drawing_count > 0 or self.text_span_count > 0

    @property
    def has_only_raster_content(self) -> bool:
        return self.image_count > 0 and not self.has_vector_or_text

    @property
    def is_raster_only(self) -> bool:
        """Return whether validation classifies the page as effectively raster-only."""
        return self.effectively_scanned


@dataclass(frozen=True, slots=True)
class PdfCapabilities:
    path: Path
    page_count: int
    pages: tuple[PageCapabilities, ...]


def _span_count(page: fitz.Page) -> int:
    total = 0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            total += sum(_span_is_visible(span) for span in line.get("spans", []))
    return total


def _span_is_visible(span: dict) -> bool:
    text = (span.get("text") or "").strip()
    if not text:
        return False
    alpha = span.get("alpha")
    if alpha is None:
        alpha = 255
    return int(alpha) > 0


def _drawing_is_visible(drawing: dict) -> bool:
    def opacity(value) -> float:
        return 1.0 if value is None else float(value)

    def is_white(value) -> bool:
        if value is None:
            return False
        if isinstance(value, int):
            return value & 0xFFFFFF == 0xFFFFFF
        try:
            return all(float(channel) >= 0.985 for channel in value)
        except TypeError:
            return False

    stroke = drawing.get("color")
    fill = drawing.get("fill")
    visible_stroke = (
        stroke is not None
        and opacity(drawing.get("stroke_opacity")) > 0.001
        and not is_white(stroke)
    )
    visible_fill = (
        fill is not None and opacity(drawing.get("fill_opacity")) > 0.001 and not is_white(fill)
    )
    return visible_stroke or visible_fill


def _union_area(rects: list[fitz.Rect], page_rect: fitz.Rect) -> float:
    if not rects:
        return 0.0
    clipped_rects = []
    for rect in rects:
        clipped = rect & page_rect
        if clipped.is_empty or clipped.width <= 0 or clipped.height <= 0:
            continue
        clipped_rects.append(clipped)
    if not clipped_rects:
        return 0.0
    x_edges = sorted({value for rect in clipped_rects for value in (rect.x0, rect.x1)})
    total = 0.0
    for left, right in zip(x_edges, x_edges[1:], strict=False):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        intervals = sorted(
            (rect.y0, rect.y1)
            for rect in clipped_rects
            if rect.x0 <= midpoint <= rect.x1 and rect.y1 > rect.y0
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= end:
                    end = max(end, next_end)
                else:
                    covered += max(0.0, end - start)
                    start, end = next_start, next_end
            covered += max(0.0, end - start)
        total += (right - left) * covered
    return min(float(page_rect.width * page_rect.height), max(0.0, total))


def _page_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    return [fitz.Rect(image["bbox"]) for image in page.get_image_info()]


def _page_structured_rects(page: fitz.Page, drawings: list[dict]) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for drawing in drawings:
        drawing_rect = drawing.get("rect")
        if drawing_rect is not None:
            rects.append(fitz.Rect(drawing_rect))
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if _span_is_visible(span):
                    rects.append(fitz.Rect(span["bbox"]))
    return rects


def _effectively_scanned(
    *,
    page: fitz.Page,
    image_rects: list[fitz.Rect],
    structured_rects: list[fitz.Rect],
    text_span_count: int,
    drawing_count: int,
    settings: ComparisonSettings,
) -> tuple[bool, float, float]:
    page_rect = page.rect
    page_area = max(float(page_rect.width * page_rect.height), 1e-9)
    image_coverage = _union_area(image_rects, page_rect) / page_area
    structured_coverage = _union_area(structured_rects, page_rect) / page_area
    structured_entities = text_span_count + drawing_count
    image_only = bool(image_rects) and structured_entities == 0
    raster_dominant = (
        bool(image_rects)
        and image_coverage >= settings.scanned_page_min_image_coverage
        and (
            structured_entities <= settings.scanned_page_max_structured_entities
            or structured_coverage <= settings.scanned_page_max_structured_coverage
        )
    )
    return image_only or raster_dominant, image_coverage, structured_coverage


def validate_pdf(
    path: str | Path,
    settings: ComparisonSettings = SETTINGS,
) -> PdfCapabilities:
    """Validate and inventory a native PDF without changing or rasterizing it."""

    candidate = Path(path).expanduser()
    if candidate.suffix.lower() != ".pdf":
        raise PdfValidationError(f"'{candidate.name}' is not a PDF. Only .pdf files are supported.")
    if not candidate.is_file():
        raise PdfValidationError(f"PDF not found: {candidate}")
    if candidate.stat().st_size == 0:
        raise PdfValidationError(f"'{candidate.name}' is empty.")
    if candidate.stat().st_size > settings.max_file_size_mb * 1024 * 1024:
        raise PdfValidationError(
            f"'{candidate.name}' exceeds the {settings.max_file_size_mb} MB size limit."
        )
    with candidate.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise PdfValidationError(f"'{candidate.name}' does not have a valid PDF header.")

    try:
        document = fitz.open(candidate)
    except Exception as exc:
        raise PdfValidationError(f"Could not open '{candidate.name}': {exc}") from exc

    try:
        if document.needs_pass:
            raise PdfValidationError(f"'{candidate.name}' is password protected.")
        if document.page_count < 1:
            raise PdfValidationError(f"'{candidate.name}' has no pages.")

        pages: list[PageCapabilities] = []
        raster_pages: list[int] = []
        for index in range(document.page_count):
            page = document.load_page(index)
            drawings = [drawing for drawing in page.get_drawings() if _drawing_is_visible(drawing)]
            drawing_count = len(drawings)
            text_span_count = _span_count(page)
            image_rects = _page_image_rects(page)
            image_count = len(image_rects)
            is_scanned, image_coverage, structured_coverage = _effectively_scanned(
                page=page,
                image_rects=image_rects,
                structured_rects=_page_structured_rects(page, drawings),
                text_span_count=text_span_count,
                drawing_count=drawing_count,
                settings=settings,
            )
            capability = PageCapabilities(
                page_index=index,
                drawing_count=drawing_count,
                text_span_count=text_span_count,
                image_count=image_count,
                image_coverage_fraction=round(image_coverage, 8),
                structured_coverage_fraction=round(structured_coverage, 8),
                effectively_scanned=is_scanned,
            )
            pages.append(capability)
            if capability.is_raster_only:
                raster_pages.append(index + 1)

        if raster_pages:
            shown = ", ".join(str(page) for page in raster_pages[:8])
            suffix = "..." if len(raster_pages) > 8 else ""
            raise RasterPdfError(
                f"'{candidate.name}' contains effectively scanned page(s) {shown}{suffix}. "
                "Pages with images but no usable vector/text entities, or dominant imagery "
                "with insufficient structured content, are treated as raster-only; this "
                "application compares PDF vector objects and text-layer spans only; "
                "OCR/Tesseract and image-comparison fallbacks are intentionally disabled."
            )
        return PdfCapabilities(candidate.resolve(), document.page_count, tuple(pages))
    finally:
        document.close()

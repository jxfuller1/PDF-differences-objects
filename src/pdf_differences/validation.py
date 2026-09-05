"""Strict PDF-only validation; there is intentionally no OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from .config import SETTINGS, ComparisonSettings
from .errors import PdfValidationError, RasterPdfError


@dataclass(frozen=True, slots=True)
class PageCapabilities:
    page_index: int
    drawing_count: int
    text_span_count: int
    image_count: int

    @property
    def has_vector_or_text(self) -> bool:
        return self.drawing_count > 0 or self.text_span_count > 0

    @property
    def is_raster_only(self) -> bool:
        return self.image_count > 0 and not self.has_vector_or_text


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
            total += sum(bool((span.get("text") or "").strip()) for span in line.get("spans", []))
    return total


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
            capability = PageCapabilities(
                page_index=index,
                drawing_count=len(page.get_drawings()),
                text_span_count=_span_count(page),
                image_count=len(page.get_images(full=True)),
            )
            pages.append(capability)
            if capability.is_raster_only:
                raster_pages.append(index + 1)

        if raster_pages:
            shown = ", ".join(str(page) for page in raster_pages[:8])
            suffix = "..." if len(raster_pages) > 8 else ""
            raise RasterPdfError(
                f"'{candidate.name}' contains raster-only page(s) {shown}{suffix}. "
                "This application compares PDF vector objects and text-layer spans only; "
                "OCR/Tesseract and image-comparison fallbacks are intentionally disabled."
            )
        return PdfCapabilities(candidate.resolve(), document.page_count, tuple(pages))
    finally:
        document.close()

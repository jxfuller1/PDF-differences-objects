from __future__ import annotations

import pytest
from helpers import make_raster_only_pdf, make_text_only_pdf

from pdf_differences.errors import PdfValidationError, RasterPdfError
from pdf_differences.validation import validate_pdf


def test_text_layer_pdf_is_accepted(tmp_path):
    path = make_text_only_pdf(tmp_path / "text.pdf")
    capabilities = validate_pdf(path)
    assert capabilities.page_count == 1
    assert capabilities.pages[0].text_span_count == 1
    assert capabilities.pages[0].drawing_count == 0


def test_raster_only_pdf_is_rejected_without_fallback(tmp_path):
    path = make_raster_only_pdf(tmp_path / "scan.pdf")
    with pytest.raises(RasterPdfError, match="OCR/Tesseract"):
        validate_pdf(path)


def test_non_pdf_extension_is_rejected(tmp_path):
    path = tmp_path / "drawing.png"
    path.write_bytes(b"not an image")
    with pytest.raises(PdfValidationError, match="Only .pdf"):
        validate_pdf(path)

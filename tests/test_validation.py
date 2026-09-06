from __future__ import annotations

import pytest
from helpers import (
    make_hidden_text_over_image_pdf,
    make_image_dominant_pdf,
    make_raster_only_pdf,
    make_text_layer_over_image_pdf,
    make_text_only_pdf,
)

from pdf_differences.config import ComparisonSettings
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
    with pytest.raises(RasterPdfError, match="effectively scanned"):
        validate_pdf(path)


def test_small_image_only_page_remains_rejected(tmp_path):
    path = make_raster_only_pdf(
        tmp_path / "small-image-only.pdf",
        image_rect=(20, 20, 60, 60),
    )
    with pytest.raises(RasterPdfError, match="effectively scanned"):
        validate_pdf(path)


def test_full_page_image_with_single_span_is_rejected(tmp_path):
    path = make_text_layer_over_image_pdf(tmp_path / "scan-with-span.pdf")
    with pytest.raises(RasterPdfError, match="insufficient structured content"):
        validate_pdf(path)


def test_effective_scan_thresholds_are_configurable(tmp_path):
    path = make_text_layer_over_image_pdf(tmp_path / "configured-mixed.pdf")
    settings = ComparisonSettings(
        scanned_page_max_structured_entities=0,
        scanned_page_max_structured_coverage=0.0,
    )
    capabilities = validate_pdf(path, settings)
    assert capabilities.pages[0].effectively_scanned is False


def test_hidden_text_over_full_page_image_is_rejected(tmp_path):
    path = make_hidden_text_over_image_pdf(tmp_path / "hidden-text.pdf")
    with pytest.raises(RasterPdfError, match="effectively scanned"):
        validate_pdf(path)


def test_small_image_with_text_layer_is_accepted(tmp_path):
    path = make_image_dominant_pdf(
        tmp_path / "logo-and-text.pdf",
        image_rect=(20, 20, 60, 60),
        overlay_texts=((80, 50, "GENERAL NOTE", 11),),
    )
    capabilities = validate_pdf(path)
    page = capabilities.pages[0]
    assert page.image_coverage_fraction < 0.85
    assert page.text_span_count == 1
    assert page.effectively_scanned is False
    assert page.is_raster_only is False


def test_thin_page_spanning_vectors_do_not_disguise_a_scan(tmp_path):
    path = make_image_dominant_pdf(
        tmp_path / "scan-with-leaders.pdf",
        overlay_lines=(
            (5, 15, 295, 185),
            (5, 35, 295, 165),
            (5, 55, 295, 145),
            (5, 75, 295, 125),
        ),
    )
    with pytest.raises(RasterPdfError, match="insufficient structured content"):
        validate_pdf(path)


def test_full_page_image_with_substantive_overlay_is_accepted(tmp_path):
    path = make_image_dominant_pdf(
        tmp_path / "mixed.pdf",
        overlay_rects=((25, 25, 115, 85), (150, 40, 250, 125)),
        overlay_texts=(
            (34, 52, "REV A", 11),
            (34, 70, "APPROVED", 11),
            (160, 68, "BOM NOTE", 11),
            (160, 95, "CHECK DIMENSIONS", 11),
        ),
    )
    capabilities = validate_pdf(path)
    assert capabilities.page_count == 1
    assert capabilities.pages[0].image_count == 1
    assert capabilities.pages[0].effectively_scanned is False
    assert capabilities.pages[0].is_raster_only is False
    assert capabilities.pages[0].image_coverage_fraction > 0.85
    assert capabilities.pages[0].structured_coverage_fraction > 0.02


def test_non_pdf_extension_is_rejected(tmp_path):
    path = tmp_path / "drawing.png"
    path.write_bytes(b"not an image")
    with pytest.raises(PdfValidationError, match="Only .pdf"):
        validate_pdf(path)

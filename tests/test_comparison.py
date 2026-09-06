from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
from helpers import make_drawing_pdf, make_image_dominant_pdf

from pdf_differences.comparison import compare_pdfs
from pdf_differences.models import ChangeCategory, ChangeType, Transform


def _make_single_shared_text_pdf(
    path: Path,
    shared_point: tuple[float, float],
    unique_text: str,
) -> Path:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text(fitz.Point(*shared_point), ".010", fontsize=10)
    page.insert_text(fitz.Point(20, 180), unique_text, fontsize=10)
    document.save(path)
    document.close()
    return path


def test_end_to_end_finds_dimension_edit_and_added_geometry(tmp_path):
    old = make_drawing_pdf(tmp_path / "old.pdf", dimension="10.0 ±0.1 mm")
    new = make_drawing_pdf(tmp_path / "new.pdf", dimension="12.0 ±0.1 mm", extra_line=True)
    result = compare_pdfs(old, new)
    assert any(
        change.change_type == ChangeType.MODIFIED
        and change.category == ChangeCategory.DIMENSION
        and change.inspection_relevant
        for change in result.changes
    )
    assert any(
        change.change_type == ChangeType.ADDED and change.category == ChangeCategory.GEOMETRY
        for change in result.changes
    )
    assert "inspection-relevant" in result.summary
    assert result.pages[0].affected_area_fraction > 0
    assert result.pages[0].relevant_area_fraction > 0


def test_identical_files_have_no_changes(tmp_path):
    old = make_drawing_pdf(tmp_path / "old.pdf")
    new = make_drawing_pdf(tmp_path / "new.pdf")
    result = compare_pdfs(old, new)
    assert result.changes == ()
    assert result.counts == {"added": 0, "removed": 0, "moved": 0, "modified": 0}


def test_global_export_shift_is_removed_by_vector_alignment(tmp_path):
    old = make_drawing_pdf(tmp_path / "old.pdf")
    new = make_drawing_pdf(tmp_path / "new.pdf", offset=(9.0, -5.0))
    result = compare_pdfs(old, new)
    assert result.pages[0].alignment.status == "aligned"
    assert result.changes == ()


def test_single_accidental_anchor_keeps_page_frames_aligned(tmp_path):
    old = _make_single_shared_text_pdf(tmp_path / "old.pdf", (150, 60), "OLD DRAWING")
    new = _make_single_shared_text_pdf(tmp_path / "new.pdf", (30, 130), "NEW DRAWING")

    result = compare_pdfs(old, new)

    alignment = result.pages[0].alignment
    assert alignment.status == "identity-unverified"
    assert alignment.anchor_count == 1
    assert alignment.transform == Transform()
    assert "page frames were overlaid" in alignment.note
    shared_text_changes = {
        change.change_type
        for change in result.changes
        if change.before_text == ".010" or change.after_text == ".010"
    }
    assert shared_text_changes == {ChangeType.ADDED, ChangeType.REMOVED}


def test_extra_page_entities_are_reported_as_added(tmp_path):
    old = make_drawing_pdf(tmp_path / "old.pdf", pages=1)
    new = make_drawing_pdf(tmp_path / "new.pdf", pages=2)
    result = compare_pdfs(old, new)
    assert any(
        change.page_index == 1 and change.change_type == ChangeType.ADDED
        for change in result.changes
    )
    assert "Page counts differ" in result.notes[0]


def test_embedded_images_are_ignored_with_an_explicit_note(tmp_path):
    overlay_rects = ((25, 25, 115, 85), (150, 40, 250, 125))
    overlay_texts = (
        (34, 52, "REV A", 11),
        (34, 70, "APPROVED", 11),
        (160, 68, "BOM NOTE", 11),
        (160, 95, "CHECK DIMENSIONS", 11),
    )
    old = make_image_dominant_pdf(
        tmp_path / "old.pdf",
        overlay_rects=overlay_rects,
        overlay_texts=overlay_texts,
    )
    new = make_image_dominant_pdf(
        tmp_path / "new.pdf",
        overlay_rects=overlay_rects,
        overlay_texts=overlay_texts,
    )
    result = compare_pdfs(old, new)
    assert result.changes == ()
    assert any("image comparison is disabled" in note for note in result.pages[0].notes)

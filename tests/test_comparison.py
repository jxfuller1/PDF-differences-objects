from __future__ import annotations

from helpers import make_drawing_pdf, make_text_layer_over_image_pdf

from pdf_differences.comparison import compare_pdfs
from pdf_differences.models import ChangeCategory, ChangeType


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
    old = make_text_layer_over_image_pdf(tmp_path / "old.pdf")
    new = make_text_layer_over_image_pdf(tmp_path / "new.pdf")
    result = compare_pdfs(old, new)
    assert result.changes == ()
    assert any("image comparison is disabled" in note for note in result.pages[0].notes)

from __future__ import annotations

import pymupdf as fitz
import pytest
from helpers import make_fragmented_callout_pdf

from pdf_differences.callouts import reconstruct_callouts
from pdf_differences.comparison import compare_pdfs
from pdf_differences.extraction import extract_page_entities
from pdf_differences.models import ChangeCategory, ChangeType


def _callouts(path):
    with fitz.open(path) as document:
        raw = extract_page_entities(document[0], 0)
    return tuple(entity for entity in reconstruct_callouts(raw) if entity.callout_category)


def test_member_edit_reports_one_change_with_the_whole_callout_bbox(tmp_path):
    old = make_fragmented_callout_pdf(
        tmp_path / "old.pdf",
        nominal=".255",
        gdt_tolerance=".010",
    )
    new = make_fragmented_callout_pdf(
        tmp_path / "new.pdf",
        nominal=".455",
        gdt_tolerance=".020",
    )

    result = compare_pdfs(old, new)
    dimension = next(
        change for change in result.changes if change.category == ChangeCategory.DIMENSION
    )
    gdt = next(change for change in result.changes if change.category == ChangeCategory.GDT)
    new_groups = {entity.callout_category: entity for entity in _callouts(new)}

    assert len(result.changes) == 2
    assert dimension.change_type == ChangeType.MODIFIED
    assert dimension.before_text == "4X DIA .255 +.010 BOTH SIDES / -.000"
    assert dimension.after_text == "4X DIA .455 +.010 BOTH SIDES / -.000"
    assert dimension.bbox == pytest.approx(new_groups[ChangeCategory.DIMENSION].bbox)
    assert len(dimension.old_member_entity_ids) == 6
    assert len(dimension.new_member_entity_ids) == 6

    assert gdt.change_type == ChangeType.MODIFIED
    assert gdt.before_text == "POSITION | .010 | A | B"
    assert gdt.after_text == "POSITION | .020 | A | B"
    assert gdt.bbox == pytest.approx(new_groups[ChangeCategory.GDT].bbox)
    assert len(gdt.old_member_entity_ids) == 8
    assert len(gdt.new_member_entity_ids) == 8


def test_whole_new_dimension_is_one_added_change_not_one_per_pdf_span(tmp_path):
    old = make_fragmented_callout_pdf(tmp_path / "old.pdf", include_dimension=False)
    new = make_fragmented_callout_pdf(tmp_path / "new.pdf", include_dimension=True)

    result = compare_pdfs(old, new)
    dimension_changes = tuple(
        change for change in result.changes if change.category == ChangeCategory.DIMENSION
    )
    new_dimension = next(
        entity for entity in _callouts(new) if entity.callout_category == ChangeCategory.DIMENSION
    )

    assert len(dimension_changes) == 1
    assert dimension_changes[0].change_type == ChangeType.ADDED
    assert dimension_changes[0].bbox == new_dimension.bbox
    assert dimension_changes[0].new_member_entity_ids == new_dimension.callout_member_ids
    assert "SECTION" not in dimension_changes[0].after_text
    assert "SCALE" not in dimension_changes[0].after_text


def test_internal_member_layout_edit_also_marks_the_whole_dimension(tmp_path):
    old = make_fragmented_callout_pdf(tmp_path / "old.pdf", nominal_x=190)
    new = make_fragmented_callout_pdf(tmp_path / "new.pdf", nominal_x=195)

    result = compare_pdfs(old, new)

    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.category == ChangeCategory.DIMENSION
    assert change.change_type == ChangeType.MODIFIED
    assert change.before_text == change.after_text
    assert "callout-structure" in change.modification_kinds
    assert len(change.old_member_entity_ids) == 6
    assert len(change.new_member_entity_ids) == 6


def test_removed_member_uses_union_of_old_and_new_callout_boxes(tmp_path):
    old = make_fragmented_callout_pdf(tmp_path / "old.pdf", include_both_sides=True)
    new = make_fragmented_callout_pdf(tmp_path / "new.pdf", include_both_sides=False)
    old_dimension = next(
        entity for entity in _callouts(old) if entity.callout_category == ChangeCategory.DIMENSION
    )
    new_dimension = next(
        entity for entity in _callouts(new) if entity.callout_category == ChangeCategory.DIMENSION
    )

    result = compare_pdfs(old, new)

    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.category == ChangeCategory.DIMENSION
    assert change.change_type == ChangeType.MODIFIED
    assert change.bbox == pytest.approx(
        (
            min(old_dimension.bbox[0], new_dimension.bbox[0]),
            min(old_dimension.bbox[1], new_dimension.bbox[1]),
            max(old_dimension.bbox[2], new_dimension.bbox[2]),
            max(old_dimension.bbox[3], new_dimension.bbox[3]),
        )
    )
    assert len(change.old_member_entity_ids) == 6
    assert len(change.new_member_entity_ids) == 5

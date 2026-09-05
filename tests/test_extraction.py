from __future__ import annotations

import pymupdf as fitz
from helpers import make_drawing_pdf

from pdf_differences.extraction import extract_page_entities
from pdf_differences.models import EntityKind


def test_extracts_stable_text_and_vector_entities(tmp_path):
    path = make_drawing_pdf(tmp_path / "drawing.pdf")
    document = fitz.open(path)
    try:
        first = extract_page_entities(document[0], 0)
        second = extract_page_entities(document[0], 0)
    finally:
        document.close()

    assert [entity.id for entity in first] == [entity.id for entity in second]
    assert sum(entity.kind == EntityKind.TEXT for entity in first) == 2
    assert sum(entity.kind == EntityKind.GEOMETRY for entity in first) == 2
    assert all(0.0 <= value <= 1.0 for entity in first for value in entity.bbox)


def test_ignores_white_background_mask_geometry(tmp_path):
    path = tmp_path / "white-mask.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.draw_rect(
        fitz.Rect(20, 30, 100, 80),
        color=(1.0, 1.0, 1.0),
        fill=(1.0, 1.0, 1.0),
    )
    page.draw_line(fitz.Point(30, 120), fitz.Point(180, 120), color=(0.0, 0.0, 0.0))
    document.save(path)
    document.close()

    with fitz.open(path) as reopened:
        entities = extract_page_entities(reopened[0], 0)

    geometry = [entity for entity in entities if entity.kind == EntityKind.GEOMETRY]
    assert len(geometry) == 1
    assert geometry[0].op_histogram == (("l", 1),)

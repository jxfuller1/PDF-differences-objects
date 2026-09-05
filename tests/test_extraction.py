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

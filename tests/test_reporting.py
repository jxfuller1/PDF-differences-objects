from __future__ import annotations

import csv
import json

import pymupdf as fitz
from helpers import make_drawing_pdf

from pdf_differences.comparison import compare_pdfs
from pdf_differences.reporting import export_annotated_pdf, export_csv, export_json


def test_json_csv_and_vector_markup_exports(tmp_path):
    old = make_drawing_pdf(tmp_path / "old.pdf")
    new = make_drawing_pdf(tmp_path / "new.pdf", dimension="12.0 ±0.1 mm")
    result = compare_pdfs(old, new)

    json_path = export_json(result, tmp_path / "result.json")
    csv_path = export_csv(result, tmp_path / "result.csv")
    marked_path = export_annotated_pdf(result, new, tmp_path / "marked.pdf")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["analysis_mode"] == "vector-and-text-entities-only"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(result.changes)
    document = fitz.open(marked_path)
    try:
        assert document.page_count == 1
        assert "PDF Differences Objects" in document.metadata["subject"]
    finally:
        document.close()

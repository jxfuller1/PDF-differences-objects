from __future__ import annotations

import ast
import tomllib
from collections import Counter
from pathlib import Path

from pdf_differences.comparison import compare_pdfs
from pdf_differences.models import ChangeCategory


def test_committed_sample_exercises_every_requested_parser_branch():
    sample = Path(__file__).resolve().parents[1] / "samples" / "mechanical_pair"
    result = compare_pdfs(sample / "baseline.pdf", sample / "revision.pdf")
    categories = Counter(change.category for change in result.changes)

    assert categories[ChangeCategory.DIMENSION] == 1
    assert categories[ChangeCategory.GDT] == 1
    assert categories[ChangeCategory.NOTE] == 1
    assert categories[ChangeCategory.REVISION] == 1
    assert categories[ChangeCategory.GEOMETRY] == 2
    assert len(result.relevant_changes) == 5
    revision = next(
        change for change in result.changes if change.category == ChangeCategory.REVISION
    )
    assert not revision.inspection_relevant


def test_production_has_no_image_ocr_pytorch_or_scipy_dependencies():
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    prohibited_dependencies = ("pytesseract", "tesseract", "torch", "opencv", "scikit-image")
    core_dependencies = " ".join(manifest["project"]["dependencies"]).casefold()
    assert not any(dependency in core_dependencies for dependency in prohibited_dependencies)
    assert "scipy" not in core_dependencies
    benchmark_extra = " ".join(
        manifest["project"].get("optional-dependencies", {}).get("benchmark", [])
    )
    assert "scipy" in benchmark_extra.casefold()

    analysis_modules = [
        root / "src" / "pdf_differences" / name
        for name in (
            "alignment.py",
            "callouts.py",
            "comparison.py",
            "extraction.py",
            "matching.py",
            "matching_algorithms.py",
            "mechanical.py",
        )
    ]
    analysis_source = "\n".join(path.read_text(encoding="utf-8") for path in analysis_modules)
    assert "get_pixmap" not in analysis_source
    imported_roots: set[str] = set()
    for path in analysis_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint({"cv2", "pytesseract", "scipy", "torch", "torch_geometric"})

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from pdf_differences.comparison import compare_pdfs  # noqa: E402
from pdf_differences.ui.app import PdfDifferencesWindow  # noqa: E402
from pdf_differences.ui.worker import ComparisonWorker  # noqa: E402


def _sample_paths() -> tuple[str, str]:
    sample = Path(__file__).resolve().parents[1] / "samples" / "mechanical_pair"
    return str(sample / "baseline.pdf"), str(sample / "revision.pdf")


def test_window_populates_table_pages_and_both_previews():
    application = QApplication.instance() or QApplication([])
    old_path, new_path = _sample_paths()
    result = compare_pdfs(old_path, new_path)
    window = PdfDifferencesWindow()
    window._old_path = old_path
    window._new_path = new_path
    window._comparison_finished(result)
    application.processEvents()

    assert window.table.rowCount() == len(result.changes)
    assert window.page_selector.count() == len(result.pages)
    assert len(window.old_view.scene().items()) > 1
    assert len(window.new_view.scene().items()) > 1
    window.deleteLater()


def test_worker_accepts_structured_progress_events():
    application = QApplication.instance() or QApplication([])
    old_path, new_path = _sample_paths()
    worker = ComparisonWorker(old_path, new_path)
    progress_events: list[tuple[str, float, str]] = []
    results = []
    worker.progress.connect(
        lambda stage, fraction, message: progress_events.append((stage, fraction, message))
    )
    worker.finished.connect(results.append)
    worker.run()
    application.processEvents()

    assert results
    assert progress_events[0][0] == "validation"
    assert progress_events[-1][0] == "complete"

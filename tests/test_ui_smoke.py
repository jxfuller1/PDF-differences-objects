from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QAbstractAnimation  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from pdf_differences.comparison import compare_pdfs  # noqa: E402
from pdf_differences.models import ChangeType  # noqa: E402
from pdf_differences.ui.app import PdfDifferencesWindow  # noqa: E402
from pdf_differences.ui.viewer import OverlayPageViewer  # noqa: E402
from pdf_differences.ui.worker import ComparisonWorker  # noqa: E402


def _sample_paths() -> tuple[str, str]:
    sample = Path(__file__).resolve().parents[1] / "samples" / "mechanical_pair"
    return str(sample / "baseline.pdf"), str(sample / "revision.pdf")


def test_window_populates_table_pages_and_overlay_preview():
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
    assert len(window.viewer.scene().items()) > 1
    assert set(window.viewer._layers) == {"old", "new"}
    assert all(
        window.table.item(row, column) is not None and window.table.item(row, column).text().strip()
        for row in range(window.table.rowCount())
        for column in range(window.table.columnCount())
    )
    added_row = next(
        row for row in range(window.table.rowCount()) if window.table.item(row, 1).text() == "added"
    )
    added_values = [
        window.table.item(added_row, column).text() for column in range(window.table.columnCount())
    ]
    assert added_values[2] == "GEOMETRY"
    assert added_values[3] == "Yes"
    assert added_values[4].startswith("Vector geometry")
    assert added_values[5].endswith("added.")
    assert added_values[6:8] == ["unmatched", "N/A"]
    assert added_values[8]
    assert window.cards["total"].text() == f"Total: {len(result.changes)}"
    window.deleteLater()


def test_overlay_blends_and_blinks_toggleable_added_and_removed_regions():
    application = QApplication.instance() or QApplication([])
    old_path, new_path = _sample_paths()
    result = compare_pdfs(old_path, new_path)
    added = next(change for change in result.changes if change.change_type == ChangeType.ADDED)
    removed = replace(
        added,
        id="test-removed",
        change_type=ChangeType.REMOVED,
        old_bbox=added.bbox,
        old_entity_id="old-test-entity",
        new_entity_id=None,
        before_text=added.after_text,
        after_text=None,
    )
    page = replace(result.pages[0], changes=(added, removed))
    viewer = OverlayPageViewer()
    viewer.load_page(old_path, new_path, page, page.changes)
    application.processEvents()

    viewer.set_blend(0)
    assert viewer._layers["old"].opacity() == 1.0
    assert viewer._layers["new"].opacity() == 0.0
    viewer.set_blend(50)
    assert viewer._layers["old"].opacity() == 1.0
    assert viewer._layers["new"].opacity() == 1.0
    viewer.set_blend(100)
    assert viewer._layers["old"].opacity() == 0.0
    assert viewer._layers["new"].opacity() == 1.0

    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Running
    viewer.show_additions(False)
    assert not viewer._regions[added.id].isVisible()
    assert viewer._regions[removed.id].isVisible()
    viewer.show_removals(False)
    assert not viewer._regions[removed.id].isVisible()
    viewer.blink_regions(False)
    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Stopped
    viewer.deleteLater()


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

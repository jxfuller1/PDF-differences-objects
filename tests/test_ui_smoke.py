from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QAbstractAnimation, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QCheckBox, QPlainTextEdit  # noqa: E402

import pdf_differences.ui.viewer_settings as viewer_settings  # noqa: E402
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
    assert set(window.viewer._layers) == {
        "old_original",
        "old_tint",
        "new_original",
        "new_tint",
    }
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
    assert isinstance(window.notes, QPlainTextEdit)
    assert window.notes.maximumHeight() == 120
    assert all(
        isinstance(toggle, QCheckBox)
        for toggle in (
            window.added_toggle,
            window.removed_toggle,
            window.regions_toggle,
            window.blink_toggle,
        )
    )
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

    assert viewer._pulse_animation.duration() == viewer_settings.BLINK_DURATION_MS
    assert viewer._region_colors[added.id].getRgb()[:3] == viewer_settings.ADDED_REGION_RGB
    assert viewer._region_colors[removed.id].getRgb()[:3] == viewer_settings.REMOVED_REGION_RGB

    viewer.set_blend(0)
    assert viewer._layers["old_original"].opacity() == 1.0
    assert viewer._layers["new_original"].opacity() == 0.0
    assert viewer._layers["old_tint"].opacity() == 0.0
    assert viewer._layers["new_tint"].opacity() == 0.0
    viewer.set_blend(50)
    assert viewer._layers["old_original"].opacity() == 0.0
    assert viewer._layers["new_original"].opacity() == 0.0
    assert viewer._layers["old_tint"].opacity() == 1.0
    assert viewer._layers["new_tint"].opacity() == 1.0
    viewer.set_blend(100)
    assert viewer._layers["old_original"].opacity() == 0.0
    assert viewer._layers["new_original"].opacity() == 1.0
    assert viewer._layers["old_tint"].opacity() == 0.0
    assert viewer._layers["new_tint"].opacity() == 0.0

    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Running
    assert all(region.scene() is viewer.scene() for region in viewer._regions.values())
    viewer.show_additions(False)
    assert not viewer._regions[added.id].isVisible()
    assert viewer._regions[removed.id].isVisible()
    viewer.show_removals(False)
    assert not viewer._regions[removed.id].isVisible()
    viewer.blink_regions(False)
    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Stopped
    viewer.deleteLater()


def test_table_filters_also_control_visible_regions():
    application = QApplication.instance() or QApplication([])
    old_path, new_path = _sample_paths()
    result = compare_pdfs(old_path, new_path)
    window = PdfDifferencesWindow()
    window._old_path = old_path
    window._new_path = new_path
    window._comparison_finished(result)
    application.processEvents()

    def table_ids() -> set[str]:
        return {
            window.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            for row in range(window.table.rowCount())
        }

    def visible_region_ids() -> set[str]:
        return {
            change_id for change_id, region in window.viewer._regions.items() if region.isVisible()
        }

    def assert_view_matches_table() -> None:
        assert visible_region_ids() == table_ids().intersection(window.viewer._regions)

    assert_view_matches_table()

    added_index = window.type_filter.findData(ChangeType.ADDED.value)
    window.type_filter.setCurrentIndex(added_index)
    assert window.table.rowCount() > 0
    assert_view_matches_table()

    window.added_toggle.setChecked(False)
    assert not visible_region_ids()
    window.added_toggle.setChecked(True)
    assert_view_matches_table()

    window.type_filter.setCurrentIndex(0)
    category = result.pages[0].changes[0].category.value
    category_index = window.category_filter.findData(category)
    window.category_filter.setCurrentIndex(category_index)
    assert_view_matches_table()

    window.category_filter.setCurrentIndex(0)
    window.search_filter.setText(result.pages[0].changes[0].label)
    assert_view_matches_table()

    window.search_filter.clear()
    window.relevant_only_filter.setChecked(True)
    assert_view_matches_table()

    window._show_page(result.pages[0].page_index)
    assert_view_matches_table()

    window.regions_toggle.setChecked(False)
    assert not visible_region_ids()
    window.regions_toggle.setChecked(True)
    assert_view_matches_table()
    window.deleteLater()


def test_table_and_view_regions_select_each_other_and_center_the_change():
    application = QApplication.instance() or QApplication([])
    old_path, new_path = _sample_paths()
    result = compare_pdfs(old_path, new_path)
    window = PdfDifferencesWindow()
    window._old_path = old_path
    window._new_path = new_path
    window._comparison_finished(result)
    window.show()
    application.processEvents()

    first_change = result.changes[0]
    first_row = window._row_for_change(first_change.id)
    assert first_row is not None
    window.table.selectRow(first_row)
    application.processEvents()
    assert window.viewer._selected_id == first_change.id
    region_center = window.viewer._regions[first_change.id].sceneBoundingRect().center()
    mapped_center = window.viewer.mapFromScene(region_center)
    viewport_center = window.viewer.viewport().rect().center()
    assert abs(mapped_center.x() - viewport_center.x()) <= 2
    assert abs(mapped_center.y() - viewport_center.y()) <= 2

    added = next(change for change in result.changes if change.change_type == ChangeType.ADDED)
    modified_index = window.type_filter.findData(ChangeType.MODIFIED.value)
    window.type_filter.setCurrentIndex(modified_index)
    assert window._row_for_change(added.id) is None
    assert not window.viewer._regions[added.id].isVisible()

    added_index = window.type_filter.findData(ChangeType.ADDED.value)
    window.type_filter.setCurrentIndex(added_index)
    assert window._row_for_change(added.id) is not None
    assert window.viewer._regions[added.id].isVisible()
    window.viewer.fit_to_page()
    application.processEvents()
    added_center = window.viewer._regions[added.id].sceneBoundingRect().center()
    QTest.mouseClick(
        window.viewer.viewport(),
        Qt.MouseButton.LeftButton,
        pos=window.viewer.mapFromScene(added_center),
    )
    application.processEvents()

    selected_item = window.table.item(window.table.currentRow(), 0)
    assert selected_item.data(Qt.ItemDataRole.UserRole) == added.id
    assert window.type_filter.currentIndex() == added_index
    assert window.viewer._selected_id == added.id
    window.close()
    application.processEvents()


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

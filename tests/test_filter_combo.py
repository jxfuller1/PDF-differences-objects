from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QSignalSpy, QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from pdf_differences.ui.filter_combo import MultiSelectComboBox  # noqa: E402


def test_checkable_combo_actions_summary_and_change_signal():
    application = QApplication.instance() or QApplication([])
    combo = MultiSelectComboBox(
        (("Added", "added"), ("Removed", "removed")),
        all_text="All change types",
        none_text="No change types",
    )
    model = combo.model()

    assert model.item(0).text() == "Select All"
    assert model.item(1).text() == "Unselect All"
    assert not model.item(0).flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not model.item(1).flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert all(
        model.item(row).flags() & Qt.ItemFlag.ItemIsUserCheckable for row in range(2, combo.count())
    )
    assert combo.checked_data() == ["added", "removed"]
    assert combo.currentText() == "All change types"

    emitted: list[tuple[str, ...]] = []
    combo.selectionChanged.connect(lambda: emitted.append(tuple(combo.checked_data())))
    combo.unselect_all()
    assert combo.checked_data() == []
    assert combo.currentText() == "No change types"
    assert emitted == [()]

    combo.unselect_all()
    assert emitted == [()]
    combo.set_checked_data({"removed"})
    assert combo.checked_data() == ["removed"]
    assert combo.currentText() == "Removed"
    assert emitted[-1] == ("removed",)
    combo.deleteLater()
    application.processEvents()


def test_checkable_combo_popup_actions_stay_open():
    application = QApplication.instance() or QApplication([])
    combo = MultiSelectComboBox((("Added", "added"), ("Removed", "removed")))
    combo.resize(220, 30)
    combo.show()
    combo.showPopup()
    application.processEvents()

    def click_row(row: int) -> None:
        index = combo.model().index(row, 0)
        position = combo.view().visualRect(index).center()
        QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton, pos=position)
        application.processEvents()
        assert combo.view().isVisible()

    click_row(0)
    assert combo.currentText() == "All selected"
    click_row(2)
    assert combo.checked_data() == ["removed"]
    click_row(1)
    assert combo.checked_data() == []
    click_row(0)
    assert combo.checked_data() == ["added", "removed"]

    combo.hidePopup()
    combo.close()
    application.processEvents()


def test_checkable_combo_bulk_actions_emit_view_update_while_open():
    application = QApplication.instance() or QApplication([])
    combo = MultiSelectComboBox((("Added", "added"), ("Removed", "removed")))
    combo.resize(220, 30)
    combo.show()
    combo.showPopup()
    application.processEvents()

    data_spy = QSignalSpy(combo.model().dataChanged)
    selection_spy = QSignalSpy(combo.selectionChanged)

    index = combo.model().index(1, 0)
    position = combo.view().visualRect(index).center()
    QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton, pos=position)
    application.processEvents()
    assert combo.checked_data() == []
    assert combo.currentText() == "None selected"
    assert len(selection_spy) == 1
    assert len(data_spy) == 2
    assert [(signal[0].row(), signal[1].row()) for signal in data_spy] == [(2, 2), (3, 3)]

    index = combo.model().index(0, 0)
    position = combo.view().visualRect(index).center()
    QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton, pos=position)
    application.processEvents()

    assert combo.view().isVisible()
    assert combo.checked_data() == ["added", "removed"]
    assert combo.currentText() == "All selected"
    assert len(selection_spy) == 2
    assert len(data_spy) == 4
    assert [(signal[0].row(), signal[1].row()) for signal in data_spy] == [
        (2, 2),
        (3, 3),
        (2, 2),
        (3, 3),
    ]

    data_spy = QSignalSpy(combo.model().dataChanged)
    index = combo.model().index(1, 0)
    position = combo.view().visualRect(index).center()
    QTest.mouseClick(combo.view().viewport(), Qt.MouseButton.LeftButton, pos=position)
    application.processEvents()

    assert combo.view().isVisible()
    assert combo.checked_data() == []
    assert combo.currentText() == "None selected"
    assert len(selection_spy) == 3
    assert len(data_spy) == 2
    assert [(signal[0].row(), signal[1].row()) for signal in data_spy] == [(2, 2), (3, 3)]

    combo.hidePopup()
    combo.close()
    application.processEvents()

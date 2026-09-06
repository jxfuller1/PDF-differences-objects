"""Excel-style multi-select combo box for filter controls."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from PyQt6.QtCore import QEvent, QModelIndex, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QComboBox, QListView


class MultiSelectComboBox(QComboBox):
    """A combo box with action rows and checkable data rows.

    ``items`` may contain strings or ``(label, data)`` pairs. Using pairs keeps
    the control compatible with :meth:`QComboBox.addItem` while allowing
    arbitrary user data.
    """

    selectionChanged = pyqtSignal()

    _SELECT_ALL = "Select All"
    _UNSELECT_ALL = "Unselect All"

    def __init__(
        self,
        items: Iterable[Any] | None = None,
        parent=None,
        *,
        all_text: str = "All selected",
        none_text: str = "None selected",
    ) -> None:
        super().__init__(parent)
        self._all_text = all_text
        self._none_text = none_text
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lineEdit().installEventFilter(self)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        view = QListView(self)
        view.setSelectionMode(QListView.SelectionMode.NoSelection)
        self.setView(view)
        view.viewport().installEventFilter(self)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self._model.itemChanged.connect(self._item_changed)

        self._append_action(self._SELECT_ALL)
        self._append_action(self._UNSELECT_ALL)
        if items is not None:
            self.addItems(items)
        self._refresh_summary()

    def _append_action(self, label: str) -> None:
        item = QStandardItem(label)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self._model.appendRow(item)

    @staticmethod
    def _item_parts(item: Any) -> tuple[str, Any]:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return str(item[0]), item[1]
        return str(item), item

    def addItem(self, text: str, userData: Any = None) -> None:  # noqa: N802 - Qt API
        item = QStandardItem(str(text))
        item.setData(userData if userData is not None else text, Qt.ItemDataRole.UserRole)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(Qt.CheckState.Checked)
        self._model.appendRow(item)

    def addItems(self, items: Iterable[Any]) -> None:  # noqa: N802 - Qt API
        for value in items:
            text, data = self._item_parts(value)
            self.addItem(text, data)

    def _option_items(self) -> list[QStandardItem]:
        return [self._model.item(row) for row in range(2, self.count())]

    def _row_clicked(self, index: QModelIndex) -> None:
        row = index.row()
        if row == 0:
            self.select_all()
        elif row == 1:
            self.unselect_all()
        elif row >= 2:
            item = self._model.item(row)
            if item is not None:
                item.setCheckState(
                    Qt.CheckState.Unchecked
                    if item.checkState() == Qt.CheckState.Checked
                    else Qt.CheckState.Checked
                )
        self._refresh_summary()

    def _item_changed(self, item: QStandardItem) -> None:
        if item.row() >= 2:
            self._refresh_summary()
            self.selectionChanged.emit()

    def _refresh_summary(self) -> None:
        options = self._option_items()
        selected = [item.text() for item in options if item.checkState() == Qt.CheckState.Checked]
        if not selected:
            summary = self._none_text
        elif len(selected) == len(options):
            summary = self._all_text
        elif len(selected) == 1:
            summary = selected[0]
        else:
            summary = f"{len(selected)} selected"
        self.setEditText(summary)

    def checked_data(self) -> list[Any]:
        """Return user data for checked options, preserving option order."""
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._option_items()
            if item.checkState() == Qt.CheckState.Checked
        ]

    def select_all(self) -> None:
        self._set_all(Qt.CheckState.Checked)

    def unselect_all(self) -> None:
        self._set_all(Qt.CheckState.Unchecked)

    def _set_all(self, state: Qt.CheckState) -> None:
        changed = any(item.checkState() != state for item in self._option_items())
        if not changed:
            return
        with QSignalBlocker(self._model):
            for item in self._option_items():
                item.setCheckState(state)
        self._refresh_summary()
        self.selectionChanged.emit()

    def set_checked_data(self, values: Iterable[Any]) -> None:
        """Check options whose user data occurs in ``values``."""
        wanted = list(values)
        options = self._option_items()
        states = [item.data(Qt.ItemDataRole.UserRole) in wanted for item in options]
        unchanged = all(
            (item.checkState() == Qt.CheckState.Checked) == state
            for item, state in zip(options, states, strict=True)
        )
        if unchanged:
            return
        with QSignalBlocker(self._model):
            for item, state in zip(options, states, strict=True):
                item.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
        self._refresh_summary()
        self.selectionChanged.emit()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self.lineEdit() and event.type() == QEvent.Type.MouseButtonPress:
            self.showPopup()
            return True
        if (
            watched is self.view().viewport()
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                self._row_clicked(index)
                return True
        return super().eventFilter(watched, event)


# A concise alias for callers that prefer the conventional widget name.
CheckableComboBox = MultiSelectComboBox

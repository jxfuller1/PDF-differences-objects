"""PyQt6 desktop workflow for vector-native drawing comparison."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QSignalBlocker, Qt, QThread
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pdf_differences.models import Change, ChangeCategory, ChangeType, ComparisonResult
from pdf_differences.reporting import export_annotated_pdf, export_csv, export_json
from pdf_differences.ui.viewer import OverlayPageViewer
from pdf_differences.ui.worker import ComparisonWorker


class PdfPathEdit(QLineEdit):
    """Path field that accepts one dropped PDF."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(
            url.toLocalFile().lower().endswith(".pdf") for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".pdf"):
                self.setText(path)
                event.acceptProposedAction()
                return


class PdfDifferencesWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PDF Differences Objects")
        self.resize(1480, 940)
        self.result: ComparisonResult | None = None
        self._thread: QThread | None = None
        self._worker: ComparisonWorker | None = None
        self._changes: list[Change] = []
        self._old_path = ""
        self._new_path = ""
        self._close_when_finished = False
        self._build()
        self.status.setText("Choose two vector/text-layer PDFs to begin.")

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(9)

        layout.addWidget(QLabel("PDF Differences Objects"))
        layout.addWidget(
            QLabel(
                "Entity-level CAD revision review · vector paths + text layer · "
                "no OCR or pixel diff"
            )
        )

        form = QFormLayout()
        self.old_path_edit = self._file_row(
            form, "Baseline", "Original / baseline PDF (or drop a PDF here)"
        )
        self.new_path_edit = self._file_row(
            form, "Revision", "New / revised PDF (or drop a PDF here)"
        )
        layout.addLayout(form)

        action_row = QHBoxLayout()
        self.compare_button = QPushButton("Compare PDFs")
        self.compare_button.clicked.connect(self.start_compare)
        action_row.addWidget(self.compare_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        action_row.addWidget(self.cancel_button)
        action_row.addStretch()
        action_row.addWidget(QLabel("Preview page"))
        self.page_selector = QComboBox()
        self.page_selector.setEnabled(False)
        self.page_selector.currentIndexChanged.connect(self._page_changed)
        action_row.addWidget(self.page_selector)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(230)
        self.progress.setVisible(False)
        action_row.addWidget(self.progress)
        self.status = QLabel()
        self.status.setMinimumWidth(210)
        action_row.addWidget(self.status)
        layout.addLayout(action_row)

        self.summary = QLabel("No comparison loaded.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.notes.setMaximumHeight(120)
        layout.addWidget(self.notes)

        metrics = QHBoxLayout()
        self.cards: dict[str, QLabel] = {}
        for key in ("total", "added", "removed", "moved", "modified", "relevant", "area"):
            card = QLabel(f"{key.title()}: 0")
            metrics.addWidget(card)
            self.cards[key] = card
        metrics.addStretch()
        layout.addLayout(metrics)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = OverlayPageViewer()
        self.viewer.set_region_clicked_handler(self._select_change_by_id)
        # Compatibility names for code that referenced either former preview pane.
        self.old_view = self.viewer
        self.new_view = self.viewer
        viewer_layout.addLayout(self._review_controls())
        viewer_layout.addWidget(self.viewer, 1)
        vertical_splitter.addWidget(viewer_panel)
        vertical_splitter.addWidget(self._results_panel())
        vertical_splitter.setSizes([520, 330])
        layout.addWidget(vertical_splitter, 1)

    def _review_controls(self) -> QHBoxLayout:
        control_row = QHBoxLayout()
        old_button = QPushButton("Old")
        old_button.clicked.connect(lambda: self.blend_slider.setValue(0))
        differences_button = QPushButton("Differences")
        differences_button.clicked.connect(lambda: self.blend_slider.setValue(50))
        new_button = QPushButton("New")
        new_button.clicked.connect(lambda: self.blend_slider.setValue(100))
        control_row.addWidget(old_button)
        control_row.addWidget(differences_button)
        control_row.addWidget(new_button)

        self.blend_label = QLabel("Old (red) ← 50% → New (blue)")
        control_row.addWidget(self.blend_label)
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_slider.setToolTip(
            "Old and New endpoints use the PDFs' original colors; the middle shows "
            "the old revision in red and the new revision in blue."
        )
        self.blend_slider.valueChanged.connect(self._set_blend)
        control_row.addWidget(self.blend_slider, 1)

        self.added_toggle = QCheckBox("Additions")
        self.added_toggle.setChecked(True)
        self.added_toggle.setToolTip("Show or hide added change regions.")
        self.added_toggle.toggled.connect(self.viewer.show_additions)
        control_row.addWidget(self.added_toggle)
        self.removed_toggle = QCheckBox("Removals")
        self.removed_toggle.setChecked(True)
        self.removed_toggle.setToolTip("Show or hide removed change regions.")
        self.removed_toggle.toggled.connect(self.viewer.show_removals)
        control_row.addWidget(self.removed_toggle)
        self.regions_toggle = QCheckBox("Regions")
        self.regions_toggle.setChecked(True)
        self.regions_toggle.setToolTip("Show or hide all change-region boxes.")
        self.regions_toggle.toggled.connect(self.viewer.show_regions)
        control_row.addWidget(self.regions_toggle)
        self.blink_toggle = QCheckBox("Blink")
        self.blink_toggle.setChecked(True)
        self.blink_toggle.setToolTip("Turn the change-region pulse animation on or off.")
        self.blink_toggle.toggled.connect(self.viewer.blink_regions)
        control_row.addWidget(self.blink_toggle)

        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.viewer.fit_to_page)
        control_row.addWidget(fit_button)
        actual_size_button = QPushButton("1:1")
        actual_size_button.clicked.connect(self.viewer.reset_view)
        control_row.addWidget(actual_size_button)
        return control_row

    def _set_blend(self, value: int) -> None:
        if value == 0:
            label = "Old (original) ← 0% → New"
        elif value == 100:
            label = "Old ← 100% → New (original)"
        else:
            label = f"Old (red) ← {value}% → New (blue)"
        self.blend_label.setText(label)
        self.viewer.set_blend(value)

    def _file_row(self, form: QFormLayout, label: str, placeholder: str) -> PdfPathEdit:
        edit = PdfPathEdit()
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda _checked=False, field=edit: self._choose(field))
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _results_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 4, 0, 0)

        filters = QHBoxLayout()
        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText(
            "Search labels, details, relevance reasons, or tiers…"
        )
        self.search_filter.textChanged.connect(self._populate_table)
        filters.addWidget(self.search_filter, 1)
        self.type_filter = QComboBox()
        self.type_filter.addItem("All change types", "")
        for change_type in ChangeType:
            self.type_filter.addItem(change_type.value.title(), change_type.value)
        self.type_filter.currentIndexChanged.connect(self._populate_table)
        filters.addWidget(self.type_filter)
        self.category_filter = QComboBox()
        self.category_filter.addItem("All categories", "")
        for category in ChangeCategory:
            self.category_filter.addItem(category.value, category.value)
        self.category_filter.currentIndexChanged.connect(self._populate_table)
        filters.addWidget(self.category_filter)
        self.relevant_only_filter = QCheckBox("Inspection-relevant only")
        self.relevant_only_filter.stateChanged.connect(self._populate_table)
        filters.addWidget(self.relevant_only_filter)
        panel_layout.addLayout(filters)

        headers = (
            "Page",
            "Type",
            "Category",
            "Inspect",
            "Label",
            "Detail",
            "Tier",
            "Score",
            "Relevance reason",
        )
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._select_change)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 3, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(4, 190)
        panel_layout.addWidget(self.table, 1)

        export_row = QHBoxLayout()
        for text, handler in (
            ("Export JSON", self._export_json),
            ("Export CSV", self._export_csv),
            ("Export annotated PDF", self._export_pdf),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            export_row.addWidget(button)
        export_row.addStretch()
        self.annotate_relevant_only = QCheckBox("PDF: relevant changes only")
        export_row.addWidget(self.annotate_relevant_only)
        panel_layout.addLayout(export_row)
        return panel

    def _choose(self, field: PdfPathEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose drawing PDF",
            str(Path(field.text()).parent) if field.text() else "",
            "PDF files (*.pdf)",
        )
        if path:
            field.setText(path)

    def start_compare(self) -> None:
        if self._thread is not None:
            return
        if not self.old_path_edit.text() or not self.new_path_edit.text():
            QMessageBox.warning(
                self, "Missing PDF", "Choose both a baseline and revised PDF before comparing."
            )
            return

        self._old_path = self.old_path_edit.text().strip()
        self._new_path = self.new_path_edit.text().strip()
        self.compare_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Starting…")

        thread = QThread(self)
        worker = ComparisonWorker(self._old_path, self._new_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._comparison_finished)
        worker.failed.connect(self._comparison_failed)
        worker.cancelled.connect(self._comparison_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_progress(self, _stage: str, fraction: float, message: str) -> None:
        self.progress.setValue(round(fraction * 100))
        self.status.setText(message)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current stage…")

    def _comparison_cancelled(self) -> None:
        self._reset_controls("Comparison cancelled")

    def _comparison_failed(self, message: str) -> None:
        self._reset_controls("Comparison failed")
        QMessageBox.critical(self, "Comparison failed", message)

    def _comparison_finished(self, result: ComparisonResult) -> None:
        self.result = result
        self._changes = list(result.changes)
        self._reset_controls("Comparison complete")
        self.summary.setText(result.summary)
        page_notes = [
            f"Page {page.page_index + 1}: {page.alignment.status}"
            + (f" — {'; '.join(page.notes)}" if page.notes else "")
            for page in result.pages
        ]
        self.notes.setPlainText("\n".join((*result.notes, *page_notes)) or "No comparison notes.")
        values = {
            "total": len(result.changes),
            "relevant": len(result.relevant_changes),
            "area": f"{result.mean_affected_area_fraction * 100:.2f}%",
            **result.counts,
        }
        for key, card in self.cards.items():
            card.setText(f"{key.title()}: {values.get(key, 0)}")

        with QSignalBlocker(self.page_selector):
            self.page_selector.clear()
            for page in result.pages:
                self.page_selector.addItem(f"{page.page_index + 1}", page.page_index)
        self.page_selector.setEnabled(bool(result.pages))
        self._populate_table()
        if result.pages:
            self._show_page(result.pages[0].page_index)

    def _reset_controls(self, message: str) -> None:
        self.compare_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self.status.setText(message)

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        if self._close_when_finished:
            self.close()

    def _filtered_changes(self) -> list[Change]:
        query = self.search_filter.text().casefold()
        selected_type = self.type_filter.currentData()
        selected_category = self.category_filter.currentData()
        relevant_only = self.relevant_only_filter.isChecked()
        filtered: list[Change] = []
        for change in self._changes:
            tier = change.match_tier.value if change.match_tier else "unmatched"
            searchable = " ".join(
                (
                    str(change.page_index + 1),
                    change.change_type.value,
                    change.category.value,
                    change.label,
                    change.detail,
                    change.relevance_reason,
                    tier,
                    change.before_text or "",
                    change.after_text or "",
                )
            ).casefold()
            if query and query not in searchable:
                continue
            if selected_type and change.change_type.value != selected_type:
                continue
            if selected_category and change.category.value != selected_category:
                continue
            if relevant_only and not change.inspection_relevant:
                continue
            filtered.append(change)
        return filtered

    def _populate_table(self, *_args) -> None:
        filtered_changes = self._filtered_changes()
        self.table.setRowCount(0)
        for change in filtered_changes:
            tier = change.match_tier.value if change.match_tier else "unmatched"
            row = self.table.rowCount()
            self.table.insertRow(row)
            score = (
                f"{change.similarity_score:.3f}" if change.similarity_score is not None else "N/A"
            )
            values = (
                str(change.page_index + 1),
                change.change_type.value,
                change.category.value,
                "Yes" if change.inspection_relevant else "No",
                change.label,
                change.detail,
                tier,
                score,
                change.relevance_reason,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, change.id)
        self.viewer.set_visible_change_ids(change.id for change in filtered_changes)

    def _select_change(self) -> None:
        if self.result is None or self.table.currentRow() < 0:
            return
        first_item = self.table.item(self.table.currentRow(), 0)
        if first_item is None:
            return
        change_id = first_item.data(Qt.ItemDataRole.UserRole)
        selected = next((change for change in self.result.changes if change.id == change_id), None)
        if selected is not None:
            if self.viewer.page_index == selected.page_index:
                self.viewer.focus_change(selected.id)
            else:
                self._show_page(selected.page_index, selected.id)

    def _select_change_by_id(self, change_id: str) -> None:
        if self.result is None:
            return
        row = self._row_for_change(change_id)
        if row is None:
            with (
                QSignalBlocker(self.search_filter),
                QSignalBlocker(self.type_filter),
                QSignalBlocker(self.category_filter),
                QSignalBlocker(self.relevant_only_filter),
            ):
                self.search_filter.clear()
                self.type_filter.setCurrentIndex(0)
                self.category_filter.setCurrentIndex(0)
                self.relevant_only_filter.setChecked(False)
            self._populate_table()
            row = self._row_for_change(change_id)
        if row is None:
            return
        item = self.table.item(row, 0)
        with QSignalBlocker(self.table):
            self.table.selectRow(row)
        self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
        self.viewer.focus_change(change_id)

    def _row_for_change(self, change_id: str) -> int | None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == change_id:
                return row
        return None

    def _page_changed(self, index: int) -> None:
        if index >= 0:
            page_index = self.page_selector.itemData(index)
            if page_index is not None:
                self._show_page(int(page_index))

    def _show_page(self, page_index: int, selected_id: str | None = None) -> None:
        if self.result is None:
            return
        for index in range(self.page_selector.count()):
            if self.page_selector.itemData(index) == page_index:
                with QSignalBlocker(self.page_selector):
                    self.page_selector.setCurrentIndex(index)
                break
        page_result = next(page for page in self.result.pages if page.page_index == page_index)
        self.viewer.load_page(
            self._old_path,
            self._new_path,
            page_result,
            page_result.changes,
            selected_id,
        )
        self.viewer.set_blend(self.blend_slider.value())

    def _destination(self, extension: str) -> str:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export comparison",
            f"comparison.{extension.lower()}",
            f"{extension.upper()} files (*.{extension.lower()})",
        )
        return path

    def _export(self, extension: str, action) -> None:
        if self.result is None:
            QMessageBox.information(self, "Nothing to export", "Run a comparison first.")
            return
        destination = self._destination(extension)
        if not destination:
            return
        if Path(destination).exists():
            choice = QMessageBox.question(
                self,
                "Confirm overwrite",
                f"Replace existing file?\n{destination}",
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        try:
            action(destination)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.status.setText(f"Exported {Path(destination).name}")

    def _export_json(self) -> None:
        self._export("json", lambda path: export_json(self.result, path, overwrite=True))

    def _export_csv(self) -> None:
        self._export("csv", lambda path: export_csv(self.result, path, overwrite=True))

    def _export_pdf(self) -> None:
        self._export(
            "pdf",
            lambda path: export_annotated_pdf(
                self.result,
                self._new_path,
                path,
                relevant_only=self.annotate_relevant_only.isChecked(),
                overwrite=True,
            ),
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is None:
            event.accept()
            return
        choice = QMessageBox.question(
            self,
            "Comparison running",
            "Cancel the comparison and close when the current stage finishes?",
        )
        if choice == QMessageBox.StandardButton.Yes:
            self._close_when_finished = True
            self._cancel()
        event.ignore()


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("PDF Differences Objects")
    application.setOrganizationName("PDF Differences Objects")
    window = PdfDifferencesWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""PyQt6 desktop workflow for vector-native drawing comparison."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QSignalBlocker, Qt, QThread
from PyQt6.QtGui import QCloseEvent, QColor
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
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import Change, ChangeCategory, ChangeType, ComparisonResult
from ..reporting import export_annotated_pdf, export_csv, export_json
from .viewer import PageViewer
from .worker import ComparisonWorker

_ROW_COLORS = {
    ChangeType.ADDED: QColor("#147a3b"),
    ChangeType.REMOVED: QColor("#b52f2f"),
    ChangeType.MOVED: QColor("#a85f00"),
    ChangeType.MODIFIED: QColor("#1c5aa6"),
}


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

        heading = QLabel("PDF Differences Objects")
        heading.setObjectName("title")
        layout.addWidget(heading)
        subtitle = QLabel(
            "Entity-level CAD revision review · vector paths + text layer · no OCR or pixel diff"
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

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
        self.compare_button.setObjectName("primary")
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
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.notes = QLabel("")
        self.notes.setObjectName("notes")
        self.notes.setWordWrap(True)
        layout.addWidget(self.notes)

        cards = QHBoxLayout()
        self.cards: dict[str, QLabel] = {}
        for key in ("total", "added", "removed", "moved", "modified", "relevant", "area"):
            card = QLabel(f"0\n{key.title()}")
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setMinimumWidth(105)
            card.setObjectName("card")
            cards.addWidget(card)
            self.cards[key] = card
        layout.addLayout(cards)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.old_view = PageViewer()
        self.new_view = PageViewer()
        preview_splitter.addWidget(self._viewer_panel("Baseline", self.old_view))
        preview_splitter.addWidget(self._viewer_panel("Revision", self.new_view))
        preview_splitter.setSizes([1, 1])
        vertical_splitter.addWidget(preview_splitter)
        vertical_splitter.addWidget(self._results_panel())
        vertical_splitter.setSizes([520, 330])
        layout.addWidget(vertical_splitter, 1)

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

    @staticmethod
    def _viewer_panel(title: str, viewer: PageViewer) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setObjectName("viewerTitle")
        panel_layout.addWidget(label)
        panel_layout.addWidget(viewer, 1)
        return panel

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
        self.notes.setText("\n".join((*result.notes, *page_notes)) or "No comparison notes.")
        values = {
            "total": len(result.changes),
            "relevant": len(result.relevant_changes),
            "area": f"{result.mean_affected_area_fraction * 100:.2f}%",
            **result.counts,
        }
        for key, card in self.cards.items():
            card.setText(f"{values.get(key, 0)}\n{key.title()}")

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

    def _populate_table(self, *_args) -> None:
        query = self.search_filter.text().casefold()
        selected_type = self.type_filter.currentData()
        selected_category = self.category_filter.currentData()
        relevant_only = self.relevant_only_filter.isChecked()
        self.table.setRowCount(0)
        for change in self._changes:
            tier = change.match_tier.value if change.match_tier else ""
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

            row = self.table.rowCount()
            self.table.insertRow(row)
            score = f"{change.similarity_score:.3f}" if change.similarity_score is not None else ""
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
                if column == 1:
                    item.setForeground(_ROW_COLORS[change.change_type])
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, change.id)

    def _select_change(self) -> None:
        if self.result is None or self.table.currentRow() < 0:
            return
        first_item = self.table.item(self.table.currentRow(), 0)
        if first_item is None:
            return
        change_id = first_item.data(Qt.ItemDataRole.UserRole)
        selected = next((change for change in self.result.changes if change.id == change_id), None)
        if selected is not None:
            self._show_page(selected.page_index, selected.id)

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
        changes = [change for change in self.result.changes if change.page_index == page_index]
        self.old_view.load_page(
            self._old_path,
            page_index,
            changes,
            selected_id,
            old_side=True,
        )
        self.new_view.load_page(
            self._new_path,
            page_index,
            changes,
            selected_id,
            old_side=False,
        )

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
    application.setStyleSheet(
        """
        QMainWindow { background: #f8fafc; }
        QLabel#title { font-size: 25px; font-weight: 700; color: #18324b; }
        QLabel#subtitle { color: #536577; padding-bottom: 3px; }
        QLabel#summary { background: #eef5ff; border-radius: 6px; padding: 8px; }
        QLabel#notes { color: #536577; font-size: 11px; }
        QLabel#viewerTitle { font-weight: 600; color: #263746; }
        QLabel#card {
            background: white;
            border: 1px solid #dbe3ea;
            border-radius: 8px;
            padding: 8px;
            font-size: 15px;
        }
        QPushButton { padding: 6px 12px; }
        QPushButton#primary { background: #246bce; color: white; font-weight: 600; }
        QLineEdit, QComboBox { padding: 5px; }
        QTableWidget { background: white; alternate-background-color: #f5f8fb; }
        """
    )
    window = PdfDifferencesWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

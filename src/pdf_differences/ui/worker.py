"""Background comparison worker."""

from __future__ import annotations

from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from pdf_differences.comparison import compare_pdfs
from pdf_differences.errors import ComparisonCancelled


class ComparisonWorker(QObject):
    progress = pyqtSignal(str, float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, old_path: str, new_path: str) -> None:
        super().__init__()
        self.old_path = old_path
        self.new_path = new_path
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = compare_pdfs(
                self.old_path,
                self.new_path,
                progress=lambda e: self.progress.emit(e.stage, e.fraction, e.message),
                cancelled=self._cancel.is_set,
            )
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        except ComparisonCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))

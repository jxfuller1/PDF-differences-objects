"""Zoomable PDF page preview with display-only change overlays."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pymupdf as fitz
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from ..models import Change, ChangeType

_COLORS = {
    ChangeType.ADDED: QColor("#219653"),
    ChangeType.REMOVED: QColor("#d64545"),
    ChangeType.MOVED: QColor("#ed8b00"),
    ChangeType.MODIFIED: QColor("#246bce"),
}


class PageViewer(QGraphicsView):
    """Render a PDF page for review; pixels never feed the comparison engine."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#e8edf2"))
        self.setToolTip("Display preview only — comparison uses PDF vector/text entities.")
        self._page_bounds: QRectF | None = None

    def _message(self, message: str) -> None:
        item = self.scene().addText(message)
        item.setDefaultTextColor(QColor("#5b6773"))
        self.scene().setSceneRect(item.boundingRect().adjusted(-20, -20, 20, 20))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @staticmethod
    def _bbox_for_side(change: Change, old_side: bool):
        if old_side:
            return None if change.change_type == ChangeType.ADDED else change.old_bbox
        return change.bbox

    def load_page(
        self,
        pdf_path: str,
        page_index: int,
        changes: Iterable[Change] = (),
        selected_id: str | None = None,
        *,
        old_side: bool = False,
    ) -> None:
        self.scene().clear()
        self.resetTransform()
        self._page_bounds = None
        source = Path(pdf_path)
        if not source.is_file():
            self._message("Page not present in this revision")
            return

        try:
            with fitz.open(source) as document:
                if page_index < 0 or page_index >= document.page_count:
                    self._message("Page not present in this revision")
                    return
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                image = QImage(
                    pixmap.samples,
                    pixmap.width,
                    pixmap.height,
                    pixmap.stride,
                    QImage.Format.Format_RGB888,
                ).copy()
        except Exception as exc:
            self._message(f"Preview unavailable: {exc}")
            return

        page_item = self.scene().addPixmap(QPixmap.fromImage(image))
        page_item.setZValue(0)
        self._page_bounds = page_item.boundingRect()
        selected_rect: QRectF | None = None
        for change in changes:
            bbox = self._bbox_for_side(change, old_side)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            rectangle = QRectF(
                x0 * pixmap.width,
                y0 * pixmap.height,
                max(2.0, (x1 - x0) * pixmap.width),
                max(2.0, (y1 - y0) * pixmap.height),
            )
            color = _COLORS[change.change_type]
            is_selected = change.id == selected_id
            pen = QPen(color, 3.0 if is_selected else 1.5)
            pen.setCosmetic(True)
            overlay = self.scene().addRect(
                rectangle,
                pen,
                QColor(color.red(), color.green(), color.blue(), 48 if is_selected else 28),
            )
            overlay.setToolTip(
                f"{change.change_type.value.title()} · {change.category.value}\n{change.detail}"
            )
            overlay.setZValue(1)
            if is_selected:
                selected_rect = rectangle

        self.scene().setSceneRect(self._page_bounds)
        if selected_rect is None:
            self.fitInView(self._page_bounds, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            padding = max(35.0, max(selected_rect.width(), selected_rect.height()) * 1.5)
            focus = selected_rect.adjusted(-padding, -padding, padding, padding)
            self.fitInView(focus.intersected(self._page_bounds), Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(selected_rect.center())

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._page_bounds is not None:
            self.fitInView(self._page_bounds, Qt.AspectRatioMode.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)

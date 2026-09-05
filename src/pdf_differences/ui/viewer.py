"""Single-view red/blue overlay for reviewing two PDF revisions."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    pyqtProperty,
)
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform, QWheelEvent
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from pdf_differences.models import Change, ChangeType, PageResult, Transform

_OLD_COLOR = QColor(220, 35, 35)
_NEW_COLOR = QColor(25, 90, 230)
_OTHER_COLOR = QColor(170, 80, 180)
_RENDER_SCALE = 1.6


class ChangeRegionItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, change_id: str, on_clicked: Callable[[str], None]) -> None:
        super().__init__(rect)
        self._change_id = change_id
        self._on_clicked = on_clicked
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)

    def mousePressEvent(self, event) -> None:
        self._on_clicked(self._change_id)
        event.accept()


class OverlayPageViewer(QGraphicsView):
    """Display aligned revisions in one scene; rendered pixels are never analyzed."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setToolTip(
            "Display preview only. Comparison still uses native PDF vector and text entities."
        )
        self._page_bounds: QRectF | None = None
        self._page_index: int | None = None
        self._selected_id: str | None = None
        self._layers: dict[str, QGraphicsPixmapItem] = {}
        self._regions: dict[str, QGraphicsRectItem] = {}
        self._region_colors: dict[str, QColor] = {}
        self._region_types: dict[str, ChangeType] = {}
        self._region_clicked: Callable[[str], None] | None = None
        self._blend = 50
        self._show_added = True
        self._show_removed = True
        self._show_regions = True
        self._blink_regions = True
        self._pulse_strength = 0.0
        self._pulse_animation = QPropertyAnimation(self, b"pulseStrength", self)
        self._pulse_animation.setDuration(1100)
        self._pulse_animation.setStartValue(0.0)
        self._pulse_animation.setKeyValueAt(0.5, 1.0)
        self._pulse_animation.setEndValue(0.0)
        self._pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_animation.setLoopCount(-1)

    @staticmethod
    def _render_page_layers(
        pdf_path: str,
        page_index: int,
        tint: QColor,
    ) -> tuple[QPixmap, QPixmap] | None:
        source = Path(pdf_path)
        if not source.is_file():
            return None
        with fitz.open(source) as document:
            if page_index < 0 or page_index >= document.page_count:
                return None
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(_RENDER_SCALE, _RENDER_SCALE),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            rgb = np.ndarray(
                (pixmap.height, pixmap.width, 3),
                dtype=np.uint8,
                buffer=pixmap.samples,
                strides=(pixmap.stride, 3, 1),
            )
            original_image = QImage(
                rgb.data,
                pixmap.width,
                pixmap.height,
                pixmap.stride,
                QImage.Format.Format_RGB888,
            ).copy()
            luminance = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(
                np.uint8
            )
            rgba = np.empty((pixmap.height, pixmap.width, 4), dtype=np.uint8)
            rgba[:, :, 0] = tint.red()
            rgba[:, :, 1] = tint.green()
            rgba[:, :, 2] = tint.blue()
            rgba[:, :, 3] = 255 - luminance
            tint_image = QImage(
                rgba.data,
                pixmap.width,
                pixmap.height,
                rgba.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            return QPixmap.fromImage(original_image), QPixmap.fromImage(tint_image)

    @staticmethod
    def _old_to_new_transform(
        transform: Transform,
        old_pixel_width: float,
        old_pixel_height: float,
        new_pixel_width: float,
        new_pixel_height: float,
    ) -> QTransform:
        cosine = math.cos(transform.rotation_radians)
        sine = math.sin(transform.rotation_radians)
        scale = transform.scale
        return QTransform(
            new_pixel_width * scale * cosine / old_pixel_width,
            new_pixel_height * scale * sine / old_pixel_width,
            -new_pixel_width * scale * sine / old_pixel_height,
            new_pixel_height * scale * cosine / old_pixel_height,
            new_pixel_width * transform.tx,
            new_pixel_height * transform.ty,
        )

    def _message(self, message: str) -> None:
        item = self.scene().addText(message)
        self.scene().setSceneRect(item.boundingRect().adjusted(-20, -20, 20, 20))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_page(
        self,
        old_pdf_path: str,
        new_pdf_path: str,
        page: PageResult,
        changes: Iterable[Change] = (),
        selected_id: str | None = None,
    ) -> None:
        self._pulse_animation.stop()
        self.scene().clear()
        self.resetTransform()
        self._page_bounds = None
        self._page_index = page.page_index
        self._selected_id = None
        self._layers.clear()
        self._regions.clear()
        self._region_colors.clear()
        self._region_types.clear()

        try:
            old_render = self._render_page_layers(old_pdf_path, page.page_index, _OLD_COLOR)
            new_render = self._render_page_layers(new_pdf_path, page.page_index, _NEW_COLOR)
        except Exception as exc:
            self._message(f"Preview unavailable: {exc}")
            return
        if old_render is None and new_render is None:
            self._message("Page is not present in either revision")
            return

        target_pixmap = new_render[0] if new_render is not None else old_render[0]
        width = float(target_pixmap.width())
        height = float(target_pixmap.height())
        self._page_bounds = QRectF(0.0, 0.0, width, height)
        background = self.scene().addRect(
            self._page_bounds,
            QPen(Qt.PenStyle.NoPen),
            QBrush(Qt.GlobalColor.white),
        )
        background.setZValue(-1)

        old_transform = None
        if old_render is not None and new_render is not None:
            old_transform = self._old_to_new_transform(
                page.alignment.transform,
                old_render[0].width(),
                old_render[0].height(),
                width,
                height,
            )
        if old_render is not None:
            for name, pixmap, z_value in (
                ("old_original", old_render[0], 0),
                ("old_tint", old_render[1], 1),
            ):
                item = self.scene().addPixmap(pixmap)
                item.setZValue(z_value)
                if old_transform is not None:
                    item.setTransform(old_transform)
                self._layers[name] = item
        if new_render is not None:
            new_item = self.scene().addPixmap(new_render[0])
            new_item.setZValue(2)
            self._layers["new_original"] = new_item
            new_tint_item = self.scene().addPixmap(new_render[1])
            new_tint_item.setZValue(3)
            self._layers["new_tint"] = new_tint_item

        selected_rect: QRectF | None = None
        for change in changes:
            x0, y0, x1, y1 = change.bbox
            rectangle = QRectF(
                x0 * width,
                y0 * height,
                max(2.0, (x1 - x0) * width),
                max(2.0, (y1 - y0) * height),
            )
            color = self._color_for_change(change.change_type)
            region = ChangeRegionItem(rectangle, change.id, self._handle_region_clicked)
            region.setData(0, change.id)
            region.setToolTip(
                f"{change.change_type.value.title()} · {change.category.value}\n{change.detail}"
            )
            region.setZValue(10)
            self.scene().addItem(region)
            self._regions[change.id] = region
            self._region_colors[change.id] = color
            self._region_types[change.id] = change.change_type
            if change.id == selected_id:
                selected_rect = rectangle

        scene_margin = max(width, height) * 0.3
        self.scene().setSceneRect(
            self._page_bounds.adjusted(-scene_margin, -scene_margin, scene_margin, scene_margin)
        )
        self._apply_state()
        if selected_rect is None:
            self.fit_to_page()
        else:
            self.focus_change(selected_id)

    @staticmethod
    def _color_for_change(change_type: ChangeType) -> QColor:
        if change_type == ChangeType.ADDED:
            return _NEW_COLOR
        if change_type == ChangeType.REMOVED:
            return _OLD_COLOR
        return _OTHER_COLOR

    def set_blend(self, value: int) -> None:
        self._blend = max(0, min(100, int(value)))
        self._apply_state()

    def blend(self) -> int:
        return self._blend

    def show_additions(self, enabled: bool) -> None:
        self._show_added = bool(enabled)
        self._apply_state()

    def show_removals(self, enabled: bool) -> None:
        self._show_removed = bool(enabled)
        self._apply_state()

    def show_regions(self, enabled: bool) -> None:
        self._show_regions = bool(enabled)
        self._apply_state()

    def blink_regions(self, enabled: bool) -> None:
        self._blink_regions = bool(enabled)
        self._apply_state()

    def _region_is_enabled(self, change_id: str) -> bool:
        change_type = self._region_types[change_id]
        if change_type == ChangeType.ADDED:
            return self._show_added
        if change_type == ChangeType.REMOVED:
            return self._show_removed
        return True

    def _apply_state(self) -> None:
        t = self._blend / 100.0
        edge = 1.0 - min(1.0, abs(2.0 * t - 1.0))
        left = max(0.0, 1.0 - 2.0 * t)
        right = max(0.0, 2.0 * t - 1.0)
        if "old_original" in self._layers:
            self._layers["old_original"].setOpacity(left)
        if "old_tint" in self._layers:
            self._layers["old_tint"].setOpacity(edge)
        if "new_original" in self._layers:
            self._layers["new_original"].setOpacity(right)
        if "new_tint" in self._layers:
            self._layers["new_tint"].setOpacity(edge)
        for change_id, region in self._regions.items():
            region.setVisible(self._show_regions and self._region_is_enabled(change_id))
        self._refresh_pulse_animation()
        self._set_pulse_strength(self._pulse_strength)

    def _get_pulse_strength(self) -> float:
        return self._pulse_strength

    def _set_pulse_strength(self, strength: float) -> None:
        self._pulse_strength = max(0.0, min(1.0, float(strength)))
        alpha = round(110 + 145 * self._pulse_strength)
        fill_alpha = round(16 + 54 * self._pulse_strength)
        width = 1.5 + 2.0 * self._pulse_strength
        for change_id, region in self._regions.items():
            color = self._region_colors[change_id]
            selected = change_id == self._selected_id
            pen_alpha = 255 if selected else alpha
            pen_width = width + 1.5 if selected else width
            pen = QPen(
                QColor(color.red(), color.green(), color.blue(), pen_alpha),
                pen_width,
            )
            pen.setCosmetic(True)
            region.setPen(pen)
            region.setBrush(
                QBrush(
                    QColor(
                        color.red(),
                        color.green(),
                        color.blue(),
                        max(fill_alpha, 72) if selected else fill_alpha,
                    )
                )
            )

    pulseStrength = pyqtProperty(float, _get_pulse_strength, _set_pulse_strength)

    def _refresh_pulse_animation(self) -> None:
        should_run = self._blink_regions and any(
            region.isVisible() for region in self._regions.values()
        )
        if should_run and self._pulse_animation.state() != QAbstractAnimation.State.Running:
            self._pulse_animation.start()
        elif not should_run and self._pulse_animation.state() == QAbstractAnimation.State.Running:
            self._pulse_animation.stop()
            self._set_pulse_strength(0.0)

    def fit_to_page(self) -> None:
        if self._page_bounds is not None:
            self.fitInView(self._page_bounds, Qt.AspectRatioMode.KeepAspectRatio)

    @property
    def page_index(self) -> int | None:
        return self._page_index

    def focus_change(self, change_id: str | None, *, zoom: bool = True) -> bool:
        if change_id is None or change_id not in self._regions:
            return False
        self._selected_id = change_id
        self.scene().clearSelection()
        region = self._regions[change_id]
        region.setSelected(True)
        self._set_pulse_strength(self._pulse_strength)
        if zoom:
            rectangle = region.sceneBoundingRect()
            center = rectangle.center()
            focus_width = max(rectangle.width() * 1.25, self._page_bounds.width() * 0.08)
            focus_height = max(rectangle.height() * 1.25, self._page_bounds.height() * 0.08)
            focus = QRectF(
                center.x() - focus_width / 2.0,
                center.y() - focus_height / 2.0,
                focus_width,
                focus_height,
            )
            self.fitInView(focus, Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(center)
            QTimer.singleShot(0, lambda ident=change_id: self._recenter_change(ident))
        return True

    def _recenter_change(self, change_id: str) -> None:
        if change_id == self._selected_id and change_id in self._regions:
            center = self._regions[change_id].sceneBoundingRect().center()
            visible = self.mapToScene(self.viewport().rect()).boundingRect()
            required = QRectF(
                center.x() - visible.width() / 2.0,
                center.y() - visible.height() / 2.0,
                visible.width(),
                visible.height(),
            )
            self.scene().setSceneRect(self.sceneRect().united(required))
            self.centerOn(center)

    def reset_view(self) -> None:
        self.resetTransform()
        if self._page_bounds is not None:
            self.centerOn(self._page_bounds.center())

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit_to_page()
        super().mouseDoubleClickEvent(event)

    def set_region_clicked_handler(self, handler: Callable[[str], None]) -> None:
        self._region_clicked = handler

    def _handle_region_clicked(self, change_id: str) -> None:
        self.focus_change(change_id)
        if self._region_clicked is not None:
            self._region_clicked(change_id)


# Kept as a compatibility alias for callers that imported the old viewer class.
PageViewer = OverlayPageViewer

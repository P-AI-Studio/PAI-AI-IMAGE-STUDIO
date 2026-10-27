# -*- coding: utf-8 -*-
"""
Interactive zoom/pan before-after preview canvas with Single, Side-by-Side, Split, and Flicker modes.
"""
from __future__ import annotations

from typing import Optional

from PIL import Image, ImageChops, ImageEnhance

from ..validation import preview_click_to_normalized

try:
    from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
    from PySide6.QtGui import (
        QColor,
        QImage,
        QMouseEvent,
        QPainter,
        QPen,
        QPixmap,
        QWheelEvent,
    )
    from PySide6.QtWidgets import QWidget
except ImportError:
    class QWidget:  # type: ignore[no-redef]
        pass

    def Signal(*args, **kwargs):  # type: ignore[no-redef]
        return None

class PreviewCanvas(QWidget):
    focusSelected = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.original: Optional[Image.Image] = None
        self.enhanced: Optional[Image.Image] = None
        self._pix_original: Optional[QPixmap] = None
        self._pix_enhanced: Optional[QPixmap] = None
        self._pix_diff: Optional[QPixmap] = None
        self.mode = "Enhanced"
        self.presentation_policy = "live_only"
        self.live_caption = "LIVE · PREVIEW"
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.pane_zoom = {"original": 1.0, "enhanced": 1.0}
        self.pane_pan = {"original": QPoint(0, 0), "enhanced": QPoint(0, 0)}
        self._dragging = False
        self._split_dragging = False
        self._active_pane = "enhanced"
        self._last_pos = QPoint()
        self.focus_point: Optional[tuple[float, float]] = None
        self.closeup_crop_box: Optional[tuple[int, int, int, int]] = None
        self.split_value = 50
        self._last_target = QRect()
        self._pane_rects: dict[str, QRect] = {}
        self._pane_image_targets: dict[str, QRect] = {}
        self._last_focus_rect = QRect()
        self._flicker_show_original = False
        self.light_studio_skin = False
        self._flicker_timer = QTimer(self)
        self._flicker_timer.setInterval(500)
        self._flicker_timer.timeout.connect(self._on_flicker_tick)

    def set_studio_skin(self, enabled: bool):
        self.light_studio_skin = bool(enabled)
        self.update()

    def _on_flicker_tick(self):
        self._flicker_show_original = not self._flicker_show_original
        self.update()

    def _sync_flicker_timer(self):
        if self.mode == "Flicker" and self.original is not None and self.enhanced is not None:
            if not self._flicker_timer.isActive():
                self._flicker_timer.start()
        else:
            self._flicker_timer.stop()

    def set_images(self, original: Optional[Image.Image], enhanced: Optional[Image.Image]):
        self.original = original.copy() if original is not None else None
        self.enhanced = enhanced.copy() if enhanced is not None else None
        # QPixmap conversion happens only when images change, never on each paint.
        self._pix_original = self._pixmap(self.original) if self.original is not None else None
        self._pix_enhanced = self._pixmap(self.enhanced) if self.enhanced is not None else None
        self._pix_diff = None
        self._pane_rects = {}
        self._pane_image_targets = {}
        if self.original is not None and self.enhanced is not None:
            try:
                original_rgb = self.original.convert("RGB").resize(self.enhanced.size, Image.Resampling.BILINEAR)
                difference = ImageChops.difference(original_rgb, self.enhanced.convert("RGB"))
                self._pix_diff = self._pixmap(ImageEnhance.Contrast(difference).enhance(3.0))
            except Exception:
                self._pix_diff = None
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.pane_zoom = {"original": 1.0, "enhanced": 1.0}
        self.pane_pan = {"original": QPoint(0, 0), "enhanced": QPoint(0, 0)}
        self._flicker_show_original = False
        self._sync_flicker_timer()
        self.update()

    def set_live_image(self, enhanced: Optional[Image.Image], *, preserve_view: bool = True):
        """Update only the Live pane without rebuilding the Before image or its difference map."""
        self.enhanced = enhanced.copy() if enhanced is not None else None
        self._pix_enhanced = self._pixmap(self.enhanced) if self.enhanced is not None else None
        # Difference rendering is not part of the enforced Live/Before workflows. If another
        # caller explicitly selected Difference, rebuild it only for that mode.
        self._pix_diff = None
        if self.mode == "Difference" and self.original is not None and self.enhanced is not None:
            try:
                original_rgb = self.original.convert("RGB").resize(self.enhanced.size, Image.Resampling.BILINEAR)
                difference = ImageChops.difference(original_rgb, self.enhanced.convert("RGB"))
                self._pix_diff = self._pixmap(ImageEnhance.Contrast(difference).enhance(3.0))
            except Exception:
                self._pix_diff = None
        if not preserve_view:
            self.zoom = 1.0
            self.pan = QPoint(0, 0)
            self.pane_zoom = {"original": 1.0, "enhanced": 1.0}
            self.pane_pan = {"original": QPoint(0, 0), "enhanced": QPoint(0, 0)}
        self._flicker_show_original = False
        self._sync_flicker_timer()
        self.update()

    def set_presentation_policy(self, policy: str):
        """Force Live-only for generation, otherwise permit a selectable view mode."""
        self.presentation_policy = policy if policy in {"live_only", "before_live", "selectable"} else "live_only"
        if self.presentation_policy == "before_live":
            self.mode = "Side by side"
        elif self.presentation_policy == "live_only":
            self.mode = "Enhanced"
        self._sync_flicker_timer()
        self.update()

    def set_live_caption(self, caption: str):
        value = str(caption or "LIVE").strip()
        self.live_caption = value[:48] or "LIVE"
        self.update()

    def set_mode(self, mode: str):
        # Generate is live-only; source workflows may choose any comparison mode.
        if self.presentation_policy == "before_live":
            mode = "Side by side"
        elif self.presentation_policy == "live_only":
            mode = "Enhanced"
        self.mode = mode
        self._sync_flicker_timer()
        self.update()

    def _uses_side_by_side_layout(self) -> bool:
        """Whether input handling should address the two forced comparison panes."""
        return bool(self._pane_rects) and (
            self.mode == "Side by side" or self.presentation_policy == "before_live"
        )

    def zoom_in(self):
        pos = self._pane_rects.get(self._active_pane, self.rect()).center() if self._uses_side_by_side_layout() else self.rect().center()
        self._zoom_about(pos, 1.2)

    def zoom_out(self):
        pos = self._pane_rects.get(self._active_pane, self.rect()).center() if self._uses_side_by_side_layout() else self.rect().center()
        self._zoom_about(pos, 1 / 1.2)

    def fit_view(self):
        self.zoom = 1.0
        self.pan = QPoint(0, 0)
        self.pane_zoom = {"original": 1.0, "enhanced": 1.0}
        self.pane_pan = {"original": QPoint(0, 0), "enhanced": QPoint(0, 0)}
        self.update()

    @staticmethod
    def _pixmap(image: Image.Image) -> QPixmap:
        rgba = image.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")
        qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _centered_target(image: Image.Image, viewport: QRect, zoom: float, pan: QPoint) -> QRect:
        fit = min(max(1, viewport.width()) / max(1, image.width), max(1, viewport.height()) / max(1, image.height))
        scale = fit * zoom
        target_w = max(1, int(round(image.width * scale)))
        target_h = max(1, int(round(image.height * scale)))
        return QRect(viewport.x() + (viewport.width() - target_w) // 2 + pan.x(),
                     viewport.y() + (viewport.height() - target_h) // 2 + pan.y(), target_w, target_h)

    def _paint_single(self, painter: QPainter, image: Image.Image, pixmap: Optional[QPixmap],
                      viewport: QRect, zoom: float, pan: QPoint) -> QRect:
        target = self._centered_target(image, viewport, zoom, pan)
        if pixmap is not None:
            painter.save()
            painter.setClipRect(viewport)
            painter.drawPixmap(target, pixmap)
            painter.restore()
        return target

    def _paint_placeholder(self, painter: QPainter, viewport: QRect, text: str):
        color = "#f0f2f9" if self.light_studio_skin else "#171a20"
        text_color = "#7f8ba6" if self.light_studio_skin else "#9aa4b2"
        painter.fillRect(viewport, QColor(color))
        painter.setPen(QColor(text_color))
        painter.drawText(viewport, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_focus_marker(self, painter: QPainter, target: Optional[QRect], viewport: Optional[QRect] = None,
                           focus_xy: Optional[tuple[float, float]] = None):
        """Draw the selected normalized focus point over the actual painted image."""
        pt = focus_xy if focus_xy is not None else self.focus_point
        if not pt or target is None or target.isEmpty():
            return
        if not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
            return
        fx = target.left() + int(pt[0] * target.width())
        fy = target.top() + int(pt[1] * target.height())
        point = QPoint(fx, fy)
        if viewport is not None and not viewport.contains(point):
            return
        painter.save()
        if viewport is not None:
            painter.setClipRect(viewport)
        painter.setPen(QPen(QColor("#ffcc66"), 2))
        painter.drawEllipse(point, 8, 8)
        painter.drawLine(fx - 13, fy, fx + 13, fy)
        painter.drawLine(fx, fy - 13, fx, fy + 13)
        painter.restore()

    def _paint_live_only(self, painter: QPainter):
        self._pane_rects = {}
        margin, label_h = 10, 28
        header = QRect(margin, margin, max(1, self.width() - margin * 2), label_h)
        viewport = QRect(margin, margin + label_h, max(1, self.width() - margin * 2), max(1, self.height() - margin * 2 - label_h))
        pane_background = "#ffffff" if self.light_studio_skin else "#202630"
        label_color = "#159a88" if self.light_studio_skin else "#6ee7d8"
        painter.fillRect(header, QColor(pane_background))
        painter.setPen(QColor(label_color))
        painter.drawText(header, Qt.AlignmentFlag.AlignCenter, self.live_caption)
        painter.setPen(QPen(QColor("#d8deed" if self.light_studio_skin else "#3c4553"), 1))
        painter.drawRect(viewport)
        self._pane_image_targets = {}
        if self.enhanced is None:
            self._paint_placeholder(painter, viewport, "Waiting for live preview…")
            self._last_target = viewport
            self._last_focus_rect = QRect()
            return
        target = self._paint_single(painter, self.enhanced, self._pix_enhanced, viewport, self.zoom, self.pan)
        self._pane_image_targets["enhanced"] = target
        self._last_target = target
        self._last_focus_rect = target
        self._draw_focus_marker(painter, target, viewport)

    def _paint_side_by_side(self, painter: QPainter):
        margin, gap, label_h = 10, 10, 26
        pane_w = max(1, (self.width() - margin * 2 - gap) // 2)
        top = margin + label_h
        pane_h = max(1, self.height() - top - margin)
        left = QRect(margin, top, pane_w, pane_h)
        right = QRect(margin + pane_w + gap, top, pane_w, pane_h)
        self._pane_rects = {"original": left, "enhanced": right}
        self._pane_image_targets = {}
        pane_background = "#ffffff" if self.light_studio_skin else "#202630"
        before_color = "#7659e8" if self.light_studio_skin else "#81c7ff"
        after_color = "#159a88" if self.light_studio_skin else "#6ee7d8"
        pane_border = "#d8deed" if self.light_studio_skin else "#3c4553"
        painter.fillRect(left.adjusted(-1, -label_h, 1, 1), QColor(pane_background))
        painter.fillRect(right.adjusted(-1, -label_h, 1, 1), QColor(pane_background))
        painter.setPen(QColor(before_color))
        painter.drawText(QRect(left.x(), margin, left.width(), label_h), Qt.AlignmentFlag.AlignCenter, "BEFORE · ORIGINAL")
        painter.setPen(QColor(after_color))
        painter.drawText(QRect(right.x(), margin, right.width(), label_h), Qt.AlignmentFlag.AlignCenter, "LIVE · PREVIEW / RESULT")
        painter.setPen(QPen(QColor(pane_border), 1))
        painter.drawRect(left)
        painter.drawRect(right)
        if self.original is not None:
            self._pane_image_targets["original"] = self._paint_single(
                painter, self.original, self._pix_original, left,
                self.pane_zoom["original"], self.pane_pan["original"])
        else:
            self._paint_placeholder(painter, left, "Waiting for source image")
        if self.enhanced is not None:
            self._pane_image_targets["enhanced"] = self._paint_single(
                painter, self.enhanced, self._pix_enhanced, right,
                self.pane_zoom["enhanced"], self.pane_pan["enhanced"])
        else:
            self._paint_placeholder(painter, right, "Waiting for live preview")
        divider_color = "#c9d2e5" if self.light_studio_skin else "#515d70"
        painter.setPen(QPen(QColor(divider_color), 1))
        painter.drawLine(self.width() // 2, 8, self.width() // 2, self.height() - 8)
        # The focus point is selected in normalized source coordinates. Render it on
        # both panes so it remains visible after the comparison layout is forced.
        self._draw_focus_marker(painter, self._pane_image_targets.get("original"), left)
        if (self.focus_point and self.closeup_crop_box is not None and self.original is not None
                and self.original.width > 0 and self.original.height > 0):
            cl, ct, cr, cb = self.closeup_crop_box
            cw, ch = max(1, cr - cl), max(1, cb - ct)
            enhanced_focus = (
                (self.focus_point[0] * self.original.width - cl) / cw,
                (self.focus_point[1] * self.original.height - ct) / ch,
            )
            self._draw_focus_marker(painter, self._pane_image_targets.get("enhanced"), right, focus_xy=enhanced_focus)
        else:
            self._draw_focus_marker(painter, self._pane_image_targets.get("enhanced"), right)

    def paintEvent(self, event):
        painter = QPainter(self)
        canvas_background = "#f0f2f9" if self.light_studio_skin else "#171a20"
        painter.fillRect(self.rect(), QColor(canvas_background))
        if self.presentation_policy == "before_live":
            self._paint_side_by_side(painter)
            return
        if self.presentation_policy == "live_only":
            self._paint_live_only(painter)
            return
        if self.mode == "Side by side" and (self.original is not None or self.enhanced is not None):
            self._paint_side_by_side(painter)
            return
        show_original_only = (self.mode == "Original" and self.original is not None)
        image = self.original if show_original_only else (self.enhanced or self.original)
        if image is None:
            placeholder_color = "#7f8ba6" if self.light_studio_skin else "#9aa4b2"
            painter.setPen(QColor(placeholder_color))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Drop an image here or add one from the queue")
            return
        target = self._centered_target(image, self.rect(), self.zoom, self.pan)
        active_pane_key = "original" if show_original_only else ("enhanced" if self.enhanced is not None else "original")
        self._pane_image_targets = {active_pane_key: target}
        self._last_target = target
        self._last_focus_rect = target
        pix = self._pix_enhanced if self.enhanced is not None else self._pix_original
        if self.mode == "Original" and self._pix_original is not None:
            painter.drawPixmap(target, self._pix_original)
        elif self.mode == "Overlay" and self._pix_original is not None and self._pix_enhanced is not None:
            painter.drawPixmap(target, self._pix_original)
            painter.save(); painter.setOpacity(0.5); painter.drawPixmap(target, self._pix_enhanced); painter.restore()
        elif self.mode == "Difference" and self._pix_diff is not None:
            painter.drawPixmap(target, self._pix_diff)
        elif self.mode == "Slider before/after" and self._pix_original is not None and self._pix_enhanced is not None:
            # Divider drag changes only the clip boundary; panning is handled elsewhere.
            painter.drawPixmap(target, self._pix_enhanced)
            clip_x = target.left() + int(target.width() * self.split_value / 100)
            painter.save()
            painter.setClipRect(QRect(target.left(), target.top(), max(0, clip_x - target.left()), target.height()))
            painter.drawPixmap(target, self._pix_original)
            painter.restore()
            painter.setPen(QPen(QColor("#eef6ff"), 2))
            painter.drawLine(clip_x, target.top(), clip_x, target.bottom())
            painter.setBrush(QColor("#eef6ff")); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(clip_x, self.height() // 2), 7, 7)
        elif self.mode == "Flicker" and self._pix_original is not None and self._pix_enhanced is not None:
            painter.drawPixmap(target, self._pix_original if self._flicker_show_original else self._pix_enhanced)
        elif pix is not None:
            painter.drawPixmap(target, pix)
        self._draw_focus_marker(painter, target, self.rect())

    def _zoom_about(self, pos: QPoint, factor: float):
        if self._uses_side_by_side_layout():
            if self._pane_rects["original"].contains(pos):
                pane = "original"
            else:
                pane = "enhanced"
            viewport = self._pane_rects[pane]
            old_zoom = self.pane_zoom[pane]
            new_zoom = max(0.25, min(8.0, old_zoom * factor))
            pan = self.pane_pan[pane]
            center_x = viewport.x() + viewport.width() / 2
            center_y = viewport.y() + viewport.height() / 2
            ratio = new_zoom / old_zoom
            new_x = pos.x() - (pos.x() - center_x - pan.x()) * ratio - center_x
            new_y = pos.y() - (pos.y() - center_y - pan.y()) * ratio - center_y
            self.pane_zoom[pane] = new_zoom
            self.pane_pan[pane] = QPoint(int(new_x), int(new_y))
            self._active_pane = pane
        else:
            old_zoom = self.zoom
            new_zoom = max(0.25, min(8.0, old_zoom * factor))
            center_x = self.width() / 2; center_y = self.height() / 2
            ratio = new_zoom / old_zoom
            new_x = pos.x() - (pos.x() - center_x - self.pan.x()) * ratio - center_x
            new_y = pos.y() - (pos.y() - center_y - self.pan.y()) * ratio - center_y
            self.zoom = new_zoom
            self.pan = QPoint(int(new_x), int(new_y))
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        self._zoom_about(event.position().toPoint(), 1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.RightButton:
            # Use the rectangle of the actual image that was painted, not the last
            # single-pane rectangle: in Side-by-Side that stale rectangle caused
            # clicks in a pane/letterbox to clamp to (1.0, 1.0), i.e. "1000,1000".
            show_original_only = (self.mode == "Original" and self.original is not None)
            pane_key = "original" if show_original_only else ("enhanced" if self.enhanced is not None else "original")
            viewport = self.rect()
            if self._uses_side_by_side_layout():
                if self._pane_rects["original"].contains(pos):
                    pane_key = "original"
                    viewport = self._pane_rects["original"]
                elif self._pane_rects["enhanced"].contains(pos):
                    pane_key = "enhanced"
                    viewport = self._pane_rects["enhanced"]
                else:
                    return
            target = self._pane_image_targets.get(pane_key)
            if target is None or not viewport.contains(pos) or not target.contains(pos):
                return
            normalized = preview_click_to_normalized(
                pos.x(), pos.y(),
                (target.left(), target.top(), target.left() + target.width(), target.top() + target.height()),
            )
            if normalized is None:
                return
            if (pane_key == "enhanced" and self.closeup_crop_box is not None
                    and self.original is not None and self.original.width > 0 and self.original.height > 0):
                cl, ct, cr, cb = self.closeup_crop_box
                cw, ch = max(1, cr - cl), max(1, cb - ct)
                fx = min(0.999, max(0.0, (cl + normalized[0] * cw) / float(self.original.width)))
                fy = min(0.999, max(0.0, (ct + normalized[1] * ch) / float(self.original.height)))
                normalized = (fx, fy)
            self.focus_point = normalized
            self.focusSelected.emit(*self.focus_point)
            self.update()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "Slider before/after" and self.original is not None and self.enhanced is not None and self._last_target.contains(pos):
                line_x = self._last_target.left() + int(self._last_target.width() * self.split_value / 100)
                if abs(pos.x() - line_x) <= 18:
                    self._split_dragging = True
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                    return
            self._active_pane = "enhanced"
            if self._uses_side_by_side_layout():
                if self._pane_rects["original"].contains(pos):
                    self._active_pane = "original"
                elif self._pane_rects["enhanced"].contains(pos):
                    self._active_pane = "enhanced"
                else:
                    self._dragging = False
                    return
            self._dragging = True
            self._last_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        if self._uses_side_by_side_layout():
            if self._pane_rects["original"].contains(pos):
                self._active_pane = "original"
            elif self._pane_rects["enhanced"].contains(pos):
                self._active_pane = "enhanced"
        if self._split_dragging and self._last_target.width() > 0:
            value = round(100 * (pos.x() - self._last_target.left()) / max(1, self._last_target.width()))
            self.split_value = max(0, min(100, int(value)))
            self.update()
        elif self._dragging:
            delta = pos - self._last_pos
            if self._uses_side_by_side_layout():
                self.pane_pan[self._active_pane] += delta
            else:
                self.pan += delta
            self._last_pos = pos
            self.update()
        elif self.mode == "Slider before/after" and self._last_target.contains(pos) and self.original is not None and self.enhanced is not None:
            line_x = self._last_target.left() + int(self._last_target.width() * self.split_value / 100)
            self.setCursor(Qt.CursorShape.SplitHCursor if abs(pos.x() - line_x) <= 18 else Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._split_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)


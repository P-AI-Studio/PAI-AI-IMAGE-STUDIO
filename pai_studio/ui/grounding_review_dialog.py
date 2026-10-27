"""Qt review UI and pure geometry contracts for semantic grounding proposals."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from ..validation import validate_normalized_bbox


@dataclass(frozen=True)
class GroundingReviewDecision:
    """A deliberate user decision for a proposed Close-up target box."""

    action: str
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        if self.action not in {"accept", "retry", "cancel"}:
            raise ValueError("Grounding review action must be accept, retry, or cancel.")
        if self.action == "accept":
            if self.bbox_xyxy is None:
                raise ValueError("Accepting a grounding proposal requires a bounding box.")
            object.__setattr__(self, "bbox_xyxy", validate_normalized_bbox(self.bbox_xyxy, "review_bbox"))
        elif self.bbox_xyxy is not None:
            raise ValueError("Retry and cancel decisions cannot include a bounding box.")


def should_accept_review_job(result_job_id: str, current_job_id: str, cancelled: bool = False) -> bool:
    """Return whether a modal review request still belongs to the active job."""
    return bool(result_job_id and current_job_id and result_job_id == current_job_id and not cancelled)


def grounding_candidate_label(index: int, description: str, bbox_xyxy) -> str:
    """Build a single-line candidate label, exposing model-generated semantics when present."""
    try:
        number = max(1, int(index))
    except (TypeError, ValueError, OverflowError):
        number = 1
    clean = " ".join(str(description or "").replace("\x00", " ").split())[:90]
    try:
        coords = tuple(float(value) for value in bbox_xyxy)
    except (TypeError, ValueError, OverflowError):
        coords = ()
    coord_text = ", ".join(f"{value:.3f}" for value in coords) if len(coords) == 4 else "coordinates unavailable"
    prefix = f"{clean} · Candidate {number}" if clean else f"Candidate {number}"
    return f"{prefix} · {coord_text}"


def map_preview_drag_to_image_box(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    content_rect: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a drag in a letterboxed preview to normalized image XYXY coordinates.

    ``content_rect`` is (left, top, width, height) of the *actual image content*
    within the preview widget, not the outer widget rectangle.
    """
    left, top, width, height = (float(v) for v in content_rect)
    image_width, image_height = (int(v) for v in image_size)
    values = (*start_xy, *end_xy, left, top, width, height)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Preview drag coordinates must be finite.")
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("Preview rectangle and image dimensions must be positive.")

    right = left + width
    bottom = top + height
    sx = min(right, max(left, float(start_xy[0])))
    ex = min(right, max(left, float(end_xy[0])))
    sy = min(bottom, max(top, float(start_xy[1])))
    ey = min(bottom, max(top, float(end_xy[1])))
    x1, x2 = sorted(((sx - left) / width, (ex - left) / width))
    y1, y2 = sorted(((sy - top) / height, (ey - top) / height))
    bbox = (x1, y1, x2, y2)
    return validate_normalized_bbox(bbox, "review_bbox")


try:
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_AVAILABLE = True
except ImportError:  # Keep geometry and decision contracts testable in headless CI.
    QT_AVAILABLE = False


if QT_AVAILABLE:
    class GroundingBoxPreview(QWidget):
        boxEdited = Signal(float, float, float, float)

        def __init__(self, image, bbox_xyxy=None, parent=None):
            super().__init__(parent)
            self.setMinimumSize(360, 260)
            self.setMouseTracking(False)
            self._image_size = (max(1, image.width), max(1, image.height))
            rgba = image.convert("RGBA")
            self._pixmap = QPixmap.fromImage(QImage(
                rgba.tobytes("raw", "RGBA"), rgba.width, rgba.height, QImage.Format.Format_RGBA8888
            ).copy())
            self._bbox = validate_normalized_bbox(bbox_xyxy, "preview_bbox") if bbox_xyxy is not None else None
            self._drag_start: Optional[QPointF] = None
            self._drag_current: Optional[QPointF] = None
            self._drag_box: Optional[tuple[float, float, float, float]] = None

        def set_box(self, bbox_xyxy):
            self._bbox = validate_normalized_bbox(bbox_xyxy, "preview_bbox")
            self.update()

        def content_rect(self) -> QRectF:
            bounds = QRectF(self.rect()).adjusted(5, 5, -5, -5)
            if bounds.width() <= 0 or bounds.height() <= 0:
                return QRectF()
            aspect = self._image_size[0] / self._image_size[1]
            width, height = bounds.width(), bounds.width() / aspect
            if height > bounds.height():
                height = bounds.height()
                width = height * aspect
            return QRectF(bounds.x() + (bounds.width() - width) / 2,
                          bounds.y() + (bounds.height() - height) / 2, width, height)

        def _to_box(self, start: QPointF, end: QPointF):
            rect = self.content_rect()
            return map_preview_drag_to_image_box(
                (start.x(), start.y()), (end.x(), end.y()),
                (rect.x(), rect.y(), rect.width(), rect.height()), self._image_size,
            )

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#20242a"))
            rect = self.content_rect()
            if rect.isNull() or rect.isEmpty():
                painter.end()
                return
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
            box = self._drag_box or self._bbox
            if box is not None:
                x1, y1, x2, y2 = box
                shape = QRectF(rect.x() + x1 * rect.width(), rect.y() + y1 * rect.height(),
                               (x2 - x1) * rect.width(), (y2 - y1) * rect.height())
                painter.setPen(QPen(QColor("#48d597"), 2))
                painter.setBrush(QColor(72, 213, 151, 35))
                painter.drawRect(shape)
            if self._drag_start is not None and self._drag_current is not None:
                painter.setPen(QPen(QColor("#ffd166"), 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(self._drag_start, self._drag_current).normalized())
            painter.end()

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton and self.content_rect().contains(event.position()):
                self._drag_start = event.position()
                self._drag_current = event.position()
                self._drag_box = None
                self.update()
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._drag_start is not None:
                self._drag_current = event.position()
                try:
                    box = self._to_box(self._drag_start, self._drag_current)
                    if (box[2] - box[0]) > 0.005 and (box[3] - box[1]) > 0.005:
                        self._drag_box = box
                except ValueError:
                    pass
                self.update()
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
                end = event.position()
                try:
                    box = self._to_box(self._drag_start, end)
                    if (box[2] - box[0]) > 0.005 and (box[3] - box[1]) > 0.005:
                        self._bbox = box
                        self.boxEdited.emit(*box)
                except ValueError:
                    pass
                self._drag_start = None
                self._drag_current = None
                self._drag_box = None
                self.update()
                event.accept()
                return
            super().mouseReleaseEvent(event)


    class GroundingReviewDialog(QDialog):
        """Review, adjust, retry or cancel a Close-up grounding proposal."""

        def __init__(self, image, result, parent=None, *, allow_retry=True):
            super().__init__(parent)
            self.setWindowTitle("Review Close-up target")
            self.setModal(True)
            self.resize(720, 650)
            self._decision = GroundingReviewDecision("cancel")
            candidates = tuple(getattr(result, "candidates", ()) or ())
            initial = getattr(result, "bbox_xyxy", None)
            if initial is not None and initial not in candidates:
                candidates = (initial, *candidates)
            candidates = tuple(b for b in candidates if b is not None)
            candidate_descriptions = tuple(getattr(result, "candidate_descriptions", ()) or ())
            # Candidate descriptions are aligned by the grounding strategy after de-duplication.
            if len(candidate_descriptions) != len(candidates):
                candidate_descriptions = ()
            self._candidates = candidates

            layout = QVBoxLayout(self)
            title = QLabel("Confirm or adjust the target box before applying the crop.")
            title.setWordWrap(True)
            layout.addWidget(title)
            self.preview = GroundingBoxPreview(image, candidates[0] if candidates else None, self)
            layout.addWidget(self.preview, 1)

            self.candidate_combo = QComboBox(self)
            for index, box in enumerate(candidates, start=1):
                description = candidate_descriptions[index - 1] if candidate_descriptions else ""
                self.candidate_combo.addItem(grounding_candidate_label(index, description, box), box)
            if candidates:
                layout.addWidget(QLabel("Detected candidates (select one or draw a rectangle):"))
                layout.addWidget(self.candidate_combo)

            form = QFormLayout()
            self.box_inputs = []
            for label, value in zip(("Left (X1)", "Top (Y1)", "Right (X2)", "Bottom (Y2)"), candidates[0] if candidates else (0.1, 0.1, 0.9, 0.9)):
                spin = QDoubleSpinBox(self)
                spin.setRange(0.0, 1.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.01)
                spin.setValue(float(value))
                self.box_inputs.append(spin)
                form.addRow(label, spin)
                spin.valueChanged.connect(self._apply_spin_values)
            layout.addLayout(form)
            self.candidate_combo.currentIndexChanged.connect(self._select_candidate)
            self.preview.boxEdited.connect(self._set_box_values)

            buttons = QHBoxLayout()
            self.accept_button = QPushButton("Use this box", self)
            self.retry_button = QPushButton("Retry detection", self)
            self.cancel_button = QPushButton("Cancel", self)
            self.retry_button.setEnabled(bool(allow_retry))
            self.accept_button.clicked.connect(self._accept_box)
            self.retry_button.clicked.connect(self._retry)
            self.cancel_button.clicked.connect(self.reject)
            buttons.addWidget(self.accept_button)
            buttons.addWidget(self.retry_button)
            buttons.addWidget(self.cancel_button)
            layout.addLayout(buttons)
            self.error_label = QLabel("")
            self.error_label.setWordWrap(True)
            layout.addWidget(self.error_label)

        @property
        def decision(self) -> GroundingReviewDecision:
            return self._decision

        def _set_box_values(self, x1, y1, x2, y2):
            for spin, value in zip(self.box_inputs, (x1, y1, x2, y2)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self.preview.set_box((x1, y1, x2, y2))

        def _apply_spin_values(self, _value=None):
            box = tuple(spin.value() for spin in self.box_inputs)
            try:
                self.preview.set_box(validate_normalized_bbox(box, "review_bbox"))
                self.error_label.setText("")
            except ValueError as exc:
                self.error_label.setText(str(exc))

        def _select_candidate(self, index):
            if 0 <= index < len(self._candidates):
                self._set_box_values(*self._candidates[index])

        def _accept_box(self):
            try:
                self._decision = GroundingReviewDecision("accept", tuple(spin.value() for spin in self.box_inputs))
            except ValueError as exc:
                self.error_label.setText(str(exc))
                return
            self.accept()

        def _retry(self):
            self._decision = GroundingReviewDecision("retry")
            self.accept()
else:
    class GroundingBoxPreview:  # pragma: no cover - only available with PySide6
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PySide6 is required for the grounding review preview.")

    class GroundingReviewDialog:  # pragma: no cover - only available with PySide6
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PySide6 is required for the grounding review dialog.")

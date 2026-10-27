# -*- coding: utf-8 -*-
"""
Main desktop studio window (App) and application entrypoint for P.AI AI IMAGE STUDIO.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageChops, ImageOps, ImageStat

try:
    import psutil
except ImportError:
    psutil = None

try:
    from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, QUrl, Signal
    from PySide6.QtGui import (
        QAction,
        QColor,
        QContextMenuEvent,
        QDesktopServices,
        QDragEnterEvent,
        QDropEvent,
        QIcon,
        QImage,
        QMouseEvent,
        QPainter,
        QPen,
        QPixmap,
        QWheelEvent,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDockWidget,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    class QMainWindow:  # type: ignore[no-redef]
        pass

    def Signal(*args, **kwargs):  # type: ignore[no-redef]
        return None

from ..constants import (
    APP_NAME,
    DEFAULTS,
    DISPLAY_VERSION,
    EDIT_PRESETS,
    FIXED_SQUARE_OUTPUT_SIZES,
    GEN_PROMPT_PRESETS,
    IMAGE_EXTENSIONS,
    MAX_INFERENCE_SIDE,
    MAX_OUTPUT_SIDE,
    PROMPT_PRESETS,
    QUALITY_PRESETS,
    QUALITY_STEPS,
    QWEN_DETAIL_PROFILES,
    REALESRGAN_MODEL_NAMES,
    SETTINGS_VERSION,
    WORKFLOW_PROMPT_PRESETS,
    qwen_detail_profile_values,
    qwen_enhance_reconstruction_strengths,
)
from ..engine import Job, LocalEngine, detect_nvidia_gpus
from ..prompts import (
    build_edit_prompt,
    calculate_qwen_output_size,
    normalize_fixed_square_side,
)
from ..grounding import resolve_closeup_grounding_gguf_pair
from ..performance import grounding_profile_description, normalize_grounding_profile
from ..preview_policy import (
    live_preview_images_for_workflow,
    preview_presentation_for_workflow,
)
from ..validation import (
    ValidationError,
    validate_and_migrate_settings,
    validate_image_file,
)
from ..workers import (
    HardwareMetricsWorker,
    ModelBundleDownloadWorker,
    ModelLoadWorker,
    Worker,
)
from ..workspace import (
    _text_encoder_embeds_vision,
    APP_DIR,
    CACHE_DIR,
    CLOSEUPS_DIR,
    DATA_ROOT,
    EDITED_DIR,
    GENERATED_DIR,
    GROUNDING_RUNTIME_ASSETS,
    INPUT_DIR,
    LLAMA_GROUNDING_ARCHIVE_FILENAME,
    LLAMA_GROUNDING_RUNTIME_DIR,
    LORA_DIR,
    MODEL_DIR,
    OUTPAINT_DIR,
    OUTPUT_DIR,
    SETTINGS_FILE,
    _extract_llama_grounding_archive_safe,
    decode_saved_path,
    encode_saved_path,
    llama_grounding_runtime_present,
    resource_path,
)
from .canvas import PreviewCanvas
from .dialogs import ModelDownloadDialog

class App(QMainWindow):
    status_signal = Signal(str)
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._pending_log_messages: list[str] = []
        self.status_signal.connect(self._apply_status_message)
        self.log_signal.connect(self._append_log_message)
        self.setWindowTitle(f"{APP_NAME} · {DISPLAY_VERSION}")
        # Keep enough room for the workflow row while still fitting typical 1366×768 displays.
        self.setMinimumSize(1120, 720)
        self.resize(1600, 1020)
        self._job_started_at: Optional[float] = None
        self._first_step_at: Optional[float] = None
        self._last_step_index = 0
        self._last_step_total = 0
        self._step_interval_ema: Optional[float] = None
        self._last_step_changed_at: Optional[float] = None
        self._progress_started_at: Optional[float] = None
        self._active_job: Optional[Job] = None
        self._active_stage_name = "Idle"
        self._active_refinement_context: Optional[dict[str, int]] = None
        self._stop_after_current = False
        self.setAcceptDrops(True)
        self.settings = self._load_settings()
        self.detected_gpus = detect_nvidia_gpus()
        self.workflow = self.settings.get("workflow_mode", DEFAULTS["workflow_mode"])
        self.engine = LocalEngine(self.log_message, self.status_message)
        self.worker: Optional[Worker] = None
        self.model_worker: Optional[ModelLoadWorker] = None
        self.queue: list[dict[str, Any]] = []
        self.current_input: Optional[str] = None
        self.current_output: Optional[str] = None
        self.current_original: Optional[Image.Image] = None
        self.current_enhanced: Optional[Image.Image] = None
        self.focus_point: Optional[tuple[float, float]] = None
        self._restore_saved_prompt_presets = True
        self._build()
        catalog_target = Path(self.model_path_edit.text().strip() or str(MODEL_DIR)).expanduser()
        if not catalog_target.is_absolute():
            catalog_target = (APP_DIR / catalog_target).resolve()
        self.model_catalog_dialog = ModelDownloadDialog(catalog_target, self)
        self._live_preview_revision = -1
        self.live_preview_timer = QTimer(self)
        self.live_preview_timer.setInterval(500)
        self.live_preview_timer.timeout.connect(self._refresh_live_generation_preview)
        self.live_preview_timer.start()
        self.apply_workflow_preset(self.workflow, save=False)
        self._restore_saved_prompt_presets = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(800)
        self._autosave_timer.timeout.connect(self.save_settings)
        selected_gpu_index = self.gpu_combo.currentData() if hasattr(self, "gpu_combo") else None
        selected_gpu = next((gpu for gpu in self.detected_gpus if gpu["index"] == selected_gpu_index), None)
        if selected_gpu:
            self.statusBar().showMessage(
                f"GPU detected: {selected_gpu['name']} · {selected_gpu['memory_total_mb'] / 1024.0:.1f} GB VRAM · "
                f"diffusion/VAE will be pinned to selected CUDA GPU · model not loaded"
            )
        else:
            self.statusBar().showMessage("No NVIDIA GPU detected · CPU fallback disabled · install/update NVIDIA driver and click Refresh GPU")
        self._populate_queue_from_workspace()

    def _load_settings(self) -> dict[str, Any]:
        # Portable-first policy: only read settings files inside the application folder.
        # No silent AppData, Roaming, home-directory or registry lookup is performed.
        candidates = [SETTINGS_FILE, APP_DIR / "settings.json"]
        for settings_path in candidates:
            try:
                obj = json.loads(settings_path.read_text(encoding="utf-8"))
                version = int(obj.get("settings_version", 0))
                if version < 16 or version > SETTINGS_VERSION:
                    continue
                decoded = {}
                for k, v in obj.items():
                    if isinstance(v, str):
                        decoded[k] = decode_saved_path(v)
                    else:
                        decoded[k] = v
                # v38: Ultimate is opt-in; reset old persisted defaults that made it run
                # automatically. Preview method is migrated separately below.
                if version < 38:
                    decoded["ultimate_qwen_upscale_enabled"] = False
                    decoded["enhance_detail_mode"] = "fast"
                # Existing releases always persisted encoder_cpu and had no auto-profile
                # switch. Preserve that explicit/legacy value rather than overriding it.
                if "encoder_cpu_auto" not in obj and "encoder_cpu" in obj:
                    decoded["encoder_cpu_auto"] = False
                decoded = validate_and_migrate_settings(obj, decode_saved_path)
                if not obj.get("encoder_cpu_user_explicit", False):
                    decoded["encoder_cpu"] = False
                # Earlier releases retried diffusion enhance up to three times by default.
                # Migrate once to one pass; users can still increase attempts manually.
                if version < 21:
                    decoded["max_attempts"] = 1
                # Legacy settings migration; current Turbo defaults use eight steps. Other step presets remain selectable.
                if version < 24:
                    decoded["quality"] = "Turbo · 8 steps"
                    decoded["num_inference_steps"] = 8
                # v41 makes TAE the migrated default for old Auto/projection settings.
                # Preserve explicit TAE/VAE/Off choices and keep projection available.
                if version < 41 and str(decoded.get("live_preview_method", "tae")).strip().lower() == "proj":
                    decoded["live_preview_method"] = "tae"
                # v39: standard Turbo uses eight steps; the first Ultimate default was eight tile steps.
                if version < 39:
                    for quality_key in ("quality", "std_quality", "gen_quality", "edit_quality"):
                        if decoded.get(quality_key) == "Turbo · 9 steps":
                            decoded[quality_key] = "Turbo · 8 steps"
                    if int(decoded.get("num_inference_steps", 8) or 8) == 9:
                        decoded["num_inference_steps"] = 8
                    if int(decoded.get("ultimate_qwen_steps", 8) or 8) == 9:
                        decoded["ultimate_qwen_steps"] = 8
                    if int(decoded.get("ultimate_qwen_tile_coverage", 50) or 50) == 60:
                        decoded["ultimate_qwen_tile_coverage"] = 50
                # v43 restores Ultimate's displayed/default step count to eight and adds
                # its own 1×–4× source scale. Turbo itself is locked to the official
                # 8-step sampling schedule inside the engine regardless of this value.
                if version < 43:
                    decoded["ultimate_qwen_steps"] = 8
                    decoded.setdefault("ultimate_qwen_scale", 2)
                if version < 44:
                    if "ultimate_qwen_scale" not in obj or int(obj.get("ultimate_qwen_scale", 1) or 1) == 1:
                        decoded["ultimate_qwen_scale"] = 2
                decoded["ultimate_qwen_scale"] = max(1, min(4, int(decoded.get("ultimate_qwen_scale", 2) or 2)))
                # v44 persists the user's selectable source-image preview mode. The
                # global Ultimate tile-strength floor stays unchanged for other workflows;
                # Enhance applies its own per-profile strengths.
                if version < 44:
                    decoded.setdefault("preview_mode", "Side by side")
                # v1.7.5: accept the first faithful Enhance result by default, avoid repeated
                # full-image diffusion passes, and prefer Auto/CPU text conditioning on 16 GB GPUs.
                if version < 28:
                    decoded["retry_diff"] = 25.0
                    decoded["warn_diff"] = 18.0
                    decoded["max_attempts"] = 1
                    decoded["gpu_memory_mode"] = "auto"
                if version < 33:
                    decoded["gpu_memory_mode"] = "auto"
                    decoded["encoder_cpu"] = False
                    decoded["encoder_cpu_auto"] = False
                    decoded["enhance_detail_mode"] = "fast"
                # Migrate pre-v1.7.3 sizing labels to the unified Qwen Output Size control.
                if "std_output_scale" not in obj:
                    legacy_size = str(decoded.get("std_size_preset", "Follow input"))
                    legacy_dims = {
                        "1024 × 1024": (1024, 1024), "2048 × 2048": (2048, 2048),
                        "1344 × 768 (16:9)": (1344, 768), "768 × 1344 (9:16)": (768, 1344),
                        "1536 × 1024 (3:2)": (1536, 1024), "1024 × 1536 (2:3)": (1024, 1536),
                    }
                    if legacy_size == "Custom dimensions":
                        decoded["std_size_preset"] = "Custom dimensions"
                    elif legacy_size in ("Follow input", "Keep Original Size (Auto)"):
                        decoded["std_size_preset"] = "Scale"
                        decoded["std_output_scale"] = 1
                    elif legacy_size in legacy_dims:
                        decoded["std_size_preset"] = "Custom dimensions"
                        decoded["std_output_width"], decoded["std_output_height"] = legacy_dims[legacy_size]
                    else:
                        decoded["std_size_preset"] = "Scale"
                        decoded["std_output_scale"] = 2
                if decoded.get("std_size_preset") not in ("Scale", "Custom dimensions"):
                    decoded["std_size_preset"] = "Scale"
                decoded["std_output_scale"] = max(1, min(4, int(decoded.get("std_output_scale", 2))))
                if decoded.get("quality") not in QUALITY_PRESETS:
                    decoded["quality"] = "Turbo · 8 steps"
                for quality_key in ("std_quality", "gen_quality", "edit_quality"):
                    if quality_key not in obj or decoded.get(quality_key) not in QUALITY_PRESETS:
                        decoded[quality_key] = decoded.get("quality", "Turbo · 8 steps")
                if "num_inference_steps" not in obj:
                    decoded["num_inference_steps"] = 8
                if "upscale_tile" not in obj:
                    decoded["upscale_tile"] = 0
                if "ultimate_qwen_upscale_enabled" not in obj:
                    decoded["ultimate_qwen_upscale_enabled"] = str(decoded.get("enhance_detail_mode", "balanced")) != "fast"
                if str(decoded.get("gpu_memory_mode", "auto")).strip().lower() == "fast":
                    decoded["gpu_memory_mode"] = "full_gpu"
                if decoded.get("edit_size") == "Keep Original Size (Auto)":
                    decoded["edit_size"] = "Original Size"
                decoded["grounding_profile"] = normalize_grounding_profile(decoded.get("grounding_profile", "balanced"))
                decoded["settings_version"] = SETTINGS_VERSION
                return {**DEFAULTS, **decoded}
            except (OSError, ValueError, TypeError):
                continue
        return dict(DEFAULTS)

    def save_settings(self):
        obj = dict(DEFAULTS)
        obj.update({
            "settings_version": SETTINGS_VERSION,
            "workflow_mode": self.workflow,
            "theme": self.theme_combo.currentText() if hasattr(self, "theme_combo") else "dark",
            "encoder_cpu": self.encoder_cpu_check.isChecked() if hasattr(self, "encoder_cpu_check") else True,
            "encoder_cpu_auto": bool(getattr(self, "_encoder_cpu_auto", False)),
            "encoder_cpu_user_explicit": bool(self.settings.get("encoder_cpu_user_explicit", False)),
            "gpu_index": int(self.gpu_combo.currentData()) if hasattr(self, "gpu_combo") and self.gpu_combo.currentData() is not None else int(self.settings.get("gpu_index", 0)),
            "gpu_memory_mode": str(self.gpu_memory_mode_combo.currentData() or "auto") if hasattr(self, "gpu_memory_mode_combo") else str(self.settings.get("gpu_memory_mode", "auto")),
            "identity_lock_enabled": self._identity_lock_enabled(),
            "model_path": encode_saved_path(self.model_path_edit.text().strip()) if hasattr(self, "model_path_edit") else encode_saved_path(str(MODEL_DIR)),
            "lora_path": encode_saved_path(self.lora_path_edit.text().strip()) if hasattr(self, "lora_path_edit") else "",
            "lora_enabled": self.lora_enabled_check.isChecked() if hasattr(self, "lora_enabled_check") else False,
            "watermark_removal_enabled": bool(self.btn_watermark_toggle.isChecked()) if hasattr(self, "btn_watermark_toggle") else bool(self.settings.get("watermark_removal_enabled", False)),
            "lora_strength": self.lora_strength_spin.value() if hasattr(self, "lora_strength_spin") else 1.0,
            "num_inference_steps": self.steps_spin.value() if hasattr(self, "steps_spin") else 8,
            "std_size_preset": self.std_size_preset.currentText() if hasattr(self, "std_size_preset") else str(self.settings.get("std_size_preset", "Follow input")),
            "std_output_scale": int(getattr(self, "std_output_scale", self.settings.get("std_output_scale", 2))),
            "std_output_width": self.std_w.value() if hasattr(self, "std_w") else int(self.settings.get("std_output_width", 1024)),
            "std_output_height": self.std_h.value() if hasattr(self, "std_h") else int(self.settings.get("std_output_height", 1024)),
            "std_prompt": self.std_prompt.toPlainText() if hasattr(self, "std_prompt") else str(self.settings.get("std_prompt", "")),
            "std_prompt_preset_name": self.std_prompt_preset.currentText() if hasattr(self, "std_prompt_preset") else str(self.settings.get("std_prompt_preset_name", "")),
            "gen_prompt": self.gen_prompt_edit.toPlainText() if hasattr(self, "gen_prompt_edit") else str(self.settings.get("gen_prompt", "")),
            "gen_prompt_preset_name": self.gen_prompt_preset.currentText() if hasattr(self, "gen_prompt_preset") else str(self.settings.get("gen_prompt_preset_name", "")),
            "edit_prompt": self.edit_prompt_edit.toPlainText() if hasattr(self, "edit_prompt_edit") else str(self.settings.get("edit_prompt", "")),
            "edit_prompt_preset_name": self.edit_preset_combo.currentText() if hasattr(self, "edit_preset_combo") else str(self.settings.get("edit_prompt_preset_name", "")),
            "gen_size": self.gen_size.currentText() if hasattr(self, "gen_size") else str(self.settings.get("gen_size", "1024 × 1024 (1:1 Square)")),
            "gen_w": self.gen_w.value() if hasattr(self, "gen_w") else int(self.settings.get("gen_w", 1024)),
            "gen_h": self.gen_h.value() if hasattr(self, "gen_h") else int(self.settings.get("gen_h", 1024)),
            "gen_count": self.gen_count.value() if hasattr(self, "gen_count") else int(self.settings.get("gen_count", 1)),
            "gen_seed": self.gen_seed.value() if hasattr(self, "gen_seed") else int(self.settings.get("gen_seed", -1)),
            "gen_seed_mode": self.gen_seed_mode.currentText() if hasattr(self, "gen_seed_mode") else str(self.settings.get("gen_seed_mode", "Random")),
            "edit_seed": self.edit_seed.value() if hasattr(self, "edit_seed") else int(self.settings.get("edit_seed", -1)),
            "edit_seed_mode": self.edit_seed_mode.currentText() if hasattr(self, "edit_seed_mode") else str(self.settings.get("edit_seed_mode", "Random")),
            "edit_size": self.edit_size.currentText() if hasattr(self, "edit_size") else str(self.settings.get("edit_size", "Keep Original Size (Auto)")),
            "edit_w": self.edit_w.value() if hasattr(self, "edit_w") else int(self.settings.get("edit_w", 1024)),
            "edit_h": self.edit_h.value() if hasattr(self, "edit_h") else int(self.settings.get("edit_h", 1024)),
            "edit_fmt": self.edit_fmt_combo.currentText() if hasattr(self, "edit_fmt_combo") else str(self.settings.get("edit_fmt", "png")),
            "std_quality": self.std_quality.currentText() if hasattr(self, "std_quality") else str(self.settings.get("std_quality", "Turbo · 8 steps")),
            "gen_quality": self.gen_quality.currentText() if hasattr(self, "gen_quality") else str(self.settings.get("gen_quality", "Turbo · 8 steps")),
            "edit_quality": self.edit_quality.currentText() if hasattr(self, "edit_quality") else str(self.settings.get("edit_quality", "Turbo · 8 steps")),
            "quality": (self.gen_quality.currentText() if hasattr(self, "gen_quality") and self.workflow == "image_generate" else self.edit_quality.currentText() if hasattr(self, "edit_quality") and self.workflow == "image_edit" else self.std_quality.currentText() if hasattr(self, "std_quality") else "Turbo · 8 steps"),
            "last_input": encode_saved_path(self.current_input or ""),
            "output_dir": encode_saved_path(self.output_dir_edit.text() if hasattr(self, "output_dir_edit") else ""),
            "input_dir": encode_saved_path(self.input_dir_edit.text().strip()) if hasattr(self, "input_dir_edit") else encode_saved_path(str(INPUT_DIR)),
            "closeup_auto_zoom": self.btn_cl_auto_zoom.isChecked() if hasattr(self, "btn_cl_auto_zoom") else True,
            "grounding_profile": normalize_grounding_profile(self.grounding_profile_combo.currentData()) if hasattr(self, "grounding_profile_combo") else normalize_grounding_profile(self.settings.get("grounding_profile", "balanced")),
            "closeup_zoom": self.closeup_zoom.value() if hasattr(self, "closeup_zoom") else 2.2,
            "closeup_side": self.closeup_side,
            "closeup_enhance": self.btn_cl_enhance.isChecked() if hasattr(self, "btn_cl_enhance") else True,
            "save_lora_caption": self.btn_caption.isChecked() if hasattr(self, "btn_caption") else True,
            "lora_trigger_word": self.trigger_edit.text() if hasattr(self, "trigger_edit") else "",
            "square_side": self.square_side,
            "square_enhance": self.btn_square_enhance.isChecked() if hasattr(self, "btn_square_enhance") else False,
            "retry_diff": self.retry_diff.value() if hasattr(self, "retry_diff") else 25.0,
            "warn_diff": self.warn_diff.value() if hasattr(self, "warn_diff") else 18.0,
            "max_attempts": self.max_attempts.value() if hasattr(self, "max_attempts") else 1,
            "enhance_detail_mode": self.enhance_detail_mode.currentData() if hasattr(self, "enhance_detail_mode") else str(self.settings.get("enhance_detail_mode", "fast")),
            "ultimate_qwen_upscale_enabled": self.ultimate_qwen_upscale_check.isChecked() if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False)),
            "ultimate_qwen_denoise_strength": self.ultimate_qwen_denoise_spin.value() if hasattr(self, "ultimate_qwen_denoise_spin") else float(self.settings.get("ultimate_qwen_denoise_strength", 0.20)),
            "ultimate_qwen_tile_coverage": self.ultimate_qwen_tile_coverage_spin.value() if hasattr(self, "ultimate_qwen_tile_coverage_spin") else int(self.settings.get("ultimate_qwen_tile_coverage", 50)),
            "ultimate_qwen_passes": self.ultimate_qwen_passes_spin.value() if hasattr(self, "ultimate_qwen_passes_spin") else int(self.settings.get("ultimate_qwen_passes", 1)),
            "ultimate_qwen_steps": self.ultimate_qwen_steps_spin.value() if hasattr(self, "ultimate_qwen_steps_spin") else int(self.settings.get("ultimate_qwen_steps", 8)),
            "ultimate_qwen_scale": int(self.ultimate_qwen_scale_combo.currentData()) if hasattr(self, "ultimate_qwen_scale_combo") and self.ultimate_qwen_scale_combo.currentData() is not None else int(self.settings.get("ultimate_qwen_scale", 2)),
            "watermark_removal_enabled": bool(self.btn_watermark_toggle.isChecked()) if hasattr(self, "btn_watermark_toggle") else bool(self.settings.get("watermark_removal_enabled", False)),
            "auto_realesrgan_enabled": self.btn_auto_realesrgan.isChecked() if hasattr(self, "btn_auto_realesrgan") else bool(self.settings.get("auto_realesrgan_enabled", False)),
            "qwen_preview_enabled": bool(self.settings.get("qwen_preview_enabled", True)),
            "preview_mode": str(getattr(self, "_preferred_preview_mode", self.settings.get("preview_mode", "Side by side"))),
            "qwen_detail_profile": self._selected_qwen_detail_profile() if hasattr(self, "qwen_detail_profile_buttons") else str(self.settings.get("qwen_detail_profile", "balanced")),
            # Real-ESRGAN auto chaining is controlled only by the explicit global toolbar toggle.
            "upscale_enabled": False,
            "upscale_model": self.upscale_model_combo.currentData() if hasattr(self, "upscale_model_combo") else "realesrgan-x4plus",
            "upscale_tile": int(self.upscale_tile_combo.currentData()) if hasattr(self, "upscale_tile_combo") and self.upscale_tile_combo.currentData() is not None else 0,
            "upscale_tta": self.upscale_tta_check.isChecked() if hasattr(self, "upscale_tta_check") else False,
        })
        try:
            SETTINGS_FILE.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self.log_message(f"Could not save settings: {exc}")

    def _selected_qwen_detail_profile(self) -> str:
        """Return the active Qwen profile button, falling back to persisted settings."""
        buttons = getattr(self, "qwen_detail_profile_buttons", {})
        for key, button in buttons.items():
            if button.isChecked():
                return key
        saved = str(getattr(self, "settings", {}).get("qwen_detail_profile", "balanced"))
        return saved if saved in QWEN_DETAIL_PROFILES else "balanced"

    def _refresh_qwen_profile_prompt_note(self, profile: str):
        # Prompt guidance is applied internally; long prompt text is intentionally hidden from UI.
        return None

    def _qwen_detail_profile_selected(self, profile: str):
        """Record the selected profile, refresh the visible prompt preview and persist it."""
        profile = profile if profile in QWEN_DETAIL_PROFILES else "balanced"
        if getattr(self, "workflow", "") == "enhance":
            strength, tile_strength = qwen_enhance_reconstruction_strengths(profile)
        else:
            strength, tile_strength = qwen_detail_profile_values(profile)
        self._refresh_qwen_profile_prompt_note(profile)
        self.log_message(
            f"Qwen detail profile selected: {profile} · main strength={strength:.2f} · detail-pass strength={tile_strength:.2f}. "
            "Profile prompt guidance and Img2Img parameters are applied to newly queued runs."
        )
        self.schedule_save()

    def schedule_save(self):
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.start()

    def _build(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        logo_path = resource_path("assets/pai_logo.png")
        if logo_path.is_file():
            self.setWindowIcon(QIcon(str(logo_path)))
            logo = QLabel()
            logo_pixmap = QPixmap(str(logo_path))
            logo.setPixmap(logo_pixmap.scaled(54, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            logo.setFixedSize(58, 58)
            top.addWidget(logo)
        brand = QVBoxLayout()
        self.brand_title_label = QLabel("P.AI AI IMAGE STUDIO")
        self.brand_title_label.setStyleSheet("font-size: 15pt; font-weight: 800; color: #7958f5;")
        self.brand_subtitle_label = QLabel("Version 1 Beta · Local AI imaging")
        self.brand_subtitle_label.setStyleSheet("font-size: 8pt; color: #7583a0;")
        brand.addWidget(self.brand_title_label); brand.addWidget(self.brand_subtitle_label)
        brand_holder = QWidget(); brand_holder.setLayout(brand)
        top.addWidget(brand_holder)
        workflow_row = QHBoxLayout()
        workflow_row.setSpacing(6)
        self.workflow_buttons = []
        self.btn_workflow_generate = QPushButton("🎨  Image Generate")
        self.btn_workflow_edit = QPushButton("✏️  Image Edit")
        self.btn_workflow_enhance = QPushButton("✨  Enhance / Upscale")
        self.btn_workflow_lora = QPushButton("◎  Close-up (Smart Zoom)")
        self.btn_workflow_square = QPushButton("▣  Square 1:1 · 1K / 2K / 4K")
        workflow_defs = [
            (self.btn_workflow_generate, "image_generate"),
            (self.btn_workflow_edit, "image_edit"),
            (self.btn_workflow_enhance, "enhance"),
            (self.btn_workflow_lora, "lora_closeup_1024"),
            (self.btn_workflow_square, "square_1_1"),
        ]
        for button, mode in workflow_defs:
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, m=mode: self.apply_workflow_preset(m))
            button.setMinimumWidth(0)
            button.setMinimumHeight(31)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setMinimumHeight(34)
            workflow_row.addWidget(button, 1)
            self.workflow_buttons.append((button, mode))
        self.btn_download_model = QPushButton("🧩 Model Catalog")
        self.btn_download_model.setToolTip("Model files, AI Upscaler, Ultimate refinement and LoRA settings")
        self.btn_download_model.clicked.connect(self.download_model_bundle)
        top.addWidget(self.btn_download_model)
        self.btn_show_left_dock = QToolButton()
        self.btn_show_left_dock.setText("☰ Queue")
        self.btn_show_left_dock.setToolTip("Show/hide Images Queue and Activity Log")
        self.btn_show_left_dock.clicked.connect(self.toggle_left_dock)
        top.addWidget(self.btn_show_left_dock)
        self.btn_settings = QPushButton("⚙ Settings")
        self.btn_settings.setMinimumHeight(38)
        self.btn_settings.clicked.connect(self.open_settings_dialog)
        top.addWidget(self.btn_settings)
        top.addStretch(1)
        # Compact live hardware monitor, placed immediately to the left of Light/Dark.
        self.hardware_monitor = QFrame()
        self.hardware_monitor.setObjectName("hardwareMonitor")
        monitor_layout = QHBoxLayout(self.hardware_monitor)
        monitor_layout.setContentsMargins(4, 2, 4, 2)
        monitor_layout.setSpacing(5)
        self.hardware_metric_labels = {}
        for metric, initial in (("cpu", "CPU --"), ("ram", "RAM --"), ("gpu", "GPU --"), ("vram", "VRAM --"), ("temp", "TEMP --")):
            label = QLabel(initial)
            label.setObjectName("hardwareMetricPill")
            label.setToolTip({"cpu": "CPU utilization", "ram": "System RAM use", "gpu": "NVIDIA GPU utilization", "vram": "Dedicated GPU memory use", "temp": "GPU temperature"}[metric])
            label.setMinimumWidth(48 if metric != "vram" else 74)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hardware_metric_labels[metric] = label
            monitor_layout.addWidget(label)
        top.addWidget(self.hardware_monitor)
        self.hardware_metrics_timer = QTimer(self)
        self.hardware_metrics_timer.setInterval(2500)
        self.hardware_metrics_timer.timeout.connect(self.refresh_system_metrics)
        self.hardware_metrics_timer.start()
        QTimer.singleShot(250, self.refresh_system_metrics)
        # One-click global appearance toggle. The symbol reflects the active theme.
        self.btn_theme_toggle = QToolButton()
        self.btn_theme_toggle.setObjectName("themeToggleButton")
        self.btn_theme_toggle.setText("☾  Dark mode")
        self.btn_theme_toggle.setToolTip("Click once to switch between Light and Dark mode")
        self.btn_theme_toggle.setMinimumHeight(38)
        self.btn_theme_toggle.setMinimumWidth(112)
        self.btn_theme_toggle.clicked.connect(self.toggle_theme)
        top.addWidget(self.btn_theme_toggle)
        outer.addLayout(top)
        outer.addLayout(workflow_row)
        # Compact switches are rendered inside Basic for Enhance/Close-up/Square and
        # inside the matching Generate/Edit studio panels. They share one live state.
        self._feature_toggle_buttons = {"ultimate": [], "esrgan": [], "lora": [], "watermark": []}
        self.btn_ultimate_toggle = self._make_feature_toggle_button("ultimate")
        self.btn_auto_realesrgan = self._make_feature_toggle_button("esrgan")
        self.btn_lora_toggle = self._make_feature_toggle_button("lora")
        self.btn_watermark_toggle = self._make_feature_toggle_button("watermark")
        workflow_feature_row = QHBoxLayout()
        workflow_feature_row.setContentsMargins(0, 0, 0, 0)
        workflow_feature_row.setSpacing(5)
        for feature_button in (self.btn_ultimate_toggle, self.btn_auto_realesrgan,
                               self.btn_lora_toggle, self.btn_watermark_toggle):
            workflow_feature_row.addWidget(feature_button, 1)
        self.workflow_feature_row = workflow_feature_row

        # Model loading, model folder, GPU selection, and LoRA live in a Settings dialog.
        self.model_settings_widget = QWidget()
        model_row = QGridLayout(self.model_settings_widget)
        model_row.setContentsMargins(14, 14, 14, 14)
        model_row.setHorizontalSpacing(12)
        model_row.setVerticalSpacing(10)
        self.model_settings_widget.setMinimumWidth(820)
        model_row.addWidget(QLabel("Local model components folder"), 0, 0)
        self.model_path_edit = QLineEdit(str(self.settings.get("model_path", MODEL_DIR)))
        self.model_path_edit.setPlaceholderText("Select a folder containing the chosen image model, encoder, VAE and optional projector")
        self.model_path_edit.textChanged.connect(self.schedule_save)
        model_row.addWidget(self.model_path_edit, 0, 1)
        self.btn_browse_model = QPushButton("Browse…")
        self.btn_browse_model.clicked.connect(self.browse_model_folder)
        model_row.addWidget(self.btn_browse_model, 0, 2)
        self.btn_load_model = QPushButton("Load Selected Models")
        self.btn_load_model.setToolTip("Load the selected local model components into the chosen CUDA GPU/runtime")
        self.btn_load_model.clicked.connect(self.load_model_async)
        model_row.addWidget(self.btn_load_model, 0, 3)
        self.btn_unload_model = QPushButton("Unload")
        self.btn_unload_model.setToolTip("Unload the local model and release GPU memory")
        self.btn_unload_model.clicked.connect(self.unload_local_model)
        model_row.addWidget(self.btn_unload_model, 0, 4)
        self.model_settings_grid = model_row
        self.lora_settings_label = QLabel("Local LoRA")
        model_row.addWidget(self.lora_settings_label, 1, 0)
        self.lora_path_edit = QLineEdit(str(self.settings.get("lora_path", "")))
        self.lora_path_edit.setPlaceholderText("Optional local LoRA weights")
        model_row.addWidget(self.lora_path_edit, 1, 1)
        self.btn_browse_lora = QPushButton("Choose LoRA…")
        self.btn_browse_lora.clicked.connect(self.browse_lora_file)
        model_row.addWidget(self.btn_browse_lora, 1, 2)
        self.lora_enabled_check = QCheckBox("Enable LoRA")
        self.lora_enabled_check.setChecked(bool(self.settings.get("lora_enabled", False)))
        model_row.addWidget(self.lora_enabled_check, 1, 3)
        self.lora_strength_spin = QDoubleSpinBox()
        self.lora_strength_spin.setRange(0.0, 2.0)
        self.lora_strength_spin.setSingleStep(0.05)
        self.lora_strength_spin.setValue(float(self.settings.get("lora_strength", 1.0)))
        self.lora_strength_spin.setToolTip("LoRA adapter strength")
        model_row.addWidget(self.lora_strength_spin, 1, 5)
        self.lora_path_edit.textChanged.connect(self.schedule_save)
        self.lora_enabled_check.toggled.connect(self._global_lora_toggled)
        self.lora_strength_spin.valueChanged.connect(self.schedule_save)
        self.encoder_cpu_check = QCheckBox("Run text encoder only on CPU (saves VRAM; diffusion stays on GPU)")
        self._encoder_cpu_auto = bool(self.settings.get("encoder_cpu_auto", False))
        saved_encoder_cpu = self.settings.get("encoder_cpu")
        if saved_encoder_cpu is None or self._encoder_cpu_auto:
            initial_gpu = next((gpu for gpu in self.detected_gpus if gpu["index"] == int(self.settings.get("gpu_index", 0))), None)
            if initial_gpu is None and self.detected_gpus:
                initial_gpu = self.detected_gpus[0]
            # Run the text encoder on the NVIDIA GPU by default; --offload-to-cpu streams stages when VRAM is shared.
            default_encoder_cpu = False
        else:
            default_encoder_cpu = bool(saved_encoder_cpu)
        self.encoder_cpu_check.setChecked(default_encoder_cpu)
        self.encoder_cpu_check.setToolTip(
            "Only text encoding runs on CPU with this option. Diffusion and VAE are explicitly assigned to the selected NVIDIA CUDA GPU.\n"
            "Keeps the text encoder weights on system RAM to reserve VRAM for Qwen diffusion/VAE. The engine keeps the encoder available and caches up to 8 prompt-conditioning results. Uncheck only when the selected GPU has enough free VRAM."
        )
        self.encoder_cpu_check.toggled.connect(self._encoder_cpu_toggled)
        model_row.addWidget(self.encoder_cpu_check, 2, 0, 1, 6)

        # Visible hardware selection: display the detected GPU model and VRAM before
        # loading weights, and use the selected CUDA index in the engine backend map.
        model_row.addWidget(QLabel("Inference GPU (CUDA)"), 3, 0)
        self.gpu_combo = QComboBox()
        self.gpu_combo.setMinimumWidth(360)
        self.gpu_combo.setToolTip("The diffusion model and VAE will run on this NVIDIA GPU. CPU inference fallback is disabled.")
        model_row.addWidget(self.gpu_combo, 3, 1, 1, 2)
        model_row.addWidget(QLabel("VRAM policy"), 4, 0)
        self.gpu_memory_mode_combo = QComboBox()
        self.gpu_memory_mode_combo.addItem("Auto · balance dedicated VRAM", "auto")
        self.gpu_memory_mode_combo.addItem("GPU-only · no CPU inference", "full_gpu")
        self.gpu_memory_mode_combo.addItem("Low VRAM · offload weights to CPU", "low_vram")
        saved_memory_mode = str(self.settings.get("gpu_memory_mode", "auto")).strip().lower()
        if saved_memory_mode == "fast":
            saved_memory_mode = "full_gpu"
        memory_idx = self.gpu_memory_mode_combo.findData(saved_memory_mode)
        self.gpu_memory_mode_combo.setCurrentIndex(memory_idx if memory_idx >= 0 else 0)
        self.gpu_memory_mode_combo.setToolTip(
            "GPU-only assigns the text encoder, vision projector, diffusion and VAE to CUDA, disables auto-fit and CPU weight offload, and fails rather than silently running model inference on CPU if VRAM is insufficient. CPU still runs the UI, file I/O and process orchestration. Auto may use CPU for text encoding; Low VRAM explicitly offloads weights."
        )
        model_row.addWidget(self.gpu_memory_mode_combo, 4, 1, 1, 3)
        self.gpu_memory_mode_combo.currentIndexChanged.connect(self._on_gpu_memory_mode_changed)
        self._on_gpu_memory_mode_changed()
        self.btn_refresh_gpu = QPushButton("Refresh GPU")
        self.btn_refresh_gpu.clicked.connect(self.refresh_gpu_choices)
        model_row.addWidget(self.btn_refresh_gpu, 3, 3)
        self.gpu_status_label = QLabel("")
        self.gpu_status_label.setWordWrap(True)
        model_row.addWidget(self.gpu_status_label, 5, 0, 1, 6)

        # Keep Load / Unload directly accessible above Workflow selection; detailed
        # model path, GPU policy and LoRA settings remain in the Settings dialog.
        model_row.removeWidget(self.btn_load_model)
        model_row.removeWidget(self.btn_unload_model)
        self.btn_load_model.setParent(None)
        self.btn_unload_model.setParent(None)
        self.model_action_row = QHBoxLayout()
        self.model_action_row.setSpacing(8)
        self.model_action_row.addWidget(QLabel("Local model"))
        self.model_action_row.addWidget(self.btn_load_model)
        self.model_action_row.addWidget(self.btn_unload_model)
        self.model_action_row.addStretch(1)

        # Qwen output size and standalone Real-ESRGAN controls are built in Basic.
        # There is deliberately no shared-by-all automatic post-processing bar.
        self.refresh_gpu_choices(persist=False)
        self.gpu_combo.currentIndexChanged.connect(self._gpu_selection_changed)
        self.model_settings_widget.setVisible(True)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sidebar = self._build_sidebar()
        # Give the settings sidebar enough room so labels and controls are not clipped.
        self.sidebar.setMinimumWidth(330)
        self.sidebar.setMaximumWidth(500)
        self.sidebar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.sidebar)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        preview_bar = QHBoxLayout()
        preview_bar.setContentsMargins(0, 0, 0, 0)
        preview_bar.setSpacing(7)
        preview_bar.addWidget(QLabel("Preview"))
        self.preview_info_label = QLabel("No image selected")
        self.preview_info_label.setMinimumWidth(0)
        self.preview_info_label.setWordWrap(False)
        self.preview_info_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.preview_info_label.setStyleSheet("color: #8fbad5; font-size: 9pt;")
        preview_bar.addWidget(self.preview_info_label, 1)
        self.btn_add = QPushButton("＋ Add Images")
        self.btn_add.setMinimumWidth(0)
        self.btn_add.clicked.connect(self.add_images)
        preview_bar.addWidget(self.btn_add)
        self.btn_open = QPushButton("Open Output")
        self.btn_open.setMinimumWidth(0)
        self.btn_open.clicked.connect(self.open_output)
        preview_bar.addWidget(self.btn_open)
        center_layout.addLayout(preview_bar)

        # Keep display mode, live-preview method, and zoom in a dedicated second row.
        # Splitting the toolbar prevents clipping in windowed mode and at 125–150% DPI.
        preview_tools = QHBoxLayout()
        preview_tools.setContentsMargins(0, 0, 0, 0)
        preview_tools.setSpacing(7)
        self.preview_mode_label = QLabel("View:")
        preview_tools.addWidget(self.preview_mode_label)
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(["Enhanced", "Original", "Slider before/after", "Side by side", "Overlay", "Difference", "Flicker"])
        self._preferred_preview_mode = str(self.settings.get("preview_mode", "Side by side"))
        if self.preview_mode.findText(self._preferred_preview_mode) < 0:
            self._preferred_preview_mode = "Side by side"
        self.preview_mode.setCurrentText(self._preferred_preview_mode)
        self.preview_mode.setMinimumWidth(135)
        self.preview_mode.setMaximumWidth(190)
        preview_tools.addWidget(self.preview_mode)
        preview_tools.addWidget(QLabel("Live preview:"))
        self.live_preview_method_combo = QComboBox()
        self.live_preview_method_combo.addItem("Projection · fastest, blockier", "proj")
        self.live_preview_method_combo.addItem("TAE · clearer decode", "tae")
        self.live_preview_method_combo.addItem("VAE · full decode (slow)", "vae")
        self.live_preview_method_combo.addItem("Off", "none")
        saved_live_method = str(self.settings.get("live_preview_method", "tae") or "tae")
        live_method_index = self.live_preview_method_combo.findData(saved_live_method)
        self.live_preview_method_combo.setCurrentIndex(live_method_index if live_method_index >= 0 else 0)
        self.live_preview_method_combo.setMinimumWidth(150)
        self.live_preview_method_combo.setMaximumWidth(230)
        self.live_preview_method_combo.setToolTip(
            "Updates the Live pane as sampling steps complete. TAE is the default clearer decode; projection is fastest but blockier, and full VAE is slowest. Applies to the next Qwen inference."
        )
        self.live_preview_method_combo.currentIndexChanged.connect(self._on_live_preview_method_changed)
        preview_tools.addWidget(self.live_preview_method_combo)
        preview_tools.addStretch(1)
        self.btn_zoom_out = QToolButton(); self.btn_zoom_out.setText("−")
        self.btn_zoom_out.setToolTip("Zoom out")
        self.btn_zoom_in = QToolButton(); self.btn_zoom_in.setText("+")
        self.btn_zoom_in.setToolTip("Zoom in")
        self.btn_fit_preview = QToolButton(); self.btn_fit_preview.setText("Fit")
        self.btn_fit_preview.setToolTip("Reset zoom and pan")
        preview_tools.addWidget(self.btn_zoom_out); preview_tools.addWidget(self.btn_zoom_in); preview_tools.addWidget(self.btn_fit_preview)
        center_layout.addLayout(preview_tools)
        self.preview = PreviewCanvas()
        self.btn_zoom_out.clicked.connect(self.preview.zoom_out)
        self.btn_zoom_in.clicked.connect(self.preview.zoom_in)
        self.btn_fit_preview.clicked.connect(self.preview.fit_view)
        self.preview_mode.currentTextChanged.connect(self._on_preview_mode_changed)
        self.preview.focusSelected.connect(self.on_focus_selected)
        center_layout.addWidget(self.preview, 1)
        self.progress_label = QLabel("Stage: Idle · ready for a local image task")
        self.progress_label.setStyleSheet("color: #c6d6e8; font-size: 9pt; font-weight: 650;")
        center_layout.addWidget(self.progress_label)
        self.progress_detail_label = QLabel("Pipeline: model loading → text encoder → Qwen diffusion → VAE → Ultimate refinement → Real-ESRGAN → save")
        self.progress_detail_label.setWordWrap(True)
        self.progress_detail_label.setStyleSheet("color: #8fa4bb; font-size: 8pt;")
        center_layout.addWidget(self.progress_detail_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setFormat("%p% · %v/100")
        self.progress.setMinimumHeight(17)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet("QProgressBar { text-align: center; border: 1px solid #4a5361; border-radius: 5px; min-height: 17px; } QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2070ca, stop:1 #36d7c5); border-radius: 4px; }")
        self.progress.hide()
        center_layout.addWidget(self.progress)
        self.main_splitter.addWidget(center)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([370, 1050])

        self.queue_panel = self._build_queue_panel()
        self.log_panel = self._build_log_panel()
        # Compact studio identity card replaces the confusing Activate Lock panel.
        # Identity-preservation prompt behavior remains available in workflow logic/settings.
        self.workspace_brand_group = QGroupBox("P.AI AI IMAGE STUDIO")
        brand_layout = QVBoxLayout(self.workspace_brand_group)
        brand_layout.setContentsMargins(8, 8, 8, 8)
        brand_layout.setSpacing(6)
        brand_mark = QLabel("P.AI")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setStyleSheet("font-size: 25pt; font-weight: 900; color: #8ecdf3; padding: 8px;")
        brand_layout.addWidget(brand_mark)
        brand_caption = QLabel("IMAGE WORKSPACE")
        brand_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_caption.setWordWrap(True)
        brand_caption.setStyleSheet("font-size: 9pt; font-weight: 700; letter-spacing: 1px; color: #aeb8c6;")
        brand_layout.addWidget(brand_caption)
        self.workspace_brand_group.setMinimumWidth(150)
        self.workspace_brand_group.setMaximumWidth(250)

        self.workspace_content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_content_splitter.setHandleWidth(6)
        self.workspace_content_splitter.addWidget(self.queue_panel)
        self.workspace_content_splitter.addWidget(self.workspace_brand_group)
        self.workspace_content_splitter.setStretchFactor(0, 5)
        self.workspace_content_splitter.setStretchFactor(1, 1)
        self.workspace_content_splitter.setCollapsible(0, False)
        self.workspace_content_splitter.setCollapsible(1, False)
        self.workspace_content_splitter.setSizes([1050, 190])

        # In Enhance / Close-up / Square workflows the settings sidebar spans the
        # full workspace height; Queue and Activity Log sit beside each other below
        # the preview instead of below the settings sidebar.
        self.workflow_bottom_row = QSplitter(Qt.Orientation.Horizontal)
        self.workflow_bottom_row.setObjectName("workflowQueueLogRow")
        self.workflow_bottom_row.setHandleWidth(6)
        self.workflow_bottom_row.setChildrenCollapsible(False)
        self.workflow_bottom_row.setStretchFactor(0, 3)
        self.workflow_bottom_row.setStretchFactor(1, 2)
        self.workflow_bottom_row.setSizes([650, 500])

        self.left_panel_shell = QWidget()
        left_panel_layout = QVBoxLayout(self.left_panel_shell)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_header = QHBoxLayout()
        panel_title = QLabel("WORKSPACE")
        panel_title.setStyleSheet("font-weight: 800; color: #8ecdf3; letter-spacing: 1px;")
        panel_header.addWidget(panel_title, 1)
        self.btn_toggle_left_dock = QToolButton(); self.btn_toggle_left_dock.setText("×")
        self.btn_toggle_left_dock.setToolTip("Hide queue and log panel")
        self.btn_toggle_left_dock.setFixedSize(26, 26)
        self.btn_toggle_left_dock.clicked.connect(self.toggle_left_dock)
        panel_header.addWidget(self.btn_toggle_left_dock)
        left_panel_layout.addLayout(panel_header)
        self.left_dock_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_dock_splitter.setHandleWidth(6)
        self.left_dock_splitter.addWidget(self.workspace_content_splitter)
        self.left_dock_splitter.addWidget(self.log_panel)
        self.left_dock_splitter.setStretchFactor(0, 5)
        self.left_dock_splitter.setStretchFactor(1, 1)
        self.left_dock_splitter.setCollapsible(0, False)
        self.left_dock_splitter.setCollapsible(1, False)
        # Image Generate / Image Edit: give Images / Queue most of the dock
        # height and keep Activity Log compact, matching the requested layout.
        self.left_dock_splitter.setSizes([480, 220])
        left_panel_layout.addWidget(self.left_dock_splitter, 1)
        self.left_panel_shell.setMinimumWidth(330)
        self.left_panel_shell.setMaximumWidth(500)
        self.left_panel_shell.setMinimumHeight(0)
        self.btn_show_left_dock.setVisible(False)

        self.gen_bottom_panel = self._build_generate_studio()
        self.edit_bottom_panel = self._build_edit_studio()
        # The bottom studios contain several rows of controls; size them for the full
        # form instead of squeezing buttons/combos into a 150–160 px strip.
        self.gen_bottom_panel.setMinimumHeight(310); self.gen_bottom_panel.setMaximumHeight(16777215)
        self.edit_bottom_panel.setMinimumHeight(330); self.edit_bottom_panel.setMaximumHeight(16777215)
        self.gen_bottom_panel.setSizePolicy(self.gen_bottom_panel.sizePolicy().horizontalPolicy(), self.gen_bottom_panel.sizePolicy().verticalPolicy())
        self.edit_bottom_panel.setSizePolicy(self.edit_bottom_panel.sizePolicy().horizontalPolicy(), self.edit_bottom_panel.sizePolicy().verticalPolicy())

        self.right_workspace = QWidget()
        right_layout = QVBoxLayout(self.right_workspace)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_vertical_splitter.setHandleWidth(6)
        self.right_vertical_splitter.addWidget(self.main_splitter)
        self.right_vertical_splitter.addWidget(self.gen_bottom_panel)
        self.right_vertical_splitter.addWidget(self.edit_bottom_panel)
        self.right_vertical_splitter.setStretchFactor(0, 4)
        self.right_vertical_splitter.setStretchFactor(1, 1)
        self.right_vertical_splitter.setStretchFactor(2, 1)
        self.right_vertical_splitter.setCollapsible(0, False)
        self.right_vertical_splitter.setSizes([520, 370, 0])
        right_layout.addWidget(self.right_vertical_splitter, 1)
        for combo in (self.gen_quality, self.edit_quality, self.std_quality):
            combo.currentTextChanged.connect(self._set_steps_from_preset)
        self._set_steps_from_preset(self.settings.get("quality", "Turbo · 8 steps"), save=False)
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.addWidget(self.left_panel_shell)
        self.workspace_splitter.addWidget(self.right_workspace)
        self.workspace_splitter.setHandleWidth(6)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setStretchFactor(0, 0); self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([410, 1150])
        outer.insertLayout(1, self.model_action_row)
        outer.addWidget(self.workspace_splitter, 1)

        # Persistent settings dialog: the model widgets remain alive between opens.
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("P.AI AI IMAGE STUDIO · Settings · Model Folder & GPU")
        self.settings_dialog.setWindowIcon(self.windowIcon())
        self.settings_dialog.resize(960, 620)
        self.settings_dialog.setMinimumSize(900, 540)
        settings_layout = QVBoxLayout(self.settings_dialog)
        settings_intro = QLabel("Choose the local model folder and GPU/VRAM policy. Model components, LoRA, Ultimate refinement and Real-ESRGAN options live in Model Catalog.")
        settings_intro.setWordWrap(True); settings_layout.addWidget(settings_intro)
        diagnostics_group = QGroupBox("Local Diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics_group)
        self.settings_diagnostics_button = QPushButton("🩺 Run local Diagnostics…")
        self.settings_diagnostics_button.setToolTip(
            "Read-only checks for the local grounding runtime, selected model files, workspace and available telemetry. "
            "No downloads or inference will be started."
        )
        self.settings_diagnostics_button.clicked.connect(self.open_diagnostics)
        diagnostics_layout.addWidget(self.settings_diagnostics_button)
        settings_layout.addWidget(diagnostics_group)
        settings_layout.addWidget(self.model_settings_widget)
        settings_close = QPushButton("Close Settings")
        settings_close.clicked.connect(self.settings_dialog.accept)
        settings_layout.addWidget(settings_close, 0, Qt.AlignmentFlag.AlignRight)
        self.setCentralWidget(root)
        saved_theme = str(self.settings.get("theme", "dark")).lower()
        if self.theme_combo.findText(saved_theme) < 0:
            saved_theme = "dark"
        self.theme_combo.setCurrentText(saved_theme)
        self._apply_theme(saved_theme)
        if self._pending_log_messages:
            pending, self._pending_log_messages = self._pending_log_messages, []
            for message in pending:
                self._append_log_message(message)
        self.refresh_qwen_grounding_status()
        self.statusBar().showMessage("Qwen-Image 2.1 local image engine · Qwen3-VL local semantic grounding for Close-up · no hosted API")

    def _build_sidebar(self):
        tabs = QTabWidget()
        basic = QWidget()
        bl = QVBoxLayout(basic)
        self.basic_tab_layout = bl
        bl.addLayout(self.workflow_feature_row)
        # Close-up and Square use fixed square output sizes instead of the shared
        # Enhance Scale/Custom controls. Persist each workflow's size independently.
        self.closeup_side = normalize_fixed_square_side(self.settings.get("closeup_side", 1024))
        self.square_side = normalize_fixed_square_side(self.settings.get("square_side", 1024))
        self.qwen_strength_group = QGroupBox("Qwen Image 2.1 · Detail / Upscale strength")
        qsl = QVBoxLayout(self.qwen_strength_group)
        profile_row = QHBoxLayout()
        profile_row.setSpacing(5)
        self.qwen_detail_profile_buttons = {}
        self.qwen_detail_profile_group = QButtonGroup(self)
        self.qwen_detail_profile_group.setExclusive(True)
        saved_profile = str(self.settings.get("qwen_detail_profile", "balanced"))
        if saved_profile not in QWEN_DETAIL_PROFILES:
            saved_profile = "balanced"
        for profile_id in ("balanced", "strong", "maximum"):
            meta = QWEN_DETAIL_PROFILES[profile_id]
            button = QPushButton(meta["short_label"])
            button.setCheckable(True)
            button.setMinimumHeight(34)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setProperty("qwenDetailProfile", profile_id)
            button.setToolTip(
                f"{meta['description']}\n"
                "Strengths are workflow-specific; Enhance uses Balanced 0.84/0.74, "
                "Strong 0.91/0.82 and Maximum 0.97/0.92. Other workflows use their own profile values."
            )
            self.qwen_detail_profile_group.addButton(button)
            self.qwen_detail_profile_buttons[profile_id] = button
            profile_column = QVBoxLayout()
            profile_column.setSpacing(2)
            profile_column.addWidget(button)
            short_descriptions = {
                "balanced": "Cleaner texture, lower change risk.",
                "strong": "More reconstruction, moderate change risk.",
                "maximum": "Strongest detail recovery, highest change risk.",
            }
            short_note = QLabel(short_descriptions[profile_id])
            short_note.setWordWrap(True)
            short_note.setStyleSheet("color: #aeb8c6; font-size: 11px; padding: 0 3px 2px 3px;")
            profile_column.addWidget(short_note)
            profile_row.addLayout(profile_column, 1)
            button.clicked.connect(lambda checked=False, key=profile_id: self._qwen_detail_profile_selected(key))
        self.qwen_detail_profile_buttons[saved_profile].setChecked(True)
        qsl.addLayout(profile_row)
        self.qwen_strength_group.setToolTip(
            "Select a Qwen reconstruction profile. Prompt guidance and Img2Img strengths apply to new jobs. "
            "Optional Real-ESRGAN post-processing is controlled by its workflow switch."
        )
        # Short per-button descriptions replace the old long prompt-guidance preview.
        self.qwen_profile_prompt_note = None  # Keep prompt guidance internal; do not display long prompt text.
        bl.addWidget(self.qwen_strength_group)

        self.enhance_detail_group = QGroupBox("Ultimate · Qwen tile refinement")
        edl = QVBoxLayout(self.enhance_detail_group)
        self.ultimate_qwen_upscale_check = QCheckBox("Enable tiled detail refinement")
        self.ultimate_qwen_upscale_check.setChecked(bool(self.settings.get("ultimate_qwen_upscale_enabled", False)))
        self.ultimate_qwen_upscale_check.setToolTip("Ultimate runs an additional detail pass over an overlapping 2×2 tile grid, using the compatible eight-step Turbo schedule where applicable.")
        edl.addWidget(self.ultimate_qwen_upscale_check)
        ultimate_form = QFormLayout()
        self.ultimate_qwen_scale_combo = QComboBox()
        for scale_label, scale_value in (("1× · Keep source dimensions", 1), ("2×", 2), ("3×", 3), ("4×", 4)):
            self.ultimate_qwen_scale_combo.addItem(scale_label, scale_value)
        saved_ultimate_scale = max(1, min(4, int(self.settings.get("ultimate_qwen_scale", 2) or 2)))
        scale_index = self.ultimate_qwen_scale_combo.findData(saved_ultimate_scale)
        self.ultimate_qwen_scale_combo.setCurrentIndex(scale_index if scale_index >= 0 else 0)
        self.ultimate_qwen_scale_combo.setToolTip(
            "Applied AFTER the normal Enhance output scale. Example: Enhance 2× then Ultimate 2× produces 4× dimensions overall. "
            "Ultimate 1× keeps the Enhance-stage dimensions but still runs Qwen detail tiles. Large outputs may need more tiles."
        )
        ultimate_form.addRow("Ultimate output scale", self.ultimate_qwen_scale_combo)
        self.ultimate_qwen_denoise_spin = QDoubleSpinBox()
        self.ultimate_qwen_denoise_spin.setRange(0.05, 0.95)
        self.ultimate_qwen_denoise_spin.setDecimals(2)
        self.ultimate_qwen_denoise_spin.setSingleStep(0.05)
        self.ultimate_qwen_denoise_spin.setValue(float(self.settings.get("ultimate_qwen_denoise_strength", 0.20)))
        self.ultimate_qwen_denoise_spin.setToolTip("Directly controls the Qwen img2img denoise strength for every Ultimate tile (0.05–0.95) across all workflows.")
        ultimate_form.addRow("Denoise strength", self.ultimate_qwen_denoise_spin)
        self.ultimate_qwen_tile_coverage_spin = QSpinBox()
        self.ultimate_qwen_tile_coverage_spin.setRange(45, 75)
        self.ultimate_qwen_tile_coverage_spin.setSuffix(" %")
        self.ultimate_qwen_tile_coverage_spin.setValue(int(self.settings.get("ultimate_qwen_tile_coverage", 50)))
        self.ultimate_qwen_tile_coverage_spin.setToolTip("Target coverage for each tile, plus an automatic overlap margin for blending. Default 50% keeps tile changes away from a hard center seam.")
        ultimate_form.addRow("Tile coverage + overlap", self.ultimate_qwen_tile_coverage_spin)
        self.ultimate_qwen_passes_spin = QSpinBox()
        self.ultimate_qwen_passes_spin.setRange(1, 2)
        self.ultimate_qwen_passes_spin.setValue(int(self.settings.get("ultimate_qwen_passes", 1)))
        self.ultimate_qwen_passes_spin.setToolTip("Number of complete overlapping-tile refinement passes. Two passes take longer and may alter more details.")
        ultimate_form.addRow("Refinement passes", self.ultimate_qwen_passes_spin)
        self.ultimate_qwen_steps_spin = QSpinBox()
        self.ultimate_qwen_steps_spin.setRange(4, 60)
        self.ultimate_qwen_steps_spin.setSingleStep(1)
        self.ultimate_qwen_steps_spin.setValue(int(self.settings.get("ultimate_qwen_steps", 8)))
        self.ultimate_qwen_steps_spin.setToolTip("Directly controls the number of executed denoising steps per tile in Ultimate mode for both Qwen-Image 2.1 Turbo and non-Turbo checkpoints.")
        ultimate_form.addRow("Steps per tile", self.ultimate_qwen_steps_spin)
        edl.addLayout(ultimate_form)
        detail_mode_note = QLabel("Ultimate is an optional, quality-focused detail stage. It processes overlapping image tiles sequentially and blends the overlaps to reduce visible boundaries.")
        detail_mode_note.setWordWrap(True)
        detail_mode_note.setStyleSheet("color: #94a6bb; font-size: 9pt;")
        edl.addWidget(detail_mode_note)
        self.ultimate_qwen_denoise_spin.valueChanged.connect(self.schedule_save)
        self.ultimate_qwen_scale_combo.currentIndexChanged.connect(self._on_ultimate_scale_changed)
        self.ultimate_qwen_scale_combo.currentIndexChanged.connect(self.schedule_save)
        self.ultimate_qwen_tile_coverage_spin.valueChanged.connect(self.schedule_save)
        self.ultimate_qwen_passes_spin.valueChanged.connect(self.schedule_save)
        self.ultimate_qwen_steps_spin.valueChanged.connect(self.schedule_save)
        # Compatibility data field used by queued jobs and existing settings migrations.
        self.enhance_detail_mode = QComboBox()
        self.enhance_detail_mode.addItem("Ultimate · overlapping Qwen tiles", "ultimate")
        self.enhance_detail_mode.addItem("Disabled", "fast")
        self.enhance_detail_mode.setCurrentIndex(0 if self.ultimate_qwen_upscale_check.isChecked() else 1)
        self.enhance_detail_mode.setVisible(False)
        self.ultimate_qwen_upscale_check.toggled.connect(lambda enabled: self.enhance_detail_mode.setCurrentIndex(0 if enabled else 1))
        self.ultimate_qwen_upscale_check.toggled.connect(self.schedule_save)
        self.ultimate_qwen_upscale_check.toggled.connect(lambda enabled: self._sync_feature_toggle_buttons("ultimate", enabled, source=self.ultimate_qwen_upscale_check))
        self.enhance_detail_mode.currentIndexChanged.connect(self.schedule_save)
        bl.addWidget(self.enhance_detail_group)

        self.lora_group = QGroupBox("Close-up · Model-Catalog Qwen3-VL Target Zoom & Qwen Image Enhance")
        lg = QVBoxLayout(self.lora_group)
        self.lora_target = QLineEdit()
        self.lora_target.setPlaceholderText("Describe any target naturally: e.g. the silver ring on the left hand")
        lg.addWidget(QLabel("Target description · interpreted by the selected Model Catalog vision-language encoder"))
        lg.addWidget(self.lora_target)
        lg.addWidget(QLabel("Auto Zoom profile · speed versus multi-scale search"))
        self.grounding_profile_combo = QComboBox()
        for label, key in (("Fast · lowest latency", "fast"), ("Balanced · recommended", "balanced"), ("Accuracy · deeper search", "accuracy")):
            self.grounding_profile_combo.addItem(label, key)
        selected_profile = normalize_grounding_profile(self.settings.get("grounding_profile", "balanced"))
        self.grounding_profile_combo.setCurrentIndex(max(0, self.grounding_profile_combo.findData(selected_profile)))
        self.grounding_profile_combo.setToolTip(grounding_profile_description(selected_profile))
        self.grounding_profile_combo.currentIndexChanged.connect(self._grounding_profile_selected)
        lg.addWidget(self.grounding_profile_combo)
        self.grounding_runtime_group = QGroupBox("Close-up Locator Runtime · Qwen3-VL Grounding (llama.cpp)")
        grl = QVBoxLayout(self.grounding_runtime_group)
        grl.setContentsMargins(10, 10, 10, 10)
        grl.setSpacing(6)
        self.btn_qwen_grounding_setup = QPushButton("⚙ Prepare Locator Runtime · ~45 MB")
        self.btn_qwen_grounding_setup.setToolTip(
            "Downloads only the small llama.cpp runtime (~45 MB). Close-up reuses the selected or available GGUF text encoder and matching vision projector on the GPU; no separate VLM weights are downloaded."
        )
        self.btn_qwen_grounding_setup.clicked.connect(self.download_qwen_grounding_bundle)
        grl.addWidget(self.btn_qwen_grounding_setup)
        self.qwen_grounding_progress = QProgressBar()
        self.qwen_grounding_progress.setRange(0, 100)
        self.qwen_grounding_progress.setValue(0)
        self.qwen_grounding_progress.setTextVisible(True)
        self.qwen_grounding_progress.setMaximumHeight(16)
        grl.addWidget(self.qwen_grounding_progress)
        self.qwen_grounding_status = QLabel("Close-up will reuse the selected Model Catalog text encoder; only the small local chat runtime may need installation.")
        self.qwen_grounding_status.setWordWrap(True)
        grl.addWidget(self.qwen_grounding_status)
        self.btn_cl_auto_zoom = QPushButton("◎  Auto Zoom: ON (Model-Catalog Qwen3-VL grounding)")
        self.btn_cl_auto_zoom.setCheckable(True)
        self.btn_cl_auto_zoom.setChecked(bool(self.settings.get("closeup_auto_zoom", True)))
        self.btn_cl_auto_zoom.clicked.connect(self.update_closeup_controls)
        lg.addWidget(self.btn_cl_auto_zoom)
        self.closeup_zoom = QDoubleSpinBox()
        self.closeup_zoom.setRange(1.1, 6.0)
        self.closeup_zoom.setSingleStep(0.1)
        self.closeup_zoom.setValue(float(self.settings.get("closeup_zoom", 2.2)))
        self.closeup_zoom_label = QLabel("Manual zoom factor")
        lg.addWidget(self.closeup_zoom_label)
        lg.addWidget(self.closeup_zoom)
        focusrow = QHBoxLayout()
        self.focus_label = QLabel("Focus point: automatic")
        self.btn_focus_reset = QPushButton("Auto")
        self.btn_focus_reset.clicked.connect(self.reset_focus)
        focusrow.addWidget(self.focus_label, 1)
        focusrow.addWidget(self.btn_focus_reset)
        lg.addLayout(focusrow)
        lg.addWidget(QLabel("Output size · square canvas"))
        closeup_size_row = QHBoxLayout()
        closeup_size_row.setSpacing(5)
        self.closeup_size_buttons = {}
        for side in FIXED_SQUARE_OUTPUT_SIZES:
            button = QPushButton(str(side))
            button.setCheckable(True)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            note = " Qwen performs a native pass plus detail-refinement tiles above 2048 px." if side > MAX_INFERENCE_SIDE else ""
            button.setToolTip(f"Set Close-up output to {side} × {side} pixels.{note}")
            button.clicked.connect(lambda checked=False, value=side: self.set_closeup_side(value))
            self.closeup_size_buttons[side] = button
            closeup_size_row.addWidget(button, 1)
        lg.addLayout(closeup_size_row)
        self.set_closeup_side(self.closeup_side, save=False)
        self.btn_cl_enhance = QPushButton("✨ Enhance: ON (Target-Aware Faithful Enhance)")
        self.btn_cl_enhance.setCheckable(True)
        self.btn_cl_enhance.setChecked(bool(self.settings.get("closeup_enhance", True)))
        self.btn_cl_enhance.clicked.connect(self.update_closeup_controls)
        lg.addWidget(self.btn_cl_enhance)
        self.btn_caption = QPushButton("📝 Auto .txt caption: ON")
        self.btn_caption.setCheckable(True)
        self.btn_caption.setChecked(bool(self.settings.get("save_lora_caption", True)))
        self.btn_caption.clicked.connect(self.update_closeup_controls)
        lg.addWidget(self.btn_caption)
        self.trigger_edit = QLineEdit(str(self.settings.get("lora_trigger_word", "")))
        self.trigger_edit.setPlaceholderText("Optional LoRA trigger word")
        lg.addWidget(self.trigger_edit)
        bl.addWidget(self.lora_group)

        self.square_mode_group = QGroupBox("Square 1:1 · Size & Enhance")
        sg = QVBoxLayout(self.square_mode_group)
        sg.addWidget(QLabel("Output size · square canvas"))
        square_size_row = QHBoxLayout()
        square_size_row.setSpacing(5)
        self.square_size_buttons = {}
        for side in FIXED_SQUARE_OUTPUT_SIZES:
            button = QPushButton(str(side))
            button.setCheckable(True)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            note = " Qwen performs a native pass plus detail-refinement tiles above 2048 px." if side > MAX_INFERENCE_SIDE else ""
            button.setToolTip(f"Set Square 1:1 output to {side} × {side} pixels.{note}")
            button.clicked.connect(lambda checked=False, value=side: self.set_square_side(value))
            self.square_size_buttons[side] = button
            square_size_row.addWidget(button, 1)
        sg.addLayout(square_size_row)
        self.set_square_side(self.square_side, save=False)
        self.btn_square_enhance = QPushButton("✨ Enhance: OFF")
        self.btn_square_enhance.setCheckable(True)
        self.btn_square_enhance.setChecked(bool(self.settings.get("square_enhance", False)))
        self.btn_square_enhance.clicked.connect(self.update_closeup_controls)
        sg.addWidget(self.btn_square_enhance)
        bl.addWidget(self.square_mode_group)

        self.workflow_queue_actions = QGroupBox("Queue controls")
        qa = QGridLayout(self.workflow_queue_actions)
        qa.setContentsMargins(8, 6, 8, 7)
        qa.setHorizontalSpacing(6)
        qa.setVerticalSpacing(5)
        self.btn_run = QPushButton("▶ Run selected")
        self.btn_run.setToolTip("Run selected queued image(s) using the active workflow.")
        self.btn_run.clicked.connect(self.run_selected)
        self.btn_run_all = QPushButton("▶ Run all")
        self.btn_run_all.setToolTip("Run every queued image using the active workflow.")
        self.btn_run_all.clicked.connect(self.run_all_queued)
        self.btn_stop_current_basic = QPushButton("■ Stop after job")
        self.btn_stop_current_basic.setToolTip("Let the active image finish, then do not start the next queued image.")
        self.btn_stop_current_basic.clicked.connect(self.stop_after_current_job)
        self.btn_stop_now = QPushButton("■ Cancel now")
        self.btn_stop_now.setToolTip("Cancel the active inference at the next safe checkpoint.")
        self.btn_stop_now.clicked.connect(self.stop_job)
        for button in (self.btn_run, self.btn_run_all, self.btn_stop_current_basic, self.btn_stop_now):
            button.setMinimumWidth(0)
            button.setMinimumHeight(28)
            button.setMaximumHeight(31)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        qa.addWidget(self.btn_run, 0, 0)
        qa.addWidget(self.btn_run_all, 0, 1)
        qa.addWidget(self.btn_stop_current_basic, 1, 0)
        qa.addWidget(self.btn_stop_now, 1, 1)
        qa.setColumnStretch(0, 1)
        qa.setColumnStretch(1, 1)
        bl.addWidget(self.workflow_queue_actions)

        files_box = QGroupBox("Input / Output files")
        self.files_box = files_box
        files_layout = QGridLayout(files_box)
        files_layout.setContentsMargins(8, 6, 8, 7)
        files_layout.setHorizontalSpacing(6)
        files_layout.setVerticalSpacing(5)
        files_layout.addWidget(QLabel("Input"), 0, 0)
        self.input_dir_edit = QLineEdit(str(self.settings.get("input_dir", INPUT_DIR)))
        self.input_dir_edit.setPlaceholderText("Input image folder")
        self.input_dir_edit.setMinimumWidth(0)
        self.input_dir_edit.setMaximumHeight(30)
        self.input_dir_edit.setToolTip("Full path to the folder used when adding input images.")
        self.input_dir_edit.textChanged.connect(self.schedule_save)
        files_layout.addWidget(self.input_dir_edit, 0, 1)
        self.btn_browse_input = QPushButton("Browse…")
        self.btn_browse_input.setMinimumWidth(0)
        self.btn_browse_input.setFixedHeight(29)
        self.btn_browse_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_browse_input.setToolTip("Choose the input image folder.")
        self.btn_browse_input.clicked.connect(self.choose_input_dir)
        files_layout.addWidget(self.btn_browse_input, 0, 2)
        self.btn_add_input = QPushButton("＋ Add images…")
        self.btn_add_input.setMinimumWidth(0)
        self.btn_add_input.setFixedHeight(29)
        self.btn_add_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_add_input.setToolTip("Add image files or a folder to Images / Queue.")
        self.btn_add_input.clicked.connect(self.add_images)
        files_layout.addWidget(self.btn_add_input, 1, 0, 1, 3)
        files_layout.addWidget(QLabel("Output"), 2, 0)
        self.output_dir_edit = QLineEdit(str(self.settings.get("output_dir", "")))
        self.output_dir_edit.setPlaceholderText("Default workflow output folder")
        self.output_dir_edit.setMinimumWidth(0)
        self.output_dir_edit.setMaximumHeight(30)
        self.output_dir_edit.setToolTip("Full path where processed images will be saved.")
        self.output_dir_edit.textChanged.connect(self.schedule_save)
        files_layout.addWidget(self.output_dir_edit, 2, 1)
        self.btn_browse_output = QPushButton("Browse…")
        self.btn_browse_output.setMinimumWidth(0)
        self.btn_browse_output.setFixedHeight(29)
        self.btn_browse_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_browse_output.clicked.connect(self.choose_output_dir)
        files_layout.addWidget(self.btn_browse_output, 2, 2)
        files_layout.setColumnStretch(1, 1)
        bl.addWidget(files_box)

        # Qwen Output Size is one shared, explicit control for all Qwen image workflows.
        # Standalone Real-ESRGAN is controlled separately below and never changes these dimensions.
        self.qwen_output_group = QGroupBox("Output Size · Qwen Image 2.1")
        qwen_size_layout = QVBoxLayout(self.qwen_output_group)
        qwen_size_layout.setContentsMargins(8, 8, 8, 8)
        qwen_size_layout.setSpacing(7)

        self.std_size_preset = QComboBox()
        self.std_size_preset.addItems(["Scale", "Custom dimensions"])
        saved_size_mode = str(self.settings.get("std_size_preset", "Scale"))
        if saved_size_mode not in ("Scale", "Custom dimensions"):
            saved_size_mode = "Scale"
        self.std_size_preset.setCurrentText(saved_size_mode)
        self.std_size_preset.setMinimumWidth(0)
        self.std_size_preset.setMinimumContentsLength(14)
        self.std_size_preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.std_size_preset.setToolTip(
            "Enhance workflow only: Scale sizes output relative to the input; Custom dimensions sets a target output canvas. "
            "Close-up and Square have dedicated 1024 / 2048 / 4096 selectors."
        )
        self.std_size_preset.currentTextChanged.connect(self.apply_std_size_preset)
        self.std_output_size_widget = QWidget()
        std_size_vlayout = QVBoxLayout(self.std_output_size_widget)
        std_size_vlayout.setContentsMargins(0, 0, 0, 0)
        std_size_vlayout.setSpacing(6)
        std_size_vlayout.addWidget(self.std_size_preset)

        self.std_output_scale = max(1, min(4, int(self.settings.get("std_output_scale", 2))))
        self.std_scale_row = QWidget()
        std_scale_layout = QGridLayout(self.std_scale_row)
        std_scale_layout.setContentsMargins(0, 0, 0, 0)
        std_scale_layout.setHorizontalSpacing(5)
        std_scale_layout.setVerticalSpacing(5)
        self.std_scale_buttons = {}
        for scale in (1, 2, 3, 4):
            button = QPushButton(f"{scale}×")
            button.setCheckable(True)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolTip(
                f"Qwen output scale {scale}×. Enhance uses the source dimensions as the base; Close-up/Square use a 1024 × 1024 base."
            )
            button.clicked.connect(lambda checked=False, value=scale: self.set_std_output_scale(value))
            self.std_scale_buttons[scale] = button
            scale_index = scale - 1
            std_scale_layout.addWidget(button, scale_index // 2, scale_index % 2)
        std_size_vlayout.addWidget(self.std_scale_row)

        self.std_custom_dims_row = QWidget()
        std_custom_row = QHBoxLayout(self.std_custom_dims_row)
        std_custom_row.setContentsMargins(0, 0, 0, 0)
        std_custom_row.setSpacing(6)
        self.std_w = QSpinBox(); self.std_w.setRange(64, MAX_OUTPUT_SIDE); self.std_w.setSingleStep(32)
        self.std_w.setKeyboardTracking(False)
        self.std_w.setValue(max(64, min(MAX_OUTPUT_SIDE, int(self.settings.get("std_output_width", 1024)))))
        self.std_w.setSuffix(" px"); self.std_w.setMinimumWidth(96)
        self.std_w.setToolTip("Custom Qwen output width in pixels (up to 8192); values are normalized to a multiple of 32 on commit.")
        self.std_h = QSpinBox(); self.std_h.setRange(64, MAX_OUTPUT_SIDE); self.std_h.setSingleStep(32)
        self.std_h.setKeyboardTracking(False)
        self.std_h.setValue(max(64, min(MAX_OUTPUT_SIDE, int(self.settings.get("std_output_height", 1024)))))
        self.std_h.setSuffix(" px"); self.std_h.setMinimumWidth(96)
        self.std_h.setToolTip("Custom Qwen output height in pixels (up to 8192); values are normalized to a multiple of 32 on commit.")
        std_custom_row.addWidget(QLabel("W")); std_custom_row.addWidget(self.std_w, 1)
        std_custom_row.addWidget(QLabel("H")); std_custom_row.addWidget(self.std_h, 1)
        std_size_vlayout.addWidget(self.std_custom_dims_row)
        self.std_w.valueChanged.connect(self.schedule_save)
        self.std_h.valueChanged.connect(self.schedule_save)
        self.std_w.editingFinished.connect(lambda: self._snap_qwen_dimension(self.std_w))
        self.std_h.editingFinished.connect(lambda: self._snap_qwen_dimension(self.std_h))
        qwen_size_layout.addWidget(self.std_output_size_widget)
        qwen_size_note = QLabel(
            "Enhance / Upscale uses Qwen Image 2.1. The output scale here runs first; if Ultimate is ON, its independent scale multiplies that result again. "
            "Large outputs use overlapping Qwen tiles; Real-ESRGAN remains a separate optional post-process."
        )
        qwen_size_note.setWordWrap(True)
        qwen_size_layout.addWidget(qwen_size_note)
        self.apply_std_size_preset(self.std_size_preset.currentText())
        self.set_std_output_scale(self.std_output_scale, save=False)
        bl.addWidget(self.qwen_output_group)

        self.model_preset_group = QGroupBox("Advanced model preset · Qwen Image 2.1")
        model_preset_layout = QVBoxLayout(self.model_preset_group)
        model_preset_layout.setContentsMargins(8, 8, 8, 8)
        self.std_quality = QComboBox(); self.std_quality.addItems(QUALITY_PRESETS)
        std_q_index = self.std_quality.findText(str(self.settings.get("std_quality", self.settings.get("quality", "Turbo · 8 steps"))))
        self.std_quality.setCurrentIndex(std_q_index if std_q_index >= 0 else 0)
        self.std_quality.setMinimumWidth(0)
        self.std_quality.setToolTip("Single sampling preset for Enhance, Close-up and Square. Qwen Turbo uses the P.AI adapted 8-step schedule; Strength for faithful Enhance is handled separately by Img2Img.")
        model_preset_layout.addWidget(self.std_quality)

        # Real-ESRGAN/Real-ESRNet is a separate task; an explicit workflow toggle may chain it after Qwen.
        self.realesrgan_box = QGroupBox("AI Upscaler · Real-ESRGAN / Real-ESRNet")
        real_layout = QVBoxLayout(self.realesrgan_box)
        real_layout.setContentsMargins(8, 8, 8, 8)
        real_layout.setSpacing(6)
        real_note = QLabel("Use manually or switch ESRGAN ON in the workflow to run after generation. Each model automatically upscales at its native scale.")
        real_note.setWordWrap(True)
        real_layout.addWidget(real_note)
        real_form = QFormLayout()
        real_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.upscale_model_combo = QComboBox()
        for label, model_name in REALESRGAN_MODEL_NAMES.items():
            self.upscale_model_combo.addItem(label, model_name)
        selected_model = str(self.settings.get("upscale_model", "realesrgan-x4plus"))
        selected_model_index = self.upscale_model_combo.findData(selected_model)
        self.upscale_model_combo.setCurrentIndex(selected_model_index if selected_model_index >= 0 else 0)
        self.upscale_model_combo.setMinimumWidth(0)
        self.upscale_model_combo.setToolTip("Photo: Real-ESRGAN x4plus (4×). Conservative restoration: Real-ESRNet (4×). Anime: Real-ESRGAN anime (4×) or AnimeVideo-v3 (4×/3×/2×). Runs automatically at the selected model's native scale.")
        real_form.addRow("Upscaler model", self.upscale_model_combo)
        self.upscale_tile_combo = QComboBox()
        for label, value in (("Auto tile", 0), ("Tile 128", 128), ("Tile 256", 256), ("Tile 384", 384), ("Tile 512", 512)):
            self.upscale_tile_combo.addItem(label, value)
        tile_idx = self.upscale_tile_combo.findData(int(self.settings.get("upscale_tile", 0)))
        self.upscale_tile_combo.setCurrentIndex(tile_idx if tile_idx >= 0 else 0)
        self.upscale_tile_combo.setMinimumWidth(0)
        self.upscale_tile_combo.setToolTip("Smaller tiles reduce VRAM use; Auto delegates tile sizing to the engine.")
        real_form.addRow("Tile size", self.upscale_tile_combo)
        real_layout.addLayout(real_form)
        self.upscale_tta_check = QCheckBox("TTA (slower, may improve some images)")
        self.upscale_tta_check.setChecked(bool(self.settings.get("upscale_tta", False)))
        self.upscale_tta_check.setToolTip("Test-time augmentation is slower and may help some images.")
        real_layout.addWidget(self.upscale_tta_check)
        self.btn_upscale_selected = QPushButton("Run AI Upscale")
        self.btn_upscale_selected.setMinimumHeight(32)
        self.btn_upscale_selected.setToolTip("Run the selected NCNN/Vulkan Real-ESRGAN or Real-ESRNet model at its native scale as a separate image-processing task. Qwen Enhance does not call it automatically.")
        self.btn_upscale_selected.clicked.connect(self.upscale_selected_image)
        real_layout.addWidget(self.btn_upscale_selected)
        # Runtime install status and download action are part of Model Catalog.
        self.upscale_model_combo.currentIndexChanged.connect(self.schedule_save)
        self.upscale_tile_combo.currentIndexChanged.connect(self.schedule_save)
        self.upscale_tta_check.toggled.connect(self.schedule_save)
        # Real-ESRGAN/Real-ESRNet is independent of Qwen workflows and remains available.
        bl.addStretch(1)

        prompt_tab = QWidget()
        pl = QVBoxLayout(prompt_tab)
        pl.addWidget(QLabel("Prompt / guidance"))
        self.std_prompt = QTextEdit()
        self.std_prompt.setPlainText(str(self.settings.get("std_prompt", "")))
        self.std_prompt.setPlaceholderText(
            "Optional instruction for Enhance, Close-up or Square. Use natural language; state the requested change first and explicitly preserve everything else."
        )
        self.std_prompt.textChanged.connect(self._std_prompt_text_edited)
        pl.addWidget(self.std_prompt, 1)
        self.std_prompt_preset = QComboBox()
        self.std_prompt_preset.setMinimumWidth(0)
        self.std_prompt_preset.setToolTip("Templates are filtered for the active Qwen workflow so Square-only outpainting instructions are not accidentally used in Enhance or Close-up.")
        self._active_std_prompt_preset = "Custom instruction (manual)"
        self.std_prompt_preset.currentTextChanged.connect(self.apply_std_prompt_preset)
        pl.addWidget(QLabel("Qwen Image 2.1 prompt templates · workflow-aware"))
        pl.addWidget(self.std_prompt_preset)
        prompt_routing_note = QLabel(
            "Enhance and Qwen Scale use the selected Enhance prompt, augmented with the selected Qwen detail profile. "
            "Close-up / Smart Zoom uses target-aware detail guidance; Square 1:1 uses outpainting guidance and a profile-aware core-enhance prompt. "
            "Image Generate and Image Edit have their own linked prompts. Standalone Real-ESRGAN / Real-ESRNet is separate and has no text prompt."
        )
        prompt_routing_note.setWordWrap(True)
        pl.addWidget(prompt_routing_note)
        self.refresh_std_prompt_presets(str(self.workflow if hasattr(self, "workflow") else "enhance"), initial=True)
        prompt_refs = QLabel(
            '<a href="https://github.com/QwenLM/Qwen-Image-2.1">Official Qwen Image 2.1 guide</a> · '
            '<a href="https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite">Official prompt rewriting examples</a>'
        )
        prompt_refs.setOpenExternalLinks(True); prompt_refs.setWordWrap(True)
        pl.addWidget(prompt_refs)

        advanced = QWidget()
        al = QFormLayout(advanced)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(4, 60); self.steps_spin.setValue(int(self.settings.get("num_inference_steps", 8)))
        self.steps_spin.setToolTip("Internal compatibility value. Qwen Turbo uses the P.AI adapted 8-step sigma schedule; this duplicate control is intentionally not shown.")
        self.max_attempts = QSpinBox(); self.max_attempts.setRange(1, 8); self.max_attempts.setValue(int(self.settings.get("max_attempts", 1)))
        self.retry_diff = QDoubleSpinBox(); self.retry_diff.setRange(0.0, 100.0); self.retry_diff.setDecimals(1); self.retry_diff.setValue(float(self.settings.get("retry_diff", 25.0)))
        self.warn_diff = QDoubleSpinBox(); self.warn_diff.setRange(0.0, 100.0); self.warn_diff.setDecimals(1); self.warn_diff.setValue(float(self.settings.get("warn_diff", 18.0)))
        al.addRow(self.model_preset_group)
        # The preset is the single user-facing sampling control. A separate Steps spinner
        # duplicates it and is controlled by the selected preset; Turbo uses an adapted 8-step sigma schedule.
        self.steps_spin.hide()
        al.addRow("Max attempts (full reruns)", self.max_attempts); al.addRow("Accept result when difference ≤", self.retry_diff); al.addRow("Warn when difference >", self.warn_diff)
        # Internal state holder for theme persistence. Theme switching is available
        # directly from the top-right button, not buried in Advanced settings.
        self.theme_combo = QComboBox(advanced); self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.hide()
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        self.theme_combo.currentTextChanged.connect(self.schedule_save)
        stop = QPushButton("■ Stop now")
        stop.clicked.connect(self.stop_job)
        al.addRow(stop)

        # Force the Basic tab content to follow the viewport width. Without Ignored
        # size-hint handling, long button labels make the inner page wider than the
        # scroll viewport and the right edge disappears because horizontal scrolling is off.
        for group in (self.enhance_detail_group, self.lora_group, self.square_mode_group, self.workflow_queue_actions,
                      files_box, self.qwen_output_group, self.model_preset_group, self.realesrgan_box):
            group.setMinimumWidth(0)
            group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Maximum)
        for control in (self.lora_target, self.btn_cl_auto_zoom, self.closeup_zoom,
                        self.btn_focus_reset, self.btn_cl_enhance, self.btn_caption,
                        self.trigger_edit, self.btn_square_enhance):
            control.setMinimumWidth(0)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, control.sizePolicy().verticalPolicy())

        basic.setMinimumWidth(0)
        basic.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        # Keep the tab contents and scroll viewport narrow enough for compact sidebars.
        basic_scroll = QScrollArea()
        self.basic_scroll = basic_scroll
        basic_scroll.setWidgetResizable(True)
        basic_scroll.setFrameShape(QFrame.Shape.NoFrame)
        basic_scroll.setWidget(basic)
        basic_scroll.setMinimumWidth(0)
        basic_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        tabs.addTab(basic_scroll, "Basic")
        tabs.addTab(prompt_tab, "Prompt")
        tabs.addTab(advanced, "Advanced")
        tabs.setMinimumWidth(0)
        tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        basic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        basic_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.sidebar_tabs = tabs
        self.update_closeup_controls()
        return tabs

    def _build_queue_panel(self):
        panel = QGroupBox("Images / Queue")
        layout = QVBoxLayout(panel)
        searchrow = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search queue…")
        self.search_box.textChanged.connect(self.refresh_queue_table)
        add = QPushButton("Add")
        add.clicked.connect(self.add_images)
        searchrow.addWidget(self.search_box, 1)
        searchrow.addWidget(add)
        layout.addLayout(searchrow)
        self.queue_table = QTableWidget(0, 6)
        self.queue_table.setHorizontalHeaderLabels(["Name", "Workflow", "Status", "Time", "Δ src", "Path"])
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(42)
        self.queue_table.setColumnHidden(5, True)
        self.queue_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.queue_table.setWordWrap(False)
        self.queue_table.setToolTip("Time is measured processing time. Δ src is the approximate mean pixel change compared with the source. Full paths remain available in cell tooltips.")
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.cellClicked.connect(self.select_queue_row)
        self.queue_table.cellDoubleClicked.connect(self.open_queue_row)
        self.queue_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue_table.customContextMenuRequested.connect(self.queue_context_menu)
        layout.addWidget(self.queue_table, 1)
        return panel

    def _build_log_panel(self):
        panel = QGroupBox("Activity Log")
        layout = QVBoxLayout(panel)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)
        return panel

    @staticmethod
    def _studio_field_cell(label_text: str, control: QWidget) -> QWidget:
        """Compact responsive field: label sits above its control instead of stealing horizontal width."""
        cell = QWidget()
        cell.setMinimumWidth(0)
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(3)
        label = QLabel(label_text)
        label.setStyleSheet("color: #aebfd2; font-size: 8.5pt; font-weight: 600;")
        label.setMinimumWidth(0)
        cell_layout.addWidget(label)
        control.setMinimumWidth(0)
        control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cell_layout.addWidget(control)
        return cell

    def _build_generate_studio(self):
        """Build a compact, non-scrollable Generate form whose controls fit in the splitter."""
        panel = QGroupBox("Image Generate Studio")
        panel.setMinimumWidth(0)
        panel.setMinimumHeight(310)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 14, 8, 7)
        outer.setSpacing(0)

        content = QWidget(panel)
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 3, 4, 3)
        content_layout.setSpacing(5)

        # Prompt heading: all three controls share one row without fixed grid columns.
        prompt_row = QHBoxLayout()
        prompt_row.setContentsMargins(0, 0, 0, 0)
        prompt_row.setSpacing(8)
        prompt_label = QLabel("Prompt")
        prompt_label.setMinimumWidth(48)
        prompt_row.addWidget(prompt_label)
        self.gen_prompt_preset = QComboBox()
        self.gen_prompt_preset.addItems(list(GEN_PROMPT_PRESETS.keys()))
        saved_gen_prompt = str(self.settings.get("gen_prompt", "")).strip()
        saved_gen_preset = str(self.settings.get("gen_prompt_preset_name", ""))
        valid_gen_preset = saved_gen_preset in GEN_PROMPT_PRESETS and saved_gen_prompt == str(GEN_PROMPT_PRESETS[saved_gen_preset]).strip()
        if not valid_gen_preset:
            saved_gen_preset = next((n for n, template in GEN_PROMPT_PRESETS.items() if saved_gen_prompt == str(template).strip()), "Custom scene (write your prompt)")
        self.gen_prompt_preset.setCurrentText(saved_gen_preset)
        self.gen_prompt_preset.setMinimumWidth(0)
        self.gen_prompt_preset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.gen_prompt_preset.setMinimumHeight(30)
        self.gen_prompt_preset.currentTextChanged.connect(self.apply_generate_prompt_preset)
        prompt_row.addWidget(self.gen_prompt_preset, 1)
        clear = QPushButton("Clear")
        clear.setMinimumWidth(76)
        clear.setMinimumHeight(30)
        prompt_row.addWidget(clear)
        content_layout.addLayout(prompt_row)

        self.gen_prompt_edit = QTextEdit()
        self.gen_prompt_edit.setPlainText(str(self.settings.get("gen_prompt", "")))
        self.gen_prompt_edit.setPlaceholderText(
            "Use a fluent natural-language prompt. Describe the subject/action, composition and spatial placement, lighting, visual medium, and palette; avoid keyword-tag piles. "
            "For visible text, quote the exact wording and specify its placement and typography."
        )
        self.gen_prompt_edit.setMinimumHeight(48)
        self.gen_prompt_edit.setMaximumHeight(58)
        self.gen_prompt_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.gen_prompt_edit.textChanged.connect(self._sync_gen_prompt_preset)
        clear.clicked.connect(self.gen_prompt_edit.clear)
        content_layout.addWidget(self.gen_prompt_edit)

        size_options = [
            "1024 × 1024 (1:1 Square)", "2048 × 2048 (1:1 High-Res)",
            "1344 × 768 (16:9 Landscape)", "768 × 1344 (9:16 Portrait)",
            "1536 × 1024 (3:2 Photo)", "1024 × 1536 (2:3 Portrait)",
            "Custom dimensions",
        ]
        self.gen_size = QComboBox()
        self.gen_size.addItems(size_options)
        self.gen_size.currentTextChanged.connect(self.apply_gen_size_preset)
        self.gen_size.currentTextChanged.connect(lambda *_: self.schedule_save())
        # High output-side capacity is intended for tiled Enhance/Upscale, not native txt2img.
        saved_gen_w = max(64, min(4096, int(self.settings.get("gen_w", 1024))))
        saved_gen_h = max(64, min(4096, int(self.settings.get("gen_h", 1024))))
        self.gen_w = QSpinBox(); self.gen_w.setRange(64, 4096); self.gen_w.setSingleStep(32); self.gen_w.setValue(saved_gen_w)
        self.gen_h = QSpinBox(); self.gen_h.setRange(64, 4096); self.gen_h.setSingleStep(32); self.gen_h.setValue(saved_gen_h)
        preset_dimensions = {
            "1024 × 1024 (1:1 Square)": (1024, 1024),
            "2048 × 2048 (1:1 High-Res)": (2048, 2048),
            "1344 × 768 (16:9 Landscape)": (1344, 768),
            "768 × 1344 (9:16 Portrait)": (768, 1344),
            "1536 × 1024 (3:2 Photo)": (1536, 1024),
            "1024 × 1536 (2:3 Portrait)": (1024, 1536),
        }
        saved_size_name = next((name for name, dims in preset_dimensions.items() if dims == (saved_gen_w, saved_gen_h)), "Custom dimensions")
        saved_size = str(self.settings.get("gen_size", saved_size_name))
        if saved_size not in size_options or (saved_size in preset_dimensions and preset_dimensions[saved_size] != (saved_gen_w, saved_gen_h)):
            saved_size = "Custom dimensions"
        self.gen_size.setCurrentText(saved_size)

        self.gen_quality = QComboBox(); self.gen_quality.addItems(QUALITY_PRESETS)
        gen_q_index = self.gen_quality.findText(str(self.settings.get("gen_quality", self.settings.get("quality", "Turbo · 8 steps"))))
        self.gen_quality.setCurrentIndex(gen_q_index if gen_q_index >= 0 else 0)
        self.gen_count = QSpinBox(); self.gen_count.setRange(1, 100); self.gen_count.setValue(int(self.settings.get("gen_count", 1)))
        self.gen_seed = QSpinBox(); self.gen_seed.setRange(-1, 2147483647); self.gen_seed.setValue(int(self.settings.get("gen_seed", -1)))
        self.gen_seed_mode = QComboBox(); self.gen_seed_mode.addItems(["Random", "Fixed", "Increment"]); self.gen_seed_mode.setCurrentText(str(self.settings.get("gen_seed_mode", "Random")))
        for control in (self.gen_w, self.gen_h, self.gen_count, self.gen_seed):
            control.setMinimumWidth(0)
            control.setMinimumHeight(30)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            control.valueChanged.connect(lambda *_: self.schedule_save())
        for control in (self.gen_quality, self.gen_size, self.gen_seed_mode):
            control.setMinimumWidth(0)
            control.setMinimumHeight(30)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.gen_seed_mode.currentTextChanged.connect(self.schedule_save)
        self.gen_w.editingFinished.connect(lambda: self._commit_gen_dimensions("width"))
        self.gen_h.editingFinished.connect(lambda: self._commit_gen_dimensions("height"))

        settings_grid = QGridLayout()
        settings_grid.setContentsMargins(0, 0, 0, 0)
        settings_grid.setHorizontalSpacing(10)
        settings_grid.setVerticalSpacing(5)
        for column in range(3):
            settings_grid.setColumnStretch(column, 1)
            settings_grid.setColumnMinimumWidth(column, 0)

        # Pair Width and Height inside one dimensions cell: two even rows, no scrolling.
        dimensions_cell = QWidget(content)
        dimensions_layout = QHBoxLayout(dimensions_cell)
        dimensions_layout.setContentsMargins(0, 0, 0, 0)
        dimensions_layout.setSpacing(7)
        dimensions_layout.addWidget(self._studio_field_cell("Width", self.gen_w), 1)
        dimensions_layout.addWidget(self._studio_field_cell("Height", self.gen_h), 1)
        fields = [
            ("Aspect / preset", self.gen_size, 0, 0),
            ("Model preset", self.gen_quality, 0, 2),
            ("Batch count", self.gen_count, 1, 0),
            ("Seed", self.gen_seed, 1, 1),
            ("Seed mode", self.gen_seed_mode, 1, 2),
        ]
        for label_text, control, row, column in fields:
            cell = self._studio_field_cell(label_text, control)
            cell.setMinimumHeight(44)
            settings_grid.addWidget(cell, row, column)
        dimensions_cell.setMinimumHeight(44)
        settings_grid.addWidget(dimensions_cell, 0, 1)
        content_layout.addLayout(settings_grid)

        content_layout.addLayout(self._make_feature_toggle_row())
        self.btn_generate = QPushButton("🎨 Generate Image(s)")
        self.btn_generate.setMinimumHeight(36)
        self.btn_generate.setMinimumWidth(0)
        self.btn_generate.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_generate.clicked.connect(self.generate_images)
        content_layout.addWidget(self.btn_generate)
        stop_row = QHBoxLayout()
        stop_row.setContentsMargins(0, 0, 0, 0)
        stop_row.setSpacing(8)
        self.btn_generate_stop_current = QPushButton("■ Stop current job")
        self.btn_generate_stop_current.setToolTip("Let the current generated image finish, then stop the remaining batch/queue.")
        self.btn_generate_stop_current.setMinimumWidth(0)
        self.btn_generate_stop_current.setMinimumHeight(30)
        self.btn_generate_stop_current.clicked.connect(self.stop_after_current_job)
        self.btn_generate_stop_now = QPushButton("■ Stop now")
        self.btn_generate_stop_now.setToolTip("Cancel the active inference at the next safe checkpoint.")
        self.btn_generate_stop_now.setMinimumWidth(0)
        self.btn_generate_stop_now.setMinimumHeight(30)
        self.btn_generate_stop_now.clicked.connect(self.stop_job)
        stop_row.addWidget(self.btn_generate_stop_current, 1)
        stop_row.addWidget(self.btn_generate_stop_now, 1)
        content_layout.addLayout(stop_row)
        outer.addWidget(content, 1)
        return panel

    def _build_edit_studio(self):
        panel = QGroupBox("Image Edit Studio")
        panel.setMinimumWidth(0)
        panel.setMinimumHeight(330)
        root = QVBoxLayout(panel)
        root.setContentsMargins(10, 14, 10, 9)
        root.setSpacing(6)
        # Shared workflow toggles span the whole studio instead of being squeezed into
        # the narrow settings column. This fixes clipped Ultimate/ESRGAN/LoRA labels.
        root.addLayout(self._make_feature_toggle_row())
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        left = QVBoxLayout()
        self.edit_preset_combo = QComboBox()
        self.edit_preset_combo.addItems(list(EDIT_PRESETS.keys()))
        saved_edit_prompt = str(self.settings.get("edit_prompt", "")).strip()
        saved_edit_preset = str(self.settings.get("edit_prompt_preset_name", ""))
        valid_edit_preset = saved_edit_preset in EDIT_PRESETS and saved_edit_prompt == str(EDIT_PRESETS[saved_edit_preset]).strip()
        if not valid_edit_preset:
            saved_edit_preset = next((n for n, template in EDIT_PRESETS.items() if saved_edit_prompt == str(template).strip()), "Custom instruction (free edit)")
        self.edit_preset_combo.setCurrentText(saved_edit_preset)
        self.edit_preset_combo.currentTextChanged.connect(self.apply_edit_preset)
        left.addWidget(QLabel("Template preset"))
        left.addWidget(self.edit_preset_combo)
        # Create the editor before connecting chip callbacks: Qt signals can fire
        # as soon as the user interacts, so the referenced widget must exist first.
        self.edit_prompt_edit = QTextEdit()
        self.edit_prompt_edit.setPlainText(str(self.settings.get("edit_prompt", "")))
        self.edit_prompt_edit.textChanged.connect(self._sync_edit_prompt_preset)
        self.edit_prompt_edit.setPlaceholderText(
            "Start with the requested change. Name only what should change and, if needed, what must stay the same. "
            "For text edits, quote the exact replacement text."
        )
        chips = QHBoxLayout()
        for text, addition in [
            ("+ Studio Lighting", "Adjust exposure and light balance with natural highlights and readable shadows."),
            ("+ Blur Background", "Blur only the background with a natural shallow depth of field; keep the foreground subject and edges sharp."),
            ("+ Clean Blemishes", "Remove temporary blemishes subtly while retaining realistic skin texture."),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked=False, s=addition: self.edit_prompt_edit.setPlainText((self.edit_prompt_edit.toPlainText().rstrip() + " " + s).strip()))
            chips.addWidget(btn)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.edit_prompt_edit.clear)
        chips.addWidget(clear)
        left.addLayout(chips)
        left.addWidget(self.edit_prompt_edit, 1)
        guards = QHBoxLayout()
        self.btn_edit_lock_id = QPushButton("🔒 Lock Facial Identity: ON")
        self.btn_edit_lock_id.setCheckable(True); self.btn_edit_lock_id.setChecked(True)
        self.btn_edit_lock_id.clicked.connect(lambda: self.btn_edit_lock_id.setText("🔒 Lock Facial Identity: " + ("ON" if self.btn_edit_lock_id.isChecked() else "OFF")))
        self.btn_edit_lock_comp = QPushButton("📐 Preserve Unmentioned Areas: ON")
        self.btn_edit_lock_comp.setCheckable(True); self.btn_edit_lock_comp.setChecked(True)
        self.btn_edit_lock_comp.clicked.connect(lambda: self.btn_edit_lock_comp.setText("📐 Preserve Unmentioned Areas: " + ("ON" if self.btn_edit_lock_comp.isChecked() else "OFF")))
        guards.addWidget(self.btn_edit_lock_id); guards.addWidget(self.btn_edit_lock_comp)
        left.addLayout(guards)
        content_layout.addLayout(left, 3)

        right_widget = QWidget()
        right_widget.setMinimumWidth(300)
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_content = QVBoxLayout(right_widget)
        right_content.setContentsMargins(2, 2, 2, 2)
        right_content.setSpacing(5)
        settings_grid = QGridLayout()
        settings_grid.setContentsMargins(0, 0, 0, 0)
        settings_grid.setHorizontalSpacing(7)
        settings_grid.setVerticalSpacing(4)
        settings_grid.setColumnStretch(0, 1)
        settings_grid.setColumnStretch(1, 1)
        self.edit_size = QComboBox()
        self.edit_size.addItems(["Original Size", "1024 × 1024", "2048 × 2048", "1344 × 768", "768 × 1344", "Custom dimensions"])
        self.edit_size.setToolTip("Use the input image's exact width × height for the final edited file. Qwen inference may use the nearest supported internal canvas and is resized back to these output dimensions.")
        saved_edit_size = str(self.settings.get("edit_size", "Original Size"))
        if saved_edit_size == "Keep Original Size (Auto)":
            saved_edit_size = "Original Size"
        if self.edit_size.findText(saved_edit_size) >= 0:
            self.edit_size.setCurrentText(saved_edit_size)
        self.edit_w = QSpinBox(); self.edit_w.setRange(64, 4096); self.edit_w.setSingleStep(32); self.edit_w.setValue(max(64, min(4096, int(self.settings.get("edit_w", 1024)))))
        self.edit_h = QSpinBox(); self.edit_h.setRange(64, 4096); self.edit_h.setSingleStep(32); self.edit_h.setValue(max(64, min(4096, int(self.settings.get("edit_h", 1024)))))
        self.edit_size.currentTextChanged.connect(self.apply_edit_size_preset)
        self.edit_quality = QComboBox(); self.edit_quality.addItems(QUALITY_PRESETS)
        edit_q_index = self.edit_quality.findText(str(self.settings.get("edit_quality", self.settings.get("quality", "Turbo · 8 steps"))))
        self.edit_quality.setCurrentIndex(edit_q_index if edit_q_index >= 0 else 0)
        self.edit_fmt_combo = QComboBox(); self.edit_fmt_combo.addItems(["png", "jpg"])
        saved_edit_fmt = str(self.settings.get("edit_fmt", "png")).lower()
        self.edit_fmt_combo.setCurrentText(saved_edit_fmt if saved_edit_fmt in ("png", "jpg") else "png")
        self.edit_seed = QSpinBox(); self.edit_seed.setRange(-1, 2147483647); self.edit_seed.setValue(int(self.settings.get("edit_seed", -1)))
        self.edit_seed_mode = QComboBox(); self.edit_seed_mode.addItems(["Random", "Fixed", "Increment"]); self.edit_seed_mode.setCurrentText(str(self.settings.get("edit_seed_mode", "Random")))
        for control in (self.edit_size, self.edit_quality, self.edit_fmt_combo, self.edit_seed_mode):
            control.setMinimumWidth(0)
            control.setMinimumHeight(29)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for control in (self.edit_w, self.edit_h, self.edit_seed):
            control.setMinimumWidth(0)
            control.setMinimumHeight(29)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            control.valueChanged.connect(lambda *_: self.schedule_save())
        self.edit_seed.valueChanged.connect(lambda *_: self.schedule_save())
        self.edit_seed_mode.currentTextChanged.connect(self.schedule_save)
        self.edit_fmt_combo.currentTextChanged.connect(self.schedule_save)

        edit_fields = [
            ("Output size", self.edit_size, 0, 0),
            ("Model preset", self.edit_quality, 0, 1),
            ("Width", self.edit_w, 1, 0),
            ("Height", self.edit_h, 1, 1),
            ("Format", self.edit_fmt_combo, 2, 0),
            ("Seed", self.edit_seed, 2, 1),
            ("Seed mode", self.edit_seed_mode, 3, 0),
        ]
        for label_text, control, row, column in edit_fields:
            cell = self._studio_field_cell(label_text, control)
            cell.setMinimumHeight(43)
            settings_grid.addWidget(cell, row, column)
        right_content.addLayout(settings_grid)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        self.btn_edit_selected = QPushButton("✏️ Edit Selected")
        self.btn_edit_selected.setMinimumWidth(0); self.btn_edit_selected.setMinimumHeight(31)
        self.btn_edit_selected.clicked.connect(self.edit_selected)
        self.btn_edit_all = QPushButton("▶ Edit All in Queue")
        self.btn_edit_all.setMinimumWidth(0); self.btn_edit_all.setMinimumHeight(31)
        self.btn_edit_all.clicked.connect(self.edit_all_in_queue)
        actions_row.addWidget(self.btn_edit_selected, 1)
        actions_row.addWidget(self.btn_edit_all, 1)
        right_content.addLayout(actions_row)

        edit_stop_row = QHBoxLayout()
        edit_stop_row.setSpacing(6)
        self.btn_edit_stop_current = QPushButton("■ Stop current job")
        self.btn_edit_stop_now = QPushButton("■ Stop now")
        self.btn_edit_stop_current.setMinimumWidth(0); self.btn_edit_stop_now.setMinimumWidth(0)
        self.btn_edit_stop_current.setMinimumHeight(29); self.btn_edit_stop_now.setMinimumHeight(29)
        self.btn_edit_stop_current.setToolTip("Let the current edit finish, then do not start the next queued image.")
        self.btn_edit_stop_now.setToolTip("Cancel the active inference at the next safe checkpoint.")
        self.btn_edit_stop_current.clicked.connect(self.stop_after_current_job)
        self.btn_edit_stop_now.clicked.connect(self.stop_job)
        edit_stop_row.addWidget(self.btn_edit_stop_current, 1)
        edit_stop_row.addWidget(self.btn_edit_stop_now, 1)
        right_content.addLayout(edit_stop_row)
        content_layout.addWidget(right_widget, 1)
        root.addLayout(content_layout, 1)
        self.apply_edit_size_preset(self.edit_size.currentText())
        return panel

    def _gpu_policy_summary(self, gpu: dict[str, Any]) -> str:
        """Describe the active memory policy, not just the GPU's VRAM capacity."""
        total_gb = float(gpu.get("memory_total_mb", 0)) / 1024.0
        free_gb = float(gpu.get("memory_free_mb", 0)) / 1024.0
        mode = str(
            self.gpu_memory_mode_combo.currentData()
            if hasattr(self, "gpu_memory_mode_combo") else self.settings.get("gpu_memory_mode", "auto")
        ).strip().lower()
        if mode == "full_gpu":
            policy = (
                "GPU-only: every model component stays on CUDA; CPU inference and CPU weight offload are forbidden. "
                "If the model does not fit in dedicated VRAM, loading/inference may fail with an out-of-memory error. "
                "CPU still runs the desktop UI, file I/O and process orchestration."
            )
        elif mode == "low_vram":
            policy = (
                "Low VRAM: model weights are explicitly placed in system RAM and staged for CUDA execution; "
                "this can reduce dedicated-VRAM pressure but may be slower."
            )
        else:
            policy = (
                "Auto: if the model bundle plus a 2.8 GiB workspace reserve will not fit, the app moves the text "
                "encoder/projector to CPU and keeps diffusion/VAE on CUDA. If even diffusion+VAE+reserve will not fit, "
                "it enables explicit weight offload; auto-fit is disabled whenever the placement is known."
            )
        capacity_note = ""
        if total_gb < 12.0:
            capacity_note = " This GPU has under 12 GiB dedicated VRAM; GPU-only mode may fail if the full model does not fit."
        elif free_gb < 6.5:
            capacity_note = " Less than 6.5 GiB dedicated VRAM is currently free."
        return (
            f"Selected: {gpu.get('name', 'NVIDIA GPU')} · {total_gb:.1f} GiB total / {free_gb:.1f} GiB free · "
            f"{policy}{capacity_note}"
        )

    def refresh_gpu_choices(self, checked: bool = False, persist: bool = True):
        """Refresh NVIDIA GPU list shown to the user; do not substitute CPU if empty."""
        if not hasattr(self, "gpu_combo"):
            return
        previous = self.gpu_combo.currentData()
        if previous is None:
            previous = self.settings.get("gpu_index", 0)
        self.detected_gpus = detect_nvidia_gpus()
        self.gpu_combo.blockSignals(True)
        self.gpu_combo.clear()
        if not self.detected_gpus:
            self.gpu_combo.addItem("No NVIDIA CUDA GPU detected · CPU fallback disabled", -1)
            self.gpu_combo.setEnabled(False)
            if hasattr(self, "gpu_status_label"):
                self.gpu_status_label.setText(
                    "GPU not detected. Install or update the NVIDIA display driver, then click Refresh GPU. "
                    "CUDA Toolkit and PyTorch are not required. Image generation will not run on CPU."
                )
        else:
            self.gpu_combo.setEnabled(True)
            for gpu in self.detected_gpus:
                total_gb = gpu["memory_total_mb"] / 1024.0
                free_gb = gpu["memory_free_mb"] / 1024.0
                label = f"{gpu['name']} · {total_gb:.1f} GB VRAM · {free_gb:.1f} GB free · CUDA {gpu['index']}"
                self.gpu_combo.addItem(label, int(gpu["index"]))
            chosen_index = self.gpu_combo.findData(int(previous)) if previous is not None else -1
            self.gpu_combo.setCurrentIndex(chosen_index if chosen_index >= 0 else 0)
            if hasattr(self, "gpu_status_label"):
                selected = self.detected_gpus[self.gpu_combo.currentIndex()]
                self.gpu_status_label.setText(self._gpu_policy_summary(selected))
        self.gpu_combo.blockSignals(False)
        if persist:
            self.schedule_save()

    def _identity_lock_enabled(self) -> bool:
        """Return identity-lock preference even when its optional UI button is absent."""
        button = getattr(self, "btn_workspace_lock", None)
        if button is not None:
            try:
                return bool(button.isChecked())
            except RuntimeError:
                pass
        return bool(getattr(self, "settings", {}).get("identity_lock_enabled", True))

    def _on_gpu_memory_mode_changed(self, *_):
        if not hasattr(self, "gpu_memory_mode_combo") or not hasattr(self, "encoder_cpu_check"):
            self.schedule_save()
            return
        full_gpu = str(self.gpu_memory_mode_combo.currentData() or "auto") == "full_gpu"
        if full_gpu and self.encoder_cpu_check.isChecked():
            self.encoder_cpu_check.blockSignals(True)
            self.encoder_cpu_check.setChecked(False)
            self.encoder_cpu_check.blockSignals(False)
            self._encoder_cpu_auto = False
        self.encoder_cpu_check.setEnabled(not full_gpu)
        self.encoder_cpu_check.setToolTip(
            "GPU-only mode: all model components are assigned to CUDA and CPU inference/offload is forbidden. If the model does not fit in dedicated VRAM, startup/inference may fail with an out-of-memory error; use Auto only if CPU text encoding is acceptable."
            if full_gpu else
            "Only the text encoder and its vision projector run on CPU with this option; diffusion and VAE computation stay on the selected CUDA GPU. This can improve speed on 16 GB cards by preserving dedicated VRAM for diffusion."
        )
        if hasattr(self, "gpu_status_label") and hasattr(self, "gpu_combo"):
            selected_index = self.gpu_combo.currentData()
            selected_gpu = next((item for item in self.detected_gpus if item["index"] == selected_index), None)
            if selected_gpu:
                self.gpu_status_label.setText(self._gpu_policy_summary(selected_gpu))
        self.schedule_save()

    def _gpu_selection_changed(self, *_):
        if hasattr(self, "gpu_combo"):
            index = self.gpu_combo.currentData()
            gpu = next((item for item in self.detected_gpus if item["index"] == index), None)
            if gpu and hasattr(self, "gpu_status_label"):
                self.gpu_status_label.setText(self._gpu_policy_summary(gpu))
                is_full_gpu = bool(
                    hasattr(self, "gpu_memory_mode_combo")
                    and str(self.gpu_memory_mode_combo.currentData() or "auto") == "full_gpu"
                )
                if (not is_full_gpu and getattr(self, "_encoder_cpu_auto", False)
                        and hasattr(self, "encoder_cpu_check")):
                    self.encoder_cpu_check.blockSignals(True)
                    self.encoder_cpu_check.setChecked(gpu["memory_total_mb"] < 20 * 1024)
                    self.encoder_cpu_check.blockSignals(False)
            self.schedule_save()

    def _encoder_cpu_toggled(self, _checked: bool):
        # Once the user touches the checkbox, stop automatically changing it when GPUs change.
        self._encoder_cpu_auto = False
        self.schedule_save()

    def download_model_bundle(self) -> bool:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Processing active", "Stop image processing before downloading or repairing model files.")
            return False
        if self.model_worker and self.model_worker.isRunning():
            QMessageBox.information(self, "Model loading", "Wait for model loading to finish before downloading or repairing model files.")
            return
        target = Path(self.model_path_edit.text().strip() or str(MODEL_DIR)).expanduser()
        if not target.is_absolute():
            target = (APP_DIR / target).resolve()
        dialog = getattr(self, "model_catalog_dialog", None)
        if dialog is None:
            dialog = ModelDownloadDialog(target, self)
            self.model_catalog_dialog = dialog
        dialog.target_dir = target
        if hasattr(dialog, "destination_label"):
            dialog.destination_label.setText(f"Model folder: {target}")
        self.refresh_qwen_grounding_status()
        dlg_res = dialog.exec()
        self.save_settings()
        if dlg_res == QDialog.DialogCode.Accepted and dialog.applied:
            self.refresh_qwen_grounding_status()
            self.model_path_edit.setText(str(target.resolve()))
            self.save_settings()
            self.statusBar().showMessage("Model choices saved. Load Selected Models to use the chosen components offline.")
            self.log_message(f"Model selection saved at {target.resolve()}")
            return True
        return False

    def browse_model_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select local model components folder", self.model_path_edit.text() or str(MODEL_DIR))
        if folder:
            self.model_path_edit.setText(folder)
            self.schedule_save()

    def browse_lora_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select local Qwen-Image 2.1 LoRA", self.lora_path_edit.text() or str(LORA_DIR), "LoRA weights (*.safetensors *.bin *.pt)")
        if filename:
            self.lora_path_edit.setText(filename)
            self.lora_enabled_check.setChecked(True)
            self.schedule_save()

    def _on_live_preview_method_changed(self, index: int = -1):
        method = str(self.live_preview_method_combo.currentData() or "tae") if hasattr(self, "live_preview_method_combo") else "proj"
        if method not in {"proj", "tae", "vae", "none"}:
            method = "tae"
        self.settings["live_preview_method"] = method
        if hasattr(self, "engine"):
            self.engine.live_preview_method = method
            self.engine.live_preview_interval = 1  # one preview request per completed sampler step
        if hasattr(self, "_autosave_timer"):
            self.schedule_save()

    def _configure_engine_from_ui(self):
        self.engine.configure(
            self.model_path_edit.text().strip(),
            self.lora_path_edit.text().strip(),
            self.lora_enabled_check.isChecked(),
            self.lora_strength_spin.value(),
            self.encoder_cpu_check.isChecked() and str(self.gpu_memory_mode_combo.currentData() or "auto") != "full_gpu",
            int(self.gpu_combo.currentData()) if self.gpu_combo.currentData() is not None else -1,
            str(self.gpu_memory_mode_combo.currentData() or "auto"),
        )
        self._on_live_preview_method_changed()

    def load_model_async(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Model busy", "Stop the active inference before loading or reconfiguring the model.")
            return
        if self.model_worker and self.model_worker.isRunning():
            return

        # Keep first-run setup staged: model weights are user-selected (and license-gated)
        # in Model Catalog, while the much larger native engine is downloaded only after
        # the required model components are present and verified.
        catalog = getattr(self, "model_catalog_dialog", None)
        if catalog is not None:
            target = Path(self.model_path_edit.text().strip() or str(MODEL_DIR)).expanduser()
            if not target.is_absolute():
                target = (APP_DIR / target).resolve()
            catalog.target_dir = target
            if hasattr(catalog, "destination_label"):
                catalog.destination_label.setText(f"Model folder: {target}")
            assets_ready, reason = catalog._profile_has_verified_catalog_assets()
            if not assets_ready:
                answer = QMessageBox.question(
                    self,
                    "Complete model setup first",
                    f"{reason}\n\nOpen Model Catalog to download or verify the selected components? "
                    "The native engine download will start only after the required model files are ready.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    # Continue through the setup stages only if the user completed
                    # download/verification in Model Catalog.
                    if self.download_model_bundle():
                        self.load_model_async()
                else:
                    self.statusBar().showMessage("Model setup incomplete · open Model Catalog to download/verify components.")
                return

        self._configure_engine_from_ui()
        # A previous Stop now must not poison the next model load attempt.
        self.engine.force_stop.clear()
        self.engine.stop_after_current.clear()
        self.btn_load_model.setEnabled(False)
        self._progress_started_at = time.monotonic()
        self.progress.setRange(0, 100); self.progress.setValue(4)
        self.progress.setFormat("Model setup · %p% estimated")
        self.progress.show()
        self.progress_label.setText("Stage: Model loading · preparing CUDA and local components")
        self.progress_detail_label.setText("Loading diffusion model, text encoder, vision projector and VAE. Component percentages are stage estimates, not byte-level measurements.")
        self.status_message("Loading selected local image model and components…")
        self.model_worker = ModelLoadWorker(self.engine)
        self.model_worker.loaded.connect(self._model_loaded)
        self.model_worker.failed.connect(self._model_load_failed)
        self.model_worker.cancelled.connect(self._model_load_cancelled)
        self.model_worker.finished.connect(lambda: self.btn_load_model.setEnabled(True))
        self.model_worker.start()

    def _model_loaded(self, message: str):
        # A ready model is an idle state, not a permanently full progress bar.
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.hide()
        self.progress_label.setText("Stage: Model ready · waiting for a workflow")
        self.progress_detail_label.setText("Model components initialized. Progress appears here when a job starts.")
        self._progress_started_at = None
        self.log_message(message)
        self.statusBar().showMessage(message)
        self.schedule_save()

    def _model_load_failed(self, message: str):
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.show()
        self.progress_label.setText("Stage: Model load failed · review the error and Activity Log")
        self.log_message("MODEL ERROR: " + message.splitlines()[0])
        QMessageBox.critical(self, "Local model load failed", message)
        self.statusBar().showMessage("Local model load failed · see error dialog")

    def _model_load_cancelled(self):
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.show()
        self.progress_label.setText("Stage: Model load cancelled")
        self.log_message("Model loading was cancelled.")
        self.statusBar().showMessage("Model load cancelled")

    def unload_local_model(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Model busy", "Stop the active inference before unloading the model.")
            return
        if self.model_worker and self.model_worker.isRunning():
            QMessageBox.information(self, "Model loading", "Wait for model loading to finish.")
            return
        self.engine.unload_model()
        self.status_message("Local model unloaded · GPU memory can now be used by other applications")
        self.log_message("Local model unloaded on user request.")

    def refresh_system_metrics(self):
        """Schedule a non-blocking system metrics read; never run nvidia-smi on the UI thread."""
        worker = getattr(self, "_hardware_metrics_worker", None)
        if worker is not None and worker.isRunning():
            return
        gpu_index = int(self.gpu_combo.currentData()) if hasattr(self, "gpu_combo") and self.gpu_combo.currentData() is not None else 0
        self._hardware_metrics_worker = HardwareMetricsWorker(gpu_index, self)
        self._hardware_metrics_worker.metrics_ready.connect(self._apply_system_metrics)
        self._hardware_metrics_worker.start()

    def _apply_system_metrics(self, metrics: dict):
        for key, payload in metrics.items():
            if key not in self.hardware_metric_labels:
                continue
            if isinstance(payload, dict):
                self.hardware_metric_labels[key].setText(str(payload.get("text", "N/A")))
                self.hardware_metric_labels[key].setToolTip(str(payload.get("tooltip", "")))
            else:
                self.hardware_metric_labels[key].setText(str(payload))

    def _feature_toggle_label(self, feature: str, enabled: Optional[bool] = None) -> str:
        # Keep the label fixed; the checked state is communicated by color only.
        names = {"ultimate": "▦ Ultimate", "esrgan": "↗ ESRGAN", "lora": "◎ LoRA", "watermark": "◌ Watermark"}
        return names.get(feature, feature)

    def _feature_default(self, feature: str) -> bool:
        keys = {"ultimate": "ultimate_qwen_upscale_enabled", "esrgan": "auto_realesrgan_enabled", "lora": "lora_enabled", "watermark": "watermark_removal_enabled"}
        return bool(self.settings.get(keys[feature], False))

    def _make_feature_toggle_button(self, feature: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("workflowFeatureToggle")
        button.setCheckable(True)
        button.setChecked(self._feature_default(feature))
        button.setText(self._feature_toggle_label(feature, button.isChecked()))
        button.setMinimumHeight(28)
        button.setMinimumWidth(0)
        button.setMaximumHeight(30)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tooltips = {
            "ultimate": "ON: run overlap-aware Qwen tile refinement. Configure output scale, coverage, and denoise in Model Catalog.",
            "esrgan": "ON: run the selected Real-ESRGAN / Real-ESRNet automatically after this workflow. Configure it in Model Catalog.",
            "lora": "Enable or disable the selected local LoRA adapter. Choose the file and strength in Model Catalog.",
            "watermark": "ON: ask the local Qwen model to remove visible watermark overlays and reconstruct the underlying texture in Image Edit, Enhance, Close-up and Square 1:1. OFF leaves watermarks untouched. Complex overlaps are not guaranteed to be removable cleanly.",
        }
        button.setToolTip(tooltips[feature])
        # Keep a cool blue enabled state consistent with the rest of the studio; all four
        # controls share the available width and never append noisy ON/OFF text.
        button.setStyleSheet(
            "QToolButton { border: 1px solid #465568; border-radius: 8px; "
            "padding: 3px 3px; background: #252e3a; color: #d2dae5; font-weight: 500; } "
            "QToolButton:checked { background: #7858f5; border-color: #7858f5; "
            "color: #ffffff; font-weight: 600; } "
            "QToolButton:hover { border-color: #79c9f1; }"
        )
        self._feature_toggle_buttons.setdefault(feature, []).append(button)
        button.toggled.connect(lambda enabled, f=feature, b=button: self._on_workflow_feature_toggled(f, enabled, b))
        return button

    def _make_feature_toggle_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for feature in ("ultimate", "esrgan", "lora", "watermark"):
            row.addWidget(self._make_feature_toggle_button(feature), 1)
        return row

    def _sync_feature_toggle_buttons(self, feature: str, enabled: bool, source=None):
        for button in self._feature_toggle_buttons.get(feature, []):
            if button is source:
                continue
            button.blockSignals(True)
            button.setChecked(bool(enabled))
            button.setText(self._feature_toggle_label(feature))
            button.blockSignals(False)
        for button in self._feature_toggle_buttons.get(feature, []):
            if button is source:
                button.blockSignals(True)
                button.setText(self._feature_toggle_label(feature))
                button.blockSignals(False)

    def _on_workflow_feature_toggled(self, feature: str, enabled: bool, source=None):
        if feature == "ultimate":
            self._global_ultimate_toggled(enabled, source=source)
        elif feature == "esrgan":
            self._global_realesrgan_toggled(enabled, source=source)
        elif feature == "lora":
            self._global_lora_toggled(enabled, source=source)
        elif feature == "watermark":
            self._global_watermark_toggled(enabled, source=source)

    def _on_ultimate_scale_changed(self, *_):
        self.schedule_save()

    def _global_ultimate_toggled(self, enabled: bool, source=None):
        if hasattr(self, "ultimate_qwen_upscale_check") and self.ultimate_qwen_upscale_check.isChecked() != bool(enabled):
            self.ultimate_qwen_upscale_check.blockSignals(True)
            self.ultimate_qwen_upscale_check.setChecked(bool(enabled))
            self.ultimate_qwen_upscale_check.blockSignals(False)
        self._sync_feature_toggle_buttons("ultimate", bool(enabled), source=source)
        if hasattr(self, "enhance_detail_mode"):
            self.enhance_detail_mode.blockSignals(True)
            self.enhance_detail_mode.setCurrentIndex(0 if enabled else 1)
            self.enhance_detail_mode.blockSignals(False)
        self.settings["enhance_detail_mode"] = "ultimate" if enabled else "fast"
        self.settings["ultimate_qwen_upscale_enabled"] = bool(enabled)
        self.schedule_save()

    def _global_realesrgan_toggled(self, enabled: bool, source=None):
        self._sync_feature_toggle_buttons("esrgan", bool(enabled), source=source)
        self.settings["auto_realesrgan_enabled"] = bool(enabled)
        self.log_message("Automatic Real-ESRGAN/Real-ESRNet post-process " + ("enabled" if enabled else "disabled") + " for all image workflows.")
        self.schedule_save()

    def _global_lora_toggled(self, enabled: bool, source=None):
        if hasattr(self, "lora_enabled_check") and self.lora_enabled_check.isChecked() != bool(enabled):
            self.lora_enabled_check.blockSignals(True)
            self.lora_enabled_check.setChecked(bool(enabled))
            self.lora_enabled_check.blockSignals(False)
        self._sync_feature_toggle_buttons("lora", bool(enabled), source=source)
        self.settings["lora_enabled"] = bool(enabled)
        self.schedule_save()

    def _global_watermark_toggled(self, enabled: bool, source=None):
        """Enable/disable watermark-removal instructions for supported AI workflows."""
        self._sync_feature_toggle_buttons("watermark", bool(enabled), source=source)
        self.settings["watermark_removal_enabled"] = bool(enabled)
        self.log_message("Watermark removal " + ("enabled" if enabled else "disabled") + " for Image Edit, Enhance, Close-up and Square 1:1 AI passes.")
        self.schedule_save()


    def toggle_theme(self):
        """Switch between Light and Dark mode with one click from the top bar."""
        if not hasattr(self, "theme_combo"):
            return
        current = self.theme_combo.currentText().strip().lower()
        self.theme_combo.setCurrentText("light" if current != "light" else "dark")
        self.schedule_save()

    def _apply_theme(self, theme: str):
        if not hasattr(self, "theme_combo"):
            return
        is_light = str(theme).strip().lower() == "light"
        if hasattr(self, "btn_theme_toggle"):
            # Show the destination theme, matching the reference screenshots:
            # in Dark mode offer Light mode (sun); in Light mode offer Dark mode (moon).
            self.btn_theme_toggle.setText("☾  Dark mode" if is_light else "☀  Light mode")
            self.btn_theme_toggle.setToolTip(
                "Click to switch to Dark mode." if is_light
                else "Click to switch to Light mode."
            )

        # Use one P.AI studio design system across all workflows. The selected theme
        # changes only the palette; workflow-specific geometry and controls stay intact.
        studio_skin = str(theme).lower() == "light"
        if hasattr(self, "preview"):
            self.preview.set_studio_skin(studio_skin)

        if studio_skin:
            self.setStyleSheet("""
                QWidget { background: #f3f5fb; color: #2b3650; font-size: 9pt; }
                QFrame#hardwareMonitor { background: #e7ebf5; border: 1px solid #d3d9e8; border-radius: 9px; }
                QLabel#hardwareMetricPill { background: #f8faff; color: #33415e; border: 1px solid #d7deed; border-radius: 6px; padding: 4px 5px; font-size: 8pt; font-weight: 700; }
                QMainWindow, QDialog { background: #f3f5fb; }
                QLabel { background: transparent; }
                QGroupBox { background: #ffffff; border: 1px solid #d9e0ef; border-radius: 9px; margin-top: 11px; padding: 8px; font-weight: 600; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #34415d; }
                QPushButton, QToolButton { background: #ffffff; color: #34415d; border: 1px solid #d6deef; border-radius: 7px; padding: 6px 9px; }
                QPushButton:hover, QToolButton:hover { background: #f3efff; border-color: #9c89fa; }
                QPushButton:pressed, QToolButton:pressed { background: #e8e1ff; }
                QPushButton:checked, QToolButton:checked { background: #7858f5; color: #ffffff; border-color: #7858f5; }
                QPushButton:disabled, QToolButton:disabled { background: #edf0f7; color: #a1abc0; border-color: #e0e5ef; }
                QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
                    background: #ffffff; color: #2c3851; border: 1px solid #d4ddef; border-radius: 6px; padding: 5px; selection-background-color: #7858f5; selection-color: #ffffff;
                }
                QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #927bf7; }
                QComboBox QAbstractItemView { background: #ffffff; color: #2c3851; selection-background-color: #7858f5; selection-color: #ffffff; border: 1px solid #d4ddef; }
                QHeaderView::section { background: #f1f4fb; color: #51607c; padding: 6px; border: 0; border-bottom: 1px solid #d9e0ef; }
                QTableWidget { alternate-background-color: #f7f8fd; gridline-color: #e5e9f3; }
                QTableWidget::item:selected { background: #e8e1ff; color: #2c2752; }
                QTabBar::tab { background: #eceff8; color: #53617d; padding: 7px 12px; margin-right: 3px; border: 1px solid #d9e0ef; border-bottom: 0; border-top-left-radius: 6px; border-top-right-radius: 6px; }
                QTabBar::tab:selected { background: #7858f5; color: #ffffff; border-color: #7858f5; }
                QProgressBar { background: #ffffff; color: #4d5871; border: 1px solid #d5ddef; border-radius: 6px; text-align: center; min-height: 13px; }
                QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7858f5, stop:1 #9b83ff); border-radius: 5px; }
                QSplitter::handle { background: #e0e6f3; }
                QScrollBar:vertical { background: #eef1f8; width: 12px; margin: 2px; border-radius: 5px; }
                QScrollBar::handle:vertical { background: #c4cde0; min-height: 24px; border-radius: 5px; }
                QScrollBar::handle:vertical:hover { background: #a99af0; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
                QToolTip { background: #ffffff; color: #2c3851; border: 1px solid #cfd8ec; padding: 4px; }
            """)
            if hasattr(self, "preview_info_label"):
                self.preview_info_label.setStyleSheet("color: #7683a0; font-size: 9pt;")
            if hasattr(self, "progress_label"):
                self.progress_label.setStyleSheet("color: #7683a0; font-size: 9pt;")
            if hasattr(self, "brand_title_label"):
                self.brand_title_label.setStyleSheet("font-size: 15pt; font-weight: 800; color: #7858f5;")
            if hasattr(self, "brand_subtitle_label"):
                self.brand_subtitle_label.setStyleSheet("font-size: 8pt; color: #7583a0;")
            return

        dark = theme.lower() == "dark"
        if dark:
            self.setStyleSheet("""
                QWidget { background: #20242b; color: #e7eaf0; font-size: 9pt; }
                QMainWindow, QDialog { background: #20242b; }
                QFrame#hardwareMonitor { background: #191d24; border: 1px solid #343b48; border-radius: 9px; }
                QLabel#hardwareMetricPill { background: #242a34; color: #dce6f4; border: 1px solid #3a4352; border-radius: 6px; padding: 4px 5px; font-size: 8pt; font-weight: 700; }
                QLabel { background: transparent; }
                QGroupBox { background: #242932; border: 1px solid #3b424d; border-radius: 9px; margin-top: 11px; padding: 8px; font-weight: 600; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #e7eaf0; }
                QPushButton, QToolButton { background: #303743; color: #e7eaf0; border: 1px solid #4a5361; border-radius: 7px; padding: 5px 8px; }
                QPushButton:hover, QToolButton:hover { background: #3c354f; border-color: #927bf7; }
                QPushButton:pressed, QToolButton:pressed { background: #44366f; }
                QPushButton:checked, QToolButton:checked { background: #7858f5; color: #ffffff; border-color: #7858f5; }
                QPushButton:disabled, QToolButton:disabled { background: #292e37; color: #7e8798; border-color: #343b46; }
                QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background: #171a20; color: #e7eaf0; border: 1px solid #424a56; border-radius: 6px; padding: 4px; selection-background-color: #7858f5; selection-color: #ffffff; }
                QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #927bf7; }
                QComboBox QAbstractItemView { background: #20242b; color: #e7eaf0; selection-background-color: #7858f5; selection-color: #ffffff; }
                QHeaderView::section { background: #303743; color: #d5d9e2; padding: 6px; border: 0; border-bottom: 1px solid #3b424d; }
                QTableWidget { alternate-background-color: #222730; gridline-color: #343b46; }
                QTableWidget::item:selected { background: #44366f; color: #ffffff; }
                QTabBar::tab { background: #303743; color: #c5cad5; padding: 7px 12px; margin-right: 3px; border: 1px solid #3b424d; border-bottom: 0; border-top-left-radius: 6px; border-top-right-radius: 6px; }
                QTabBar::tab:selected { background: #7858f5; color: #ffffff; border-color: #7858f5; }
                QProgressBar { background: #171a20; color: #e7eaf0; border: 1px solid #4a5361; border-radius: 6px; text-align: center; min-height: 13px; }
                QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7858f5, stop:1 #9b83ff); border-radius: 5px; }
                QSplitter::handle { background: #343b46; }
                QScrollBar:vertical { background: #242932; width: 10px; margin: 2px; border-radius: 5px; }
                QScrollBar::handle:vertical { background: #505867; min-height: 24px; border-radius: 5px; }
                QScrollBar::handle:vertical:hover { background: #927bf7; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
                QToolTip { background: #242932; color: #e7eaf0; border: 1px solid #505867; padding: 4px; }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background: #f3f5f8; color: #20242b; font-size: 10pt; }
                QGroupBox { border: 1px solid #c8ced8; border-radius: 7px; margin-top: 9px; padding: 8px; font-weight: 600; }
                QPushButton { background: #e3e8ef; border: 1px solid #bbc4d1; border-radius: 5px; padding: 6px 9px; }
                QPushButton:hover { background: #d6e2f0; }
                QPushButton:checked { background: #b7d8f3; border-color: #4b9bd2; }
                QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget { background: white; border: 1px solid #c8ced8; border-radius: 4px; padding: 4px; }
                QHeaderView::section { background: #e3e8ef; padding: 5px; border: 0; }
                QTabBar::tab { background: #e3e8ef; padding: 7px 12px; }
                QTabBar::tab:selected { background: #b7d8f3; }
            """)
        if hasattr(self, "brand_title_label"):
            self.brand_title_label.setStyleSheet("font-size: 15pt; font-weight: 800; color: #72d9f0;" if dark else "font-size: 15pt; font-weight: 800; color: #7858f5;")
        if hasattr(self, "brand_subtitle_label"):
            self.brand_subtitle_label.setStyleSheet("font-size: 8pt; color: #9aa6b7;" if dark else "font-size: 8pt; color: #7583a0;")

    def _append_log_message(self, formatted_message: str):
        # This slot is delivered by Qt on the GUI thread. Buffer early messages until
        # the Activity Log widget has been created during initial construction.
        if hasattr(self, "log_edit"):
            self.log_edit.appendPlainText(formatted_message)
        else:
            self._pending_log_messages.append(formatted_message)

    def log_message(self, message: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_signal.emit(f"[{stamp}] {message}")

    def status_message(self, message: str):
        # A Qt signal ensures worker-thread status updates are queued onto the GUI thread.
        self.status_signal.emit(str(message))

    def _on_preview_mode_changed(self, mode: str):
        """Apply a user-selected comparison view and persist it for source workflows."""
        if getattr(getattr(self, "preview", None), "presentation_policy", "live_only") == "selectable":
            if mode in {"Enhanced", "Original", "Slider before/after", "Side by side", "Overlay", "Difference", "Flicker"}:
                self._preferred_preview_mode = mode
                self.settings["preview_mode"] = mode
                self.schedule_save()
        self.preview.set_mode(mode)

    def _apply_preview_presentation(self, workflow: str):
        """Generate is live-only; source workflows allow the user to choose a view mode."""
        policy = preview_presentation_for_workflow(workflow)
        live_only = policy == "live_only"
        if hasattr(self, "preview"):
            self.preview.set_presentation_policy(policy)
            self.preview.set_live_caption("LIVE · PREVIEW")
        if hasattr(self, "preview_mode_label"):
            self.preview_mode_label.setVisible(not live_only)
        if hasattr(self, "preview_mode"):
            self.preview_mode.setVisible(not live_only)
            self.preview_mode.setEnabled(not live_only)
            desired_mode = "Enhanced" if live_only else str(getattr(self, "_preferred_preview_mode", "Side by side"))
            if self.preview_mode.currentText() != desired_mode:
                self.preview_mode.blockSignals(True)
                self.preview_mode.setCurrentText(desired_mode)
                self.preview_mode.blockSignals(False)
            if hasattr(self, "preview"):
                self.preview.set_mode(desired_mode)

    def _switch_workflow_layout(self, workflow: str):
        self._apply_preview_presentation(workflow)
        if not getattr(self, "_restore_saved_prompt_presets", False):
            self.refresh_std_prompt_presets(workflow)
        studio = workflow in ("image_generate", "image_edit")
        queue_below_preview = workflow in ("enhance", "lora_closeup_1024", "square_1_1")
        self.gen_bottom_panel.setVisible(workflow == "image_generate")
        self.edit_bottom_panel.setVisible(workflow == "image_edit")
        self.lora_group.setVisible(workflow == "lora_closeup_1024")
        self.square_mode_group.setVisible(workflow == "square_1_1")
        self.workflow_queue_actions.setVisible(queue_below_preview)
        qwen_workflow = queue_below_preview
        # The shared Scale / Custom output controls belong only to Enhance.
        # Close-up and Square expose their dedicated fixed 1K / 2K / 4K buttons.
        self.qwen_output_group.setVisible(workflow == "enhance")
        self.model_preset_group.setVisible(qwen_workflow)
        self.std_output_size_widget.setVisible(workflow == "enhance")
        self.std_size_preset.setEnabled(workflow == "enhance")
        self.qwen_output_group.setMinimumWidth(0)
        self.realesrgan_box.setMinimumWidth(0)
        # Enhance / Close-up / Square: Queue controls first, Input/Output second, settings next.
        # Queue controls collapse in Generate/Edit, leaving Input/Output at the top there too.
        basic_layout = getattr(self, "basic_tab_layout", None)
        files_box = getattr(self, "files_box", None)
        if basic_layout is not None and files_box is not None:
            basic_layout.removeWidget(self.workflow_queue_actions)
            basic_layout.removeWidget(files_box)
            basic_layout.insertWidget(0, self.workflow_queue_actions)
            basic_layout.insertWidget(1, files_box)

        if queue_below_preview:
            # Keep Settings as a full-height left column; the central workspace owns
            # the preview on top and a horizontal Queue | Activity Log row below it.
            if self.sidebar.parentWidget() is self.main_splitter:
                self.sidebar.setParent(None)
            if self.left_panel_shell.parentWidget() is self.workspace_splitter:
                self.left_panel_shell.setParent(None)
            if self.sidebar.parentWidget() is not self.workspace_splitter:
                self.workspace_splitter.insertWidget(0, self.sidebar)
            self.sidebar.setVisible(True)
            self.sidebar.setMinimumWidth(330)
            self.sidebar.setMaximumWidth(500)
            self.workspace_splitter.setSizes([410, 1150])
            # main_splitter now contains only the preview; it takes the full right pane.
            self.main_splitter.setSizes([0, 1000])

            # Reparent Queue and Activity Log into a single side-by-side row.
            for panel in (self.queue_panel, self.log_panel):
                if panel.parentWidget() is not None:
                    panel.setParent(None)
                if self.workflow_bottom_row.indexOf(panel) < 0:
                    self.workflow_bottom_row.addWidget(panel)
                panel.setVisible(True)
            self.workflow_bottom_row.setStretchFactor(0, 3)
            self.workflow_bottom_row.setStretchFactor(1, 2)
            self.workflow_bottom_row.setSizes([650, 500])
            self.workspace_brand_group.setVisible(False)
            self.left_panel_shell.setVisible(False)
            if self.workflow_bottom_row.parentWidget() is not self.right_vertical_splitter:
                if self.workflow_bottom_row.parentWidget() is not None:
                    self.workflow_bottom_row.setParent(None)
                self.right_vertical_splitter.insertWidget(1, self.workflow_bottom_row)
            self.workflow_bottom_row.setMinimumHeight(150)
            self.right_vertical_splitter.setStretchFactor(0, 4)
            self.right_vertical_splitter.setStretchFactor(1, 1)
            self.right_vertical_splitter.setStretchFactor(2, 0)
            self.right_vertical_splitter.setStretchFactor(3, 0)
            self.right_vertical_splitter.setSizes([700, 250, 0, 0])
            self.left_panel_shell.setMinimumWidth(0)
            self.left_panel_shell.setMaximumWidth(0)
            self.btn_show_left_dock.setVisible(False)
            self.btn_toggle_left_dock.setVisible(False)
        else:
            # Image Generate / Image Edit retain their existing studio arrangement:
            # their workflow-specific panel is shown under the preview, while the
            # Queue shell returns to the left workspace column.
            if self.workflow_bottom_row.parentWidget() is self.right_vertical_splitter:
                self.workflow_bottom_row.setParent(None)
            self.workflow_bottom_row.setMinimumHeight(0)

            if self.sidebar.parentWidget() is self.workspace_splitter:
                self.sidebar.setParent(None)
            if self.sidebar.parentWidget() is not self.main_splitter:
                self.main_splitter.insertWidget(0, self.sidebar)
            self.sidebar.setVisible(False)
            self.main_splitter.setSizes([370, 1050])

            # Restore Queue and Activity Log to the original outer dock arrangement.
            for panel in (self.queue_panel, self.log_panel):
                if panel.parentWidget() is not None:
                    panel.setParent(None)
            self.workspace_content_splitter.insertWidget(0, self.queue_panel)
            self.workspace_content_splitter.insertWidget(1, self.workspace_brand_group)
            self.left_dock_splitter.insertWidget(0, self.workspace_content_splitter)
            self.left_dock_splitter.insertWidget(1, self.log_panel)
            self.left_dock_splitter.setStretchFactor(0, 5)
            self.left_dock_splitter.setStretchFactor(1, 1)
            self.left_dock_splitter.setSizes([480, 220])
            self.workspace_brand_group.setVisible(False)

            if self.left_panel_shell.parentWidget() is not self.workspace_splitter:
                if self.left_panel_shell.parentWidget() is not None:
                    self.left_panel_shell.setParent(None)
                self.workspace_splitter.insertWidget(0, self.left_panel_shell)
            self.left_panel_shell.setVisible(True)
            self.left_panel_shell.setMinimumWidth(330)
            self.left_panel_shell.setMaximumWidth(500)
            self.left_panel_shell.setMinimumHeight(0)
            self.workspace_splitter.setSizes([410, 1150])
            self.right_vertical_splitter.setStretchFactor(0, 4)
            if workflow == "image_generate":
                self.right_vertical_splitter.setSizes([520, 370, 0])
            else:
                self.right_vertical_splitter.setSizes([510, 0, 380])
            self.btn_show_left_dock.setVisible(False)
            self.btn_toggle_left_dock.setVisible(True)

        if hasattr(self, "theme_combo"):
            self._apply_theme(self.theme_combo.currentText())

    def apply_workflow_preset(self, workflow: str, save: bool = True):
        allowed = {"image_generate", "image_edit", "enhance", "lora_closeup_1024", "square_1_1"}
        if workflow not in allowed:
            workflow = "image_generate"
        self.workflow = workflow
        # Switching/starting a workflow always begins by showing the source. A result-specific
        # comparison mode is applied only after that workflow completes successfully.
        if hasattr(self, "preview_mode"):
            self.preview_mode.blockSignals(True)
            self.preview_mode.setCurrentText("Original")
            self.preview_mode.blockSignals(False)
            self.preview.set_mode("Original")
        if workflow == "image_generate" and hasattr(self, "gen_quality"):
            self._set_steps_from_preset(self.gen_quality.currentText(), save=False)
        elif workflow == "image_edit" and hasattr(self, "edit_quality"):
            self._set_steps_from_preset(self.edit_quality.currentText(), save=False)
        elif hasattr(self, "std_quality"):
            self._set_steps_from_preset(self.std_quality.currentText(), save=False)
        for button, mode in self.workflow_buttons:
            button.setChecked(mode == workflow)
        self._switch_workflow_layout(workflow)
        self.statusBar().showMessage(f"Workflow: {workflow}")
        if save:
            self.schedule_save()

    def toggle_left_dock(self):
        visible = self.left_panel_shell.isVisible()
        self.left_panel_shell.setVisible(not visible)
        self.btn_show_left_dock.setVisible(visible)
        self.btn_toggle_left_dock.setVisible(not visible)
        self.btn_show_left_dock.setText("☰ Queue" if visible else "☰ Hide Queue")

    def open_settings_dialog(self):
        if hasattr(self, "settings_dialog"):
            self.model_settings_widget.setVisible(True)
            self.settings_dialog.exec()
            self.schedule_save()

    def stop_after_current_job(self):
        """Finish the active image/job, then stop before starting another queued image."""
        if not self.worker or not self.worker.isRunning():
            self.log_message("No active job to stop after the current image.")
            return
        self._stop_after_current = True
        self.engine.stop_after_current.set()
        self.log_message("Stop current job requested; the active image will finish, then queued work will stop.")

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "estimating…"
        seconds = int(round(seconds))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m {seconds % 60:02d}s"

    def _balance_live_preview_colors(self, frame: Image.Image) -> Image.Image:
        """Correct the yellow/warm channel bias of Qwen 16-channel fast latent projection (proj)."""
        method = str(getattr(self.engine, "live_preview_method", "proj") or "proj").strip().lower()
        if method != "proj":
            return frame
        try:
            import numpy as np
            arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return frame
            ref_img = getattr(self, "current_original", None)
            if ref_img is not None and getattr(self, "workflow", "") != "image_generate":
                ref_small = ImageOps.exif_transpose(ref_img).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                ref_arr = np.asarray(ref_small, dtype=np.float32)
                for ch in range(3):
                    p_mean = float(arr[:, :, ch].mean())
                    p_std = max(6.0, float(arr[:, :, ch].std()))
                    r_mean = float(ref_arr[:, :, ch].mean())
                    r_std = max(6.0, float(ref_arr[:, :, ch].std()))
                    corrected = (arr[:, :, ch] - p_mean) * (r_std / p_std) + r_mean
                    arr[:, :, ch] = 0.85 * corrected + 0.15 * arr[:, :, ch]
            else:
                # Damped gray-world + percentile auto-levels for text-to-image latent projection
                ch_means = [float(arr[:, :, c].mean()) for c in range(3)]
                ch_stds = [max(8.0, float(arr[:, :, c].std())) for c in range(3)]
                target_mean = sum(ch_means) / 3.0
                target_std = sum(ch_stds) / 3.0
                for ch in range(3):
                    centered = (arr[:, :, ch] - ch_means[ch]) * (target_std / ch_stds[ch]) + target_mean
                    arr[:, :, ch] = 0.82 * centered + 0.18 * arr[:, :, ch]
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        except Exception:
            return frame

    def _refresh_live_generation_preview(self):
        """Render the latest native intermediate frame as it arrives from sd-server."""
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning() or getattr(self, "workflow", "") == "upscale_only":
            return
        if hasattr(self, "live_preview_method_combo") and self.live_preview_method_combo.currentData() == "none":
            return
        preview = getattr(self.engine, "latest_preview_image", None)
        if preview is None:
            return
        try:
            revision = int(getattr(self.engine, "latest_preview_revision", 0) or 0)
            if revision == self._live_preview_revision:
                return
            self._live_preview_revision = revision
            preview = self._balance_live_preview_colors(preview)
            active_workflow = str(getattr(getattr(self, "_active_job", None), "workflow", self.workflow) or self.workflow)
            source_image, live_image = live_preview_images_for_workflow(active_workflow, self.current_original, preview)
            self._apply_preview_presentation(active_workflow)
            if active_workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:
                self.preview.set_mode("Side by side")
                if hasattr(self, "preview_mode") and self.preview_mode.currentText() != "Side by side":
                    self.preview_mode.blockSignals(True)
                    self.preview_mode.setCurrentText("Side by side")
                    self.preview_mode.blockSignals(False)
            active_ctx = str(getattr(self.engine, "_active_pipeline_context", "") or "").strip()
            tile_marker = getattr(self.engine, "_latest_preview_marker", None)
            tile_caption = "LIVE · PREVIEW"
            tile_badge = ""
            if isinstance(tile_marker, tuple) and len(tile_marker) >= 2 and isinstance(tile_marker[1], tuple):
                stage_tag = str(tile_marker[1][0])
                if stage_tag in {"ultimate-tile-active", "ultimate-tile-complete"} and len(tile_marker[1]) >= 4:
                    t_idx, t_tot = tile_marker[1][2], tile_marker[1][3]
                    tile_caption = f"LIVE · ULTIMATE TILE {t_idx}/{t_tot}"
                    tile_badge = f" · Ultimate Tile {t_idx}/{t_tot}"
            elif "Ultimate" in active_ctx:
                tile_caption = "LIVE · ULTIMATE TILE"
                tile_badge = " · Ultimate Tile"
            self.preview.set_live_caption(tile_caption)
            # The input is established at job start; only seed it here if a workflow switch
            # left the canvas without a source. Regular sampler frames only update the Live pane.
            if source_image is not None and self.preview.original is None:
                self.preview.set_images(source_image, None)
            self.preview.set_live_image(live_image, preserve_view=True)
            if hasattr(self, "preview_info_label"):
                step = int(getattr(self.engine, "latest_preview_step", 0) or 0)
                total = int(getattr(self.engine, "latest_preview_total_steps", 0) or 0)
                pass_no = int(getattr(self.engine, "latest_preview_pass", 0) or 0)
                step_text = f" · Step {step}/{total}" if step and total else ""
                pass_text = f" · Pass {pass_no}" if pass_no else ""
                self.preview_info_label.setText(
                    f"Live preview{tile_badge}{step_text}{pass_text} · {preview.width}×{preview.height}"
                )
        except Exception as exc:
            self.log_message(f"Could not update live preview: {exc}")

    def _apply_status_message(self, message: str):
        """Map native backend callbacks to stage, step, global progress and an ETA estimate."""
        message = str(message or "").strip()
        self.statusBar().showMessage(message)
        if not hasattr(self, "progress_label"):
            return

        now = time.monotonic()
        active_worker = bool((self.worker and self.worker.isRunning()) or (self.model_worker and self.model_worker.isRunning()))
        elapsed_start = self._job_started_at if self.worker and self.worker.isRunning() else self._progress_started_at
        elapsed = max(0.0, now - elapsed_start) if elapsed_start is not None else None
        lower = message.casefold()
        step_match = re.search(r"step\s+(\d+)\s*/\s*(\d+)", message, re.IGNORECASE)
        refinement_match = re.search(r"(?:pass\s+(\d+)\s*/\s*(\d+)).*?tile\s+(\d+)\s*/\s*(\d+)", message, re.IGNORECASE)
        image_match = re.search(r"image\s+(\d+)\s*/\s*(\d+)", message, re.IGNORECASE)
        explicit_pct = re.search(r"(\d{1,3})%\s+estimated", message, re.IGNORECASE)
        stage = "Working"
        overall_pct = 10
        detail = "Waiting for the next backend checkpoint."
        step_eta_text: Optional[str] = None

        # A model that has loaded successfully is idle, not at 100% on a fake progress bar.
        if not active_worker and any(term in lower for term in ("gpu ready", "model ready", "local model unloaded")):
            self.progress.hide()
            self.progress_label.setText("Stage: Model ready · waiting for a workflow" if "unloaded" not in lower else "Stage: Local model unloaded")
            self.progress_detail_label.setText("The stage tracker starts when model loading or a workflow is active.")
            return

        if "model load" in lower or "loading model on nvidia" in lower or "loading selected local image model" in lower or "checking local engine" in lower:
            stage = "Model loading / GPU preparation"
            if explicit_pct:
                overall_pct = max(4, min(90, int(explicit_pct.group(1))))
                detail = message
            else:
                overall_pct = max(5, self.progress.value())
                detail = "Loading diffusion weights, text encoder, vision projector and VAE; percentage is an estimate."
        elif "text encoder" in lower or "text conditioning" in lower or "prompt conditioning" in lower or "conditioning cache" in lower:
            if "ultimate qwen refinement" in lower or "qwen detail refinement" in lower:
                ctx = self._active_refinement_context or {}
                stage = str(ctx.get("stage", "Ultimate Qwen refinement" if "ultimate" in lower else "Qwen detail refinement"))
                overall_pct = max(78, int(78 + 15 * ((max(0, ctx.get("tile", 1) - 1)) / max(1, ctx.get("tile_total", 1)))))
                detail = message
            else:
                stage = "Text encoder / prompt conditioning"
                overall_pct = 13
                detail = message
                self._last_step_index = 0
                self._last_step_total = 0
                self._first_step_at = None
                self._step_interval_ema = None
                self._last_step_changed_at = None
                self._active_refinement_context = None
        elif "vae encode" in lower or "encode_first_stage" in lower or "target t_enc" in lower:
            stage = "VAE · encoding reference image"
            overall_pct = 19
            detail = message
        elif step_match:
            step, total = int(step_match.group(1)), max(1, int(step_match.group(2)))
            self._last_step_total = total
            if step != self._last_step_index:
                if self._first_step_at is None or step < self._last_step_index:
                    self._first_step_at = now
                    self._step_interval_ema = None
                elif self._last_step_index > 0 and self._last_step_changed_at is not None:
                    interval = (now - self._last_step_changed_at) / max(1, step - self._last_step_index)
                    if 0 < interval < 600:
                        self._step_interval_ema = interval if self._step_interval_ema is None else (0.65 * self._step_interval_ema + 0.35 * interval)
                self._last_step_index = step
                self._last_step_changed_at = now
            is_refinement = bool(self._active_refinement_context)
            if is_refinement:
                ctx = self._active_refinement_context or {}
                pass_idx = max(1, ctx.get("pass", 1))
                pass_total = max(1, ctx.get("pass_total", 1))
                tile_idx = max(1, ctx.get("tile", 1))
                tile_total = max(1, ctx.get("tile_total", 1))
                unit_fraction = ((pass_idx - 1) * tile_total + tile_idx - 1 + step / total) / max(1, pass_total * tile_total)
                base = 78 if "ultimate" in lower or "ultimate" in str(ctx.get("stage", "")).casefold() else 76
                span = 15
                overall_pct = int(base + span * unit_fraction)
                stage = str(ctx.get("stage", "Qwen detail refinement"))
                detail = f"Pass {pass_idx}/{pass_total} · Tile {tile_idx}/{tile_total} · Denoising step {step}/{total} ({step} steps completed)"
            else:
                overall_pct = int(17 + 62 * step / total)
                stage = "Qwen diffusion · denoising"
                detail = f"Denoising step {step}/{total} · {step} of {total} steps completed"
            if step >= total:
                stage = (str((self._active_refinement_context or {}).get("stage", "Qwen detail refinement")) + " · VAE decode") if is_refinement else "VAE · decoding generated image"
                overall_pct = max(overall_pct, 88 if is_refinement else 82)
                detail += " · denoising complete; decoding the image"
            elif step > 1 and self._first_step_at is not None:
                avg = self._step_interval_ema or ((now - self._first_step_at) / max(1, step - 1))
                step_eta_text = self._format_duration(max(0.0, total - step) * avg)
        elif "vae decode complete" in lower or "vae decode" in lower or "decode_first_stage" in lower or "latent 1 decoded" in lower:
            stage = "VAE · decoding generated image"
            overall_pct = 87
            detail = message
            if self._active_refinement_context:
                ctx = self._active_refinement_context
                stage = str(ctx.get("stage", "Qwen detail refinement")) + " · VAE decode"
                detail = f"{ctx.get('stage')} · Pass {ctx.get('pass', 1)}/{ctx.get('pass_total', 1)} · Tile {ctx.get('tile', 1)}/{ctx.get('tile_total', 1)} · VAE decoding the tile"
                overall_pct = max(78, int(78 + 15 * ((ctx.get('tile', 1) - 1) / max(1, ctx.get('tile_total', 1)))))
        elif refinement_match or "qwen detail refinement" in lower or "ultimate qwen refinement" in lower:
            if refinement_match:
                pass_idx, pass_total, tile_idx, tile_total = [int(x) for x in refinement_match.groups()]
                stage = "Ultimate Qwen refinement" if "ultimate" in lower else "Qwen detail refinement"
                self._active_refinement_context = {"stage": stage, "pass": pass_idx, "pass_total": pass_total,
                                                   "tile": tile_idx, "tile_total": tile_total}
                unit_fraction = ((pass_idx - 1) * tile_total + tile_idx - 1) / max(1, pass_total * tile_total)
                overall_pct = int(78 + 15 * unit_fraction)
                detail = f"Pass {pass_idx}/{pass_total} · Tile {tile_idx}/{tile_total} · starting Qwen detail reconstruction"
            else:
                stage = "Ultimate Qwen refinement" if "ultimate" in lower else "Qwen detail refinement"
                overall_pct = 79
                detail = message
        elif "esrgan" in lower or "ai upscale" in lower or "vulkan upscaler" in lower:
            stage = "Real-ESRGAN / Real-ESRNet · Vulkan upscaling"
            if image_match:
                image_i, image_total = [int(x) for x in image_match.groups()]
                overall_pct = int(92 + 6 * (image_i - 1) / max(1, image_total))
                detail = f"Image {image_i}/{image_total} · selected NCNN model on Vulkan"
                if "complete" in lower or "saved" in lower:
                    overall_pct = int(92 + 6 * image_i / max(1, image_total))
                    detail = f"Image {image_i}/{image_total} · upscaled output saved"
            elif "complete" in lower:
                overall_pct = 98
                detail = message
            else:
                overall_pct = 94
                detail = message + " · progress depends on the Vulkan CLI, which does not expose reliable per-tile percentages"
            self._active_refinement_context = None
        elif "finalizing output" in lower or "validating saved files" in lower:
            stage = "Finalizing · verifying outputs and queue"
            overall_pct = 99
            detail = message
            self._active_refinement_context = None
        elif "memory shortage" in lower or "retrying once" in lower:
            stage = "CUDA memory recovery"
            overall_pct = max(12, self.progress.value())
            detail = message
        elif "download" in lower:
            stage = "Downloading model / runtime components"
            overall_pct = max(4, self.progress.value())
            detail = message
        elif "vae decode complete" in lower or "local inference finished" in lower:
            stage = "Image tensor received · VAE decode complete"
            overall_pct = max(82, self.progress.value())
            detail = message
            if self._active_refinement_context:
                ctx = self._active_refinement_context
                detail = f"{ctx.get('stage')} · Pass {ctx.get('pass', 1)}/{ctx.get('pass_total', 1)} · Tile {ctx.get('tile', 1)}/{ctx.get('tile_total', 1)} complete"
                overall_pct = max(overall_pct, int(78 + 15 * ((ctx.get('tile', 1)) / max(1, ctx.get('tile_total', 1)))))
        else:
            stage = message or "Working"
            overall_pct = max(5, self.progress.value()) if active_worker else 0
            detail = message or "Waiting for status."

        if not active_worker:
            self.progress.hide()
            self.progress_label.setText("Stage: " + stage)
            self.progress_detail_label.setText(detail)
            return

        # Keep the measured denoising-step ETA separate from the coarse end-to-end estimate.
        # The latter is intentionally labelled approximate because VAE, refinement and Vulkan
        # phases have different costs and some backends expose no per-phase progress percentage.
        overall_eta_text = "estimating…"
        if elapsed is not None and overall_pct >= 18 and overall_pct < 100:
            overall_eta = elapsed * (100 - overall_pct) / max(1, overall_pct)
            overall_eta_text = self._format_duration(overall_eta)
        if step_eta_text:
            eta = f"Step ETA ~{step_eta_text} · overall ETA ~{overall_eta_text} (approx.)"
        elif overall_pct < 100:
            eta = f"Overall ETA ~{overall_eta_text} (approx.)"
        else:
            eta = "ETA ~completing"
        pct = max(3, min(99, int(overall_pct)))
        self.progress.setRange(0, 100)
        self.progress.setValue(max(self.progress.value(), pct))
        self.progress.setFormat(f"Overall {self.progress.value()}% · %v/100")
        self.progress.show()
        self._active_stage_name = stage
        elapsed_text = self._format_duration(elapsed) if elapsed is not None else "estimating…"
        unit_status = ""
        if step_match:
            step_number, step_total = int(step_match.group(1)), max(1, int(step_match.group(2)))
            unit_status = f"Step {step_number}/{step_total}"
        elif refinement_match:
            pass_number, pass_total, tile_number, tile_total = [int(v) for v in refinement_match.groups()]
            unit_status = f"Pass {pass_number}/{pass_total} · Tile {tile_number}/{tile_total}"
        elif image_match and any(token in lower for token in ("esrgan", "ai upscale", "vulkan upscaler")):
            unit_status = f"Image {int(image_match.group(1))}/{max(1, int(image_match.group(2)))}"
        headline = f"Stage: {stage}"
        if unit_status:
            headline += f" · {unit_status}"
        headline += f" · {self.progress.value()}% overall"
        self.progress_label.setText(headline)
        if unit_status:
            self.progress.setFormat(f"{unit_status} · Overall {self.progress.value()}%")
        else:
            self.progress.setFormat(f"Overall {self.progress.value()}% · %v/100")
        self.progress_detail_label.setText(f"{detail} · elapsed {elapsed_text} · {eta}")

    @staticmethod
    def estimate_pixel_change(original: Optional[Image.Image], image: Image.Image) -> Optional[float]:
        if original is None:
            return None
        try:
            output = image.convert("RGB")
            scale = min(1.0, 512.0 / max(1, output.width, output.height))
            sample_size = (max(1, int(round(output.width * scale))), max(1, int(round(output.height * scale))))
            # Difference scoring is only a coarse UI warning. Sampling at <=512 px
            # avoids blocking the Qt event loop on multi-megapixel outputs.
            source = ImageOps.fit(original.convert("RGB"), sample_size, method=Image.Resampling.BILINEAR)
            output_small = output.resize(sample_size, Image.Resampling.BILINEAR)
            means = ImageStat.Stat(ImageChops.difference(source, output_small)).mean
            return (sum(means) / 3 / 255) * 100
        except Exception:
            return None

    def update_preview_info(self, image: Optional[Image.Image] = None, path: Optional[str] = None,
                            original: Optional[Image.Image] = None, elapsed: Optional[float] = None):
        if image is None:
            image = self.current_enhanced or self.current_original
        if image is None:
            self.preview_info_label.setText("No image selected")
            return
        output_resolution = f"{image.width}×{image.height}"
        if original is not None:
            source_resolution = f"{original.width}×{original.height}"
            resolution = f"Source {source_resolution} → Output {output_resolution} · {image.width * image.height / 1_000_000:.2f} MP output"
        else:
            resolution = f"Image {output_resolution} · {image.width * image.height / 1_000_000:.2f} MP"
        file_info = ""
        if path and Path(path).is_file():
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                fmt = Image.registered_extensions().get(Path(path).suffix.lower(), Path(path).suffix.upper().lstrip("."))
                file_info = f" · {fmt} · {size_mb:.2f} MB"
            except OSError:
                pass
        parts = [resolution + file_info]
        if elapsed is not None:
            parts.append(f"Time {self._format_duration(elapsed)}")
        if original is not None:
            change_pct = self.estimate_pixel_change(original, image)
            if change_pct is not None:
                parts.append(f"Mean pixel change ≈ {change_pct:.1f}%")
            else:
                parts.append("Change vs source: unavailable")
        self.preview_info_label.setText("  ·  ".join(parts))

    def apply_std_prompt_preset(self, name: str):
        """Apply one curated Qwen prompt template without stacking conflicting templates."""
        self._active_std_prompt_preset = name if name in PROMPT_PRESETS else "Custom instruction (manual)"
        self.std_prompt.setPlainText(PROMPT_PRESETS.get(name, ""))

    def _std_prompt_text_edited(self):
        """Mark hand-edited template text as custom so the selector never lies."""
        current_name = getattr(self, "_active_std_prompt_preset", "Custom instruction (manual)")
        expected = str(PROMPT_PRESETS.get(current_name, "")).strip()
        actual = self.std_prompt.toPlainText().strip()
        if current_name in PROMPT_PRESETS and actual != expected:
            self._active_std_prompt_preset = "Custom instruction (manual)"
            combo = getattr(self, "std_prompt_preset", None)
            if combo is not None and combo.findText("Custom instruction (manual)") >= 0:
                combo.blockSignals(True)
                combo.setCurrentText("Custom instruction (manual)")
                combo.blockSignals(False)
        self.schedule_save()

    def refresh_std_prompt_presets(self, workflow: str, initial: bool = False):
        """Keep prompt presets workflow-specific and ensure selector/text always agree."""
        combo = getattr(self, "std_prompt_preset", None)
        editor = getattr(self, "std_prompt", None)
        if combo is None:
            return
        workflow = workflow if workflow in WORKFLOW_PROMPT_PRESETS else "enhance"
        allowed = list(WORKFLOW_PROMPT_PRESETS[workflow])
        square_default = (
            "Square 1:1 · Enhance + Outpaint"
            if bool(getattr(getattr(self, "btn_square_enhance", None), "isChecked", lambda: False)())
            else "Square 1:1 · Outpaint only (No Enhance)"
        )
        default_by_workflow = {
            "enhance": "Faithful enhance (default)",
            "lora_closeup_1024": "Close-up · target detail cleanup",
            "square_1_1": square_default,
        }
        default_name = default_by_workflow[workflow]
        current_name = str(getattr(self, "_active_std_prompt_preset", "Custom instruction (manual)"))
        current_text = editor.toPlainText().strip() if editor is not None else ""
        current_template = str(PROMPT_PRESETS.get(current_name, "")).strip()
        exact_managed_template = current_name in PROMPT_PRESETS and current_text == current_template
        is_custom_text = current_name == "Custom instruction (manual)" and bool(current_text)

        combo.blockSignals(True)
        combo.clear()
        combo.addItems(allowed)
        apply_prompt = False
        if initial:
            saved_name = str(self.settings.get("std_prompt_preset_name", ""))
            saved_template_matches = (
                saved_name in PROMPT_PRESETS
                and saved_name in allowed
                and current_text == str(PROMPT_PRESETS.get(saved_name, "")).strip()
            )
            # If an older settings file saved the prompt text but not its preset name,
            # infer the exact matching preset before treating the text as user-authored.
            matched_allowed = next((
                name for name in allowed
                if name != "Custom instruction (manual)"
                and current_text == str(PROMPT_PRESETS.get(name, "")).strip()
            ), None)
            if saved_template_matches:
                combo.setCurrentText(saved_name)
            elif matched_allowed:
                combo.setCurrentText(matched_allowed)
            elif is_custom_text:
                combo.setCurrentText("Custom instruction (manual)")
            elif current_text:
                # Existing text from a preset not available in the active workflow is not
                # a valid preset match; keep the text as a clearly named manual instruction.
                combo.setCurrentText("Custom instruction (manual)")
                self._active_std_prompt_preset = "Custom instruction (manual)"
            else:
                combo.setCurrentText(default_name)
                apply_prompt = True
        elif is_custom_text:
            # Preserve an actual user-authored instruction across workflow switches.
            combo.setCurrentText("Custom instruction (manual)")
        elif current_name in allowed and not (
            exact_managed_template
            and current_name != default_name
            and (
                current_name in default_by_workflow.values()
                or current_name in {"Square 1:1 · Outpaint only (No Enhance)", "Square 1:1 · Enhance + Outpaint"}
            )
        ):
            combo.setCurrentText(current_name)
        else:
            # The prior managed preset is inappropriate for the new workflow (or was a
            # workflow default); switch both the label and prompt to the correct default.
            combo.setCurrentText(default_name)
            apply_prompt = True
        selected_name = combo.currentText()
        combo.blockSignals(False)
        self._active_std_prompt_preset = selected_name
        if editor is not None and apply_prompt:
            editor.setPlainText(str(PROMPT_PRESETS.get(selected_name, "")))

    def _selected_closeup_grounding_parts(self) -> tuple[Optional[Path], Optional[Path], str]:
        """Return the current Model Catalog encoder/projector paths and a user-facing compatibility note."""
        model_folder = Path(self.model_path_edit.text().strip() or str(MODEL_DIR)).expanduser()
        try:
            parts = self.engine._find_model_parts(model_folder)
        except Exception as exc:
            return None, None, str(exc).splitlines()[0]
        model = parts.get("llm")
        mmproj = parts.get("llm_vision")
        if model is None or not Path(model).is_file():
            return None, None, "The Model Catalog text encoder is not available on disk. Choose/download the encoder in Model Catalog first."
        model_path = Path(model).resolve()
        mmproj_resolved = Path(mmproj).resolve() if mmproj and Path(mmproj).is_file() else None
        if model_path.suffix.lower() != ".gguf" or _text_encoder_embeds_vision(model_path) or mmproj_resolved is None:
            gguf_m, gguf_v = resolve_closeup_grounding_gguf_pair(model_path, mmproj_resolved)
            if gguf_m is not None and gguf_v is not None:
                return gguf_m, gguf_v, ""
            return model_path, None, ""
        return model_path, mmproj_resolved, ""

    def refresh_qwen_grounding_status(self):
        runtime_ready = llama_grounding_runtime_present()
        button = getattr(self, "btn_qwen_grounding_setup", None)
        status = getattr(self, "qwen_grounding_status", None)
        if button is not None:
            button.setText("✓ Locator Runtime Ready · Verify / Repair" if runtime_ready else "⚙ Prepare Locator Runtime · ~45 MB")
        model, mmproj, problem = self._selected_closeup_grounding_parts() if hasattr(self, "model_path_edit") else (None, None, "")
        if status is not None:
            if problem:
                status.setText("Model Catalog selection: " + problem)
            elif model is not None and mmproj is not None and model.suffix.lower() == ".gguf":
                if runtime_ready:
                    status.setText(f"Ready: Close-up reuses Qwen3-VL GGUF encoder '{model.name}' + '{mmproj.name}' on GPU via Locator Runtime.")
                else:
                    status.setText(f"Selected GGUF grounding pair: {model.name} + {mmproj.name}. Click Prepare Locator Runtime (~45 MB) to enable GPU Qwen3-VL grounding.")
            elif model is not None and (model.suffix.lower() != ".gguf" or _text_encoder_embeds_vision(model)):
                status.setText(
                    f"Ready: Close-up works with non-GGUF / ConvRot encoder '{model.name}' "
                    "(embedded vision; no mmproj required)."
                )
            elif model is not None:
                status.setText(
                    f"Ready: Close-up works with '{model.name}' without mmproj "
                    "(optional mmproj in Model Catalog enables Qwen3-VL GGUF VLM grounding)."
                )

    def download_qwen_grounding_bundle(self):
        """Download/extract only llama.cpp runtime; model weights always come from Model Catalog."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Image workflow is active", "Finish or stop the current image job before preparing the grounding runtime.")
            return
        current = getattr(self, "qwen_grounding_worker", None)
        if current is not None and current.isRunning():
            QMessageBox.information(self, "Download in progress", "The local grounding runtime is already downloading.")
            return
        model, mmproj, problem = self._selected_closeup_grounding_parts()
        if problem:
            QMessageBox.warning(self, "Model Catalog encoder required", problem)
            self.refresh_qwen_grounding_status()
            return
        if not llama_grounding_runtime_present():
            message = (
                f"Close-up will reuse the selected Model Catalog text encoder ({model.name}) and its matching vision projector. "
                "This download contains only the small llama.cpp Windows CPU runtime (about 45 MB), not another text encoder or vision model.\n\n"
                "The selected encoder/projector files must be GGUF and compatible with llama.cpp. Continue?"
            )
            answer = QMessageBox.question(self, "Prepare local grounding runtime", message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.btn_qwen_grounding_setup.setEnabled(False)
        self.qwen_grounding_progress.setValue(0)
        self.qwen_grounding_status.setText("Downloading/verifying the small llama.cpp runtime only; selected Model Catalog weights are not downloaded again…")
        worker = ModelBundleDownloadWorker(DATA_ROOT / "runtime-downloads", GROUNDING_RUNTIME_ASSETS, self)
        worker.manifest["bundle"] = "llama-cpp-grounding-runtime-only-v1"
        self.qwen_grounding_worker = worker
        worker.progress.connect(self._qwen_grounding_progress_changed)
        worker.completed.connect(self._qwen_grounding_download_completed)
        worker.failed.connect(self._qwen_grounding_download_failed)
        worker.cancelled.connect(lambda: self.qwen_grounding_status.setText("Runtime preparation cancelled. Partial files are retained where possible."))
        worker.finished.connect(self._qwen_grounding_download_thread_finished)
        worker.start()

    def _qwen_grounding_progress_changed(self, value: int, message: str):
        if hasattr(self, "qwen_grounding_progress"):
            self.qwen_grounding_progress.setValue(value)
        if hasattr(self, "qwen_grounding_status"):
            self.qwen_grounding_status.setText(message)

    def _qwen_grounding_download_completed(self, path: str):
        try:
            archive = DATA_ROOT / "runtime-downloads" / LLAMA_GROUNDING_ARCHIVE_FILENAME
            if not archive.is_file():
                raise FileNotFoundError(f"Local llama.cpp runtime archive is missing: {archive}")
            _extract_llama_grounding_archive_safe(archive, LLAMA_GROUNDING_RUNTIME_DIR)
            if not llama_grounding_runtime_present():
                raise RuntimeError("llama-server.exe was not found after extracting the official runtime archive.")
            self.qwen_grounding_progress.setValue(100)
            self.qwen_grounding_status.setText("Local chat runtime is ready. Close-up will use the text encoder/projector already selected in Model Catalog.")
            self.log_message("Prepared the local llama.cpp CPU runtime only. No separate Qwen3-VL model weights were downloaded.")
        except Exception as exc:
            self.qwen_grounding_status.setText("Runtime setup failed: " + str(exc))
            QMessageBox.critical(self, "Locator runtime setup failed", str(exc))
        finally:
            self.refresh_qwen_grounding_status()

    def _qwen_grounding_download_failed(self, message: str):
        self.qwen_grounding_status.setText("Runtime download / verification failed: " + message.splitlines()[0])
        QMessageBox.critical(self, "Locator runtime download failed", message)

    def _qwen_grounding_download_thread_finished(self):
        if hasattr(self, "btn_qwen_grounding_setup"):
            self.btn_qwen_grounding_setup.setEnabled(True)
        self.refresh_qwen_grounding_status()
        if self.qwen_grounding_worker is not None:
            self.qwen_grounding_worker.wait()

    def update_closeup_controls(self):
        auto = self.btn_cl_auto_zoom.isChecked()
        enhance = self.btn_cl_enhance.isChecked()
        if not auto:
            auto_text = "✋  Auto Zoom: OFF (Manual Factor / Focus Point)"
        elif not enhance:
            auto_text = "◎  Auto Zoom: ON (Qwen3-VL semantic crop only)"
        else:
            auto_text = "◎  Auto Zoom: ON (Qwen3-VL grounding + faithful Enhance)"
        self.btn_cl_auto_zoom.setText(auto_text)
        # All Close-up modes now form a real square crop before any Qwen inference.
        # Keep zoom adjustable even in Auto mode; Auto selects the anchor, not the magnification.
        self.closeup_zoom.setEnabled(True)
        if hasattr(self, "closeup_zoom_label"):
            self.closeup_zoom_label.setVisible(not auto)
        self.closeup_zoom.setVisible(not auto)
        self.closeup_zoom.setToolTip(
            "Controls the square crop size after Qwen3-VL returns a semantic bounding box. The target description is passed as free-form language; no keyword-to-body-part map is used. "
            "Right-click the exact item in the preview to override automatic localization. The selected Model Catalog encoder/projector run locally via llama.cpp bound to 127.0.0.1. No second model weights are downloaded; no hosted inference API or external workflow service is used. "
            "Changing input images clears stale focus points."
        )
        self.btn_cl_enhance.setText("✨ Enhance: ON (Faithful Qwen Detail Recovery)" if enhance else "✨ Enhance: OFF (Pure Crop / Resample)")
        self.btn_caption.setText("📝 Auto .txt caption: ON" if self.btn_caption.isChecked() else "📝 Auto .txt caption: OFF")
        self.btn_square_enhance.setText("✨ Enhance: ON" if self.btn_square_enhance.isChecked() else "✨ Enhance: OFF")
        if getattr(self, "workflow", None) == "square_1_1" and hasattr(self, "std_prompt_preset"):
            self.refresh_std_prompt_presets("square_1_1")
        self.schedule_save()

    def set_closeup_side(self, side: int, save: bool = True):
        """Set one of the fixed Close-up output sizes and update its radio-like buttons."""
        side = normalize_fixed_square_side(side)
        self.closeup_side = side
        for value, button in getattr(self, "closeup_size_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(value == side)
            button.blockSignals(False)
        if save and hasattr(self, "_autosave_timer"):
            self.schedule_save()

    def set_square_side(self, side: int, save: bool = True):
        """Set one of the fixed Square output sizes, independent of Close-up."""
        side = normalize_fixed_square_side(side)
        self.square_side = side
        for value, button in getattr(self, "square_size_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(value == side)
            button.blockSignals(False)
        if save and hasattr(self, "_autosave_timer"):
            self.schedule_save()

    def reset_focus(self):
        self.focus_point = None
        self.preview.focus_point = None
        self.focus_label.setText("Focus point: automatic")
        self.update_closeup_controls()

    def on_focus_selected(self, fx: float, fy: float):
        self.focus_point = (fx, fy)
        self.focus_label.setText(f"Focus point: custom ({fx:.3f}, {fy:.3f})")
        if self.workflow == "lora_closeup_1024":
            self.log_message("Preview focus point selected; this exact point takes priority for the next Close-up, including when Auto Zoom is ON.")
        elif self.workflow != "lora_closeup_1024":
            self.log_message("Preview focus point selected for Close-up; it is saved until the input image changes or focus is reset.")
        self.update_closeup_controls()
        self.schedule_save()

    def _set_steps_from_preset(self, text: str, save: bool = True):
        steps = QUALITY_STEPS.get(str(text), 8)
        if hasattr(self, "steps_spin"):
            self.steps_spin.blockSignals(True)
            self.steps_spin.setValue(steps)
            self.steps_spin.blockSignals(False)
        if save:
            self.schedule_save()

    def apply_gen_size_preset(self, text: str):
        dims = {
            "1024 × 1024 (1:1 Square)": (1024, 1024),
            "2048 × 2048 (1:1 High-Res)": (2048, 2048),
            "1344 × 768 (16:9 Landscape)": (1344, 768),
            "768 × 1344 (9:16 Portrait)": (768, 1344),
            "1536 × 1024 (3:2 Photo)": (1536, 1024),
            "1024 × 1536 (2:3 Portrait)": (1024, 1536),
        }
        if text in dims:
            self.gen_w.setValue(dims[text][0]); self.gen_h.setValue(dims[text][1])

    def _commit_gen_dimensions(self, edited_axis: str = ""):
        """Normalize custom generation dimensions and keep the preset label truthful."""
        for spin in (self.gen_w, self.gen_h):
            self._snap_qwen_dimension(spin)
        presets = {
            "1024 × 1024 (1:1 Square)": (1024, 1024),
            "2048 × 2048 (1:1 High-Res)": (2048, 2048),
            "1344 × 768 (16:9 Landscape)": (1344, 768),
            "768 × 1344 (9:16 Portrait)": (768, 1344),
            "1536 × 1024 (3:2 Photo)": (1536, 1024),
            "1024 × 1536 (2:3 Portrait)": (1024, 1536),
        }
        dimensions = (self.gen_w.value(), self.gen_h.value())
        matched = next((name for name, size in presets.items() if size == dimensions), "Custom dimensions")
        self.gen_size.blockSignals(True)
        self.gen_size.setCurrentText(matched)
        self.gen_size.blockSignals(False)
        self.schedule_save()

    def apply_edit_size_preset(self, text: str):
        dims = {"1024 × 1024": (1024, 1024), "2048 × 2048": (2048, 2048), "1344 × 768": (1344, 768), "768 × 1344": (768, 1344)}
        if text in dims:
            self.edit_w.setValue(dims[text][0]); self.edit_h.setValue(dims[text][1])
        custom = str(text) == "Custom dimensions"
        # Width / Height are editable only for custom dimensions. Fixed presets
        # display their actual values without allowing the controls to disagree.
        for widget_name in ("edit_w", "edit_h", "edit_w_label", "edit_h_label"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(custom)
        if hasattr(self, "_autosave_timer"):
            self.schedule_save()

    @staticmethod
    def _snap_qwen_dimension(spin: QSpinBox):
        """Qwen Image dimensions should be multiples of 32; normalize manual entries on commit."""
        value = int(spin.value())
        snapped = max(64, min(MAX_OUTPUT_SIDE, int(round(value / 32.0)) * 32))
        if snapped != value:
            spin.setValue(snapped)

    def set_std_output_scale(self, scale: int, save: bool = True):
        """Set Qwen Output Size scale; this is unrelated to Real-ESRGAN's scale."""
        self.std_output_scale = max(1, min(4, int(scale)))
        for value, button in getattr(self, "std_scale_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(value == self.std_output_scale)
            button.blockSignals(False)
        if save:
            self.schedule_save()

    def apply_std_size_preset(self, text: str):
        is_custom = str(text) == "Custom dimensions"
        if hasattr(self, "std_custom_dims_row"):
            self.std_custom_dims_row.setVisible(is_custom)
        if hasattr(self, "std_scale_row"):
            self.std_scale_row.setVisible(not is_custom)
        if hasattr(self, "std_output_scale") and hasattr(self, "std_scale_buttons"):
            self.set_std_output_scale(self.std_output_scale, save=False)
        self.schedule_save()

    def apply_generate_prompt_preset(self, name: str):
        """Load a Qwen-native natural-language template into Image Generate."""
        if name not in GEN_PROMPT_PRESETS:
            return
        self.gen_prompt_edit.setPlainText(GEN_PROMPT_PRESETS.get(name, ""))
        self.schedule_save()

    def apply_edit_preset(self, name: str):
        # Every preset, including the empty Custom preset, replaces the old prompt so
        # stale instructions cannot survive a preset switch.
        if name not in EDIT_PRESETS:
            return
        self.edit_prompt_edit.setPlainText(EDIT_PRESETS[name])
        self.schedule_save()

    def _sync_gen_prompt_preset(self):
        """Keep Generate's template selector truthful when the prompt is edited by hand."""
        if not hasattr(self, "gen_prompt_preset") or not hasattr(self, "gen_prompt_edit"):
            return
        actual = self.gen_prompt_edit.toPlainText().strip()
        matched = next((n for n, template in GEN_PROMPT_PRESETS.items() if actual == str(template).strip()), None)
        desired = matched or "Custom scene (write your prompt)"
        combo = self.gen_prompt_preset
        if combo.currentText() != desired and combo.findText(desired) >= 0:
            combo.blockSignals(True)
            combo.setCurrentText(desired)
            combo.blockSignals(False)
        self.schedule_save()

    def _sync_edit_prompt_preset(self):
        """Keep Edit's template selector truthful when the prompt is edited by hand."""
        if not hasattr(self, "edit_preset_combo") or not hasattr(self, "edit_prompt_edit"):
            return
        actual = self.edit_prompt_edit.toPlainText().strip()
        matched = next((n for n, template in EDIT_PRESETS.items() if actual == str(template).strip()), None)
        desired = matched or "Custom instruction (free edit)"
        combo = self.edit_preset_combo
        if combo.currentText() != desired and combo.findText(desired) >= 0:
            combo.blockSignals(True)
            combo.setCurrentText(desired)
            combo.blockSignals(False)
        self.schedule_save()

    def choose_input_dir(self):
        initial = self.input_dir_edit.text().strip() or str(INPUT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Choose input image folder", initial)
        if folder:
            self.input_dir_edit.setText(folder)
            self.add_paths([folder])
            self.schedule_save()

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir_edit.text() or str(DATA_ROOT))
        if folder:
            self.output_dir_edit.setText(folder)
            self.schedule_save()

    def add_images(self):
        initial = self.input_dir_edit.text().strip() if hasattr(self, "input_dir_edit") else str(INPUT_DIR)
        paths, _ = QFileDialog.getOpenFileNames(self, "Add image files", initial or str(INPUT_DIR),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if paths:
            self.add_paths(paths)

    def add_paths(self, paths):
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                for child in p.rglob("*"):
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                        try:
                            validate_image_file(child)
                            self._add_queue_item(str(child))
                        except ValidationError as exc:
                            self.log_message(f"Skipped invalid image {child.name}: {exc}")
            elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    validate_image_file(p)
                    self._add_queue_item(str(p))
                except ValidationError as exc:
                    self.log_message(f"Skipped invalid image {p.name}: {exc}")
        self.refresh_queue_table()

    def _load_queue_metrics_cache(self) -> dict[str, dict[str, Any]]:
        cache_file = CACHE_DIR / "queue_metrics.json"
        if not cache_file.is_file():
            return {}
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_queue_metrics_cache(self, path_key: str, elapsed_seconds: Optional[float],
                                  change_pct: Optional[float], workflow: str,
                                  source_path: Optional[str] = None) -> None:
        cache_file = CACHE_DIR / "queue_metrics.json"
        try:
            data = self._load_queue_metrics_cache()
            data[str(Path(path_key).resolve())] = {
                "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
                "change_pct": float(change_pct) if change_pct is not None else None,
                "workflow": str(workflow or ""),
                "source_path": str(source_path) if source_path else None,
            }
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _add_queue_item(self, path: str, status: str = "Queued", is_output: bool = False):
        path = str(Path(path).resolve())
        existing = next((item for item in self.queue if item["path"] == path), None)
        if existing is not None:
            # A newly produced image is retained as completed history. If the user
            # explicitly adds that old output as an input later, make it runnable.
            if is_output:
                existing["status"] = status
                existing["is_output"] = True
            elif status == "Queued" and existing.get("is_output", False):
                existing["status"] = "Queued"
                existing["is_output"] = False
                existing["source_path"] = None
            return
        item_entry = {"path": path, "status": status, "workflow": self.workflow,
                      "elapsed_seconds": None, "change_pct": None,
                      "source_path": None, "is_output": bool(is_output)}
        cached = self._load_queue_metrics_cache().get(path)
        if isinstance(cached, dict):
            if cached.get("elapsed_seconds") is not None:
                item_entry["elapsed_seconds"] = float(cached["elapsed_seconds"])
            if cached.get("change_pct") is not None:
                item_entry["change_pct"] = float(cached["change_pct"])
            if cached.get("workflow"):
                item_entry["workflow"] = str(cached["workflow"])
            if cached.get("source_path"):
                item_entry["source_path"] = str(cached["source_path"])
        self.queue.append(item_entry)

    def _populate_queue_from_workspace(self):
        # Populate runnable inputs from the remembered user folder. Old outputs remain
        # visible as completed history for preview and before/after comparison.
        input_root = Path(str(self.settings.get("input_dir", INPUT_DIR))).expanduser()
        if hasattr(self, "input_dir_edit") and self.input_dir_edit.text().strip():
            input_root = Path(self.input_dir_edit.text().strip()).expanduser()
        if not input_root.is_dir():
            input_root = INPUT_DIR
        try:
            for p in input_root.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    self._add_queue_item(str(p), status="Queued", is_output=False)
        except OSError as exc:
            self.log_message(f"Could not scan input directory {input_root}: {exc}")
        for folder in (GENERATED_DIR, EDITED_DIR, OUTPUT_DIR, CLOSEUPS_DIR, OUTPAINT_DIR):
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    self._add_queue_item(str(p), status="Completed", is_output=True)
        self.refresh_queue_table()

    def refresh_queue_table(self):
        if not hasattr(self, "queue_table"):
            return
        query = self.search_box.text().strip().lower()
        visible = [item for item in self.queue if query in item["path"].lower() or query in item["status"].lower()]
        self.queue_table.setRowCount(len(visible))
        for row, item in enumerate(visible):
            elapsed = item.get("elapsed_seconds")
            if elapsed is None and item.get("status") == "Running" and getattr(self, "_job_started_at", None) is not None:
                elapsed = max(0.0, time.monotonic() - self._job_started_at)
            time_label = f"{float(elapsed):.1f}s" if elapsed is not None else "0.0s"
            delta = item.get("change_pct")
            delta_label = f"{float(delta):.1f}%" if delta is not None else "0.0%"
            workflow_labels = {
                "image_generate": "Generate", "image_edit": "Edit", "enhance": "Enhance",
                "lora_closeup_1024": "Close-up", "square_1_1": "Square", "upscale_only": "Upscale",
            }
            workflow_name = str(item.get("workflow", ""))
            values = [Path(item["path"]).name, workflow_labels.get(workflow_name, workflow_name), item["status"], time_label, delta_label, item["path"]]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setData(Qt.ItemDataRole.UserRole, item["path"])
                cell.setToolTip(f"{value}\n{item['path']}\nWorkflow: {workflow_name}")
                if col == 0:
                    cell.setStatusTip(item["path"])
                self.queue_table.setItem(row, col, cell)

    def _selected_paths(self) -> list[str]:
        rows = sorted({index.row() for index in self.queue_table.selectionModel().selectedRows()})
        return [self.queue_table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in rows if self.queue_table.item(row, 0)]

    def select_queue_row(self, row: int, column: int):
        table_item = self.queue_table.item(row, 0)
        if not table_item:
            return
        path = table_item.data(Qt.ItemDataRole.UserRole)
        info = next((queue_item for queue_item in self.queue if queue_item["path"] == path), None)
        try:
            with Image.open(path) as image_file:
                selected_image = ImageOps.exif_transpose(image_file).copy()
            effective_input = (
                str(Path(info["source_path"]).resolve())
                if info and info.get("is_output") and info.get("source_path") and Path(str(info["source_path"])).is_file()
                else path
            )
            if effective_input != self.current_input:
                if path != self.current_input:
                    self.focus_point = None
                    if hasattr(self, "preview"):
                        self.preview.focus_point = None
                    if hasattr(self, "focus_label"):
                        self.focus_label.setText("Focus point: automatic")
            self.current_input = effective_input
            if info and info.get("is_output"):
                self.current_enhanced = selected_image
                self.current_output = path
                source_path = info.get("source_path")
                source_image = None
                if source_path and Path(source_path).is_file():
                    try:
                        with Image.open(source_path) as source_file:
                            source_image = ImageOps.exif_transpose(source_file).copy()
                    except Exception:
                        source_image = None
                self.current_original = source_image
                result_workflow = str(info.get("workflow") or self.workflow)
                self._apply_preview_presentation(result_workflow)
                self.preview.set_live_caption("LIVE · RESULT")
                source_image, result_image = live_preview_images_for_workflow(result_workflow, self.current_original, self.current_enhanced)
                self.preview.closeup_crop_box = self._extract_closeup_crop_box(path) if result_workflow == "lora_closeup_1024" else None
                self.preview.set_images(source_image, result_image)
                preferred_view = "Enhanced" if result_workflow == "image_generate" else str(getattr(self, "_preferred_preview_mode", "Side by side"))
                self.preview.set_mode(preferred_view)
                self.preview_mode.blockSignals(True)
                self.preview_mode.setCurrentText(preferred_view)
                self.preview_mode.blockSignals(False)
                self.update_preview_info(self.current_enhanced, path, self.current_original, info.get("elapsed_seconds"))
            else:
                self.current_original = selected_image
                self.current_enhanced = None
                self.current_output = None
                self._apply_preview_presentation(self.workflow)
                source_image, _empty_live = live_preview_images_for_workflow(self.workflow, self.current_original, None)
                self.preview.set_live_caption("LIVE · PREVIEW")
                self.preview.closeup_crop_box = None
                self.preview.set_images(source_image, None)
                self.preview.set_mode("Enhanced" if self.workflow == "image_generate" else str(getattr(self, "_preferred_preview_mode", "Side by side")))
                self.update_preview_info(self.current_original, path)
        except Exception as exc:
            self.log_message(f"Could not preview {path}: {exc}")

    def open_queue_row(self, row: int, column: int):
        item = self.queue_table.item(row, 0)
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.ItemDataRole.UserRole)))

    def queue_context_menu(self, point):
        menu = QMenu(self)
        open_action = menu.addAction("Open image")
        remove_action = menu.addAction("Remove from queue")
        selected = menu.exec(self.queue_table.viewport().mapToGlobal(point))
        if selected == open_action:
            paths = self._selected_paths()
            if paths:
                QDesktopServices.openUrl(QUrl.fromLocalFile(paths[0]))
        elif selected == remove_action:
            paths = set(self._selected_paths())
            self.queue = [item for item in self.queue if item["path"] not in paths]
            self.refresh_queue_table()

    def _upscale_custom_from_ui(self) -> dict[str, Any]:
        """Return Real-ESRGAN options (native model scale) and the explicit optional post-process toggle."""
        return {
            "upscale_enabled": False,
            "auto_realesrgan_enabled": bool(self.btn_auto_realesrgan.isChecked()) if hasattr(self, "btn_auto_realesrgan") else bool(self.settings.get("auto_realesrgan_enabled", False)),
            "upscale_model": str(self.upscale_model_combo.currentData() or "realesrgan-x4plus"),
            "upscale_tile": int(self.upscale_tile_combo.currentData() if self.upscale_tile_combo.currentData() is not None else 0),
            "upscale_tta": bool(self.upscale_tta_check.isChecked()),
            "ultimate_qwen_denoise_strength": float(self.ultimate_qwen_denoise_spin.value()) if hasattr(self, "ultimate_qwen_denoise_spin") else float(self.settings.get("ultimate_qwen_denoise_strength", 0.20)),
            "ultimate_qwen_tile_coverage": (float(self.ultimate_qwen_tile_coverage_spin.value()) / 100.0) if hasattr(self, "ultimate_qwen_tile_coverage_spin") else float(self.settings.get("ultimate_qwen_tile_coverage", 50)) / 100.0,
            "ultimate_qwen_passes": self.ultimate_qwen_passes_spin.value() if hasattr(self, "ultimate_qwen_passes_spin") else int(self.settings.get("ultimate_qwen_passes", 1)),
            "ultimate_qwen_steps": self.ultimate_qwen_steps_spin.value() if hasattr(self, "ultimate_qwen_steps_spin") else int(self.settings.get("ultimate_qwen_steps", 8)),
            "ultimate_qwen_scale": int(self.ultimate_qwen_scale_combo.currentData()) if hasattr(self, "ultimate_qwen_scale_combo") and self.ultimate_qwen_scale_combo.currentData() is not None else int(self.settings.get("ultimate_qwen_scale", 2)),
            "watermark_removal_enabled": bool(self.btn_watermark_toggle.isChecked()) if hasattr(self, "btn_watermark_toggle") else bool(self.settings.get("watermark_removal_enabled", False)),
        }

    def upscale_selected_image(self):
        """Start a standalone AI upscaling job without loading the Qwen diffusion model."""
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Wait for the current job or stop it before starting another upscale.")
            return
        selected = self._selected_paths() if hasattr(self, "queue_table") else []
        input_path = selected[0] if selected else (self.current_input or "")
        if not input_path:
            input_path, _ = QFileDialog.getOpenFileName(
                self, "Choose image to upscale", str(INPUT_DIR),
                "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
            )
        if not input_path:
            return
        try:
            custom = self._upscale_custom_from_ui()
            custom["upscale_enabled"] = True
            job = Job(
                "upscale_only", input_path=input_path,
                output_dir=self.output_dir_edit.text().strip(),
                custom=custom,
            )
            self.start_job(job)
        except Exception as exc:
            QMessageBox.warning(self, "AI Upscale", str(exc))

    def _job_from_ui(self, workflow: Optional[str] = None, input_path: Optional[str] = None) -> Job:
        workflow = workflow or self.workflow
        input_path = input_path or self.current_input
        upscale = self._upscale_custom_from_ui()
        if workflow == "upscale_only":
            if not input_path:
                raise ValueError("Select an input image first.")
            return Job(workflow, input_path=input_path, output_dir=self.output_dir_edit.text().strip(), custom=upscale)
        if workflow == "image_generate":
            return Job(workflow, prompt=self.gen_prompt_edit.toPlainText().strip(), output_dir=self.output_dir_edit.text().strip(),
                width=self.gen_w.value(), height=self.gen_h.value(), quality=self.gen_quality.currentText(),
                seed=self.gen_seed.value(), seed_mode=self.gen_seed_mode.currentText(), batch_count=self.gen_count.value(),
                custom={**upscale, "num_inference_steps": self.steps_spin.value(), "qwen_detail_profile": self._selected_qwen_detail_profile() if hasattr(self, "qwen_detail_profile_buttons") else "balanced",
                "ultimate_qwen_upscale_enabled": self.ultimate_qwen_upscale_check.isChecked() if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False))})
        if workflow == "image_edit":
            if not input_path:
                raise ValueError("Select an input image in the queue first.")
            prompt = build_edit_prompt(self.edit_prompt_edit.toPlainText(), self.btn_edit_lock_id.isChecked(), self.btn_edit_lock_comp.isChecked())
            if self.edit_size.currentText() in {"Original Size", "Keep Original Size (Auto)"}:
                with Image.open(input_path) as im:
                    width, height = ImageOps.exif_transpose(im).size
            else:
                width, height = self.edit_w.value(), self.edit_h.value()
            return Job(workflow, input_path=input_path, prompt=prompt, output_dir=self.output_dir_edit.text().strip(),
                width=width, height=height, quality=self.edit_quality.currentText(), seed=self.edit_seed.value(),
                seed_mode=self.edit_seed_mode.currentText(), output_format=self.edit_fmt_combo.currentText(),
                custom={**upscale, "num_inference_steps": self.steps_spin.value(), "qwen_detail_profile": self._selected_qwen_detail_profile() if hasattr(self, "qwen_detail_profile_buttons") else "balanced",
                "ultimate_qwen_upscale_enabled": self.ultimate_qwen_upscale_check.isChecked() if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False))})
        if workflow == "lora_closeup_1024":
            if not input_path:
                raise ValueError("Select an input image in the queue first.")
            side = normalize_fixed_square_side(self.closeup_side)
            self.closeup_side = side
            prompt = self.std_prompt.toPlainText().strip()
            return Job(workflow, input_path=input_path, prompt=prompt, output_dir=self.output_dir_edit.text().strip(),
                width=side, height=side, quality=self.std_quality.currentText(), seed=-1,
                target=self.lora_target.text().strip() or "the main subject detail",
                auto_zoom=self.btn_cl_auto_zoom.isChecked(), zoom_factor=self.closeup_zoom.value(),
                focus_point=self.focus_point, enhance=self.btn_cl_enhance.isChecked(), side=side,
                save_caption=self.btn_caption.isChecked(), trigger_word=self.trigger_edit.text().strip(),
                custom={**upscale, "identity_lock": self._identity_lock_enabled(), "num_inference_steps": self.steps_spin.value(), "max_attempts": self.max_attempts.value(), "retry_diff": self.retry_diff.value(), "warn_diff": self.warn_diff.value(), "qwen_detail_profile": self._selected_qwen_detail_profile() if hasattr(self, "qwen_detail_profile_buttons") else "balanced",
                "grounding_profile": normalize_grounding_profile(self.grounding_profile_combo.currentData()) if hasattr(self, "grounding_profile_combo") else normalize_grounding_profile(self.settings.get("grounding_profile", "balanced")),
                "ultimate_qwen_upscale_enabled": self.ultimate_qwen_upscale_check.isChecked() if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False))})
        if workflow == "square_1_1":
            if not input_path:
                raise ValueError("Select an input image in the queue first.")
            side = normalize_fixed_square_side(self.square_side)
            self.square_side = side
            prompt = self.std_prompt.toPlainText().strip()
            return Job(workflow, input_path=input_path, prompt=prompt,
                output_dir=self.output_dir_edit.text().strip(), width=side, height=side,
                quality=self.std_quality.currentText(), side=side, enhance=self.btn_square_enhance.isChecked(),
                custom={**upscale, "identity_lock": self._identity_lock_enabled(), "num_inference_steps": self.steps_spin.value(), "max_attempts": self.max_attempts.value(), "retry_diff": self.retry_diff.value(), "warn_diff": self.warn_diff.value(), "qwen_detail_profile": self._selected_qwen_detail_profile() if hasattr(self, "qwen_detail_profile_buttons") else "balanced",
                "ultimate_qwen_upscale_enabled": self.ultimate_qwen_upscale_check.isChecked() if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False))})
        if not input_path:
            raise ValueError("Select an input image in the queue first.")
        with Image.open(input_path) as im:
            original_size = ImageOps.exif_transpose(im).size
        qwen_size_mode = self.std_size_preset.currentText()
        qwen_scale = int(getattr(self, "std_output_scale", 2))
        ultimate_enabled = bool(self.ultimate_qwen_upscale_check.isChecked()) if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False))
        # The Qwen Enhance scale is stage one. Ultimate applies its own additional
        # scale after this stage, then runs overlapping Qwen detail tiles on that canvas.
        # Do not let Ultimate overwrite std_output_scale in this job.
        width, height = calculate_qwen_output_size(
            original_size, workflow, qwen_size_mode, qwen_scale, self.std_w.value(), self.std_h.value()
        )
        prompt = self.std_prompt.toPlainText().strip()
        return Job(workflow, input_path=input_path, prompt=prompt, output_dir=self.output_dir_edit.text().strip(),
            width=width, height=height, quality=self.std_quality.currentText(), custom={
                **upscale, "identity_lock": self._identity_lock_enabled(),
                "num_inference_steps": self.steps_spin.value(), "max_attempts": self.max_attempts.value(),
                "retry_diff": self.retry_diff.value(), "warn_diff": self.warn_diff.value(),
                "enhance_detail_mode": self.enhance_detail_mode.currentData() if hasattr(self, "enhance_detail_mode") else "fast",
                "qwen_detail_profile": self._selected_qwen_detail_profile() if hasattr(self, "qwen_detail_profile_buttons") else "balanced",
                "ultimate_qwen_upscale_enabled": self.ultimate_qwen_upscale_check.isChecked() if hasattr(self, "ultimate_qwen_upscale_check") else bool(self.settings.get("ultimate_qwen_upscale_enabled", False))})

    def generate_images(self):
        if not self.gen_prompt_edit.toPlainText().strip():
            QMessageBox.warning(self, "Prompt required", "Enter a prompt for image generation.")
            return
        self.start_job(self._job_from_ui("image_generate"))

    def edit_selected(self):
        self.apply_workflow_preset("image_edit")
        try:
            self.start_job(self._job_from_ui("image_edit"))
        except Exception as exc:
            QMessageBox.warning(self, "Image Edit", str(exc))

    def edit_all_in_queue(self):
        self.apply_workflow_preset("image_edit")
        self.run_all_queued(force_workflow="image_edit")

    @staticmethod
    def _extract_closeup_crop_box(path: str | Path) -> Optional[tuple[int, int, int, int]]:
        try:
            with Image.open(path) as im:
                raw_box = im.info.get("crop_box")
            if not raw_box:
                return None
            nums = [int(x) for x in re.findall(r"-?\d+", str(raw_box))]
            if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
                return (nums[0], nums[1], nums[2], nums[3])
        except Exception:
            pass
        return None

    def run_selected(self):
        paths = self._selected_paths()
        resolved_paths: list[str] = []
        for p in paths:
            q_info = next((item for item in self.queue if item["path"] == p), None)
            if q_info and q_info.get("is_output") and q_info.get("source_path") and Path(str(q_info["source_path"])).is_file():
                resolved_paths.append(str(Path(q_info["source_path"]).resolve()))
            else:
                resolved_paths.append(p)
        paths = resolved_paths
        if not paths and self.current_input:
            paths = [self.current_input]
        if not paths and self.workflow != "image_generate":
            QMessageBox.warning(self, "No input selected", "Select an image in the queue first.")
            return
        try:
            job = self._job_from_ui(self.workflow, paths[0] if paths else None)
            self.start_job(job)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start job", str(exc))

    def run_all_queued(self, force_workflow: Optional[str] = None):
        queued = [x["path"] for x in self.queue if x["status"] == "Queued"]
        if not queued:
            QMessageBox.information(self, "Queue empty", "There are no queued images.")
            return
        self._run_queue_sequence(queued, force_workflow or self.workflow)

    def _run_queue_sequence(self, paths: list[str], workflow: str):
        if not paths:
            return
        try:
            job = self._job_from_ui(workflow, paths[0])
            job.custom["queue_remaining"] = paths[1:]
            self.start_job(job)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start queue", str(exc))

    def start_job(self, job: Job):
        job.validate(require_existing_input=(job.workflow != "image_generate"))
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "A job is already running. Stop it before starting another.")
            return
        self._configure_engine_from_ui()
        self.engine.force_stop.clear()
        self.engine.stop_after_current.clear()
        self._stop_after_current = False
        self._job_started_at = time.monotonic()
        self._progress_started_at = self._job_started_at
        self._active_job = job
        if job.workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:
            self._preferred_preview_mode = "Side by side"
        self._apply_preview_presentation(job.workflow)
        if job.workflow == "image_generate":
            self.current_original = None
            self.current_enhanced = None
            self.preview.closeup_crop_box = None
            self.preview.set_images(None, None)
        elif job.input_path and Path(job.input_path).is_file():
            try:
                with Image.open(job.input_path) as source_file:
                    self.current_original = ImageOps.exif_transpose(source_file).copy()
                self.current_input = str(Path(job.input_path).resolve())
                self.current_enhanced = None
                self.preview.closeup_crop_box = None
                self.preview.set_images(self.current_original, None)
                if self.current_original is not None:
                    self.preview.set_live_caption("LIVE · SOURCE WHILE PREPARING")
                    self.preview.set_live_image(self.current_original, preserve_view=False)
                if job.workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:
                    self.preview.set_mode("Side by side")
                    if hasattr(self, "preview_mode"):
                        self.preview_mode.blockSignals(True)
                        self.preview_mode.setCurrentText("Side by side")
                        self.preview_mode.blockSignals(False)
            except Exception as exc:
                self.log_message(f"Could not prepare Before preview: {type(exc).__name__}")
        self.preview.set_live_caption("LIVE · PREVIEW")
        self._active_stage_name = "Starting"
        self._active_refinement_context = None
        self.engine._active_pipeline_context = ""
        self.engine.latest_preview_image = None
        self.engine.latest_preview_revision = int(getattr(self.engine, "latest_preview_revision", 0) or 0) + 1
        self.engine._latest_preview_marker = None
        self._live_preview_revision = self.engine.latest_preview_revision
        self.engine.latest_preview_step = 0
        self.engine.latest_preview_total_steps = 0
        self.engine.latest_preview_pass = 0
        self._first_step_at = None; self._last_step_index = 0; self._last_step_total = 0; self._step_interval_ema = None
        self._last_step_changed_at = None
        self.progress.setRange(0, 100); self.progress.setValue(3); self.progress.setFormat("Preparing · %p%")
        self.progress.show()
        self.progress_label.setText(f"Stage: Preparing {job.workflow}")
        self.progress_detail_label.setText("Waiting for local engine, prompt conditioning and image-generation stages…")
        if job.workflow == "image_generate" and str(getattr(self.engine, "live_preview_method", "tae")) != "none":
            self.preview_info_label.setText("Live preview · waiting for the first sampler step…")
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.btn_edit_selected.setEnabled(False)
        if job.workflow == "upscale_only":
            self.log_message("Starting standalone Real-ESRGAN/Real-ESRNet Vulkan upscale; Qwen diffusion model will not be loaded.")
        else:
            self.log_message(f"Starting local workflow {job.workflow} (Qwen-Image 2.1; no external workflow server process/API)")
        self.worker = Worker(self.engine, job)
        self.worker.progress.connect(self.status_message)
        self.worker.grounding_review_requested.connect(self._handle_grounding_review_request)
        self.worker.completed.connect(lambda paths, j=job: self.job_completed(paths, j))
        self.worker.failed.connect(self.job_failed)
        self.worker.aborted.connect(self.job_aborted)
        self.engine.grounding_review_callback = self.worker.request_grounding_review
        self.worker.start()
        for item in self.queue:
            if item["path"] == job.input_path:
                item["status"] = "Running"
        self.refresh_queue_table()

    def _grounding_profile_selected(self, _index=None):
        profile = normalize_grounding_profile(self.grounding_profile_combo.currentData())
        self.settings["grounding_profile"] = profile
        self.grounding_profile_combo.setToolTip(grounding_profile_description(profile))
        self.schedule_save()

    def open_diagnostics(self):
        """Run read-only diagnostics in a worker and show the report dialog."""
        from ..ui.diagnostics_dialog import DiagnosticsDialog

        model, mmproj, problem = self._selected_closeup_grounding_parts()
        context = {
            "runtime_dir": LLAMA_GROUNDING_RUNTIME_DIR,
            "encoder_path": str(model) if model else None,
            "projector_path": str(mmproj) if mmproj else None,
            "workspace_dir": str(DATA_ROOT),
            "gpu_telemetry_supported": None,
        }
        if problem:
            # Surface only the compatibility problem, not full paths or prompts.
            context["technical_details"] = problem.splitlines()[0]
        dialog = DiagnosticsDialog(context, self)
        dialog.exec()

    def _handle_grounding_review_request(self, request: dict[str, Any]):
        """Display grounding review only for the currently active, uncancelled job."""
        from ..ui.grounding_review_dialog import GroundingReviewDecision, GroundingReviewDialog, should_accept_review_job

        event = request.get("event")
        try:
            active_job = self._active_job
            if (not should_accept_review_job(
                    str(request.get("job_id", "")), str(getattr(active_job, "job_id", "")),
                    cancelled=self.engine.force_stop.is_set())
                    or active_job is None):
                request["decision"] = GroundingReviewDecision("cancel")
                return
            image = request.get("image")
            result = request.get("result")
            if image is None or result is None:
                request["decision"] = GroundingReviewDecision("cancel")
                return
            dialog = GroundingReviewDialog(
                image, result, self, allow_retry=bool(request.get("allow_retry", True))
            )
            dialog.exec()
            request["decision"] = dialog.decision
        except Exception as exc:
            self.log_message(f"Grounding review dialog could not be displayed: {type(exc).__name__}.")
            request["decision"] = GroundingReviewDecision("cancel")
        finally:
            if event is not None:
                event.set()

    def job_completed(self, paths: list[Path], job: Job):
        elapsed = (time.monotonic() - self._job_started_at) if self._job_started_at is not None else None
        self.progress.setRange(0, 100); self.progress.setValue(100); self.progress.setFormat("Complete · %p%"); self.progress.show()
        self.progress_label.setText(f"Stage: Complete · elapsed {self._format_duration(elapsed)} · outputs saved")
        self.progress_detail_label.setText("All requested image and post-processing stages completed.")
        self._active_stage_name = "Complete"
        self._active_refinement_context = None
        self.btn_run.setEnabled(True); self.btn_run_all.setEnabled(True); self.btn_generate.setEnabled(True); self.btn_edit_selected.setEnabled(True)
        if elapsed is not None:
            self.log_message(f"Workflow {job.workflow} completed in {self._format_duration(elapsed)}.")
        if job.workflow == "upscale_only":
            if hasattr(self, "upscale_tab_status"):
                self.upscale_tab_status.setText("Runtime: installed and ready (portable Vulkan engine; cached locally for offline use).")
        for queue_item in self.queue:
            if job.input_path and queue_item["path"] == str(Path(job.input_path).resolve()):
                queue_item["status"] = "Completed"
        if job.workflow == "image_generate":
            self.current_original = None
        elif job.input_path and Path(job.input_path).is_file():
            try:
                with Image.open(job.input_path) as src_im:
                    self.current_original = ImageOps.exif_transpose(src_im).copy()
            except Exception:
                pass
        for path in paths:
            self.log_message(f"Completed: {path}")
            self._add_queue_item(str(path), status="Completed", is_output=True)
            try:
                with Image.open(path) as im:
                    output = ImageOps.exif_transpose(im).copy()
                self.current_output = str(path)
                self.current_enhanced = output
                image_elapsed = self.engine.output_durations.get(str(path), elapsed)
                if image_elapsed is None and elapsed is not None:
                    image_elapsed = elapsed
                delta_pct = self.estimate_pixel_change(self.current_original, output)
                if delta_pct is None and job.workflow == "image_generate":
                    delta_pct = 100.0
                resolved_output = str(Path(path).resolve())
                resolved_input = str(Path(job.input_path).resolve()) if job.input_path and Path(job.input_path).is_file() else None
                for queue_item in self.queue:
                    if queue_item["path"] == resolved_output:
                        queue_item["status"] = "Completed"
                        queue_item["workflow"] = job.workflow
                        queue_item["elapsed_seconds"] = image_elapsed
                        queue_item["change_pct"] = delta_pct
                        queue_item["source_path"] = resolved_input
                        queue_item["is_output"] = True
                    elif resolved_input and queue_item["path"] == resolved_input:
                        queue_item["status"] = "Completed"
                        queue_item["workflow"] = job.workflow
                        queue_item["elapsed_seconds"] = image_elapsed
                        queue_item["change_pct"] = delta_pct
                self._save_queue_metrics_cache(resolved_output, image_elapsed, delta_pct, job.workflow, resolved_input)
                if resolved_input:
                    self._save_queue_metrics_cache(resolved_input, image_elapsed, delta_pct, job.workflow, resolved_input)
                self._apply_preview_presentation(job.workflow)
                self.preview.set_live_caption("LIVE · RESULT")
                source_image, result_image = live_preview_images_for_workflow(job.workflow, self.current_original, self.current_enhanced)
                self.preview.closeup_crop_box = self._extract_closeup_crop_box(path) if job.workflow == "lora_closeup_1024" else None
                self.preview.set_images(source_image, result_image)
                preferred_view = "Enhanced" if job.workflow == "image_generate" else str(getattr(self, "_preferred_preview_mode", "Side by side"))
                self.preview.set_mode(preferred_view)
                self.preview_mode.blockSignals(True)
                self.preview_mode.setCurrentText(preferred_view)
                self.preview_mode.blockSignals(False)
                self.update_preview_info(output, str(path), self.current_original, image_elapsed)
            except Exception as exc:
                self.log_message(f"Output preview failed: {exc}")
        self.refresh_queue_table()
        remaining = job.custom.get("queue_remaining", [])
        if self._stop_after_current:
            self._stop_after_current = False
            self.engine.stop_after_current.clear()
            remaining = []
            self.log_message("Stop current job honored: the active job finished and no next queued image will start.")
        if remaining and job.workflow != "image_generate":
            try:
                next_job = self._job_from_ui(job.workflow, remaining[0])
                next_job.custom["queue_remaining"] = remaining[1:]
                QTimer.singleShot(50, lambda j=next_job: self.start_job(j))
            except Exception as exc:
                self.log_message(f"Queue stopped: {exc}")
        self.schedule_save()

    def job_failed(self, message: str):
        self._stop_after_current = False; self.engine.stop_after_current.clear()
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setFormat("Failed · %p%"); self.progress.show()
        self.progress_label.setText("Stage: Failed · inspect Activity Log for details")
        self.progress_detail_label.setText("The workflow stopped before completion. The error dialog and Activity Log contain the details.")
        self._active_stage_name = "Failed"
        self._active_refinement_context = None
        self.btn_run.setEnabled(True); self.btn_run_all.setEnabled(True); self.btn_generate.setEnabled(True); self.btn_edit_selected.setEnabled(True)
        self.log_message("LOCAL INFERENCE ERROR: " + message.splitlines()[0])
        QMessageBox.critical(self, "Local inference failed", message)
        for item in self.queue:
            if item["status"] == "Running":
                item["status"] = "Failed"
        self.refresh_queue_table()

    def job_aborted(self):
        self._stop_after_current = False; self.engine.stop_after_current.clear()
        self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setFormat("Cancelled · %p%"); self.progress.show()
        self.progress_label.setText("Stage: Cancelled · inference stopped at a safe checkpoint")
        self.progress_detail_label.setText("Generation was cancelled. The current output may be incomplete.")
        self._active_stage_name = "Cancelled"
        self._active_refinement_context = None
        self.btn_run.setEnabled(True); self.btn_run_all.setEnabled(True); self.btn_generate.setEnabled(True); self.btn_edit_selected.setEnabled(True)
        self.log_message("Job cancelled.")
        for item in self.queue:
            if item["status"] == "Running":
                item["status"] = "Cancelled"
        self.refresh_queue_table()

    def stop_job(self):
        """Cancel the active inference at the engine's next safe checkpoint."""
        self.engine.force_stop.set()
        self._stop_after_current = False
        self.engine.stop_after_current.clear()
        if self.worker and self.worker.isRunning():
            self.log_message("Stop now requested; inference will cancel at the next safe checkpoint.")
        else:
            self.log_message("No active inference to stop.")

    def open_output(self):
        path = self.current_output or self.current_input
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            folder = self.output_dir_edit.text().strip() or str(OUTPUT_DIR)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        self.add_paths(paths)
        event.acceptProposedAction()

    def closeEvent(self, event):
        self.save_settings()
        active_threads = (self.worker and self.worker.isRunning()) or (self.model_worker and self.model_worker.isRunning())
        if active_threads:
            event.ignore()
            if not getattr(self, "_shutdown_requested", False):
                self._shutdown_requested = True
                self.engine.force_stop.set()
                self.statusBar().showMessage("Stopping local inference safely before exit…")
                self._shutdown_timer = QTimer(self)
                self._shutdown_timer.setInterval(250)
                self._shutdown_timer.timeout.connect(self._finish_shutdown_when_idle)
                self._shutdown_timer.start()
            return
        self.engine.unload_model()
        event.accept()

    def _finish_shutdown_when_idle(self):
        if (self.worker and self.worker.isRunning()) or (self.model_worker and self.model_worker.isRunning()):
            return
        self._shutdown_timer.stop()
        self.engine.unload_model()
        self.close()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("P.AI")
    window = App()
    window.show()
    sys.exit(app.exec())


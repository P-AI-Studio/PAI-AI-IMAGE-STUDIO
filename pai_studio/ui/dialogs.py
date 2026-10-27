# -*- coding: utf-8 -*-
"""
Tabbed Model Catalog dialog for Qwen Image 2.1 weights, Real-ESRGAN/Real-ESRNet,
Ultimate refinement settings, and LoRA management.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

try:
    from PySide6.QtCore import QTimer, Qt, QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    class QDialog:  # type: ignore[no-redef]
        pass

from ..constants import (
    MODEL_CATALOG,
    MODEL_CATALOG_BY_ID,
    MODEL_DEFAULTS,
    MODEL_ROLE_LABELS,
    MODEL_ROLE_ORDER,
    MODEL_ROLE_REQUIRED,
    QWEN_LICENSE_URL,
    QWEN_REPO_URL,
    QWEN_VAE_REPO_URL,
    REALESRGAN_CATALOG,
    REALESRGAN_CATALOG_BY_ID,
    REALESRGAN_MODEL_ZOO_URL,
    REALESRGAN_NCNN_URL,
    REALESRGAN_NCNN_URLS,
    _human_gb,
    _realesrgan_model_assets_present,
)
from ..workers import (
    LoRAInfoLookupWorker,
    ModelBundleDownloadWorker,
    RealESRGANRuntimeWorker,
)
from ..validation import (
    ValidationError,
    validate_model_file,
)
from ..workspace import (
    DATA_ROOT,
    _model_role_mismatch,
    _text_encoder_embeds_vision,
    inspect_lora_file,
)

class ModelDownloadDialog(QDialog):
    """Tabbed model and workflow settings for Qwen, the AI upscaler, Ultimate and LoRA."""
    def __init__(self, target_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("P.AI · Model Catalog & Workflow Settings")
        self.setMinimumSize(1040, 720)
        self.resize(1360, 900)
        self.setStyleSheet("""
            QDialog { background: #171b22; color: #e8edf5; }
            QLabel { color: #dce4ef; }
            QGroupBox { border: 1px solid #394452; border-radius: 10px; margin-top: 12px; padding: 10px 10px 8px 10px; font-weight: 650; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #8fc8f5; }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { min-height: 28px; border: 1px solid #3b4655; border-radius: 7px; padding: 2px 8px; background: #202630; color: #f0f4fa; }
            QPushButton, QToolButton { min-height: 30px; border: 1px solid #414d5e; border-radius: 8px; padding: 4px 11px; background: #293240; color: #eef3fa; }
            QPushButton:hover, QToolButton:hover { background: #344255; border-color: #5e8fbd; }
            QPushButton:disabled { color: #7e8998; background: #222832; }
            QCheckBox { spacing: 8px; padding: 3px; }
            QProgressBar { min-height: 15px; border: 1px solid #3c4654; border-radius: 6px; text-align: center; background: #202630; }
            QProgressBar::chunk { border-radius: 5px; background: #3e9de0; }
            QTabWidget::pane { border: 1px solid #354050; border-radius: 9px; }
            QScrollArea { border: none; }
        """)
        self.target_dir = target_dir
        self.worker: Optional[ModelBundleDownloadWorker] = None
        self.upscaler_worker: Optional[RealESRGANRuntimeWorker] = None
        self.applied = False
        self.completed_successfully = False
        self.combos: dict[str, QComboBox] = {}
        self.browse_buttons: dict[str, QPushButton] = {}
        self.detail_labels: dict[str, QLabel] = {}
        self.profile_warnings: list[str] = []
        self._loading_profile = self._read_profile()

        root = QVBoxLayout(self)
        heading = QLabel("Models & Workflow Settings")
        heading.setStyleSheet("font-size: 17pt; font-weight: 800; color: #82c9f5; padding: 3px 0;")
        root.addWidget(heading)
        intro = QLabel("Choose Qwen model components, then configure the selected learned upscaler and tile refinement together in Ultimate. LoRA has its own tab.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QGridLayout()
        form.setColumnStretch(1, 1)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(5)
        form.addWidget(QLabel("Component"), 0, 0)
        form.addWidget(QLabel("Model / variant"), 0, 1)
        form.addWidget(QLabel("Local file"), 0, 2)
        for row, role in enumerate(MODEL_ROLE_ORDER, start=1):
            label = QLabel(MODEL_ROLE_LABELS[role])
            label.setWordWrap(True)
            form.addWidget(label, row, 0)
            combo = QComboBox()
            combo.setMinimumWidth(360)
            self.combos[role] = combo
            if role == "vision_projector":
                combo.addItem(
                    "No projector · not needed for Safetensors/ConvRot (optional, GGUF only)",
                    {"kind": "none", "id": "none", "role": role},
                )
            for meta in [item for item in MODEL_CATALOG if item["role"] == role]:
                file_path = target_dir / meta["filename"]
                status = "✓ present" if file_path.is_file() else "↓ download"
                combo.addItem(f"{meta['title']}  [{status}]", {"kind": "catalog", "id": meta["id"]})
            # Existing files not in the downloadable catalog can be selected explicitly.
            for local_file in self._local_candidates(role):
                combo.addItem(f"Local file · {local_file.name}", {"kind": "local", "path": str(local_file.resolve()), "filename": local_file.name, "role": role, "title": local_file.name})
            saved = self._loading_profile.get(role, {}) if isinstance(self._loading_profile, dict) else {}
            saved_id = saved.get("id") if isinstance(saved, dict) else None
            saved_path = saved.get("path") if isinstance(saved, dict) else None
            selected_index = -1
            if saved_id:
                for i in range(combo.count()):
                    data = combo.itemData(i)
                    if data and data.get("id") == saved_id:
                        selected_index = i
                        break
            if selected_index < 0 and saved_path and Path(saved_path).is_file() and not _model_role_mismatch(role, Path(saved_path).name):
                for i in range(combo.count()):
                    data = combo.itemData(i)
                    if data and data.get("path") and Path(data["path"]).resolve() == Path(saved_path).resolve():
                        selected_index = i
                        break
                if selected_index < 0:
                    combo.addItem(f"Selected local file · {Path(saved_path).name}", {"kind": "local", "path": str(Path(saved_path).resolve()), "filename": Path(saved_path).name, "role": role, "title": Path(saved_path).name})
                    selected_index = combo.count() - 1
            elif selected_index < 0 and saved_path and Path(saved_path).is_file():
                problem = _model_role_mismatch(role, Path(saved_path).name)
                if problem:
                    self.profile_warnings.append(f"{MODEL_ROLE_LABELS[role]}: {problem}")
            if selected_index < 0:
                default_id = MODEL_DEFAULTS.get(role)
                for i in range(combo.count()):
                    data = combo.itemData(i)
                    if data and data.get("id") == default_id:
                        selected_index = i
                        break
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
            combo.currentIndexChanged.connect(lambda _i, r=role: self._refresh_details(r))
            form.addWidget(combo, row, 1)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _checked=False, r=role: self._browse_local(r))
            self.browse_buttons[role] = browse
            form.addWidget(browse, row, 2)
            details = QLabel()
            details.setWordWrap(True)
            details.setStyleSheet("color: #94a6bb; padding: 1px 3px 5px 3px; font-size: 9pt;")
            self.detail_labels[role] = details
            form.addWidget(details, row, 3, 1, 3)
        # Keep model selection separate from optional image-processing settings.
        components_card = QGroupBox("Qwen model components")
        components_layout = QVBoxLayout(components_card)
        components_layout.setContentsMargins(12, 12, 12, 12)
        components_layout.addLayout(form)

        # Settings live on this single modern tabbed page. Workflow cards are moved
        # here once, so no separate Real-ESRGAN popup or duplicate controls are created.
        app_parent = self.parent()
        self.catalog_tabs = QTabWidget()
        self.catalog_tabs.setDocumentMode(True)
        self.catalog_tabs.setMovable(False)
        self.catalog_tabs.setUsesScrollButtons(True)

        def add_settings_tab(title: str):
            content = QWidget()
            page_layout = QVBoxLayout(content)
            page_layout.setContentsMargins(12, 12, 12, 12)
            page_layout.setSpacing(10)
            page_scroll = QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setFrameShape(QFrame.Shape.NoFrame)
            page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page_scroll.setWidget(content)
            self.catalog_tabs.addTab(page_scroll, title)
            return page_layout

        models_page = add_settings_tab("Models")
        models_page.addWidget(components_card)

        # Move the actual live controls from Basic into their settings tabs.
        ultimate_card = getattr(app_parent, "enhance_detail_group", None)
        esrgan_card = getattr(app_parent, "realesrgan_box", None)
        grounding_card = getattr(app_parent, "grounding_runtime_group", None)
        for card in (ultimate_card, esrgan_card, grounding_card):
            if card is not None:
                basic_layout = getattr(app_parent, "basic_tab_layout", None)
                if basic_layout is not None:
                    basic_layout.removeWidget(card)
                card.setParent(None)
        if grounding_card is not None:
            models_page.addWidget(grounding_card)
        models_page.addStretch(1)

        # LoRA settings use the same widgets consumed by the local engine and saved profile.
        self.lora_settings_group = QGroupBox("LoRA adapter")
        lo = QGridLayout(self.lora_settings_group)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setHorizontalSpacing(10)
        lo.setVerticalSpacing(9)
        model_settings_grid = getattr(app_parent, "model_settings_grid", None)
        lora_widgets = (
            getattr(app_parent, "lora_settings_label", None),
            getattr(app_parent, "lora_path_edit", None),
            getattr(app_parent, "btn_browse_lora", None),
            getattr(app_parent, "lora_enabled_check", None),
            getattr(app_parent, "lora_strength_spin", None),
        )
        for widget in lora_widgets:
            if widget is not None:
                if model_settings_grid is not None:
                    model_settings_grid.removeWidget(widget)
                widget.setParent(None)
        lo.addWidget(QLabel("Adapter file"), 0, 0)
        lo.addWidget(app_parent.lora_path_edit, 0, 1, 1, 2)
        lo.addWidget(app_parent.btn_browse_lora, 0, 3)
        lo.addWidget(app_parent.lora_enabled_check, 1, 0, 1, 2)
        lo.addWidget(QLabel("Strength"), 1, 2)
        lo.addWidget(app_parent.lora_strength_spin, 1, 3)
        lo.setColumnStretch(1, 1)
        self.lora_settings_group.setToolTip("Choose a local adapter, inspect its metadata, and adjust its multiplier. Workflow switches stay inside each workflow panel.")

        self.lora_info_card = QGroupBox("LoRA Info")
        info_layout = QVBoxLayout(self.lora_info_card)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(5)
        self.lora_info_summary = QLabel("Choose a LoRA file to inspect its metadata.")
        self.lora_info_summary.setWordWrap(True)
        self.lora_info_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lora_info_summary.setStyleSheet("font-weight: 600; color: #d9e8f7; padding: 2px;")
        info_layout.addWidget(self.lora_info_summary)
        self.lora_trigger_words = QPlainTextEdit()
        self.lora_trigger_words.setReadOnly(True)
        self.lora_trigger_words.setPlaceholderText("Trigger / trained words (when present in metadata)")
        self.lora_trigger_words.setMaximumHeight(58)
        self.lora_trigger_words.setMinimumHeight(42)
        self.lora_trigger_words.setStyleSheet("QPlainTextEdit { background: #202630; border: 1px solid #394452; border-radius: 6px; padding: 4px; }")
        info_layout.addWidget(self.lora_trigger_words)
        example_label = QLabel("Example prompts")
        example_label.setStyleSheet("font-weight: 600; color: #bdcfe3; padding-top: 2px;")
        info_layout.addWidget(example_label)
        self.lora_example_prompts = QPlainTextEdit()
        self.lora_example_prompts.setReadOnly(True)
        self.lora_example_prompts.setPlaceholderText("Example prompts from metadata, when available")
        self.lora_example_prompts.setMinimumHeight(42)
        self.lora_example_prompts.setMaximumHeight(68)
        self.lora_example_prompts.setStyleSheet("QPlainTextEdit { background: #202630; border: 1px solid #394452; border-radius: 6px; padding: 4px; }")
        info_layout.addWidget(self.lora_example_prompts)
        self.lora_description_label = QLabel("Description: not available")
        self.lora_description_label.setWordWrap(True)
        self.lora_description_label.setStyleSheet("color: #9fb0c3; font-size: 9pt;")
        info_layout.addWidget(self.lora_description_label)
        self.lora_info_source = QLabel("Local metadata is checked first; online lookup is optional.")
        self.lora_info_source.setWordWrap(True)
        self.lora_info_source.setStyleSheet("color: #8193a7; font-size: 8pt;")
        info_layout.addWidget(self.lora_info_source)
        info_buttons = QHBoxLayout()
        self.btn_refresh_lora_info = QPushButton("Refresh local info")
        self.btn_refresh_lora_info.clicked.connect(self._refresh_lora_info)
        info_buttons.addWidget(self.btn_refresh_lora_info)
        self.btn_lookup_lora_info = QPushButton("Look up CivitAI info")
        self.btn_lookup_lora_info.setToolTip("Optional: hashes this local file and looks for an exact public CivitAI match. The file itself is not uploaded.")
        self.btn_lookup_lora_info.clicked.connect(self._lookup_lora_online)
        info_buttons.addWidget(self.btn_lookup_lora_info)
        info_buttons.addStretch(1)
        info_layout.addLayout(info_buttons)
        self.lora_online_status = QLabel("")
        self.lora_online_status.setWordWrap(True)
        self.lora_online_status.setStyleSheet("color: #92c8f2; font-size: 8pt;")
        info_layout.addWidget(self.lora_online_status)
        self.lora_civitai_link = QLabel("")
        self.lora_civitai_link.setOpenExternalLinks(True)
        self.lora_civitai_link.setWordWrap(True)
        info_layout.addWidget(self.lora_civitai_link)
        lo.addWidget(self.lora_info_card, 2, 0, 1, 4)
        self._lora_local_info: dict[str, Any] = {}
        self._lora_online_info: Optional[dict[str, Any]] = None
        self._lora_lookup_path = ""
        self.lora_lookup_worker: Optional[LoRAInfoLookupWorker] = None
        app_parent.lora_path_edit.textChanged.connect(self._refresh_lora_info)

        ultimate_page = add_settings_tab("Ultimate")
        if ultimate_card is not None:
            ultimate_page.addWidget(ultimate_card)
        if esrgan_card is not None:
            # The selected learned upscaler is part of this one Ultimate pipeline,
            # not a second independent section or a duplicate post-processing toggle.
            ultimate_page.addWidget(esrgan_card)
        ultimate_note = QLabel(
            "Ultimate combines three stages: Qwen reconstruction, the selected learned NCNN upscaler "
            "(Real-ESRGAN / Real-ESRNet or a curated community model), then full-coverage Qwen tile refinement "
            "with color matching and soft overlap blending. If coverage is below 50%, the grid adds rows/columns "
            "so every output pixel is covered. The same selected model powers the optional standalone upscale action."
        )
        ultimate_note.setWordWrap(True)
        ultimate_note.setStyleSheet("color: #94a6bb; padding: 0 3px;")
        ultimate_page.addWidget(ultimate_note)

        # Model, runtime setup, and Ultimate refinement controls are intentionally in
        # this one tab. Model weights are fetched on first use and checksum-verified.
        self.upscaler_combo = app_parent.upscale_model_combo
        upscaler_group = QGroupBox("Ultimate AI Upscaler Runtime")
        upscaler_layout = QVBoxLayout(upscaler_group)
        upscaler_layout.setContentsMargins(12, 12, 12, 12)
        upscaler_layout.setSpacing(8)
        upscaler_intro = QLabel(
            "Install or verify the selected model's local Vulkan runner and model weights. "
            "The compact installer does not bundle these files; downloads are staged on first use."
        )
        upscaler_intro.setWordWrap(True)
        upscaler_layout.addWidget(upscaler_intro)
        self.upscaler_detail = QLabel()
        self.upscaler_detail.setWordWrap(True)
        self.upscaler_detail.setStyleSheet("color: #94a6bb; padding: 2px 0; font-size: 9pt;")
        upscaler_layout.addWidget(self.upscaler_detail)
        links = QLabel(
            f'<a href="{REALESRGAN_NCNN_URL}">Real-ESRGAN Windows runtime</a> · '
            f'<a href="{REALESRGAN_NCNN_URLS[1]}">Runtime mirror</a> · '
            f'<a href="{REALESRGAN_MODEL_ZOO_URL}">Official model references</a> · '
            '<a href="https://github.com/upscayl/custom-models">Community NCNN model catalogue</a>'
        )
        links.setOpenExternalLinks(True)
        links.setWordWrap(True)
        upscaler_layout.addWidget(links)
        upscaler_buttons = QHBoxLayout()
        self.btn_use_upscaler = QPushButton("Save selected model")
        self.btn_use_upscaler.clicked.connect(self.apply_upscaler_choice)
        upscaler_buttons.addWidget(self.btn_use_upscaler)
        self.btn_download_upscaler = QPushButton("Install / verify selected model")
        self.btn_download_upscaler.clicked.connect(self.start_upscaler_download)
        upscaler_buttons.addWidget(self.btn_download_upscaler)
        upscaler_layout.addLayout(upscaler_buttons)
        self.upscaler_status = QLabel()
        self.upscaler_status.setWordWrap(True)
        upscaler_layout.addWidget(self.upscaler_status)
        self.upscaler_combo.currentIndexChanged.connect(self._refresh_upscaler_details)
        self._refresh_upscaler_details()
        ultimate_page.addWidget(upscaler_group)
        ultimate_page.addStretch(1)

        lora_page = add_settings_tab("LoRA")
        lora_page.addWidget(self.lora_settings_group)
        lora_note = QLabel("Reads embedded details first. CivitAI lookup is optional and matches the local file by SHA-256.")
        lora_note.setWordWrap(True)
        lora_note.setStyleSheet("color: #94a6bb; padding: 0 3px;")
        lora_page.addWidget(lora_note)
        lora_page.addStretch(1)
        self._refresh_lora_info()

        root.addWidget(self.catalog_tabs, 1)

        self.destination_label = QLabel(f"Model folder · {target_dir}")
        self.destination_label.setWordWrap(True)
        self.destination_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.destination_label)
        self.total_label = QLabel()
        self.total_label.setStyleSheet("font-weight: 650; color: #8fc8f5;")
        root.addWidget(self.total_label)
        licenses = QLabel(
            f'<a href="{QWEN_LICENSE_URL}">Qwen license</a> · '
            f'<a href="{QWEN_REPO_URL}">Encoder / projector source</a> · '
            f'<a href="{QWEN_VAE_REPO_URL}">VAE source</a>'
        )
        licenses.setOpenExternalLinks(True)
        licenses.setWordWrap(True)
        root.addWidget(licenses)
        self.accept_license = QCheckBox("I accept the selected model licenses")
        self.accept_license.stateChanged.connect(self._update_enabled)
        root.addWidget(self.accept_license)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        root.addWidget(self.bar)
        initial_status = "Choose components, then Apply or Download."
        if self.profile_warnings:
            initial_status = (
                "Some saved selections were invalid; choose the correct local files.\n" + "\n".join(self.profile_warnings)
            )
        self.status = QLabel(initial_status)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #aebed0; font-size: 9pt;")
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_apply = QPushButton("Apply installed")
        self.btn_apply.clicked.connect(self.apply_installed)
        buttons.addWidget(self.btn_apply)
        self.btn_download = QPushButton("Download & apply")
        self.btn_download.clicked.connect(self.start_download)
        buttons.addWidget(self.btn_download)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.cancel_or_close)
        buttons.addWidget(self.btn_close)
        root.addLayout(buttons)
        for role in MODEL_ROLE_ORDER:
            self._refresh_details(role)
        self._update_enabled()

    def _selected_encoder_is_non_gguf(self) -> bool:
        """Return True when the currently selected text encoder is a Safetensors/ConvRot file."""
        if "text_encoder" not in self.combos:
            return False
        data = self._current_data("text_encoder")
        kind = data.get("kind")
        if kind == "catalog" and data.get("id") in MODEL_CATALOG_BY_ID:
            filename = str(MODEL_CATALOG_BY_ID[data["id"]].get("filename", ""))
        else:
            filename = str(data.get("filename") or data.get("path") or "")
        return bool(filename) and (not filename.lower().endswith(".gguf") or _text_encoder_embeds_vision(filename))

    def _refresh_lora_info(self, *_):
        """Show safe local metadata whenever the selected LoRA path changes."""
        app_parent = self.parent()
        path = str(app_parent.lora_path_edit.text().strip()) if hasattr(app_parent, "lora_path_edit") else ""
        self._lora_online_info = None
        self._lora_local_info = inspect_lora_file(path)
        self.lora_online_status.setText("")
        self.lora_civitai_link.setText("")
        self._render_lora_info()

    def _render_lora_info(self):
        local = self._lora_local_info or {}
        online = self._lora_online_info or {}
        if not local.get("exists"):
            path = str(local.get("path", ""))
            self.lora_info_summary.setText("No LoRA file selected." if not path else f"File not found: {Path(path).name}")
            self.lora_trigger_words.setPlainText("")
            self.lora_example_prompts.setPlainText("")
            self.lora_description_label.setText("Description: not available")
            self.lora_info_source.setText("Select an existing local LoRA. Safetensors headers and JSON sidecars are read without loading the tensor data.")
            self.btn_lookup_lora_info.setEnabled(False)
            return
        title = online.get("name") or local.get("file_name", "LoRA")
        size = local.get("size_label") or "Size unknown"
        base_model = online.get("base_model") or local.get("base_model") or "Not specified"
        creator = online.get("creator") or local.get("creator") or "Not specified"
        base_model = base_model if base_model not in ("", "Not specified") else "Not specified"
        creator = creator if creator not in ("", "Not specified") else "Not specified"
        local_triggers = local.get("trigger_words") or []
        trigger_words = online.get("trigger_words") or local_triggers
        local_examples = local.get("example_prompts") or []
        examples = online.get("example_prompts") or local_examples
        description = online.get("description") or local.get("description") or "No description available."
        source = online.get("source") or local.get("source") or "Local file"
        self.lora_info_summary.setText(f"{title}  ·  {size}\nBase model: {base_model}    ·    Creator: {creator}")
        self.lora_trigger_words.setPlainText(", ".join(trigger_words) if trigger_words else "No trigger / trained words found in available metadata.")
        self.lora_example_prompts.setPlainText("\n\n".join(examples) if examples else "No example prompts found in available metadata.")
        self.lora_description_label.setText("Description: " + str(description))
        self.lora_info_source.setText("Source: " + str(source))
        self.btn_lookup_lora_info.setEnabled(True)
        if online.get("url"):
            url = str(online["url"])
            self.lora_civitai_link.setText(f'<a href="{url}">Open matched CivitAI model page</a>')

    def _lookup_lora_online(self):
        if self.lora_lookup_worker is not None and self.lora_lookup_worker.isRunning():
            return
        app_parent = self.parent()
        path = str(app_parent.lora_path_edit.text().strip()) if hasattr(app_parent, "lora_path_edit") else ""
        if not path or not Path(path).is_file():
            self.lora_online_status.setText("Choose an existing LoRA file first.")
            return
        self._lora_lookup_path = str(Path(path).expanduser().resolve())
        self.btn_lookup_lora_info.setEnabled(False)
        self.lora_online_status.setText("Preparing lookup…")
        worker = LoRAInfoLookupWorker(self._lora_lookup_path, self)
        self.lora_lookup_worker = worker
        worker.progress.connect(self.lora_online_status.setText)
        worker.found.connect(self._lora_info_online_found)
        worker.failed.connect(self._lora_info_online_failed)
        worker.finished.connect(self._lora_info_lookup_finished)
        worker.start()

    def _lora_info_online_found(self, info: dict[str, Any]):
        current_path = str(Path(self.parent().lora_path_edit.text().strip()).expanduser().resolve()) if self.parent().lora_path_edit.text().strip() else ""
        if current_path != self._lora_lookup_path:
            return  # user changed the selected adapter while the lookup was running
        self._lora_online_info = dict(info)
        self.lora_online_status.setText("Matched by SHA-256 · local file was not uploaded.")
        self._render_lora_info()

    def _lora_info_online_failed(self, message: str):
        current_path = str(Path(self.parent().lora_path_edit.text().strip()).expanduser().resolve()) if self.parent().lora_path_edit.text().strip() else ""
        if current_path == self._lora_lookup_path:
            self.lora_online_status.setText(message)

    def _lora_info_lookup_finished(self):
        self.btn_lookup_lora_info.setEnabled(bool(self._lora_local_info.get("exists")))

    def _read_profile(self) -> dict[str, Any]:
        try:
            obj = json.loads((self.target_dir / "model-selection.json").read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _local_candidates(self, role: str) -> list[Path]:
        """Offer local checkpoint files for explicit assignment to any role.

        Extension alone cannot prove architecture compatibility; the role is assigned by
        the user. Any *.safetensors file is allowed as requested, plus GGUF for sd.cpp.
        """
        try:
            files = [p for p in self.target_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in {".safetensors", ".gguf"}]
        except OSError:
            return []
        catalog_names = {meta["filename"].lower() for meta in MODEL_CATALOG}
        return sorted((p for p in files if p.name.lower() not in catalog_names
                       and not _model_role_mismatch(role, p.name)), key=lambda q: q.name.lower())

    def _current_data(self, role: str) -> dict[str, Any]:
        combo = self.combos[role]
        data = combo.currentData()
        if not isinstance(data, dict):
            return {"kind": "none", "id": "none", "role": role}
        return data

    def _selected_profile(self) -> dict[str, Any]:
        profile: dict[str, Any] = {"schema_version": 1, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        non_gguf_encoder = self._selected_encoder_is_non_gguf()
        for role in MODEL_ROLE_ORDER:
            data = self._current_data(role)
            kind = data.get("kind")
            if role == "vision_projector" and non_gguf_encoder:
                profile[role] = {"id": "none", "filename": "", "path": "", "title": "No vision projector (embedded in Safetensors/ConvRot)"}
                continue
            if kind == "none":
                profile[role] = {"id": "none", "filename": "", "path": "", "title": "No vision projector"}
            elif kind == "catalog":
                meta = MODEL_CATALOG_BY_ID[data["id"]]
                if role == "vision_projector" and not (self.target_dir / meta["filename"]).is_file():
                    profile[role] = {"id": "none", "filename": "", "path": "", "title": "No vision projector"}
                else:
                    profile[role] = {"id": meta["id"], "filename": meta["filename"], "path": "", "title": meta["title"]}
            else:
                local_path = Path(str(data.get("path", ""))).expanduser().resolve()
                profile[role] = {"id": "local", "filename": local_path.name, "path": str(local_path), "title": data.get("title", local_path.name)}
        upscaler_id = str(self.upscaler_combo.currentData() or "realesrgan-x4plus")
        upscaler_meta = REALESRGAN_CATALOG_BY_ID.get(upscaler_id, REALESRGAN_CATALOG[0])
        profile["upscaler"] = {
            "id": upscaler_meta["id"], "title": upscaler_meta["title"],
            "backend": upscaler_meta.get("backend", "official"),
            "runtime_url": REALESRGAN_NCNN_URL if upscaler_meta.get("backend", "official") == "official" else "https://github.com/upscayl/upscayl-ncnn/releases/tag/20251207-174704",
            "runtime_urls": list(REALESRGAN_NCNN_URLS) if upscaler_meta.get("backend", "official") == "official" else ["https://github.com/upscayl/upscayl-ncnn/releases/tag/20251207-174704"],
            "model_zoo_url": upscaler_meta.get("model_source_url", REALESRGAN_MODEL_ZOO_URL),
            "pytorch_reference_url": upscaler_meta.get("pytorch_weights_url", upscaler_meta.get("model_source_url", "")),
        }
        return profile

    def _profile_file_write(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)
        profile = self._selected_profile()
        tmp = self.target_dir / "model-selection.json.tmp"
        tmp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.target_dir / "model-selection.json")
        notices = [
            "P.AI AI IMAGE STUDIO — selected AtomicChat Qwen-Image Turbo model component licenses",
            "",
            "P.AI does not relicense model weights. Review each upstream model page and license before use or redistribution.",
            "",
        ]
        for role in MODEL_ROLE_ORDER:
            data = profile.get(role, {})
            model_id = data.get("id")
            if model_id in MODEL_CATALOG_BY_ID:
                meta = MODEL_CATALOG_BY_ID[model_id]
                notices.append(f"{MODEL_ROLE_LABELS[role]}: {meta['title']}")
                notices.append(f"  Source: {meta['source']}")
                notices.append(f"  License: {meta['license']} — {meta['license_url']}")
        (self.target_dir / "MODEL-LICENSE-NOTICE.txt").write_text("\n".join(notices) + "\n", encoding="utf-8")

    def _selected_catalog_assets(self) -> list[dict[str, Any]]:
        assets = []
        seen = set()
        non_gguf_encoder = self._selected_encoder_is_non_gguf()
        for role in MODEL_ROLE_ORDER:
            if role == "vision_projector" and non_gguf_encoder:
                continue
            data = self._current_data(role)
            if data.get("kind") != "catalog":
                continue
            meta = MODEL_CATALOG_BY_ID[data["id"]]
            if meta["filename"] not in seen:
                assets.append(meta)
                seen.add(meta["filename"])
        return assets

    def _selected_bytes(self) -> int:
        return sum(int(meta["size_hint"]) for meta in self._selected_catalog_assets())

    def _refresh_details(self, role: str):
        if role == "text_encoder" and "vision_projector" in self.combos:
            if self._selected_encoder_is_non_gguf():
                vp_combo = self.combos["vision_projector"]
                for idx in range(vp_combo.count()):
                    vp_data = vp_combo.itemData(idx)
                    if isinstance(vp_data, dict) and vp_data.get("kind") == "none":
                        if vp_combo.currentIndex() != idx:
                            vp_combo.blockSignals(True)
                            vp_combo.setCurrentIndex(idx)
                            vp_combo.blockSignals(False)
                        break
            if "vision_projector" in self.detail_labels:
                self._refresh_details("vision_projector")
        data = self._current_data(role)
        if data.get("kind") == "none":
            if self._selected_encoder_is_non_gguf():
                self.detail_labels[role].setText("Not needed — Safetensors/ConvRot encoders include vision support directly.")
            else:
                self.detail_labels[role].setText("Optional — only used when a GGUF text encoder needs VLM mmproj.")
        elif data.get("kind") == "local":
            local_path = Path(data.get("path", ""))
            status = "Available" if local_path.is_file() else "Not found"
            self.detail_labels[role].setText(f"Local file · {status}: {local_path.name}")
        else:
            meta = MODEL_CATALOG_BY_ID[data["id"]]
            path = self.target_dir / meta["filename"]
            status = "Available; verify the hash to confirm" if path.is_file() else "Selected for download"
            self.detail_labels[role].setText(f"{_human_gb(int(meta['size_hint']))} · {meta['license']} · {status}")
        selected_size = _human_gb(self._selected_bytes())
        self.total_label.setText(f"Download size · {selected_size}")
        self._update_enabled()

    def _browse_local(self, role: str):
        filters = "Checkpoint files (*.safetensors *.gguf);;Safetensors (*.safetensors);;GGUF (*.gguf);;All files (*.*)"
        selected = QFileDialog.getOpenFileName(self, f"Choose local {role.replace('_', ' ')} file", str(self.target_dir), filters)[0]
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        mismatch = _model_role_mismatch(role, path.name)
        if mismatch:
            QMessageBox.warning(self, "Wrong model component", mismatch)
            return
        try:
            validate_model_file(path, role=role)
        except ValidationError as exc:
            QMessageBox.warning(self, "Invalid model file", str(exc))
            return
        combo = self.combos[role]
        data = {"kind": "local", "path": str(path), "filename": path.name, "role": role, "title": path.name}
        combo.addItem(f"Local file · {path.name}", data)
        combo.setCurrentIndex(combo.count() - 1)

    def _upscaler_runtime_executable(self) -> Optional[Path]:
        model_id = str(self.upscaler_combo.currentData() or "realesrgan-x4plus")
        meta = REALESRGAN_CATALOG_BY_ID.get(model_id, {})
        if meta.get("backend") == "upscayl":
            runtime_root = DATA_ROOT / "runtime" / "upscayl-ncnn"
            exe = next((path for path in runtime_root.rglob("upscayl-bin.exe") if path.is_file()), None) if runtime_root.is_dir() else None
            if exe is None:
                return None
            model_dir = exe.parent / "models"
            stem = str(meta.get("model_file", model_id))
            files_ok = all(
                (model_dir / f"{stem}.{suffix}").is_file()
                and (model_dir / f"{stem}.{suffix}").stat().st_size == int(meta[f"{suffix}_size"])
                for suffix in ("param", "bin")
            )
            return exe if files_ok else None
        runtime_root = DATA_ROOT / "runtime" / "realesrgan-ncnn-vulkan"
        if not runtime_root.is_dir():
            return None
        exe = next((path for path in runtime_root.rglob("realesrgan-ncnn-vulkan.exe") if path.is_file()), None)
        if not exe:
            return None
        model_dir = exe.parent / "models"
        model_files = {str(path) for path in model_dir.glob("*") if path.is_file()} if model_dir.is_dir() else set()
        return exe if _realesrgan_model_assets_present(model_files) else None

    def _refresh_upscaler_details(self, *_):
        model_id = str(self.upscaler_combo.currentData() or "realesrgan-x4plus")
        meta = REALESRGAN_CATALOG_BY_ID.get(model_id, REALESRGAN_CATALOG[0])
        self.upscaler_detail.setOpenExternalLinks(True)
        self.upscaler_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        source_url = meta.get("model_source_url") or meta.get("pytorch_weights_url") or REALESRGAN_MODEL_ZOO_URL
        source_label = "Community NCNN source" if meta.get("backend") == "upscayl" else "Upstream model source"
        backend_label = "Upscayl-NCNN" if meta.get("backend") == "upscayl" else "Real-ESRGAN NCNN"
        self.upscaler_detail.setText(
            f"{meta['description']} · Backend: <b>{backend_label}</b> · NCNN model: <code>{meta['id']}</code> · "
            f'<a href="{source_url}">{source_label}</a>'
        )
        installed = self._upscaler_runtime_executable()
        self.upscaler_status.setText("Selected model and Vulkan runtime ready" if installed else "Selected model/runtime will be downloaded and verified on demand")

    def apply_upscaler_choice(self):
        model_id = str(self.upscaler_combo.currentData() or "realesrgan-x4plus")
        app_parent = self.parent()
        combo = getattr(app_parent, "upscale_model_combo", None)
        if combo is not None:
            index = combo.findData(model_id)
            if index >= 0:
                combo.setCurrentIndex(index)
                if hasattr(app_parent, "schedule_save"):
                    app_parent.schedule_save()
        # Save this independent selection without requiring Qwen models to be downloaded.
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            profile_path = self.target_dir / "model-selection.json"
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                if not isinstance(profile, dict):
                    profile = {}
            except (OSError, ValueError, TypeError):
                profile = {}
            profile["upscaler"] = self._selected_profile()["upscaler"]
            tmp = profile_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, profile_path)
        except OSError as exc:
            self.status.setText(f"Upscaler selected in the active session; could not save catalog profile: {exc}")
            return
        self.upscaler_status.setText(f"Selected · {REALESRGAN_CATALOG_BY_ID[model_id]['title']}")

    def start_upscaler_download(self):
        if self.upscaler_worker is not None and self.upscaler_worker.isRunning():
            return
        app_parent = self.parent()
        active_worker = getattr(app_parent, "worker", None)
        model_worker = getattr(app_parent, "model_worker", None)
        if (active_worker is not None and active_worker.isRunning()) or (model_worker is not None and model_worker.isRunning()):
            QMessageBox.information(self, "Processing active", "Wait for image processing/model loading to finish before downloading the upscaler.")
            return
        engine = getattr(app_parent, "engine", None)
        if engine is None:
            QMessageBox.warning(self, "Upscaler unavailable", "The application engine is not available. Use the official download link shown above.")
            return
        if hasattr(engine, "force_stop"):
            engine.force_stop.clear()
        model_id = str(self.upscaler_combo.currentData() or "realesrgan-x4plus")
        self.upscaler_status.setText(f"Installing/verifying selected AI model · {model_id}…")
        self.upscaler_worker = RealESRGANRuntimeWorker(engine, self, model_id)
        self.upscaler_worker.completed.connect(self._upscaler_download_completed)
        self.upscaler_worker.failed.connect(self._upscaler_download_failed)
        self.upscaler_worker.start()
        self._update_enabled()

    def _upscaler_download_completed(self, path: str):
        self.upscaler_status.setText("Selected model and Vulkan runtime installed/verified.")
        self._refresh_upscaler_details()
        self._update_enabled()

    def _upscaler_download_failed(self, message: str):
        self.upscaler_status.setText("Runtime installation failed: " + message.split("\n", 1)[0])
        QMessageBox.warning(self, "AI upscaler setup failed", message)
        self._update_enabled()

    def _profile_has_verified_catalog_assets(self) -> tuple[bool, str]:
        manifest = {}
        try:
            loaded = json.loads((self.target_dir / "model-manifest.json").read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded.get("files", {})
        except (OSError, ValueError, TypeError):
            pass
        non_gguf_encoder = self._selected_encoder_is_non_gguf()
        for role in MODEL_ROLE_ORDER:
            if role == "vision_projector" and non_gguf_encoder:
                continue
            data = self._current_data(role)
            if data.get("kind") == "none":
                if role in MODEL_ROLE_REQUIRED:
                    return False, f"{MODEL_ROLE_LABELS[role]} is required."
                continue
            if data.get("kind") == "local":
                local_path = Path(data.get("path", ""))
                if not local_path.is_file():
                    if role == "vision_projector":
                        continue
                    return False, f"Local file for {MODEL_ROLE_LABELS[role]} was not found."
                mismatch = _model_role_mismatch(role, local_path.name)
                if mismatch:
                    return False, mismatch
                continue
            meta = MODEL_CATALOG_BY_ID[data["id"]]
            path = self.target_dir / meta["filename"]
            if not path.is_file():
                if role == "vision_projector":
                    # Vision projector (mmproj) is strictly optional; if not downloaded, allow applying without it.
                    continue
                return False, f"{meta['title']} has not been downloaded. Click Download / Verify & Apply."
            stat = path.stat()
            stored = manifest.get(meta["filename"], {}) if isinstance(manifest, dict) else {}
            expected_hash = str(meta.get("sha256", "")).lower()
            stored_hash = str(stored.get("sha256", "")).lower()
            hash_ok = (stored_hash == expected_hash) if expected_hash else (len(stored_hash) == 64)
            if not (hash_ok and stored.get("size") == stat.st_size and stored.get("mtime_ns") == stat.st_mtime_ns):
                return False, f"{meta['filename']} is not yet registered in a valid manifest. Click Download / Verify & Apply to calculate/verify SHA-256."
        return True, ""

    def apply_installed(self):
        app_parent = self.parent()
        if app_parent is not None and hasattr(app_parent, "save_settings"):
            app_parent.save_settings()
        ok, reason = self._profile_has_verified_catalog_assets()
        if not ok:
            QMessageBox.information(self, "Components need verification", reason)
            return
        try:
            self._profile_file_write()
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save model selection", str(exc))
            return
        self.applied = True
        self.completed_successfully = True
        self.accept()

    def _update_enabled(self, *_):
        busy = ((self.worker is not None and self.worker.isRunning()) or
                (self.upscaler_worker is not None and self.upscaler_worker.isRunning()))
        self.btn_use_upscaler.setEnabled(not busy)
        self.btn_download_upscaler.setEnabled(not busy)
        self.btn_download.setEnabled(not busy)
        self.btn_download.setToolTip(
            "Downloads/verifies the selected catalog components. If license acknowledgment is not checked, "
            "the app will ask you to confirm before proceeding."
        )
        self.btn_apply.setEnabled(not busy)
        self.btn_close.setEnabled(not busy)

    def start_download(self):
        if self.worker is not None and self.worker.isRunning():
            return
        assets = self._selected_catalog_assets()
        if not assets:
            local_roles = [
                role for role in MODEL_ROLE_ORDER
                if self._current_data(role).get("kind") == "local"
                and Path(str(self._current_data(role).get("path", ""))).is_file()
            ]
            if local_roles:
                QMessageBox.information(
                    self, "No catalog downloads selected",
                    "The selected components are local files, so there is nothing to download. "
                    "Use ‘Apply installed choices’ to save this selection.\n\n"
                    "The selected components are local files; catalog downloads are not required for them. "
                    "Use ‘Apply installed choices’ to save these selections."
                )
            else:
                QMessageBox.information(
                    self, "No catalog files selected",
                    "Select a catalog component to download/verify, or use Browse for a local model file."
                )
            self._update_enabled()
            return
        if not self.accept_license.isChecked():
            answer = QMessageBox.question(
                self,
                "License review required",
                "Before downloading or verifying these components, review the license links shown in this window. "
                "Confirm that you have reviewed and will comply with the license terms for each selected component.\n\n"
                "Have you reviewed the licenses for the selected components and agree to comply with their terms?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.status.setText("Download not started. Review the selected component licenses and try again.")
                return
            self.accept_license.setChecked(True)
        # Required components cannot be omitted. Vision projector is the only optional role.
        for role in MODEL_ROLE_REQUIRED:
            data = self._current_data(role)
            if data.get("kind") == "none":
                QMessageBox.warning(self, "Missing component", f"{MODEL_ROLE_LABELS[role]} cannot be disabled.")
                return
        self.btn_download.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.accept_license.setEnabled(False)
        for combo in self.combos.values():
            combo.setEnabled(False)
        for browse in self.browse_buttons.values():
            browse.setEnabled(False)
        self.status.setText("Preparing selected model downloads…")
        self.worker = ModelBundleDownloadWorker(self.target_dir, assets, self)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._thread_finished)
        self.worker.start()

    def _progress(self, value: int, message: str):
        self.bar.setValue(value)
        self.status.setText(message)

    def _completed(self, path: str):
        try:
            self._profile_file_write()
            self.completed_successfully = True
            self.applied = True
            self.status.setText(f"Selected components validated and profile saved:\n{path}")
            QTimer.singleShot(0, self._accept_when_worker_is_idle)
        except Exception as exc:
            self.status.setText(f"Files verified, but could not save model profile: {exc}")
            QMessageBox.critical(self, "Cannot save model selection", str(exc))

    def _accept_when_worker_is_idle(self):
        if self.applied and (self.worker is None or not self.worker.isRunning()):
            if self.worker is not None:
                self.worker.wait()
            self.accept()

    def _thread_finished(self):
        if self.worker is not None:
            self.worker.wait()
        if self.applied:
            self.accept()
        else:
            self.accept_license.setEnabled(True)
            self.btn_close.setEnabled(True)
            for combo in self.combos.values():
                combo.setEnabled(True)
            for browse in self.browse_buttons.values():
                browse.setEnabled(True)
            self._update_enabled()

    def _failed(self, message: str):
        self.status.setText("Download / verification failed. Partial .part files are kept when possible.")
        self.accept_license.setEnabled(True)
        QMessageBox.critical(self, "Model component download failed", message)

    def _cancelled(self):
        self.status.setText("Download cancelled. Partial .part files are retained for retry.")
        self.accept_license.setEnabled(True)

    def cancel_or_close(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel()
            self.status.setText("Cancelling after the current network block…")
            self.btn_close.setEnabled(False)
            return
        self.reject()

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel()
            self.status.setText("Cancelling download…")
            return
        app_parent = self.parent()
        if app_parent is not None and hasattr(app_parent, "save_settings"):
            app_parent.save_settings()
        super().reject()

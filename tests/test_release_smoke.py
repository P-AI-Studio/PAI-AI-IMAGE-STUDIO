from pathlib import Path
import ast
import json
import re
import zipfile
import math
import textwrap
from typing import Optional
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pai_ai_image_studio.py"
PACKAGE_MODULE_ORDER = (
    "pai_studio/constants.py",
    "pai_studio/preview_policy.py",
    "pai_studio/workspace.py",
    "pai_studio/grounding.py",
    "pai_studio/prompts.py",
    "pai_studio/image_ops.py",
    "pai_studio/engine.py",
    "pai_studio/workers.py",
    "pai_studio/ui/dialogs.py",
    "pai_studio/ui/canvas.py",
    "pai_studio/ui/main_window.py",
    "pai_studio/validation.py",
)


def read_source():
    parts = ["from __future__ import annotations\n"]
    for rel in PACKAGE_MODULE_ORDER:
        text = (ROOT / rel).read_text(encoding="utf-8")
        # Strip module-level relative imports and duplicate __future__ imports so the
        # concatenated source forms a valid single AST module for legacy smoke inspections.
        cleaned_lines = []
        skip_parens = False
        for line in text.splitlines():
            if skip_parens:
                if ")" in line:
                    skip_parens = False
                continue
            if line.strip() == "from __future__ import annotations":
                continue
            if line.startswith("from .") or line.startswith("from .."):
                if "(" in line and ")" not in line:
                    skip_parens = True
                continue
            cleaned_lines.append(line)
        parts.append("\n".join(cleaned_lines))
    return "\n\n".join(parts)


def test_native_log_window_reads_large_logs_incrementally(tmp_path):
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_read_log_window")
    helper_src = textwrap.dedent(ast.get_source_segment(source, helper_node))
    namespace = {"Path": Path, "Optional": Optional}
    exec(helper_src, namespace)
    log_path = tmp_path / "sd-server.log"
    log_path.write_bytes(b"phase-one\nphase-two\n")

    offset, first = namespace["_read_log_window"](log_path, 0, max_bytes=10)
    next_offset, second = namespace["_read_log_window"](log_path, offset, max_bytes=10)
    assert first == "phase-one\n"
    assert second == "phase-two\n"
    assert next_offset == log_path.stat().st_size


def test_lora_info_reads_safetensors_header_without_deserializing_tensor_data(tmp_path):
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inspect_lora_file")
    helper_src = ast.get_source_segment(source, helper_node)
    namespace = {"Path": Path, "json": json, "re": re, "Any": object}
    exec(helper_src, namespace)

    metadata = {
        "modelspec.architecture": "qwen-image-2.1",
        "modelspec.author": "P.AI test creator",
        "modelspec.description": "<b>Fabric detail</b> adapter for testing.",
        "modelspec.prompt": "fabric detail close-up, soft studio light\nmacro weave, natural light",
        "ss_trigger_words": "fabric_detail, fine_texture",
    }
    header = json.dumps({"__metadata__": metadata}).encode("utf-8")
    lora_path = tmp_path / "sample.safetensors"
    lora_path.write_bytes(len(header).to_bytes(8, "little") + header + b"tensor-payload-is-not-decoded")

    info = namespace["inspect_lora_file"](lora_path)
    assert info["exists"] is True
    assert info["base_model"] == "qwen-image-2.1"
    assert info["creator"] == "P.AI test creator"
    assert info["trigger_words"] == ["fabric_detail", "fine_texture"]
    assert info["description"] == "Fabric detail adapter for testing."
    assert info["example_prompts"] == ["fabric detail close-up, soft studio light", "macro weave, natural light"]
    assert "safetensors metadata" in info["source"]


def test_lora_info_settings_use_local_metadata_and_optional_hash_matched_civitai_lookup():
    source = read_source()
    assert 'self.lora_info_card = QGroupBox("LoRA Info")' in source
    assert 'example_label = QLabel("Example prompts")' in source
    assert 'QPushButton("Look up CivitAI info")' in source
    assert '/api/v1/model-versions/by-hash/' in source
    assert 'The file itself is not uploaded.' in source
    assert 'Safetensors headers and JSON sidecars are read without loading the tensor data.' in source
    assert 'trainedWords' in source


def test_square_outpaint_uses_context_filled_rgb_canvas_and_masks_only_new_margins():
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_square_outpaint_inputs")
    helper_src = ast.get_source_segment(source, helper_node)
    namespace = {"Image": Image, "ImageOps": ImageOps, "ImageFilter": __import__("PIL.ImageFilter", fromlist=["ImageFilter"])}
    exec(helper_src, namespace)
    src = Image.new("RGB", (80, 120))
    pix = src.load()
    for y in range(120):
        for x in range(80):
            pix[x, y] = (x * 3, y * 2, (x + y) % 255)
    canvas, mask = namespace["build_square_outpaint_inputs"](src, 120, (20, 0, 100, 120), feather=12)
    assert canvas.mode == "RGB" and canvas.size == (120, 120)
    assert mask.mode == "L" and mask.size == (120, 120)
    assert canvas.getpixel((0, 60)) != canvas.getpixel((10, 60))
    assert mask.getpixel((0, 60)) == 255
    assert mask.getpixel((60, 60)) == 0
    assert 'payload["init_image"]' in source and 'payload["mask_image"]' in source
    assert 'payload["strength"] = max(0.0, min(1.0, float(strength)))' in source
    assert "composite_square_outpaint_result" in source


def test_realesrgan_archive_validation_does_not_require_every_optional_model_family():
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_realesrgan_model_assets_present")
    helper_src = ast.get_source_segment(source, helper_node)
    namespace = {"Path": Path}
    exec(helper_src, namespace)
    check = namespace["_realesrgan_model_assets_present"]
    assert check({"models/realesrgan-x4plus.param", "models/realesrgan-x4plus.bin"})
    assert check({"models/realesr-animevideov3-x2.param", "models/realesr-animevideov3-x2.bin"})
    assert not check({"models/realesrgan-x4plus.param"})
    assert 'REALESRGAN_HF_NCNN_BASE = "https://huggingface.co/edgetools/realesrgan/resolve/main/"' in source
    assert 'https://huggingface.co/HNoise7676/RealESRGAN_x4plus_anime_6B_NCNN/resolve/main/' in source
    assert 'remote_stem = "RealESRGAN_x4plus_anime_6B"' in source
    assert "The downloaded file is not a valid Real-ESRGAN Windows ZIP" in source


def test_runtime_progress_names_the_native_archives_explicitly():
    source = read_source()
    assert 'Downloading sd-master-1d0a58a-bin-win-cuda12-x64.zip' in source
    assert 'Downloading cudart-sd-bin-win-cu12-x64.zip' in source
    assert 'skipping sd-master-1d0a58a-bin-win-cuda12-x64.zip' in source


def test_qwen_enhance_large_outputs_use_qwen_tiles_not_standalone_realesrgan():
    source = read_source()
    worker = source[source.index('    def _enhance_qwen_output'):source.index('    def process(self, job: Job)')]
    assert 'self._qwen_detail_refine_large_output' in worker
    assert 'self._qwen_detail_refine_large_output' in worker
    assert 'self._run_upscale_file' not in worker
    detail = source[source.index('    def _qwen_detail_refine_large_output'):source.index('    def _enhance_qwen_output')]
    assert 'Do not simply sharpen, add halos, overemphasize grain, outpaint, remove scene objects' in detail
    assert 'only permitted content removal is a visible' in detail
    assert 'prompt.strip() + \"\\n\\nLOCAL DETAIL REFINEMENT PASS' not in detail
    assert 'def _run_upscale_file' in source
    assert 'job.workflow == "upscale_only"' in source


def test_source_parses():
    ast.parse(read_source())


def test_no_torch_or_diffusers_runtime_imports():
    tree = ast.parse(read_source())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "torch" not in imports
    assert "diffusers" not in imports


def test_atomicchat_catalog_pairs_turbo_denoiser_with_uncensored_encoder_and_projector():
    source = read_source()
    block = source[source.index("MODEL_CATALOG: list["):source.index("MODEL_CATALOG_BY_ID =")]
    assert "AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main/Qwen-Image-2.1-Turbo-AD-Q5_K.gguf" in block
    assert "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main/Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf" in block
    assert "mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf" in block
    assert "Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors" in block
    for role in ("diffusion", "text_encoder", "vision_projector", "vae"):
        assert f'"role": "{role}"' in block
    assert '"default": True' in block
    # These upstream model cards do not publish pinned SHA-256 hashes; no hash is fabricated.
    assert '"sha256": ""' in block
    assert "AD-Q4_K" in block and "AD-Q5_K" in block and "AD-Q6_K" in block and "Q8_0" in block


def test_closeup_reuses_model_catalog_encoder_instead_of_downloading_separate_weights():
    source = read_source()
    assert 'QWEN_REPO_URL = "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF"' in source
    assert 'QWEN_TURBO_REPO_URL = "https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF"' in source
    assert 'model_parts = self._find_model_parts(Path(self.model_path).expanduser().resolve())' in source
    assert 'self._locate_closeup_target(' in source
    assert 'model_path=model_path' in source and 'mmproj_path=mmproj_path' in source
    assert 'GROUNDING_RUNTIME_ASSETS: list[dict[str, Any]]' in source
    assert 'Qwen3VL-8B-Instruct-Q4_K_M.gguf' not in source
    assert 'Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf' not in source
    assert "abenzerps/Qwen-Image-2.1-Uncensored-Quantized" not in source


def test_turbo_uses_eight_step_schedule_cfg_and_sigmas():
    source = read_source()
    assert 'is_turbo = "turbo" in diffusion_name' in source
    assert 'sample_params.update(qwen_turbo_sampling_params())' in source
    assert 'steps overridden to the required 8-step Euler schedule (CFG=1.0)' in source
    assert 'def qwen_turbo_sampling_params()' in source
    assert '"sample_steps": 8' in source
    assert '"guidance": {"txt_cfg": 1.0}' in source
    assert 'turbo_steps = requested_steps if custom_ultimate_steps else 8' not in source
    for sigma in ("0.978453", "0.95418", "0.926626", "0.89508", "0.845148", "0.704534", "0.414568"):
        assert sigma in source


def test_turbo_sampler_helper_always_returns_checkpoint_sigmas_and_eight_steps():
    tree = ast.parse(read_source())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qwen_turbo_sampling_params")
    source = ast.get_source_segment(read_source(), function)
    namespace = {"Any": object, "QWEN_TURBO_SIGMAS": (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "qwen_turbo_helper.py", "exec"), namespace)
    params = namespace["qwen_turbo_sampling_params"]()
    assert params["sample_steps"] == 8
    assert params["custom_sigmas"] == list(namespace["QWEN_TURBO_SIGMAS"])
    assert len(params["custom_sigmas"]) == 9
    assert params["guidance"] == {"txt_cfg": 1.0}


def test_memory_placement_keeps_diffusion_on_cuda_when_full_bundle_exceeds_16gb_budget():
    source = read_source()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "plan_qwen_memory_placement")
    namespace = {"Any": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "memory_policy.py", "exec"), namespace)
    plan = namespace["plan_qwen_memory_placement"]
    placement = plan(free_vram_gb=15.0, total_vram_gb=16.0, diffusion_gb=6.7, vae_gb=0.66,
                     component_file_gb=13.8, memory_mode="full_gpu")
    assert placement["force_cpu_te_for_capacity"] is False
    assert placement["text_backend"] == "cuda"
    assert placement["offload_to_cpu"] is False
    assert placement["auto_fit_mode"] == "off"
    full_fit = plan(free_vram_gb=24.0, total_vram_gb=24.0, diffusion_gb=6.7, vae_gb=0.66,
                    component_file_gb=13.8, memory_mode="full_gpu")
    assert full_fit["force_cpu_te_for_capacity"] is False
    assert full_fit["text_backend"] == "cuda"
    assert full_fit["auto_fit_mode"] == "off"
    tight = plan(free_vram_gb=8.0, total_vram_gb=16.0, diffusion_gb=6.7, vae_gb=0.66,
                 component_file_gb=13.8, memory_mode="auto")
    assert tight["force_cpu_te_for_capacity"] is True
    assert tight["offload_to_cpu"] is True
    telemetry_unknown = plan(free_vram_gb=0.0, total_vram_gb=16.0, diffusion_gb=6.7, vae_gb=0.66,
                             component_file_gb=12.0, memory_mode="full_gpu")
    assert telemetry_unknown["force_cpu_te_for_capacity"] is False
    assert telemetry_unknown["text_backend"] == "cuda"
    assert telemetry_unknown["offload_to_cpu"] is False
    assert telemetry_unknown["auto_fit_mode"] == "off"


def test_workflow_feature_toggle_labels_are_state_independent_and_color_driven():
    source = read_source()
    assert 'return names.get(feature, feature)' in source
    assert "'ON' if enabled else 'OFF'" not in source
    assert 'QToolButton:checked { background: #7858f5; border-color: #7858f5' in source
    assert 'button.setText(self._feature_toggle_label(feature))' in source


def test_memory_policy_avoids_unnecessary_cpu_offload_and_retries_oom_once():
    source = read_source()
    assert '"gpu_memory_mode": "auto"' in source
    assert '"GPU-only · no CPU inference", "full_gpu"' in source
    assert '"Low VRAM · offload weights to CPU", "low_vram"' in source
    assert 'diffusion_gb >= total_vram_gb * 0.52' in source
    assert 'free_vram_gb < 6.5' in source
    assert 'if offload_to_cpu:' in source
    assert 'args.append("--offload-to-cpu")' in source
    assert 'def _infer_with_auto_offload_retry' in source
    assert 'CUDA memory allocation failed. Retrying this image once' in source
    assert 'diffusion/VAE computation remain on the selected CUDA GPU' in source


def test_eight_step_default_preserves_other_quality_presets():
    source = read_source()
    assert '"quality": "Turbo · 8 steps"' in source
    assert '"num_inference_steps": 8' in source
    assert 'QUALITY_PRESETS = ["Turbo · 8 steps", "Fast · 12 steps", "Balanced · 20 steps", "Standard · 28 steps"]' in source
    assert "def _set_steps_from_preset" in source


def test_status_updates_use_a_qt_signal_for_gui_thread_delivery():
    source = read_source()
    assert 'status_signal = Signal(str)' in source
    assert 'self.status_signal.connect(self._apply_status_message)' in source
    assert 'self.status_signal.emit(str(message))' in source


def test_verified_huggingface_file_hashes_are_pinned_for_default_and_reference_assets():
    source = read_source()
    for sha in (
        "4bb73c53cbe284bbd6d69b9fdc59539c531f0389dd2aa567663d777f8130bc59",
        "c56db41508c4c3c0d5782e40e4f153c38a1154ac9466c1c5c50a6e4bd09f6a69",
        "8ab2888f605030a894c6238d978f51427bfde6732f4a112c71cff0439e265d04",
        "9849f4f5606e7130dafa8ca4fe6d773d91d83b0eed983f6245de3195c9105949",
        "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
    ):
        assert sha in source


def test_catalog_integrity_records_local_hash_when_upstream_pin_unavailable():
    source = read_source()
    assert '"sha256_source": "upstream-pinned" if expected_hash else "computed-locally-no-upstream-pin"' in source
    assert "Downloaded file {key} is too small" in source
    assert "Existing file is implausibly small" in source
    assert "len(stored_hash) == 64" in source
    assert "SHA-256 verification failed" in source


def test_local_engine_loopback_binding_cuda_and_optional_vision_are_explicit():
    source = read_source()
    assert '"--listen-ip", "127.0.0.1"' in source
    assert '"--listen-port", str(port)' in source
    assert '"--backend", backend_spec' in source
    assert 'f"diffusion=cuda{cuda_device_index},te={text_backend},vae=cuda{cuda_device_index}"' in source
    assert 'args.extend(["--llm_vision", str(parts["llm_vision"])])' in source
    assert 'env["CUDA_VISIBLE_DEVICES"] = gpu_uuid' in source
    assert "CPU fallback for diffusion is disabled" in source
    assert "No NVIDIA CUDA GPU was found" in source


def test_model_controls_are_above_workflows_and_configuration_remains_in_settings():
    source = read_source()
    build = source[source.index("    def _build(self):"):source.index("    def _build_sidebar(self):")]
    assert 'QPushButton("🧩 Model Catalog")' in build
    assert 'QPushButton("⚙ Settings")' in build
    assert 'self.settings_layout' not in build  # dialog is persistent, not a second floating settings widget
    assert 'settings_layout.addWidget(self.model_settings_widget)' in build
    assert 'QPushButton("Load Selected Models")' in build
    assert 'QPushButton("Unload")' in build
    assert 'model_row.removeWidget(self.btn_load_model)' in build
    assert 'model_row.removeWidget(self.btn_unload_model)' in build
    assert 'outer.insertLayout(1, self.model_action_row)' in build
    assert 'self.model_path_edit' in build


def test_queue_and_activity_log_are_in_left_dock_with_compact_close_and_stop_modes():
    source = read_source()
    assert 'self.left_panel_shell = QWidget()' in source
    assert 'self.workspace_content_splitter.addWidget(self.queue_panel)' in source
    assert 'self.left_dock_splitter.addWidget(self.workspace_content_splitter)' in source
    assert 'self.left_dock_splitter.addWidget(self.log_panel)' in source
    assert 'self.btn_toggle_left_dock = QToolButton(); self.btn_toggle_left_dock.setText("×")' in source
    assert 'QCheckBox("Stop after this image")' not in source
    assert 'QPushButton("■ Stop current job")' in source
    assert 'QPushButton("■ Stop now")' in source
    assert 'QPushButton("■ Stop current job")' in source
    assert 'Stop-after-current-image requested' not in source
    assert 'self.engine.stop_after_current.set()' in source
    assert 'def stop_after_current_job(self):' in source
    assert 'Stop current job honored' in source
    assert 'self.btn_generate_stop_current.clicked.connect(self.stop_after_current_job)' in source
    assert 'self.btn_generate_stop_now.clicked.connect(self.stop_job)' in source
    assert 'self.btn_edit_stop_current.clicked.connect(self.stop_after_current_job)' in source
    assert 'self.btn_edit_stop_now.clicked.connect(self.stop_job)' in source


def test_preview_has_zoom_pan_independent_side_by_side_and_draggable_slider():
    source = read_source()
    preview = source[source.index("class PreviewCanvas(QWidget):"):source.index("class App(QMainWindow):")]
    assert 'self.pane_zoom = {"original": 1.0, "enhanced": 1.0}' in preview
    assert 'self.pane_pan = {"original": QPoint(0, 0), "enhanced": QPoint(0, 0)}' in preview
    assert "def _paint_side_by_side" in preview
    assert "def _zoom_about" in preview
    assert "self._split_dragging = True" in preview
    assert '"Slider before/after"' in preview
    assert '"Side by side"' in preview
    assert 'def fit_view(self)' in preview


def test_image_queue_includes_per_output_seconds_and_change_vs_source():
    source = read_source()
    assert 'QTableWidget(0, 6)' in source
    assert '["Name", "Workflow", "Status", "Time", "Δ src", "Path"]' in source
    assert '"elapsed_seconds": None, "change_pct": None' in source
    assert 'self.output_durations[str(path)] = time.monotonic() - image_started_at' in source
    assert 'queue_item["elapsed_seconds"] = image_elapsed' in source
    assert 'queue_item["change_pct"] = delta_pct' in source
    assert 'def estimate_pixel_change' in source
    assert 'source_path = info.get("source_path")' in source


def test_progress_shows_phase_steps_percent_elapsed_and_eta():
    source = read_source()
    assert "def _apply_status_message" in source
    assert 're.search(r"step\\s+(\\d+)\\s*/\\s*(\\d+)"' in source
    assert "Step ETA ~{step_eta_text} · overall ETA ~{overall_eta_text} (approx.)" in source
    assert "elapsed {elapsed_text}" in source
    assert "steps completed" in source
    assert "Text encoder / prompt conditioning" in source
    assert "VAE · encoding reference image" in source
    assert "VAE · decoding generated image" in source
    assert "Real-ESRGAN / Real-ESRNet · Vulkan upscaling" in source
    assert "Finalizing · verifying outputs and queue" in source
    assert "def update_preview_info" in source
    assert "Mean pixel change ≈" in source
    assert "{image.width}×{image.height}" in source

def test_queue_fits_without_horizontal_scroll_and_keeps_full_paths_in_tooltips():
    source = read_source()
    assert 'self.queue_table.setColumnHidden(5, True)' in source
    assert 'self.queue_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)' in source
    assert '"Time", "Δ src"' in source
    assert 'cell.setToolTip(f"{value}' in source
    assert 'Workflow: {workflow_name}' in source
    assert '"image_generate": "Generate"' in source
    assert 'self.left_panel_shell.setMinimumWidth(330)' in source


def test_progress_is_determinate_and_has_stage_fill_instead_of_marquee():
    source = read_source()
    assert 'self.progress.setRange(0, 0)' not in source
    assert 'self.progress.setTextVisible(True)' in source
    assert 'Overall {self.progress.value()}% · %v/100' in source
    assert 'self.progress_label.setText(headline)' in source
    assert 'unit_status = f"Step {step_number}/{step_total}"' in source
    assert 'self.progress_detail_label.setText(f"{detail} · elapsed {elapsed_text} · {eta}")' in source
    assert 'self.progress.hide()' in source
    assert 'Model components initialized. Progress appears here when a job starts.' in source


def test_real_esrgan_is_scale_only_and_separate_from_qwen_output_size():
    source = read_source()
    assert 'self.upscale_scale_combo' not in source
    assert 'self.upscale_mode_combo' not in source
    assert 'self.upscale_custom_w' not in source
    assert 'self.upscale_custom_h' not in source
    assert 'scale = realesrgan_native_scale(model_name)' in source
    # Lanczos is allowed only for the Qwen Ultimate source canvas, not as a hidden replacement for Real-ESRGAN.
    upscale_start = source.index('    def _run_upscale_file')
    upscale_end = source.index('    def _translate_image_prompt_to_english', upscale_start)
    upscale_impl = source[upscale_start:upscale_end]
    assert 'Image.Resampling.LANCZOS' not in upscale_impl
    assert 'Image.Resampling.LANCZOS' in source[source.index('    def _enhance_qwen_output'):source.index('    def process(self, job: Job)')]
    assert 'QGroupBox("Output Size · Qwen Image 2.1")' in source
    assert 'self.realesrgan_box = QGroupBox("AI Upscaler · Real-ESRGAN / Real-ESRNet")' in source
    assert 'upscale_page.addWidget(esrgan_card)' in source
    assert 'def _upscale_custom_from_ui(self)' in source
    assert '"upscale_enabled": False' in source
    assert 'self.std_output_size_widget.setVisible(workflow == "enhance")' in source
    assert 'self._upscale_size_in_basic' not in source
    assert 'self.upscale_enabled_check' not in source


def test_closeup_prompt_enforces_a_tight_target_frame_and_uses_lora_trigger_word():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_closeup_prompt")
    namespace = {
        "Optional": __import__("typing").Optional,
        "CLOSEUP_SEMANTIC_AUTO_PROMPT": "Reframe the reference as a close-up of {target}.",
        "CLOSEUP_MANUAL_CROP_PROMPT": "Refine the existing crop of {target}.",
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<closeup-prompt>", "exec"), namespace)
    build = namespace["build_closeup_prompt"]
    auto = build("left eye", "keep natural texture", True, "my_lora_token")
    assert auto.startswith("my_lora_token,")
    assert "left eye" in auto
    assert "geometric crop has already been made" in auto
    assert "do not zoom back out" in auto
    assert "keep natural texture" in auto
    manual = build("left eye", "", False, "")
    assert "existing crop" in manual
    assert "visibly tighter square close-up" not in manual


def test_cuda_bootstrap_reuses_installed_windows_dependencies_and_path_engine():
    source = read_source()
    assert "def _windows_cuda_dependency_dirs() -> list[Path]:" in source
    assert 'os.environ.get("PATH", "").split(os.pathsep)' in source
    assert 'key.upper() == "CUDA_PATH" or key.upper().startswith("CUDA_PATH_V")' in source
    assert "for directory in system_cuda_dirs:" in source
    assert "Prefer the pinned package, but reuse dependencies already installed in Windows." in source
    assert 'shutil.which("sd-server.exe") or shutil.which("sd-server")' in source
    assert 'dll_dirs.update(str(path) for path in self._windows_cuda_dependency_dirs())' in source


def test_closeup_control_labels_match_the_offline_crop_or_ai_path():
    source = read_source()
    assert 'Auto Zoom: ON (Qwen3-VL semantic crop only)' in source
    assert 'Auto Zoom: ON (Qwen3-VL grounding + faithful Enhance)' in source
    assert 'self.closeup_zoom.setEnabled(True)' in source
    assert 'self.update_closeup_controls()' in source[source.index('def on_focus_selected'):source.index('def _set_steps_from_preset')]
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    assert 'build_closeup_prompt(job.target, job.prompt, True, active_trigger)' in process
    assert 'raise RuntimeError(' in process
    assert 'No center crop was substituted' in process
    assert 'target-guided crop + Qwen enhance' in process
    assert '"crop_box": crop_box' in process
    assert '"focus_point": job.focus_point' in process
    assert 'user-selected preview point (takes priority)' in process


def test_square_and_enhance_workflows_log_applied_settings_and_no_ai_passthrough():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    assert 'f"Enhance settings · source={source_original.width}×{source_original.height}' in process
    assert 'f"Square 1:1 mode={mode}' in process
    assert 'source is already square/nearly square and base Enhance is OFF; resize/crop only' in process
    assert 'optional Ultimate pass applied afterward to the finished output if enabled' in process
    assert 'if mode in ("copy", "crop") and not job.enhance:' in process


def test_closeup_and_square_use_fixed_output_sizes_in_their_own_panels():
    source = read_source()
    assert 'FIXED_SQUARE_OUTPUT_SIZES = (1024, 2048, 4096)' in source
    assert 'def normalize_fixed_square_side(value: int) -> int:' in source
    assert 'self.std_size_preset.addItems(["Scale", "Custom dimensions"])' in source
    assert 'self.std_output_scale = max(1, min(4, int(self.settings.get("std_output_scale", 2))))' in source
    for scale in (1, 2, 3, 4):
        assert f'for scale in (1, 2, 3, 4)' in source
    assert 'def calculate_qwen_output_size(' in source
    assert 'def set_std_output_scale(self, scale: int, save: bool = True)' in source
    assert 'self.qwen_output_group.setVisible(workflow == "enhance")' in source
    assert 'self.std_output_size_widget.setVisible(workflow == "enhance")' in source
    assert 'self.closeup_size_buttons = {}' in source
    assert 'self.square_size_buttons = {}' in source
    assert 'side = normalize_fixed_square_side(self.closeup_side)' in source
    assert 'side = normalize_fixed_square_side(self.square_side)' in source
    assert 'self.std_size_preset.setCurrentText("Scale")' not in source[source.index('def set_closeup_side'):source.index('def reset_focus')]
    assert 'self.std_w.setMinimumWidth(96)' in source
    assert 'self.std_h.setMinimumWidth(96)' in source
    assert 'self.std_w.editingFinished.connect' in source
    assert 'self.std_h.editingFinished.connect' in source


def test_fixed_square_sizes_are_normalized_to_supported_choices():
    source = read_source()
    tree = ast.parse(source)
    wanted = {"normalize_fixed_square_side"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = {"FIXED_SQUARE_OUTPUT_SIZES": (1024, 2048, 4096)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<fixed-square-sizes>", "exec"), namespace)
    normalize = namespace["normalize_fixed_square_side"]
    assert normalize(1024) == 1024
    assert normalize(2048) == 2048
    assert normalize(4096) == 4096
    assert normalize(3000) == 2048
    assert normalize(3900) == 4096
    assert normalize("invalid") == 1024


def test_generate_and_edit_studio_controls_use_compact_consistent_sizes():
    source = read_source()
    assert 'clear.setMinimumWidth(76)' in source
    assert 'self.gen_prompt_edit.setMinimumHeight(48)' in source
    assert 'self.gen_prompt_edit.setMaximumHeight(58)' in source
    assert 'self.btn_generate.setMinimumHeight(36)' in source
    assert 'settings_grid.setVerticalSpacing(5)' in source
    assert 'control.setMinimumHeight(30)' in source
    gen = source[source.index('    def _build_generate_studio(self):'):source.index('    def _build_edit_studio(self):')]
    edit = source[source.index('    def _build_edit_studio(self):'):source.index('    def _gpu_policy_summary')]
    assert 'QScrollArea' not in gen and 'generateStudioScroll' not in gen
    assert 'QScrollArea' not in edit and 'editStudioSettingsScroll' not in edit
    assert 'dimensions_cell' in gen
    assert 'panel.setMinimumHeight(310)' in gen
    assert 'panel.setMinimumHeight(330)' in edit
    assert 'def _studio_field_cell(label_text: str, control: QWidget) -> QWidget:' in source
    assert 'custom = str(text) == "Custom dimensions"' in source
    assert 'self.edit_w.setEnabled(custom)' in source or 'widget.setEnabled(custom)' in source
    assert '"edit_size": self.edit_size.currentText()' in source
    assert '"edit_w": self.edit_w.value()' in source
    assert '"edit_h": self.edit_h.value()' in source

def test_qwen_output_size_is_separate_from_real_esrgan_and_saved():
    source = read_source()
    assert '"std_size_preset": "Scale"' in source
    assert '"std_output_scale": 2' in source
    assert '"std_output_width": 1024' in source
    assert '"std_output_height": 1024' in source
    assert '"std_size_preset": self.std_size_preset.currentText()' in source
    assert '"std_output_scale": int(getattr(self, "std_output_scale"' in source
    assert '"std_output_width": self.std_w.value()' in source
    assert '"std_output_height": self.std_h.value()' in source
    assert 'self.std_w.setSuffix(" px")' in source
    assert 'self.std_h.setSuffix(" px")' in source
    assert 'The output scale here runs first; if Ultimate is ON, its independent scale multiplies that result again.' in source
    assert 'Run AI Upscale' in source
    assert 'AI Upscale · shared by all workflows' not in source
    assert '"upscale_custom_w"' not in source
    assert '"upscale_custom_h"' not in source


def test_basic_pane_is_wider_and_has_no_horizontal_scrollbar():
    source = read_source()
    assert 'tabs.setMinimumWidth(0)' in source
    assert 'self.sidebar.setMinimumWidth(330)' in source
    assert 'self.sidebar.setMaximumWidth(500)' in source
    assert 'QSizePolicy.Policy.Ignored' in source
    assert 'self.main_splitter.setStretchFactor(0, 0)' in source
    assert 'self.main_splitter.setStretchFactor(1, 1)' in source
    assert 'self.workspace_splitter.setSizes([410, 1150])' in source
    assert 'basic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)' in source
    assert 'basic_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)' in source


def test_model_catalog_offers_real_esrgan_models_and_download_links():
    source = read_source()
    assert 'REALESRGAN_CATALOG: list[dict[str, str]]' in source
    assert 'Real-ESRGAN x4plus · Photo / general images' in source
    assert 'Real-ESRNet x4plus · Conservative photo restoration' in source
    assert 'Real-ESRGAN x4plus anime · Anime / illustration' in source
    assert 'AnimeVideo-v3 · Lightweight animation upscaler' in source
    assert 'Official Windows runtime' in source
    assert 'Model references' in source
    assert 'btn_download_upscaler.clicked.connect(self.start_upscaler_download)' in source
    assert 'btn_use_upscaler.clicked.connect(self.apply_upscaler_choice)' in source
    assert 'Model references' in source
    assert 'NCNN model' in source


def test_prompt_presets_are_connected_to_all_qwen_workflows():
    source = read_source()
    assert 'GEN_PROMPT_PRESETS = OrderedDict([' in source
    assert 'self.gen_prompt_preset.currentTextChanged.connect(self.apply_generate_prompt_preset)' in source
    assert 'self.edit_preset_combo.currentTextChanged.connect(self.apply_edit_preset)' in source
    assert 'def build_enhance_prompt(user_instruction: str = "")' in source
    assert 'def build_closeup_prompt(target: str, custom_prompt: Optional[str] = None' in source
    assert 'def build_square_prompt(user_instruction: str = "", enhance: bool = False)' in source
    assert 'Official prompt rewriting examples' in source
    assert 'self.std_prompt.textChanged.connect(self._std_prompt_text_edited)' in source
    assert 'self.gen_prompt_edit.textChanged.connect(self._sync_gen_prompt_preset)' in source
    assert 'def _sync_gen_prompt_preset(self):' in source
    assert 'self.schedule_save()' in source[source.index('def _sync_gen_prompt_preset'):source.index('def _sync_edit_prompt_preset')]
    assert 'self.edit_prompt_edit.textChanged.connect(self._sync_edit_prompt_preset)' in source
    assert 'def _sync_edit_prompt_preset(self):' in source
    assert 'self.schedule_save()' in source[source.index('def _sync_edit_prompt_preset'):source.index('def choose_input_dir')]


def test_qwen_size_helper_scales_enhance_and_forces_square_modes():
    source = read_source()
    tree = ast.parse(source)
    wanted = {"snap_qwen_dimension_value", "normalize_fixed_square_side", "calculate_qwen_output_size"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = {"MAX_OUTPUT_SIDE": 4096, "FIXED_SQUARE_OUTPUT_SIZES": (1024, 2048, 4096)}
    compiled = compile(ast.Module(body=nodes, type_ignores=[]), "<qwen-size-helper>", "exec")
    exec(compiled, namespace)
    calc = namespace["calculate_qwen_output_size"]
    assert calc((1000, 500), "enhance", "Scale", 1, 1024, 1024) == (1000, 500)
    assert calc((1000, 500), "enhance", "Scale", 2, 1024, 1024) == (2000, 1000)
    assert calc((1000, 500), "lora_closeup_1024", "Scale", 3, 2048, 1024) == (2048, 2048)
    assert calc((1000, 500), "square_1_1", "Custom dimensions", 2, 4096, 1536) == (4096, 4096)
    assert calc((1000, 500), "enhance", "Custom dimensions", 2, 1023, 777) == (1024, 768)

def test_input_folder_is_editable_and_populates_queue():
    source = read_source()
    assert 'self.input_dir_edit = QLineEdit' in source
    assert 'self.btn_browse_input.clicked.connect(self.choose_input_dir)' in source
    assert 'def choose_input_dir(self):' in source
    assert 'self.add_paths([folder])' in source
    assert '"input_dir": encode_saved_path(self.input_dir_edit.text().strip())' in source


def test_brand_assets_exist_and_are_passed_to_pyinstaller():
    assets = ROOT / "assets"
    for file in ("pai_logo.png", "pai_icon.png", "pai_icon.ico"):
        path = assets / file
        assert path.is_file() and path.stat().st_size > 10_000
    build = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert "assets\\pai_icon.ico" in build
    assert "assets\\pai_logo.png" in build
    assert "--add-data" in build
    # OpenCV is optional at source import time, but its Haar-cascade XML data must
    # be included in the frozen app for offline face localization to work in EXE builds.
    assert "--collect-all cv2" not in build
    assert "Qwen3-VL" in read_source()


def test_release_metadata_and_setup_files_exist():
    assert (ROOT / "build_release.ps1").is_file()
    assert (ROOT / "installer.iss").is_file()
    assert (ROOT / "requirements-build.txt").is_file()
    assert (ROOT / "build_release.bat").is_file()
    assert (ROOT / "Install_Update_Requirements.bat").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "licenses" / "THIRD_PARTY_NOTICES.txt").is_file()
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "README_FA.md").is_file()


def test_settings_accept_previous_versions_and_migrate_to_eight_step_defaults():
    source = read_source()
    from pai_studio.constants import SETTINGS_VERSION
    assert f"SETTINGS_VERSION = {SETTINGS_VERSION}" in source
    assert 'if version < 16 or version > SETTINGS_VERSION:' in source
    assert 'if version < 24:' in source
    assert 'if version < 41 and str(decoded.get("live_preview_method", "tae")).strip().lower() == "proj":' in source
    assert 'decoded["live_preview_method"] = "tae"' in source
    assert 'SETTINGS_VERSION = 44' in source
    assert 'watermark_removal_enabled' in source
    assert 'ultimate_qwen_steps' in source
    assert '"live_preview_method": "tae"' in source
    assert 'decoded["quality"] = "Turbo · 8 steps"' in source
    assert '"gen_count": self.gen_count.value()' in source
    assert '"gen_seed": self.gen_seed.value()' in source
    assert '"gen_seed_mode": self.gen_seed_mode.currentText()' in source
    assert '"edit_seed": self.edit_seed.value()' in source
    assert '"edit_seed_mode": self.edit_seed_mode.currentText()' in source
    assert 'self.gen_seed.setValue(int(self.settings.get("gen_seed", -1)))' in source
    assert 'self.edit_seed.setValue(int(self.settings.get("edit_seed", -1)))' in source
    assert 'saved_theme = str(self.settings.get("theme", "dark")).lower()' in source
    assert 'self.theme_combo.currentTextChanged.connect(self.schedule_save)' in source
    assert 'self.gen_seed_mode.currentTextChanged.connect(self.schedule_save)' in source
    assert 'self.edit_seed_mode.currentTextChanged.connect(self.schedule_save)' in source

def test_real_esrgan_upscale_is_standalone_and_local_with_scale_tile_and_tta():
    source = read_source()
    assert 'REALESRGAN_NCNN_URL' in source
    assert 'realesrgan-x4plus' in source
    assert 'realesrnet-x4plus' in source
    assert 'realesrgan-x4plus-anime' in source
    assert 'realesr-animevideov3' in source
    assert "realesrgan_native_scale" in source
    assert "self.upscale_scale_combo" not in source
    assert '"upscale_tile": 0' in source
    assert 'if bool(job.custom.get("upscale_enabled", False))' not in source[source.index('    def process(self, job: Job)'):source.index('class Worker')]
    assert 'if job.workflow == "upscale_only"' in source
    assert '"-s", str(scale), "-t", str(tile)' in source
    assert "def _realesrgan_model_assets_present" in source
    assert 'self.realesrgan_box = QGroupBox("AI Upscaler · Real-ESRGAN / Real-ESRNet")' in source
    assert 'self.realesrgan_box = QGroupBox("AI Upscaler · Real-ESRGAN / Real-ESRNet")' in source
    assert 'upscale_page.addWidget(esrgan_card)' in source
    assert 'Use manually or switch ESRGAN ON in the workflow to run after generation.' in source
    assert 'The shared controls in the top bar' not in source
    assert 'AI Upscale · shared by all workflows' not in source
    assert 'right_layout.addWidget(upscale_box, 0)' not in source
    assert 'tabs.addTab(size_tab, "Inference Size")' not in source
    assert 'tabs.addTab(upscale_tab, "AI Upscale")' not in source


def test_requirements_bat_uses_system_python_and_nvidia_driver():
    bat = (ROOT / "Install_Update_Requirements.bat").read_text(encoding="utf-8")
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    launcher = (ROOT / "Run_PAI_AI_IMAGE_STUDIO.bat").read_text(encoding="utf-8")
    assert "py -%%V" in bat
    assert "pip install --user --upgrade -r" in bat
    assert "pip check" in bat
    assert "nvidia-smi --query-gpu" in bat
    assert "does not install a second Python copy" in bat
    assert "system Python" in launcher
    assert ".venv\\Scripts\\python.exe" not in launcher
    assert "PySide6" in req and "Pillow" in req and "certifi" in req and "opencv-python-headless" not in req


def test_release_version_is_semver_beta_2():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert version == "1.0.0-beta.2"
    assert f'APP_VERSION = "{version}"' in read_source()
    installer = (ROOT / "installer.iss").read_text(encoding="utf-8")
    assert f'#define AppVersion "{version}"' in installer
    assert "OutputBaseFilename=PAI-AI-Image-Studio-Setup-{#AppVersion}" in installer
    build_script = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert '"/DAppVersion=$AppVersion"' in build_script
    assert f"`{version}`" in (ROOT / "README.md").read_text(encoding="utf-8")



def test_product_branding_and_generate_edit_skin_are_present():
    source = read_source()
    assert 'APP_NAME = "P.AI AI IMAGE STUDIO"' in source
    assert 'DISPLAY_VERSION = "Version 1 Beta"' in source
    assert 'studio_skin = str(theme).lower() == "light"' in source
    assert 'self.preview.set_studio_skin(studio_skin)' in source
    installer = (ROOT / "installer.iss").read_text(encoding="utf-8")
    assert '#define AppName "P.AI AI IMAGE STUDIO"' in installer
    assert 'AppVerName={#AppVerName}' in installer

def test_pinned_cuda_runtime_is_required_and_bundled():
    source = read_source()
    release_script = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert "def _ensure_cuda_runtime_payload" in source
    assert "cudart-sd-bin-win-cu12-x64.zip" in source
    assert "ggml-cuda.dll" in source
    for dll in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
        assert dll in source
        assert dll in release_script
    assert "Download-CudaRuntime" in release_script
    assert "CPU-only runtime" in release_script


def test_package_zip_layout_after_build_includes_no_generated_executable_claim():
    # This test deliberately checks project source artifacts only; a Windows EXE is built on Windows.
    assert (ROOT / "assets" / "pai_logo.png").is_file()
    assert (ROOT / "build_release.bat").is_file()


# Regression tests for fixes requested in Pasted text(2).txt.
def test_worker_logs_are_delivered_through_a_gui_thread_signal():
    source = read_source()
    assert 'log_signal = Signal(str)' in source
    assert 'self.log_signal.connect(self._append_log_message)' in source
    assert 'self.log_signal.emit(f"[{stamp}] {message}")' in source
    log_block = source[source.index('    def _append_log_message'):source.index('    def status_message', source.index('    def _append_log_message'))]
    assert 'QTimer.singleShot' not in log_block
    assert 'self.log_edit.appendPlainText(formatted_message)' in log_block


def test_model_load_clears_previous_cancel_flag_and_handles_cancelled_signal():
    source = read_source()
    load_method = source[source.index('    def load_model_async'):source.index('    def _model_loaded', source.index('    def load_model_async'))]
    assert 'self.engine.force_stop.clear()' in load_method
    assert 'self.engine.stop_after_current.clear()' in load_method
    assert 'self.model_worker.cancelled.connect(self._model_load_cancelled)' in load_method
    assert 'def _model_load_cancelled(self):' in source
    assert 'self.progress.setRange(0, 100)' in source[source.index('    def _model_load_cancelled'):source.index('    def unload_local_model')]


def test_preview_caches_pixmaps_and_difference_and_flicker_uses_timer():
    source = read_source()
    preview = source[source.index('class PreviewCanvas(QWidget):'):source.index('class App(QMainWindow):')]
    assert 'self._pix_original: Optional[QPixmap] = None' in preview
    assert 'self._pix_enhanced: Optional[QPixmap] = None' in preview
    assert 'self._pix_diff: Optional[QPixmap] = None' in preview
    assert 'self._flicker_timer.setInterval(500)' in preview
    paint = preview[preview.index('    def paintEvent'):preview.index('    def _zoom_about')]
    assert 'self._pixmap(' not in paint
    assert 'self.update()' not in paint
    assert 'self._pix_diff = self._pixmap' in preview
    assert 'self._flicker_timer.timeout.connect(self._on_flicker_tick)' in preview


def test_square_feather_ramp_does_not_overflow_255():
    source = read_source()
    block = source[source.index('def build_square_feather_mask'):source.index('class Aborted')]
    assert '127.5 * (1.0 - math.cos(math.pi * i / max(1, feather - 1)))))' in block
    assert ')))) * 2)' not in block


def test_workspace_queue_marks_old_outputs_completed_not_runnable():
    source = read_source()
    block = source[source.index('    def _populate_queue_from_workspace'):source.index('    def refresh_queue_table', source.index('    def _populate_queue_from_workspace'))]
    assert 'for p in input_root.iterdir()' in block
    assert 'status="Queued", is_output=False' in block
    assert 'for folder in (GENERATED_DIR, EDITED_DIR, OUTPUT_DIR, CLOSEUPS_DIR, OUTPAINT_DIR)' in block
    assert 'status="Completed", is_output=True' in block
    assert 'for item in self.queue:' not in source[source.index('        remaining = job.custom.get("queue_remaining", [])', source.index('    def job_completed'))-300:source.index('    def job_failed')]


def test_seed_modes_have_explicit_random_fixed_and_increment_semantics():
    source = read_source()
    process = source[source.index('    def process(self, job: Job)'):source.index('\n\nclass Worker(QThread):')]
    assert 'if seed_mode == "Random":' in process
    assert 'random.randint(0, 2**31 - 1)' in process
    assert 'if seed_mode == "Increment":' in process
    assert 'return int(base_seed) + index' in process
    assert 'current_seed = seed_for(index)' in process
    assert 'edit_seed = seed_for(0)' in process
    assert 'square_seed = seed_for(0)' in process


def test_queue_status_uses_completed_consistently_and_outputs_are_marked_output():
    source = read_source()
    assert 'self._add_queue_item(str(path), status="Completed", is_output=True)' in source
    assert 'item["status"] = "Done"' not in source


def test_square_feather_mask_ramp_runtime_values_are_gradual():
    source = read_source()
    function_and_next = source[source.index('def build_square_feather_mask'):source.index('class Aborted')]
    namespace = {"Image": Image, "math": math}
    exec(function_and_next, namespace)
    mask = namespace["build_square_feather_mask"]((12, 12), (2, 2, 10, 10), 4)
    # Values progress along the feather instead of clipping most of it to white.
    assert mask.getpixel((2, 5)) == 0
    assert 50 <= mask.getpixel((3, 5)) <= 80
    assert 170 <= mask.getpixel((4, 5)) <= 210
    assert mask.getpixel((5, 5)) == 255


def test_requested_workflow_layout_places_settings_full_height_and_queue_log_side_by_side():
    source = read_source()
    assert 'queue_below_preview = workflow in ("enhance", "lora_closeup_1024", "square_1_1")' in source
    assert 'self.workspace_splitter.insertWidget(0, self.sidebar)' in source
    assert 'self.right_vertical_splitter.insertWidget(1, self.workflow_bottom_row)' in source
    assert 'self.workflow_bottom_row.addWidget(panel)' in source
    assert 'self.workspace_content_splitter.addWidget(self.workspace_brand_group)' in source
    assert 'QGroupBox("P.AI AI IMAGE STUDIO")' in source
    assert 'self.workspace_lock_group' not in source
    assert 'self.btn_workspace_lock.setCheckable(True)' not in source
    assert 'self.right_vertical_splitter.setHandleWidth(6)' in source
    assert 'self.left_dock_splitter.setHandleWidth(6)' in source
    assert 'self.workspace_content_splitter.setHandleWidth(6)' in source
    assert 'self.workflow_queue_actions = QGroupBox("Queue controls")' in source
    assert 'self.btn_run.clicked.connect(self.run_selected)' in source
    assert 'self.btn_run_all.clicked.connect(self.run_all_queued)' in source
    assert 'self.btn_stop_now.clicked.connect(self.stop_job)' in source
    assert 'workflow_row.addWidget(button, 1)' in source



def test_theme_toggle_is_one_click_at_top_right_and_not_in_advanced_tab():
    source = read_source()
    build = source[source.index("    def _build(self):"):source.index("    def _build_sidebar(self):")]
    sidebar = source[source.index("    def _build_sidebar(self):"):source.index("    def _build_queue_panel(self):")]
    theme = source[source.index("    def toggle_theme(self):"):source.index("    def _append_log_message(self, formatted_message: str):")]
    assert 'self.btn_theme_toggle.clicked.connect(self.toggle_theme)' in build
    assert 'top.addWidget(self.btn_theme_toggle)' in build
    assert 'self.btn_theme_toggle.setText("☾  Dark mode" if is_light else "☀  Light mode")' in theme
    assert 'self.theme_combo.hide()' in sidebar
    assert 'self.theme_combo.currentTextChanged.connect(self.schedule_save)' in sidebar
    assert 'al.addRow("Theme", self.theme_combo)' not in sidebar

def test_identity_lock_is_idempotent_and_can_be_disabled():
    tree = ast.parse(read_source())
    names = {"IDENTITY_LOCK_PROMPT", "apply_identity_lock"}
    nodes = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
             and ((isinstance(node, ast.FunctionDef) and node.name in names)
                  or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    apply_lock = namespace["apply_identity_lock"]
    base = "Improve fine details without changing the composition."
    locked = apply_lock(base, True)
    assert base in locked and namespace["IDENTITY_LOCK_PROMPT"] in locked
    assert apply_lock(locked, True) == locked
    assert apply_lock(base, False) == base


def test_gpu_first_policy_uses_capacity_aware_text_encoder_fallback():
    source = read_source()
    assert 'new_memory_mode not in {"auto", "full_gpu", "low_vram"}' in source
    assert '"GPU-only · no CPU inference", "full_gpu"' in source
    assert 'def plan_qwen_memory_placement(' in source
    assert 'free_vram_gb < component_file_gb + reserve_gb' in source
    assert 'text_backend = "cpu" if (text_encoder_on_cpu or force_cpu_te)' in source
    assert 'auto_fit_mode = "off" if (' in source
    assert 'if offload_to_cpu:' in source and 'args.append("--offload-to-cpu")' in source
    assert '"--auto-fit", auto_fit_mode' in source
    assert 'auto_fit={auto_fit_mode}' in source
    assert 'self.encoder_cpu_check.isChecked() and str(self.gpu_memory_mode_combo.currentData() or "auto") != "full_gpu"' in source
    assert 'self.encoder_cpu_check.setEnabled(not full_gpu)' in source
    assert 'not force_cpu_te_for_capacity and free_vram_gb > 0 and free_vram_gb < estimated_weight_gb + 1.5' in source
    assert "def _gpu_policy_summary(self, gpu: dict[str, Any]) -> str:" in source
    assert "GPU-only assigns the text encoder, vision projector, diffusion and VAE to CUDA" in source
    assert "disables auto-fit and CPU weight offload" in source


def test_model_autodiscovery_prefers_turbo_encoder_and_matching_projector():
    source = read_source()
    curated = '"qwen-image-2.1-turbo-uncensored" in n'
    generic = '"qwen3vl-8b-instruct-q4_k_m" in n'
    assert curated in source and generic in source
    assert source.index(curated) < source.index(generic)
    assert '"mmproj-qwen-image-2.1-turbo-uncensored" in n' in source
    assert '"mmproj-qwen3vl-8b-instruct" in n' in source


def test_model_role_guard_rejects_qwen3vl_as_diffusion_but_accepts_it_as_text_encoder():
    tree = ast.parse(read_source())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_model_role_mismatch")
    namespace = {}
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    mismatch = namespace["_model_role_mismatch"]
    assert "text encoder" in mismatch("diffusion", "qwen3vl_8b_int8_convrot.safetensors").lower()
    assert mismatch("text_encoder", "qwen3vl_8b_int8_convrot.safetensors") == ""
    assert "vision projector" in mismatch("vision_projector", "qwen3vl_8b_int8_convrot.safetensors").lower()
    assert mismatch("diffusion", "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf") == ""
    assert mismatch("text_encoder", "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf")


def test_model_role_guard_is_used_when_loading_and_browsing_local_files():
    source = read_source()
    assert "mismatch = _model_role_mismatch(role, candidate.name)" in source
    assert "_model_role_mismatch(role, p.name)" in source
    assert "Qwen3-VL/text encoder" in source
    assert "profile_mismatches[role] = mismatch" in source
    assert "Wrong model component" in source


def test_qwen3vl_safetensors_uses_embedded_vision_without_external_mmproj():
    tree = ast.parse(read_source())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_text_encoder_embeds_vision")
    namespace = {"Path": Path}
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    has_embedded = namespace["_text_encoder_embeds_vision"]
    assert has_embedded("qwen3vl_8b_int8_convrot.safetensors")
    assert has_embedded("Qwen3-VL-8B-Instruct-BF16.safetensors")
    assert not has_embedded("Qwen3VL-8B-Instruct-Q4_K_M.gguf")
    assert not has_embedded("mmproj-Qwen3VL-8B-Instruct-F16.gguf")


def test_image_edit_accepts_embedded_vision_safetensors_without_external_projector():
    source = read_source()
    assert "has_embedded_vision = _text_encoder_embeds_vision(llm_path)" in source
    assert "and not has_embedded_vision" in source
    assert "embedded in Qwen3-VL safetensors" in source


def test_preview_reports_source_and_output_dimensions():
    source = read_source()
    assert 'source_resolution = f"{original.width}×{original.height}"' in source
    assert 'resolution = f"Source {source_resolution} → Output {output_resolution}' in source
    assert 'output_resolution = f"{image.width}×{image.height}"' in source


def test_prompt_templates_follow_specific_change_and_preserve_other_content():
    source = read_source()
    assert 'Change only what is requested.' in source
    assert 'Keep all unmentioned content, framing, perspective, lighting, and background unchanged.' in source
    assert 'Reconstruct this photograph with Qwen image-to-image as a careful photographic restoration and re-render' in source
    assert 'Never merely enlarge the original' in source
    assert 'natural skin microtexture' in source
    assert 'Remove random luminance/chroma noise' in source
    assert 'wrinkles, acne, folds or unsupported patterns' in source
    assert 'Start with the requested change.' in source
    assert 'For visible text, quote the exact wording and specify its placement and typography.' in source


def test_custom_qwen_output_dimensions_snap_to_multiples_of_32():
    source = read_source()
    assert "def _snap_qwen_dimension(spin: QSpinBox):" in source
    assert "round(value / 32.0)" in source and "* 32" in source
    assert "self.std_w.editingFinished.connect" in source
    assert "self.std_h.editingFinished.connect" in source


def test_enhance_uses_balanced_img2img_and_keeps_reference_conditioning():
    source = read_source()
    assert 'ENHANCE_IMG2IMG_STRENGTH = 0.46' in source
    assert 'ENHANCE_TILE_IMG2IMG_STRENGTH = 0.38' in source
    body = source[source.index('def enhance_once('):source.index('def _enhance_with_retries(')]
    assert 'prompt, source, (width, height), seed, quality, steps' in body
    assert 'init_image=source' in body
    assert 'strength=strength' in body
    assert 'ENHANCE_IMG2IMG_STRENGTH if strength is None' in body
    assert 'source used as reference + init image' in body
    assert 'use_img2img: bool = True' in body
    retry_body = source[source.index('def _enhance_with_retries('):source.index('def _tile_starts(')]
    assert 'preserve_source_structure_with_ai_detail' not in retry_body
    assert 'return best' in retry_body


def test_detail_refinement_uses_img2img_and_preserves_square_reference_mode():
    source = read_source()
    body = source[source.index('def _qwen_detail_refine_large_output('):source.index('def _enhance_qwen_output(')]
    assert 'init_image=tile' in body
    assert 'strength=effective_tile_strength' in body
    assert 'if use_img2img:' in body
    assert 'Compatibility path for callers explicitly requesting reference-only refinement.' not in body
    assert 'Current user-facing Qwen profiles use Img2Img with a selected strength.' not in body
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    assert 'use_img2img=True' in process
    assert 'tile_strength=profile_tile_strength' in process
    closeup = process[process.index('elif job.workflow == "lora_closeup_1024"'):process.index('elif job.workflow == "square_1_1"')]
    assert 'cropped, effective_prompt, side, side, closeup_seed' in closeup
    assert 'raw_crop, effective_prompt' not in closeup
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    assert 'square_output_prompt' in square
    assert 'result, square_output_prompt, side, side, square_seed + 1000' in square

def test_enhance_large_output_does_not_mutate_requested_dimensions():
    source = read_source()
    body = source[source.index('def _enhance_qwen_output('):source.index('def process(')]
    assert 'stage_width, stage_height = int(width), int(height)' in body and 'requested_width = min(MAX_OUTPUT_SIDE' in body
    assert 'width, height = self._inference_size((width, height))' not in body
    assert 'final file will retain the requested' in body
    assert 'result.resize((requested_width, requested_height)' in body


def test_enhance_large_output_runtime_invariant_keeps_requested_dimensions():
    source = read_source()
    tree = ast.parse(source)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocalEngine")
    method_node = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "_enhance_qwen_output")
    namespace = {"Image": Image, "MAX_INFERENCE_SIDE": 2048, "MAX_OUTPUT_SIDE": 8192, "Optional": __import__('typing').Optional}
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), "<enhance-size-regression>", "exec"), namespace)

    class FakeEngine:
        _enhance_qwen_output = namespace["_enhance_qwen_output"]
        _publish_live_preview = lambda self, image, marker=None: None

        def _inference_size(self, size):
            w, h = size
            scale = min(1.0, 2048 / max(w, h))
            return max(64, int(round(w * scale / 32) * 32)), max(64, int(round(h * scale / 32) * 32))

        def _enhance_with_retries(self, original, prompt, width, height, *args, **kwargs):
            # Simulate a native pass returning only its capped dimensions.
            return Image.new("RGB", (2048, 1024))

        def log(self, message):
            pass

    engine = FakeEngine()
    result = engine._enhance_qwen_output(Image.new("RGB", (1024, 512)), "prompt", 4096, 2048,
                                         123, "quality", 1, 25.0, 18.0, 8, "fast")
    assert result.size == (4096, 2048)


def test_closeup_grounding_releases_diffusion_server_before_gpu_locator():
    source = read_source()
    body = source[source.index('elif job.workflow == "lora_closeup_1024"'):source.index('elif job.workflow == "square_1_1"')]
    assert 'self.unload_model()' in body
    assert 'stopping the image diffusion server first to release GPU memory' in body


def test_closeup_zoom_factor_is_not_overridden_by_context_padding():
    source = read_source()
    body = source[source.index('def crop_closeup_from_bbox('):source.index('def build_edit_prompt(')]
    assert 'minimum_target_crop = target_extent * 1.12' in body
    assert 'crop_side = int(round(max(minimum_target_crop, zoom_crop)))' in body


def test_square_enhance_preset_explicitly_prompts_for_outpainting():
    source = read_source()
    process = source[source.index('    def process(self, job: Job)'):source.index('class Worker(QThread)')]
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    # Base enhancement works on the fitted core; Ultimate is applied only after the complete square result exists.
    copy_crop = square[square.index('if mode in ("copy", "crop"):'):square.index('else:', square.index('if mode in ("copy", "crop"):'))]
    assert 'detail_mode="fast"' in copy_crop
    assert 'tile_strength=profile_tile_strength, detail_profile=profile_key' in copy_crop
    assert "SQUARE_ENHANCE_OUTPAINT_PROMPT = (" in source
    assert "Extend the original scene naturally to the requested square canvas" in source
    assert "build_square_outpaint_inputs" in source
    assert 'mask_image=_full_mask, strength=0.94' in source
    assert 'joint_prompt, source_original, (side, side)' in source
    assert 'JOINT OUTPAINT PASS' in source
    assert 'Square 1:1 edge plan (offline): ' in source
    assert '("Square 1:1 · Enhance + Outpaint", SQUARE_ENHANCE_OUTPAINT_PROMPT)' in source
    assert 'Ultimate Qwen refinement is running on the completed final square output' in source


def test_input_buttons_use_full_labels_and_expand_without_clipping():
    source = read_source()
    assert 'self.btn_browse_input = QPushButton("Browse…")' in source
    assert 'self.btn_browse_input.setMinimumWidth(0)' in source
    assert 'self.btn_add_input.setMinimumWidth(0)' in source
    assert 'self.btn_browse_output = QPushButton("Browse…")' in source
    assert 'QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed' in source


def test_prompt_templates_are_filtered_by_workflow_and_manual_text_is_preserved():
    source = read_source()
    assert 'def refresh_std_prompt_presets(self, workflow: str, initial: bool = False)' in source
    assert 'self.refresh_std_prompt_presets(workflow)' in source
    assert 'WORKFLOW_PROMPT_PRESETS = {' in source
    assert '"enhance": (' in source
    assert '"lora_closeup_1024": (' in source
    assert '"square_1_1": (' in source
    assert 'allowed = list(WORKFLOW_PROMPT_PRESETS[workflow])' in source
    assert 'Square 1:1 · Outpaint only (No Enhance)' in source
    assert 'Square 1:1 · Enhance + Outpaint' in source
    assert 'self.std_prompt.textChanged.connect(self._std_prompt_text_edited)' in source
    assert 'self._active_std_prompt_preset = "Custom instruction (manual)"' in source
    assert 'saved_template_matches' in source
    assert 'Preserve an actual user-authored instruction across workflow switches.' in source
    assert 'current_name in {"Square 1:1 · Outpaint only (No Enhance)", "Square 1:1 · Enhance + Outpaint"}' in source


def test_square_prompt_templates_are_idempotent_and_obey_enhance_toggle():
    source = read_source()
    assert 'instruction in (SQUARE_OUTPAINT_ONLY_PROMPT, SQUARE_ENHANCE_OUTPAINT_PROMPT)' in source
    assert 'return SQUARE_ENHANCE_OUTPAINT_PROMPT if enhance else SQUARE_OUTPAINT_ONLY_PROMPT' in source
    assert 'if not instruction or instruction == FAITHFUL_ENHANCE_PROMPT:' in source





def test_generate_studio_uses_non_scrollable_compact_rows_and_persists_dimensions():
    source = read_source()
    gen = source[source.index('    def _build_generate_studio(self):'):source.index('    def _build_edit_studio(self):')]
    edit = source[source.index('    def _build_edit_studio(self):'):source.index('    def _gpu_policy_summary')]
    assert 'QScrollArea' not in gen
    assert 'generateStudioScroll' not in gen
    assert 'settings_grid.setColumnStretch(column, 1)' in gen
    assert 'settings_grid.addWidget(cell, row, column)' in gen
    assert 'self.gen_prompt_edit.setMinimumHeight(48)' in gen
    assert 'self.gen_prompt_edit.setMaximumHeight(58)' in gen
    assert 'dimensions_cell' in gen
    assert 'QScrollArea' not in edit
    assert 'editStudioSettingsScroll' not in edit
    assert 'right_content.addLayout(settings_grid)' in edit
    assert 'root.addLayout(content_layout, 1)' in edit
    assert '"gen_w": self.gen_w.value()' in source
    assert '"gen_h": self.gen_h.value()' in source
    assert '"gen_size": self.gen_size.currentText()' in source
    assert 'self.gen_w.editingFinished.connect(lambda: self._commit_gen_dimensions("width"))' in source
    assert 'self.gen_h.editingFinished.connect(lambda: self._commit_gen_dimensions("height"))' in source
    assert 'self.gen_size.setCurrentText(matched)' in source

def test_all_workflows_share_the_selected_studio_theme():
    source = read_source()
    theme = source[source.index("    def _apply_theme(self, theme: str):"):source.index("    def _append_log_message(self, formatted_message: str):")]
    assert 'studio_skin = str(theme).lower() == "light"' in theme
    assert 'self.preview.set_studio_skin(studio_skin)' in theme
    assert '#7858f5' in theme and '#927bf7' in theme
    assert 'deliberately scoped to Image' not in theme


def test_removed_workspace_lock_button_does_not_break_job_creation():
    source = read_source()
    assert "def _identity_lock_enabled(self) -> bool:" in source
    assert '"identity_lock": self._identity_lock_enabled()' in source
    assert '"identity_lock_enabled": self._identity_lock_enabled()' in source


# Regression checks for portable packaging and prompt/model preset routing.
def test_all_app_owned_settings_and_default_data_are_project_local():
    source = read_source()
    assert 'USER_DATA_ROOT = APP_DIR / "data"' in source
    assert 'SETTINGS_FILE = DATA_ROOT / "settings.json"' in source
    assert 'MODEL_DIR = DATA_ROOT / "models" / "Qwen-Image-2.1"' in source
    assert 'CACHE_DIR = WORKSPACE_DIRS["cache"]' in source
    assert 'LEGACY_USER_DATA_ROOT' not in source
    assert 'LEGACY_ROAMING_ROOT' not in source
    assert 'QSettings' not in source


def test_model_preset_control_is_in_advanced_and_not_basic():
    source = read_source()
    assert 'al.addRow(self.model_preset_group)' in source
    basic = source[source.index('    def _build_sidebar(self):'):source.index('    def _build_queue_panel(self):')]
    assert 'bl.addWidget(self.model_preset_group)' not in basic
    assert 'self.model_preset_group.setVisible(qwen_workflow)' in source
    assert 'self.std_quality.currentText()' in source
    assert 'self.steps_spin' in source


def test_prompt_template_handlers_replace_prompts_and_keep_presets_synchronized():
    source = read_source()
    assert 'self.std_prompt_preset.currentTextChanged.connect(self.apply_std_prompt_preset)' in source
    assert 'self.gen_prompt_preset.currentTextChanged.connect(self.apply_generate_prompt_preset)' in source
    assert 'self.edit_preset_combo.currentTextChanged.connect(self.apply_edit_preset)' in source
    assert 'self.std_prompt.setPlainText(PROMPT_PRESETS.get(name, ""))' in source
    assert 'self.gen_prompt_edit.setPlainText(GEN_PROMPT_PRESETS.get(name, ""))' in source
    assert 'self.edit_prompt_edit.setPlainText(EDIT_PRESETS[name])' in source
    assert 'self.std_prompt_preset.currentText() if hasattr(self, "std_prompt_preset")' in source
    assert 'self.gen_prompt_preset.currentText() if hasattr(self, "gen_prompt_preset")' in source
    assert 'self.edit_preset_combo.currentText() if hasattr(self, "edit_preset_combo")' in source


def test_real_esrgan_catalog_has_official_pytorch_references_and_runtime_mirror():
    source = read_source()
    assert 'REALESRGAN_NCNN_URLS = (' in source
    assert 'https://sourceforge.net/projects/real-esrgan.mirror/files/v0.2.5.0/' in source
    for url_fragment in (
        'RealESRGAN_x4plus.pth',
        'RealESRNet_x4plus.pth',
        'RealESRGAN_x4plus_anime_6B.pth',
        'realesr-animevideov3.pth',
    ):
        assert url_fragment in source
    assert 'All app-owned settings and default data' not in source or 'USER_DATA_ROOT = APP_DIR / "data"' in source


def test_native_build_downloads_have_primary_and_mirror_sources_in_local_cache():
    release_script = (ROOT / 'build_release.ps1').read_text(encoding='utf-8')
    assert "$CacheRoot = Join-Path $ProjectRoot '.build-cache\\stable-diffusion-cpp-master-964-1d0a58a'" in release_script
    assert 'sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/' in release_script
    assert 'Download-Verified @($BinaryUrl, $BinaryMirrorUrl) $BinaryZip $BinarySha' in release_script
    assert 'Download-CudaRuntime @($CudaRuntimeUrl, $CudaRuntimeMirrorUrl) $CudaRuntimeZip' in release_script
    assert '$script:CudaRuntimeSourceUrl' in release_script


def test_download_helpers_retry_hf_mirror_without_relaxing_pinned_hashes():
    source = read_source()
    assert 'def model_download_sources(meta: dict[str, Any]) -> tuple[str, ...]:' in source
    assert 'hf-mirror.com' in source
    assert 'SHA-256 verification failed' in source
    assert 'Never resume a partial response from one host against another mirror' in source
    assert 'install only if that source independently matches the same pin' in source
    assert 'if source_index < len(source_urls) - 1:' in source
    assert 'REALESRGAN_NCNN_URLS' in source
    assert 'archive.testzip()' not in source  # zip validation is performed via zipfile.ZipFile.testzip()
    assert 'zf.testzip()' in source


def test_github_runner_script_and_manifest_docs_are_present():
    assert (ROOT / 'Run_PAI_AI_IMAGE_STUDIO.bat').is_file()
    assert (ROOT / '.gitignore').is_file()
    assert (ROOT / 'README.md').is_file()
    assert (ROOT / 'README_FA.md').is_file()
    assert (ROOT / 'pai_studio' / 'validation.py').is_file()
    assert (ROOT / '.github' / 'workflows' / 'tests.yml').is_file()


def test_std_prompt_restore_recognizes_exact_template_when_saved_preset_name_is_missing():
    source = read_source()
    refresh = source[source.index("def refresh_std_prompt_presets"):source.index("def update_closeup_controls")]
    assert "matched_allowed = next((" in refresh
    assert 'current_text == str(PROMPT_PRESETS.get(name, "")).strip()' in refresh
    assert refresh.index("elif matched_allowed:") < refresh.index("elif is_custom_text:")


def test_launcher_and_setup_use_system_python_without_project_bootstrap():
    root = ROOT
    setup = (root / "Install_Update_Requirements.bat").read_text(encoding="utf-8")
    launcher = (root / "Run_PAI_AI_IMAGE_STUDIO.bat").read_text(encoding="utf-8")
    build = (root / "build_release.ps1").read_text(encoding="utf-8")
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "py -%%V" in setup and "pip install --user" in setup
    assert "does not install a second Python copy" in setup
    assert "Python 3.11, 3.12 or 3.13 x64" in setup
    assert "system Python" in launcher
    assert "Get-Command python" in build
    assert ".system-python-path.txt" in ignore
    assert not (root / "Install_Local_Python.ps1").exists()
    assert ".python\\python.exe" not in setup
    assert ".venv\\Scripts\\python.exe" not in launcher


def test_advanced_model_preset_is_shared_by_enhance_scale_closeup_and_square():
    source = read_source()
    advanced = source[source.index("advanced = QWidget()"):source.index("# Force the Basic tab content")]
    layout = source[source.index("def _switch_workflow_layout"):source.index("def apply_workflow_preset")]
    assert 'QGroupBox("Advanced model preset · Qwen Image 2.1")' in source
    assert "al.addRow(self.model_preset_group)" in advanced
    assert 'qwen_workflow = queue_below_preview' in layout
    assert 'self.model_preset_group.setVisible(qwen_workflow)' in layout
    assert '"enhance", "lora_closeup_1024", "square_1_1"' in layout


def test_prompt_presets_are_replaced_and_jobs_read_the_correct_editor():
    source = read_source()
    assert 'self.std_prompt_preset.currentTextChanged.connect(self.apply_std_prompt_preset)' in source
    assert 'self.gen_prompt_preset.currentTextChanged.connect(self.apply_generate_prompt_preset)' in source
    assert 'self.edit_preset_combo.currentTextChanged.connect(self.apply_edit_preset)' in source
    assert 'self.std_prompt.setPlainText(PROMPT_PRESETS.get(name, ""))' in source
    assert 'self.gen_prompt_edit.setPlainText(GEN_PROMPT_PRESETS.get(name, ""))' in source
    assert 'self.edit_prompt_edit.setPlainText(EDIT_PRESETS[name])' in source
    jobs = source[source.index('def _job_from_ui'):source.index('def generate_images')]
    assert 'prompt=self.gen_prompt_edit.toPlainText().strip()' in jobs
    assert 'build_edit_prompt(self.edit_prompt_edit.toPlainText()' in jobs
    assert 'prompt = self.std_prompt.toPlainText().strip()' in jobs


def test_download_sources_use_official_urls_and_fallbacks_with_integrity_validation():
    source = read_source()
    assert "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/" in source and "realesrgan-ncnn-vulkan-20220424-windows.zip" in source
    assert "sourceforge.net/projects/real-esrgan.mirror/files/v0.2.5.0/" in source and "realesrgan-ncnn-vulkan-20220424-windows.zip/download" in source
    assert 'https://github.com/leejet/stable-diffusion.cpp/releases/download/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip' in (ROOT / 'build_release.ps1').read_text(encoding='utf-8')
    assert 'https://sourceforge.net/projects/stable-diffusion-cpp.mirror/files/master-964-1d0a58a/sd-master-1d0a58a-bin-win-cuda12-x64.zip/download' in (ROOT / 'build_release.ps1').read_text(encoding='utf-8')
    assert 'def model_download_sources(meta: dict[str, Any])' in source
    assert 'hf-mirror.com' in source
    assert 'zipfile.ZipFile(path, "r")' in source or "zipfile.ZipFile(path, 'r')" in source
    assert 'SHA-256 verification failed for {key}' in source
    assert 'REALESRGAN_MODEL_ZOO_URL' in source

def test_qwen3vl_gguf_mmproj_is_allowed_in_vision_projector_slot():
    source = read_source()
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_model_role_mismatch")
    namespace = {}
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    assert namespace["_model_role_mismatch"](
        "vision_projector", "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
    ) == ""



def test_qwen_image_uncensored_gguf_projector_is_accepted():
    source = read_source()
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_model_role_mismatch")
    namespace = {}
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    assert namespace["_model_role_mismatch"](
        "vision_projector", "mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf"
    ) == ""


def test_native_engine_reuses_existing_runtime_from_previous_pai_download_folder():
    source = read_source()
    runtime_method = source[source.index("    def _runtime_executable(self)"):source.index("    def _model_profile_signature", source.index("    def _runtime_executable(self)"))]
    assert 'search_roots: list[Path] = [APP_DIR.parent, APP_DIR.parent.parent]' in runtime_method
    assert 'downloads_root = Path.home() / "Downloads"' in runtime_method
    assert 'project_root.glob(f"**/data/runtime/{runtime_tag}/sd-server.exe")' in runtime_method
    assert 'skipping sd-master-1d0a58a-bin-win-cuda12-x64.zip' in runtime_method
    assert 'self._ensure_cuda_runtime_payload(payload_root)' in runtime_method


def test_enhance_defaults_accept_first_result_without_repeated_full_passes():
    source = read_source()
    assert '"retry_diff": 25.0' in source
    assert '"warn_diff": 18.0' in source
    assert '"max_attempts": 1' in source
    assert 'Max attempts (full reruns)' in source
    assert 'Accept result when difference ≤' in source
    assert 'if diff <= retry_diff:' in source
    assert 'Warning: best structural difference is' in source


def test_native_text_conditioning_cache_is_enabled_and_encoder_can_be_cpu_resident():
    source = read_source()
    assert '"--conditioning-cache-size", "8"' in source
    assert 'text conditioning LRU cache=8 prompts' in source
    assert 'force_cpu_te_for_capacity' in source
    assert 'text_backend == "cpu"' in source
    assert '"force_cpu_te_for_capacity": force_cpu_te' in source
    assert '"text_backend": text_backend' in source
    assert '"auto_fit_mode": auto_fit_mode' in source


def test_closeup_target_description_is_passed_as_open_vocabulary_to_qwen3vl():
    source = read_source()
    assert 'Target description, to interpret literally: {target_text}' in source
    assert 'interpret the complete natural-language description semantically' in source
    assert 'target-region heuristic' not in source
    assert 'generic_person_terms' not in source
    assert 'estimate_closeup_focus_point' not in source

def test_square_composite_preserves_source_core_until_output_enhance_and_enhances_completed_canvas():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'composite_square_outpaint_result')
    namespace = {'Image': Image, 'ImageOps': ImageOps, 'np': __import__('numpy')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<square-composite>', 'exec'), namespace)
    original = Image.new('RGB', (80, 120), (90, 120, 150))
    generated = Image.new('RGB', (120, 120), (0, 0, 0))
    output = namespace['composite_square_outpaint_result'](original, generated, 120, (20, 0, 100, 120), False, 12)
    assert output.size == (120, 120)
    assert output.getpixel((60, 60)) == (90, 120, 150)
    assert output.getpixel((0, 60)) == (0, 0, 0)
    enhanced = namespace['composite_square_outpaint_result'](original, generated, 120, (20, 0, 100, 120), True, 12)
    assert enhanced.getpixel((60, 60)) == (0, 0, 0)
    assert output.getpixel((60, 60)) != enhanced.getpixel((60, 60))
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    assert 'if job.enhance:' in square and 'square_output_prompt' in square
    assert 'result, square_output_prompt, side, side, square_seed + 1000' in square
    assert 'source_core, core_prompt, source_core.width, source_core.height' not in square
    assert 'version < 28' in source and 'decoded["gpu_memory_mode"] = "auto"' in source

def test_ultimate_style_qwen_four_tile_refinement_is_toggleable_and_persisted():
    source = read_source()
    assert '"enhance_detail_mode": "fast"' in source
    assert '"ultimate_qwen_upscale_enabled": False' in source
    assert 'QGroupBox("Ultimate · Qwen tile refinement")' in source
    assert 'QCheckBox("Enable tiled detail refinement")' in source
    assert 'self.ultimate_qwen_upscale_check.toggled.connect' in source
    assert 'self.ultimate_qwen_upscale_check.isChecked()' in source
    detail = source[source.index('    def _qwen_detail_refine_large_output'):source.index('    def _enhance_qwen_output')]
    assert 'detail_mode == "ultimate"' in detail
    assert 'getattr(self, "_active_ultimate_denoise_strength", 0.20)' in detail
    assert 'x_starts = [0, max(0, width - tile_w)] if width > 1 else [0]' in detail
    assert 'y_starts = [0, max(0, height - tile_h)] if height > 1 else [0]' in detail
    assert 'getattr(self, "_active_ultimate_tile_coverage", 0.50)' in detail
    assert 'getattr(self, "_active_ultimate_refinement_passes", 1)' in detail
    assert '"ultimate_qwen_passes": self.ultimate_qwen_passes_spin.value()' in source
    assert '"ultimate_qwen_passes", 1' in source
    assert 'self._active_ultimate_refinement_passes = max(1, min(2, int(job.custom.get("ultimate_qwen_passes", 1))))' in source
    assert 'ultimate_qwen_steps' in source and 'ultimate_qwen_steps_spin' in source
    assert 'tile_quality = f"Ultimate · {tile_steps} steps"' in detail
    assert 'self.settings["enhance_detail_mode"] = "ultimate" if enabled else "fast"' in source
    assert 'if version < 38:' in source and 'decoded["ultimate_qwen_upscale_enabled"] = False' in source
    assert 'detail_mode == "fast"' in detail
    assert ('Qwen overlapping tiles' in detail or 'Qwen 4-tile' in detail)
    assert 'Real-ESRGAN' in source and 'Real-ESRNet' in source



def test_image_prompts_are_translated_by_the_selected_local_encoder_without_remote_service():
    source = read_source()
    assert 'def prepare_native_prompt(prompt: str, log_callback=None) -> str:' in source
    assert 'def translate_prompt_to_english(' in source
    assert 'self._translate_image_prompt_to_english(job.prompt, job.workflow)' in source
    assert 'gpu_layers=0' in source
    assert 'using the original prompt' in source
    assert 'http://127.0.0.1:11434' not in source
    assert 'PAI_OLLAMA_URL' not in source
    assert 'PAI_VLM_MODEL' not in source
    assert 'def _ensure_local_ollama_server' not in source
    assert 'def _ollama_chat_local' not in source


def test_closeup_has_visual_bbox_grounding_and_bbox_centered_crop():
    source = read_source()
    assert 'def detect_target_bbox_local' in source
    assert 'def crop_closeup_from_bbox' in source
    assert 'grounding_result = self._locate_closeup_target(' in source
    assert 'self._locate_closeup_target(' in source
    assert 'model_path=model_path' in source and 'mmproj_path=mmproj_path' in source
    assert 'Qwen3-VL open-vocabulary semantic grounding' in source



def test_release_contains_no_remote_inference_dependency():
    source = read_source().lower()
    assert "ollama" not in source
    assert "setup_local_qwen3vl" not in source
    assert "def _ollama_chat_local" not in source
    assert "def translate_prompt_to_english" in source
    assert "translate_prompt_via_local_encoder" in source
    assert 'def prepare_native_prompt' in source
    assert '"--host", "127.0.0.1"' in source
    assert 'no hosted inference api' in source

def test_preview_starts_original_then_uses_workflow_specific_comparison_after_output():
    source = read_source()
    assert 'def preferred_preview_mode_for_workflow(workflow: str) -> str:' in source
    assert 'if workflow in {"image_edit", "enhance", "upscale_only", "lora_closeup_1024", "square_1_1"}:' in source
    assert 'return "Side by side"' in source
    assert 'return "Enhanced"' in source
    preset = source[source.index('def apply_workflow_preset('):source.index('def toggle_left_dock(')]
    assert 'self.preview_mode.setCurrentText("Original")' in preset
    assert 'self.preview.set_mode("Original")' in preset
    assert 'preferred_view = "Enhanced" if job.workflow == "image_generate" else str(getattr(self, "_preferred_preview_mode", "Side by side"))' in source



def test_qwen3vl_grounding_response_parser_normalizes_coordinates_and_rejects_invalid_boxes():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_qwen3vl_grounding_response')
    namespace = {'re': re, 'json': __import__('json'), 'Optional': __import__('typing').Optional, 'Any': __import__('typing').Any}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<qwen3vl-grounding-parser>', 'exec'), namespace)
    parser = namespace['parse_qwen3vl_grounding_response']
    assert parser('{"found":true,"bbox_2d":[100,200,600,800]}') == (0.1, 0.2, 0.6, 0.8)
    assert parser('```json\n{"found": true, "bbox_2d": [0.2, 0.1, 0.8, 0.9]}\n```') == (0.2, 0.1, 0.8, 0.9)
    assert parser('{"found": false}') is None
    assert parser('{"found": true, "bbox_2d": [0, 0, 1000, 1000]}') is None
    assert parser('{"found": true, "bbox_2d": [400, 200, 300, 700]}') is None

def test_closeup_detected_bbox_crop_stays_inside_source_and_is_square():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "crop_closeup_from_bbox")
    namespace = {"Image": Image, "ImageOps": ImageOps, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<closeup-crop>", "exec"), namespace)
    original = Image.new("RGB", (100, 200), (120, 80, 40))
    result, box, zoom = namespace["crop_closeup_from_bbox"](original, 128, (0.2, 0.25, 0.8, 0.75), padding=3.0)
    left, top, right, bottom = box
    assert result.size == (128,128)
    assert right-left == bottom-top
    assert 0 <= left < right <= original.width
    assert 0 <= top < bottom <= original.height
    assert zoom >= 1.0


def test_square_canvas_uses_actual_edge_context_not_blurred_center_cover():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_square_outpaint_inputs")
    namespace = {"Image": Image, "ImageOps": ImageOps, "ImageFilter": __import__("PIL.ImageFilter", fromlist=["ImageFilter"])}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<square-init>", "exec"), namespace)
    original = Image.new("RGB", (80, 120))
    pix = original.load()
    for y in range(120):
        for x in range(80):
            pix[x,y] = (x*3, y*2, (x+y)%255)
    canvas, mask = namespace["build_square_outpaint_inputs"](original, 120, (20,0,100,120), feather=12)
    assert canvas.size == (120,120) and mask.size == (120,120)
    # Edge context should be an extrapolation of the left/right border, not a crop
    # from the middle of the source image as the older implementation did.
    assert abs(canvas.getpixel((19,60))[0] - canvas.getpixel((20,60))[0]) < 15
    assert abs(canvas.getpixel((99,60))[0] - canvas.getpixel((100,60))[0]) < 15
    assert mask.getpixel((0,60)) == 255 and mask.getpixel((60,60)) == 0


def test_square_outpaint_uses_one_joint_masked_generation_pass():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    assert 'joint_prompt, source_original, (side, side)' in square
    assert 'mask_image=_full_mask, strength=0.94' in square
    assert 'JOINT OUTPAINT PASS' in square
    assert 'one Qwen generation pass for all missing margins' in square
    assert 'for edge_index, edge_name in enumerate(edge_names)' not in square
    assert 'Square 1:1 edge plan (offline): ' in square
    assert 'result = composite_square_outpaint_result(' in square


def test_square_ultimate_refines_only_the_completed_output_after_joint_outpaint():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    joint_infer = square.index('joint_prompt, source_original, (side, side)')
    final_composite = square.index('result = composite_square_outpaint_result(')
    ultimate_marker = square.index('Ultimate Qwen refinement is running on the completed final square output')
    final_refine = square.index('result = self._qwen_detail_refine_large_output(', ultimate_marker)
    save_output = square.index('save_image_with_metadata(result, path, source_original')
    assert joint_infer < final_composite < ultimate_marker < final_refine < save_output
    assert 'Completed square 1:1 image; preserve the finished framing' in square
    assert 'detail_mode="ultimate", use_img2img=True' in square[ultimate_marker:save_output]
    assert 'detail_mode="fast"' in square[:ultimate_marker]


def test_closeup_uses_qwen3vl_and_manual_focus_still_wins():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    focus = process.index('if job.focus_point is not None:')
    detection = process.index('grounding_result = self._locate_closeup_target(')
    assert focus < detection
    assert 'if grounding_result.status != "found" or grounding_result.bbox_xyxy is None:' in process
    assert 'No center crop was substituted' in process
    assert 'semantic model inference is skipped for target localization' in process

def test_square_edge_mask_exposes_only_requested_margin():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_square_outpaint_inputs")
    namespace = {"Image": Image, "ImageOps": ImageOps, "ImageFilter": __import__("PIL.ImageFilter", fromlist=["ImageFilter"])}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<square-mask>", "exec"), namespace)
    image = Image.new("RGB", (80,120), (120,100,80))
    _canvas, left_mask = namespace["build_square_outpaint_inputs"](image,120,(20,0,100,120),12,editable_sides=("left",))
    assert left_mask.getpixel((0,60)) == 255
    assert left_mask.getpixel((119,60)) == 0
    assert left_mask.getpixel((60,60)) == 0
    _canvas, right_mask = namespace["build_square_outpaint_inputs"](image,120,(20,0,100,120),12,editable_sides=("right",))
    assert right_mask.getpixel((0,60)) == 0
    assert right_mask.getpixel((119,60)) == 255


def test_square_side_compositor_only_accepts_the_requested_edge():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "composite_square_outpaint_side")
    namespace = {"Image": Image}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<square-side-composite>", "exec"), namespace)
    base = Image.new("RGB", (120,120), (200,10,10))
    generated = Image.new("RGB", (120,120), (10,20,220))
    result = namespace["composite_square_outpaint_side"](base, generated, (20,0,100,120), "left", 8)
    assert result.getpixel((0,60)) == (10,20,220)
    assert result.getpixel((119,60)) == (200,10,10)
    assert result.getpixel((60,60)) == (200,10,10)


def test_closeup_grounding_uses_selected_encoder_and_only_downloads_runtime():
    source = read_source()
    assert 'GROUNDING_RUNTIME_ASSETS: list[dict[str, Any]]' in source
    assert 'def download_qwen_grounding_bundle(self):' in source
    assert 'def _start_server_locked(self, timeout_seconds: float = 300.0, gpu_layers: int = 99)' in source
    assert '"--host", "127.0.0.1"' in source
    assert '"bbox_2d"' in source and 'bbox_2d' in source
    assert 'Right-click the exact item in the preview' in source
    assert 'target-region heuristic' not in source
    assert 'OpenCV face cascade + target-region heuristic' not in source
    assert 'selected Model Catalog text encoder' in source
    assert 'No second text encoder will be downloaded' in source
    assert 'Qwen3VL-8B-Instruct-Q4_K_M.gguf' not in source

def test_square_outpaint_has_no_external_edge_analyzer_and_uses_offline_edge_plan():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    assert 'Square 1:1 edge plan (offline): ' in process
    assert 'describe_square_outpaint_context_local' not in source
    assert 'mask_image=_full_mask, strength=0.94' in process
    assert 'joint_prompt, source_original, (side, side)' in process
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    assert 'JOINT OUTPAINT PASS' in square
    assert 'for edge_index, edge_name in enumerate(edge_names)' not in square


def test_square_outpaint_only_skips_qwen_tile_regeneration_at_large_output_sizes():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    assert 'if job.enhance:' in square
    assert 'Square 1:1 base Enhance is OFF · completed outpaint is preserved' in square
    assert 'Ultimate Qwen refinement is running on the completed final square output' in square
    assert 'source_core, core_prompt, source_core.width, source_core.height' not in square

def test_closeup_bbox_crop_honors_zoom_slider_when_box_allows_it():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "crop_closeup_from_bbox")
    namespace = {"Image": Image, "ImageOps": ImageOps, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<closeup-zoom>", "exec"), namespace)
    crop_fn = namespace["crop_closeup_from_bbox"]
    image = Image.new("RGB", (100, 200), (90, 100, 110))
    bbox = (0.45, 0.40, 0.55, 0.50)
    _loose, _box_loose, zoom_loose = crop_fn(image, 128, bbox, zoom_factor=1.2)
    _tight, _box_tight, zoom_tight = crop_fn(image, 128, bbox, zoom_factor=4.0)
    assert zoom_loose <= 1.21
    assert zoom_tight > zoom_loose


def test_closeup_manual_point_overrides_auto_zoom_and_stale_focus_is_cleared():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    assert 'if job.auto_zoom:' in process
    assert 'if job.auto_zoom and job.focus_point is None' not in process
    assert 'if job.focus_point is not None:' in process
    assert 'exact user-selected preview point' in process
    assert 'zoom_factor=job.zoom_factor' in process
    select = source[source.index('def select_queue_row'):source.index('def queue_context_menu')]
    assert 'if path != self.current_input:' in select
    assert 'self.preview.focus_point = None' in select


def test_square_high_res_detail_refinement_operates_on_completed_square_output():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    square = process[process.index('elif job.workflow == "square_1_1"'):]
    assert 'generated = composite_square_outpaint_result(' in square
    assert 'result = composite_square_outpaint_result(' in square
    assert 'square_output_prompt' in square
    assert 'result, square_output_prompt, side, side, square_seed + 1000' in square
    assert 'generated.paste(refined_core, (left, top))' not in square
    assert 'generated, "", square_seed + 1000' not in square

def test_fidelity_composite_preserves_source_low_frequency_and_caps_ai_detail_delta():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preserve_source_structure_with_ai_detail")
    namespace = {"Image": Image, "ImageOps": ImageOps, "ImageFilter": __import__("PIL.ImageFilter", fromlist=["ImageFilter"])}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<fidelity-composite>", "exec"), namespace)
    original = Image.new("RGB", (96, 96), (120, 100, 80))
    generated = Image.new("RGB", (96, 96), (230, 40, 180))
    output = namespace["preserve_source_structure_with_ai_detail"](original, generated, detail_strength=0.24)
    # The generated image has only a constant color shift, hence its high-frequency band
    # is zero; it must not recolor the original photograph.
    assert output.getpixel((48, 48)) == original.getpixel((48, 48))
    assert output.size == original.size


def test_square_plan_outpaints_near_square_instead_of_cropping_source_edges():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plan_square_outpaint")
    namespace = {"Image": Image, "ImageOps": ImageOps}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<square-plan>", "exec"), namespace)
    original = Image.new("RGB", (118, 120), (40, 80, 120))
    mode, _fitted, margins = namespace["plan_square_outpaint"](original, 120)
    assert mode == "outpaint"
    assert margins is not None
    assert (margins[2] - margins[0]) == 118
    assert (margins[3] - margins[1]) == 120


def test_auto_closeup_uses_qwen3vl_not_fixed_subject_keyword_lists():
    source = read_source()
    assert 'class Qwen3VLGrounder' in source
    assert 'Target description, to interpret literally: {target_text}' in source
    assert 'generic_person_terms' not in source
    assert 'OpenCV face anchor + head/shoulder subject estimate' not in source
    assert 'not Qwen3-VL phrase grounding' not in source

def test_only_mit_licensed_llama_runtime_is_downloaded_for_grounding_setup():
    source = read_source()
    block = source[source.index('GROUNDING_RUNTIME_ASSETS: list['):source.index('def llama_grounding_runtime_present')]
    assert '"license": "MIT"' in block
    assert 'https://github.com/ggml-org/llama.cpp/releases/download/b11744/llama-b11744-bin-win-cuda-12.4-x64.zip' in block
    assert 'runtime-only download' in source
    assert 'Prepare Locator Runtime' in source
    assert 'Qwen3VL-8B-Instruct-Q4_K_M.gguf' not in source
    assert 'selected Model Catalog text encoder' in source



def test_v189_stronger_detail_strengths_and_square_core_remain_protected():
    source = read_source()
    assert 'prompt, source, (width, height), seed, quality, steps' in source
    assert 'Qwen-Image 2.1 full-frame enhancement pass complete' in source
    assert '_boost_aligned_luminance_detail' not in source
    assert 'prepare_enhance_base_image' not in source
    assert 'mask_image=_full_mask, strength=0.94' in source
    assert 'JOINT OUTPAINT PASS' in source
    assert 'Ultimate Qwen refinement is running on the completed final square output' in source
    assert 'qwen_turbo_tile_sampling_params' in source


def test_preview_mode_policy_has_exact_requested_result_defaults():
    source = read_source()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preferred_preview_mode_for_workflow")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<preview-policy>", "exec"), namespace)
    fn = namespace["preferred_preview_mode_for_workflow"]
    assert fn("image_generate") == "Enhanced"
    assert fn("image_edit") == "Side by side"
    assert fn("enhance") == "Side by side"
    assert fn("upscale_only") == "Side by side"
    assert fn("lora_closeup_1024") == "Side by side"
    assert fn("square_1_1") == "Side by side"
    assert 'self.preview_mode.setCurrentText("Original")' in source




def test_focus_point_uses_the_real_painted_image_bounds_and_does_not_clamp_outside_clicks():
    source = read_source()
    canvas = source[source.index("class PreviewCanvas"):source.index("class App(") if "class App(" in source else source.index("class MainWindow")]
    assert 'self._pane_image_targets["original"] = self._paint_single' in canvas
    assert 'self._pane_image_targets["enhanced"] = self._paint_single' in canvas
    assert "preview_click_to_normalized(" in canvas
    mouse_press = canvas[canvas.index("def mousePressEvent"):canvas.index("def mouseMoveEvent")]
    assert "if target is None or not viewport.contains(pos) or not target.contains(pos):" in mouse_press
    assert "max(0.0, min(1.0, fx))" not in mouse_press


def test_live_preview_refresh_uses_revision_counter_not_full_jpeg_hash_on_ui_thread():
    source = read_source()
    refresh = source[source.index("def _refresh_live_generation_preview"):source.index("def _apply_status_message")]
    assert "latest_preview_revision" in refresh
    assert 'preview.save(buffer, format="JPEG", quality=78, optimize=True)' not in refresh
    assert "hashlib.sha256" not in refresh
    infer = source[source.index("def _infer("):source.index("def _looks_like_cuda_oom")]
    assert 'timeout=1.5' in infer
    assert 'self.force_stop.wait(0.10)' in infer

def test_edit_original_size_is_default_and_final_size_is_restored_after_internal_canvas_rounding():
    source = read_source()
    assert '"edit_size": "Original Size"' in source
    assert 'self.edit_size.addItems(["Original Size",' in source
    edit_job = source[source.index('if workflow == "image_edit":'):source.index('if workflow == "lora_closeup_1024":')]
    assert 'if self.edit_size.currentText() in {"Original Size", "Keep Original Size (Auto)"}' in edit_job
    assert 'width, height = ImageOps.exif_transpose(im).size' in edit_job
    infer = source[source.index('def _infer('):source.index('def _looks_like_cuda_oom(')]
    assert 'if result_image.size != size:' in infer
    assert 'result_image = result_image.resize(size, Image.Resampling.BICUBIC)' in infer


def test_qwen3vl_grounding_runtime_is_cpu_local_and_uses_natural_language_target_verbatim():
    source = read_source()
    grounder = source[source.index('class Qwen3VLGrounder'):source.index('_QWEN3VL_GROUNDER:')]
    assert '"--n-gpu-layers", str(requested_gpu_layers)' in grounder
    assert 'def translate_prompt_to_english(' in grounder
    assert '"--host", "127.0.0.1"' in grounder
    assert '"/v1/chat/completions"' in grounder
    assert 'f"Target description, to interpret literally: {target_text}"' in grounder
    assert 'right-click the intended target in the preview' in source



def test_enhance_once_sends_same_source_as_reference_and_init_image():
    source = read_source()
    tree = ast.parse(source)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocalEngine")
    method = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "enhance_once")
    namespace = {"ENHANCE_IMG2IMG_STRENGTH": 0.46, "Image": Image, "Optional": __import__('typing').Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<balanced-enhance>", "exec"), namespace)

    class FakeEngine:
        enhance_once = namespace["enhance_once"]

        def __init__(self):
            self.captured = None

        def log(self, _message):
            pass

        def _infer_with_auto_offload_retry(self, prompt, reference, size, seed, quality, steps, **kwargs):
            self.captured = (prompt, reference, size, seed, quality, steps, kwargs)
            return Image.new("RGB", size)

    engine = FakeEngine()
    image = Image.new("RGB", (320, 480), (90, 100, 110))
    out = engine.enhance_once(image, "preserve identity", 640, 960, 7, "Balanced", 20)
    assert out.size == (640, 960)
    assert engine.captured[1] is image
    assert engine.captured[6]["init_image"] is image
    assert engine.captured[6]["strength"] == 0.46
    assert engine.captured[6]["fit_source_to_output"] is True


def test_square_reference_mode_is_propagated_to_high_resolution_tiles():
    source = read_source()
    tree = ast.parse(source)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocalEngine")
    method = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "_enhance_qwen_output")
    namespace = {"Image": Image, "MAX_INFERENCE_SIDE": 2048, "MAX_OUTPUT_SIDE": 8192, "Optional": __import__('typing').Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<square-refinement-mode>", "exec"), namespace)

    class FakeEngine:
        _enhance_qwen_output = namespace["_enhance_qwen_output"]
        _publish_live_preview = lambda self, image, marker=None: None

        def __init__(self):
            self.tile_kwargs = None

        def _inference_size(self, size):
            w, h = size
            scale = min(1.0, 2048 / max(w, h))
            return max(64, round(w * scale / 32) * 32), max(64, round(h * scale / 32) * 32)

        def log(self, _message):
            pass

        def _enhance_with_retries(self, original, prompt, width, height, *args, **kwargs):
            return Image.new("RGB", (width, height), (20, 30, 40))

        def _qwen_detail_refine_large_output(self, image, prompt, seed, quality, steps, detail_mode, **kwargs):
            self.tile_kwargs = kwargs
            return image

    engine = FakeEngine()
    original = Image.new("RGB", (1000, 500))
    result = engine._enhance_qwen_output(original, "square faithful", 4096, 2048,
                                         15, "Balanced", 1, 25.0, 18.0, 8,
                                         detail_mode="balanced", use_img2img=False)
    assert result.size == (4096, 2048)
    assert engine.tile_kwargs == {"use_img2img": False, "tile_strength": None, "detail_profile": "balanced", "force": False}


def test_realesrgan_selected_model_pair_can_be_restored_from_cached_zip(tmp_path):
    import os
    import shutil
    source = read_source()
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "restore_realesrgan_model_pair_from_archive")
    namespace = {"Path": Path, "zipfile": zipfile, "shutil": shutil, "os": os}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<realesrgan-repair>", "exec"), namespace)
    archive = tmp_path / "official-runtime.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("realesrgan-ncnn-vulkan-20220424-windows/models/realesrnet-x4plus.param", b"params-content")
        zf.writestr("realesrgan-ncnn-vulkan-20220424-windows/models/realesrnet-x4plus.bin", b"weights-content")
    model_dir = tmp_path / "data" / "models"
    restore = namespace["restore_realesrgan_model_pair_from_archive"]
    assert restore(archive, model_dir, "realesrnet-x4plus") is True
    assert (model_dir / "realesrnet-x4plus.param").read_bytes() == b"params-content"
    assert (model_dir / "realesrnet-x4plus.bin").read_bytes() == b"weights-content"
    assert restore(archive, model_dir, "not-a-real-model") is False


def test_realesrgan_pair_repair_runs_before_network_fallback():
    source = read_source()
    worker = source[source.index('    def _ensure_realesrgan_model_pair('):source.index('    def _ensure_realesrgan_runtime(')]
    assert worker.index('restore_realesrgan_model_pair_from_archive') < worker.index('self._ensure_hf_realesrgan_pair(model_dir, stem)')
    assert 'minimum model sizes before running inference' in source


def test_github_release_files_and_dependency_declarations_exist():
    required = [
        "README.md", "README_FA.md", "CHANGELOG.md", "VERSION", "LICENSE",
        "CONTRIBUTING.md", "SECURITY.md", "pai_studio/__init__.py",
        "pai_studio/validation.py", "requirements.txt", "requirements-build.txt",
        "requirements-test.txt", ".github/workflows/tests.yml",
        ".github/PULL_REQUEST_TEMPLATE.md", ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), f"Missing release file: {relative}"
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "PySide6" in requirements and "Pillow" in requirements
    assert "certifi" in requirements and "numpy" in requirements
    assert "Pillow>=10.0,<12" in requirements  # conservative cap avoids known conflicts with packages requiring Pillow <12
    assert "pytest" not in requirements and "PyInstaller" not in requirements
    test_requirements = (ROOT / "requirements-test.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in test_requirements and "pytest" in test_requirements


def test_release_version_display_is_consistent_in_active_metadata():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    display = "Version 1 Beta"
    source = read_source()
    assert f'APP_VERSION = "{version}"' in source
    assert f'DISPLAY_VERSION = "{display}"' in source
    assert f'#define AppVerName "P.AI AI IMAGE STUDIO {display}"' in (ROOT / "installer.iss").read_text(encoding="utf-8")
    assert f"`{version}`" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{version}`" in (ROOT / "README_FA.md").read_text(encoding="utf-8")
    assert "Version 1 Beta · Local AI imaging" in source


def test_setup_scripts_verify_required_imports_without_blocking_on_unrelated_pip_check():
    bat = (ROOT / "Install_Update_Requirements.bat").read_text(encoding="utf-8")
    build = (ROOT / "build_release.ps1").read_text(encoding="utf-8")
    assert 'import PySide6, PIL, certifi, numpy' in bat
    assert 'import PySide6, PIL, certifi, numpy' in build
    assert 'pip check found a conflict in the shared system Python environment' in build
    assert 'shared system Python environment' in bat



def test_enhance_scale_is_stage_one_and_ultimate_scale_is_independent_stage_two():
    source = read_source()
    job_builder = source[source.index('    def _job_from_ui('):source.index('    def generate_images(')]
    assert 'qwen_scale = int(getattr(self, "std_output_scale", 2))' in job_builder
    assert 'ultimate_qwen_scale_combo.currentData()' not in job_builder
    enhance = source[source.index('elif job.workflow == "enhance":'):source.index('elif job.workflow == "lora_closeup_1024":')]
    assert 'ultimate_scale=int(job.custom.get("ultimate_qwen_scale", 2) or 2) if ultimate_enabled else 1' in enhance
    assert 'stage_result.resize((requested_width, requested_height), Image.Resampling.LANCZOS)' in source

def test_qwen_detail_profiles_keep_current_default_and_offer_stronger_modes():
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qwen_detail_profile_values")
    helper_src = ast.get_source_segment(source, helper_node)
    profiles_node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QWEN_DETAIL_PROFILES" for t in node.targets))
    profiles_src = ast.get_source_segment(source, profiles_node)
    namespace = {}
    exec(profiles_src + "\n" + helper_src, namespace)
    balanced = namespace["qwen_detail_profile_values"]("balanced")
    strong = namespace["qwen_detail_profile_values"]("strong")
    maximum = namespace["qwen_detail_profile_values"]("maximum")
    assert balanced == (0.46, 0.38)
    assert strong[0] > balanced[0] and strong[1] > balanced[1]
    assert maximum[0] > strong[0] and maximum[1] > strong[1]
    assert namespace["qwen_detail_profile_values"]("unknown") == balanced
    assert '"qwen_detail_profile": "balanced"' in source


def test_enhance_uses_stronger_reconstruction_strengths_without_enabling_ultimate():
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qwen_enhance_reconstruction_strengths")
    helper_src = ast.get_source_segment(source, helper_node)
    namespace = {}
    exec(helper_src, namespace)
    balanced = namespace["qwen_enhance_reconstruction_strengths"]("balanced")
    strong = namespace["qwen_enhance_reconstruction_strengths"]("strong")
    maximum = namespace["qwen_enhance_reconstruction_strengths"]("maximum")
    assert balanced == (0.84, 0.74)
    assert strong == (0.91, 0.82)
    assert maximum == (0.97, 0.92)
    assert strong[0] > balanced[0] and strong[1] > balanced[1]
    assert maximum[0] > strong[0] and maximum[1] > strong[1]
    process = source[source.index('elif job.workflow == "enhance":'):source.index('elif job.workflow == "lora_closeup_1024":')]
    assert 'detail_mode = enhance_detail_mode_for_ultimate(ultimate_enabled)' in process
    assert 'qwen_enhance_reconstruction_strengths(profile_key)' in process
    assert "Ultimate=ON" in process and "Ultimate tiles" in process


def test_qwen_enhancement_uses_lanczos_only_for_ultimate_source_canvas_and_keeps_standalone_realesrgan_visible():
    source = read_source()
    upscale_start = source.index('    def _run_upscale_file')
    upscale_end = source.index('    def _translate_image_prompt_to_english', upscale_start)
    assert "Image.Resampling.LANCZOS" not in source[upscale_start:upscale_end]
    enhance_start = source.index('    def _enhance_qwen_output')
    enhance_end = source.index('    def process(self, job: Job)', enhance_start)
    assert "Image.Resampling.LANCZOS" in source[enhance_start:enhance_end]
    assert 'self.realesrgan_box = QGroupBox("AI Upscaler · Real-ESRGAN / Real-ESRNet")' in source
    assert 'upscale_page.addWidget(esrgan_card)' in source
    assert 'upscale_page.addWidget(esrgan_card)' in source
    assert 'catalog_tabs.addTab(page_scroll, title)' in source
    assert 'AI Upscaler' in source and 'Real-ESRGAN remains a separate optional post-process' in source
    assert 'Qwen-Image 2.1 full-frame enhancement pass complete' in source


def test_qwen_detail_profile_prompts_and_strengths_are_distinct_and_routed_across_workflows():
    source = read_source()
    assert '"balanced": {"label": "Balanced · preserve identity", "short_label": "Balanced", "strength": 0.46, "tile_strength": 0.38' in source
    assert '"strong": {"label": "Strong · more detail", "short_label": "Strong", "strength": 0.64, "tile_strength": 0.54' in source
    assert '"maximum": {"label": "Maximum · experimental", "short_label": "Maximum", "strength": 0.82, "tile_strength": 0.72' in source
    assert 'Active prompt guidance ({profile})' not in source
    assert 'Strongest detail recovery, highest change risk.' in source
    assert 'self._refresh_qwen_profile_prompt_note(profile)' in source
    process = source[source.index('    def process(self, job: Job)'):source.index('class Worker(QThread)')]
    for workflow in ('image_generate', 'image_edit', 'enhance', 'lora_closeup_1024', 'square_1_1'):
        assert f'job.workflow == "{workflow}"' in process
    assert 'apply_qwen_detail_profile_prompt(prepare_native_prompt(translated_prompt, self.log), profile_key)' in process
    assert 'self._translate_image_prompt_to_english(job.prompt, job.workflow)' in process
    assert 'apply_qwen_detail_profile_prompt(build_enhance_prompt(job.prompt), profile_key)' in process
    assert 'apply_qwen_detail_profile_prompt(build_square_core_enhance_prompt(job.prompt), profile_key)' in process


def test_maximum_profile_runs_a_real_second_qwen_reconstruction_pass():
    source = read_source()
    detail = source[source.index('    def _qwen_detail_refine_large_output'):source.index('    def _enhance_qwen_output')]
    assert 'refinement_passes = 2 if profile_key == "maximum" else 1' in detail
    assert 'for pass_index in range(refinement_passes)' in detail
    assert 'pass_input = result.copy()' in detail
    assert 'profile_limit = QWEN_DETAIL_PROFILE_TILE_LIMITS[profile_key]' in detail
    assert 'SECOND CLEANUP PASS' in detail
    assert 'strength=effective_tile_strength' in detail


def test_qwen_profile_metadata_matches_strength_tiles_passes_and_fast_mode():
    import copy
    source = read_source()
    tree = ast.parse(source)
    targets = {"QWEN_DETAIL_PROFILES", "QWEN_DETAIL_PROFILE_TILE_LIMITS"}
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in targets:
                    assignments[target.id] = ast.literal_eval(node.value)
    helper_names = {"qwen_detail_profile_values", "qwen_detail_refinement_metadata"}
    helpers = [copy.deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    for helper in helpers:
        helper.decorator_list = []
    namespace = {
        "QWEN_DETAIL_PROFILES": assignments["QWEN_DETAIL_PROFILES"],
        "QWEN_DETAIL_PROFILE_TILE_LIMITS": assignments["QWEN_DETAIL_PROFILE_TILE_LIMITS"],
        "Any": __import__("typing").Any,
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "<profile-metadata>", "exec"), namespace)
    metadata = namespace["qwen_detail_refinement_metadata"]
    balanced = metadata("balanced", "balanced")
    strong = metadata("strong", "balanced")
    maximum = metadata("maximum", "max_detail")
    fast = metadata("maximum", "fast")
    disabled = metadata("maximum", "max_detail", enabled=False)
    assert (balanced["qwen_tile_strength"], balanced["qwen_tile_limit"], balanced["qwen_detail_refinement_passes"]) == (0.38, 2048, 1)
    assert (strong["qwen_tile_strength"], strong["qwen_tile_limit"], strong["qwen_detail_refinement_passes"]) == (0.54, 1536, 1)
    assert (maximum["qwen_tile_strength"], maximum["qwen_tile_limit"], maximum["qwen_detail_refinement_passes"]) == (0.72, 1280, 2)
    assert fast["qwen_tile_limit"] is None and fast["qwen_detail_refinement_passes"] == 0
    assert disabled["qwen_detail_profile"] is None and disabled["qwen_tile_strength"] is None


def test_profile_prompts_are_distinct_and_maximum_runs_two_real_refinement_calls():
    import copy
    from PIL import ImageChops
    source = read_source()
    tree = ast.parse(source)
    profiles_node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QWEN_DETAIL_PROFILES" for t in node.targets))
    prompts_node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QWEN_DETAIL_PROFILE_PROMPTS" for t in node.targets))
    limits_node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QWEN_DETAIL_PROFILE_TILE_LIMITS" for t in node.targets))
    profile_data = ast.literal_eval(profiles_node.value)
    prompt_data = ast.literal_eval(prompts_node.value)
    tile_limits = ast.literal_eval(limits_node.value)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_qwen_detail_profile_prompt")
    helper = copy.deepcopy(helper); helper.decorator_list = []
    namespace = {"QWEN_DETAIL_PROFILE_PROMPTS": prompt_data, "QWEN_DETAIL_PROFILES": profile_data}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<profile-prompt-helper>", "exec"), namespace)
    base_prompt = "Faithfully enhance the supplied photograph. Keep the same person, geometry and colors."
    profiled = {key: namespace["apply_qwen_detail_profile_prompt"](base_prompt, key) for key in ("balanced", "strong", "maximum")}
    assert len(set(profiled.values())) == 3
    assert all("Keep the same person" in value for value in profiled.values())
    assert "suppress fine random luminance/chroma noise" in profiled["balanced"]
    assert "clearly stronger Qwen" in profiled["strong"]
    assert "aggressively suppress visible random" in profiled["maximum"]

    local_engine_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LocalEngine")
    method_names = {"_qwen_detail_refine_large_output", "_tile_starts", "_tile_fade_mask"}
    methods = []
    for node in local_engine_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in method_names:
            method = copy.deepcopy(node); method.decorator_list = []
            methods.append(method)
    namespace.update({
        "Image": Image, "ImageChops": ImageChops, "Optional": __import__("typing").Optional,
        "MAX_INFERENCE_SIDE": 2048, "ENHANCE_TILE_IMG2IMG_STRENGTH": 0.38,
        "math": math, "QWEN_DETAIL_PROFILE_PROMPTS": prompt_data, "QWEN_DETAIL_PROFILES": profile_data,
        "QWEN_DETAIL_PROFILE_TILE_LIMITS": tile_limits,
        "Aborted": type("Aborted", (Exception,), {}),
    })
    exec(compile(ast.Module(body=methods, type_ignores=[]), "<detail-refinement>", "exec"), namespace)

    class NeverStop:
        def is_set(self):
            return False

    class FakeEngine:
        _qwen_detail_refine_large_output = namespace["_qwen_detail_refine_large_output"]
        _tile_starts = staticmethod(namespace["_tile_starts"])
        _tile_fade_mask = staticmethod(namespace["_tile_fade_mask"])
        _publish_live_preview = lambda self, image, marker=None: None

        def __init__(self):
            self.force_stop = NeverStop()
            self.calls = []
            self.logs = []
            self.status_callback = lambda _message: None

        def log(self, message):
            self.logs.append(message)

        def _infer_with_auto_offload_retry(self, prompt, tile, size, seed, quality, steps, **kwargs):
            self.calls.append({"prompt": prompt, "strength": kwargs.get("strength"), "seed": seed, "size": size})
            return tile.copy()

    calls_by_profile = {}
    for key, strength in (("balanced", 0.38), ("strong", 0.54), ("maximum", 0.72)):
        engine = FakeEngine()
        sample = Image.new("RGB", (256, 192), (70, 90, 110))
        result = engine._qwen_detail_refine_large_output(
            sample, profiled[key], 12, "Quality", 8, detail_mode="balanced",
            use_img2img=True, tile_strength=strength, detail_profile=key, force=True,
        )
        assert result.size == sample.size
        assert all(call["strength"] == strength for call in engine.calls)
        calls_by_profile[key] = len(engine.calls)
        assert any(prompt_data[key] in call["prompt"] for call in engine.calls)
    assert calls_by_profile == {"balanced": 1, "strong": 1, "maximum": 2}

    # Ultimate-style mode must use the selected profile strength instead of silently forcing 0.20.
    engine = FakeEngine()
    sample = Image.new("RGB", (256, 192), (70, 90, 110))
    result = engine._qwen_detail_refine_large_output(
        sample, profiled["balanced"], 24, "Quality", 8, detail_mode="ultimate",
        use_img2img=True, tile_strength=0.72, detail_profile="maximum", force=True,
    )
    assert result.size == sample.size
    assert len(engine.calls) == 4
    assert all(call["strength"] == 0.72 for call in engine.calls)
    assert any("overlapping tiles" in message.lower() or "qwen 4-tile" in message.lower() for message in engine.logs)


def test_queue_and_input_files_are_first_for_enhance_closeup_and_square():
    source = read_source()
    body = source[source.index('    def _switch_workflow_layout(self, workflow: str)'):source.index('    def ' , source.index('    def _switch_workflow_layout(self, workflow: str)')+8)]
    assert 'queue_below_preview = workflow in ("enhance", "lora_closeup_1024", "square_1_1")' in body
    assert 'basic_layout.insertWidget(0, self.workflow_queue_actions)' in body
    assert 'basic_layout.insertWidget(1, files_box)' in body


def test_real_esrgan_is_independent_and_visible_in_basic_and_model_catalog():
    source = read_source()
    assert 'self.realesrgan_box = QGroupBox("AI Upscaler · Real-ESRGAN / Real-ESRNet")' in source
    assert 'upscale_page.addWidget(esrgan_card)' in source
    assert 'upscale_page.addWidget(esrgan_card)' in source
    assert 'add_settings_tab("AI Upscaler")' in source
    assert 'Run AI Upscale' in source
    assert 'btn_realesrgan_settings_shortcut' not in source
    assert 'show_realesrgan_settings' not in source
    process = source[source.index('    def process(self, job: Job)'):source.index('class Worker(QThread)')]
    enhance_segment = process[process.index('elif job.workflow == "enhance"'):process.index('elif job.workflow == "lora_closeup_1024"')]
    assert '_run_upscale_file(' not in enhance_segment
    assert 'if job.workflow == "upscale_only"' in source



def test_watermark_toggle_is_opt_in_and_wired_to_supported_ai_workflows():
    source = read_source()
    prompts = (ROOT / "pai_studio" / "prompts.py").read_text(encoding="utf-8")
    assert '"watermark_removal_enabled": False' in source
    assert '"watermark": "◌ Watermark"' in source
    assert 'self._global_watermark_toggled(enabled, source=source)' in source
    assert '"watermark_removal_enabled": bool(self.btn_watermark_toggle.isChecked())' in source
    assert 'def apply_watermark_removal_prompt(prompt: str, enabled: bool = False)' in prompts
    assert 'if not enabled:' in prompts
    assert 'WATERMARK REMOVAL TASK:' in prompts
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    for workflow in ('image_edit', 'enhance', 'lora_closeup_1024', 'square_1_1'):
        assert f'job.workflow == "{workflow}"' in process
    assert 'Watermark removal ON · Close-up Enhance is OFF' in process
    assert 'Watermark removal ON · Square Enhance is OFF' in process


def test_hardware_monitor_is_compact_and_gracefully_handles_missing_metrics():
    source = read_source()
    for metric in ('"cpu"', '"ram"', '"gpu"', '"vram"', '"temp"'):
        assert metric in source
    assert 'self.hardware_metrics_timer.setInterval(2500)' in source
    assert 'if psutil is not None:' in source
    assert '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name' in source
    assert 'def refresh_system_metrics(self):' in source


def test_image_edit_and_closeup_pass_selected_qwen_strength_to_img2img():
    source = read_source()
    assert 'init_image=source_original, strength=profile_strength' in source
    assert 'strength=profile_strength' in source
    assert 'tile_strength=profile_tile_strength' in source
    assert 'Qwen detail profile=' in source



def test_new_pai_logo_assets_are_present_and_nonempty():
    for relative in ("assets/pai_logo.png", "assets/pai_icon.png", "assets/pai_icon.ico"):
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size > 1000


def test_hardware_monitor_runs_metric_collection_on_worker_thread():
    source = read_source()
    worker = source[source.index("class HardwareMetricsWorker"):source.index("class App(QMainWindow)")]
    assert "class HardwareMetricsWorker(QThread)" in worker
    assert "self.metrics_ready.emit(metrics)" in worker
    assert "timeout=1.0" in worker
    refresh = source[source.index("    def refresh_system_metrics(self):"):source.index("    def toggle_theme(self):")]
    assert "self._hardware_metrics_worker.start()" in refresh
    assert "subprocess.run" not in refresh



def test_square_core_enhancement_uses_enhance_prompt_not_outpaint_instructions():
    source = read_source()
    assert 'def build_square_core_enhance_prompt(user_instruction: str = "")' in source
    helper = source[source.index('def build_square_core_enhance_prompt'):source.index('IDENTITY_LOCK_PROMPT =')]
    assert 'SQUARE_OUTPAINT_ONLY_PROMPT' in helper
    assert 'SQUARE_ENHANCE_OUTPAINT_PROMPT' in helper
    assert 'return build_enhance_prompt(instruction)' in helper
    assert 'build_square_core_enhance_prompt(job.prompt)' in source




def test_ultimate_is_opt_in_and_workflow_toggle_updates_legacy_enhance_mode():
    source = read_source()
    assert '"ultimate_qwen_upscale_enabled": False' in source
    assert 'return bool(self.settings.get(keys[feature], False))' in source
    toggle = source[source.index('    def _global_ultimate_toggled'):source.index('    def _global_realesrgan_toggled')]
    assert 'self.enhance_detail_mode.setCurrentIndex(0 if enabled else 1)' in toggle
    assert 'self.settings["enhance_detail_mode"] = "ultimate" if enabled else "fast"' in toggle
    assert 'decoded["ultimate_qwen_upscale_enabled"] = False' in source

def test_ultimate_steps_are_configurable_and_reach_each_tile_inference():
    source = read_source()
    assert 'self.ultimate_qwen_steps_spin.setRange(4, 60)' in source
    assert 'ultimate_form.addRow("Steps per tile", self.ultimate_qwen_steps_spin)' in source
    assert 'self._active_ultimate_steps = max(4, min(60, int(job.custom.get("ultimate_qwen_steps", 8))))' in source
    detail = source[source.index('    def _qwen_detail_refine_large_output'):source.index('    def _enhance_qwen_output')]
    assert 'tile_quality = f"Ultimate · {tile_steps} steps"' in detail
    assert 'Directly controls the number of executed denoising steps per tile in Ultimate mode' in source
    assert 'sample_params.update(qwen_turbo_sampling_params())' in source
    assert 'turbo_steps = requested_steps if custom_ultimate_steps else 8' not in source
    assert '"ultimate_qwen_steps": 8' in source
    assert '"ultimate_qwen_tile_coverage": 50' in source
    assert 'int(round(width * coverage))' in detail
    assert 'int(round(height * coverage))' in detail

def test_high_resolution_text_to_image_uses_qwen_tiles_and_selected_profile():
    source = read_source()
    process = source[source.index('def process(self, job: Job)'):source.index('class Worker(QThread)')]
    generate = process[process.index('if job.workflow == "image_generate"'):process.index('elif job.workflow == "image_edit"')]
    assert 'if max(job.width, job.height) > MAX_INFERENCE_SIDE or profile_key in {"balanced", "strong", "maximum"}:' in generate
    assert 'self._qwen_detail_refine_large_output(' in generate
    assert 'detail_mode="ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast", use_img2img=True, tile_strength=profile_tile_strength' in generate
    assert 'qwen_detail_profile_values(profile_key)' in generate
    # Profile metadata must be saved even for 1024/2048 outputs; otherwise users cannot
    # compare which Qwen profile actually generated a file outside the high-res path.
    assert 'qwen_detail_refinement_metadata(profile_key, "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast")' in generate
    assert 'qwen_detail_refinement_metadata(profile_key, "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast")' in process
    assert 'qwen_detail_refinement_metadata(profile_key, detail_mode)' in process
    assert 'qwen_detail_refinement_metadata(job.custom.get("qwen_detail_profile", "balanced"), "ultimate" if bool(job.custom.get("ultimate_qwen_upscale_enabled", False)) else "fast", enabled=bool(job.enhance) and bool(job.custom.get("ultimate_qwen_upscale_enabled", False)))' in process
    assert 'if max(job.width, job.height) > MAX_INFERENCE_SIDE else None' not in generate


def test_prompt_guidance_cfg_is_one_for_turbo_and_non_turbo():
    source = read_source()
    assert 'sample_params["guidance"] = {"txt_cfg": 1.0}' in source
    assert 'sample_params["guidance"] = {"txt_cfg": 4.0}' not in source

def test_ultimate_mode_metadata_and_prompt_profile_remain_connected():
    source = read_source()
    assert '"ultimate":' not in source[source.index('QWEN_DETAIL_PROFILES ='):source.index('QWEN_DETAIL_PROFILE_TILE_LIMITS =')]
    assert 'apply_qwen_detail_profile_prompt((' in source
    assert 'detail_mode="ultimate"' in source
    assert 'strength=effective_tile_strength' in source


def test_lora_uses_relative_api_path_and_explicit_lora_model_directory():
    source = read_source()
    assert 'args.extend(["--lora-model-dir", str(lora_file.parent)])' in source
    assert 'payload["lora"] = [{"path": lora.name, "multiplier": float(self.lora_strength), "is_high_noise": False}]' in source
    assert 'lora_runtime_changed = new_lora_path != self.lora_path or bool(lora_enabled) != self.lora_enabled' in source


def test_global_feature_toggles_and_optional_low_resolution_preview_are_wired():
    source = read_source()
    assert 'self.btn_ultimate_toggle = self._make_feature_toggle_button("ultimate")' in source
    assert 'self.btn_auto_realesrgan = self._make_feature_toggle_button("esrgan")' in source
    assert '"auto_realesrgan_enabled": False' in source
    assert 'if bool(job.custom.get("auto_realesrgan_enabled", False)) and outputs' in source
    assert '"preview": str(self.live_preview_method or "tae") if str(self.live_preview_method or "tae") in {"proj", "tae", "vae"} else "none"' in source
    assert '"preview_interval": max(1, min(20, int(self.live_preview_interval or 1)))' in source
    assert 'self._refresh_live_generation_preview' in source
    assert 'def _publish_live_preview(self, image: Image.Image, marker: Any = None)' in source
    assert 'marker = (job_id, int(step_num or 0), int(total_num or 0), int(preview_pass or 0))' in source
    assert 'self._publish_live_preview(result, marker=("tile-pass-complete", pass_index + 1, tiles_per_pass))' in source
    assert 'self._suppress_live_preview = True' in source
    assert 'QUALITY_PRESETS = ["Turbo · 8 steps", "Fast · 12 steps", "Balanced · 20 steps", "Standard · 28 steps"]' in source
    assert 'Draft · 8 steps' not in source

def test_workflow_feature_toggles_are_below_workflow_selector_and_settings_are_in_app():
    text = read_source()
    assert 'self.btn_ultimate_toggle = self._make_feature_toggle_button("ultimate")' in text
    assert 'self.btn_auto_realesrgan = self._make_feature_toggle_button("esrgan")' in text
    assert 'outer.addLayout(workflow_row)' in text
    assert 'bl.addLayout(self.workflow_feature_row)' in text
    assert 'self.catalog_tabs.addTab(page_scroll, title)' in text
    assert 'models_page = add_settings_tab("Models")' in text
    assert 'upscale_page = add_settings_tab("AI Upscaler")' in text
    assert 'ultimate_page = add_settings_tab("Ultimate")' in text
    assert 'lora_page = add_settings_tab("LoRA")' in text
    assert 'upscale_page.addWidget(esrgan_card)' in text
    assert 'ultimate_page.addWidget(ultimate_card)' in text
    assert 'lora_page.addWidget(self.lora_settings_group)' in text
    assert 'self.btn_lora_toggle = self._make_feature_toggle_button("lora")' in text
    assert 'content_layout.addLayout(self._make_feature_toggle_row())' in text
    assert 'root.addLayout(self._make_feature_toggle_row())' in text
    assert 'right.addRow(self._make_feature_toggle_row())' not in text
    assert 'no separate Real-ESRGAN popup' in text
    assert 'top.addWidget(self.btn_ultimate_toggle)' not in text
    assert 'top.addWidget(self.btn_auto_realesrgan)' not in text
    assert 'btn_realesrgan_settings_shortcut' not in text
    assert 'show_realesrgan_settings' not in text


def test_sdcpp_live_preview_parser_reads_native_preview_object_every_step():
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_parse_sdcpp_preview_status")
    helper_src = textwrap.dedent(ast.get_source_segment(source, helper_node))
    import base64
    import io
    from PIL import ImageOps
    image = Image.new("RGB", (64, 48), (37, 130, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    namespace = {
        "Any": object,
        "Optional": Optional,
        "Image": Image,
        "ImageOps": ImageOps,
        "base64": base64,
        "io": io,
    }
    exec(helper_src, namespace)
    parsed, step, total, sample_pass = namespace["_parse_sdcpp_preview_status"]({
        "status": "generating",
        "preview": {
            "step": 4,
            "total_steps": 9,
            "pass": 1,
            "b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"),
        },
    })
    assert parsed is not None and parsed.size == (512, 384)
    assert parsed.getpixel((0, 0)) == (37, 130, 220)
    assert (step, total, sample_pass) == (4, 9, 1)


def test_sdcpp_live_preview_parser_accepts_old_server_aliases_and_malformed_frames():
    source = read_source()
    tree = ast.parse(source)
    helper_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_parse_sdcpp_preview_status")
    helper_src = textwrap.dedent(ast.get_source_segment(source, helper_node))
    import base64
    import io
    from PIL import ImageOps
    namespace = {"Any": object, "Optional": Optional, "Image": Image, "ImageOps": ImageOps, "base64": base64, "io": io}
    exec(helper_src, namespace)
    parse_preview = namespace["_parse_sdcpp_preview_status"]
    frame, step, total, _ = parse_preview({"preview_info": {"step": "2", "total_steps": "9"}, "preview_base64": "not-valid-image"})
    assert frame is None
    assert (step, total) == (2, 9)


def test_live_preview_uses_native_every_step_mode_and_step_is_visible_in_stage_headline():
    source = read_source()
    assert 'self.live_preview_method_combo.addItem("Projection · fastest, blockier", "proj")' in source
    assert 'self.live_preview_method_combo.addItem("TAE · clearer decode", "tae")' in source
    assert 'self.live_preview_method_combo.addItem("VAE · full decode (slow)", "vae")' in source
    assert '"live_preview_method": "tae"' in source
    assert '"preview_interval": max(1, min(20, int(self.live_preview_interval or 1)))' in source
    assert 'self.progress_label.setText(headline)' in source
    assert 'unit_status = f"Step {step_number}/{step_total}"' in source
    assert 'self.live_preview_method_combo.currentIndexChanged.connect(self._on_live_preview_method_changed)' in source
    assert "TAE is the default clearer decode" in source
    assert 'live_preview_images_for_workflow(active_workflow, self.current_original, preview)' in source
    assert 'self.preview_canvas.set_images(None, preview)' not in source

def test_workflow_studio_fields_and_feature_toggles_are_responsive_and_full_width_without_scroll():
    source = read_source()
    assert 'def _studio_field_cell(label_text: str, control: QWidget) -> QWidget:' in source
    assert 'settings_grid.addWidget(cell, row, column)' in source
    gen_source = source[source.index('    def _build_generate_studio(self):'):source.index('    def _build_edit_studio(self):')]
    edit_source = source[source.index('    def _build_edit_studio(self):'):source.index('    def _gpu_policy_summary')]
    assert 'QScrollArea' not in gen_source and 'generateStudioScroll' not in gen_source
    assert 'QScrollArea' not in edit_source and 'editStudioSettingsScroll' not in edit_source
    assert 'dimensions_cell' in gen_source
    assert 'root.addLayout(self._make_feature_toggle_row())' in edit_source
    assert 'right.addRow(self._make_feature_toggle_row())' not in edit_source
    assert 'root.addLayout(content_layout, 1)' in edit_source



def test_focus_and_mouse_interactions_follow_forced_before_live_layout():
    source = (ROOT / "pai_studio" / "ui" / "canvas.py").read_text(encoding="utf-8")
    canvas = source[source.index("class PreviewCanvas(QWidget):"): ]
    assert "def _uses_side_by_side_layout(self) -> bool:" in canvas
    assert "self.presentation_policy == \"before_live\"" in canvas
    assert canvas.count("self._uses_side_by_side_layout()") >= 5
    set_images = canvas[canvas.index("    def set_images"):canvas.index("    def set_live_image")]
    assert "self._pane_rects = {}" in set_images
    live_only = canvas[canvas.index("    def _paint_live_only"):canvas.index("    def _paint_side_by_side")]
    assert "self._pane_rects = {}" in live_only


def test_gpu_only_never_routes_text_encoder_or_weights_to_cpu():
    source = read_source()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "plan_qwen_memory_placement")
    namespace = {"Any": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "memory_policy.py", "exec"), namespace)
    plan = namespace["plan_qwen_memory_placement"]
    for free_vram in (0.0, 2.0, 15.0):
        placement = plan(free_vram_gb=free_vram, total_vram_gb=16.0, diffusion_gb=7.0, vae_gb=1.0,
                         component_file_gb=14.0, text_encoder_on_cpu=True, memory_mode="full_gpu", auto_offload_override=True)
        assert placement["text_backend"] == "cuda"
        assert placement["force_cpu_te_for_capacity"] is False
        assert placement["offload_to_cpu"] is False
        assert placement["auto_fit_mode"] == "off"


def test_realesrgan_output_dimensions_are_checked_and_tile_default_is_conservative():
    source = read_source()
    assert 'int(options.get("upscale_tile", 0))' in source
    assert "Real-ESRGAN returned unexpected dimensions" in source
    assert "incompatible or mismatched NCNN model pair" in source


def test_ultimate_runs_global_reconstruction_before_tile_refinement():
    source = read_source()
    start = source.index('    def _enhance_qwen_output(')
    end = source.index('    def process(self, job: Job)', start)
    body = source[start:end]
    assert 'Ultimate two-stage upscale' in body
    assert 'base = self._enhance_with_retries(' in body
    assert 'Enhance canvas {stage_width}×{stage_height}' in body
    assert 'requested_ultimate_scale = max(1, min(4, int(ultimate_scale or 2)))' in body

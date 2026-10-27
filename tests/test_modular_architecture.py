# -*- coding: utf-8 -*-
"""
Tests verifying the modular `pai_studio` package structure, direct imports without AST/exec
workarounds, and removal of repository bloat files.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_all_modular_submodules_import_cleanly():
    modules = (
        "pai_studio",
        "pai_studio.constants",
        "pai_studio.validation",
        "pai_studio.workspace",
        "pai_studio.prompts",
        "pai_studio.image_ops",
        "pai_studio.grounding",
        "pai_studio.grounding_strategy",
        "pai_studio.performance",
        "pai_studio.diagnostics",
        "pai_studio.engine",
        "pai_studio.workers",
        "pai_studio.ui",
        "pai_studio.ui.canvas",
        "pai_studio.ui.dialogs",
        "pai_studio.ui.grounding_review_dialog",
        "pai_studio.ui.diagnostics_dialog",
        "pai_studio.ui.main_window",
        "pai_ai_image_studio",
    )
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None


def test_unnecessary_bloat_files_are_removed_for_github_release():
    removed_files = (
        "RELEASE_VALIDATION.md",
        "PUBLIC_RELEASE_CHECKLIST.md",
        "README_RELEASE_FA.md",
        "docs/RUNTIME_AND_WORKFLOW_FIXES.md",
        "docs/DOWNLOADS_AND_PROMPTS.md",
    )
    for rel in removed_files:
        assert not (ROOT / rel).exists(), f"Bloat file should be removed: {rel}"


def test_direct_function_calls_across_modules(tmp_path):
    from pai_studio.constants import APP_VERSION
    from pai_studio.grounding import parse_qwen3vl_grounding_response
    from pai_studio.image_ops import crop_closeup_square, plan_square_outpaint
    from pai_studio.prompts import build_edit_prompt, calculate_qwen_output_size
    from pai_studio.workspace import app_directory, inspect_lora_file

    assert APP_VERSION == "1.0.0-beta.2"
    assert app_directory() == ROOT
    assert calculate_qwen_output_size((1000, 500), "enhance", "Scale", 2, 1024, 1024) == (2000, 1000)
    assert "Preserve the main subject" in build_edit_prompt("change jacket color", lock_identity=True)
    assert parse_qwen3vl_grounding_response('{"found": true, "bbox_2d": [100, 200, 500, 600]}') == (0.1, 0.2, 0.5, 0.6)

    src = Image.new("RGB", (400, 200), (100, 120, 140))
    mode, fitted, margins = plan_square_outpaint(src, target=256)
    assert mode == "outpaint"
    assert margins == (0, 64, 256, 192)

    cropped, box, zoom = crop_closeup_square(src, 256, zoom_factor=2.0)
    assert cropped.size == (256, 256)
    assert zoom >= 2.0

    missing_lora = inspect_lora_file(tmp_path / "nonexistent.safetensors")
    assert missing_lora["exists"] is False

def test_reused_external_runtime_scans_executable_parent_for_cuda_dlls():
    engine_src = (ROOT / "pai_studio" / "engine.py").read_text(encoding="utf-8")
    assert 'for runtime_root in (executable.parent, executable.parent.parent, APP_DIR / "runtime", DATA_ROOT / "runtime"):' in engine_src
    assert "def _stage_beside_backend() -> None:" in engine_src


def test_no_unresolved_global_symbols_in_any_module():
    import builtins
    import symtable

    builtin_names = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__all__", "annotations"}
    for pyfile in sorted((ROOT / "pai_studio").rglob("*.py")):
        src = pyfile.read_text(encoding="utf-8")
        top = symtable.symtable(src, str(pyfile), "exec")
        top_defined = set(builtin_names)
        for sym in top.get_symbols():
            if sym.is_imported() or sym.is_assigned() or sym.is_namespace():
                top_defined.add(sym.get_name())

        missing = set()
        def walk_table(tbl):
            for sym in tbl.get_symbols():
                name = sym.get_name()
                if sym.is_referenced() and sym.is_global() and name not in top_defined:
                    missing.add(name)
            for child in tbl.get_children():
                walk_table(child)

        walk_table(top)
        assert not missing, f"Unresolved globals in {pyfile.relative_to(ROOT)}: {sorted(missing)}"


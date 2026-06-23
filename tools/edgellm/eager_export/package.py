# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact helpers for eager component exports.

This module owns the files that sit beside an engine: contract
JSON, summary JSON, optional packager output, and the final engine
path. Think of it as the packaging layer after capture/compile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.edgellm.contracts.action_contracts import write_action_contract_manifest
from tools.edgellm.contracts.language_contracts import write_language_contract_manifest
from tools.edgellm.contracts.vision_contracts import write_vision_contract_manifest

from .capture import ExampleInputs
from .loader import call_with_supported_kwargs, import_object
from .manifest import EagerExportManifest, EagerExportRole


def resolve_output_dir(
    manifest: EagerExportManifest,
    role: EagerExportRole,
    *,
    output_root: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the directory where one role should write artifacts.

    Priority is: role-specific ``output_dir``, manifest-level
    ``outputs`` by role/component name, then the CLI ``output_root``
    fallback. Returning ``None`` means the caller must raise a clear
    error because there is nowhere safe to write.
    """
    raw = (
        role.output_dir
        or manifest.outputs.get(role.name)
        or manifest.outputs.get(role.component)
    )
    if raw:
        return Path(raw).expanduser().resolve()
    if output_root:
        return Path(output_root).expanduser().resolve() / role.name
    return None


def resolve_engine_path(
    output_dir: Path,
    role: EagerExportRole,
) -> Path:
    """Resolve the exact TensorRT engine path for a compiled role.

    A manifest can force a full engine path, otherwise we place the
    conventional engine filename inside the resolved output directory.
    """
    if role.engine_path:
        return Path(role.engine_path).expanduser().resolve()
    return output_dir / role.default_engine_filename()


def write_role_contract_manifest(
    output_dir: Path,
    role: EagerExportRole,
    examples: ExampleInputs,
    *,
    input_names: list[str],
    output_names: list[str],
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Write the runtime contract manifest for a role.

    The engine knows tensor names and shapes, but it does not know
    what those tensors mean to the Edge runtime. Contract JSON fills
    that semantic gap and lets generic C++ runners bind inputs
    without model-id-specific code.
    """
    if not role.contract:
        return None

    metadata = {
        "eager_export_role": role.name,
        "module_path": role.module_path,
        "exported_program": role.exported_program,
        **dict(role.metadata),
    }
    dynamic_axes = dict(role.dynamic_axes or examples.dynamic_axes or {})
    if role.component == "language":
        return write_language_contract_manifest(
            output_dir,
            role.contract,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            metadata=metadata,
            artifacts=artifacts,
        )
    if role.component == "visual":
        return write_vision_contract_manifest(
            output_dir,
            role.contract,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            metadata=metadata,
            artifacts=artifacts,
        )
    if role.component == "action":
        return write_action_contract_manifest(
            output_dir,
            role.contract,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            metadata=metadata,
            artifacts=artifacts,
        )
    return None


def write_eager_export_summary(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    """Write a small summary of what this export produced.

    This file is for humans and automation: it records which module
    was exported, where the engine went, and which contract file was
    emitted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eager_export_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def call_packager_hook(
    role: EagerExportRole,
    *,
    manifest: EagerExportManifest,
    output_dir: Path,
    engine_path: Optional[Path],
    exported_program_path: Optional[Path],
    loaded: Any,
    module: Any,
    examples: ExampleInputs,
    input_names: list[str],
    output_names: list[str],
) -> Any:
    """Call an optional role-specific packaging hook.

    Some roles need more than a generic contract. For example, VLA
    action export writes diffusion-step metadata derived from the
    Python module. The hook receives all export context and can write
    any extra files it needs beside the engine.
    """
    if not role.packager:
        return None
    packager = import_object(role.packager)
    return call_with_supported_kwargs(
        packager,
        manifest=manifest,
        role=role,
        output_dir=str(output_dir),
        engine_path=str(engine_path) if engine_path is not None else None,
        exported_program_path=(
            str(exported_program_path) if exported_program_path is not None else None
        ),
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        processor=loaded.processor,
        loaded=loaded,
        module=module,
        examples=examples,
        input_names=input_names,
        output_names=output_names,
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest schema for eager model export.

The manifest is intentionally a front-end description. Runtime compatibility is
still governed by the component contracts emitted beside compiled engines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_ROLE_COMPONENTS = {"language", "visual", "action"}


def _as_dict(value: Any, field_name: str) -> Dict[str, Any]:
    """Validate a JSON manifest field that should be an object.

    Manifests are user-authored JSON. This helper turns missing
    fields into an empty dict and raises a targeted error when the
    user passes the wrong shape. That makes failures point at the
    manifest field, not at some later internal AttributeError.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Manifest field {field_name!r} must be an object")


def _as_list(value: Any, field_name: str) -> List[str]:
    """Normalize a manifest field that names one or more strings.

    The CLI and JSON files often accept either ``"a,b"`` or
    ``["a", "b"]``. Internally we keep one representation: a list
    of strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Manifest field {field_name!r} must be a list or string")


def _normalize_axis_map(value: Any) -> Dict[str, Dict[int, str]]:
    """Convert JSON dynamic-axis metadata into Python-friendly keys.

    JSON object keys are always strings, but Torch/TensorRT axis
    indices are integers. This converts ``{"input": {"0":
    "batch"}}`` into ``{"input": {0: "batch"}}``.
    """
    result: Dict[str, Dict[int, str]] = {}
    for tensor_name, axes in _as_dict(value, "dynamic_axes").items():
        result[str(tensor_name)] = {
            int(axis): str(axis_name)
            for axis, axis_name in _as_dict(axes, "dynamic_axes.*").items()
        }
    return result


@dataclass(frozen=True)
class EagerExportRole:
    """One model component to capture or compile from an eager root model.

    A role is the bridge between a Python model and an Edge runtime
    component. For example, one eager policy model may expose an
    ``action`` role whose module path points at an action-head
    wrapper, plus metadata describing its example inputs, output
    names, runtime contract, and destination directory.
    """

    name: str
    component: str
    module_path: str = ""
    exported_program: Optional[str] = None
    contract: Optional[str] = None
    example_inputs: Optional[str] = None
    example_kwargs: Mapping[str, Any] = field(default_factory=dict)
    packager: Optional[str] = None
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=lambda: ["output"])
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    output_dir: Optional[str] = None
    engine_path: Optional[str] = None
    engine_filename: Optional[str] = None
    compile_torchtrt: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_manifest(name: str, value: Any) -> "EagerExportRole":
        """Parse one ``roles.<name>`` block from a manifest.

        The compact form ``"action": "model.action_head"`` is
        expanded into the same structure as the verbose object form.
        Validation happens here so the exporter can use typed
        dataclasses afterward.
        """
        if isinstance(value, str):
            data = {"module": value}
        else:
            data = _as_dict(value, f"roles.{name}")

        component = str(data.get("component") or data.get("role") or name)
        if component not in _ROLE_COMPONENTS:
            component = str(data.get("component") or "custom")

        module_path = str(
            data.get("module")
            or data.get("path")
            or data.get("module_path")
            or ""
        )
        exported_program = data.get("exported_program") or data.get(
            "exported_program_path") or data.get("pt2")
        if not module_path and not exported_program:
            raise ValueError(
                f"Role {name!r} must specify either module/path or exported_program"
            )

        return EagerExportRole(
            name=name,
            component=component,
            module_path=module_path,
            exported_program=exported_program,
            contract=data.get("contract"),
            example_inputs=data.get("example_inputs"),
            example_kwargs=_as_dict(data.get("example_kwargs"), "example_kwargs"),
            packager=data.get("packager"),
            input_names=_as_list(data.get("input_names"), "input_names"),
            output_names=_as_list(data.get("output_names"), "output_names"),
            dynamic_axes=_normalize_axis_map(data.get("dynamic_axes")),
            output_dir=data.get("output_dir"),
            engine_path=data.get("engine_path"),
            engine_filename=data.get("engine_filename"),
            compile_torchtrt=data.get("compile_torchtrt"),
            metadata=_as_dict(data.get("metadata"), "metadata"),
        )

    def default_engine_filename(self) -> str:
        """Return the conventional engine filename for this role.

        Runtime code expects stable names such as ``llm.engine``,
        ``visual.engine``, and ``action.engine``. Custom roles fall
        back to ``<role-name>.engine``.
        """
        if self.engine_filename:
            return self.engine_filename
        if self.component == "language":
            return "llm.engine"
        if self.component == "visual":
            return "visual.engine"
        if self.component == "action":
            return "action.engine"
        return f"{self.name}.engine"


@dataclass(frozen=True)
class EagerExportManifest:
    """Top-level eager export manifest.

    The manifest says how to load an already-working eager model,
    which roles to extract from it, and where to place exported
    artifacts. It intentionally does not encode model-family logic;
    that belongs in HF strategies.
    """

    loader: Optional[str] = None
    loader_kwargs: Mapping[str, Any] = field(default_factory=dict)
    example_inputs: Optional[str] = None
    example_kwargs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)
    roles: Mapping[str, EagerExportRole] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "EagerExportManifest":
        """Validate raw JSON data and convert it into a manifest."""
        raw = dict(data)
        loader = raw.get("loader")

        roles_raw = _as_dict(raw.get("roles"), "roles")
        if not roles_raw:
            raise ValueError("Eager export manifest requires at least one role")

        return EagerExportManifest(
            loader=str(loader) if loader else None,
            loader_kwargs=_as_dict(raw.get("loader_kwargs"), "loader_kwargs"),
            example_inputs=raw.get("example_inputs"),
            example_kwargs=_as_dict(raw.get("example_kwargs"), "example_kwargs"),
            outputs={
                str(key): str(value)
                for key, value in _as_dict(raw.get("outputs"), "outputs").items()
            },
            roles={
                str(name): EagerExportRole.from_manifest(str(name), value)
                for name, value in roles_raw.items()
            },
            metadata=_as_dict(raw.get("metadata"), "metadata"),
        )


def load_manifest(path: str | Path) -> EagerExportManifest:
    """Load and validate an eager export manifest JSON file.

    This is the main entry point used by
    ``EdgeExport.from_manifest_path``.
    """
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return EagerExportManifest.from_dict(data)

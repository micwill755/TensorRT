# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared component contract schema for Edge-LLM exports.

Component contracts describe the tensor ABI of independently exported model
pieces such as visual encoders, language models, and action policies.  The
contract is intentionally about runtime I/O, not about a particular model id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


COMPONENT_CONTRACT_FORMAT_VERSION = 1
COMPONENT_VISUAL = "visual"
COMPONENT_LANGUAGE = "language"
COMPONENT_ACTION = "action"


@dataclass(frozen=True)
class ComponentContract:
    """Stable tensor contract for one exported model component."""

    component: str
    name: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...] = ("output",)
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    runtime_contract: str = "torchtrt"
    description: str = ""
    input_bindings: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    output_bindings: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "name": self.name,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "dynamic_axes": json_dynamic_axes(self.dynamic_axes),
            "runtime_contract": self.runtime_contract,
            "description": self.description,
            "binding_plan": build_binding_plan(
                list(self.input_names),
                list(self.output_names),
                input_bindings=self.input_bindings,
                output_bindings=self.output_bindings,
            ),
        }


def json_dynamic_axes(
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]],
) -> Dict[str, Dict[str, str]]:
    """Return JSON-friendly dynamic axes with stringified axis indices."""
    return {
        name: {str(axis): axis_name for axis, axis_name in axes.items()}
        for name, axes in (dynamic_axes or {}).items()
    }


def metadata_without_none(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Drop unset metadata values before writing manifests."""
    return {
        key: value
        for key, value in (metadata or {}).items()
        if value is not None
    }


def _binding_by_name(
    bindings: Optional[Tuple[Mapping[str, Any], ...] | List[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(binding["name"]): dict(binding)
        for binding in (bindings or [])
        if "name" in binding
    }


def _default_input_binding(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "source": name,
        "required": True,
    }


def _default_output_binding(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "semantic": name,
    }


def build_binding_plan(
    input_names: List[str],
    output_names: List[str],
    *,
    input_bindings: Optional[Tuple[Mapping[str, Any], ...] | List[Mapping[str, Any]]] = None,
    output_bindings: Optional[Tuple[Mapping[str, Any], ...] | List[Mapping[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a runtime binding plan from exported engine tensor names.

    The plan maps TensorRT tensor names to semantic runtime producers/consumers.
    It intentionally avoids model-name dispatch: C++ can bind by `source` and
    expose by `semantic` while preserving the concrete engine tensor names.
    """
    inputs_by_name = _binding_by_name(input_bindings)
    outputs_by_name = _binding_by_name(output_bindings)

    inputs = []
    for name in input_names:
        binding = dict(inputs_by_name.get(name, _default_input_binding(name)))
        binding["name"] = name
        binding.setdefault("source", name)
        binding.setdefault("required", True)
        inputs.append(binding)

    outputs = []
    for name in output_names:
        binding = dict(outputs_by_name.get(name, _default_output_binding(name)))
        binding["name"] = name
        binding.setdefault("semantic", name)
        outputs.append(binding)

    return {
        "inputs": inputs,
        "outputs": outputs,
    }


def build_component_contract_manifest(
    contract: ComponentContract,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    export_dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    contract_manifest: Optional[Mapping[str, Any]] = None,
    input_bindings: Optional[List[Mapping[str, Any]]] = None,
    output_bindings: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the common manifest shape for an exported component."""
    resolved_input_names = list(input_names or contract.input_names)
    resolved_output_names = list(output_names or contract.output_names)
    manifest: Dict[str, Any] = {
        "format_version": COMPONENT_CONTRACT_FORMAT_VERSION,
        "component": contract.component,
        "contract": dict(contract_manifest or contract.to_manifest_dict()),
        "engine_io": {
            "input_names": resolved_input_names,
            "output_names": resolved_output_names,
            "dynamic_axes": json_dynamic_axes(dynamic_axes or contract.dynamic_axes),
        },
        "binding_plan": build_binding_plan(
            resolved_input_names,
            resolved_output_names,
            input_bindings=input_bindings or contract.input_bindings,
            output_bindings=output_bindings or contract.output_bindings,
        ),
        "metadata": metadata_without_none(metadata),
    }
    if export_dynamic_axes is not None:
        manifest["engine_io"]["export_dynamic_axes"] = json_dynamic_axes(
            export_dynamic_axes
        )
    if artifacts is not None:
        manifest["artifacts"] = dict(artifacts)
    return manifest


def write_component_contract_manifest(
    output_dir: str | Path,
    contract: ComponentContract,
    *,
    filename: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    export_dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    contract_manifest: Optional[Mapping[str, Any]] = None,
    input_bindings: Optional[List[Mapping[str, Any]]] = None,
    output_bindings: Optional[List[Mapping[str, Any]]] = None,
) -> Path:
    """Write a component contract manifest with stable formatting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = build_component_contract_manifest(
        contract,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_dynamic_axes=export_dynamic_axes,
        metadata=metadata,
        artifacts=artifacts,
        contract_manifest=contract_manifest,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    manifest_path = output_path / filename
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path

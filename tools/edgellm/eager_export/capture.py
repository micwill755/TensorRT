# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example-input normalization and torch.export capture helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Any, Mapping, Optional

from .loader import call_with_supported_kwargs, import_object
from .manifest import EagerExportManifest, EagerExportRole


_EXAMPLE_SPEC_KEYS = {
    "args",
    "kwargs",
    "input_names",
    "output_names",
    "dynamic_axes",
    "dynamic_shapes",
}


@dataclass
class ExampleInputs:
    """Normalized example inputs for one eager role."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=lambda: ["output"])
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    dynamic_shapes: Any = None


def _normalize_axes(value: Any) -> dict[str, dict[int, str]]:
    if not value:
        return {}
    return {
        str(name): {
            int(axis): str(axis_name)
            for axis, axis_name in dict(axes).items()
        }
        for name, axes in dict(value).items()
    }


def normalize_example_inputs(value: Any) -> ExampleInputs:
    """Normalize hook output into args/kwargs plus optional export metadata."""
    if isinstance(value, ExampleInputs):
        return value

    if isinstance(value, Mapping):
        data = dict(value)
        if _EXAMPLE_SPEC_KEYS.intersection(data):
            args_value = data.get("args", ())
            if args_value is None:
                args = ()
            elif isinstance(args_value, tuple):
                args = args_value
            elif isinstance(args_value, list):
                args = tuple(args_value)
            else:
                args = (args_value,)
            return ExampleInputs(
                args=args,
                kwargs=dict(data.get("kwargs") or {}),
                input_names=[str(name) for name in data.get("input_names", [])],
                output_names=[str(name) for name in data.get("output_names", [])]
                or ["output"],
                dynamic_axes=_normalize_axes(data.get("dynamic_axes")),
                dynamic_shapes=data.get("dynamic_shapes"),
            )
        return ExampleInputs(kwargs=data)

    if isinstance(value, tuple):
        return ExampleInputs(args=value)
    if isinstance(value, list):
        return ExampleInputs(args=tuple(value))
    if value is None:
        return ExampleInputs()
    return ExampleInputs(args=(value,))


def _infer_forward_input_names(module: Any, count: int) -> list[str]:
    try:
        params = list(signature(module.forward).parameters)
    except (AttributeError, TypeError, ValueError):
        params = []
    return [
        params[idx] if idx < len(params) else f"input_{idx}"
        for idx in range(count)
    ]


def infer_input_names(module: Any, examples: ExampleInputs) -> list[str]:
    """Infer top-level input names when the manifest/hook omits them."""
    if examples.input_names:
        return list(examples.input_names)
    names = _infer_forward_input_names(module, len(examples.args))
    names.extend(str(name) for name in examples.kwargs)
    return names


def call_example_input_hook(
    hook_path: str,
    *,
    manifest: EagerExportManifest,
    role: EagerExportRole,
    module: Any,
    loaded: Any,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    hook_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    hook = import_object(hook_path)
    return call_with_supported_kwargs(
        hook,
        **dict(hook_kwargs or {}),
        manifest=manifest,
        role=role,
        module=module,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        processor=loaded.processor,
        loaded=loaded,
        device=device,
        dtype=dtype,
    )


def resolve_role_examples(
    manifest: EagerExportManifest,
    role: EagerExportRole,
    module: Any,
    loaded: Any,
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> ExampleInputs:
    """Resolve examples from role-specific hook or top-level hook."""
    if role.example_inputs:
        raw = call_example_input_hook(
            role.example_inputs,
            manifest=manifest,
            role=role,
            module=module,
            loaded=loaded,
            device=device,
            dtype=dtype,
            hook_kwargs=role.example_kwargs,
        )
        return normalize_example_inputs(raw)

    if not manifest.example_inputs:
        return ExampleInputs()

    raw_all = call_example_input_hook(
        manifest.example_inputs,
        manifest=manifest,
        role=role,
        module=module,
        loaded=loaded,
        device=device,
        dtype=dtype,
        hook_kwargs=manifest.example_kwargs,
    )
    if isinstance(raw_all, Mapping) and role.name in raw_all:
        return normalize_example_inputs(raw_all[role.name])
    if isinstance(raw_all, Mapping) and role.component in raw_all:
        return normalize_example_inputs(raw_all[role.component])
    return normalize_example_inputs(raw_all)


def capture_exported_program(
    module: Any,
    examples: ExampleInputs,
    *,
    strict: bool = False,
) -> Any:
    """Capture a role with torch.export and return the ExportedProgram."""
    import torch

    if hasattr(module, "eval"):
        module.eval()
    with torch.inference_mode():
        try:
            return torch.export.export(
                module,
                examples.args,
                kwargs=examples.kwargs or None,
                dynamic_shapes=examples.dynamic_shapes,
                strict=strict,
            )
        except Exception:
            from torch.export._trace import _export

            return _export(
                module,
                examples.args,
                kwargs=examples.kwargs or None,
                dynamic_shapes=examples.dynamic_shapes,
                strict=strict,
                prefer_deferred_runtime_asserts_over_guards=True,
            )


def load_exported_program(path: str | Path) -> Any:
    """Load a torch.export ExportedProgram from disk."""
    import torch

    return torch.export.load(Path(path).expanduser().resolve())


def save_exported_program(exported_program: Any, output_path: str | Path) -> Path:
    """Save an ExportedProgram to disk."""
    import torch

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, path)
    return path

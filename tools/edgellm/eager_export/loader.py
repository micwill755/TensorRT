# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import and loading helpers for eager export manifests."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .manifest import EagerExportManifest


@dataclass
class LoadedEagerModel:
    """Normalized result from a user-provided eager model loader.

    Loader hooks are allowed to return several shapes: a model, a
    tuple, or a mapping. The exporter converts all of them into this
    single object so later stages do not need to care how loading was
    implemented.
    """

    model: Any
    tokenizer: Any = None
    processor: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


def import_object(path: str) -> Any:
    """Import a Python object from a string path.

    Manifests name hooks as strings so they can live outside this
    package. Both ``package.module:object`` and
    ``package.module.object`` forms are accepted. Nested attributes
    after the object name are resolved one by one.
    """
    if not path or not path.strip():
        raise ValueError("Expected a fully-qualified import path")
    normalized = path.strip()
    if ":" in normalized:
        module_name, object_name = normalized.split(":", 1)
    else:
        module_name, object_name = normalized.rsplit(".", 1)
    if not module_name or not object_name:
        raise ValueError(f"Expected a fully-qualified import path, got {path!r}")
    module = importlib.import_module(module_name)
    current = module
    for part in object_name.split("."):
        current = getattr(current, part)
    return current


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a user hook while tolerating optional exporter arguments.

    Hooks can opt into only the parameters they need. If the hook
    accepts ``**kwargs`` we pass everything; otherwise we inspect the
    signature and drop unsupported names. This keeps old hooks working
    as the exporter grows new context arguments.
    """
    try:
        params = signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)

    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in params
    }
    return fn(**accepted)


def normalize_loaded_model(value: Any) -> LoadedEagerModel:
    """Normalize common loader return shapes into ``LoadedEagerModel``.

    Supported returns are:
    - ``LoadedEagerModel``: already normalized.
    - mapping with ``model`` or ``root_model``: extra keys become
      ``extra`` metadata.
    - tuple: interpreted as ``(model, tokenizer, processor, ...)``.
    - anything else: treated as the model itself.
    """
    if isinstance(value, LoadedEagerModel):
        return value

    if isinstance(value, Mapping):
        data = dict(value)
        model = data.pop("model", data.pop("root_model", None))
        if model is None:
            raise ValueError(
                "Loader returned a mapping but did not include 'model'"
            )
        tokenizer = data.pop("tokenizer", None)
        processor = data.pop("processor", None)
        return LoadedEagerModel(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            extra=data,
        )

    if isinstance(value, tuple):
        if not value:
            raise ValueError("Loader returned an empty tuple")
        extra: Dict[str, Any] = {}
        if len(value) > 3:
            extra["loader_tail"] = value[3:]
        return LoadedEagerModel(
            model=value[0],
            tokenizer=value[1] if len(value) > 1 else None,
            processor=value[2] if len(value) > 2 else None,
            extra=extra,
        )

    return LoadedEagerModel(model=value)


def load_eager_model(
    manifest: EagerExportManifest,
    *,
    manifest_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> LoadedEagerModel:
    """Call the manifest loader and normalize its result.

    If no loader is configured, the manifest must refer to an
    already-saved ExportedProgram instead of a live Python module.
    """
    if not manifest.loader:
        return LoadedEagerModel(model=None)
    loader = import_object(manifest.loader)
    loaded = call_with_supported_kwargs(
        loader,
        **dict(manifest.loader_kwargs),
        manifest=manifest,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        device=device,
        dtype=dtype,
    )
    return normalize_loaded_model(loaded)

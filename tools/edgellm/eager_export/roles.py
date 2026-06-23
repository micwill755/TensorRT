# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Role resolution helpers for eager export.

A manifest uses strings such as ``model.action_head``. This module
turns those strings into concrete Python objects before capture or
compilation begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .manifest import EagerExportManifest, EagerExportRole


@dataclass(frozen=True)
class ResolvedRole:
    """A manifest role resolved to a concrete Python module/object.

    ``spec`` keeps the manifest metadata. ``module`` is the actual
    object that will be exported, or ``None`` when the role points at
    an existing ExportedProgram on disk.
    """

    spec: EagerExportRole
    module: Any = None


def resolve_dotted_path(root: Any, path: str) -> Any:
    """Resolve dot-separated attributes, mapping keys, or list indices.

    This lets one manifest syntax work for normal modules, dict-like
    loader returns, and simple lists/tuples. The leading ``model``
    part is tolerated so users can write paths the same way they
    reason about the root model.
    """
    if path in ("", ".", "self"):
        return root

    current = root
    parts = [part for part in path.split(".") if part]
    if parts and parts[0] == "model" and not hasattr(current, "model"):
        parts = parts[1:]

    for part in parts:
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def resolve_roles(
    model: Any,
    manifest: EagerExportManifest,
    selected: list[str] | None = None,
) -> dict[str, ResolvedRole]:
    """Resolve selected manifest roles against the eager root model.

    The result is a mapping from role name to the resolved Python
    module plus its original manifest spec. This is the handoff into
    ``EdgeExport.run``.
    """
    wanted = selected or list(manifest.roles)
    resolved: dict[str, ResolvedRole] = {}
    for name in wanted:
        if name not in manifest.roles:
            known = ", ".join(sorted(manifest.roles))
            raise ValueError(f"Unknown eager export role {name!r}. Known: {known}")
        spec = manifest.roles[name]
        module = None
        if spec.module_path:
            if model is None:
                raise ValueError(
                    f"Role {name!r} specifies module_path but manifest has no loader"
                )
            module = resolve_dotted_path(model, spec.module_path)
        resolved[name] = ResolvedRole(
            spec=spec,
            module=module,
        )
    return resolved

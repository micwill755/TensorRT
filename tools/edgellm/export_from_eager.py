# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export already-running eager PyTorch model components.

This is the eager-model export path: the user owns Hugging Face/custom loading
and proves the model runs in eager mode. Edge-LLM then captures selected roles
and emits artifacts that can target the existing generic runtimes.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.edgellm.edge_export import EdgeExport, EdgeExportOptions


def _split_roles(values: Optional[list[str]]) -> Optional[list[str]]:
    """Normalize repeated/comma-separated ``--role`` flags.

    Returning ``None`` means "use every role in the manifest".
    """
    if not values:
        return None
    roles: list[str] = []
    for value in values:
        roles.extend(item.strip() for item in value.split(",") if item.strip())
    return roles or None


def main() -> None:
    """Parse CLI flags and run the eager-manifest export path.

    This command starts after the user has solved model loading. The
    manifest points at a loader or ExportedProgram plus the runtime
    roles to export.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Capture or compile Edge-LLM components from a user-provided eager "
            "PyTorch model manifest."
        )
    )
    parser.add_argument("--manifest",
                        type=str,
                        required=True,
                        help="Path to eager export manifest JSON")
    parser.add_argument("--role",
                        action="append",
                        default=None,
                        help="Role name(s) to export. May be repeated or comma-separated.")
    parser.add_argument("--output_root",
                        type=str,
                        default=None,
                        help="Fallback output root when a role has no output_dir")
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        help="Device hint passed to loader/example hooks")
    parser.add_argument("--dtype",
                        type=str,
                        default=None,
                        help="Dtype hint passed to loader/example hooks")
    parser.add_argument("--dry_run",
                        action="store_true",
                        help="Load and resolve roles without capture or compile")
    parser.add_argument("--capture",
                        action="store_true",
                        help="Save torch.export ExportedProgram artifacts")
    parser.add_argument("--compile_torchtrt",
                        action="store_true",
                        help="Compile compatible roles through the existing Torch-TRT helper")
    parser.add_argument("--strict_export",
                        action="store_true",
                        help="Use strict torch.export capture")
    args = parser.parse_args()

    try:
        exporter = EdgeExport.from_manifest_path(Path(args.manifest))
        exporter.run(
            EdgeExportOptions(
                selected_roles=_split_roles(args.role),
                output_root=args.output_root,
                device=args.device,
                dtype=args.dtype,
                dry_run=args.dry_run,
                capture=args.capture,
                compile_torchtrt=args.compile_torchtrt,
                strict_export=args.strict_export,
            )
        )
        print("Eager export completed successfully!")
    except Exception as exc:
        print(f"Error during eager export: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

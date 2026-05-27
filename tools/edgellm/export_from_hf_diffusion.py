# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype HF role-resolution front door for Edge exports.

This script implements the first lightweight version of the "diffusion"
resolver: start from noisy HF metadata, progressively denoise it into ranked
language / visual / action candidates, and write an inspectable report.  It is
additive to ``export_from_hf.py`` and does not compile engines by itself yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.edgellm.hf_export.resolution import (
    candidate_manifest_from_plan,
    resolve_hf_roles,
    write_resolution_report,
)
from tools.edgellm.hf_export.source import HFModelSource, HF_STRATEGY_FAMILIES


def _split_roles(values: Optional[list[str]]) -> tuple[str, ...]:
    """Normalize repeated or comma-separated ``--role`` flags."""
    if not values:
        return ()
    roles: list[str] = []
    for value in values:
        roles.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(roles)


def _default_report_path(args: argparse.Namespace) -> Path:
    """Choose a stable report path when the caller does not provide one."""
    if args.report_output:
        return Path(args.report_output).expanduser().resolve()
    if args.output_root:
        return Path(args.output_root).expanduser().resolve() / "hf_resolution_report.json"
    return Path.cwd() / "hf_resolution_report.json"


def _default_manifest_path(args: argparse.Namespace) -> Path:
    """Choose a stable candidate-manifest path."""
    if args.candidate_manifest_output:
        return Path(args.candidate_manifest_output).expanduser().resolve()
    if args.output_root:
        return Path(args.output_root).expanduser().resolve() / "hf_resolution_candidate_manifest.json"
    return Path.cwd() / "hf_resolution_candidate_manifest.json"


def main() -> None:
    """Resolve likely Edge roles from a HF model id/path and write a report."""
    parser = argparse.ArgumentParser(
        description=(
            "Prototype diffusion-style HF resolver for Edge-LLM exports. "
            "Writes a ranked role-resolution report and optional candidate manifest."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--task", default=None, help="Optional task/family hint")
    parser.add_argument(
        "--family",
        default="auto",
        choices=["auto", *HF_STRATEGY_FAMILIES],
        help="Family override. Default: auto detect.",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=None,
        help="Role(s) to resolve. May be repeated or comma-separated: language, visual, action.",
    )
    parser.add_argument("--output_root", default=None, help="Root directory for generated resolver artifacts")
    parser.add_argument("--report_output", default=None, help="Where to write hf_resolution_report.json")
    parser.add_argument(
        "--write_candidate_manifest",
        action="store_true",
        help="Also write an eager-style candidate manifest from selected roles.",
    )
    parser.add_argument(
        "--candidate_manifest_output",
        default=None,
        help="Where to write the optional candidate manifest.",
    )
    parser.add_argument("--model_class", default=None, help="Optional fully-qualified custom model class")
    parser.add_argument("--language_module", default=None, help="Explicit language module path hint")
    parser.add_argument("--lm_head_module", default=None, help="Explicit LM head module path hint")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path hint")
    parser.add_argument("--vision_module", default=None, help="Explicit visual module path hint")
    parser.add_argument("--projector_module", default=None, help="Explicit visual projector path hint")
    parser.add_argument("--processor_model", default=None, help="Processor id/path hint")
    parser.add_argument("--action_module", default=None, help="Explicit action module path hint")
    parser.add_argument("--attn_implementation", default="eager", help="Attention implementation hint")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_kv_cache_capacity", type=int, default=4096)
    parser.add_argument("--action_batch_size", type=int, default=2)
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.55,
        help="Minimum candidate confidence required for automatic selection.",
    )
    parser.add_argument("--print_report", action="store_true", help="Print the generated report JSON")
    args = parser.parse_args()

    try:
        source = HFModelSource.from_args(args, model_attr="model")
        roles = _split_roles(args.role)
        plan = resolve_hf_roles(
            source,
            roles=roles,
            min_confidence=float(args.min_confidence),
        )

        report_path = write_resolution_report(plan, _default_report_path(args))
        print(f"[export_from_hf_diffusion] Wrote resolution report to {report_path}")

        if args.write_candidate_manifest:
            manifest = candidate_manifest_from_plan(
                plan,
                source,
                output_root=args.output_root,
                action_batch_size=int(args.action_batch_size),
                max_kv_cache_capacity=int(args.max_kv_cache_capacity),
            )
            manifest_path = _default_manifest_path(args)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[export_from_hf_diffusion] Wrote candidate manifest to {manifest_path}")

        if args.print_report:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))

        if plan.unresolved_roles:
            unresolved = ", ".join(plan.unresolved_roles)
            print(
                "[export_from_hf_diffusion] Unresolved role(s): "
                f"{unresolved}. Pass explicit module hints or lower --min_confidence."
            )
    except Exception as exc:
        print(f"Error during HF diffusion resolution: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct Hugging Face front door for Edge-LLM exports.

This follows the run_hf.py strategy shape: detect a broad HF family, build an
Edge role manifest where needed, then reuse the Edge component exporters.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.edgellm.edge_export import EdgeExport, EdgeExportOptions
from tools.edgellm.hf_export.source import HFModelSource
from tools.edgellm.hf_export.strategies.base import HFExportConfig
from tools.edgellm.hf_export.strategies.vla import VLAEdgeStrategy


def _split_roles(values: Optional[list[str]]) -> tuple[str, ...]:
    if not values:
        return ("action",)
    roles: list[str] = []
    for value in values:
        roles.extend(item.strip() for item in value.split(",") if item.strip())
    return tuple(roles or ("action",))


def _ordered_roles(roles: tuple[str, ...]) -> tuple[str, ...]:
    ordered = [role for role in ("language", "visual", "action") if role in roles]
    ordered.extend(role for role in roles if role not in ordered)
    return tuple(ordered)


def _build_strategy(family: str, cfg: HFExportConfig):
    if family == "vla":
        return VLAEdgeStrategy(cfg)
    raise NotImplementedError(
        "Direct Hugging Face Edge export currently wires the VLA family first. "
        f"Family {family!r} was detected/requested. LLM/VLM families should use "
        "the existing component exporters until their role strategies are folded in."
    )


def _role_output_dir(args: argparse.Namespace, role: str) -> str:
    explicit = {
        "language": args.llm_output_dir,
        "visual": args.visual_output_dir,
        "action": args.action_output_dir,
    }.get(role)
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    if not args.output_root:
        option = "--llm_output_dir" if role == "language" else f"--{role}_output_dir"
        raise ValueError(f"{option} is required when --output_root is not provided")
    output_root = Path(args.output_root).expanduser().resolve()
    if role == "language":
        return str(output_root / "language")
    return str(output_root / role)


def _default_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_output:
        return Path(args.manifest_output).expanduser().resolve()
    if args.output_root:
        return Path(args.output_root).expanduser().resolve() / "hf_export_manifest.json"
    if args.action_output_dir:
        return Path(args.action_output_dir).expanduser().resolve().parent / "hf_export_manifest.json"
    return Path.cwd() / "hf_export_manifest.json"


def _export_language_role(args: argparse.Namespace, source: HFModelSource) -> None:
    from tensorrt_edgellm.onnx_export.llm_export import export_llm_model

    output_dir = _role_output_dir(args, "language")
    print(f"[export_from_hf] Exporting language role to {output_dir}")
    export_llm_model(
        **source.language_kwargs(),
        output_dir=output_dir,
        device=args.device or "cuda",
        fp8_kv_cache=args.fp8_kv_cache,
        trt_native_ops=args.trt_native_ops,
        fp8_embedding=args.fp8_embedding,
        torchtrt=args.compile_torchtrt,
        torchtrt_engine_path=str(Path(output_dir) / "llm.engine") if args.compile_torchtrt else None,
    )


def _export_visual_role(args: argparse.Namespace, source: HFModelSource) -> None:
    from tensorrt_edgellm.onnx_export.visual_export import visual_export

    output_dir = _role_output_dir(args, "visual")
    print(f"[export_from_hf] Exporting visual role to {output_dir}")
    visual_export(
        **source.visual_kwargs(),
        output_dir=output_dir,
        dtype=args.dtype,
        quantization=args.visual_quantization,
        dataset_dir=args.visual_dataset_dir,
        device=args.device or "cuda",
        torchtrt=args.compile_torchtrt,
        torchtrt_engine_path=str(Path(output_dir) / "visual.engine") if args.compile_torchtrt else None,
        prompt=args.visual_prompt,
        image_size=args.image_size,
        enable_vit_attention_plugin=not args.no_vit_attention_plugin,
    )


def _run_edge_export(args: argparse.Namespace, manifest_path: Path, roles: tuple[str, ...]) -> None:
    exporter = EdgeExport.from_manifest_path(manifest_path)
    exporter.run(
        EdgeExportOptions(
            selected_roles=roles,
            output_root=args.output_root,
            device=args.device,
            dtype=args.dtype,
            dry_run=args.dry_run,
            capture=args.capture,
            compile_torchtrt=args.compile_torchtrt,
            strict_export=args.strict_export,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Edge-LLM roles directly from a Hugging Face model path/id."
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--task", default=None, help="Optional HF task/family override")
    parser.add_argument(
        "--family",
        default="auto",
        choices=["auto", "llm", "vlm", "vla"],
        help="Family override. Default: auto detect.",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=None,
        help="Role(s) to export. May be repeated or comma-separated: language, visual, action.",
    )
    parser.add_argument("--output_root", default=None, help="Root output directory for role artifacts")
    parser.add_argument("--llm_output_dir", default=None, help="Output directory for language/LLM artifacts")
    parser.add_argument("--visual_output_dir", default=None, help="Output directory for visual role artifacts")
    parser.add_argument("--action_output_dir", default=None, help="Output directory for action role artifacts")
    parser.add_argument("--manifest_output", default=None, help="Where to write the generated eager manifest")
    parser.add_argument("--manifest_only", action="store_true", help="Only write the generated eager manifest")
    parser.add_argument("--model_class", default=None, help="Optional fully-qualified model class")
    parser.add_argument("--language_module", default=None, help="Dotted module path for language role extraction")
    parser.add_argument("--lm_head_module", default=None, help="Dotted module path for LM head extraction")
    parser.add_argument("--instantiate_from_config", action="store_true", help="Instantiate custom LLM class from config")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path for language role exports")
    parser.add_argument("--vision_module", default=None, help="Dotted module path for visual role extraction")
    parser.add_argument("--projector_module", default=None, help="Dotted module path for visual projector extraction")
    parser.add_argument("--processor_model", default=None, help="Processor model id/path for visual role exports")
    parser.add_argument("--no_processor", action="store_true", help="Do not load/save visual processor metadata")
    parser.add_argument("--add_common_vlm_aliases", action="store_true", help="Install common VLM aliases while loading")
    parser.add_argument("--attn_implementation", default="eager", help="Attention implementation hint while loading")
    parser.add_argument("--device", default=None, help="Device hint passed to role exporters")
    parser.add_argument("--dtype", default="fp16", help="Dtype hint passed to role exporters")
    parser.add_argument("--visual_quantization", choices=["fp8"], default=None, help="Visual quantization mode")
    parser.add_argument("--visual_dataset_dir", default="lmms-lab/MMMU", help="Dataset for visual quantization")
    parser.add_argument("--visual_prompt", default="Describe this image.", help="Prompt for processor-derived visual sample")
    parser.add_argument("--image_size", type=int, default=None, help="Optional square visual image size override")
    parser.add_argument("--no_vit_attention_plugin", action="store_true", help="Disable ViT attention plugin replacement")
    parser.add_argument("--fp8_kv_cache", action="store_true", help="Use FP8 KV cache for language role")
    parser.add_argument("--trt_native_ops", action="store_true", help="Use TensorRT native ops for language role")
    parser.add_argument("--fp8_embedding", action="store_true", help="Use FP8 embedding table for language role")
    parser.add_argument("--max_kv_cache_capacity", type=int, default=4096)
    parser.add_argument("--action_batch_size", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true", help="Resolve roles without capture/compile")
    parser.add_argument("--capture", action="store_true", help="Capture ExportedProgram artifacts for eager roles")
    parser.add_argument("--compile_torchtrt", action="store_true", help="Compile role engines with Torch-TRT")
    parser.add_argument("--strict_export", action="store_true", help="Use strict torch.export capture")
    args = parser.parse_args()

    try:
        roles = _ordered_roles(_split_roles(args.role))
        unknown_roles = set(roles) - {"language", "visual", "action"}
        if unknown_roles:
            raise ValueError(f"Unsupported role(s): {', '.join(sorted(unknown_roles))}")

        source = HFModelSource.from_args(args, model_attr="model")
        family = source.detected_family()
        print(f"[export_from_hf] Family = {family}")
        cfg = HFExportConfig(
            model=source.model,
            family=family,
            roles=roles,
            output_root=args.output_root,
            action_output_dir=args.action_output_dir,
            model_class=source.model_class,
            attn_implementation=source.attn_implementation,
            max_kv_cache_capacity=args.max_kv_cache_capacity,
            action_batch_size=args.action_batch_size,
            source=source,
        )

        action_manifest_path: Optional[Path] = None
        if "action" in roles:
            action_cfg = replace(cfg, roles=("action",))
            strategy = _build_strategy(family, action_cfg)
            manifest = strategy.build_manifest()
            action_manifest_path = _default_manifest_path(args)
            action_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            action_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[export_from_hf] Wrote eager manifest to {action_manifest_path}")
        elif args.manifest_only:
            print("[export_from_hf] No eager manifest is generated for language/visual-only exports.")

        if args.manifest_only:
            return

        if args.dry_run:
            if "action" in roles and action_manifest_path is not None:
                _run_edge_export(args, action_manifest_path, ("action",))
            for role in roles:
                if role in {"language", "visual"}:
                    print(f"[export_from_hf] Dry run: resolved {role} role.")
            return

        for role in roles:
            if role == "language":
                _export_language_role(args, source)
            elif role == "visual":
                _export_visual_role(args, source)
            elif role == "action":
                if action_manifest_path is None:
                    raise RuntimeError("Internal error: action role requested without an eager manifest.")
                _run_edge_export(args, action_manifest_path, ("action",))
    except Exception as exc:
        print(f"Error during HF Edge export: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

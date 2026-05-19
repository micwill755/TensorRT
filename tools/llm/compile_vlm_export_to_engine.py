#!/usr/bin/env python3
"""Compile an existing VLM vision torch.export artifact directly to a TRT engine.

This is the engine-only companion to export_vlm_to_tensorrt.py's compile path.
It starts from a saved ExportedProgram and manifest, skips Hugging Face model
loading, and calls Torch-TensorRT's direct serialized-engine API:

    torch.export .pt2 -> serialized TensorRT .engine

The ViT custom op and Torch-TensorRT converter are registered by importing
plugin_utils_vit. The Edge-LLM plugin library is loaded before conversion so
TensorRT can find ViTAttentionPlugin in its plugin registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from plugin_utils_vit import load_plugin


DEFAULT_OUTPUT_DIR = "/tmp/vlm_vision_tensorrt_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a saved VLM vision torch.export artifact directly to a "
            "serialized TensorRT engine."
        )
    )
    parser.add_argument(
        "--input_export",
        required=True,
        help="Path to a torch.export .pt2 artifact.",
    )
    parser.add_argument(
        "--input_manifest",
        default=None,
        help=(
            "Manifest describing the export inputs. Defaults to manifest.json "
            "next to --input_export."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for the direct engine and manifest. Defaults to the "
            "input export directory."
        ),
    )
    parser.add_argument(
        "--output_engine",
        default=None,
        help="Optional explicit output .engine path.",
    )
    parser.add_argument(
        "--output_manifest",
        default=None,
        help=(
            "Optional explicit output manifest path. Defaults to "
            "direct_engine_manifest.json in --output_dir."
        ),
    )
    parser.add_argument(
        "--plugin_path",
        default=None,
        help="Optional path to libNvInfer_edgellm_plugin.so.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target compile device, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--min_block_size",
        type=int,
        default=1,
        help="Torch-TensorRT minimum partition block size.",
    )
    parser.add_argument(
        "--workspace_size",
        type=int,
        default=0,
        help="TensorRT workspace size in bytes. 0 uses Torch-TensorRT default.",
    )
    parser.add_argument(
        "--optimization_level",
        type=int,
        default=None,
        help="Optional TensorRT builder optimization level.",
    )
    parser.add_argument(
        "--require_full_compilation",
        action="store_true",
        help="Require the exported program to compile fully to TensorRT.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32. By default this script mirrors export_vlm_to_tensorrt.py and disables TF32.",
    )
    parser.add_argument(
        "--no_fp32_acc",
        action="store_true",
        help="Disable FP32 accumulation for matmul layers.",
    )
    parser.add_argument(
        "--skip_engine_inspection",
        action="store_true",
        help="Skip deserializing the produced engine to record I/O tensor names.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def dtype_from_string(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.")
    if normalized in ("float16", "bfloat16", "float32"):
        return dtype_from_name(normalized)
    return {
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.long,
        "bool": torch.bool,
    }[normalized]


def load_manifest(export_path: Path, manifest_path: str | None) -> tuple[Path, Dict[str, Any]]:
    path = Path(manifest_path) if manifest_path is not None else export_path.parent / "manifest.json"
    if not path.exists():
        raise RuntimeError(
            f"Could not find manifest for {export_path}. Pass --input_manifest explicitly."
        )
    return path, json.loads(path.read_text())


def _manifest_tensor_specs(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    specs.update(manifest.get("tensor_inputs", {}))
    inputs = manifest.get("inputs", {})
    pixel_shape = inputs.get("pixel_values_shape")
    pixel_dtype = inputs.get("pixel_values_dtype")
    if pixel_shape is not None and pixel_dtype is not None:
        specs["pixel_values"] = {
            "shape": pixel_shape,
            "dtype": pixel_dtype,
        }
    specs.update(manifest.get("core_inputs", {}))
    return specs


def _candidate_input_names(name: str) -> List[str]:
    candidates = [name]
    if name.startswith("arg"):
        candidates.append(name[3:])
    if "_" in name:
        candidates.append(name.split("_", 1)[1])
    return candidates


def compile_inputs_from_manifest(
    exported_program: torch.export.ExportedProgram,
    manifest: Dict[str, Any],
) -> List[Any]:
    import torch_tensorrt

    specs = _manifest_tensor_specs(manifest)
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_inputs = list(getattr(graph_signature, "user_inputs", ()))
    if not user_inputs:
        user_inputs = ["pixel_values", *manifest.get("core_inputs", {}).keys()]

    compile_inputs: List[Any] = []
    missing: List[str] = []
    for name in user_inputs:
        spec = None
        for candidate in _candidate_input_names(str(name)):
            if candidate in specs:
                spec = specs[candidate]
                break
        if spec is None:
            missing.append(str(name))
            continue

        compile_inputs.append(
            torch_tensorrt.Input(
                shape=tuple(spec["shape"]),
                dtype=dtype_from_string(spec["dtype"]),
            )
        )

    if missing:
        raise RuntimeError(
            "Could not infer Torch-TensorRT input specs for exported inputs "
            f"{missing}. Known manifest tensors: {sorted(specs.keys())}"
        )
    return compile_inputs


def artifact_prefix_from_export_path(export_path: Path) -> str:
    export_stem = export_path.stem
    suffix = "_exported_program"
    return export_stem[: -len(suffix)] if export_stem.endswith(suffix) else export_stem


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def inspect_serialized_engine(engine_bytes: bytes) -> Dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the direct engine output.")

    input_names: List[str] = []
    output_names: List[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    return {
        "input_binding_names": input_names,
        "output_binding_names": output_names,
        "num_io_tensors": engine.num_io_tensors,
    }


def add_direct_engine_metadata(
    manifest: Dict[str, Any],
    engine_path: Path,
    output_dir: Path,
    engine_bytes: bytes,
    args: argparse.Namespace,
    engine_info: Dict[str, Any],
) -> None:
    manifest.setdefault("artifacts", {})
    manifest["artifacts"]["direct_tensorrt_engine"] = relative_or_absolute(
        engine_path,
        output_dir,
    )
    manifest["direct_tensorrt_engine"] = {
        "path": relative_or_absolute(engine_path, output_dir),
        "bytes": len(engine_bytes),
        **engine_info,
    }
    manifest["direct_tensorrt_compile"] = {
        "api": "torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine",
        "min_block_size": args.min_block_size,
        "workspace_size": args.workspace_size,
        "optimization_level": args.optimization_level,
        "require_full_compilation": bool(args.require_full_compilation),
        "disable_tf32": not args.allow_tf32,
        "use_fp32_acc": not args.no_fp32_acc,
    }
    manifest["direct_engine_runtime_requirements"] = {
        "custom_op_module": "plugin_utils_vit",
        "tensorrt_plugin_path": args.plugin_path,
        "load_order": [
            "import plugin_utils_vit",
            "ctypes.CDLL(tensorrt_plugin_path)",
            "tensorrt.Runtime(...).deserialize_cuda_engine(engine_bytes)",
        ],
    }


def main() -> None:
    args = parse_args()
    export_path = Path(args.input_export)
    if not export_path.exists():
        raise RuntimeError(f"Could not find input export at {export_path}.")

    output_dir = Path(args.output_dir) if args.output_dir is not None else export_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_prefix = artifact_prefix_from_export_path(export_path)
    engine_path = (
        Path(args.output_engine)
        if args.output_engine is not None
        else output_dir / f"{artifact_prefix}_direct.engine"
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    output_manifest_path = (
        Path(args.output_manifest)
        if args.output_manifest is not None
        else output_dir / "direct_engine_manifest.json"
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path, manifest = load_manifest(export_path, args.input_manifest)
    exported_program = torch.export.load(export_path)
    compile_inputs = compile_inputs_from_manifest(exported_program, manifest)

    import torch_tensorrt
    from torch_tensorrt.dynamo import convert_exported_program_to_serialized_trt_engine

    device = torch.device(args.device)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    trt_device = (
        torch_tensorrt.Device(f"cuda:{device_index}")
        if device.type == "cuda"
        else torch_tensorrt.Device(str(device))
    )

    load_plugin(args.plugin_path)
    engine_bytes = convert_exported_program_to_serialized_trt_engine(
        exported_program,
        arg_inputs=compile_inputs,
        device=trt_device,
        min_block_size=args.min_block_size,
        workspace_size=args.workspace_size,
        optimization_level=args.optimization_level,
        require_full_compilation=args.require_full_compilation,
        disable_tf32=not args.allow_tf32,
        use_fp32_acc=not args.no_fp32_acc,
    )
    engine_path.write_bytes(engine_bytes)

    engine_info: Dict[str, Any] = {}
    if not args.skip_engine_inspection:
        engine_info = inspect_serialized_engine(engine_bytes)

    manifest.setdefault("artifacts", {})
    manifest["artifacts"].setdefault("torch_export", relative_or_absolute(export_path, output_dir))
    manifest["source_manifest"] = relative_or_absolute(manifest_path, output_manifest_path.parent)
    add_direct_engine_metadata(
        manifest,
        engine_path,
        output_dir,
        engine_bytes,
        args,
        engine_info,
    )
    output_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Saved direct TensorRT engine to {engine_path}")
    print(f"Saved direct engine manifest to {output_manifest_path}")
    print(json.dumps(manifest["direct_tensorrt_engine"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a VLM vision tower from Hugging Face directly to a TensorRT engine.

This script is the "from model" companion to compile_vlm_export_to_engine.py:

    HF/PyTorch vision tower -> torch.export -> serialized TensorRT .engine

It reuses export_vlm_to_tensorrt.py for Hugging Face loading, processor input
preparation, vision-contract handling, and ViT attention replacement. The final
compile step uses Torch-TensorRT's direct serialized-engine API instead of
building a Torch-TensorRT runtime module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from compile_vlm_export_to_engine import inspect_serialized_engine, relative_or_absolute
from export_vlm_to_tensorrt import (
    DEFAULT_OUTPUT_DIR,
    VIT_INPUT_CONTRACT_STATIC_GRID_THW,
    ViTPluginWrapper,
    call_vision_reference,
    compile_inputs_from_tensors,
    count_vit_plugin_attention_modules,
    dtype_from_name,
    export_vision,
    get_vision_module,
    load_model,
    load_plugin,
    load_processor_for_export,
    normalize_conv2d_valid_padding,
    prepare_processor_inputs,
    prepare_synthetic_pixel_inputs,
    prepare_vision_contract,
    replace_vit_attention_with_plugin,
    safe_model_tag,
    save_executorch_program,
    save_sample_tensors,
    set_vit_plugin_config_from_visual,
    tensor_specs,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a VLM vision tower and compile it directly to a serialized "
            "TensorRT engine."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local model path to export.",
    )
    parser.add_argument(
        "--model_class",
        default=None,
        help="Optional import path for a custom model class with from_pretrained.",
    )
    parser.add_argument(
        "--vision_module",
        default=None,
        help="Optional dotted module path inside the loaded model to export.",
    )
    parser.add_argument(
        "--no_processor",
        action="store_true",
        help="Skip AutoProcessor and build a synthetic pixel_values tensor directly.",
    )
    parser.add_argument(
        "--dataclass_kw_only_imports",
        action="store_true",
        help="Temporarily make dataclass-decorated classes keyword-only during custom import.",
    )
    parser.add_argument(
        "--instantiate_from_config",
        action="store_true",
        help="For --model_class, construct from config instead of from_pretrained().",
    )
    parser.add_argument(
        "--add_common_vlm_aliases",
        action="store_true",
        help="Install process-local compatibility aliases on loaded VLM modules.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--processor_model",
        default=None,
        help="Optional HF model id/path to load the processor from.",
    )
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Synthetic square RGB image size used for the export input contract.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Export and compile device.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Optional Hugging Face attention implementation override.",
    )
    parser.add_argument(
        "--plugin_path",
        default=None,
        help="Optional path to libNvInfer_edgellm_plugin.so.",
    )
    parser.add_argument(
        "--no_vit_attention_plugin",
        action="store_true",
        help="Use the Python fallback wrapper instead of the ViTAttentionPlugin custom op.",
    )
    parser.add_argument(
        "--save_executorch",
        action="store_true",
        help="Also lower the exported program to ExecuTorch and save a .pte file.",
    )
    parser.add_argument(
        "--save_sample_tensors",
        action="store_true",
        help="Save export sample inputs and PyTorch reference output for replay tests.",
    )
    parser.add_argument(
        "--output_engine",
        default=None,
        help="Optional explicit output .engine path.",
    )
    parser.add_argument(
        "--skip_engine_inspection",
        action="store_true",
        help="Skip deserializing the produced engine to record I/O tensor names.",
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
        help="Allow TF32. By default this script disables TF32.",
    )
    parser.add_argument(
        "--no_fp32_acc",
        action="store_true",
        help="Disable FP32 accumulation for matmul layers.",
    )
    return parser.parse_args()


def resolve_trt_device(device: torch.device) -> Any:
    import torch_tensorrt

    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return torch_tensorrt.Device(f"cuda:{device_index}")
    return torch_tensorrt.Device(str(device))


def compile_direct_engine(
    exported_program: torch.export.ExportedProgram,
    compile_args: Tuple[torch.Tensor, ...],
    compile_kwargs: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> bytes:
    from torch_tensorrt.dynamo import convert_exported_program_to_serialized_trt_engine

    compile_inputs = compile_inputs_from_tensors(compile_args, compile_kwargs)
    if not args.no_vit_attention_plugin or args.plugin_path is not None:
        load_plugin(args.plugin_path)
    return convert_exported_program_to_serialized_trt_engine(
        exported_program,
        arg_inputs=compile_inputs,
        device=resolve_trt_device(device),
        min_block_size=args.min_block_size,
        workspace_size=args.workspace_size,
        optimization_level=args.optimization_level,
        require_full_compilation=args.require_full_compilation,
        disable_tf32=not args.allow_tf32,
        use_fp32_acc=not args.no_fp32_acc,
    )


def add_direct_engine_manifest_fields(
    manifest: Dict[str, Any],
    engine_path: Path,
    output_dir: Path,
    engine_bytes: bytes,
    args: argparse.Namespace,
    engine_info: Dict[str, Any],
) -> None:
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
    manifest["runtime_requirements"] = {
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    model = load_model(
        args.model,
        dtype,
        device,
        prefer_generation_model=False,
        model_class=args.model_class,
        dataclass_kw_only_imports=args.dataclass_kw_only_imports,
        instantiate_from_config=args.instantiate_from_config,
        use_common_vlm_aliases=args.add_common_vlm_aliases,
        move_to_device=False,
        attn_implementation=args.attn_implementation,
    )

    visual = get_vision_module(model, args.vision_module).to(device=device, dtype=dtype).eval()
    normalize_conv2d_valid_padding(visual)

    if args.no_processor:
        processor_inputs, input_metadata = prepare_synthetic_pixel_inputs(
            args.image_size,
            device,
            dtype,
        )
    else:
        processor = load_processor_for_export(
            model,
            args.model,
            args.processor_model,
        )
        processor_inputs, input_metadata = prepare_processor_inputs(
            processor,
            args.prompt,
            args.image_size,
            device,
            dtype,
        )

    pixel_values = processor_inputs["pixel_values"]
    input_contract, core_inputs, max_window_seq_len = prepare_vision_contract(
        visual,
        processor_inputs,
        pixel_values,
        device,
        dtype,
    )

    with torch.no_grad():
        reference = call_vision_reference(
            visual,
            pixel_values,
            processor_inputs,
            input_contract,
            core_inputs,
        )

    use_vit_attention_plugin = not args.no_vit_attention_plugin
    set_vit_plugin_config_from_visual(visual, pixel_values)
    vision_config = getattr(
        visual,
        "config",
        getattr(getattr(model, "config", None), "vision_config", model.config),
    )
    replace_vit_attention_with_plugin(
        visual,
        vision_config,
        use_plugin_op=use_vit_attention_plugin,
    )
    attention_modules = count_vit_plugin_attention_modules(visual)
    static_grid_thw = (
        processor_inputs["image_grid_thw"]
        if input_contract == VIT_INPUT_CONTRACT_STATIC_GRID_THW
        else None
    )
    wrapper = ViTPluginWrapper(
        visual,
        input_contract=input_contract,
        max_window_seq_len=max_window_seq_len,
        static_grid_thw=static_grid_thw,
    ).eval().to(device)

    exported_program = export_vision(wrapper, pixel_values, core_inputs)
    artifact_prefix = f"{safe_model_tag(args.model)}_vision"
    export_path = output_dir / f"{artifact_prefix}_exported_program.pt2"
    torch.export.save(exported_program, export_path)

    sample_inputs = {"pixel_values": pixel_values, **core_inputs}
    compile_args = (pixel_values,)
    compile_kwargs = core_inputs
    engine_path = (
        Path(args.output_engine)
        if args.output_engine is not None
        else output_dir / f"{artifact_prefix}_direct.engine"
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    engine_bytes = compile_direct_engine(
        exported_program,
        compile_args,
        compile_kwargs,
        device,
        args,
    )
    engine_path.write_bytes(engine_bytes)

    engine_info: Dict[str, Any] = {}
    if not args.skip_engine_inspection:
        engine_info = inspect_serialized_engine(engine_bytes)

    output_metadata = {
        "vision_embeddings_shape": list(reference.shape),
        "vision_embeddings_dtype": str(reference.dtype),
    }
    manifest: Dict[str, Any] = {
        "model": args.model,
        "component": "vision",
        "format_version": 1,
        "input_contract": input_contract,
        "model_class": args.model_class,
        "processor_model": args.processor_model,
        "vision_module": args.vision_module,
        "uses_vit_attention_plugin": use_vit_attention_plugin,
        "vit_attention_modules": attention_modules,
        "max_window_seq_len": max_window_seq_len,
        "artifacts": {
            "torch_export": export_path.name,
        },
        "inputs": input_metadata,
        "tensor_inputs": tensor_specs(sample_inputs),
        "core_inputs": tensor_specs(core_inputs),
        "outputs": output_metadata,
    }

    if args.save_executorch:
        pte_path = output_dir / f"{artifact_prefix}.pte"
        save_executorch_program(exported_program, pte_path)
        manifest["artifacts"]["executorch"] = pte_path.name

    if args.save_sample_tensors:
        sample_path = output_dir / f"{artifact_prefix}_test_tensors.pt"
        save_sample_tensors(sample_path, sample_inputs, reference)
        manifest["artifacts"]["sample_tensors"] = sample_path.name

    add_direct_engine_manifest_fields(
        manifest,
        engine_path,
        output_dir,
        engine_bytes,
        args,
        engine_info,
    )
    write_manifest(output_dir, manifest)

    print(f"Saved artifacts to {output_dir}")
    print(f"Saved direct TensorRT engine to {engine_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

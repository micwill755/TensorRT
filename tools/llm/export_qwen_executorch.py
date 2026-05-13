#!/usr/bin/env python3
"""Standalone Qwen VLM export scaffold for ExecuTorch/Torch-TensorRT experiments.

This script intentionally does not import run_vlm.py. It is meant as a small
x86-first artifact path for the next project direction:

    Hugging Face Qwen2.5-VL -> torch.export -> optional ExecuTorch .pte
    -> optional Torch-TensorRT saved artifact

The first target is the vision tower because it has a compact tensor contract:
pixel_values + image_grid_thw -> image embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

from plugin_utils_vit import (
    VIT_INPUT_CONTRACT_WINDOWED_ROPE,
    ViTPluginWrapper,
    count_vit_plugin_attention_modules,
    load_plugin,
    replace_vit_attention_with_plugin,
    set_vit_plugin_config,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qwen2.5-VL vision tower through torch.export / ExecuTorch."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default="/tmp/qwen_executorch_artifacts")
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
        help="Export device. Use cuda for Torch-TensorRT compile.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--save_executorch",
        action="store_true",
        help="Lower the exported program to ExecuTorch and save a .pte file.",
    )
    parser.add_argument(
        "--compile_torchtrt",
        action="store_true",
        help="Compile the exported vision tower with Torch-TensorRT and save it.",
    )
    parser.add_argument(
        "--plugin_path",
        default=None,
        help="Optional path to libNvInfer_edgellm_plugin.so.",
    )
    parser.add_argument(
        "--no_vit_attention_plugin",
        action="store_true",
        help=(
            "Replace Qwen attention with the Python fallback wrapper instead of "
            "the TensorRT ViT attention plugin custom op."
        ),
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def get_qwen_visual(model: nn.Module) -> nn.Module:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        visual = getattr(owner, "visual", None)
        if isinstance(visual, nn.Module):
            return visual
    raise ValueError("Could not find Qwen visual tower at model.visual or model.model.visual")


def _extract_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor-like model output, got {type(output)!r}")
    return output


def _get_config_attr(config: Any, names: Tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _infer_patch_count(vision_config: Any, pixel_values: torch.Tensor) -> int:
    if pixel_values.dim() in (2, 3):
        return int(pixel_values.shape[-2])

    image_size = _get_config_attr(vision_config, ("image_size",))
    patch_size = _get_config_attr(vision_config, ("patch_size",))
    if image_size is None or patch_size is None:
        return 0

    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size[:2]
    else:
        image_h = image_w = image_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size[:2]
    else:
        patch_h = patch_w = patch_size
    return int((image_h // patch_h) * (image_w // patch_w) + 1)


def set_vit_plugin_config_from_qwen_visual(
    visual: nn.Module, pixel_values: torch.Tensor
) -> None:
    vision_config = getattr(visual, "config", None)
    if vision_config is None:
        raise ValueError("Cannot infer ViT plugin config: visual.config is missing.")

    num_heads = _get_config_attr(
        vision_config, ("num_heads", "num_attention_heads", "attention_heads")
    )
    if num_heads is None:
        raise ValueError("Cannot infer ViT plugin num_attention_heads from config.")

    head_dim = _get_config_attr(vision_config, ("head_dim",))
    if head_dim is None:
        hidden_size = _get_config_attr(vision_config, ("hidden_size", "embed_dim", "dim"))
        if hidden_size is None:
            raise ValueError("Cannot infer ViT plugin hidden_size from config.")
        head_dim = int(hidden_size) // int(num_heads)

    set_vit_plugin_config(
        num_attention_heads=int(num_heads),
        head_dim=int(head_dim),
        num_patches=_infer_patch_count(vision_config, pixel_values),
    )


def make_windowed_rope_core_inputs(
    visual: nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Precompute Qwen window/RoPE tensors outside torch.export tracing."""

    with torch.no_grad():
        rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)

    window_index = window_index.to(device=device, dtype=torch.long)
    reverse_window_index = torch.argsort(window_index)

    seq_len = pixel_values.shape[0]
    attention_mask = torch.zeros(1, seq_len, seq_len, dtype=dtype, device=device)
    window_attention_mask = torch.full(
        (1, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    cu_window_seqlens_cpu = torch.as_tensor(
        cu_window_seqlens, device="cpu", dtype=torch.long
    )
    cu_window_seqlens_list = torch.unique_consecutive(cu_window_seqlens_cpu).tolist()
    max_window_seq_len = max(
        end - start
        for start, end in zip(cu_window_seqlens_list[:-1], cu_window_seqlens_list[1:])
    )

    for start, end in zip(cu_window_seqlens_list[:-1], cu_window_seqlens_list[1:]):
        window_attention_mask[:, start:end, start:end] = 0

    return (
        {
            "rotary_pos_emb": rotary_pos_emb.to(device=device),
            "attention_mask": attention_mask,
            "window_attention_mask": window_attention_mask,
            "cu_window_seqlens": torch.tensor(
                cu_window_seqlens_list, dtype=torch.int32, device=device
            ),
            "window_index": window_index,
            "reverse_window_index": reverse_window_index,
        },
        max_window_seq_len,
    )


def make_synthetic_image(size: int) -> Image.Image:
    # A deterministic, non-uniform image keeps preprocessing realistic enough
    # while avoiding network or dataset dependencies.
    y = torch.linspace(0, 255, size, dtype=torch.uint8).view(size, 1)
    x = torch.linspace(0, 255, size, dtype=torch.uint8).view(1, size)
    red = x.expand(size, size)
    green = y.expand(size, size)
    blue = ((red.to(torch.int16) + green.to(torch.int16)) // 2).to(torch.uint8)
    image = torch.stack((red, green, blue), dim=-1).cpu().numpy()
    return Image.fromarray(image, mode="RGB")


def prepare_inputs(
    processor: Any,
    prompt: str,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    image = make_synthetic_image(image_size)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    metadata = {
        "input_ids_shape": list(inputs["input_ids"].shape),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
        "image_grid_thw": image_grid_thw.detach().cpu().tolist(),
    }
    return pixel_values, image_grid_thw, metadata


def export_vision(
    wrapper: nn.Module,
    pixel_values: torch.Tensor,
    core_inputs: Dict[str, torch.Tensor],
) -> torch.export.ExportedProgram:
    with torch.no_grad():
        return torch.export.export(
            wrapper,
            args=(pixel_values,),
            kwargs=core_inputs,
            strict=False,
        )


def save_executorch_program(
    exported_program: torch.export.ExportedProgram, output_path: Path
) -> None:
    try:
        from executorch.exir import to_edge
    except ImportError as exc:
        raise RuntimeError(
            "ExecuTorch is not installed in this environment. Install it or rerun "
            "without --save_executorch."
        ) from exc

    edge_program = to_edge(exported_program)
    executorch_program = edge_program.to_executorch()
    output_path.write_bytes(executorch_program.buffer)


def compile_and_save_torchtrt(
    exported_program: torch.export.ExportedProgram,
    example_inputs: Tuple[torch.Tensor, ...],
    example_kwargs: Dict[str, torch.Tensor],
    output_path: Path,
    device: torch.device,
    plugin_path: str | None,
) -> None:
    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError(
            "Torch-TensorRT is not installed in this environment. Install it or rerun "
            "without --compile_torchtrt."
        ) from exc

    try:
        import torch_tensorrt._C  # noqa: F401
    except ImportError:
        # Some local source builds register the runtime ops through the packaged
        # runtime libraries without exposing a torch_tensorrt._C Python module.
        pass

    if not hasattr(torch.ops.tensorrt, "execute_engine"):
        raise RuntimeError(
            "torch.ops.tensorrt.execute_engine is not registered. The serialized "
            ".pt2 execution artifact requires the Torch-TensorRT C++ runtime op."
        )

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    trt_device = torch_tensorrt.Device(f"cuda:{device_index}")

    load_plugin(plugin_path)
    compile_inputs = [
        torch_tensorrt.Input(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        for tensor in list(example_inputs) + list(example_kwargs.values())
    ]

    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=compile_inputs,
        use_explicit_typing=True,
        use_fp32_acc=True,
        disable_tf32=True,
        device=trt_device,
        min_block_size=1,
    )
    torch_tensorrt.save(trt_model, str(output_path), retrace=False)


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    visual = get_qwen_visual(model)
    pixel_values, image_grid_thw, input_metadata = prepare_inputs(
        processor, args.prompt, args.image_size, device, dtype
    )

    with torch.no_grad():
        reference = _extract_tensor(visual(pixel_values, grid_thw=image_grid_thw))

    use_vit_attention_plugin = not args.no_vit_attention_plugin
    set_vit_plugin_config_from_qwen_visual(visual, pixel_values)
    replace_vit_attention_with_plugin(
        visual,
        getattr(visual, "config", model.config),
        use_plugin_op=use_vit_attention_plugin,
    )
    attention_modules = count_vit_plugin_attention_modules(visual)
    core_inputs, max_window_seq_len = make_windowed_rope_core_inputs(
        visual, pixel_values, image_grid_thw, device, dtype
    )
    wrapper = ViTPluginWrapper(
        visual,
        input_contract=VIT_INPUT_CONTRACT_WINDOWED_ROPE,
        max_window_seq_len=max_window_seq_len,
    ).eval().to(device)

    exported_program = export_vision(wrapper, pixel_values, core_inputs)
    export_path = output_dir / "qwen_vision_exported_program.pt2"
    torch.export.save(exported_program, export_path)

    manifest: Dict[str, Any] = {
        "model": args.model,
        "component": "vision",
        "format_version": 1,
        "input_contract": VIT_INPUT_CONTRACT_WINDOWED_ROPE,
        "uses_vit_attention_plugin": use_vit_attention_plugin,
        "vit_attention_modules": attention_modules,
        "max_window_seq_len": max_window_seq_len,
        "artifacts": {
            "torch_export": export_path.name,
        },
        "inputs": input_metadata,
        "core_inputs": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in core_inputs.items()
        },
        "outputs": {
            "vision_embeddings_shape": list(reference.shape),
            "vision_embeddings_dtype": str(reference.dtype),
        },
    }

    if args.save_executorch:
        pte_path = output_dir / "qwen_vision.pte"
        save_executorch_program(exported_program, pte_path)
        manifest["artifacts"]["executorch"] = pte_path.name

    if args.compile_torchtrt:
        trt_path = output_dir / "qwen_vision_torchtrt.pt2"
        compile_and_save_torchtrt(
            exported_program,
            (pixel_values,),
            core_inputs,
            trt_path,
            device,
            args.plugin_path,
        )
        manifest["artifacts"]["torch_tensorrt"] = trt_path.name

    write_manifest(output_dir, manifest)
    print(f"Saved artifacts to {output_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Standalone VLM export scaffold for Torch-TensorRT experiments.

This script intentionally does not import run_vlm.py. It is meant as a small
x86-first artifact path for the next project direction:

    Hugging Face VLM -> torch.export -> optional ExecuTorch .pte
    -> optional Torch-TensorRT saved artifact -> raw TensorRT engine

The first target is the vision tower because it has a compact tensor contract.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

from plugin_utils_vit import (
    VIT_INPUT_CONTRACT_NATIVE,
    VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO,
    VIT_INPUT_CONTRACT_WINDOWED_ROPE,
    ViTPluginWrapper,
    count_vit_plugin_attention_modules,
    load_plugin,
    replace_vit_attention_with_plugin,
    set_vit_plugin_config,
)
from utils import (
    extract_vision_tensor,
    get_qwen_position_ids,
    get_vision_model as get_generic_vision_model,
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
        "--component",
        default="vision",
        choices=("vision", "vlm_prefill"),
        help=(
            "Artifact to export. 'vision' exports the visual tower only; "
            "'vlm_prefill' exports one full VLM prefill forward ending at "
            "last-token logits."
        ),
    )
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
        "--input_export",
        default=None,
        help=(
            "Load an existing torch.export .pt2 and compile it with Torch-TensorRT. "
            "When set, model loading/export is skipped."
        ),
    )
    parser.add_argument(
        "--input_manifest",
        default=None,
        help=(
            "Manifest from a previous export run. Defaults to manifest.json next to "
            "--input_export when compiling an existing export."
        ),
    )
    parser.add_argument(
        "--save_sample_tensors",
        action="store_true",
        help="Save the export sample inputs and PyTorch reference output for engine tests.",
    )
    parser.add_argument(
        "--sample_tensors",
        default=None,
        help="Path to sample tensors saved by --save_sample_tensors.",
    )
    parser.add_argument(
        "--run_engine",
        default=None,
        help="Run a raw TensorRT engine with saved sample tensors and compare output.",
    )
    parser.add_argument(
        "--run_bundle",
        action="store_true",
        help=(
            "Run the raw TensorRT engine and sample tensors listed in "
            "--output_dir/manifest.json."
        ),
    )
    parser.add_argument("--verify_atol", type=float, default=2e-1)
    parser.add_argument("--verify_rtol", type=float, default=2e-1)
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


def get_qwen_visual(model: nn.Module) -> nn.Module:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        visual = getattr(owner, "visual", None)
        if isinstance(visual, nn.Module):
            return visual
    raise ValueError("Could not find Qwen visual tower at model.visual or model.model.visual")


def model_loader_candidates(prefer_generation_model: bool) -> List[Any]:
    generation_candidates: List[Any] = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            generation_candidates.append(getattr(module, class_name))
        except (ImportError, AttributeError):
            pass
    candidates: List[Any] = (
        generation_candidates + [AutoModel]
        if prefer_generation_model
        else [AutoModel] + generation_candidates
    )
    return candidates


def load_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    prefer_generation_model: bool,
    move_to_device: bool = True,
) -> nn.Module:
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    last_error: Exception | None = None
    with torch.no_grad():
        for loader in model_loader_candidates(prefer_generation_model):
            try:
                model = loader.from_pretrained(
                    model_name,
                    **model_kwargs,
                ).eval()
                return model.to(device) if move_to_device else model
            except (KeyError, ValueError, AttributeError) as exc:
                last_error = exc
    raise ValueError(f"Could not load {model_name!r} with available AutoModel classes.") from last_error


def get_qwen_language_model(model: nn.Module) -> nn.Module:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        language_model = getattr(owner, "language_model", None)
        if isinstance(language_model, nn.Module):
            return language_model
    for attr_name in ("language_model", "model"):
        language_model = getattr(model, attr_name, None)
        if isinstance(language_model, nn.Module) and language_model is not model:
            return language_model
    raise ValueError("Could not find Qwen language model.")


def get_input_embedding_layer(model: nn.Module, language_model: nn.Module) -> nn.Module:
    for owner in (language_model, model):
        if hasattr(owner, "get_input_embeddings"):
            emb = owner.get_input_embeddings()
            if isinstance(emb, nn.Module):
                return emb
    raise ValueError("Could not find Qwen input embedding layer.")


def get_lm_head(model: nn.Module, language_model: nn.Module) -> nn.Module:
    for owner in (model, language_model):
        lm_head = getattr(owner, "lm_head", None)
        if isinstance(lm_head, nn.Module):
            return lm_head
    raise ValueError("Could not find Qwen lm_head.")


def get_image_token_id(model: nn.Module) -> int:
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Could not find model.config.image_token_id.")
    return int(image_token_id)


def _extract_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor-like model output, got {type(output)!r}")
    return output


def compare_tensors(
    label: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    atol: float,
    rtol: float,
) -> bool:
    reference = reference.detach().float().cpu()
    actual = actual.detach().float().cpu()
    diff = (reference - actual).abs()
    allclose = torch.allclose(reference, actual, atol=atol, rtol=rtol)
    print(
        f"{label}: allclose={allclose}, "
        f"max_abs={diff.max().item():.6f}, "
        f"mean_abs={diff.mean().item():.6f}, "
        f"ref_norm={reference.norm().item():.6f}, "
        f"trt_norm={actual.norm().item():.6f}"
    )
    return allclose


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


def has_windowed_rope_contract(
    visual: nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> bool:
    return (
        isinstance(inputs.get("image_grid_thw"), torch.Tensor)
        and hasattr(visual, "get_window_index")
        and hasattr(visual, "rot_pos_emb")
        and hasattr(visual, "patch_embed")
        and hasattr(visual, "blocks")
    )


def has_tiled_aspect_ratio_contract(inputs: Dict[str, torch.Tensor]) -> bool:
    return isinstance(inputs.get("aspect_ratio_ids"), torch.Tensor) and (
        isinstance(inputs.get("aspect_ratio_mask"), torch.Tensor)
        or isinstance(inputs.get("attention_mask"), torch.Tensor)
    )


def prepare_vision_contract(
    visual: nn.Module,
    inputs: Dict[str, torch.Tensor],
    pixel_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[str, Dict[str, torch.Tensor], int]:
    if has_windowed_rope_contract(visual, inputs):
        return (
            VIT_INPUT_CONTRACT_WINDOWED_ROPE,
            *make_windowed_rope_core_inputs(
                visual,
                pixel_values,
                inputs["image_grid_thw"],
                device,
                dtype,
            ),
        )

    if has_tiled_aspect_ratio_contract(inputs):
        attention_mask = inputs.get("aspect_ratio_mask")
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = inputs["attention_mask"]
        return (
            VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO,
            {
                "aspect_ratio_ids": inputs["aspect_ratio_ids"],
                "attention_mask": attention_mask,
            },
            0,
        )

    return VIT_INPUT_CONTRACT_NATIVE, {}, 0


def call_vision_reference(
    visual: nn.Module,
    pixel_values: torch.Tensor,
    processor_inputs: Dict[str, torch.Tensor],
    input_contract: str,
    core_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        if input_contract == VIT_INPUT_CONTRACT_WINDOWED_ROPE:
            return _extract_tensor(
                visual(pixel_values, grid_thw=processor_inputs["image_grid_thw"])
            )

        if input_contract == VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO:
            kwargs = {
                "aspect_ratio_ids": core_inputs["aspect_ratio_ids"],
                "aspect_ratio_mask": core_inputs["attention_mask"],
                "output_hidden_states": True,
            }
            try:
                return extract_vision_tensor(visual(pixel_values, **kwargs))
            except TypeError:
                kwargs.pop("output_hidden_states", None)
                try:
                    return extract_vision_tensor(visual(pixel_values, **kwargs))
                except TypeError:
                    return extract_vision_tensor(
                        visual(
                            pixel_values,
                            aspect_ratio_ids=core_inputs["aspect_ratio_ids"],
                            attention_mask=core_inputs["attention_mask"],
                        )
                    )

        return extract_vision_tensor(visual(pixel_values))


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


def prepare_processor_inputs(
    processor: Any,
    prompt: str,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
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
    tensor_inputs: Dict[str, torch.Tensor] = {
        name: value.to(device=device)
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    tensor_inputs["pixel_values"] = tensor_inputs["pixel_values"].to(dtype=dtype)
    metadata = {
        "input_ids_shape": list(inputs["input_ids"].shape)
        if "input_ids" in inputs
        else None,
        "pixel_values_shape": list(tensor_inputs["pixel_values"].shape),
        "pixel_values_dtype": str(tensor_inputs["pixel_values"].dtype),
    }
    for optional_name in (
        "image_grid_thw",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
        "attention_mask",
    ):
        optional_tensor = tensor_inputs.get(optional_name)
        if isinstance(optional_tensor, torch.Tensor):
            metadata[optional_name] = optional_tensor.detach().cpu().tolist()
    return tensor_inputs, metadata


def prepare_inputs(
    processor: Any,
    prompt: str,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    inputs, metadata = prepare_processor_inputs(
        processor, prompt, image_size, device, dtype
    )
    return inputs["pixel_values"], inputs["image_grid_thw"], metadata


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


class QwenVlmPrefillWrapper(nn.Module):
    """Full Qwen VLM prefill wrapper ending at last-token logits."""

    def __init__(
        self,
        vision: nn.Module,
        embeddings: nn.Module,
        language_model: nn.Module,
        lm_head: nn.Module,
        image_token_id: int,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.embeddings = embeddings
        self.language_model = language_model
        self.lm_head = lm_head
        self.image_token_id = image_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        window_attention_mask: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        window_index: torch.Tensor,
        reverse_window_index: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.vision(
            pixel_values,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            window_attention_mask=window_attention_mask,
            cu_window_seqlens=cu_window_seqlens,
            window_index=window_index,
            reverse_window_index=reverse_window_index,
        )
        inputs_embeds = self.embeddings(input_ids)
        image_mask = input_ids == self.image_token_id
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        hidden_states = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0] if isinstance(output, (tuple, list)) else output
        )
        return self.lm_head(hidden_states[:, -1, :])


def export_vlm_prefill(
    wrapper: nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    position_ids: torch.Tensor,
    core_inputs: Dict[str, torch.Tensor],
) -> torch.export.ExportedProgram:
    with torch.no_grad():
        return torch.export.export(
            wrapper,
            args=(input_ids, pixel_values, position_ids),
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


def save_sample_tensors(
    output_path: Path,
    sample_inputs: Dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> None:
    tensors = {
        "reference": reference.detach().cpu(),
    }
    tensors.update({name: value.detach().cpu() for name, value in sample_inputs.items()})
    torch.save(tensors, output_path)


def _torch_dtype_from_trt(trt_dtype: Any) -> torch.dtype:
    import tensorrt as trt

    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "BF16"):
        dtype_map[trt.DataType.BF16] = torch.bfloat16
    if trt_dtype not in dtype_map:
        raise TypeError(f"Unsupported TensorRT dtype for torch allocation: {trt_dtype}")
    return dtype_map[trt_dtype]


def load_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Could not find manifest at {manifest_path}.")
    return json.loads(manifest_path.read_text())


def resolve_run_artifacts(
    args: argparse.Namespace,
) -> Tuple[Path, Path, str | None]:
    manifest: Dict[str, Any] | None = None
    output_dir = Path(args.output_dir)

    if args.run_bundle:
        manifest = load_manifest(output_dir)
        engines = manifest.get("artifacts", {}).get("tensorrt_engines", [])
        if len(engines) != 1:
            raise RuntimeError(
                f"--run_bundle expected exactly one TensorRT engine, found {len(engines)}."
            )
        engine_path = output_dir / engines[0]["path"]
    elif args.run_engine is not None:
        engine_path = Path(args.run_engine)
    else:
        raise RuntimeError("--run_engine or --run_bundle is required.")

    if args.sample_tensors is not None:
        sample_path = Path(args.sample_tensors)
    else:
        if manifest is None and (output_dir / "manifest.json").exists():
            manifest = load_manifest(output_dir)
        sample_name = (
            manifest.get("artifacts", {}).get("sample_tensors")
            if manifest is not None
            else None
        )
        if sample_name is None:
            sample_path = output_dir / "qwen_vision_test_tensors.pt"
        else:
            sample_path = output_dir / sample_name

    plugin_path = args.plugin_path
    if plugin_path is None and manifest is not None:
        plugin_path = (
            manifest.get("runtime_requirements", {}) or {}
        ).get("tensorrt_plugin_path")

    return engine_path, sample_path, plugin_path


def run_raw_tensorrt_engine(args: argparse.Namespace) -> None:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are required for --run_engine.") from exc

    engine_path, sample_path, plugin_path = resolve_run_artifacts(args)

    if plugin_path is None:
        raise RuntimeError("--plugin_path is required to run an engine with ViTAttentionPlugin.")

    if not sample_path.exists():
        raise RuntimeError(
            f"Could not find sample tensors at {sample_path}. Rerun export with "
            "--save_sample_tensors first."
        )

    ctypes.CDLL(plugin_path)
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize {engine_path}.")

    samples = torch.load(sample_path, map_location="cpu")
    device = torch.device(args.device)
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream(device=device)
    bound_tensors: Dict[str, torch.Tensor] = {}

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        dtype = _torch_dtype_from_trt(engine.get_tensor_dtype(name))

        if mode == trt.TensorIOMode.INPUT:
            if name not in samples:
                raise RuntimeError(
                    f"Engine input {name!r} is missing from {sample_path}."
                )
            tensor = samples[name].to(device=device, dtype=dtype).contiguous()
            if hasattr(context, "set_input_shape"):
                context.set_input_shape(name, tuple(tensor.shape))
        else:
            shape = tuple(context.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=dtype, device=device)

        bound_tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))

    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 returned false.")
    torch.cuda.synchronize(device)

    output_names = [
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
    ]
    if len(output_names) != 1:
        raise RuntimeError(f"Expected one engine output, found {output_names}.")
    if "reference" not in samples:
        raise RuntimeError(f"Sample tensor file {sample_path} does not contain reference.")

    compare_tensors(
        "raw TensorRT engine output vs PyTorch reference",
        samples["reference"],
        bound_tensors[output_names[0]],
        args.verify_atol,
        args.verify_rtol,
    )


def compile_and_save_torchtrt(
    exported_program: torch.export.ExportedProgram,
    compile_inputs: List[Any],
    output_path: Path,
    device: torch.device,
    plugin_path: str | None,
) -> List[Dict[str, Any]]:
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

    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=compile_inputs,
        use_explicit_typing=True,
        use_fp32_acc=True,
        disable_tf32=True,
        device=trt_device,
        min_block_size=1,
    )
    engine_entries = save_raw_tensorrt_engines(trt_model, output_path.parent)
    torch_tensorrt.save(trt_model, str(output_path), retrace=False)
    return engine_entries


def compile_inputs_from_tensors(
    example_inputs: Tuple[torch.Tensor, ...],
    example_kwargs: Dict[str, torch.Tensor],
) -> List[Any]:
    import torch_tensorrt

    return [
        torch_tensorrt.Input(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        for tensor in list(example_inputs) + list(example_kwargs.values())
    ]


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


def _safe_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("_"))
    return sanitized or "tensorrt_engine"


def safe_model_tag(model_name: str) -> str:
    return _safe_artifact_name(model_name.rsplit("/", 1)[-1].lower())


def save_raw_tensorrt_engines(
    trt_model: nn.Module,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    engine_dir = output_dir / "engines"
    engine_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for module_name, module in trt_model.named_modules():
        serialized_engine = getattr(module, "serialized_engine", None)
        if not serialized_engine:
            continue

        engine_path = engine_dir / f"{_safe_artifact_name(module_name)}.engine"
        engine_bytes = bytes(serialized_engine)
        engine_path.write_bytes(engine_bytes)
        entries.append(
            {
                "name": module_name,
                "path": str(engine_path.relative_to(output_dir)),
                "bytes": len(engine_bytes),
                "input_binding_names": list(
                    getattr(module, "input_binding_names", [])
                ),
                "output_binding_names": list(
                    getattr(module, "output_binding_names", [])
                ),
            }
        )

    if not entries:
        raise RuntimeError("Torch-TensorRT compile did not produce any serialized engines.")
    return entries


def load_manifest_for_export(args: argparse.Namespace, export_path: Path) -> Dict[str, Any]:
    manifest_path = (
        Path(args.input_manifest)
        if args.input_manifest is not None
        else export_path.parent / "manifest.json"
    )
    if not manifest_path.exists():
        raise RuntimeError(
            f"Could not find manifest for {export_path}. Pass --input_manifest explicitly."
        )
    return json.loads(manifest_path.read_text())


def add_runtime_requirements(
    manifest: Dict[str, Any],
    plugin_path: str | None,
) -> None:
    manifest["runtime_requirements"] = {
        "custom_op_module": "plugin_utils_vit",
        "tensorrt_plugin_path": plugin_path,
        "load_order": [
            "import torch_tensorrt",
            "import plugin_utils_vit",
            "ctypes.CDLL(tensorrt_plugin_path)",
            "torch_tensorrt.load(torch_tensorrt_artifact)",
        ],
    }


def maybe_register_sdpa_converter(model_name: str, model_config: Any | None) -> None:
    try:
        from torchtrt_ext import register_sdpa
    except ImportError:
        return
    register_sdpa.enable_sdpa_converter(model_name, model_config)


def compile_existing_export(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_path = Path(args.input_export)
    manifest = load_manifest_for_export(args, export_path)
    if manifest.get("component") == "vlm_prefill":
        maybe_register_sdpa_converter(manifest.get("model", args.model), None)
    exported_program = torch.export.load(export_path)
    compile_inputs = compile_inputs_from_manifest(exported_program, manifest)

    export_stem = export_path.stem
    artifact_prefix = (
        export_stem[: -len("_exported_program")]
        if export_stem.endswith("_exported_program")
        else export_stem
    )
    trt_path = output_dir / f"{artifact_prefix}_torchtrt.pt2"
    engine_entries = compile_and_save_torchtrt(
        exported_program,
        compile_inputs,
        trt_path,
        torch.device(args.device),
        args.plugin_path,
    )

    manifest.setdefault("artifacts", {})
    try:
        manifest["artifacts"]["torch_export"] = str(export_path.relative_to(output_dir))
    except ValueError:
        manifest["artifacts"]["torch_export"] = str(export_path)
    manifest["artifacts"]["torch_tensorrt"] = trt_path.name
    manifest["artifacts"]["tensorrt_engines"] = engine_entries
    add_runtime_requirements(manifest, args.plugin_path)
    write_manifest(output_dir, manifest)

    print(f"Compiled {export_path} to {trt_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def tensor_specs(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tensors.items()
    }


def main() -> None:
    args = parse_args()
    if args.run_engine is not None or args.run_bundle:
        run_raw_tensorrt_engine(args)
        return

    if args.input_export is not None:
        compile_existing_export(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    model = load_model(
        args.model,
        dtype,
        device,
        prefer_generation_model=args.component == "vlm_prefill",
        move_to_device=args.component != "vision",
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    visual = get_generic_vision_model(model).to(device=device, dtype=dtype).eval()
    processor_inputs, input_metadata = prepare_processor_inputs(
        processor, args.prompt, args.image_size, device, dtype
    )
    pixel_values = processor_inputs["pixel_values"]
    image_grid_thw = processor_inputs.get("image_grid_thw")
    input_ids = processor_inputs.get("input_ids")
    attention_mask = processor_inputs.get("attention_mask")
    input_contract, core_inputs, max_window_seq_len = prepare_vision_contract(
        visual,
        processor_inputs,
        pixel_values,
        device,
        dtype,
    )

    with torch.no_grad():
        if args.component == "vision":
            reference = call_vision_reference(
                visual,
                pixel_values,
                processor_inputs,
                input_contract,
                core_inputs,
            )
        else:
            if input_contract != VIT_INPUT_CONTRACT_WINDOWED_ROPE:
                raise ValueError(
                    "--component vlm_prefill currently supports Qwen-style "
                    "windowed-RoPE VLMs only. Use --component vision for Llama vision."
                )
            if input_ids is None or image_grid_thw is None:
                raise ValueError("Qwen VLM prefill requires input_ids and image_grid_thw.")
            reference_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            if hasattr(reference_output, "logits"):
                reference = reference_output.logits[:, -1, :]
            else:
                language_model = get_qwen_language_model(model)
                lm_head = get_lm_head(model, language_model)
                hidden_states = (
                    reference_output.last_hidden_state
                    if hasattr(reference_output, "last_hidden_state")
                    else reference_output[0]
                )
                reference = lm_head(hidden_states[:, -1, :])

    use_vit_attention_plugin = not args.no_vit_attention_plugin
    set_vit_plugin_config_from_qwen_visual(visual, pixel_values)
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
    wrapper = ViTPluginWrapper(
        visual,
        input_contract=input_contract,
        max_window_seq_len=max_window_seq_len,
    ).eval().to(device)

    if args.component == "vision":
        exported_program = export_vision(wrapper, pixel_values, core_inputs)
        artifact_prefix = (
            "qwen_vision"
            if args.model == DEFAULT_MODEL
            else f"{safe_model_tag(args.model)}_vision"
        )
        sample_inputs = {"pixel_values": pixel_values, **core_inputs}
        compile_args = (pixel_values,)
        compile_kwargs = core_inputs
        tensor_input_specs = tensor_specs(sample_inputs)
        output_metadata = {
            "vision_embeddings_shape": list(reference.shape),
            "vision_embeddings_dtype": str(reference.dtype),
        }
    else:
        language_model = get_qwen_language_model(model)
        embeddings = get_input_embedding_layer(model, language_model)
        lm_head = get_lm_head(model, language_model)
        position_ids = get_qwen_position_ids(
            model,
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        full_wrapper = QwenVlmPrefillWrapper(
            wrapper,
            embeddings,
            language_model,
            lm_head,
            get_image_token_id(model),
        ).eval().to(device)
        exported_program = export_vlm_prefill(
            full_wrapper,
            input_ids,
            pixel_values,
            position_ids,
            core_inputs,
        )
        artifact_prefix = "qwen_vlm_prefill"
        sample_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "position_ids": position_ids,
            **core_inputs,
        }
        compile_args = (input_ids, pixel_values, position_ids)
        compile_kwargs = core_inputs
        tensor_input_specs = tensor_specs(sample_inputs)
        output_metadata = {
            "last_token_logits_shape": list(reference.shape),
            "last_token_logits_dtype": str(reference.dtype),
        }

    export_path = output_dir / f"{artifact_prefix}_exported_program.pt2"
    torch.export.save(exported_program, export_path)

    manifest: Dict[str, Any] = {
        "model": args.model,
        "component": args.component,
        "format_version": 1,
        "input_contract": input_contract,
        "uses_vit_attention_plugin": use_vit_attention_plugin,
        "vit_attention_modules": attention_modules,
        "max_window_seq_len": max_window_seq_len,
        "artifacts": {
            "torch_export": export_path.name,
        },
        "inputs": input_metadata,
        "tensor_inputs": tensor_input_specs,
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

    if args.compile_torchtrt:
        if args.component == "vlm_prefill":
            maybe_register_sdpa_converter(args.model, model.config)
        trt_path = output_dir / f"{artifact_prefix}_torchtrt.pt2"
        compile_inputs = compile_inputs_from_tensors(
            compile_args,
            compile_kwargs,
        )
        engine_entries = compile_and_save_torchtrt(
            exported_program,
            compile_inputs,
            trt_path,
            device,
            args.plugin_path,
        )
        manifest["artifacts"]["torch_tensorrt"] = trt_path.name
        manifest["artifacts"]["tensorrt_engines"] = engine_entries
        add_runtime_requirements(manifest, args.plugin_path)

    write_manifest(output_dir, manifest)
    print(f"Saved artifacts to {output_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

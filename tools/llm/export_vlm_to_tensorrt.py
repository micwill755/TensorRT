#!/usr/bin/env python3
"""Compile path for VLM vision towers.

This script takes a Hugging Face/PyTorch model vision tower through:

    HF/PyTorch model -> torch.export -> Torch-TensorRT compile
    -> serialized TensorRT engine

The exported artifact is focused on the vision tower because it has a compact
tensor contract that can be validated independently from text generation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from plugin_utils_vit import (
    VIT_INPUT_CONTRACT_GRID_THW,
    VIT_INPUT_CONTRACT_NATIVE,
    VIT_INPUT_CONTRACT_STATIC_GRID_THW,
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
    get_vision_model as get_generic_vision_model,
)


DEFAULT_OUTPUT_DIR = "/tmp/vlm_vision_tensorrt_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a VLM vision tower through torch.export and Torch-TensorRT."
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Hugging Face model id or local model path to export.",
    )
    parser.add_argument(
        "--model_class",
        default=None,
        help=(
            "Optional import path for a custom model class with from_pretrained, "
            "for models not registered with Transformers AutoModel."
        ),
    )
    parser.add_argument(
        "--vision_module",
        default=None,
        help=(
            "Optional dotted module path inside the loaded model to export, "
            "for example 'backbone.vision_model'."
        ),
    )
    parser.add_argument(
        "--no_processor",
        action="store_true",
        help=(
            "Skip AutoProcessor and build a synthetic pixel_values tensor directly. "
            "Use this for standalone vision modules or policy checkpoints."
        ),
    )
    parser.add_argument(
        "--dataclass_kw_only_imports",
        action="store_true",
        help=(
            "Temporarily make dataclass-decorated classes keyword-only while "
            "importing/loading a custom model class. This can help older packages "
            "import on Python 3.12 without editing their source."
        ),
    )
    parser.add_argument(
        "--instantiate_from_config",
        action="store_true",
        help=(
            "For --model_class, load the HF config and call the class constructor "
            "instead of class.from_pretrained(). Useful when the outer model uses "
            "nested from_pretrained calls that conflict with HF meta initialization."
        ),
    )
    parser.add_argument(
        "--add_common_vlm_aliases",
        action="store_true",
        help=(
            "Install process-local compatibility aliases on loaded VLM modules "
            "for packages that expect common .language_model/.visual attributes."
        ),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--processor_model",
        default=None,
        help=(
            "Optional HF model id/path to load the processor from. Defaults to "
            "--model. Useful when exporting a nested backbone from a larger policy."
        ),
    )
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument(
        "--component",
        default="vision",
        choices=("vision",),
        help="Artifact to export. Currently exports the model's vision tower.",
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
        "--attn_implementation",
        default=None,
        choices=("eager", "sdpa", "flash_attention_2"),
        help=(
            "Optional Hugging Face attention implementation to pass to "
            "from_pretrained."
        ),
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
            "Replace supported vision attention with the Python fallback wrapper "
            "instead of the TensorRT ViT attention plugin custom op."
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


def import_object(import_path: str) -> Any:
    module_name, _, object_name = import_path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(
            f"Expected a dotted import path like package.module.Class, got {import_path!r}."
        )
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def set_attn_implementation_on_config(
    config: Any,
    attn_implementation: str,
) -> None:
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or id(value) in visited:
            return
        visited.add(id(value))

        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        elif hasattr(value, "__dict__"):
            for attr_name in (
                "attn_implementation",
                "_attn_implementation",
                "_attn_implementation_internal",
            ):
                setattr(value, attr_name, attn_implementation)
            children = vars(value).values()
        else:
            return

        for child in children:
            visit(child)

    visit(config)


def load_config_with_attn_implementation(
    model_name: str,
    attn_implementation: str,
) -> Any:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    set_attn_implementation_on_config(config, attn_implementation)
    return config


def ensure_tied_weight_keys_compat(model: nn.Module) -> None:
    if hasattr(model, "all_tied_weights_keys"):
        return

    tied_keys = getattr(model, "_tied_weights_keys", None)
    if tied_keys is None:
        model.all_tied_weights_keys = {}
    elif isinstance(tied_keys, dict):
        model.all_tied_weights_keys = tied_keys
    else:
        model.all_tied_weights_keys = {key: None for key in tied_keys}


@contextmanager
def force_attn_implementation_during_init(attn_implementation: str | None):
    if attn_implementation is None:
        yield
        return

    from transformers import PreTrainedModel

    original_init = PreTrainedModel.__init__

    def init_with_forced_attn(self, config, *args, **kwargs):
        set_attn_implementation_on_config(config, attn_implementation)
        result = original_init(self, config, *args, **kwargs)
        ensure_tied_weight_keys_compat(self)
        return result

    PreTrainedModel.__init__ = init_with_forced_attn
    try:
        yield
    finally:
        PreTrainedModel.__init__ = original_init


@contextmanager
def default_device_for_loading(device: str):
    get_default_device = getattr(torch, "get_default_device", None)
    set_default_device = getattr(torch, "set_default_device", None)
    if get_default_device is None or set_default_device is None:
        yield
        return

    previous_device = get_default_device()
    set_default_device(device)
    try:
        yield
    finally:
        set_default_device(previous_device)


def enable_dataclass_kw_only_import_compat(
    module_prefixes: Tuple[str, ...],
) -> None:
    import dataclasses

    if getattr(dataclasses.dataclass, "_vlm_export_kw_only_compat", False):
        return

    original_dataclass = dataclasses.dataclass

    def dataclass_compat(cls=None, /, **kwargs):
        def should_force_kw_only(inner_cls: Any) -> bool:
            module_name = getattr(inner_cls, "__module__", "")
            return any(
                module_name == prefix or module_name.startswith(f"{prefix}.")
                for prefix in module_prefixes
            )

        if cls is None:
            return lambda inner_cls: original_dataclass(
                inner_cls,
                **(
                    {**kwargs, "kw_only": True}
                    if should_force_kw_only(inner_cls)
                    else kwargs
                ),
            )
        if should_force_kw_only(cls):
            kwargs.setdefault("kw_only", True)
        return original_dataclass(cls, **kwargs)

    dataclass_compat._vlm_export_kw_only_compat = True  # type: ignore[attr-defined]
    dataclasses.dataclass = dataclass_compat


def add_common_vlm_aliases(model: nn.Module) -> nn.Module:
    if not hasattr(model, "language_model"):
        language_model = None
        nested_model = getattr(model, "model", None)
        if isinstance(nested_model, nn.Module):
            if hasattr(nested_model, "language_model"):
                language_model = nested_model.language_model
            elif hasattr(nested_model, "layers"):
                language_model = nested_model
        if isinstance(language_model, nn.Module):
            model.language_model = language_model

    if not hasattr(model, "visual"):
        for attr_name in ("vision_model", "visual", "vision_tower"):
            candidate = getattr(model, attr_name, None)
            if isinstance(candidate, nn.Module):
                model.visual = candidate
                break
        else:
            nested_model = getattr(model, "model", None)
            if isinstance(nested_model, nn.Module):
                for attr_name in ("visual", "vision_model", "vision_tower"):
                    candidate = getattr(nested_model, attr_name, None)
                    if isinstance(candidate, nn.Module):
                        model.visual = candidate
                        break
    return model


def enable_common_vlm_alias_hook() -> None:
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    original_from_pretrained = PreTrainedModel.from_pretrained
    original_from_pretrained_fn = getattr(
        original_from_pretrained,
        "__func__",
        original_from_pretrained,
    )
    if getattr(original_from_pretrained_fn, "_vlm_export_common_aliases", False):
        return

    def from_pretrained_with_aliases(cls, *args, **kwargs):
        model = original_from_pretrained_fn(cls, *args, **kwargs)
        return add_common_vlm_aliases(model)

    from_pretrained_with_aliases._vlm_export_common_aliases = True  # type: ignore[attr-defined]
    PreTrainedModel.from_pretrained = classmethod(from_pretrained_with_aliases)


def load_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    prefer_generation_model: bool,
    model_class: str | None = None,
    dataclass_kw_only_imports: bool = False,
    instantiate_from_config: bool = False,
    use_common_vlm_aliases: bool = False,
    move_to_device: bool = True,
    attn_implementation: str | None = None,
) -> nn.Module:
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if attn_implementation is not None:
        model_kwargs["config"] = load_config_with_attn_implementation(
            model_name,
            attn_implementation,
        )
    if model_class is not None:
        if dataclass_kw_only_imports:
            enable_dataclass_kw_only_import_compat(
                (model_class.split(".", maxsplit=1)[0],)
            )
        if use_common_vlm_aliases:
            enable_common_vlm_alias_hook()
        loader = import_object(model_class)
        with torch.no_grad():
            with default_device_for_loading("cpu"):
                with force_attn_implementation_during_init(attn_implementation):
                    if instantiate_from_config:
                        config = loader.config_class.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                        )
                        if attn_implementation is not None:
                            set_attn_implementation_on_config(
                                config,
                                attn_implementation,
                            )
                        model = loader(config).eval()
                    else:
                        model = loader.from_pretrained(model_name, **model_kwargs).eval()
            if use_common_vlm_aliases:
                model = add_common_vlm_aliases(model)
            return model.to(device) if move_to_device else model

    last_error: Exception | None = None
    with torch.no_grad():
        if use_common_vlm_aliases:
            enable_common_vlm_alias_hook()
        for loader in model_loader_candidates(prefer_generation_model):
            try:
                with default_device_for_loading("cpu"):
                    with force_attn_implementation_during_init(attn_implementation):
                        model = loader.from_pretrained(
                            model_name,
                            **model_kwargs,
                        ).eval()
                if use_common_vlm_aliases:
                    model = add_common_vlm_aliases(model)
                return model.to(device) if move_to_device else model
            except (KeyError, ValueError, AttributeError) as exc:
                last_error = exc
    raise ValueError(f"Could not load {model_name!r} with available AutoModel classes.") from last_error


def get_nested_module(model: nn.Module, module_path: str) -> nn.Module:
    current: Any = model
    for attr_name in module_path.split("."):
        if not attr_name:
            raise ValueError(f"Invalid empty path component in {module_path!r}.")
        if attr_name.isdigit() and isinstance(current, (nn.ModuleList, list, tuple)):
            current = current[int(attr_name)]
        else:
            current = getattr(current, attr_name)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{module_path!r} did not resolve to a torch.nn.Module.")
    return current


def get_vision_module(model: nn.Module, module_path: str | None) -> nn.Module:
    if module_path is not None:
        return get_nested_module(model, module_path)
    return get_generic_vision_model(model)


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


def set_vit_plugin_config_from_visual(
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
    """Precompute window/RoPE tensors outside torch.export tracing."""

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


def has_grid_thw_contract(visual: nn.Module, inputs: Dict[str, torch.Tensor]) -> bool:
    if not isinstance(inputs.get("image_grid_thw"), torch.Tensor):
        return False
    try:
        signature = inspect.signature(visual.forward)
    except (TypeError, ValueError):
        return False
    return "grid_thw" in signature.parameters


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

    if has_grid_thw_contract(visual, inputs):
        # Some HF vision towers branch on grid_thw values internally while
        # constructing positional embeddings. Keep the sampled grid static for
        # this fixed-shape export so torch.export does not need to specialize a
        # data-dependent tensor input.
        return (
            VIT_INPUT_CONTRACT_STATIC_GRID_THW,
            {},
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

        if input_contract in (
            VIT_INPUT_CONTRACT_GRID_THW,
            VIT_INPUT_CONTRACT_STATIC_GRID_THW,
        ):
            grid_thw = core_inputs.get("grid_thw", processor_inputs["image_grid_thw"])
            return extract_vision_tensor(
                visual(pixel_values, grid_thw=grid_thw)
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


def prepare_synthetic_pixel_inputs(
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    y = torch.linspace(0, 1, image_size, dtype=torch.float32).view(1, image_size, 1)
    x = torch.linspace(0, 1, image_size, dtype=torch.float32).view(1, 1, image_size)
    red = x.expand(1, image_size, image_size)
    green = y.expand(1, image_size, image_size)
    blue = (red + green) * 0.5
    pixel_values = torch.stack((red, green, blue), dim=1)
    pixel_values = (pixel_values - 0.5) / 0.5
    pixel_values = pixel_values.to(device=device, dtype=dtype).contiguous()
    metadata = {
        "input_source": "synthetic_pixel_values",
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
    }
    return {"pixel_values": pixel_values}, metadata


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
            sample_path = output_dir / "vision_test_tensors.pt"
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


def compile_existing_export(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_path = Path(args.input_export)
    manifest = load_manifest_for_export(args, export_path)
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

    if args.model is None:
        raise RuntimeError("--model is required when exporting a new artifact.")

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
    if args.no_processor:
        processor_inputs, input_metadata = prepare_synthetic_pixel_inputs(
            args.image_size, device, dtype
        )
    else:
        processor = AutoProcessor.from_pretrained(
            args.processor_model or args.model,
            trust_remote_code=True,
            use_fast=True,
        )
        processor_inputs, input_metadata = prepare_processor_inputs(
            processor, args.prompt, args.image_size, device, dtype
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
    sample_inputs = {"pixel_values": pixel_values, **core_inputs}
    compile_args = (pixel_values,)
    compile_kwargs = core_inputs
    tensor_input_specs = tensor_specs(sample_inputs)
    output_metadata = {
        "vision_embeddings_shape": list(reference.shape),
        "vision_embeddings_dtype": str(reference.dtype),
    }

    export_path = output_dir / f"{artifact_prefix}_exported_program.pt2"
    torch.export.save(exported_program, export_path)

    manifest: Dict[str, Any] = {
        "model": args.model,
        "component": args.component,
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

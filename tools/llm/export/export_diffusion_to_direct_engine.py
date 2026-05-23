#!/usr/bin/env python3
"""Export a VLA diffusion/action expert directly to a TensorRT engine.

This is the action-side companion to export_vlm_to_direct_engine.py and
export_lm_to_direct_engine.py. It loads a root model, resolves the action
components directly from the model structure or explicit module paths, and
serializes one fused denoising step:

    x, t, prefix_k, prefix_v, position_ids, attention_mask -> action_update

The runtime owns the diffusion sampling loop. This engine owns only the hot
denoising function: action_in_proj + expert + action_out_proj.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from utils.direct_engine_utils import (
    add_direct_engine_manifest_fields as add_shared_direct_engine_manifest_fields,
    patch_torchtrt_output_names,
    relative_or_absolute,
)
from utils.engine_io import save_trt_engine


DEFAULT_OUTPUT_DIR = "/tmp/diffusion_direct_tensorrt_artifacts"

logger = logging.getLogger(__name__)


class CacheLayerView:
    __slots__ = ("keys", "values")

    def __init__(self, keys: torch.Tensor | None = None, values: torch.Tensor | None = None):
        self.keys = keys
        self.values = values


class PrefixKVCache:
    """Minimal stacked-prefix KV cache accepted by HF decoder blocks."""

    __prefix_kv_cache__ = True

    def __init__(self, prefix_k: torch.Tensor, prefix_v: torch.Tensor):
        self._k = prefix_k
        self._v = prefix_v
        self.layers = [CacheLayerView() for _ in range(prefix_k.shape[0])]
        self._sync_layer_views()

    @property
    def key_cache(self) -> torch.Tensor:
        return self._k

    @property
    def value_cache(self) -> torch.Tensor:
        return self._v

    def _sync_layer_views(self) -> None:
        for index, layer in enumerate(self.layers):
            layer.keys = self._k[index]
            layer.values = self._v[index]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        return (
            torch.cat([self._k[layer_idx], key_states], dim=-2),
            torch.cat([self._v[layer_idx], value_states], dim=-2),
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self._k.shape[3]

    def get_max_cache_shape(self) -> None:
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        del layer_idx
        return self._k.shape[3] + cache_position.shape[0], 0

    def __len__(self) -> int:
        return int(self._k.shape[0])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].keys, self.layers[layer_idx].values


class StaticKVDiffusionStepModule(nn.Module):
    """Fused action projection, expert, and action output projection."""

    def __init__(
        self,
        action_in_proj: nn.Module,
        expert: nn.Module,
        action_out_proj: nn.Module,
        n_diffusion_tokens: int,
        action_space_dims: tuple[int, ...],
    ):
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        self.n_diffusion_tokens = int(n_diffusion_tokens)
        self.action_space_dims = tuple(int(dim) for dim in action_space_dims)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        future_token_embeds = self.action_in_proj(x, t)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, self.n_diffusion_tokens, -1)

        expert_out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=PrefixKVCache(prefix_k, prefix_v),
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_hidden = expert_out.last_hidden_state[:, -self.n_diffusion_tokens :]
        return self.action_out_proj(last_hidden).view(-1, *self.action_space_dims)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def import_object(import_path: str) -> Any:
    module_name, object_name = import_path.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module_name), object_name)


def safe_model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.strip("/")) or "model"


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_nested_attr(root: Any, attr_path: str) -> Any:
    current = root
    for part in attr_path.split("."):
        if not hasattr(current, part):
            raise ValueError(
                f"Cannot resolve path {attr_path!r}: {type(current).__name__} has no {part!r}."
            )
        current = getattr(current, part)
    return current


def get_nested_module(root: nn.Module, module_path: str) -> nn.Module:
    module = get_nested_attr(root, module_path)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{module_path!r} resolved to {type(module).__name__}, not nn.Module.")
    return module


def module_at_path_or_none(root: nn.Module, module_path: str) -> Optional[nn.Module]:
    try:
        return get_nested_module(root, module_path)
    except (AttributeError, TypeError, ValueError):
        return None


def parse_int_tuple(value: Optional[str]) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--action_space_dims must contain at least one integer.")
    return tuple(int(part) for part in parts)


def resolve_module(root: nn.Module, explicit_path: Optional[str], candidate_paths: tuple[str, ...], name: str) -> nn.Module:
    if explicit_path:
        return get_nested_module(root, explicit_path)
    for candidate_path in candidate_paths:
        module = module_at_path_or_none(root, candidate_path)
        if isinstance(module, nn.Module):
            return module
    raise ValueError(f"Could not resolve {name}. Pass --{name} explicitly.")


def resolve_action_space_dims(root: nn.Module, explicit_dims: Optional[tuple[int, ...]]) -> tuple[int, ...]:
    if explicit_dims is not None:
        return explicit_dims
    for candidate_path in ("action_space", "model.action_space"):
        try:
            action_space = get_nested_attr(root, candidate_path)
        except ValueError:
            continue
        get_dims = getattr(action_space, "get_action_space_dims", None)
        if callable(get_dims):
            return tuple(int(dim) for dim in get_dims())
    raise ValueError("Could not infer action space dims. Pass --action_space_dims, e.g. 8,2.")


def resolve_expert_config(expert: nn.Module) -> Any:
    config = getattr(expert, "config", None)
    if config is None:
        raise ValueError("Selected expert module does not expose .config.")
    if not hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = config.num_attention_heads
    if not hasattr(config, "head_dim") or config.head_dim is None:
        config.head_dim = config.hidden_size // config.num_attention_heads
    return config


def inspect_serialized_engine(engine_bytes: bytes) -> Dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the direct diffusion engine output.")
    inputs = []
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return {
        "input_names": inputs,
        "output_names": outputs,
        "num_io_tensors": int(engine.num_io_tensors),
        "num_optimization_profiles": int(engine.num_optimization_profiles),
    }


def make_sample_inputs(
    cfg: Dict[str, Any],
    *,
    min_prefix_len: int,
    max_prefix_len: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...], Any]:
    n_diffusion_tokens = int(cfg["n_diffusion_tokens"])
    action_space_dims = tuple(int(dim) for dim in cfg["action_space_dims"])
    num_layers = int(cfg["num_layers"])
    num_kv_heads = int(cfg["num_kv_heads"])
    head_dim = int(cfg["head_dim"])
    opt_prefix_len = (min_prefix_len + max_prefix_len) // 2

    def make_inputs(prefix_len: int) -> list[torch.Tensor]:
        return [
            torch.randn(batch_size, *action_space_dims, dtype=dtype, device=device),
            torch.zeros(batch_size, 1, 1, dtype=dtype, device=device),
            torch.zeros(num_layers, batch_size, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.zeros(num_layers, batch_size, num_kv_heads, prefix_len, head_dim, dtype=dtype, device=device),
            torch.arange(n_diffusion_tokens, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(3, batch_size, -1)
            .clone(),
            torch.zeros(
                batch_size,
                1,
                n_diffusion_tokens,
                prefix_len + n_diffusion_tokens,
                dtype=torch.float32,
                device=device,
            ),
        ]

    sample_inputs = tuple(make_inputs(opt_prefix_len))
    prefix_dim = torch.export.Dim("prefix_len", min=min_prefix_len, max=max_prefix_len)
    mask_dim = prefix_dim + n_diffusion_tokens
    dynamic_shapes = (
        None,
        None,
        {3: prefix_dim},
        {3: prefix_dim},
        None,
        {3: mask_dim},
    )
    return sample_inputs, dynamic_shapes, make_inputs


def export_diffusion_module(
    module: nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: tuple[Any, ...],
) -> "torch.export.ExportedProgram":
    with torch.no_grad():
        try:
            return torch.export.export(
                module,
                sample_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
        except Exception as exc:
            logger.warning(
                "torch.export.export failed (%s), retrying with deferred runtime asserts",
                exc,
            )
            return torch.export._trace._export(
                module,
                sample_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )


def build_trt_input_specs(
    make_inputs: Any,
    min_prefix_len: int,
    opt_prefix_len: int,
    max_prefix_len: int,
) -> list[Any]:
    import torch_tensorrt

    min_inputs = make_inputs(min_prefix_len)
    opt_inputs = make_inputs(opt_prefix_len)
    max_inputs = make_inputs(max_prefix_len)
    return [
        torch_tensorrt.Input(
            min_shape=tuple(t_min.shape),
            opt_shape=tuple(t_opt.shape),
            max_shape=tuple(t_max.shape),
            dtype=t_min.dtype,
        )
        for t_min, t_opt, t_max in zip(min_inputs, opt_inputs, max_inputs)
    ]


def tensor_specs(tensors: tuple[torch.Tensor, ...]) -> Dict[str, Dict[str, Any]]:
    names = ("x", "t", "prefix_k", "prefix_v", "position_ids", "attention_mask")
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in zip(names, tensors)
    }


def direct_compile_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "truncate_double": True,
        "min_block_size": args.min_block_size,
        "workspace_size": args.workspace_size,
        "optimization_level": args.optimization_level,
        "require_full_compilation": bool(args.require_full_compilation),
        "disable_tf32": not args.allow_tf32,
        "use_fp32_acc": not args.no_fp32_acc,
        "use_explicit_typing": True,
        "immutable_weights": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a VLA diffusion/action expert step to a serialized TensorRT engine."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--model_class", default=None, help="Optional custom class import path.")
    parser.add_argument(
        "--instantiate_from_config",
        action="store_true",
        help="For --model_class, construct from config instead of from_pretrained().",
    )
    parser.add_argument("--action_in_proj", default=None, help="Dotted path to action input projection.")
    parser.add_argument("--expert_module", default=None, help="Dotted path to action expert/decoder module.")
    parser.add_argument("--action_out_proj", default=None, help="Dotted path to action output projection.")
    parser.add_argument(
        "--action_space_dims",
        default=None,
        help="Comma-separated action shape, e.g. 8,2. Defaults to model.action_space.get_action_space_dims().",
    )
    parser.add_argument(
        "--n_diffusion_tokens",
        type=int,
        default=None,
        help="Number of future/action tokens. Defaults to action_space_dims[0].",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_engine", default=None, help="Optional explicit output .engine/.trt path.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--min_prefix_len", type=int, default=1)
    parser.add_argument("--max_prefix_len", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--attn_implementation", default="sdpa", help="Expert attention implementation override.")
    parser.add_argument("--min_block_size", type=int, default=1)
    parser.add_argument("--workspace_size", type=int, default=0)
    parser.add_argument("--optimization_level", type=int, default=None)
    parser.add_argument("--require_full_compilation", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--no_fp32_acc", action="store_true")
    parser.add_argument("--save_sample_tensors", action="store_true")
    parser.add_argument("--skip_engine_inspection", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.min_prefix_len < 1:
        raise ValueError("--min_prefix_len must be >= 1.")
    if args.max_prefix_len < args.min_prefix_len:
        raise ValueError("--max_prefix_len must be >= --min_prefix_len.")
    if args.dtype != "float16":
        raise ValueError("The direct diffusion export path currently expects --dtype float16.")


def load_root_model(args: argparse.Namespace, dtype: torch.dtype) -> nn.Module:
    if args.model_class is not None:
        loader = import_object(args.model_class)
        if args.instantiate_from_config:
            config = loader.config_class.from_pretrained(args.model, trust_remote_code=True)
            return loader(config).eval()
        return loader.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).eval()

    from transformers import AutoModel

    return AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
    ).eval()


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    root_model = load_root_model(args, dtype).to(device=device, dtype=dtype).eval()
    action_in_proj = resolve_module(
        root_model,
        args.action_in_proj,
        ("action_in_proj", "model.action_in_proj"),
        "action_in_proj",
    )
    expert = resolve_module(
        root_model,
        args.expert_module,
        ("expert", "model.expert", "action_expert"),
        "expert_module",
    )
    action_out_proj = resolve_module(
        root_model,
        args.action_out_proj,
        ("action_out_proj", "model.action_out_proj"),
        "action_out_proj",
    )
    if args.attn_implementation and hasattr(expert, "config"):
        expert.config._attn_implementation = args.attn_implementation

    action_space_dims = resolve_action_space_dims(root_model, parse_int_tuple(args.action_space_dims))
    n_diffusion_tokens = int(args.n_diffusion_tokens or action_space_dims[0])
    expert_config = resolve_expert_config(expert)

    module = (
        StaticKVDiffusionStepModule(
            action_in_proj=action_in_proj,
            expert=expert,
            action_out_proj=action_out_proj,
            n_diffusion_tokens=n_diffusion_tokens,
            action_space_dims=action_space_dims,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    cfg = {
        "n_diffusion_tokens": n_diffusion_tokens,
        "action_space_dims": action_space_dims,
        "num_layers": int(expert_config.num_hidden_layers),
        "num_kv_heads": int(expert_config.num_key_value_heads),
        "head_dim": int(expert_config.head_dim),
    }

    sample_inputs, dynamic_shapes, make_inputs = make_sample_inputs(
        cfg,
        min_prefix_len=args.min_prefix_len,
        max_prefix_len=args.max_prefix_len,
        dtype=dtype,
        device=device,
        batch_size=args.batch_size,
    )
    exported = export_diffusion_module(module, sample_inputs, dynamic_shapes)
    opt_prefix_len = (args.min_prefix_len + args.max_prefix_len) // 2
    trt_input_specs = build_trt_input_specs(
        make_inputs,
        args.min_prefix_len,
        opt_prefix_len,
        args.max_prefix_len,
    )

    artifact_prefix = f"{safe_model_tag(args.model)}_diffusion"
    engine_path = (
        Path(args.output_engine)
        if args.output_engine is not None
        else output_dir / f"{artifact_prefix}_direct.engine"
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    import torch_tensorrt

    trt_settings: Dict[str, Any] = {
        "truncate_double": True,
        "min_block_size": args.min_block_size,
        "workspace_size": args.workspace_size,
        "optimization_level": args.optimization_level,
        "require_full_compilation": bool(args.require_full_compilation),
        "disable_tf32": not args.allow_tf32,
        "use_fp32_acc": not args.no_fp32_acc,
        "use_explicit_typing": True,
        "immutable_weights": True,
    }
    with torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext():
        with patch_torchtrt_output_names(["action_update"]):
            engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exported,
                inputs=trt_input_specs,
                **trt_settings,
            )

    metadata = {
        "component": "action",
        "save_format": "raw_trt_engine",
        "runtime_contract": "vla_diffusion_step",
        "precision": "FP16",
        "input_names": ["x", "t", "prefix_k", "prefix_v", "position_ids", "attention_mask"],
        "output_names": ["action_update"],
        "batch_size": int(args.batch_size),
        "min_prefix_len": int(args.min_prefix_len),
        "max_prefix_len": int(args.max_prefix_len),
        "n_diffusion_tokens": n_diffusion_tokens,
        "action_space_dims": list(action_space_dims),
        "num_layers": cfg["num_layers"],
        "num_kv_heads": cfg["num_kv_heads"],
        "head_dim": cfg["head_dim"],
    }
    save_trt_engine(engine_bytes, engine_path, metadata)

    engine_info: Dict[str, Any] = {}
    if not args.skip_engine_inspection:
        engine_info = inspect_serialized_engine(engine_bytes)

    sidecar_path = Path(str(engine_path) + ".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    manifest: Dict[str, Any] = {
        "model": args.model,
        "component": "action",
        "format_version": 1,
        "runtime_contract": "vla_diffusion_step",
        "model_class": args.model_class,
        "action_in_proj": args.action_in_proj,
        "expert_module": args.expert_module,
        "action_out_proj": args.action_out_proj,
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "min_prefix_len": int(args.min_prefix_len),
        "max_prefix_len": int(args.max_prefix_len),
        "n_diffusion_tokens": n_diffusion_tokens,
        "action_space_dims": list(action_space_dims),
        "tensor_inputs": tensor_specs(sample_inputs),
        "artifacts": {
            "engine_sidecar": relative_or_absolute(sidecar_path, output_dir),
        },
        "engine_sidecar": sidecar,
    }
    if args.save_sample_tensors:
        sample_path = output_dir / f"{artifact_prefix}_test_tensors.pt"
        torch.save(
            {name: tensor.detach().cpu() for name, tensor in zip(metadata["input_names"], sample_inputs)},
            sample_path,
        )
        manifest["artifacts"]["sample_tensors"] = sample_path.name

    add_shared_direct_engine_manifest_fields(
        manifest,
        engine_path=engine_path,
        output_dir=output_dir,
        engine_bytes=engine_bytes,
        compile_options=direct_compile_options(args),
        engine_info=engine_info,
        custom_op_module=None,
        plugin_path=None,
        runtime_requirements_extra={"action_engine_contract": "vla_diffusion_step"},
    )
    write_manifest(output_dir, manifest)

    del module, exported, sample_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Saved artifacts to {output_dir}")
    print(f"Saved direct TensorRT action/diffusion engine to {engine_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export a Hugging Face language module directly to a plugin TensorRT engine.

This mirrors export_vlm_to_direct_engine.py's structure: load the model from
Hugging Face or a custom class, resolve the component directly from the model
structure (optionally with explicit module paths), and serialize the selected LM
component with the Edge-LLM AttentionPlugin.

The emitted engine contract is the VLA/plugin-LM contract:

    inputs_embeds, kv_caches, ctx_len, ds_stack -> logits, present_key_values.*

For text-only models, ds_stack is a zero tensor with one physical layer and
num_ds_layers=0 in the manifest. VLM/VLA runtimes can pass real deepstack
features by exporting with --num_ds_layers > 0.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from utils.direct_engine_utils import (
    add_direct_engine_manifest_fields as add_shared_direct_engine_manifest_fields,
    relative_or_absolute,
)


DEFAULT_OUTPUT_DIR = "/tmp/lm_direct_tensorrt_artifacts"

logger = logging.getLogger(__name__)
FP16 = torch.float16


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


def tensor_specs(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in tensors.items()
    }


def get_nested_module(model: nn.Module, module_path: str) -> nn.Module:
    current: Any = model
    for part in module_path.split("."):
        if not hasattr(current, part):
            raise ValueError(
                f"Cannot resolve module path {module_path!r}: "
                f"{type(current).__name__} has no attribute {part!r}."
            )
        current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{module_path!r} resolved to {type(current).__name__}, not nn.Module.")
    return current


def inspect_serialized_engine(engine_bytes: bytes) -> Dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the direct engine output.")
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



def create_kv_caches_from_config(
    config: Any,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    return [
        torch.zeros(
            batch_size,
            2,
            int(config.num_key_value_heads),
            max_seq_len,
            int(config.head_dim),
            dtype=dtype,
            device=device,
        )
        for _ in range(int(config.num_hidden_layers))
    ]




class PluginWrapperDSInput(nn.Module):
    """LM forward that uses plugin self-attention and optional deepstack features."""

    def __init__(self, lm: nn.Module, lm_head: nn.Module, num_ds: int):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.num_ds = int(num_ds)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden = inputs_embeds
        seq_len = inputs_embeds.shape[1]
        new_kvs: list[torch.Tensor] = []
        for index, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                past_key_value=kv_caches[index],
                ctx_len=ctx_len,
            )
            hidden = residual + hidden
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            new_kvs.append(kv)
            if index < self.num_ds:
                hidden = hidden + ds_stack[index, :, :seq_len, :]
        hidden = self.lm.norm(hidden)
        return self.lm_head(hidden), new_kvs


class FlatPluginWrapperDSInput(PluginWrapperDSInput):
    """Plugin LM wrapper variant with flat TensorRT outputs."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
        ds_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        logits, new_kvs = super().forward(inputs_embeds, kv_caches, ctx_len, ds_stack)
        return (logits, *tuple(new_kvs))


def plugin_lm_output_names(num_layers: int) -> List[str]:
    return ["logits"] + [f"present_key_values.{index}" for index in range(num_layers)]


def find_rotary_embedding(lm: nn.Module) -> nn.Module:
    if hasattr(lm, "rotary_emb"):
        return lm.rotary_emb
    layers = getattr(lm, "layers", None)
    if layers is not None and len(layers) > 0:
        self_attn = getattr(layers[0], "self_attn", None)
        rotary_emb = getattr(self_attn, "rotary_emb", None)
        if rotary_emb is not None:
            return rotary_emb
    raise ValueError("Cannot find rotary embedding on the selected language module.")


def build_rope_cache(
    lm: nn.Module,
    prefill_seq_len: int,
    position_ids: Optional[torch.Tensor],
    rope_deltas: Optional[torch.Tensor],
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is not None and rope_deltas is not None and hasattr(lm, "rotary_emb"):
        try:
            with torch.no_grad():
                decode_positions = torch.arange(prefill_seq_len, max_seq_len, device=device).float()
                decode_positions = decode_positions + rope_deltas.to(device).float().squeeze()
                decode_position_ids = decode_positions.view(1, 1, -1).expand(3, 1, -1).long()
                full_position_ids = torch.cat([position_ids.to(device), decode_position_ids], dim=2)
                cos, sin = lm.rotary_emb(torch.ones(1, device=device, dtype=FP16), full_position_ids)
                half_head_dim = head_dim // 2
                return torch.cat(
                    [
                        cos[:, :max_seq_len, :half_head_dim].float(),
                        sin[:, :max_seq_len, :half_head_dim].float(),
                    ],
                    dim=-1,
                )
        except Exception as exc:
            logger.warning("Falling back to text RoPE cache: %s", exc)

    from utils.plugin.plugin_utils import get_plugin_rope_cache

    return get_plugin_rope_cache(find_rotary_embedding(lm), max_seq_len, head_dim, device)


def install_plugin_attention(lm: nn.Module, config: Any, rope_cache: torch.Tensor) -> None:
    from utils.plugin.plugin_utils import PluginAttention

    for index, layer in enumerate(lm.layers):
        layer.self_attn = PluginAttention(layer.self_attn, config, index, rope_cache)


def export_plugin_wrapper(
    wrapper: nn.Module,
    example_embeds: torch.Tensor,
    example_kvs: list[torch.Tensor],
    example_ctx: torch.Tensor,
    example_ds: torch.Tensor,
    num_layers: int,
    max_seq_len: int,
) -> "torch.export.ExportedProgram":
    seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    dynamic_shapes = {
        "inputs_embeds": {1: seq_dim},
        "kv_caches": [{}] * num_layers,
        "ctx_len": {},
        "ds_stack": {},
    }
    export_args = (example_embeds, example_kvs, example_ctx, example_ds)
    try:
        return torch.export.export(
            wrapper,
            args=export_args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    except Exception:
        return torch.export._trace._export(
            wrapper,
            export_args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def build_plugin_lm_input_specs(
    *,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    num_ds_layers: int,
    dtype: torch.dtype,
    opt_seq_len: int,
) -> list[Any]:
    import torch_tensorrt

    seq_opt = max(1, min(int(opt_seq_len), int(max_seq_len)))
    embeds = torch_tensorrt.Input(
        min_shape=(batch_size, 1, hidden_size),
        opt_shape=(batch_size, seq_opt, hidden_size),
        max_shape=(batch_size, max_seq_len, hidden_size),
        dtype=dtype,
    )
    kvs = [
        torch_tensorrt.Input(
            shape=(batch_size, 2, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]
    ctx = torch_tensorrt.Input(shape=(batch_size,), dtype=torch.int32)
    ds = torch_tensorrt.Input(
        shape=(num_ds_layers, batch_size, max_seq_len, hidden_size),
        dtype=dtype,
    )
    return [embeds, kvs, ctx, ds]


def save_plugin_lm_engine(
    language_model: nn.Module,
    lm_head: nn.Module,
    path: str | Path,
    prefill_seq_len: int,
    num_ds_layers: int = 0,
    position_ids: Optional[torch.Tensor] = None,
    rope_deltas: Optional[torch.Tensor] = None,
    max_seq_len: int = 4096,
    batch_size: int = 1,
    device: str = "cuda",
    plugin_path: Optional[str] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> bool:
    import torch_tensorrt

    from utils.direct_engine_utils import patch_torchtrt_output_names
    from utils.engine_io import save_trt_engine
    from utils.plugin.plugin_utils import (
        load_plugin,
        register_plugin_op,
        set_plugin_config_from_model,
    )

    dev = torch.device(device)
    path = Path(path)
    load_plugin(plugin_path)
    register_plugin_op()

    config = language_model.config
    head_dim = int(config.head_dim)
    num_kv_heads = int(config.num_key_value_heads)
    num_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    set_plugin_config_from_model(config, max_seq_len)

    lm = copy.deepcopy(language_model).to(dtype=FP16, device=dev).eval()
    rope_cache = build_rope_cache(
        lm=lm,
        prefill_seq_len=prefill_seq_len,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        device=dev,
    )
    install_plugin_attention(lm, config, rope_cache)

    lm_head = copy.deepcopy(lm_head).to(dtype=FP16, device=dev).eval()
    wrapper = FlatPluginWrapperDSInput(lm, lm_head, num_ds_layers).to(device=dev).eval()

    batch = int(batch_size)
    example_embeds = torch.randn(batch, 3, hidden_size, dtype=FP16, device=dev)
    example_ctx = torch.tensor([3] * batch, dtype=torch.int32, device=dev)
    example_kvs = create_kv_caches_from_config(config, max_seq_len, batch, dev, FP16)
    ds_stack_layers = max(1, int(num_ds_layers))
    example_ds = torch.zeros(ds_stack_layers, batch, max_seq_len, hidden_size, dtype=FP16, device=dev)

    logger.info("Exporting plugin LM wrapper for serialized TensorRT engine")
    exported = export_plugin_wrapper(
        wrapper=wrapper,
        example_embeds=example_embeds,
        example_kvs=example_kvs,
        example_ctx=example_ctx,
        example_ds=example_ds,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    input_specs = build_plugin_lm_input_specs(
        batch_size=batch,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        num_ds_layers=ds_stack_layers,
        dtype=FP16,
        opt_seq_len=int(prefill_seq_len),
    )
    output_names = plugin_lm_output_names(num_layers)
    trt_settings: Dict[str, Any] = {
        "enabled_precisions": {torch.float32},
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "disable_tf32": True,
        "min_block_size": 1,
        "dryrun": False,
        "device": dev,
        "use_python_runtime": True,
        "decompose_attention": True,
    }

    logger.info("Serializing plugin LM engine with convert_exported_program_to_serialized_trt_engine")
    try:
        with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
            with patch_torchtrt_output_names(output_names):
                engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                    exported,
                    inputs=input_specs,
                    **trt_settings,
                )
    except Exception as exc:
        logger.error("Plugin LM TensorRT serialization failed: %s", exc)
        return False

    metadata = {
        "component": "llm_plugin",
        "save_format": "raw_trt_engine",
        "runtime_contract": "vlm_plugin_lm_ds_input",
        "precision": "FP16",
        "input_names": ["inputs_embeds", "kv_caches", "ctx_len", "ds_stack"],
        "output_names": output_names,
        "batch_size": batch,
        "prefill_seq_len": int(prefill_seq_len),
        "max_seq_len": int(max_seq_len),
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "num_ds_layers": int(num_ds_layers),
        "ds_stack_layers": int(ds_stack_layers),
        "kv_cache_shape": [batch, 2, num_kv_heads, int(max_seq_len), head_dim],
        "ds_stack_shape": [int(ds_stack_layers), batch, int(max_seq_len), hidden_size],
        "attention_plugin": "AttentionPlugin",
        "tensorrt_plugin_path": plugin_path,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    save_trt_engine(engine_bytes, path, metadata)

    del wrapper, lm, lm_head, exported, example_embeds, example_kvs, example_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Plugin LM engine saved to %s", path)
    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face language module to a serialized plugin TensorRT engine."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--model_class", default=None, help="Optional custom class import path.")
    parser.add_argument("--language_module", default=None, help="Optional dotted path to the LM module.")
    parser.add_argument("--lm_head_module", default=None, help="Optional dotted path to the LM head.")
    parser.add_argument(
        "--instantiate_from_config",
        action="store_true",
        help="For --model_class, construct from config instead of from_pretrained().",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_engine", default=None, help="Optional explicit output .engine/.trt path.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path. Defaults to --model.")
    parser.add_argument("--prompt", default="What is parallel programming?")
    parser.add_argument("--random_input", action="store_true", help="Use synthetic input ids.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_seq_len", type=int, default=3, help="Example text length for embeddings.")
    parser.add_argument(
        "--prefill_seq_len",
        type=int,
        default=None,
        help="RoPE/plugin prefill length. Defaults to --input_seq_len.",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048, help="KV-cache capacity.")
    parser.add_argument("--num_ds_layers", type=int, default=0, help="Number of active deepstack layers.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Optional HF attention implementation override while loading.",
    )
    parser.add_argument("--plugin_path", default=None, help="Optional libNvInfer_edgellm_plugin.so path.")
    parser.add_argument(
        "--position_ids_pt",
        default=None,
        help="Optional torch-saved position_ids tensor for VLM/VLA RoPE prefill.",
    )
    parser.add_argument(
        "--rope_deltas_pt",
        default=None,
        help="Optional torch-saved rope_deltas tensor for VLM/VLA RoPE prefill.",
    )
    parser.add_argument("--save_sample_tensors", action="store_true")
    parser.add_argument("--skip_engine_inspection", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def module_at_path_or_none(model: nn.Module, module_path: str) -> Optional[nn.Module]:
    try:
        return get_nested_module(model, module_path)
    except (AttributeError, TypeError, ValueError):
        return None


def looks_like_lm_leaf(module: Any) -> bool:
    return isinstance(module, nn.Module) and hasattr(module, "embed_tokens") and hasattr(module, "config")


def resolve_language_module(root_model: nn.Module, module_path: Optional[str]) -> nn.Module:
    if module_path:
        return get_nested_module(root_model, module_path)
    if looks_like_lm_leaf(root_model):
        return root_model
    for candidate_path in (
        "model.language_model",
        "language_model",
        "vlm.model.language_model",
        "vlm.language_model",
        "model",
        "transformer",
        "decoder",
    ):
        module = module_at_path_or_none(root_model, candidate_path)
        if looks_like_lm_leaf(module):
            return module
    raise ValueError("Could not resolve a language module. Pass --language_module explicitly.")


def resolve_lm_head(root_model: nn.Module, language_model: nn.Module, module_path: Optional[str]) -> nn.Module:
    if module_path:
        return get_nested_module(root_model, module_path)
    direct_head = getattr(language_model, "lm_head", None)
    if isinstance(direct_head, nn.Module):
        return direct_head
    get_language_output_embeddings = getattr(language_model, "get_output_embeddings", None)
    if callable(get_language_output_embeddings):
        head = get_language_output_embeddings()
        if isinstance(head, nn.Module):
            return head
    root_head = getattr(root_model, "lm_head", None)
    if isinstance(root_head, nn.Module):
        return root_head
    get_output_embeddings = getattr(root_model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        head = get_output_embeddings()
        if isinstance(head, nn.Module):
            return head
    for candidate_path in (
        "vlm.lm_head",
        "model.lm_head",
        "language_model.lm_head",
        "model.language_model.lm_head",
    ):
        head = module_at_path_or_none(root_model, candidate_path)
        if isinstance(head, nn.Module):
            return head
    raise ValueError("Could not resolve an LM head. Pass --lm_head_module explicitly.")


def resolve_text_config(language_model: nn.Module) -> Any:
    config = getattr(language_model, "config", None)
    for attr_name in ("text_config", "llm_config", "language_config"):
        nested = getattr(config, attr_name, None)
        if nested is not None:
            config = nested
            break
    if config is None:
        raise ValueError("Could not find a config on the selected language module.")
    if not hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = config.num_attention_heads
    if not hasattr(config, "head_dim") or config.head_dim is None:
        config.head_dim = config.hidden_size // config.num_attention_heads
    return config


def disable_use_cache(module: nn.Module) -> None:
    for config in (
        getattr(module, "config", None),
        getattr(getattr(module, "config", None), "text_config", None),
        getattr(getattr(module, "config", None), "llm_config", None),
        getattr(getattr(module, "config", None), "language_config", None),
        getattr(module, "generation_config", None),
    ):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def load_root_model(args: argparse.Namespace, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.model_class is not None:
        loader = import_object(args.model_class)
        if args.instantiate_from_config:
            config = loader.config_class.from_pretrained(args.model, trust_remote_code=True)
            return loader(config).eval()
        return loader.from_pretrained(args.model, **model_kwargs).eval()

    return AutoModelForCausalLM.from_pretrained(
        args.model,
        ignore_mismatched_sizes=True,
        **model_kwargs,
    ).eval()


def make_input_ids(args: argparse.Namespace, config: Any, device: torch.device) -> torch.Tensor:
    if args.random_input:
        if not hasattr(config, "vocab_size"):
            raise ValueError("--random_input requires config.vocab_size.")
        return torch.randint(
            low=1,
            high=int(config.vocab_size),
            size=(args.batch_size, args.input_seq_len),
            dtype=torch.long,
            device=device,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [args.prompt] * args.batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.input_seq_len,
    )
    input_ids = encoded["input_ids"].to(device=device, dtype=torch.long)
    if input_ids.shape[1] < args.input_seq_len:
        pad_width = args.input_seq_len - input_ids.shape[1]
        padding = torch.full(
            (input_ids.shape[0], pad_width),
            int(tokenizer.pad_token_id),
            dtype=input_ids.dtype,
            device=device,
        )
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


def load_optional_tensor(path: Optional[str], device: torch.device) -> Optional[torch.Tensor]:
    if path is None:
        return None
    value = torch.load(path, map_location=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, found {type(value).__name__}.")
    return value.to(device)


def sample_tensor_specs(
    language_model: nn.Module,
    config: Any,
    input_ids: torch.Tensor,
    max_seq_len: int,
    num_ds_layers: int,
    dtype: torch.dtype,
) -> Dict[str, Dict[str, Any]]:
    with torch.no_grad():
        inputs_embeds = language_model.embed_tokens(input_ids).to(dtype=dtype)
    kv_caches = create_kv_caches_from_config(config, max_seq_len, input_ids.shape[0], input_ids.device, dtype)
    ctx_len = torch.full((input_ids.shape[0],), input_ids.shape[1], dtype=torch.int32, device=input_ids.device)
    ds_stack = torch.zeros(
        max(1, int(num_ds_layers)),
        input_ids.shape[0],
        max_seq_len,
        int(config.hidden_size),
        dtype=dtype,
        device=input_ids.device,
    )
    return tensor_specs(
        {
            "inputs_embeds": inputs_embeds,
            "kv_caches.0": kv_caches[0] if kv_caches else torch.empty(0, device=input_ids.device),
            "ctx_len": ctx_len,
            "ds_stack": ds_stack,
        }
    )


def direct_compile_options() -> Dict[str, Any]:
    return {
        "enabled_precisions": ["float32"],
        "use_explicit_typing": True,
        "use_fp32_acc": True,
        "disable_tf32": True,
        "min_block_size": 1,
        "use_python_runtime": True,
        "decompose_attention": True,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.input_seq_len < 1:
        raise ValueError("--input_seq_len must be >= 1.")
    if args.max_seq_len < args.input_seq_len:
        raise ValueError("--max_seq_len must be >= --input_seq_len.")
    if args.num_ds_layers < 0:
        raise ValueError("--num_ds_layers must be >= 0.")
    if args.dtype != "float16":
        raise ValueError("The Edge-LLM AttentionPlugin path currently expects --dtype float16.")
    if (args.position_ids_pt is None) != (args.rope_deltas_pt is None):
        raise ValueError("Pass both --position_ids_pt and --rope_deltas_pt, or neither.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)

    root_model = load_root_model(args, dtype)
    disable_use_cache(root_model)
    language_model = resolve_language_module(root_model, args.language_module).to(device=device, dtype=dtype).eval()
    lm_head = resolve_lm_head(root_model, language_model, args.lm_head_module).to(device=device, dtype=dtype).eval()
    disable_use_cache(language_model)
    config = resolve_text_config(language_model)

    input_ids = make_input_ids(args, config, device)
    prefill_seq_len = int(args.prefill_seq_len or input_ids.shape[1])
    position_ids = load_optional_tensor(args.position_ids_pt, device)
    rope_deltas = load_optional_tensor(args.rope_deltas_pt, device)

    artifact_prefix = f"{safe_model_tag(args.model)}_lm"
    engine_path = (
        Path(args.output_engine)
        if args.output_engine is not None
        else output_dir / f"{artifact_prefix}_direct.engine"
    )
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    ok = save_plugin_lm_engine(
        language_model=language_model,
        lm_head=lm_head,
        path=engine_path,
        prefill_seq_len=prefill_seq_len,
        num_ds_layers=args.num_ds_layers,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        plugin_path=args.plugin_path,
        metadata_extra={
            "model": args.model,
            "model_class": args.model_class,
            "language_module": args.language_module,
            "lm_head_module": args.lm_head_module,
            "position_ids_source": args.position_ids_pt,
            "rope_deltas_source": args.rope_deltas_pt,
        },
        debug=args.debug,
    )
    if not ok:
        raise RuntimeError("Failed to serialize plugin LM engine.")

    engine_bytes = engine_path.read_bytes()
    engine_info: Dict[str, Any] = {}
    if not args.skip_engine_inspection:
        engine_info = inspect_serialized_engine(engine_bytes)

    sidecar_path = Path(str(engine_path) + ".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    manifest: Dict[str, Any] = {
        "model": args.model,
        "component": "lm",
        "format_version": 1,
        "runtime_contract": "vlm_plugin_lm_ds_input",
        "model_class": args.model_class,
        "language_module": args.language_module,
        "lm_head_module": args.lm_head_module,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "input_seq_len": int(input_ids.shape[1]),
        "prefill_seq_len": prefill_seq_len,
        "max_seq_len": args.max_seq_len,
        "num_ds_layers": args.num_ds_layers,
        "uses_attention_plugin": True,
        "attention_plugin": "AttentionPlugin",
        "artifacts": {
            "engine_sidecar": relative_or_absolute(sidecar_path, output_dir),
        },
        "tensor_inputs": sample_tensor_specs(
            language_model,
            config,
            input_ids,
            args.max_seq_len,
            args.num_ds_layers,
            dtype,
        ),
        "engine_sidecar": sidecar,
    }

    if args.save_sample_tensors:
        sample_path = output_dir / f"{artifact_prefix}_test_tensors.pt"
        torch.save({"input_ids": input_ids.detach().cpu()}, sample_path)
        manifest["artifacts"]["sample_tensors"] = sample_path.name

    add_shared_direct_engine_manifest_fields(
        manifest,
        engine_path=engine_path,
        output_dir=output_dir,
        engine_bytes=engine_bytes,
        compile_options=direct_compile_options(),
        engine_info=engine_info,
        custom_op_module="utils.plugin.plugin_utils",
        plugin_path=args.plugin_path,
        runtime_requirements_extra={"attention_plugin": "AttentionPlugin"},
    )
    write_manifest(output_dir, manifest)

    print(f"Saved artifacts to {output_dir}")
    print(f"Saved direct TensorRT engine to {engine_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

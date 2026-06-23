import ctypes
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import tensorrt as trt
import torch
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

logger = logging.getLogger(__name__)


def resolve_edge_plugin_path(plugin_path: Optional[str] = None) -> str:
    if plugin_path:
        return str(Path(plugin_path).expanduser().resolve())

    env_path = os.environ.get("EDGELLM_TRT_PLUGIN_SO", "").strip()
    if env_path:
        return str(Path(env_path).expanduser().resolve())

    legacy_env_path = os.environ.get("EDGELLM_PLUGIN_PATH", "").strip()
    if legacy_env_path:
        return str(Path(legacy_env_path).expanduser().resolve())

    candidates = [
        Path.cwd() / "build" / "cpp" / "libNvInfer_edgellm_plugin.so",
        Path.cwd() / "build" / "libNvInfer_edgellm_plugin.so",
        Path.cwd() / "libNvInfer_edgellm_plugin.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(candidates[-1].resolve())


def load_edge_plugin(plugin_path: Optional[str] = None) -> str:
    path = resolve_edge_plugin_path(plugin_path)
    if not Path(path).is_file():
        raise RuntimeError(f"Edge-LLM TensorRT plugin not found at {path}")
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    logger.info("Loaded Edge-LLM TensorRT plugin: %s", path)
    return path


def _has_torch_op(namespace: str, name: str) -> bool:
    return hasattr(torch.ops, namespace) and hasattr(getattr(torch.ops, namespace), name)


if not _has_torch_op("trt", "attention_plugin"):

    @torch.library.custom_op("trt::attention_plugin", mutates_args=())
    def attention_plugin(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        enable_tree_attention: bool,
        head_size: int,
        enable_fp8_kv_cache: bool,
        sliding_window_size: int = -1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        qkv_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del k, v, context_lengths, rope_rotary_cos_sin, kvcache_start_index
        del num_kv_heads, enable_tree_attention, enable_fp8_kv_cache
        del sliding_window_size, attention_mask, position_ids, qkv_scales
        batch_size, seq_len, _ = q.shape
        attn_output = torch.zeros(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )
        return attn_output, past_key_value.clone()

    @attention_plugin.register_fake
    def _attention_plugin_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        enable_tree_attention: bool,
        head_size: int,
        enable_fp8_kv_cache: bool,
        sliding_window_size: int = -1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        qkv_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del k, v, context_lengths, rope_rotary_cos_sin, kvcache_start_index
        del num_kv_heads, enable_tree_attention, enable_fp8_kv_cache
        del sliding_window_size, attention_mask, position_ids, qkv_scales
        batch_size, seq_len, _ = q.shape
        attn_output = torch.empty(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )
        return attn_output, torch.empty_like(past_key_value)


if not _has_torch_op("trt", "vit_attention_plugin"):

    @torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
    def vit_attention_plugin(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        del k, v, cu_seqlens, max_seqlen_carrier, num_heads, head_size
        return torch.zeros_like(q)

    @vit_attention_plugin.register_fake
    def _vit_attention_plugin_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        del k, v, cu_seqlens, max_seqlen_carrier, num_heads, head_size
        return torch.empty_like(q)


if not _has_torch_op("trt", "vit_masked_attention_plugin"):

    @torch.library.custom_op("trt::vit_masked_attention_plugin", mutates_args=())
    def vit_masked_attention_plugin(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        num_heads: int,
        head_size: int,
        qkv_fused: int = 1,
        mask_type: int = 0,
        max_seq_len: int = 0,
        mask_block_size: int = 0,
    ) -> torch.Tensor:
        del cos, sin, mask, qkv_fused, mask_type, max_seq_len, mask_block_size
        batch_size, seq_len, _ = qkv.shape
        return torch.zeros(
            batch_size,
            seq_len,
            num_heads * head_size,
            dtype=qkv.dtype,
            device=qkv.device,
        )

    @vit_masked_attention_plugin.register_fake
    def _vit_masked_attention_plugin_fake(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        num_heads: int,
        head_size: int,
        qkv_fused: int = 1,
        mask_type: int = 0,
        max_seq_len: int = 0,
        mask_block_size: int = 0,
    ) -> torch.Tensor:
        del cos, sin, mask, qkv_fused, mask_type, max_seq_len, mask_block_size
        return qkv.new_empty((qkv.shape[0], qkv.shape[1], num_heads * head_size))


@dynamo_tensorrt_converter(
    torch.ops.trt.attention_plugin.default, supports_dynamic_shapes=True
)
def convert_edge_attention_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, kv, ctx_len, rope, kv_cache_start_idx = args[:7]
    num_q_heads = args[7]
    num_kv_heads = args[8]
    enable_tree_attention = args[9]
    head_size = args[10]
    enable_fp8_kv_cache = args[11]
    sliding_window_size = args[12] if len(args) > 12 else -1
    attention_mask = args[13] if len(args) > 13 else None
    position_ids = args[14] if len(args) > 14 else None
    qkv_scales = args[15] if len(args) > 15 else None

    creator = trt.get_plugin_registry().get_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            field_name,
            np.array([field_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        )
        for field_name, field_val in [
            ("num_q_heads", int(num_q_heads)),
            ("num_kv_heads", int(num_kv_heads)),
            ("head_size", int(head_size)),
            ("enable_tree_attention", int(enable_tree_attention)),
            ("enable_fp8_kv_cache", int(enable_fp8_kv_cache)),
            ("sliding_window_size", int(sliding_window_size)),
        ]
    ]
    if bool(enable_fp8_kv_cache) and qkv_scales is not None:
        field_list.append(
            trt.PluginField(
                "qkv_scales",
                np.array(list(qkv_scales), dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            )
        )

    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create AttentionPlugin")

    plugin_inputs = [q, k, v, kv, ctx_len, rope, kv_cache_start_idx]
    if bool(enable_tree_attention):
        plugin_inputs.extend([attention_mask, position_ids])

    inputs = [
        get_trt_tensor(ctx, tensor, f"{name}_i{idx}")
        if not isinstance(tensor, trt.ITensor)
        else tensor
        for idx, tensor in enumerate(plugin_inputs)
    ]

    kv_cache_start_idx_input_idx = 6
    if (
        len(inputs[kv_cache_start_idx_input_idx].shape) == 2
        and inputs[kv_cache_start_idx_input_idx].shape[1] == 1
    ):
        shuffle_layer = ctx.net.add_shuffle(inputs[kv_cache_start_idx_input_idx])
        shuffle_layer.reshape_dims = (inputs[kv_cache_start_idx_input_idx].shape[0],)
        inputs[kv_cache_start_idx_input_idx] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    layer.name = name
    return layer.get_output(0), layer.get_output(1)


@dynamo_tensorrt_converter(
    torch.ops.trt.vit_attention_plugin.default, supports_dynamic_shapes=True
)
def convert_edge_vit_attention_plugin(ctx: ConversionContext, target, args, kwargs, name):
    del target, kwargs
    args = list(args)
    q, k, v, cu_seqlens, max_seqlen_carrier = args[:5]
    num_heads = args[5]
    head_size = args[6]

    creator = trt.get_plugin_registry().get_plugin_creator("ViTAttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("ViTAttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            "num_heads", np.array([int(num_heads)], dtype=np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            "head_size", np.array([int(head_size)], dtype=np.int32), trt.PluginFieldType.INT32
        ),
    ]
    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create ViTAttentionPlugin")

    inputs = []
    for idx, tensor in enumerate([q, k, v, cu_seqlens, max_seqlen_carrier]):
        tensor_name = f"{name}_i{idx}"
        trt_tensor = (
            get_trt_tensor(ctx, tensor, tensor_name)
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        if not trt_tensor.name:
            trt_tensor.name = tensor_name
        inputs.append(trt_tensor)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    layer.name = name
    output = layer.get_output(0)
    if not output.name:
        output.name = f"{name}_output"
    return output


def _converter_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dynamo_tensorrt_converter(
    torch.ops.trt.vit_masked_attention_plugin.default, supports_dynamic_shapes=True
)
def convert_edge_vit_masked_attention_plugin(
    ctx: ConversionContext, target, args, kwargs, name
):
    del target
    args = list(args)
    qkv, cos, sin, mask = args[:4]
    num_heads = args[4] if len(args) > 4 else kwargs.get("num_heads")
    head_size = args[5] if len(args) > 5 else kwargs.get("head_size")
    qkv_fused = args[6] if len(args) > 6 else kwargs.get("qkv_fused", 1)
    mask_type = args[7] if len(args) > 7 else kwargs.get("mask_type", 0)
    max_seq_len = args[8] if len(args) > 8 else kwargs.get("max_seq_len", 0)
    mask_block_size = args[9] if len(args) > 9 else kwargs.get("mask_block_size", 0)

    creator = trt.get_plugin_registry().get_plugin_creator(
        "ViTMaskedAttentionPlugin", "1", ""
    )
    if creator is None:
        raise RuntimeError("ViTMaskedAttentionPlugin not found in TensorRT plugin registry")

    field_list = [
        trt.PluginField(
            "num_heads",
            np.array([_converter_int(num_heads)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "head_size",
            np.array([_converter_int(head_size)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "qkv_fused",
            np.array([_converter_int(qkv_fused, 1)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "mask_type",
            np.array([_converter_int(mask_type)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "max_seq_len",
            np.array([_converter_int(max_seq_len)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "mask_block_size",
            np.array([_converter_int(mask_block_size)], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
    ]
    plugin = creator.create_plugin(name, trt.PluginFieldCollection(field_list))
    if plugin is None:
        raise RuntimeError("Failed to create ViTMaskedAttentionPlugin")

    inputs = []
    for idx, tensor in enumerate([qkv, cos, sin, mask]):
        tensor_name = f"{name}_i{idx}"
        trt_tensor = (
            get_trt_tensor(ctx, tensor, tensor_name)
            if not isinstance(tensor, trt.ITensor)
            else tensor
        )
        if not trt_tensor.name:
            trt_tensor.name = tensor_name
        inputs.append(trt_tensor)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    layer.name = name
    output = layer.get_output(0)
    if not output.name:
        output.name = f"{name}_output"
    return output

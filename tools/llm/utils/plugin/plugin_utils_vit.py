"""
Plugin utilities for TensorRT ViT inference with custom attention plugins.

This module provides Vision Transformer-specific utilities for using TensorRT
attention plugins with ViT models. Unlike LLMs, ViT models:
- Do not use KV caching (full bidirectional attention)
- Do not use RoPE (learnable/absolute position embeddings)
- Process fixed-size image patches at once
"""

import ctypes
import importlib
import inspect
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt

_TENSORRT_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKSPACE_ROOT = os.path.dirname(_TENSORRT_REPO_ROOT)

# Default plugin path for ViT attention plugin. TensorRT-Edge-LLM is checked out
# next to TensorRT in this workspace, not inside the TensorRT repository.
DEFAULT_PLUGIN_PATH = os.path.join(
    _WORKSPACE_ROOT,
    "TensorRT-Edge-LLM",
    "build",
    "libNvInfer_edgellm_plugin.so",
)

# Global configuration for ViT plugin converter
_VIT_PLUGIN_CONFIG: Dict[str, Any] = {}
VIT_MASK_TYPE_DENSE_ADDITIVE = 0
VIT_MASK_TYPE_PACKED_CU_SEQLENS = 1
VIT_MASK_TYPE_COMPACT_BLOCK = 2

def load_plugin(plugin_path: Optional[str] = None) -> bool:
    """
    Load the TensorRT attention plugin library.

    Args:
        plugin_path: Path to the plugin .so file. If None, uses DEFAULT_PLUGIN_PATH.

    Returns:
        True if plugin was loaded successfully, False otherwise.

    Raises:
        RuntimeError: If plugin file does not exist.
    """
    path = plugin_path or os.environ.get("TRT_EDGE_LLM_PLUGIN_PATH") or DEFAULT_PLUGIN_PATH
    if not os.path.exists(path):
        raise RuntimeError(f"Plugin not found at {path}")
    ctypes.CDLL(path)
    print(f"Loaded plugin: {path}")
    return True


def set_vit_plugin_config(
    num_attention_heads: int,
    head_dim: int,
    num_patches: int,
    max_batch_size: int = 4,
    mask_type: int = 0,
    max_seq_len: int = 0,
    mask_block_size: int = 0,
) -> None:
    """
    Set global configuration for the ViT plugin converter.

    Args:
        num_attention_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        num_patches: Number of image patches (including [CLS] token).
        max_batch_size: Maximum batch size.
        mask_type: Plugin mask mode. 0=dense additive mask, 1=packed cu_seqlens,
            2=compact block validity mask.
        max_seq_len: Maximum packed segment length for cu_seqlens FMHA.
        mask_block_size: Number of sequence tokens represented by one compact
            block-mask element when mask_type=2.
    """
    global _VIT_PLUGIN_CONFIG
    _VIT_PLUGIN_CONFIG = {
        "num_attention_heads": num_attention_heads,
        "head_dim": head_dim,
        "num_patches": num_patches,
        "max_batch_size": max_batch_size,
        "mask_type": mask_type,
        "max_seq_len": max_seq_len,
        "mask_block_size": mask_block_size,
    }

def get_vit_plugin_config() -> Dict[str, Any]:
    """Get the current ViT plugin configuration."""
    return _VIT_PLUGIN_CONFIG.copy()

def set_vit_plugin_config_from_model(model_config: Any) -> None:
    """
    Set ViT plugin configuration from a HuggingFace vision model config.

    Args:
        model_config: HuggingFace model configuration object.
    """
    # HuggingFace vision configs use slightly different field names across
    # families.
    num_heads = getattr(model_config, "num_attention_heads", None) or getattr(
        model_config, "attention_heads"
    )
    head_dim = model_config.hidden_size // num_heads
    
    # Calculate number of patches from image size
    image_size = model_config.image_size
    patch_size = model_config.patch_size
    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size
    else:
        image_h = image_w = image_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size
    num_patches = (image_h // patch_h) * (image_w // patch_w) + 1  # +1 for [CLS]

    set_vit_plugin_config(
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_patches=num_patches,
    )


# -----------------------------------------------------------------------------
# Plugin Op Registration
# -----------------------------------------------------------------------------

def _register_vit_plugin_op_impl() -> None:
    """
    Internal implementation to register the tensorrt_vit::attention custom op for PyTorch.

    ViT attention differs from LLM attention:
    - No KV cache - full bidirectional attention
    - Simple fused QKV input
    - Single output - no separate KV output
    """

    @torch.library.custom_op("tensorrt_vit::attention", mutates_args=())
    def attn(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask_or_cu_seqlens: torch.Tensor,
        num_heads: int,
        head_dim: int,
        qkv_fused: int = 1,
        mask_type: int = 0,
        max_seq_len: int = 0,
        mask_block_size: int = 0,
    ) -> torch.Tensor:
        """
        ViT attention operation.

        Args:
            qkv: Fused [Q, K, V] tensor of shape [B, S, (H*D*3)].
            cos: RoPE cosine tensor of shape [S, D].
            sin: RoPE sine tensor of shape [S, D].
            mask_or_cu_seqlens: Dense additive mask or INT32 cu_seqlens.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            qkv_fused: Whether QKV is fused (1=yes, 0=no).
            mask_type: 0 for dense additive mask, 1 for packed cu_seqlens,
                2 for compact block validity mask.
            max_seq_len: Max segment length when mask_type=1.
            mask_block_size: Tokens per compact mask block when mask_type=2.

        Returns:
            Attention output of shape [B, S, H*D].
        """
        batch_size, seq_len, _ = qkv.shape
        output_dim = num_heads * head_dim
        attn_out = torch.zeros(
            batch_size, seq_len, output_dim, dtype=qkv.dtype, device=qkv.device
        )
        return attn_out

    @torch.library.register_fake("tensorrt_vit::attention")
    def _(qkv, cos, sin, mask_or_cu_seqlens, num_heads, head_dim, qkv_fused=1, mask_type=0, max_seq_len=0, mask_block_size=0):
        batch_size, seq_len, _ = qkv.shape
        output_dim = num_heads * head_dim
        attn_out = torch.empty(
            batch_size, seq_len, output_dim, dtype=qkv.dtype, device=qkv.device
        )
        return attn_out


def register_vit_plugin_op() -> None:
    """
    Register the tensorrt_vit::attention custom op for PyTorch.

    This function is idempotent - safe to call multiple times.
    """
    if hasattr(torch.ops, "tensorrt_vit") and hasattr(
        torch.ops.tensorrt_vit, "attention"
    ):
        return
    _register_vit_plugin_op_impl()


# Register the op at module import time so the converter decorator works
if not (
    hasattr(torch.ops, "tensorrt_vit")
    and hasattr(torch.ops.tensorrt_vit, "attention")
):
    _register_vit_plugin_op_impl()

# Importing plugin_converter_vit at the bottom of this file registers the
# Torch-TensorRT converter for tensorrt_vit::attention.

from .plugin_converter_vit import (  # noqa: E402 (must be after op registration)
    convert_vit_attention,
    get_vit_plugin_conversion_count,
    reset_vit_plugin_conversion_count,
)

# -----------------------------------------------------------------------------
# Plugin Attention Module
# -----------------------------------------------------------------------------

class ViTPluginAttention(nn.Module):
    """
    Model-agnostic ViT attention wrapper using the TensorRT ViT attention plugin.

    The wrapper follows the same idea as the LLM plugin path: infer the attention
    module layout from the original module instead of requiring a separate hand
    written implementation for every model family. It supports common vision
    attention layouts:
    - fused QKV projection: qkv + proj
    - separate Q/K/V: q_proj/k_proj/v_proj + o_proj
    - HuggingFace ViT: query/key/value + output.dense

    RoPE is also inferred from the forward inputs. Models that pass
    position_embeddings=(cos, sin) use those tensors; models without visual RoPE
    get identity cos/sin tensors.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        config: Any,
        layer_idx: int,
        return_tuple: bool = False,
        use_plugin_op: bool = True,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        self.return_tuple = return_tuple
        self.use_plugin_op = use_plugin_op

        self.projection_layout = self._detect_projection_layout(original_attn)
        self.output_proj = self._detect_output_projection(original_attn)
        self.num_heads = self._detect_num_heads(original_attn, config)
        self.head_dim = self._detect_head_dim(original_attn, config, self.num_heads)
        self.q_norm = self._detect_optional_norm(
            original_attn,
            ("q_norm", "query_norm", "q_layernorm", "query_layernorm"),
        )
        self.k_norm = self._detect_optional_norm(
            original_attn,
            ("k_norm", "key_norm", "k_layernorm", "key_layernorm"),
        )

    def _detect_projection_layout(self, original_attn: nn.Module) -> str:
        if hasattr(original_attn, "qkv"):
            return "fused_qkv"
        if all(hasattr(original_attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            return "separate_qkv"
        if all(hasattr(original_attn, name) for name in ("query", "key", "value")):
            return "hf_vit_qkv"
        raise ValueError(
            "Unsupported ViT attention projection layout. Expected qkv, "
            "q_proj/k_proj/v_proj, or query/key/value projections."
        )

    def _detect_output_projection(self, original_attn: nn.Module) -> nn.Module:
        for name in ("proj", "o_proj", "out_proj", "out"):
            if hasattr(original_attn, name):
                return getattr(original_attn, name)
        if hasattr(original_attn, "output"):
            output = original_attn.output
            return output.dense if hasattr(output, "dense") else output
        raise ValueError(
            "Unsupported ViT attention output projection layout. Expected proj, "
            "o_proj, out_proj, out, or output(.dense)."
        )

    def _detect_num_heads(self, original_attn: nn.Module, config: Any) -> int:
        for source in (original_attn, config):
            for name in ("num_heads", "attention_heads", "num_attention_heads"):
                value = getattr(source, name, None)
                if value is not None:
                    return int(value)
        raise ValueError("Could not infer number of attention heads for ViT plugin.")

    def _detect_head_dim(
        self, original_attn: nn.Module, config: Any, num_heads: int
    ) -> int:
        for source in (original_attn, config):
            value = getattr(source, "head_dim", None)
            if value is not None:
                return int(value)

        hidden_size = None
        for source in (config, original_attn):
            for name in ("hidden_size", "embed_dim", "dim"):
                value = getattr(source, name, None)
                if value is not None:
                    hidden_size = int(value)
                    break
            if hidden_size is not None:
                break
        if hidden_size is None:
            raise ValueError("Could not infer hidden size for ViT plugin head_dim.")
        return hidden_size // num_heads

    def _detect_optional_norm(
        self, original_attn: nn.Module, names: Tuple[str, ...]
    ) -> Optional[nn.Module]:
        for name in names:
            norm = getattr(original_attn, name, None)
            if isinstance(norm, nn.Module):
                return norm
        return None

    def _norm_last_dim(self, norm: nn.Module) -> Optional[int]:
        normalized_shape = getattr(norm, "normalized_shape", None)
        if isinstance(normalized_shape, int):
            return normalized_shape
        if isinstance(normalized_shape, (tuple, list)) and normalized_shape:
            return int(normalized_shape[-1])
        weight = getattr(norm, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() > 0:
            return int(weight.shape[-1])
        return None

    def _apply_optional_norm(
        self, tensor: torch.Tensor, norm: Optional[nn.Module]
    ) -> torch.Tensor:
        if norm is None:
            return tensor

        batch_size, seq_len, hidden_size = tensor.shape
        norm_last_dim = self._norm_last_dim(norm)
        if norm_last_dim == self.head_dim:
            tensor = tensor.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            tensor = norm(tensor)
            return tensor.reshape(batch_size, seq_len, hidden_size)
        return norm(tensor)

    def _project_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection_layout == "fused_qkv":
            return self.original_attn.qkv(hidden_states)
        if self.projection_layout == "separate_qkv":
            q = self.original_attn.q_proj(hidden_states)
            k = self.original_attn.k_proj(hidden_states)
            v = self.original_attn.v_proj(hidden_states)
            q = self._apply_optional_norm(q, self.q_norm)
            k = self._apply_optional_norm(k, self.k_norm)
            return torch.cat([q, k, v], dim=-1)

        q = self.original_attn.query(hidden_states)
        k = self.original_attn.key(hidden_states)
        v = self.original_attn.value(hidden_states)
        q = self._apply_optional_norm(q, self.q_norm)
        k = self._apply_optional_norm(k, self.k_norm)
        return torch.cat([q, k, v], dim=-1)

    def _get_rope_tensors(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[-2]
        if position_embeddings is None:
            cos = torch.ones(
                seq_len,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            sin = torch.zeros_like(cos)
            return cos, sin

        cos, sin = position_embeddings
        if not self.use_plugin_op:
            return cos.to(device=hidden_states.device), sin.to(device=hidden_states.device)
        return (
            cos.to(device=hidden_states.device, dtype=hidden_states.dtype),
            sin.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)

        if attention_mask.dim() == 1 and attention_mask.dtype == torch.int32:
            return attention_mask
        if (
            attention_mask.dim() == 2
            and attention_mask.dtype in (torch.int32, torch.bool)
            and seq_len % attention_mask.shape[-1] == 0
        ):
            return attention_mask.to(device=device, dtype=torch.int32)

        attention_mask = attention_mask.to(dtype=dtype)
        if attention_mask.dim() == 4:
            if attention_mask.shape[1] == 1:
                attention_mask = attention_mask[:, 0, :, :]
            else:
                attention_mask = attention_mask.reshape(
                    attention_mask.shape[0] * attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )
        return attention_mask

    def _compact_block_mask_to_dense(
        self,
        block_mask: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_blocks = block_mask.shape[-1]
        if num_blocks <= 0 or seq_len % num_blocks != 0:
            raise ValueError(
                "Compact block mask requires seq_len to be divisible by the number of blocks."
            )
        block_size = seq_len // num_blocks
        token_mask = block_mask.to(device=device, dtype=torch.bool).repeat_interleave(
            block_size,
            dim=-1,
        )
        dense_mask = torch.zeros(
            token_mask.shape[0],
            seq_len,
            seq_len,
            dtype=dtype,
            device=device,
        )
        return dense_mask.masked_fill(~token_mask.unsqueeze(1), torch.finfo(dtype).min)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _torch_attention(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(v.dtype)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_seq_len: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        squeeze_batch = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_batch = True

        batch_size, seq_len, _ = hidden_states.shape
        qkv = self._project_qkv(hidden_states)
        cos, sin = self._get_rope_tensors(hidden_states, position_embeddings)
        attention_mask = self._normalize_attention_mask(
            attention_mask,
            batch_size,
            seq_len,
            hidden_states.dtype,
            hidden_states.device,
        )
        mask_type = VIT_MASK_TYPE_DENSE_ADDITIVE
        mask_block_size = 0
        if attention_mask.dim() == 1 and attention_mask.dtype == torch.int32:
            mask_type = VIT_MASK_TYPE_PACKED_CU_SEQLENS
            if max_seq_len <= 0:
                max_seq_len = seq_len
        elif attention_mask.dim() == 2 and attention_mask.dtype == torch.int32:
            num_blocks = attention_mask.shape[-1]
            if num_blocks <= 0 or seq_len % num_blocks != 0:
                raise ValueError(
                    "Compact block attention mask requires seq_len to be divisible "
                    "by the number of blocks."
                )
            mask_type = VIT_MASK_TYPE_COMPACT_BLOCK
            mask_block_size = seq_len // num_blocks

        if self.use_plugin_op:
            attn_out = torch.ops.tensorrt_vit.attention.default(
                qkv,
                cos,
                sin,
                attention_mask,
                self.num_heads,
                self.head_dim,
                1,
                mask_type,
                max_seq_len,
                mask_block_size,
            )
        else:
            if mask_type == VIT_MASK_TYPE_PACKED_CU_SEQLENS:
                raise ValueError("PyTorch reference attention requires a dense mask.")
            if mask_type == VIT_MASK_TYPE_COMPACT_BLOCK:
                attention_mask = self._compact_block_mask_to_dense(
                    attention_mask,
                    seq_len,
                    hidden_states.dtype,
                    hidden_states.device,
                )
            attn_out = self._torch_attention(qkv, cos, sin, attention_mask)
        output = self.output_proj(attn_out)
        output = output.squeeze(0) if squeeze_batch else output
        if self.return_tuple:
            return output, None
        return output


# -----------------------------------------------------------------------------
# Model Wrappers
# -----------------------------------------------------------------------------

VIT_INPUT_CONTRACT_NATIVE = "native"
VIT_INPUT_CONTRACT_GRID_THW = "grid_thw"
VIT_INPUT_CONTRACT_STATIC_GRID_THW = "static_grid_thw"
VIT_INPUT_CONTRACT_WINDOWED_ROPE = "windowed_rope"
VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO = "tiled_aspect_ratio"
VIT_INPUT_CONTRACT_EDGE_LLM_SINGLE_INPUT = "edgellm_single_input"
VIT_INPUT_CONTRACT_EDGE_LLM_ROTARY = "edgellm_rotary"
VIT_INPUT_CONTRACT_EDGE_LLM_WINDOWED_ROTARY = "edgellm_windowed_rotary"
VIT_INPUT_CONTRACT_EDGE_LLM_FAST_POS_DEEPSTACK = "edgellm_fast_pos_deepstack"

def _require_tensor(value: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"ViT plugin forward requires {name}.")
    return value


def _supports_static_grid_thw_internal_forward(model: nn.Module) -> bool:
    return all(
        hasattr(model, attr_name)
        for attr_name in (
            "patch_embed",
            "fast_pos_embed_interpolate",
            "rot_pos_emb",
            "blocks",
        )
    )


def _make_static_grid_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    grid_thw_list = grid_thw.detach().cpu().tolist()
    cu_seqlens = [0]
    total = 0
    for num_frames, height, width in grid_thw_list:
        for _ in range(int(num_frames)):
            total += int(height) * int(width)
            cu_seqlens.append(total)
    return torch.tensor(cu_seqlens, dtype=torch.int32, device=grid_thw.device)


def _forward_static_grid_thw_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    pos_embeds: Optional[torch.Tensor],
    rotary_pos_emb: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    pos_embeds = _require_tensor(pos_embeds, "static_grid_pos_embeds")
    rotary_pos_emb = _require_tensor(rotary_pos_emb, "static_grid_rotary_pos_emb")
    cu_seqlens = _require_tensor(cu_seqlens, "static_grid_cu_seqlens")

    hidden_states = model.patch_embed(pixel_values)
    hidden_states = hidden_states + pos_embeds.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = cu_seqlens.to(device=hidden_states.device)

    for block in model.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
    
    '''
        For Alpamayo/Qwen-style vision towers, the image is first converted into patch embeddings, 
        passed through the ViT transformer blocks, and then sent through a final merger/projector 
        before being used by the VLM. The tensor after the ViT blocks is only an intermediate representation, 
        in our case [784, 1152], meaning 784 patch tokens with hidden size 1152. Alpamayo’s 
        final vision output runs this through the merger, which combines patch groups and projects 
        the features to [196, 4096], the embedding shape expected by the language/multimodal side. 
        
        Our TensorRT engine was originally returning the intermediate ViT output, while the PyTorch 
        reference was using the final merged output, which caused the shape mismatch.
    '''
    if hasattr(model, "merger"):
        hidden_states = model.merger(hidden_states)
    return hidden_states


def _get_windowed_rope_visual_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "patch_embed") and hasattr(model, "blocks"):
        return model
    raise ValueError("Cannot find a windowed-RoPE visual backbone.")


def _get_windowed_rope_blocks(visual_model: nn.Module) -> nn.ModuleList:
    if hasattr(visual_model, "blocks"):
        return visual_model.blocks
    raise ValueError("Cannot find windowed-RoPE visual blocks.")


def _forward_windowed_rope_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    window_attention_mask: Optional[torch.Tensor] = None,
    cu_window_seqlens: Optional[torch.Tensor] = None,
    window_index: Optional[torch.Tensor] = None,
    reverse_window_index: Optional[torch.Tensor] = None,
    max_window_seq_len: int = 0,
    **kwargs,
) -> torch.Tensor:
    visual = _get_windowed_rope_visual_model(model)
    rotary_pos_emb = _require_tensor(rotary_pos_emb, "rotary_pos_emb")
    attention_mask = _require_tensor(attention_mask, "attention_mask")
    window_attention_mask = _require_tensor(
        window_attention_mask, "window_attention_mask"
    )
    cu_window_seqlens = _require_tensor(cu_window_seqlens, "cu_window_seqlens")
    window_index = _require_tensor(window_index, "window_index")
    reverse_window_index = _require_tensor(reverse_window_index, "reverse_window_index")

    hidden_states = visual.patch_embed(pixel_values)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    blocks = _get_windowed_rope_blocks(visual)
    for layer_idx, block in enumerate(blocks):
        full_attention = layer_idx in visual.fullatt_block_indexes
        attention_mask_now = attention_mask if full_attention else window_attention_mask

        residual = hidden_states
        hidden_states = block.norm1(hidden_states)
        hidden_states = block.attn(
            hidden_states,
            attention_mask=attention_mask_now,
            position_embeddings=position_embeddings,
            max_seq_len=0,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = block.norm2(hidden_states)
        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = visual.merger(hidden_states)
    hidden_states = hidden_states[reverse_window_index, :]
    return hidden_states




def _forward_tiled_aspect_ratio_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    aspect_ratio_ids = _require_tensor(aspect_ratio_ids, "aspect_ratio_ids")
    attention_mask = _require_tensor(attention_mask, "attention_mask")
    vision = model.vision_model if hasattr(model, "vision_model") else model
    batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
        pixel_values.shape
    )
    pixel_values = pixel_values.reshape(
        batch_size * num_concurrent_media * num_tiles,
        num_channels,
        height,
        width,
    )
    aspect_ratio_ids = aspect_ratio_ids.reshape(
        batch_size * num_concurrent_media, -1
    )
    if attention_mask.dim() == 3:
        compact_attention_mask = attention_mask.reshape(
            batch_size * num_concurrent_media,
            -1,
        )
    elif attention_mask.dim() == 2:
        compact_attention_mask = attention_mask
    else:
        raise ValueError(
            "Tiled aspect-ratio vision requires a compact mask with shape "
            "[B, media, tiles] or [B*media, tiles]."
        )
    compact_attention_mask = compact_attention_mask.to(
        device=pixel_values.device,
        dtype=torch.int32,
    )

    target_dtype = vision.patch_embedding.weight.dtype
    target_device = vision.patch_embedding.weight.device
    patch_input = pixel_values.to(target_device, target_dtype)
    patch_padding = vision.patch_embedding.padding
    if patch_padding == "valid":
        patch_padding = (0, 0)
    patch_embeds = torch.nn.functional.conv2d(
        patch_input,
        vision.patch_embedding.weight,
        vision.patch_embedding.bias,
        vision.patch_embedding.stride,
        patch_padding,
        vision.patch_embedding.dilation,
        vision.patch_embedding.groups,
    )
    hidden_state = patch_embeds.flatten(2).transpose(1, 2)

    _, num_patches, dim = hidden_state.shape
    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        -1,
        dim,
    )
    hidden_state = vision.pre_tile_positional_embedding(
        hidden_state,
        aspect_ratio_ids,
    )

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media * num_tiles,
        num_patches,
        dim,
    )
    hidden_state = vision.apply_class_embedding(hidden_state)
    num_patches += 1

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches,
        dim,
    )
    hidden_state = vision.gated_positional_embedding(
        hidden_state,
        aspect_ratio_ids,
    )
    hidden_state = vision.layernorm_pre(hidden_state)

    num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
    hidden_state = torch.nn.functional.pad(
        hidden_state,
        (0, 0, 0, num_padding_patches),
        mode="constant",
        value=0,
    )
    slice_index = -num_padding_patches if num_padding_patches > 0 else None

    hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
    transformer_kwargs = {"attention_mask": compact_attention_mask}
    if "output_hidden_states" in inspect.signature(
        vision.transformer.forward
    ).parameters:
        transformer_kwargs["output_hidden_states"] = True
    output = vision.transformer(hidden_state, **transformer_kwargs)
    hidden_state = output.last_hidden_state
    hidden_state = vision.layernorm_post(hidden_state)

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches + num_padding_patches,
        dim,
    )
    hidden_state = vision.post_tile_positional_embedding(
        hidden_state,
        aspect_ratio_ids,
    )
    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles * (num_patches + num_padding_patches),
        dim,
    )
    global_output = vision.global_transformer(
        hidden_state,
        attention_mask=compact_attention_mask,
    )
    hidden_state = global_output.last_hidden_state

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches + num_padding_patches,
        dim,
    )
    hidden_state = hidden_state[:, :, :slice_index]
    hidden_state = hidden_state.reshape(
        batch_size,
        num_concurrent_media,
        num_tiles,
        num_patches,
        dim,
    )

    all_intermediate_hidden_states = [
        output.hidden_states[i] for i in vision.intermediate_layers_indices
    ]
    intermediate_hidden_states = torch.stack(
        all_intermediate_hidden_states,
        dim=-1,
    )
    intermediate_hidden_states = intermediate_hidden_states.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches + num_padding_patches,
        -1,
    )
    intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
    intermediate_hidden_states = intermediate_hidden_states.reshape(
        batch_size,
        num_concurrent_media,
        num_tiles,
        num_patches,
        -1,
    )

    return torch.cat([hidden_state, intermediate_hidden_states], dim=-1)


def _forward_native_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    output = model(pixel_values, **kwargs)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@dataclass
class RuntimeVisionContract:
    """Concrete tensor contract for a runtime-facing visual engine."""

    name: str
    wrapper: nn.Module
    core_inputs: Dict[str, torch.Tensor]
    output_names: List[str]


def _dense_visual_attention_mask(
    pixel_values: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    seq_len = int(pixel_values.shape[0])
    return torch.zeros(1, seq_len, seq_len, dtype=dtype, device=device)


def _fast_pos_embed_interpolate_inputs(
    visual: nn.Module,
    image_grid_thw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build Edge-LLM fast-position embedding inputs from structural vision attrs.

    The logic mirrors the optimized Qwen-style export path, but this helper is
    keyed by required attributes instead of model_type strings.
    """
    if hasattr(visual, "fast_pos_embed_interpolate_optimized"):
        return visual.fast_pos_embed_interpolate_optimized(image_grid_thw)

    missing = [
        name
        for name in ("num_grid_per_side", "config", "pos_embed")
        if not hasattr(visual, name)
    ]
    if missing:
        raise ValueError(
            "Cannot build fast position embedding inputs; missing visual "
            f"attributes: {missing}"
        )

    grid_thw = image_grid_thw.detach().to(device="cpu")
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    num_grid_per_side = int(getattr(visual, "num_grid_per_side"))
    merge_size = int(getattr(visual.config, "spatial_merge_size"))

    idx_list: List[List[int]] = [[] for _ in range(4)]
    weight_list: List[List[float]] = [[] for _ in range(4)]

    for t_value, h_value, w_value in zip(grid_ts, grid_hs, grid_ws):
        h = int(h_value)
        w = int(w_value)
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clip(max=num_grid_per_side - 1)
        w_ceil = (w_floor + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        base_h = h_floor * num_grid_per_side
        base_h_ceil = h_ceil * num_grid_per_side

        merged_h = h // merge_size
        merged_w = w // merge_size
        if merged_h <= 0 or merged_w <= 0:
            raise ValueError(
                "Invalid visual grid for Edge-LLM fast position embedding: "
                f"grid_thw entry {(int(t_value), h, w)} with merge_size={merge_size}"
            )

        indices = [
            (base_h.reshape(merged_h, 1, merge_size, 1)
             + w_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
            (base_h.reshape(merged_h, 1, merge_size, 1)
             + w_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
            (base_h_ceil.reshape(merged_h, 1, merge_size, 1)
             + w_floor.reshape(1, merged_w, 1, merge_size)).flatten(),
            (base_h_ceil.reshape(merged_h, 1, merge_size, 1)
             + w_ceil.reshape(1, merged_w, 1, merge_size)).flatten(),
        ]

        weights = [
            ((1 - dh).reshape(merged_h, 1, merge_size, 1)
             * (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
            ((1 - dh).reshape(merged_h, 1, merge_size, 1)
             * dw.reshape(1, merged_w, 1, merge_size)).flatten(),
            (dh.reshape(merged_h, 1, merge_size, 1)
             * (1 - dw).reshape(1, merged_w, 1, merge_size)).flatten(),
            (dh.reshape(merged_h, 1, merge_size, 1)
             * dw.reshape(1, merged_w, 1, merge_size)).flatten(),
        ]

        for index in range(4):
            idx_list[index].extend(int(value) for value in indices[index].tolist())
            weight_list[index].extend(float(value) for value in weights[index].tolist())

    device = visual.pos_embed.weight.device
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(
        weight_list,
        dtype=visual.pos_embed.weight.dtype,
        device=device,
    )
    return idx_tensor, weight_tensor


class EdgeLLMSingleInputVisualWrapper(nn.Module):
    """Expose a native visual tower with Edge-LLM's single-input binding name."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _forward_native_vision(self.model, input)


class EdgeLLMRotaryVisualWrapper(nn.Module):
    """Compose a rotary visual tower from runtime-provided RoPE and mask tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.patch_embed(input)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for block in self.model.blocks:
            residual = hidden_states
            hidden_states = block.norm1(hidden_states)
            hidden_states = block.attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = block.norm2(hidden_states)
            hidden_states = block.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return self.model.merger(hidden_states)


class EdgeLLMWindowedRotaryVisualWrapper(nn.Module):
    """Compose a windowed rotary visual tower from Edge-LLM runtime tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        window_attention_mask: torch.Tensor,
        window_index: torch.Tensor,
        reverse_window_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.patch_embed(input)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.model.spatial_merge_unit,
            self.model.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.model.spatial_merge_unit,
            self.model.spatial_merge_unit,
            -1,
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, block in enumerate(self.model.blocks):
            attention_mask_now = (
                attention_mask
                if layer_num in self.model.fullatt_block_indexes
                else window_attention_mask
            )
            residual = hidden_states
            hidden_states = block.norm1(hidden_states)
            hidden_states = block.attn(
                hidden_states,
                attention_mask=attention_mask_now,
                position_embeddings=position_embeddings,
                max_seq_len=0,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = block.norm2(hidden_states)
            hidden_states = block.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.model.merger(hidden_states)
        return hidden_states[reverse_window_index, :]


class EdgeLLMFastPosDeepstackVisualWrapper(nn.Module):
    """Compose a fast-position/deepstack visual tower for Edge-LLM bindings."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.deepstack_visual_indexes = [
            int(index) for index in getattr(model, "deepstack_visual_indexes", [])
        ]

    def forward(
        self,
        input: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor,
        fast_pos_embed_idx: torch.Tensor,
        fast_pos_embed_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        hidden_states = self.model.patch_embed(input)
        pos_embeds = (
            self.model.pos_embed(fast_pos_embed_idx)
            * fast_pos_embed_weight[:, :, None]
        )
        hidden_states = hidden_states + (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_features: List[torch.Tensor] = []
        for layer_num, block in enumerate(self.model.blocks):
            residual = hidden_states
            hidden_states = block.norm1(hidden_states)
            hidden_states = block.attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = block.norm2(hidden_states)
            hidden_states = block.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if layer_num in self.deepstack_visual_indexes:
                deepstack_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self.model.deepstack_merger_list[deepstack_idx](hidden_states)
                )

        output = self.model.merger(hidden_states)
        return (output, *deepstack_features)


def _has_attrs(module: nn.Module, names: Tuple[str, ...]) -> bool:
    return all(hasattr(module, name) for name in names)


def _prepare_edge_llm_rotary_inputs(
    visual: nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    return {
        "rotary_pos_emb": visual.rot_pos_emb(image_grid_thw).to(device=device),
        "attention_mask": _dense_visual_attention_mask(pixel_values, dtype, device),
    }


def prepare_edge_llm_runtime_contract(
    visual: nn.Module,
    processor_inputs: Dict[str, torch.Tensor],
    pixel_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> RuntimeVisionContract:
    """
    Build an Edge-LLM-compatible visual engine contract from module structure.

    This intentionally avoids model-name registries. It uses the runtime-facing
    tensor contract implied by the selected visual module's capabilities.
    """
    image_grid_thw = processor_inputs.get("image_grid_thw")

    if (
        isinstance(image_grid_thw, torch.Tensor)
        and _has_attrs(
            visual,
            (
                "patch_embed",
                "pos_embed",
                "rot_pos_emb",
                "blocks",
                "merger",
                "deepstack_visual_indexes",
                "deepstack_merger_list",
            ),
        )
    ):
        fast_pos_embed_idx, fast_pos_embed_weight = _fast_pos_embed_interpolate_inputs(
            visual,
            image_grid_thw,
        )
        core_inputs = {
            **_prepare_edge_llm_rotary_inputs(
                visual,
                pixel_values,
                image_grid_thw,
                device,
                dtype,
            ),
            "fast_pos_embed_idx": fast_pos_embed_idx.to(device=device, dtype=torch.long),
            "fast_pos_embed_weight": fast_pos_embed_weight.to(
                device=device,
                dtype=dtype,
            ),
        }
        deepstack_count = len(getattr(visual, "deepstack_visual_indexes", []))
        return RuntimeVisionContract(
            name=VIT_INPUT_CONTRACT_EDGE_LLM_FAST_POS_DEEPSTACK,
            wrapper=EdgeLLMFastPosDeepstackVisualWrapper(visual).eval().to(device),
            core_inputs=core_inputs,
            output_names=[
                "output",
                *[f"deepstack_features.{index}" for index in range(deepstack_count)],
            ],
        )

    if (
        isinstance(image_grid_thw, torch.Tensor)
        and _has_attrs(
            visual,
            (
                "patch_embed",
                "rot_pos_emb",
                "blocks",
                "merger",
                "fullatt_block_indexes",
                "spatial_merge_unit",
                "get_window_index",
            ),
        )
    ):
        window_inputs, _ = make_windowed_rope_core_inputs(
            visual,
            pixel_values,
            image_grid_thw,
            device,
            dtype,
        )
        core_inputs = {
            name: window_inputs[name]
            for name in (
                "rotary_pos_emb",
                "attention_mask",
                "window_attention_mask",
                "window_index",
                "reverse_window_index",
            )
        }
        return RuntimeVisionContract(
            name=VIT_INPUT_CONTRACT_EDGE_LLM_WINDOWED_ROTARY,
            wrapper=EdgeLLMWindowedRotaryVisualWrapper(visual).eval().to(device),
            core_inputs=core_inputs,
            output_names=["output"],
        )

    if (
        isinstance(image_grid_thw, torch.Tensor)
        and _has_attrs(visual, ("patch_embed", "rot_pos_emb", "blocks", "merger"))
    ):
        return RuntimeVisionContract(
            name=VIT_INPUT_CONTRACT_EDGE_LLM_ROTARY,
            wrapper=EdgeLLMRotaryVisualWrapper(visual).eval().to(device),
            core_inputs=_prepare_edge_llm_rotary_inputs(
                visual,
                pixel_values,
                image_grid_thw,
                device,
                dtype,
            ),
            output_names=["output"],
        )

    return RuntimeVisionContract(
        name=VIT_INPUT_CONTRACT_EDGE_LLM_SINGLE_INPUT,
        wrapper=EdgeLLMSingleInputVisualWrapper(visual).eval().to(device),
        core_inputs={},
        output_names=["output"],
    )


class ViTPluginWrapper(nn.Module):
    """
    Generic wrapper for vision models with plugin attention.

    The caller chooses the tensor input contract and provides the corresponding
    tensors during export/runtime. The contract describes the vision tower's
    tensor interface, not a concrete model name.
    """

    def __init__(
        self,
        model: nn.Module,
        input_contract: str = VIT_INPUT_CONTRACT_NATIVE,
        max_window_seq_len: int = 0,
        static_grid_thw: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.input_contract = input_contract
        self.max_window_seq_len = max_window_seq_len
        self.uses_static_grid_internal_forward = False
        if static_grid_thw is not None:
            static_grid_thw = static_grid_thw.clone()
            if _supports_static_grid_thw_internal_forward(model):
                with torch.no_grad():
                    self.register_buffer(
                        "static_grid_pos_embeds",
                        model.fast_pos_embed_interpolate(static_grid_thw).detach(),
                    )
                    self.register_buffer(
                        "static_grid_rotary_pos_emb",
                        model.rot_pos_emb(static_grid_thw).detach(),
                    )
                    self.register_buffer(
                        "static_grid_cu_seqlens",
                        _make_static_grid_cu_seqlens(static_grid_thw),
                    )
                self.uses_static_grid_internal_forward = True
                self.static_grid_thw = None
            else:
                self.register_buffer("static_grid_thw", static_grid_thw)
        else:
            self.static_grid_thw = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        window_attention_mask: Optional[torch.Tensor] = None,
        cu_window_seqlens: Optional[torch.Tensor] = None,
        window_index: Optional[torch.Tensor] = None,
        reverse_window_index: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_contract == VIT_INPUT_CONTRACT_WINDOWED_ROPE:
            return _forward_windowed_rope_vision(
                self.model,
                pixel_values,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                window_attention_mask=window_attention_mask,
                cu_window_seqlens=cu_window_seqlens,
                window_index=window_index,
                reverse_window_index=reverse_window_index,
                max_window_seq_len=self.max_window_seq_len,
            )
        if self.input_contract == VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO:
            return _forward_tiled_aspect_ratio_vision(
                self.model,
                pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                attention_mask=attention_mask,
            )
        if self.input_contract == VIT_INPUT_CONTRACT_GRID_THW:
            return _forward_native_vision(
                self.model,
                pixel_values,
                grid_thw=_require_tensor(grid_thw, "grid_thw"),
            )
        if self.input_contract == VIT_INPUT_CONTRACT_STATIC_GRID_THW:
            if self.uses_static_grid_internal_forward:
                return _forward_static_grid_thw_vision(
                    self.model,
                    pixel_values,
                    self.static_grid_pos_embeds,
                    self.static_grid_rotary_pos_emb,
                    self.static_grid_cu_seqlens,
                )
            return _forward_native_vision(
                self.model,
                pixel_values,
                grid_thw=_require_tensor(self.static_grid_thw, "static_grid_thw"),
            )
        if self.input_contract == VIT_INPUT_CONTRACT_NATIVE:
            return _forward_native_vision(self.model, pixel_values)
        raise ValueError(f"Unsupported ViT plugin input contract: {self.input_contract}")



# -----------------------------------------------------------------------------
# VLM Export Shared Helpers
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/tmp/vlm_vision_tensorrt_artifacts"


def dtype_from_name(name: str) -> torch.dtype:
    """Map a short CLI dtype name to the matching torch dtype."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def dtype_from_string(name: str) -> torch.dtype:
    """Map manifest dtype strings back to torch dtypes."""
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
    """Return generic HF AutoModel loaders in the desired fallback order."""
    from transformers import AutoModel

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
    """Import an object from a dotted path such as package.module.Class."""
    module_name, _, object_name = import_path.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(
            f"Expected a dotted import path like package.module.Class, got {import_path!r}."
        )
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def set_attn_implementation_on_config(
    config: Any,
    attn_implementation: str | None = None,
    *,
    public_attn_implementation: str | None = None,
    internal_attn_implementation: str | None = None,
) -> None:
    """Force attention implementation values on a config and nested configs."""
    if attn_implementation is not None:
        public_attn_implementation = (
            attn_implementation
            if public_attn_implementation is None
            else public_attn_implementation
        )
        internal_attn_implementation = (
            attn_implementation
            if internal_attn_implementation is None
            else internal_attn_implementation
        )

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
            if public_attn_implementation is not None:
                setattr(value, "attn_implementation", public_attn_implementation)
            if internal_attn_implementation is not None:
                for attr_name in ("_attn_implementation", "_attn_implementation_internal"):
                    setattr(value, attr_name, internal_attn_implementation)
            children = vars(value).values()
        else:
            return

        for child in children:
            visit(child)

    visit(config)


def load_config_with_attn_implementation(
    model_name: str,
    attn_implementation: str,
    config_loader: Any | None = None,
) -> Any:
    """Load an HF config and apply an attention override before model init."""
    if config_loader is None:
        from transformers import AutoConfig

        config_loader = AutoConfig
    config = config_loader.from_pretrained(model_name, trust_remote_code=True)
    set_attn_implementation_on_config(config, attn_implementation)
    return config


def ensure_tied_weight_keys_compat(model: nn.Module) -> None:
    """Bridge older remote HF models to newer tied-weight loader expectations."""
    if hasattr(model, "all_tied_weights_keys"):
        return

    tied_keys = getattr(model, "_tied_weights_keys", None)
    if tied_keys is None:
        model.all_tied_weights_keys = {}
    elif isinstance(tied_keys, dict):
        model.all_tied_weights_keys = tied_keys
    else:
        model.all_tied_weights_keys = {key: None for key in tied_keys}


def hf_model_supports_attn_implementation(
    model: nn.Module,
    attn_implementation: str,
) -> bool:
    """Return whether a HF model advertises support for an attention backend."""
    if attn_implementation == "eager":
        return True

    support_attr = {
        "sdpa": "_supports_sdpa",
        "flash_attention_2": "_supports_flash_attn_2",
        "flex_attention": "_supports_flex_attn",
    }.get(attn_implementation)
    if support_attr is None:
        return True
    return bool(getattr(model, support_attr, False))


def select_hf_internal_attn_implementation(
    model: nn.Module,
    requested_attn_implementation: str,
    unsupported_fallback: str = "eager",
) -> str:
    """Choose a HF internal attention dispatch for the model being initialized."""
    if hf_model_supports_attn_implementation(model, requested_attn_implementation):
        return requested_attn_implementation
    return unsupported_fallback


def install_legacy_tie_weights_compat(model: nn.Module) -> None:
    """Allow older ``tie_weights(self)`` overrides to run on newer Transformers."""
    if getattr(model, "_vlm_export_legacy_tie_weights_compat", False):
        return

    tie_weights = getattr(model, "tie_weights", None)
    if tie_weights is None:
        return

    def tie_weights_compat(*args, **kwargs):
        try:
            return tie_weights(*args, **kwargs)
        except TypeError as exc:
            if "recompute_mapping" in str(exc):
                return tie_weights()
            raise

    object.__setattr__(model, "tie_weights", tie_weights_compat)
    object.__setattr__(model, "_vlm_export_legacy_tie_weights_compat", True)


@contextmanager
def legacy_tie_weights_compat_during_init():
    """Patch HF post-init so legacy custom tie_weights signatures are accepted."""
    from transformers import PreTrainedModel

    original_post_init = PreTrainedModel.post_init

    def post_init_with_legacy_tie_weights(self, *args, **kwargs):
        install_legacy_tie_weights_compat(self)
        return original_post_init(self, *args, **kwargs)

    PreTrainedModel.post_init = post_init_with_legacy_tie_weights
    try:
        yield
    finally:
        PreTrainedModel.post_init = original_post_init


@contextmanager
def default_torch_tensor_device_for_init(device: str):
    """Give torch.tensor a real default device while custom modules initialize."""
    original_tensor = torch.tensor

    def tensor_with_default_device(*args, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = device
        return original_tensor(*args, **kwargs)

    torch.tensor = tensor_with_default_device
    try:
        yield
    finally:
        torch.tensor = original_tensor


@contextmanager
def torch_save_pickle_protocol(protocol: int):
    """Force torch.save callers in this scope to use a large-object protocol."""
    original_save = torch.save

    def save_with_protocol(*args, **kwargs):
        kwargs.setdefault("pickle_protocol", protocol)
        return original_save(*args, **kwargs)

    torch.save = save_with_protocol
    try:
        yield
    finally:
        torch.save = original_save


@contextmanager
def force_attn_implementation_during_init(
    attn_implementation: str | None,
    unsupported_fallback: str = "eager",
):
    """Temporarily force nested HF model configs during PreTrainedModel init."""
    if attn_implementation is None:
        yield
        return

    from transformers import PreTrainedModel

    original_init = PreTrainedModel.__init__

    def init_with_forced_attn(self, config, *args, **kwargs):
        internal_attn_implementation = select_hf_internal_attn_implementation(
            self,
            attn_implementation,
            unsupported_fallback=unsupported_fallback,
        )
        set_attn_implementation_on_config(
            config,
            public_attn_implementation=attn_implementation,
            internal_attn_implementation=internal_attn_implementation,
        )
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
    """Temporarily set torch default device while HF constructs modules."""
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
    """Patch dataclass imports for packages with Python 3.12 field-order issues."""
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
    """Add common .language_model and .visual aliases when a package expects them."""
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
    """Install a process-local HF from_pretrained hook that adds VLM aliases."""
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
    """Load a VLM/policy model through HF AutoModel or a provided class path."""
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if model_class is not None:
        if dataclass_kw_only_imports:
            enable_dataclass_kw_only_import_compat(
                (model_class.split(".", maxsplit=1)[0],)
            )
        if use_common_vlm_aliases:
            enable_common_vlm_alias_hook()
        loader = import_object(model_class)
        custom_model_kwargs = dict(model_kwargs)
        if attn_implementation is not None:
            custom_model_kwargs["config"] = load_config_with_attn_implementation(
                model_name,
                attn_implementation,
                getattr(loader, "config_class", None),
            )
        with torch.no_grad():
            with default_device_for_loading("cpu"):
                with default_torch_tensor_device_for_init("cpu"):
                    with legacy_tie_weights_compat_during_init():
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
                                model = loader.from_pretrained(
                                    model_name,
                                    **custom_model_kwargs,
                                ).eval()
            if use_common_vlm_aliases:
                model = add_common_vlm_aliases(model)
            return model.to(device) if move_to_device else model

    last_error: Exception | None = None
    auto_model_kwargs = dict(model_kwargs)
    if attn_implementation is not None:
        auto_model_kwargs["config"] = load_config_with_attn_implementation(
            model_name,
            attn_implementation,
        )
    with torch.no_grad():
        if use_common_vlm_aliases:
            enable_common_vlm_alias_hook()
        for loader in model_loader_candidates(prefer_generation_model):
            try:
                with default_device_for_loading("cpu"):
                    with default_torch_tensor_device_for_init("cpu"):
                        with legacy_tie_weights_compat_during_init():
                            with force_attn_implementation_during_init(attn_implementation):
                                model = loader.from_pretrained(
                                    model_name,
                                    **auto_model_kwargs,
                                ).eval()
                if use_common_vlm_aliases:
                    model = add_common_vlm_aliases(model)
                return model.to(device) if move_to_device else model
            except (KeyError, ValueError, AttributeError) as exc:
                last_error = exc
    raise ValueError(f"Could not load {model_name!r} with available AutoModel classes.") from last_error


def get_nested_module(model: nn.Module, module_path: str) -> nn.Module:
    """Resolve a dotted module path inside a loaded model."""
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
    """Return the selected vision tower, either explicit or auto-detected."""
    if module_path is not None:
        return get_nested_module(model, module_path)
    from utils.utils import get_vision_model as get_generic_vision_model

    return get_generic_vision_model(model)


def normalize_conv2d_valid_padding(module: nn.Module) -> None:
    """Rewrite PyTorch string padding that older Torch-TensorRT converters reject."""
    for child in module.modules():
        if isinstance(child, nn.Conv2d) and child.padding == "valid":
            child.padding = (0, 0)


def resolve_processor_candidates(model: nn.Module, model_name: str, processor_model: str | None) -> List[str]:
    """Return processor model ids to try, ordered from most to least explicit."""
    candidates: List[str] = []
    for candidate in (
        processor_model,
        getattr(getattr(model, "config", None), "vlm_name_or_path", None),
        getattr(getattr(model, "config", None), "processor_name_or_path", None),
        model_name,
    ):
        if isinstance(candidate, str) and candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def load_processor_for_export(
    model: nn.Module,
    model_name: str,
    processor_model: str | None,
) -> Any:
    """Load a processor for export, falling back to processors built by wrapper models."""
    from transformers import AutoProcessor

    errors: List[str] = []
    for candidate in resolve_processor_candidates(model, model_name, processor_model):
        try:
            return AutoProcessor.from_pretrained(
                candidate,
                trust_remote_code=True,
                use_fast=True,
            )
        except (OSError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")

    processor = getattr(model, "processor", None)
    if processor is not None:
        print("Using processor attached to loaded model.")
        return processor

    joined_errors = "\n".join(errors)
    raise ValueError(
        "Could not load an AutoProcessor and the model has no attached processor. "
        f"Tried:\n{joined_errors}"
    )


def _extract_tensor(output: Any) -> torch.Tensor:
    """Normalize common model output containers to a single tensor."""
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor-like model output, got {type(output)!r}")
    return output


def _get_config_attr(config: Any, names: Tuple[str, ...]) -> Any:
    """Return the first present config attribute from a list of aliases."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _infer_patch_count(vision_config: Any, pixel_values: torch.Tensor) -> int:
    """Infer the vision token count used to configure the ViT attention plugin."""
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
    """Populate global ViT plugin metadata from the selected vision tower config."""
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
    """Precompute Qwen-style window/RoPE tensors outside torch.export tracing."""
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
    """Detect vision towers that need grid-aware RoPE and window metadata."""
    return (
        isinstance(inputs.get("image_grid_thw"), torch.Tensor)
        and hasattr(visual, "get_window_index")
        and hasattr(visual, "rot_pos_emb")
        and hasattr(visual, "patch_embed")
        and hasattr(visual, "blocks")
    )


def has_tiled_aspect_ratio_contract(inputs: Dict[str, torch.Tensor]) -> bool:
    """Detect Llama-style tiled image inputs with aspect-ratio metadata."""
    return isinstance(inputs.get("aspect_ratio_ids"), torch.Tensor) and (
        isinstance(inputs.get("aspect_ratio_mask"), torch.Tensor)
        or isinstance(inputs.get("attention_mask"), torch.Tensor)
    )


def has_grid_thw_contract(visual: nn.Module, inputs: Dict[str, torch.Tensor]) -> bool:
    """Detect vision towers whose forward method accepts generic grid_thw."""
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
    """Choose the wrapper input contract and prepare its non-pixel tensors."""
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
    """Run the original vision tower once to capture a PyTorch reference output."""
    from utils.utils import extract_vision_tensor

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


def make_synthetic_image(size: int) -> Any:
    """Create a deterministic RGB image for processor-based export samples."""
    from PIL import Image

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
    """Use an HF processor to create model-ready tensors and input metadata."""
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
    """Create pixel_values directly when a model should bypass AutoProcessor."""
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
    """Legacy helper returning pixel values and image_grid_thw from a processor."""
    inputs, metadata = prepare_processor_inputs(
        processor, prompt, image_size, device, dtype
    )
    return inputs["pixel_values"], inputs["image_grid_thw"], metadata


def export_vision(
    wrapper: nn.Module,
    pixel_values: torch.Tensor,
    core_inputs: Dict[str, torch.Tensor],
) -> torch.export.ExportedProgram:
    """Export the wrapped vision tower with a fixed tensor contract."""
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
    """Optionally lower an exported program to an ExecuTorch artifact."""
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
    """Save replay inputs and reference output for engine validation."""
    tensors = {
        "reference": reference.detach().cpu(),
    }
    tensors.update({name: value.detach().cpu() for name, value in sample_inputs.items()})
    torch.save(tensors, output_path)


def compile_inputs_from_tensors(
    example_inputs: Tuple[torch.Tensor, ...],
    example_kwargs: Dict[str, torch.Tensor],
) -> List[Any]:
    """Create Torch-TensorRT input specs from live example tensors."""
    import torch_tensorrt

    return [
        torch_tensorrt.Input(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        for tensor in list(example_inputs) + list(example_kwargs.values())
    ]


def _safe_artifact_name(name: str) -> str:
    """Convert model/module names to filesystem-safe artifact names."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("_"))
    return sanitized or "tensorrt_engine"


def safe_model_tag(model_name: str) -> str:
    """Create a stable artifact prefix from an HF model id or local path."""
    return _safe_artifact_name(model_name.rsplit("/", 1)[-1].lower())


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    """Write the export manifest with stable formatting."""
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def tensor_specs(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """Summarize tensor shapes and dtypes for the manifest."""
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tensors.items()
    }

# -----------------------------------------------------------------------------
# Model Modification Functions
# -----------------------------------------------------------------------------


def replace_vit_attention_with_plugin(
    model: nn.Module,
    config: Any,
    use_plugin_op: bool = True,
) -> nn.Module:
    """
    Replace all supported vision attention modules with plugin attention.

    This is the vision-side equivalent of the LLM helper: callers use one
    replacement entry point, and the function detects the model structure:
    - block-based vision stacks: ``blocks[*].attn``
    - transformer/global-transformer stacks: ``*.layers[*].self_attn``
    - HF ViT-style encoders: ``encoder.layer[*].attention.self``

    Args:
        model: The HuggingFace vision model or visual tower to modify.
        config: Model configuration.

    Returns:
        The modified model with plugin attention.
    """
    replacement_count = 0

    # Block-based visual tower: model.visual.blocks or visual.blocks.
    visual_model = model.visual if hasattr(model, "visual") else model
    if hasattr(visual_model, "blocks"):
        for i, block in enumerate(visual_model.blocks):
            if hasattr(block, "attn"):
                block.attn = ViTPluginAttention(
                    block.attn, config, i, use_plugin_op=use_plugin_op
                )
                replacement_count += 1
        if replacement_count:
            return model

    # Some self-attention modules return (hidden_state, attn_weights), so
    # these replacements ask the generic plugin wrapper to return a tuple.
    vision_model = model.vision_model if hasattr(model, "vision_model") else model
    layer_idx = 0
    for encoder_name in ("transformer", "global_transformer"):
        encoder = getattr(vision_model, encoder_name, None)
        if encoder is None or not hasattr(encoder, "layers"):
            continue

        for layer in encoder.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn = ViTPluginAttention(
                    layer.self_attn,
                    config,
                    layer_idx,
                    return_tuple=True,
                    use_plugin_op=use_plugin_op,
                )
                layer_idx += 1
                replacement_count += 1

    if layer_idx:
        return model

    # HF SigLIP/SigLIP2-style tower: model.vision_model.encoder.layers or
    # model.encoder.layers. These attention modules return
    # (hidden_state, attn_weights), so the replacement returns a tuple.
    if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
        for i, layer in enumerate(vision_model.encoder.layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn = ViTPluginAttention(
                    layer.self_attn,
                    config,
                    i,
                    return_tuple=True,
                    use_plugin_op=use_plugin_op,
                )
                replacement_count += 1

    if replacement_count:
        return model

    # HF ViT-style tower: model.vision_model.encoder.layer or model.encoder.layer.
    if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layer"):
        for i, layer in enumerate(vision_model.encoder.layer):
            if hasattr(layer, "attention"):
                layer.attention.self = ViTPluginAttention(
                    layer.attention.self, config, i, use_plugin_op=use_plugin_op
                )
                replacement_count += 1

    if replacement_count == 0:
        raise ValueError("Cannot find supported ViT attention modules")

    return model


def count_vit_plugin_attention_modules(model: nn.Module) -> int:
    """Count ViT attention modules replaced with the plugin wrapper."""
    return sum(1 for module in model.modules() if isinstance(module, ViTPluginAttention))

# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------

def compile_vit_plugin_model(
    model: nn.Module,
    example_inputs: Optional[Tuple[torch.Tensor, ...]],
    device: torch.device,
    example_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Callable:
    """
    Compile a ViT/VLM visual wrapper with plugin attention.

    Model-specific wrappers own input preparation and forward signatures. This
    helper owns the shared torch.export -> Torch-TensorRT compile path.

    Args:
        model: The vision wrapper or model to export.
        example_inputs: Example tensor inputs matching ``model.forward``.
        example_kwargs: Optional named tensor inputs matching ``model.forward``.
        dynamic_shapes: Optional torch.export dynamic shape spec.
        device: Device for compilation.
        debug: Whether to enable debug logging.

    Returns:
        Compiled TensorRT model function.
    """
    if dynamic_shapes is None:
        dynamic_shapes = {}
    if example_inputs is None:
        example_inputs = ()
    if example_kwargs is None:
        example_kwargs = {}
    if dynamic_shapes:
        dynamic_shapes = {
            name: shape
            for name, shape in dynamic_shapes.items()
            if isinstance(example_kwargs.get(name), torch.Tensor)
        }

    ep = torch.export.export(
        model,
        args=example_inputs,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    compile_inputs = list(example_inputs) + [
        value for value in example_kwargs.values() if isinstance(value, torch.Tensor)
    ]
    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=compile_inputs,
            use_explicit_typing=True,
            use_fp32_acc=True,
            device=device,
            disable_tf32=True,
            min_block_size=1,
        )

    return trt_model


# -----------------------------------------------------------------------------
# Inference Utilities
# -----------------------------------------------------------------------------


def inference_vit_plugin(
    model_func: Callable,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """
    Run inference on a compiled ViT plugin model.

    Args:
        model_func: The compiled TensorRT model function.
        pixel_values: Input images [batch, channels, height, width].

    Returns:
        Model output (logits or embeddings depending on model).
    """
    return model_func(pixel_values)


# Benchmark utilities

def measure_vit_latency(
    fn: Callable,
    num_warmup: int = 5,
    num_runs: int = 10,
) -> Tuple[float, float, float]:
    """
    Measure function latency with GPU synchronization.

    Args:
        fn: Function to benchmark.
        num_warmup: Number of warmup runs.
        num_runs: Number of timing runs.

    Returns:
        Tuple of (mean_latency_ms, std_latency_ms, median_latency_ms).
    """
    import statistics

    # Warmup
    for _ in range(num_warmup):
        fn()

    torch.cuda.synchronize()
    times = []

    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))

    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time = statistics.median(times)

    return mean_time, stdev_time, median_time


def measure_vit_memory(
    model: nn.Module,
    pixel_values: torch.Tensor,
) -> Tuple[float, float]:
    """
    Measure model memory usage.

    Args:
        model: The model.
        pixel_values: Sample input.

    Returns:
        Tuple of (peak_memory_mb, reserved_memory_mb).
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(pixel_values)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    reserved_memory = torch.cuda.memory_reserved() / 1e6

    return peak_memory, reserved_memory


# Importing this module registers the Torch-TensorRT converter for
# tensorrt_vit::attention, matching the LLM plugin_utils/plugin_converter split.
from .plugin_converter_vit import convert_vit_attention  # noqa: F401,E402

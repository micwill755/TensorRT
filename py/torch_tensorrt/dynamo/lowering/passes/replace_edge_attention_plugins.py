import logging
from typing import Any, Optional

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def _node_target_name(node: torch.fx.Node) -> str:
    return str(getattr(node, "target", ""))


def _is_existing_edge_plugin_call(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target_name = _node_target_name(node)
    return (
        "torch.ops.trt.attention_plugin" in target_name
        or "torch.ops.trt.vit_attention_plugin" in target_name
        or "torch.ops.trt.vit_masked_attention_plugin" in target_name
        or "trt.attention_plugin" in target_name
        or "trt.vit_attention_plugin" in target_name
        or "trt.vit_masked_attention_plugin" in target_name
    )


def _is_sdpa_call(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    sdpa_targets = {
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.nn.functional.scaled_dot_product_attention,
    }
    c_nn_sdpa = getattr(torch._C._nn, "scaled_dot_product_attention", None)
    if c_nn_sdpa is not None:
        sdpa_targets.add(c_nn_sdpa)
    return node.target in sdpa_targets


def _arg_or_kw(
    node: torch.fx.Node, index: int, name: str, default: Any = None
) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _meta_value(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return value.meta.get("val") or value.meta.get("tensor_meta")
    return value


def _shape(value: Any) -> Optional[tuple[Any, ...]]:
    meta = _meta_value(value)
    shape = getattr(meta, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(shape)
    except TypeError:
        return None


def _dtype(value: Any) -> Optional[torch.dtype]:
    return getattr(_meta_value(value), "dtype", None)


def _device(value: Any) -> Optional[torch.device]:
    return getattr(_meta_value(value), "device", None)


def _static_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted


def _is_false(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, float)):
        return not bool(value)
    return False


def _is_supported_sdpa_for_vit_plugin(node: torch.fx.Node) -> bool:
    query = _arg_or_kw(node, 0, "query")
    key = _arg_or_kw(node, 1, "key")
    value = _arg_or_kw(node, 2, "value")
    attn_mask = _arg_or_kw(node, 3, "attn_mask")
    dropout_p = _arg_or_kw(node, 4, "dropout_p", 0.0)
    is_causal = _arg_or_kw(node, 5, "is_causal", False)
    scale = node.kwargs.get("scale")
    enable_gqa = node.kwargs.get("enable_gqa", False)

    if not _is_false(dropout_p) or not _is_false(is_causal):
        return False
    if scale is not None or not _is_false(enable_gqa):
        return False

    query_shape = _shape(query)
    key_shape = _shape(key)
    value_shape = _shape(value)
    if query_shape is None or key_shape is None or value_shape is None:
        return False
    if len(query_shape) != 4 or key_shape != query_shape or value_shape != query_shape:
        return False

    batch_size, num_heads, seq_len, head_dim = (_static_int(dim) for dim in query_shape)
    if None in (batch_size, num_heads, seq_len, head_dim):
        return False
    if min(batch_size, num_heads, seq_len, head_dim) <= 0:
        return False

    if _dtype(query) != torch.float16:
        return False

    if attn_mask is not None:
        mask_shape = _shape(attn_mask)
        mask_dtype = _dtype(attn_mask)
        if mask_shape is None or mask_dtype in (torch.bool, torch.int32, torch.int64):
            return False
        if len(mask_shape) == 3:
            return tuple(_static_int(dim) for dim in mask_shape[-2:]) == (seq_len, seq_len)
        if len(mask_shape) == 4 and _static_int(mask_shape[1]) == 1:
            return tuple(_static_int(dim) for dim in mask_shape[-2:]) == (seq_len, seq_len)
        return False

    return True


def _call_aten(
    graph: torch.fx.Graph, target: Any, args: tuple[Any, ...], kwargs: Optional[dict[str, Any]] = None
) -> torch.fx.Node:
    return graph.call_function(target, args=args, kwargs=kwargs or {})


def _rewrite_sdpa_to_vit_masked_plugin(node: torch.fx.Node) -> bool:
    if not _is_supported_sdpa_for_vit_plugin(node):
        return False

    try:
        from torch_tensorrt.dynamo.conversion import edge_plugins  # noqa: F401
    except Exception as exc:
        logger.debug("Skipping Edge ViT plugin rewrite; plugin ops unavailable: %s", exc)
        return False

    plugin_target = getattr(torch.ops.trt, "vit_masked_attention_plugin", None)
    if plugin_target is None:
        return False

    query = _arg_or_kw(node, 0, "query")
    key = _arg_or_kw(node, 1, "key")
    value = _arg_or_kw(node, 2, "value")
    attn_mask = _arg_or_kw(node, 3, "attn_mask")
    query_shape = _shape(query)
    if query_shape is None:
        return False

    batch_size, num_heads, seq_len, head_dim = [int(dim) for dim in query_shape]
    hidden_size = num_heads * head_dim
    dtype = _dtype(query) or torch.float16
    device = _device(query)
    graph = node.graph

    with graph.inserting_before(node):
        q_bshd = _call_aten(graph, torch.ops.aten.permute.default, (query, [0, 2, 1, 3]))
        k_bshd = _call_aten(graph, torch.ops.aten.permute.default, (key, [0, 2, 1, 3]))
        v_bshd = _call_aten(graph, torch.ops.aten.permute.default, (value, [0, 2, 1, 3]))
        q_bsh = _call_aten(graph, torch.ops.aten.reshape.default, (q_bshd, [batch_size, seq_len, hidden_size]))
        k_bsh = _call_aten(graph, torch.ops.aten.reshape.default, (k_bshd, [batch_size, seq_len, hidden_size]))
        v_bsh = _call_aten(graph, torch.ops.aten.reshape.default, (v_bshd, [batch_size, seq_len, hidden_size]))
        qkv = _call_aten(graph, torch.ops.aten.cat.default, ([q_bsh, k_bsh, v_bsh], -1))

        tensor_kwargs = {"dtype": dtype}
        if device is not None:
            tensor_kwargs["device"] = device
        cos = _call_aten(graph, torch.ops.aten.ones.default, ([seq_len, head_dim],), tensor_kwargs)
        sin = _call_aten(graph, torch.ops.aten.zeros_like.default, (cos,))

        if attn_mask is None:
            mask = _call_aten(
                graph,
                torch.ops.aten.zeros.default,
                ([batch_size, seq_len, seq_len],),
                tensor_kwargs,
            )
        else:
            mask_shape = _shape(attn_mask) or ()
            if len(mask_shape) == 4:
                mask = _call_aten(graph, torch.ops.aten.squeeze.dim, (attn_mask, 1))
            else:
                mask = attn_mask

        plugin_out = graph.call_function(
            torch.ops.trt.vit_masked_attention_plugin.default,
            args=(qkv, cos, sin, mask, num_heads, head_dim, 1, 0, 0, 0),
            kwargs={},
        )
        plugin_bshd = _call_aten(
            graph,
            torch.ops.aten.reshape.default,
            (plugin_out, [batch_size, seq_len, num_heads, head_dim]),
        )
        replacement = _call_aten(
            graph, torch.ops.aten.permute.default, (plugin_bshd, [0, 2, 1, 3])
        )

    replacement.meta.update(node.meta)
    node.replace_all_uses_with(replacement)
    graph.erase_node(node)
    logger.debug("Rewrote SDPA node %s to trt.vit_masked_attention_plugin", node.name)
    return True


def replace_edge_attention_plugins(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rewrite supported Edge-LLM attention patterns to plugin ops.

    This pass is the Torch-TensorRT-owned landing zone for Naren-style export:
    the model is captured first with ``torch.export``, then Dynamo lowering owns
    attention/plugin graph transformation before TRTInterpreter runs.

    The first concrete rewrite handles a safe ViT-style SDPA form:
    ``scaled_dot_product_attention(q, k, v)`` with static ``[B,H,S,D]`` FP16
    tensors, no dropout, no causal masking, no GQA, and either no mask or a
    dense additive mask. That pattern is lowered to
    ``torch.ops.trt.vit_masked_attention_plugin`` and reshaped back to the SDPA
    output layout so downstream graph consumers do not change.
    """
    del settings
    changed = False

    plugin_nodes = [node for node in gm.graph.nodes if _is_existing_edge_plugin_call(node)]
    if plugin_nodes:
        logger.debug(
            "Edge attention plugin lowering pass found existing plugin op(s): %s",
            ", ".join(_node_target_name(node) for node in plugin_nodes),
        )

    for node in list(gm.graph.nodes):
        if _is_sdpa_call(node):
            changed = _rewrite_sdpa_to_vit_masked_plugin(node) or changed

    if changed or plugin_nodes:
        clean_up_graph_after_modifications(gm)
        gm.recompile()
    return gm

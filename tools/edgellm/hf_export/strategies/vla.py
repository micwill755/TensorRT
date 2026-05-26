# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic VLA Hugging Face strategy for Edge-LLM role export.

This mirrors the run_hf.py family-strategy idea: VLA is the strategy, while
runtime mechanics are discovered structurally as Edge contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

_ACTION_WRAPPER_SYMBOLS = None


def _action_wrapper_symbols():
    """Lazy-load Edge-LLM action wrapper classes.

    Importing the full Edge-LLM model package can pull in heavy
    optional dependencies. Delaying this import lets ``--help`` and
    family detection work even on systems that are not ready to run
    action export.
    """
    global _ACTION_WRAPPER_SYMBOLS
    if _ACTION_WRAPPER_SYMBOLS is None:
        from tensorrt_edgellm.action_models.gr00t_model import (
            GR00TStateFlowActionStep,
            _use_export_friendly_attention_processors,
        )
        from tensorrt_edgellm.action_models.pi05_model import PI05PrefixKVActionStep

        _ACTION_WRAPPER_SYMBOLS = (
            PI05PrefixKVActionStep,
            GR00TStateFlowActionStep,
            _use_export_friendly_attention_processors,
        )
    return _ACTION_WRAPPER_SYMBOLS


from tools.edgellm.contracts.action_contracts import (
    ACTION_CONTRACT_PREFIX_KV_FLOW_STEP,
    ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP,
    write_action_contract_manifest,
)
from tools.edgellm.eager_export.loader import LoadedEagerModel, import_object

from .base import EdgeHFStrategy, HFExportConfig


def _torch_dtype(dtype: Optional[str]) -> torch.dtype:
    """Translate CLI dtype strings into ``torch.dtype`` objects."""
    if dtype in (None, "fp16", "float16", "half"):
        return torch.float16
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported VLA export dtype: {dtype}")


def _filter_supported_kwargs(fn: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments that a loader does not accept.

    Custom Hugging Face policy classes are not perfectly consistent
    about their ``from_pretrained`` signatures. Filtering keeps the
    generic exporter tolerant without hard-coding each class.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in params}


def _set_attn_implementation(config: Any, attn_implementation: Optional[str]) -> None:
    """Best-effort set attention implementation on nested configs.

    Many HF configs nest language and vision configs. We recursively
    set both public and private attention fields where they exist,
    and ignore objects that do not expose those fields.
    """
    if config is None or attn_implementation is None:
        return
    for attr_name in ("attn_implementation", "_attn_implementation"):
        if hasattr(config, attr_name):
            try:
                setattr(config, attr_name, attn_implementation)
            except Exception:
                pass
    for child_name in ("text_config", "llm_config", "language_config", "vision_config"):
        child = getattr(config, child_name, None)
        if child is not None and child is not config:
            _set_attn_implementation(child, attn_implementation)


def _load_with_class(
    model_dir: str,
    *,
    model_class: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: Optional[str],
    extra_kwargs: Mapping[str, Any],
) -> nn.Module:
    """Load a custom VLA class from ``--model_class``.

    This is the path used for LeRobot/OpenPI-style policies whose
    root config is not a standard Transformers model type.
    """
    loader = import_object(model_class)
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **dict(extra_kwargs),
    }
    config = load_kwargs.get("config")
    if config is None and attn_implementation is not None:
        config_class = getattr(loader, "config_class", None)
        if config_class is not None:
            try:
                config = config_class.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
                _set_attn_implementation(config, attn_implementation)
                load_kwargs["config"] = config
            except Exception as exc:
                print(
                    "Could not preload custom VLA config to set "
                    f"attn_implementation={attn_implementation!r}: {exc}."
                )

    from_pretrained = getattr(loader, "from_pretrained")
    try:
        model = from_pretrained(model_dir, **load_kwargs)
    except TypeError:
        model = from_pretrained(model_dir, **_filter_supported_kwargs(from_pretrained, load_kwargs))
    if config is None:
        _set_attn_implementation(getattr(model, "config", None), attn_implementation)
        for module in model.modules():
            _set_attn_implementation(getattr(module, "config", None), attn_implementation)
    return model.eval()


def _load_auto_vla(
    model_dir: str,
    *,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[nn.Module, Any, Any]:
    """Try generic HF Auto classes for VLA-like repos.

    This is a convenience path for models that are already compatible
    with Transformers auto loading. If it fails, users should provide
    ``--model_class`` so the exporter can call the repo-specific class.
    """
    processor = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    except Exception:
        processor = None

    errors: list[str] = []
    for class_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoModel"):
        try:
            import transformers

            loader = getattr(transformers, class_name)
            model = loader.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            ).eval()
            return model, getattr(processor, "tokenizer", None), processor
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    detail = "\n".join(errors[-3:])
    raise ValueError(
        "Could not load VLA model with generic HF auto classes. Pass --model_class."
        + (f"\nRecent errors:\n{detail}" if detail else "")
    )


def _maybe_to_device(model: nn.Module, *, device: str, torch_dtype: torch.dtype) -> nn.Module:
    """Move a model to the requested device and dtype.

    Some modules accept ``to(device=..., dtype=...)`` while others
    only accept a device first. This helper handles both styles.
    """
    try:
        return model.to(device=device, dtype=torch_dtype)
    except TypeError:
        model = model.to(device)
        return model.to(dtype=torch_dtype)


def _supports_prefix_kv_flow_step(model: nn.Module) -> bool:
    """Detect PI0.5/OpenPI-style prefix-KV action structure.

    This is structural detection: we look for capabilities the wrapper
    needs, not for a particular model id.
    """
    core = getattr(model, "model", model)
    return all(
        hasattr(core, attr)
        for attr in ("embed_suffix", "paligemma_with_expert", "action_out_proj")
    )


def _supports_state_conditioned_flow_step(model: nn.Module) -> bool:
    """Detect GR00T-style state-conditioned action structure."""
    action_head = getattr(model, "action_head", None)
    return action_head is not None and all(
        hasattr(action_head, attr)
        for attr in (
            "action_encoder",
            "action_decoder",
            "state_encoder",
            "model",
            "vlln",
            "vl_self_attention",
        )
    )


def _wrap_action_role(model: nn.Module) -> tuple[str, nn.Module]:
    """Choose the action contract and wrapper for a loaded VLA model.

    The loaded root model is usually too large and irregular to export
    directly. The wrapper exposes one clean forward signature matching
    a runtime contract.
    """
    (
        PI05PrefixKVActionStep,
        GR00TStateFlowActionStep,
        _use_export_friendly_attention_processors,
    ) = _action_wrapper_symbols()
    if _supports_prefix_kv_flow_step(model):
        return ACTION_CONTRACT_PREFIX_KV_FLOW_STEP, PI05PrefixKVActionStep(model).eval()
    if _supports_state_conditioned_flow_step(model):
        _use_export_friendly_attention_processors(model.action_head)
        return ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP, GR00TStateFlowActionStep(model).eval()
    raise ValueError(
        "Could not discover a supported VLA action contract from model structure. "
        "Known structural contracts: prefix_kv_flow_step and state_conditioned_flow_step."
    )


def load_vla_model(
    model_dir: str,
    *,
    model_class: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = "eager",
    trust_remote_code: bool = True,
    **from_pretrained_kwargs: Any,
) -> LoadedEagerModel:
    """Load a VLA eager model and expose structural Edge roles.

    The returned ``LoadedEagerModel`` uses a mapping as its model:
    ``root`` keeps the original HF policy and ``action`` is the small
    wrapper module that the action role will export.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _torch_dtype(dtype)
    tokenizer = None
    processor = None

    if model_class:
        model = _load_with_class(
            model_dir,
            model_class=model_class,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            extra_kwargs=from_pretrained_kwargs,
        )
    else:
        model, tokenizer, processor = _load_auto_vla(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    model = _maybe_to_device(model, device=device, torch_dtype=torch_dtype).eval()
    action_contract, action = _wrap_action_role(model)
    action = action.to(device).eval()

    core = getattr(model, "model", model)
    tokenizer = tokenizer or getattr(model, "tokenizer", None) or getattr(core, "tokenizer", None)
    processor = processor or getattr(model, "processor", None) or getattr(core, "processor", None)
    return LoadedEagerModel(
        model={"root": model, "action": action},
        tokenizer=tokenizer,
        processor=processor,
        extra={
            "model_dir": model_dir,
            "model_class": model_class,
            "action_contract": action_contract,
        },
    )


def _prefix_kv_action_inputs(
    module: PI05PrefixKVActionStep,
    *,
    device: str,
    torch_dtype: torch.dtype,
    batch_size: int,
    max_kv_cache_capacity: int,
    prefix_len: Optional[int],
) -> dict[str, Any]:
    """Build example tensors for a prefix-KV action flow step.

    These tensors describe the graph signature. They are not used as
    real robot actions; they simply tell Torch export the batch size,
    action dimensions, prefix KV cache layout, and dtype.
    """
    cfg = module.core.config
    expert_cfg = module.core.paligemma_with_expert.gemma_expert.model.config
    prefix_len = int(prefix_len or min(int(max_kv_cache_capacity), 256))
    num_layers = int(expert_cfg.num_hidden_layers)
    num_kv_heads = int(expert_cfg.num_key_value_heads)
    head_dim = int(getattr(expert_cfg, "head_dim", expert_cfg.hidden_size // expert_cfg.num_attention_heads))

    noisy_actions = torch.randn(
        int(batch_size), int(cfg.chunk_size), int(cfg.max_action_dim), device=device, dtype=torch.float32
    )
    time_steps_t0 = torch.tensor([1.0], device=device, dtype=torch.float32)
    time_steps_t1 = torch.tensor([1.0 - 1.0 / int(cfg.num_inference_steps)], device=device, dtype=torch.float32)
    prefix_pad_mask = torch.ones(int(batch_size), prefix_len, device=device, dtype=torch.bool)
    prefix_k = torch.zeros(
        num_layers, int(batch_size), num_kv_heads, prefix_len, head_dim, device=device, dtype=torch_dtype
    )
    prefix_v = torch.zeros_like(prefix_k)
    return {
        "args": (noisy_actions, time_steps_t0, time_steps_t1, prefix_pad_mask, prefix_k, prefix_v),
        "input_names": [
            "noisy_actions",
            "time_steps_t0",
            "time_steps_t1",
            "prefix_pad_mask",
            "prefix_k",
            "prefix_v",
        ],
        "output_names": ["denoised_actions"],
        "dynamic_axes": {
            "noisy_actions": {0: "batch_size"},
            "prefix_pad_mask": {0: "batch_size", 1: "kv_cache_len"},
            "prefix_k": {1: "batch_size", 3: "kv_cache_len"},
            "prefix_v": {1: "batch_size", 3: "kv_cache_len"},
            "denoised_actions": {0: "batch_size"},
        },
    }


def _state_conditioned_action_inputs(
    module: GR00TStateFlowActionStep,
    *,
    device: str,
    torch_dtype: torch.dtype,
    batch_size: int,
    backbone_seq_len: Optional[int],
) -> dict[str, Any]:
    """Build example tensors for a state-conditioned action flow step.

    The GR00T-style action engine consumes noisy actions plus VLM
    backbone features, attention masks, robot state, embodiment id,
    and image mask.
    """
    cfg = module.config
    action_horizon = int(cfg.action_horizon)
    action_dim = int(cfg.max_action_dim)
    state_history = int(getattr(cfg, "state_history_length", 1))
    state_dim = int(cfg.max_state_dim)
    seq_len = int(backbone_seq_len or min(int(getattr(cfg, "max_seq_len", 1024)), 1024))
    backbone_dim = int(cfg.backbone_embedding_dim)

    actions = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=torch_dtype)
    timestep = torch.zeros(batch_size, device=device, dtype=torch.int64)
    backbone_features = torch.randn(batch_size, seq_len, backbone_dim, device=device, dtype=torch_dtype)
    backbone_attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    state = torch.randn(batch_size, state_history, state_dim, device=device, dtype=torch_dtype)
    embodiment_id = torch.zeros(batch_size, device=device, dtype=torch.int64)
    image_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    return {
        "args": (actions, timestep, backbone_features, backbone_attention_mask, state, embodiment_id, image_mask),
        "input_names": [
            "actions",
            "timestep",
            "backbone_features",
            "backbone_attention_mask",
            "state",
            "embodiment_id",
            "image_mask",
        ],
        "output_names": ["action_velocity"],
        "dynamic_axes": {
            "actions": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "backbone_features": {0: "batch_size", 1: "backbone_seq_len"},
            "backbone_attention_mask": {0: "batch_size", 1: "backbone_seq_len"},
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "image_mask": {0: "batch_size", 1: "backbone_seq_len"},
            "action_velocity": {0: "batch_size"},
        },
    }


def vla_action_inputs(
    module: nn.Module,
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    batch_size: int = 2,
    max_kv_cache_capacity: int = 4096,
    prefix_len: Optional[int] = None,
    backbone_seq_len: Optional[int] = None,
    **_: Any,
) -> dict[str, Any]:
    """Return example inputs for the discovered VLA action contract.

    ``EdgeExport`` calls this hook after resolving the ``action`` role.
    The hook dispatches to the right example builder based on the
    wrapper module type.
    """
    PI05PrefixKVActionStep, GR00TStateFlowActionStep, _ = _action_wrapper_symbols()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _torch_dtype(dtype)
    if isinstance(module, PI05PrefixKVActionStep) or hasattr(module, "core"):
        return _prefix_kv_action_inputs(
            module,
            device=device,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            max_kv_cache_capacity=max_kv_cache_capacity,
            prefix_len=prefix_len,
        )
    if isinstance(module, GR00TStateFlowActionStep) or hasattr(module, "action_head"):
        return _state_conditioned_action_inputs(
            module,
            device=device,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            backbone_seq_len=backbone_seq_len,
        )
    raise ValueError(f"Unsupported VLA action module type: {type(module).__name__}")


def _prefix_runtime_metadata(
    module: PI05PrefixKVActionStep,
    *,
    max_kv_cache_capacity: int,
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    """Collect runtime metadata for prefix-KV diffusion execution.

    The C++ action runner needs values such as denoise steps, action
    chunk size, head layout, and max KV capacity. The engine alone
    does not reliably encode all of that semantic information.
    """
    cfg = module.core.config
    expert_cfg = module.core.paligemma_with_expert.gemma_expert.model.config
    head_dim = int(getattr(expert_cfg, "head_dim", expert_cfg.hidden_size // expert_cfg.num_attention_heads))
    return {
        "wrapper": type(module).__name__,
        "execution_mode": "prefix_kv_flow_denoise",
        "runtime": {
            "denoise_steps": int(cfg.num_inference_steps),
            "chunk_size": int(cfg.chunk_size),
            "max_action_dim": int(cfg.max_action_dim),
            "num_layers": int(expert_cfg.num_hidden_layers),
            "num_kv_heads": int(expert_cfg.num_key_value_heads),
            "head_dim": head_dim,
            "max_kv_cache_capacity": int(max_kv_cache_capacity),
            "torch_dtype": str(torch_dtype).replace("torch.", ""),
        },
    }


def _state_runtime_metadata(module: GR00TStateFlowActionStep, *, torch_dtype: torch.dtype) -> dict[str, Any]:
    """Collect runtime metadata for state-conditioned diffusion."""
    cfg = module.config
    return {
        "wrapper": type(module).__name__,
        "execution_mode": "state_conditioned_flow_denoise",
        "runtime": {
            "denoise_steps": int(cfg.num_inference_timesteps),
            "num_timestep_buckets": int(cfg.num_timestep_buckets),
            "action_horizon": int(cfg.action_horizon),
            "max_action_dim": int(cfg.max_action_dim),
            "max_state_dim": int(cfg.max_state_dim),
            "state_history_length": int(getattr(cfg, "state_history_length", 1)),
            "backbone_embedding_dim": int(cfg.backbone_embedding_dim),
            "torch_dtype": str(torch_dtype).replace("torch.", ""),
        },
    }


def package_vla_action(
    *,
    module: nn.Module,
    role: Any,
    output_dir: str,
    engine_path: Optional[str] = None,
    exported_program_path: Optional[str] = None,
    input_names: list[str],
    output_names: list[str],
    examples: Any = None,
    dtype: Optional[str] = None,
    **_: Any,
) -> dict[str, Any]:
    """Write the action contract discovered from the VLA action module.

    This packager runs after capture/compile. It writes the
    ``action_contract.json`` file with both the stable contract name
    and model-derived runtime metadata.
    """
    PI05PrefixKVActionStep, GR00TStateFlowActionStep, _ = _action_wrapper_symbols()
    torch_dtype = _torch_dtype(dtype)
    if examples is not None and getattr(examples, "args", None):
        for value in examples.args:
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
                torch_dtype = value.dtype
                break

    if isinstance(module, PI05PrefixKVActionStep) or hasattr(module, "core"):
        contract = ACTION_CONTRACT_PREFIX_KV_FLOW_STEP
        max_kv_cache_capacity = int(role.example_kwargs.get("max_kv_cache_capacity", 4096))
        metadata = _prefix_runtime_metadata(
            module,
            max_kv_cache_capacity=max_kv_cache_capacity,
            torch_dtype=torch_dtype,
        )
    elif isinstance(module, GR00TStateFlowActionStep) or hasattr(module, "action_head"):
        contract = ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP
        metadata = _state_runtime_metadata(module, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported VLA action module type: {type(module).__name__}")

    dynamic_axes = dict(role.dynamic_axes or {})
    if not dynamic_axes and examples is not None:
        dynamic_axes = dict(getattr(examples, "dynamic_axes", {}) or {})

    contract_path = write_action_contract_manifest(
        output_dir,
        contract,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        metadata=metadata,
        artifacts={
            "engine": str(Path(engine_path).resolve()) if engine_path else None,
            "exported_program": str(Path(exported_program_path).resolve()) if exported_program_path else None,
        },
    )
    return {"action_contract": str(contract_path), "metadata": metadata}


class VLAEdgeStrategy(EdgeHFStrategy):
    """Build an eager-export manifest for VLA model roles.

    Today this strategy emits the action role first. Language and
    visual roles still delegate to existing component exporters until
    those paths are folded into strategy-generated manifests too.
    """

    def build_manifest(self) -> dict:
        """Return a manifest that exports the discovered VLA action role."""
        roles = set(self.cfg.roles or ("action",))
        unsupported = roles - {"action"}
        if unsupported:
            raise NotImplementedError(
                "The first direct-HF VLA strategy wires the action role. "
                f"Unsupported requested roles: {', '.join(sorted(unsupported))}. "
                "Use the existing component exporters for language/visual while "
                "those roles are folded into this strategy."
            )

        if not self.cfg.action_output_dir and not self.cfg.output_root:
            raise ValueError("VLA action export needs --action_output_dir or --output_root")
        action_output = self.cfg.action_output_dir or str(Path(self.cfg.output_root) / "action")
        loader_kwargs = (
            self.cfg.source.vla_loader_kwargs()
            if self.cfg.source is not None
            else {
                "model_dir": self.cfg.model,
                "model_class": self.cfg.model_class,
                "attn_implementation": self.cfg.attn_implementation,
                "trust_remote_code": self.cfg.trust_remote_code,
            }
        )
        source_metadata = (
            self.cfg.source.to_manifest_metadata()
            if self.cfg.source is not None
            else {}
        )
        return {
            "loader": "tools.edgellm.hf_export.strategies.vla:load_vla_model",
            "loader_kwargs": loader_kwargs,
            "roles": {
                "action": {
                    "component": "action",
                    "module": "action",
                    "example_inputs": "tools.edgellm.hf_export.strategies.vla:vla_action_inputs",
                    "example_kwargs": {
                        "batch_size": int(self.cfg.action_batch_size),
                        "max_kv_cache_capacity": int(self.cfg.max_kv_cache_capacity),
                    },
                    "packager": "tools.edgellm.hf_export.strategies.vla:package_vla_action",
                    "output_dir": action_output,
                    "engine_filename": "action.engine",
                }
            },
            "metadata": {
                "source": "hf_export",
                "family": "vla",
                "model": self.cfg.model,
                **source_metadata,
                **dict(self.cfg.metadata or {}),
            },
        }

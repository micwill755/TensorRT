# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named language runtime contracts for Edge-LLM exports.

Language contracts describe the tensor ABI for the independently exported text
engines. Layer-expanded tensors such as KV caches are written exactly in each
manifest's ``engine_io`` block; the registered contract names describe the
reusable runtime family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .component_contracts import (
    COMPONENT_LANGUAGE,
    ComponentContract,
    write_component_contract_manifest,
)


LANGUAGE_CONTRACT_MANIFEST_FILENAME = "language_contract.json"

LANGUAGE_CONTRACT_DECODER_KV_CACHE = "decoder_kv_cache"
LANGUAGE_CONTRACT_DECODER_KV_CACHE_DEEPSTACK = "decoder_kv_cache_deepstack"
LANGUAGE_CONTRACT_EAGLE_BASE = "eagle_base"
LANGUAGE_CONTRACT_EAGLE_DRAFT = "eagle_draft"
LANGUAGE_CONTRACT_TRT_NATIVE_DECODER = "trt_native_decoder"
LANGUAGE_CONTRACT_TRT_NATIVE_EAGLE_DRAFT = "trt_native_eagle_draft"
LANGUAGE_CONTRACT_HYBRID_DECODER = "hybrid_decoder"
LANGUAGE_CONTRACT_OMNI_TALKER = "omni_talker"
LANGUAGE_CONTRACT_OMNI_CODE_PREDICTOR = "omni_code_predictor"


@dataclass(frozen=True)
class LanguageRuntimeContract:
    """Stable tensor contract family for a language engine export."""

    name: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...] = ("logits",)
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    description: str = ""
    runtime_contract: str = "edgellm"

    def as_component_contract(self) -> ComponentContract:
        return ComponentContract(
            component=COMPONENT_LANGUAGE,
            name=self.name,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            runtime_contract=self.runtime_contract,
            description=self.description,
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        return self.as_component_contract().to_manifest_dict()


_LANGUAGE_RUNTIME_CONTRACTS: Dict[str, LanguageRuntimeContract] = {}


def register_language_runtime_contract(
    contract: LanguageRuntimeContract,
    *,
    replace: bool = False,
) -> LanguageRuntimeContract:
    """Register a language runtime contract by name."""
    if not replace and contract.name in _LANGUAGE_RUNTIME_CONTRACTS:
        raise ValueError(f"Language contract already registered: {contract.name}")
    _LANGUAGE_RUNTIME_CONTRACTS[contract.name] = contract
    return contract


def get_language_runtime_contract(name: str) -> LanguageRuntimeContract:
    """Look up a language contract and fail with a useful message."""
    try:
        return _LANGUAGE_RUNTIME_CONTRACTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_LANGUAGE_RUNTIME_CONTRACTS))
        raise ValueError(f"Unknown language contract {name!r}. Known: {known}") from exc


def list_language_runtime_contracts() -> List[LanguageRuntimeContract]:
    """Return registered language contracts in deterministic name order."""
    return [
        _LANGUAGE_RUNTIME_CONTRACTS[name]
        for name in sorted(_LANGUAGE_RUNTIME_CONTRACTS)
    ]


def write_language_contract_manifest(
    output_dir: str | Path,
    contract_name: str,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    export_dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    filename: str = LANGUAGE_CONTRACT_MANIFEST_FILENAME,
) -> Path:
    """Write language contract metadata beside an exported language artifact."""
    contract = get_language_runtime_contract(contract_name)
    return write_component_contract_manifest(
        output_dir,
        contract.as_component_contract(),
        filename=filename,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_dynamic_axes=export_dynamic_axes,
        metadata=metadata,
        artifacts=artifacts,
        contract_manifest=contract.to_manifest_dict(),
    )


def _register_builtin_contracts() -> None:
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_DECODER_KV_CACHE,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
            ),
            output_names=("logits", "present_key_values_*"),
            description="Autoregressive decoder with per-layer KV cache tensors.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_DECODER_KV_CACHE_DEEPSTACK,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
                "deepstack_embeds_*",
            ),
            output_names=("logits", "hidden_states", "present_key_values_*"),
            description=(
                "Autoregressive decoder with KV cache tensors and visual "
                "deepstack embedding side inputs."
            ),
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_EAGLE_BASE,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
                "attention_pos_id",
                "attention_mask",
            ),
            output_names=("logits", "hidden_states", "present_key_values_*"),
            description="EAGLE base decoder that also exports hidden states.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_EAGLE_DRAFT,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
                "hidden_states_input",
                "hidden_states_from_draft",
                "attention_pos_id",
                "attention_mask",
            ),
            output_names=("logits", "hidden_states", "present_key_values_*"),
            description="EAGLE draft decoder with base and draft hidden-state inputs.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_TRT_NATIVE_DECODER,
            input_names=(
                "inputs_embeds",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "k_cache_*",
                "v_cache_*",
                "kvcache_start_index",
            ),
            output_names=("logits", "present_k_cache_*", "present_v_cache_*"),
            description="TensorRT-native decoder with split key/value cache tensors.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_TRT_NATIVE_EAGLE_DRAFT,
            input_names=(
                "inputs_embeds",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "k_cache_*",
                "v_cache_*",
                "kvcache_start_index",
                "hidden_states_input",
                "hidden_states_from_draft",
                "attention_pos_id",
                "attention_mask",
            ),
            output_names=(
                "logits",
                "hidden_states",
                "present_k_cache_*",
                "present_v_cache_*",
            ),
            description="TensorRT-native EAGLE draft decoder with split key/value caches.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_HYBRID_DECODER,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "conv_state_*",
                "recurrent_state_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
            ),
            output_names=(
                "logits",
                "present_key_values_*",
                "present_conv_state_*",
                "present_recurrent_state_*",
            ),
            description="Hybrid attention/state-space decoder runtime contract.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_OMNI_TALKER,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
            ),
            output_names=("logits", "hidden_states", "present_key_values_*"),
            description="Qwen3-Omni talker decoder runtime contract.",
        )
    )
    register_language_runtime_contract(
        LanguageRuntimeContract(
            name=LANGUAGE_CONTRACT_OMNI_CODE_PREDICTOR,
            input_names=(
                "inputs_embeds",
                "past_key_values_*",
                "rope_rotary_cos_sin",
                "context_lengths",
                "last_token_ids",
                "kvcache_start_index",
                "lm_head_weight",
            ),
            output_names=("logits", "hidden_states", "present_key_values_*"),
            description="Qwen3-Omni code predictor decoder runtime contract.",
        )
    )


_register_builtin_contracts()

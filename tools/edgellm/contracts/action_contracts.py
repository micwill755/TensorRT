# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named action runtime contracts for Edge-LLM exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .component_contracts import (
    COMPONENT_ACTION,
    ComponentContract,
    write_component_contract_manifest,
)


ACTION_CONTRACT_MANIFEST_FILENAME = "action_contract.json"

ACTION_CONTRACT_ALPAMAYO_FLOW_STEP = "alpamayo_flow_step"
ACTION_CONTRACT_PREFIX_KV_FLOW_STEP = "prefix_kv_flow_step"
ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP = "state_conditioned_flow_step"

# Backward-compatible aliases for earlier commands that used model-family names.
# The contracts themselves describe runtime mechanics, not model identity.
ACTION_CONTRACT_PI05_PREFIX_KV_FLOW_STEP = ACTION_CONTRACT_PREFIX_KV_FLOW_STEP
ACTION_CONTRACT_GR00T_STATE_FLOW_STEP = ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP


@dataclass(frozen=True)
class ActionRuntimeContract:
    """Stable tensor contract family for an action engine export."""

    name: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...] = ("output",)
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    description: str = ""
    runtime_contract: str = "edgellm"

    def as_component_contract(self) -> ComponentContract:
        return ComponentContract(
            component=COMPONENT_ACTION,
            name=self.name,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            runtime_contract=self.runtime_contract,
            description=self.description,
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        return self.as_component_contract().to_manifest_dict()


_ACTION_RUNTIME_CONTRACTS: Dict[str, ActionRuntimeContract] = {}


def register_action_runtime_contract(
    contract: ActionRuntimeContract,
    *,
    replace: bool = False,
) -> ActionRuntimeContract:
    """Register an action runtime contract by name."""
    if not replace and contract.name in _ACTION_RUNTIME_CONTRACTS:
        raise ValueError(f"Action contract already registered: {contract.name}")
    _ACTION_RUNTIME_CONTRACTS[contract.name] = contract
    return contract


def get_action_runtime_contract(name: str) -> ActionRuntimeContract:
    """Look up an action contract and fail with a useful message."""
    try:
        return _ACTION_RUNTIME_CONTRACTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ACTION_RUNTIME_CONTRACTS))
        raise ValueError(f"Unknown action contract {name!r}. Known: {known}") from exc


def list_action_runtime_contracts() -> List[ActionRuntimeContract]:
    """Return registered action contracts in deterministic name order."""
    return [
        _ACTION_RUNTIME_CONTRACTS[name]
        for name in sorted(_ACTION_RUNTIME_CONTRACTS)
    ]


def write_action_contract_manifest(
    output_dir: str | Path,
    contract_name: str,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    export_dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    filename: str = ACTION_CONTRACT_MANIFEST_FILENAME,
) -> Path:
    """Write action contract metadata beside an exported action artifact."""
    contract = get_action_runtime_contract(contract_name)
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
    register_action_runtime_contract(
        ActionRuntimeContract(
            name=ACTION_CONTRACT_ALPAMAYO_FLOW_STEP,
            input_names=(
                "noise_trajectory",
                "time_steps_t0",
                "time_steps_t1",
                "kvcache_start_index",
                "rope_rotary_cos_sin",
                "attention_pos_id",
                "k_cache_*",
                "v_cache_*",
            ),
            output_names=(
                "denoised_trajectory",
                "present_k_cache_*",
                "present_v_cache_*",
            ),
            description=(
                "Alpamayo-style single flow-matching denoising step with "
                "split key/value cache tensors."
            ),
        )
    )

    register_action_runtime_contract(
        ActionRuntimeContract(
            name=ACTION_CONTRACT_PREFIX_KV_FLOW_STEP,
            input_names=(
                "noisy_actions",
                "time_steps_t0",
                "time_steps_t1",
                "prefix_pad_mask",
                "prefix_k",
                "prefix_v",
            ),
            output_names=("denoised_actions",),
            dynamic_axes={
                "noisy_actions": {0: "batch_size"},
                "prefix_pad_mask": {
                    0: "batch_size",
                    1: "kv_cache_len",
                },
                "prefix_k": {
                    1: "batch_size",
                    3: "kv_cache_len",
                },
                "prefix_v": {
                    1: "batch_size",
                    3: "kv_cache_len",
                },
                "denoised_actions": {0: "batch_size"},
            },
            description=(
                "Prefix-KV action flow step using prefetched language-prefix "
                "KV cache tensors plus noisy action tokens."
            ),
        )
    )
    register_action_runtime_contract(
        ActionRuntimeContract(
            name=ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP,
            input_names=(
                "actions",
                "timestep",
                "backbone_features",
                "backbone_attention_mask",
                "state",
                "embodiment_id",
                "image_mask",
            ),
            output_names=("action_velocity",),
            dynamic_axes={
                "actions": {0: "batch_size"},
                "timestep": {0: "batch_size"},
                "backbone_features": {
                    0: "batch_size",
                    1: "backbone_seq_len",
                },
                "backbone_attention_mask": {
                    0: "batch_size",
                    1: "backbone_seq_len",
                },
                "state": {0: "batch_size"},
                "embodiment_id": {0: "batch_size"},
                "image_mask": {
                    0: "batch_size",
                    1: "backbone_seq_len",
                },
                "action_velocity": {0: "batch_size"},
            },
            description=(
                "State-conditioned action flow step using VLM backbone features, "
                "robot state, embodiment id, and noisy actions."
            ),
        )
    )


_register_builtin_contracts()

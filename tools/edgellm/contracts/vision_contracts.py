# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named visual input contracts for generic VLM export paths.

A visual input contract is the tensor ABI for an exported visual engine: input
names, output names, and dynamic axes.  It is deliberately not keyed by Hugging
Face model id, because different policy checkpoints can wrap the same kind of
vision tower under different Python module paths.

Export still needs concrete sample tensors so Torch export / Torch-TRT can trace
the graph.  Those samples should come from the strongest available source:
processor outputs when a processor is available, otherwise model/vision config
and module structure, then local HF preprocessing metadata, and finally an
explicit CLI override.  The sample value itself is not runtime data; only its
shape, dtype, and side-input structure define the engine contract.

For `--no_processor` exports, this means we do not invent a fake preprocessing
path.  We infer `pixel_values` shape from the selected vision module and record
the provenance in `visual_contract.json` metadata, so runtime integration can
see what was model-derived and what was user-overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .component_contracts import (
    COMPONENT_VISUAL,
    ComponentContract,
    write_component_contract_manifest,
)


VISION_CONTRACT_MANIFEST_FILENAME = "visual_contract.json"

VIT_INPUT_CONTRACT_NATIVE = "native"
VIT_INPUT_CONTRACT_GRID_THW = "grid_thw"
VIT_INPUT_CONTRACT_STATIC_GRID_THW = "static_grid_thw"
VIT_INPUT_CONTRACT_WINDOWED_ROPE = "windowed_rope"
VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO = "tiled_aspect_ratio"
VIT_INPUT_CONTRACT_FAST_POS_DEEPSTACK = "fast_pos_deepstack"

@dataclass(frozen=True)
class VisionInputContract:
    """Stable tensor contract for one visual-engine export shape.

    `requires_processor_sample` means the contract usually needs a real exemplar
    to resolve shape-dependent side inputs.  The exemplar may be produced by a
    processor or inferred from the model contract when the caller explicitly uses
    `--no_processor`.
    """

    name: str
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...] = ("output",)
    dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    description: str = ""
    requires_processor_sample: bool = True
    runtime_contract: str = "torchtrt"
    input_bindings: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    output_bindings: Tuple[Mapping[str, Any], ...] = field(
        default_factory=lambda: ({"name": "output", "semantic": "vision_embeddings"},)
    )

    def as_component_contract(self) -> ComponentContract:
        return ComponentContract(
            component=COMPONENT_VISUAL,
            name=self.name,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            runtime_contract=self.runtime_contract,
            description=self.description,
            input_bindings=self.input_bindings,
            output_bindings=self.output_bindings,
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        data = self.as_component_contract().to_manifest_dict()
        data["requires_processor_sample"] = self.requires_processor_sample
        return data


_VISION_INPUT_CONTRACTS: Dict[str, VisionInputContract] = {}


def register_vision_input_contract(
    contract: VisionInputContract,
    *,
    replace: bool = False,
) -> VisionInputContract:
    """Register a visual input contract by name."""
    if not replace and contract.name in _VISION_INPUT_CONTRACTS:
        raise ValueError(f"Visual input contract already registered: {contract.name}")
    _VISION_INPUT_CONTRACTS[contract.name] = contract
    return contract


def get_vision_input_contract(name: str) -> VisionInputContract:
    """Look up a visual input contract and fail with a useful message."""
    try:
        return _VISION_INPUT_CONTRACTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_VISION_INPUT_CONTRACTS))
        raise ValueError(f"Unknown visual input contract {name!r}. Known: {known}") from exc


def list_vision_input_contracts() -> List[VisionInputContract]:
    """Return registered contracts in deterministic name order."""
    return [
        _VISION_INPUT_CONTRACTS[name]
        for name in sorted(_VISION_INPUT_CONTRACTS)
    ]


def write_vision_contract_manifest(
    output_dir: str | Path,
    contract_name: str,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    output_bindings: Optional[List[Mapping[str, Any]]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    export_dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    filename: str = VISION_CONTRACT_MANIFEST_FILENAME,
) -> Path:
    """Write visual contract metadata beside an exported visual artifact."""
    contract = get_vision_input_contract(contract_name)
    return write_component_contract_manifest(
        output_dir,
        contract.as_component_contract(),
        filename=filename,
        input_names=input_names,
        output_names=output_names,
        output_bindings=output_bindings,
        dynamic_axes=dynamic_axes,
        export_dynamic_axes=export_dynamic_axes,
        metadata=metadata,
        artifacts=artifacts,
        contract_manifest=contract.to_manifest_dict(),
    )


def _register_builtin_contracts() -> None:
    register_vision_input_contract(
        VisionInputContract(
            name=VIT_INPUT_CONTRACT_NATIVE,
            input_names=("input",),
            input_bindings=(
                {"name": "input", "source": "image_pixels_nchw"},
            ),
            dynamic_axes={"input": {0: "hw"}},
            description="Call the selected visual module with pixel/input tensor only.",
            requires_processor_sample=True,
        )
    )
    register_vision_input_contract(
        VisionInputContract(
            name=VIT_INPUT_CONTRACT_GRID_THW,
            input_names=("input", "grid_thw"),
            input_bindings=(
                {"name": "input", "source": "packed_image_patches"},
                {"name": "grid_thw", "source": "image_grid_thw"},
            ),
            dynamic_axes={"input": {0: "hw"}, "grid_thw": {0: "num_images"}},
            description="Call the visual module with explicit image grid_thw.",
            requires_processor_sample=True,
        )
    )
    register_vision_input_contract(
        VisionInputContract(
            name=VIT_INPUT_CONTRACT_STATIC_GRID_THW,
            input_names=("input",),
            input_bindings=(
                {"name": "input", "source": "packed_image_patches"},
            ),
            dynamic_axes={"input": {0: "hw"}},
            description=(
                "Bake the processor-derived image_grid_thw side inputs into the "
                "wrapper and expose only the pixel/input tensor."
            ),
            requires_processor_sample=True,
        )
    )
    register_vision_input_contract(
        VisionInputContract(
            name=VIT_INPUT_CONTRACT_WINDOWED_ROPE,
            input_names=(
                "input",
                "rotary_pos_emb",
                "attention_mask",
                "window_attention_mask",
                "cu_window_seqlens",
                "window_index",
                "reverse_window_index",
            ),
            input_bindings=(
                {"name": "input", "source": "packed_image_patches"},
                {"name": "rotary_pos_emb", "source": "rotary_pos_emb"},
                {"name": "attention_mask", "source": "attention_mask"},
                {"name": "window_attention_mask", "source": "window_attention_mask"},
                {"name": "cu_window_seqlens", "source": "cu_window_seqlens"},
                {"name": "window_index", "source": "window_index"},
                {"name": "reverse_window_index", "source": "reverse_window_index"},
            ),
            dynamic_axes={
                "input": {0: "hw"},
                "rotary_pos_emb": {0: "hw"},
                "attention_mask": {1: "hw", 2: "hw"},
                "window_attention_mask": {1: "hw", 2: "hw"},
                "cu_window_seqlens": {0: "num_windows_plus_1"},
                "window_index": {0: "window_tokens"},
                "reverse_window_index": {0: "window_tokens"},
            },
            description="Expose Qwen-style windowed RoPE side tensors explicitly.",
            requires_processor_sample=True,
        )
    )
    register_vision_input_contract(
        VisionInputContract(
            name=VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO,
            input_names=("input", "aspect_ratio_ids", "attention_mask"),
            input_bindings=(
                {"name": "input", "source": "tiled_image_pixels"},
                {"name": "aspect_ratio_ids", "source": "aspect_ratio_ids"},
                {"name": "attention_mask", "source": "tile_attention_mask"},
            ),
            dynamic_axes={
                "input": {0: "batch", 1: "media", 2: "tiles"},
                "attention_mask": {0: "batch"},
                "aspect_ratio_ids": {0: "batch", 1: "media"},
                "output": {0: "batch"},
            },
            description="Expose tiled/aspect-ratio visual inputs for tile-based towers.",
            requires_processor_sample=True,
        )
    )
    register_vision_input_contract(
        VisionInputContract(
            name=VIT_INPUT_CONTRACT_FAST_POS_DEEPSTACK,
            input_names=(
                "input",
                "rotary_pos_emb",
                "cu_seqlens",
                "max_seqlen_carrier",
                "fast_pos_embed_idx",
                "fast_pos_embed_weight",
            ),
            input_bindings=(
                {"name": "input", "source": "packed_image_patches"},
                {"name": "rotary_pos_emb", "source": "rotary_pos_emb"},
                {"name": "cu_seqlens", "source": "cu_seqlens"},
                {"name": "max_seqlen_carrier", "source": "max_seqlen_carrier"},
                {"name": "fast_pos_embed_idx", "source": "fast_pos_embed_idx"},
                {"name": "fast_pos_embed_weight", "source": "fast_pos_embed_weight"},
            ),
            dynamic_axes={
                "input": {0: "hw"},
                "rotary_pos_emb": {0: "hw"},
                "cu_seqlens": {0: "batch_size + 1"},
                "max_seqlen_carrier": {0: "max_seqlen"},
                "fast_pos_embed_idx": {1: "hw"},
                "fast_pos_embed_weight": {1: "hw"},
                "output": {0: "image_token_len"},
            },
            description=(
                "Edge-LLM Qwen/Cosmos-style packed visual contract with explicit "
                "RoPE, cu_seqlens, max sequence carrier, and fast positional "
                "embedding tensors."
            ),
            requires_processor_sample=True,
            runtime_contract="edgellm",
        )
    )


_register_builtin_contracts()

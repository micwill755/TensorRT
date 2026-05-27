# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diffusion-inspired role resolution for Hugging Face Edge exports.

This module is intentionally deterministic.  The "diffusion" idea here is a
progressive denoising pass over weak Hugging Face signals: model id, config
keys, user hints, and known runtime contracts.  The output is a ranked set of
language / visual / action role candidates plus the evidence used to choose
them.  A later export step can use those candidates directly, ask the user for
overrides, or run a heavier model-probe pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from tools.edgellm.contracts.action_contracts import (
    ACTION_CONTRACT_ALPAMAYO_FLOW_STEP,
    ACTION_CONTRACT_PREFIX_KV_FLOW_STEP,
    ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP,
)
from tools.edgellm.contracts.language_contracts import (
    LANGUAGE_CONTRACT_DECODER_KV_CACHE,
    LANGUAGE_CONTRACT_DECODER_KV_CACHE_DEEPSTACK,
)
from tools.edgellm.contracts.vision_contracts import (
    VIT_INPUT_CONTRACT_FAST_POS_DEEPSTACK,
    VIT_INPUT_CONTRACT_NATIVE,
    VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO,
    VIT_INPUT_CONTRACT_WINDOWED_ROPE,
)
from tools.edgellm.hf_export.detect import _config_dict, is_vision_encoder_model_type
from tools.edgellm.hf_export.source import HFModelSource


ROLE_ORDER = ("language", "visual", "action")
ROLE_COMPONENTS = {
    "language": "language",
    "visual": "visual",
    "action": "action",
}

EDGE_DEFAULT_ROLE_FAMILIES = {
    "llm": ("language",),
    "vlm": ("language", "visual"),
    "vla": ROLE_ORDER,
    "multimodal": ("visual",),
}

EDGE_UNSUPPORTED_FAMILY_NOTES = {
    "audio": "audio models are detected, but no Edge audio runtime contract is wired yet",
    "detection": "detection models are detected, but no Edge detection runtime contract is wired yet",
    "diffusion": "diffusion pipelines are detected, but they do not map to the current Edge LLM runtime roles",
    "llm_tp": "tensor-parallel LLM is a Torch-TensorRT compile family, not an Edge role contract yet",
    "seq2seq": "seq2seq models need an encoder/decoder contract before Edge export can default roles",
    "video_diffusion": "video diffusion pipelines are detected, but no Edge video runtime contract is wired yet",
}


@dataclass(frozen=True)
class RoleCandidate:
    """One possible mapping from a HF model to an Edge export role."""

    role: str
    component: str
    module_path: str
    contract: str
    confidence: float
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate for JSON reports."""
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


@dataclass(frozen=True)
class ResolutionStep:
    """A compact record of one denoising stage."""

    name: str
    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this step for JSON reports."""
        return {
            "name": self.name,
            "summary": self.summary,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class HFRoleResolutionPlan:
    """Ranked role candidates plus the selected export plan."""

    model: str
    family: str
    roles: tuple[str, ...]
    selected: Mapping[str, RoleCandidate]
    candidates: Mapping[str, tuple[RoleCandidate, ...]]
    unresolved_roles: tuple[str, ...]
    steps: tuple[ResolutionStep, ...]
    config_keys: tuple[str, ...]
    family_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full resolution plan for humans and automation."""
        return {
            "model": self.model,
            "family": self.family,
            "roles": list(self.roles),
            "selected": {
                role: candidate.to_dict()
                for role, candidate in self.selected.items()
            },
            "candidates": {
                role: [candidate.to_dict() for candidate in candidates]
                for role, candidates in self.candidates.items()
            },
            "unresolved_roles": list(self.unresolved_roles),
            "steps": [step.to_dict() for step in self.steps],
            "config_keys": list(self.config_keys),
            "family_notes": list(self.family_notes),
        }


def _default_roles_for_family(
    family: str,
    *,
    source: HFModelSource,
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return default Edge roles and explanatory notes for a family."""
    if family in EDGE_DEFAULT_ROLE_FAMILIES:
        roles = EDGE_DEFAULT_ROLE_FAMILIES[family]
        if family == "multimodal":
            return roles, (
                "multimodal defaults to visual role only; CLIP-style text encoders are not LLM decoder contracts",
            )
        return roles, ()

    if family == "encoder":
        model_type = str(config.get("model_type", "")).lower()
        config_blob = _config_text(config)
        looks_visual = (
            is_vision_encoder_model_type(model_type)
            or _contains_any(
                config_blob,
                ("image_size", "num_channels", "patch_size", "vision_config"),
            )
            or any(token in source.model.lower() for token in ("vit", "resnet", "dinov2", "convnext", "swin"))
        )
        if looks_visual:
            return ("visual",), (
                "encoder defaults to visual role because the config looks like a vision encoder",
            )
        return (), (
            "encoder family detected, but text encoders do not match the Edge LLM decoder contract",
        )

    note = EDGE_UNSUPPORTED_FAMILY_NOTES.get(
        family,
        f"family {family!r} has no default Edge role mapping yet",
    )
    return (), (note,)


def _requested_roles(
    source: HFModelSource,
    roles: Optional[Sequence[str]],
    *,
    family: str,
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Choose requested roles in runtime dependency order."""
    raw_roles = tuple(roles or source.roles or ())
    if raw_roles:
        requested = {
            part.strip()
            for role in raw_roles
            for part in str(role).split(",")
            if part.strip()
        }
        notes: tuple[str, ...] = ()
    else:
        default_roles, notes = _default_roles_for_family(
            family,
            source=source,
            config=config,
        )
        requested = set(default_roles)

    unknown = requested - set(ROLE_ORDER)
    if unknown:
        raise ValueError(f"Unsupported role(s): {', '.join(sorted(unknown))}")
    return tuple(role for role in ROLE_ORDER if role in requested), notes


def _flatten_config_keys(value: Any, *, prefix: str = "", depth: int = 0) -> tuple[str, ...]:
    """Collect stable config-key paths without dumping huge values."""
    if depth > 4:
        return ()
    if isinstance(value, Mapping):
        keys: list[str] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_config_keys(child, prefix=path, depth=depth + 1))
        return tuple(keys)
    if isinstance(value, list):
        keys = []
        for idx, child in enumerate(value[:4]):
            path = f"{prefix}[{idx}]"
            keys.extend(_flatten_config_keys(child, prefix=path, depth=depth + 1))
        return tuple(keys)
    return ()


def _config_text(config: Mapping[str, Any]) -> str:
    """Return a lowercase searchable config blob."""
    return json.dumps(config, sort_keys=True, default=str).lower()


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    """Check whether any marker appears in the text."""
    return any(marker.lower() in text for marker in markers)


def _add_candidate(
    candidates: dict[str, list[RoleCandidate]],
    candidate: RoleCandidate,
) -> None:
    """Insert a candidate, merging exact duplicates by confidence/evidence."""
    bucket = candidates.setdefault(candidate.role, [])
    for idx, existing in enumerate(bucket):
        if (
            existing.module_path == candidate.module_path
            and existing.contract == candidate.contract
        ):
            merged_evidence = tuple(dict.fromkeys((*existing.evidence, *candidate.evidence)))
            bucket[idx] = RoleCandidate(
                role=existing.role,
                component=existing.component,
                module_path=existing.module_path,
                contract=existing.contract,
                confidence=max(existing.confidence, candidate.confidence),
                evidence=merged_evidence,
            )
            return
    bucket.append(candidate)


def _language_contract(config_blob: str) -> str:
    """Pick the most likely language runtime contract family."""
    if _contains_any(config_blob, ("deepstack", "fast_pos", "visual_hidden_states")):
        return LANGUAGE_CONTRACT_DECODER_KV_CACHE_DEEPSTACK
    return LANGUAGE_CONTRACT_DECODER_KV_CACHE


def _vision_contract(config_blob: str, model_name: str) -> str:
    """Pick the most likely visual runtime contract family."""
    text = f"{config_blob} {model_name}".lower()
    if _contains_any(text, ("qwen3_vl", "qwen3-vl", "qwen2_vl", "qwen2-vl", "mrope", "window_index")):
        return VIT_INPUT_CONTRACT_WINDOWED_ROPE
    if _contains_any(text, ("mllama", "llama-vision", "aspect_ratio_ids", "num_tiles")):
        return VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO
    if _contains_any(text, ("deepstack", "fast_pos_embed")):
        return VIT_INPUT_CONTRACT_FAST_POS_DEEPSTACK
    return VIT_INPUT_CONTRACT_NATIVE


def _action_contract(config_blob: str, model_name: str) -> str:
    """Pick the most likely action runtime contract family."""
    text = f"{config_blob} {model_name}".lower()
    if _contains_any(text, ("pi05", "openpi", "paligemma_with_expert", "prefix_kv")):
        return ACTION_CONTRACT_PREFIX_KV_FLOW_STEP
    if _contains_any(text, ("gr00t", "state_conditioned", "embodiment_id", "action_head")):
        return ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP
    if _contains_any(text, ("alpamayo", "noise_trajectory")):
        return ACTION_CONTRACT_ALPAMAYO_FLOW_STEP
    return ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP


def _seed_explicit_hints(
    source: HFModelSource,
    candidates: dict[str, list[RoleCandidate]],
    config_blob: str,
) -> None:
    """Add high-confidence user-provided module hints."""
    hint_specs = (
        ("language", source.language_module, _language_contract(config_blob), "explicit --language_module"),
        ("visual", source.vision_module, _vision_contract(config_blob, source.model), "explicit --vision_module"),
        ("action", source.action_module, _action_contract(config_blob, source.model), "explicit --action_module"),
    )
    for role, module_path, contract, evidence in hint_specs:
        if module_path:
            _add_candidate(
                candidates,
                RoleCandidate(
                    role=role,
                    component=ROLE_COMPONENTS[role],
                    module_path=module_path,
                    contract=contract,
                    confidence=1.0,
                    evidence=(evidence,),
                ),
            )


def _seed_language_candidates(
    source: HFModelSource,
    candidates: dict[str, list[RoleCandidate]],
    config_blob: str,
) -> None:
    """Add language candidates from HF config and common module names."""
    contract = _language_contract(config_blob)
    markers = ("causal", "decoder", "is_decoder", "llm_config", "language_config")
    has_markers = _contains_any(config_blob, markers)
    if has_markers or source.detected_family() in {"llm", "vlm", "vla"}:
        evidence = [
            "config has language-model markers"
            if has_markers
            else "requested/detected family implies a language role"
        ]
        if "pi05" in source.model.lower() or "paligemma" in config_blob:
            module_path = "model.paligemma_with_expert.paligemma.model.language_model"
            evidence.append("PI0.5/PaliGemma language path heuristic")
            confidence = 0.86
        else:
            module_path = "language_model"
            confidence = 0.64
        _add_candidate(
            candidates,
            RoleCandidate(
                role="language",
                component="language",
                module_path=module_path,
                contract=contract,
                confidence=confidence,
                evidence=tuple(evidence),
            ),
        )


def _seed_visual_candidates(
    source: HFModelSource,
    candidates: dict[str, list[RoleCandidate]],
    config_blob: str,
) -> None:
    """Add visual candidates from HF config and common module names."""
    contract = _vision_contract(config_blob, source.model)
    markers = ("vision_config", "visual_config", "image_size", "image_token", "vision_feature")
    has_markers = _contains_any(config_blob, markers)
    family = source.detected_family()
    looks_visual_encoder = family == "encoder" and (
        has_markers
        or any(token in source.model.lower() for token in ("vit", "resnet", "dinov2", "convnext", "swin"))
    )
    if has_markers or family in {"vlm", "vla", "multimodal"} or looks_visual_encoder:
        evidence = [
            "config has vision-model markers"
            if has_markers
            else "requested/detected family implies a visual role"
        ]
        if family == "encoder":
            module_path = "model"
            evidence.append("vision encoder family heuristic")
            confidence = 0.72
        elif "pi05" in source.model.lower() or "paligemma" in config_blob:
            module_path = "model.paligemma_with_expert.paligemma.model.vision_tower"
            evidence.append("PI0.5/PaliGemma vision path heuristic")
            confidence = 0.84
        elif _contains_any(f"{config_blob} {source.model}", ("qwen", "mrope")):
            module_path = "visual"
            evidence.append("Qwen-style visual contract markers")
            confidence = 0.76
        else:
            module_path = "vision_model"
            confidence = 0.62
        _add_candidate(
            candidates,
            RoleCandidate(
                role="visual",
                component="visual",
                module_path=module_path,
                contract=contract,
                confidence=confidence,
                evidence=tuple(evidence),
            ),
        )


def _seed_action_candidates(
    source: HFModelSource,
    candidates: dict[str, list[RoleCandidate]],
    config_blob: str,
) -> None:
    """Add action candidates from HF config and common VLA markers."""
    text = f"{config_blob} {source.model}".lower()
    markers = (
        "action_head",
        "action_horizon",
        "max_action_dim",
        "state_history_length",
        "embodiment",
        "policy_type",
        "robot",
        "vla",
        "pi05",
        "gr00t",
    )
    if _contains_any(text, markers) or source.detected_family() == "vla":
        contract = _action_contract(config_blob, source.model)
        evidence = ["config/model id has VLA action markers"]
        confidence = 0.66
        if contract == ACTION_CONTRACT_PREFIX_KV_FLOW_STEP:
            evidence.append("prefix-KV/OpenPI marker match")
            confidence = 0.88
        elif contract == ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP:
            evidence.append("state-conditioned action marker match")
            confidence = 0.78
        _add_candidate(
            candidates,
            RoleCandidate(
                role="action",
                component="action",
                module_path="action",
                contract=contract,
                confidence=confidence,
                evidence=tuple(evidence),
            ),
        )


def resolve_hf_roles(
    source: HFModelSource,
    *,
    roles: Optional[Sequence[str]] = None,
    min_confidence: float = 0.55,
) -> HFRoleResolutionPlan:
    """Resolve likely Edge roles from HF metadata.

    The pass is deliberately cheap: it does not instantiate the full model.
    That makes it suitable as the first stage before a more expensive probe
    that validates module paths and example inputs.
    """
    family = source.detected_family()
    config = _config_dict(source.model)
    requested, family_notes = _requested_roles(
        source,
        roles,
        family=family,
        config=config,
    )
    config_blob = _config_text(config)
    config_keys = tuple(sorted(set(_flatten_config_keys(config))))[:256]
    candidates: dict[str, list[RoleCandidate]] = {role: [] for role in requested}
    steps: list[ResolutionStep] = [
        ResolutionStep(
            name="seed",
            summary="Seeded resolver from HF source, requested roles, and coarse family detection.",
            details={"family": family, "roles": list(requested), "family_notes": list(family_notes)},
        ),
        ResolutionStep(
            name="config_scan",
            summary="Scanned HF config text for language, visual, and action markers.",
            details={"config_key_count": len(config_keys), "has_config": bool(config)},
        ),
    ]

    if family_notes:
        steps.append(
            ResolutionStep(
                name="family_policy",
                summary="Applied Edge-specific default-role policy for the detected family.",
                details={"notes": list(family_notes)},
            )
        )

    _seed_explicit_hints(source, candidates, config_blob)
    if "language" in requested:
        _seed_language_candidates(source, candidates, config_blob)
    if "visual" in requested:
        _seed_visual_candidates(source, candidates, config_blob)
    if "action" in requested:
        _seed_action_candidates(source, candidates, config_blob)

    ranked: dict[str, tuple[RoleCandidate, ...]] = {}
    selected: dict[str, RoleCandidate] = {}
    unresolved: list[str] = []
    for role in requested:
        role_candidates = tuple(sorted(candidates.get(role, ()), key=lambda item: item.confidence, reverse=True))
        ranked[role] = role_candidates
        if role_candidates and role_candidates[0].confidence >= min_confidence:
            selected[role] = role_candidates[0]
        else:
            unresolved.append(role)

    steps.append(
        ResolutionStep(
            name="rank_candidates",
            summary="Ranked candidates and selected the highest-confidence role above threshold.",
            details={
                "min_confidence": min_confidence,
                "selected_roles": sorted(selected),
                "unresolved_roles": unresolved,
            },
        )
    )
    return HFRoleResolutionPlan(
        model=source.model,
        family=family,
        roles=requested,
        selected=selected,
        candidates=ranked,
        unresolved_roles=tuple(unresolved),
        steps=tuple(steps),
        config_keys=config_keys,
        family_notes=tuple(family_notes),
    )


def candidate_manifest_from_plan(
    plan: HFRoleResolutionPlan,
    source: HFModelSource,
    *,
    output_root: Optional[str] = None,
    action_batch_size: int = 2,
    max_kv_cache_capacity: int = 4096,
) -> dict[str, Any]:
    """Create an inspectable eager-style manifest from selected candidates.

    The action role is executable for VLA strategies that load through
    ``load_vla_model``.  Language and visual entries are intentionally marked
    as candidate-only until their direct-HF strategy packagers are folded into
    this resolver.
    """
    roles: dict[str, dict[str, Any]] = {}
    root = Path(output_root).expanduser().resolve() if output_root else None
    for role, candidate in plan.selected.items():
        output_dir = str(root / role) if root else None
        role_data: dict[str, Any] = {
            "component": candidate.component,
            "module": candidate.module_path,
            "contract": candidate.contract,
            "metadata": {
                "confidence": candidate.confidence,
                "evidence": list(candidate.evidence),
                "candidate_only": role in {"language", "visual"},
            },
        }
        if output_dir:
            role_data["output_dir"] = output_dir
        if role == "language":
            role_data["engine_filename"] = "llm.engine"
        elif role == "visual":
            role_data["engine_filename"] = "visual.engine"
        elif role == "action":
            role_data.update(
                {
                    "module": "action",
                    "example_inputs": "tools.edgellm.hf_export.strategies.vla:vla_action_inputs",
                    "example_kwargs": {
                        "batch_size": int(action_batch_size),
                        "max_kv_cache_capacity": int(max_kv_cache_capacity),
                    },
                    "packager": "tools.edgellm.hf_export.strategies.vla:package_vla_action",
                    "engine_filename": "action.engine",
                }
            )
        roles[role] = role_data

    manifest = {
        "roles": roles,
        "metadata": {
            "source": "hf_resolution_diffusion",
            "family": plan.family,
            "model": plan.model,
            "candidate_manifest": True,
            "notes": (
                "This manifest is generated from lightweight HF metadata. "
                "Run a model-probe pass or provide explicit module overrides "
                "before treating language/visual candidates as final."
            ),
        },
    }
    if plan.family == "vla" and "action" in plan.selected:
        manifest["loader"] = "tools.edgellm.hf_export.strategies.vla:load_vla_model"
        manifest["loader_kwargs"] = source.vla_loader_kwargs()
    else:
        manifest["loader"] = None
        manifest["loader_kwargs"] = {}
    return manifest


def write_resolution_report(plan: HFRoleResolutionPlan, output_path: str | Path) -> Path:
    """Write a JSON report and return its path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

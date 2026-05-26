# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight family detection for direct Hugging Face Edge exports."""

from __future__ import annotations

import json
from typing import Any, Optional


_FAMILIES = {"llm", "vlm", "vla"}


def _config_dict(model: str) -> dict[str, Any]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        return dict(getattr(cfg, "__dict__", {}))
    except Exception:
        try:
            from transformers import PretrainedConfig

            data, _ = PretrainedConfig.get_config_dict(model)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def detect_family(model: str, *, task_override: Optional[str] = None) -> str:
    """Detect a coarse HF family: llm, vlm, or vla.

    This intentionally mirrors the run_hf.py shape: broad family selection first,
    then strategy-specific structural detection later. It is not a model-id
    switchboard.
    """
    if task_override:
        task = task_override.strip().lower().replace("_", "-")
        if task in _FAMILIES:
            return task
        if any(token in task for token in ("action", "robot", "vla")):
            return "vla"
        if any(token in task for token in ("vision", "image", "multimodal")):
            return "vlm"
        if any(token in task for token in ("text", "causal", "generation")):
            return "llm"

    cfg = _config_dict(model)
    cfg_text = json.dumps(cfg, sort_keys=True, default=str).lower()
    model_lower = str(model).lower()

    vla_markers = (
        "action_head",
        "action_horizon",
        "max_action_dim",
        "max_state_dim",
        "state_history_length",
        "embodiment",
        "policy_type",
        "vla",
        "robot",
    )
    if any(marker in cfg_text for marker in vla_markers):
        return "vla"
    if any(marker in model_lower for marker in ("vla", "robot", "gr00t", "pi05", "openvla", "spatialvla")):
        return "vla"

    vlm_markers = (
        "vision_config",
        "visual_config",
        "image_token",
        "image_processor",
        "vision_feature",
        "multi_modal",
        "multimodal",
    )
    if any(marker in cfg_text for marker in vlm_markers):
        return "vlm"

    return "llm"

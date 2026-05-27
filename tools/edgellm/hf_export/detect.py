# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight family detection for direct Hugging Face Edge exports.

Family detection is a coarse first routing step. It should answer
"what kind of HF model is this?" without loading model weights. The Edge
exporter then maps that family to role candidates and runtime contracts.

This follows the broader ``tools/hf`` strategy shape: detect families such as
LLM, VLM, VLA, encoder, diffusion, audio, and detection, then let role
resolution decide what Edge artifacts are actually supported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


FAMILIES = {
    "llm",
    "llm_tp",
    "vlm",
    "vla",
    "encoder",
    "seq2seq",
    "detection",
    "diffusion",
    "video_diffusion",
    "audio",
    "multimodal",
}

_LLM_TYPES = {
    "bloom", "codegen", "cohere", "cohere2", "dbrx", "deepseek",
    "deepseek_v2", "deepseek_v3", "exaone", "falcon", "gemma",
    "gemma2", "gemma3", "gpt2", "gpt_bigcode", "gpt_neo",
    "gpt_neox", "gptj", "internlm", "internlm2", "llama", "mistral",
    "mixtral", "mpt", "nemotron", "olmo", "olmo2", "opt",
    "persimmon", "phi", "phi3", "qwen2", "qwen2_moe", "qwen3",
    "qwen3_moe", "starcoder2", "stablelm",
}

_ENCODER_TYPES = {
    "albert", "beit", "bert", "bit", "camembert", "convnext",
    "data2vec_vision", "deberta", "deberta-v2", "deit",
    "depth_anything", "dinov2", "distilbert", "dpt", "efficientnet",
    "electra", "glpn", "levit", "mobilenet_v2", "mobilevit",
    "mobilevitv2", "poolformer", "regnet", "resnet", "roberta",
    "segformer", "swin", "vit", "xlm", "xlm-roberta",
}

_VISION_ENCODER_TYPES = {
    "beit", "bit", "convnext", "data2vec_vision", "deit",
    "depth_anything", "dinov2", "dpt", "efficientnet", "glpn", "levit",
    "mobilenet_v2", "mobilevit", "mobilevitv2", "poolformer", "regnet",
    "resnet", "segformer", "swin", "vit",
}

_SEQ2SEQ_TYPES = {"bart", "fsmt", "longt5", "m2m_100", "marian", "mbart", "mt5", "pegasus", "t5"}

_DIFFUSION_TYPES = {
    "flux", "stable-diffusion", "stable_diffusion", "stable-diffusion-xl",
    "stable_diffusion_xl", "unet-2d-condition",
}

_VIDEO_DIFFUSION_TYPES = {"cogvideox", "unet-spatio-temporal-condition", "unet_spatio_temporal_condition"}

_VIDEO_PIPELINE_KEYWORDS = (
    "cogvideox", "animatediff", "stablevideo", "stablevideodiffusion",
    "img2vid", "imagetovideo", "texttovideo", "i2vgen",
    "videocrafter", "modelscope", "zeroscope",
)

_AUDIO_TYPES = {"hubert", "wav2vec2", "wavlm", "whisper"}

_MULTIMODAL_TYPES = {"align", "blip", "blip-2", "clip", "clipseg", "flava", "groupvit", "siglip", "x_clip"}

_VLA_TYPES = {"openvla", "pi0", "prismatic", "spatialvla", "tinyvla"}

_VLM_TYPES = {
    "aria", "chameleon", "florence2", "idefics2", "idefics3", "llava",
    "llava_next", "llava_next_video", "llava_onevision", "paligemma",
    "phi3_v", "qwen2_vl", "qwen2_5_vl", "qwen3_vl",
}

_DETECTION_TYPES = {
    "conditional_detr", "dab_detr", "deta", "detr", "grounding_dino",
    "owlv2", "owlvit", "rt_detr", "rt_detr_resnet", "sam",
    "table_transformer", "yolos",
}

_TASK_MAP = {
    "text-generation": "llm",
    "text2text-generation": "seq2seq",
    "text-classification": "encoder",
    "token-classification": "encoder",
    "fill-mask": "encoder",
    "feature-extraction": "encoder",
    "image-classification": "encoder",
    "image-segmentation": "detection",
    "object-detection": "detection",
    "zero-shot-object-detection": "detection",
    "zero-shot-image-classification": "multimodal",
    "visual-question-answering": "vlm",
    "image-to-text": "vlm",
    "robot-action": "vla",
    "vision-language-action": "vla",
    "image-generation": "diffusion",
    "image-to-image": "diffusion",
    "text-to-video": "video_diffusion",
    "image-to-video": "video_diffusion",
    "video-generation": "video_diffusion",
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
}


def _config_dict(model: str) -> dict[str, Any]:
    """Best-effort load of a Hugging Face config as a dictionary."""
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


def _diffusers_family(model: str) -> Optional[str]:
    """Detect diffusers pipelines from model_index.json when available."""
    index: dict[str, Any] = {}
    local_index = Path(model).expanduser() / "model_index.json"
    if local_index.exists():
        try:
            index = json.loads(local_index.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model, "model_index.json")
            index = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            index = {}

    class_name = str(index.get("_class_name", "")).lower()
    if any(keyword in class_name for keyword in _VIDEO_PIPELINE_KEYWORDS):
        return "video_diffusion"
    if "pipeline" in class_name or any(keyword in class_name for keyword in ("diffusion", "flux")):
        return "diffusion"
    return None


def _model_type_to_family(model_type: str, model: str) -> Optional[str]:
    """Map a Hugging Face model_type to a broad strategy family."""
    model_type = model_type.lower()
    if model_type in _LLM_TYPES:
        return "llm"
    if model_type in _VLA_TYPES:
        return "vla"
    if model_type in _VLM_TYPES:
        return "vlm"
    if model_type in _ENCODER_TYPES:
        return "encoder"
    if model_type in _SEQ2SEQ_TYPES:
        return "seq2seq"
    if model_type in _DETECTION_TYPES:
        return "detection"
    if model_type in _VIDEO_DIFFUSION_TYPES:
        return "video_diffusion"
    if model_type in _DIFFUSION_TYPES:
        return "diffusion"
    if model_type in _AUDIO_TYPES:
        return "audio"
    if model_type in _MULTIMODAL_TYPES:
        return "multimodal"

    lower_model = model.lower()
    if any(token in lower_model for token in ("openvla", "spatialvla", "tinyvla", "prismatic-vla", "/vla-", "pi05", "/pi0")):
        return "vla"
    if any(token in lower_model for token in ("llava", "paligemma", "qwen2-vl", "qwen2vl", "qwen3-vl", "smolvlm", "idefics", "moondream")):
        return "vlm"
    if any(token in lower_model for token in ("detr", "rtdetr", "sam-vit", "owlvit", "owlv2")):
        return "detection"
    if any(token in lower_model for token in _VIDEO_PIPELINE_KEYWORDS):
        return "video_diffusion"
    if any(token in lower_model for token in ("diffusion", "flux", "sd-", "sdxl")):
        return "diffusion"
    if any(token in lower_model for token in ("whisper", "wav2vec", "wavlm", "asr")):
        return "audio"
    if any(token in lower_model for token in ("llama", "gpt", "mistral", "qwen", "gemma")):
        return "llm"
    if any(token in lower_model for token in ("bert", "vit", "resnet")):
        return "encoder"
    return None


def _task_to_family(task: str) -> str:
    """Map a user task/family override to a broad family."""
    normalized = task.strip().lower().replace("_", "-")
    underscore = normalized.replace("-", "_")
    if normalized in FAMILIES:
        return normalized
    if underscore in FAMILIES:
        return underscore
    family = _TASK_MAP.get(normalized)
    if family is not None:
        return family
    if any(token in normalized for token in ("action", "robot", "vla")):
        return "vla"
    if any(token in normalized for token in ("vision-language", "image-to-text", "vqa")):
        return "vlm"
    if any(token in normalized for token in ("vision", "image")):
        return "encoder"
    if any(token in normalized for token in ("text", "causal", "generation")):
        return "llm"
    raise ValueError(f"Unknown HF task/family override: {task!r}")


def detect_family(model: str, *, task_override: Optional[str] = None) -> str:
    """Detect a broad HF family such as ``llm``, ``vlm``, or ``vla``.

    The returned family is only a routing hint. Edge role resolution
    still decides which runtime components and contracts are supported.
    """
    if task_override:
        return _task_to_family(task_override)

    cfg = _config_dict(model)
    cfg_text = json.dumps(cfg, sort_keys=True, default=str).lower()
    model_lower = str(model).lower()

    vla_markers = (
        "action_head", "action_horizon", "max_action_dim", "max_state_dim",
        "state_history_length", "embodiment", "policy_type", "vla", "robot",
    )
    if any(marker in cfg_text for marker in vla_markers):
        return "vla"
    if any(marker in model_lower for marker in ("vla", "robot", "gr00t", "pi05", "openvla", "spatialvla")):
        return "vla"

    model_type = str(cfg.get("model_type", "")).lower()
    if model_type:
        family = _model_type_to_family(model_type, str(model))
        if family is not None:
            return family

    diffusers_family = _diffusers_family(str(model))
    if diffusers_family is not None:
        return diffusers_family

    vlm_markers = (
        "vision_config", "visual_config", "image_token", "image_processor",
        "vision_feature", "multi_modal", "multimodal",
    )
    if any(marker in cfg_text for marker in vlm_markers):
        return "vlm"

    family = _model_type_to_family(model_type, str(model)) if model_type else None
    if family is not None:
        return family

    # Preserve the previous behavior for unknown text-generation-style repos: if
    # no strong non-LLM signal exists, default to LLM so users can still provide
    # explicit module/tokenizer hints.
    return "llm"


def is_vision_encoder_model_type(model_type: str) -> bool:
    """Return whether a model_type is a vision encoder family."""
    return model_type.lower() in _VISION_ENCODER_TYPES

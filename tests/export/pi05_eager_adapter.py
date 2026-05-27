# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5 PyTorch eager adapter for ``benchmark_edge_vs_eager.py``.

The benchmark harness keeps model-family code behind a small adapter hook so
the Edge subprocess benchmark and the PyTorch eager benchmark can share one
report format.  This adapter loads the LeRobot PI0.5 policy once, builds a
minimal observation from the same request JSON used by ``action_inference``,
and times repeated ``select_action`` calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace
import inspect

import torch


def _import_object(path: str) -> Any:
    """Import ``module.Class`` or ``module:Class``."""
    import importlib

    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _first_request_content(input_file: str) -> tuple[str, str | None]:
    """Return the first text prompt and image path from an action JSON file."""
    data = json.loads(Path(input_file).read_text(encoding="utf-8"))
    request = data["requests"][0]
    prompt = "What action should the robot take?"
    image_path = None
    for message in request.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, str):
            prompt = content
            continue
        for item in content:
            if item.get("type") == "text":
                prompt = item.get("text", prompt)
            elif item.get("type") == "image":
                image_path = item.get("image")
    return prompt, image_path


def _image_tensor(path: str | None, *, device: torch.device, size: int = 224) -> torch.Tensor:
    """Load an RGB image as a ``[1, 3, H, W]`` float tensor in [0, 1]."""
    if path is None:
        return torch.zeros(1, 3, size, size, device=device, dtype=torch.float32)

    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _feature_items(config: Any) -> dict[str, Any]:
    """Return PI0.5 input feature specs from config object or dictionary."""
    features = getattr(config, "input_features", None)
    if features is None and isinstance(config, dict):
        features = config.get("input_features")
    return dict(features or {})


def _feature_type(spec: Any) -> str:
    value = getattr(spec, "type", None)
    if value is None and isinstance(spec, dict):
        value = spec.get("type")
    return str(value)


def _feature_shape(spec: Any) -> tuple[int, ...]:
    shape = getattr(spec, "shape", None)
    if shape is None and isinstance(spec, dict):
        shape = spec.get("shape")
    return tuple(int(dim) for dim in (shape or ()))


def _build_batch(policy: Any, input_file: str, device: torch.device) -> dict[str, Any]:
    """Build the minimal LeRobot observation batch used by PI0.5."""
    prompt, image_path = _first_request_content(input_file)
    config = getattr(policy, "config", getattr(getattr(policy, "model", None), "config", None))
    features = _feature_items(config)

    image = _image_tensor(image_path, device=device)
    batch: dict[str, Any] = {"task": [prompt]}

    for name, spec in features.items():
        kind = _feature_type(spec).upper()
        shape = _feature_shape(spec)
        if kind == "VISUAL":
            batch[name] = image
        elif kind == "STATE":
            state_shape = (1, *(shape or (32,)))
            batch[name] = torch.zeros(state_shape, device=device, dtype=torch.float32)

    if not any(_feature_type(spec).upper() == "VISUAL" for spec in features.values()):
        batch["observation.images.base_0_rgb"] = image
    if not any(_feature_type(spec).upper() == "STATE" for spec in features.values()):
        batch["observation.state"] = torch.zeros(1, 32, device=device, dtype=torch.float32)
    return batch


def _add_language_tokens(batch: dict[str, Any], policy: Any, *, device: torch.device) -> None:
    """Add PI0.5 language token tensors expected by select_action."""
    prompt = batch.get("task", ["What action should the robot take?"])[0]
    config = getattr(policy, "config", getattr(getattr(policy, "model", None), "config", None))
    max_length = int(getattr(config, "tokenizer_max_length", 200) or 200)

    tokenizer = (
        getattr(policy, "tokenizer", None)
        or getattr(getattr(policy, "model", None), "tokenizer", None)
        or getattr(getattr(getattr(policy, "model", None), "paligemma_with_expert", None), "tokenizer", None)
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    encoded = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch["observation.language.tokens"] = encoded["input_ids"].to(device=device, dtype=torch.long)
    batch["observation.language.attention_mask"] = encoded["attention_mask"].to(device=device, dtype=torch.bool)


def _patch_lerobot_create_causal_mask() -> None:
    """Filter newer LeRobot causal-mask kwargs on older helper signatures."""
    for module_name in ("lerobot.policies.pi_gemma", "lerobot.policies.pi05.modeling_pi05"):
        try:
            module = __import__(module_name, fromlist=["create_causal_mask"])
        except Exception:
            continue
        fn = getattr(module, "create_causal_mask", None)
        if fn is None or getattr(fn, "_edgellm_kw_filter_patch", False):
            continue
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            continue
        accepted = set(signature.parameters)

        def wrapped_create_causal_mask(*call_args: Any, __fn=fn, __accepted=accepted, **call_kwargs: Any):
            if "input_embeds" in __accepted and "input_embeds" not in call_kwargs and "inputs_embeds" in call_kwargs:
                call_kwargs["input_embeds"] = call_kwargs["inputs_embeds"]
            filtered_kwargs = {key: value for key, value in call_kwargs.items() if key in __accepted}
            return __fn(*call_args, **filtered_kwargs)

        wrapped_create_causal_mask._edgellm_kw_filter_patch = True
        setattr(module, "create_causal_mask", wrapped_create_causal_mask)


def _patch_paligemma_image_features(policy: Any) -> None:
    """Normalize PI0.5 PaliGemma image-feature output across Transformers versions."""
    core = getattr(policy, "model", policy)
    owner = getattr(getattr(core, "paligemma_with_expert", None), "paligemma", None)
    target = getattr(owner, "model", None)
    get_image_features = getattr(target, "get_image_features", None)
    if target is None or not callable(get_image_features):
        return
    if getattr(target, "_edgellm_pooler_output_patch", False):
        return

    def wrapped_get_image_features(*call_args: Any, **call_kwargs: Any):
        output = get_image_features(*call_args, **call_kwargs)
        if isinstance(output, torch.Tensor):
            return SimpleNamespace(pooler_output=output, last_hidden_state=output)
        if not hasattr(output, "pooler_output") and isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return SimpleNamespace(pooler_output=first, last_hidden_state=first)
        return output

    target.get_image_features = wrapped_get_image_features
    target._edgellm_pooler_output_patch = True


def _root_model_from_loaded(loaded: Any) -> Any:
    """Return the root eager policy from the benchmark-loaded model bundle."""
    if loaded is None:
        raise ValueError("PI0.5 adapter requires the benchmark to pass loaded= from HFModelSource/load_vla_model")
    model = getattr(loaded, "model", loaded)
    if isinstance(model, dict):
        if "root" in model:
            return model["root"]
        if "model" in model:
            return model["model"]
        return next(iter(model.values()))
    return model


def _model_device(model: Any) -> torch.device:
    """Best-effort device lookup for a loaded eager model."""
    try:
        return next(iter(model.parameters())).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_runner(args: Any, loaded: Any = None, source: Any = None, family: str | None = None):
    """Create a PI0.5 eager runner from the benchmark-loaded HF model."""
    del source, family
    _patch_lerobot_create_causal_mask()
    policy = _root_model_from_loaded(loaded)
    device = _model_device(policy)
    policy.eval()
    _patch_paligemma_image_features(policy)
    base_batch = _build_batch(policy, args.input_file, device)
    _add_language_tokens(base_batch, policy, device=device)

    def create_inputs() -> dict[str, Any]:
        """Return a fresh batch for each timed eager iteration."""
        cloned: dict[str, Any] = {}
        for key, value in base_batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        return cloned

    def run_once(*, iteration: int, phase: str, output_path: Path, seed: int, input_file: str):
        del iteration, phase, input_file
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        batch = create_inputs()
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    output = policy.select_action(batch) if hasattr(policy, "select_action") else policy(batch)
            else:
                output = policy.select_action(batch) if hasattr(policy, "select_action") else policy(batch)
        if isinstance(output, torch.Tensor):
            payload: dict[str, Any] = {
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
            }
        else:
            payload = {"output_type": type(output).__name__, "output_repr": repr(output)[:500]}
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    return run_once

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central VLA eager adapter for ``benchmark_edge_vs_eager.py``.

This file keeps the benchmark CLI stable while still allowing model-family
specific eager setup.  It uses the same HF source and role-resolution helpers as
the Edge export path, then dispatches to the narrow adapters we already know how
to run.  A small generic VLA fallback handles Hugging Face models with standard
processor + ``predict_action`` / ``generate`` APIs.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.edgellm.contracts.action_contracts import (
    ACTION_CONTRACT_PREFIX_KV_FLOW_STEP,
    ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP,
)
from tools.edgellm.hf_export.resolution import HFRoleResolutionPlan, resolve_hf_roles
from tools.edgellm.hf_export.source import HFModelSource


def _root_model_from_loaded(loaded: Any) -> Any:
    """Return the root eager model from the benchmark-loaded bundle."""
    model = getattr(loaded, "model", loaded)
    if isinstance(model, dict):
        if "root" in model:
            return model["root"]
        if "model" in model:
            return model["model"]
        return next(iter(model.values()))
    return model


def _ensure_source(args: Any, source: Any) -> HFModelSource:
    """Use the provided source or rebuild it from benchmark args."""
    if isinstance(source, HFModelSource):
        return source
    return HFModelSource.from_args(args, model_attr="model")


def _resolve_plan(source: HFModelSource) -> HFRoleResolutionPlan:
    """Resolve all VLA roles; fall back to source defaults if a model is partial."""
    try:
        return resolve_hf_roles(source, roles=("language", "visual", "action"))
    except Exception:
        return resolve_hf_roles(source)


def _select_adapter(root_model: Any, source: HFModelSource, plan: HFRoleResolutionPlan) -> str:
    """Choose the most specific eager adapter available for this VLA model."""
    model_name = str(source.model).lower()
    type_name = f"{type(root_model).__module__}.{type(root_model).__name__}".lower()
    action_contract = plan.selected.get("action").contract if "action" in plan.selected else ""

    if (
        action_contract == ACTION_CONTRACT_PREFIX_KV_FLOW_STEP
        or "pi05" in model_name
        or "pi05" in type_name
        or hasattr(root_model, "select_action")
    ):
        return "pi05"

    if (
        action_contract == ACTION_CONTRACT_STATE_CONDITIONED_FLOW_STEP
        or "gr00t" in model_name
        or "gr00t" in type_name
        or (hasattr(root_model, "get_action") and hasattr(root_model, "action_head"))
    ):
        return "gr00t"

    return "generic"


def _request_content(input_file: str | Path) -> tuple[str, str | None]:
    """Return the first prompt and image path from an Edge request JSON."""
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
    return prompt, _resolve_path(image_path)


def _resolve_path(path: str | None) -> str | None:
    """Resolve container paths when invoked from a host checkout."""
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if path.startswith("/workspace/"):
        scratch_candidate = Path("/mnt/scratch/workspace") / path.removeprefix("/workspace/")
        if scratch_candidate.exists():
            return str(scratch_candidate)
    return path


def _load_image(path: str | None):
    """Load a PIL RGB image or create a black placeholder."""
    from PIL import Image

    if path is None:
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    return Image.open(path).convert("RGB")


def _processor_from_loaded(loaded: Any, source: HFModelSource) -> Any:
    """Return the loaded processor or load a generic HF processor."""
    processor = getattr(loaded, "processor", None)
    if processor is not None:
        return processor

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(source.model, trust_remote_code=source.trust_remote_code)


def _model_device(model: Any) -> torch.device:
    try:
        return next(iter(model.parameters())).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_tensor_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
    """Move tensor-like processor outputs to the model device."""
    result: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _generic_vla_inputs(processor: Any, input_file: str | Path, device: torch.device) -> dict[str, Any]:
    """Build generic VLA inputs following Naren's processor-first strategy."""
    prompt, image_path = _request_content(input_file)
    image = _load_image(image_path)

    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return _move_tensor_inputs(
                processor(text=[text], images=[image], return_tensors="pt", padding=True),
                device,
            )
        except Exception:
            pass

    try:
        return _move_tensor_inputs(processor(prompt, image, return_tensors="pt"), device)
    except Exception:
        return _move_tensor_inputs(
            processor(text=[prompt], images=[image], return_tensors="pt", padding=True),
            device,
        )


def _summarize_output(output: Any) -> dict[str, Any]:
    """Write a compact JSON-friendly output summary."""
    if isinstance(output, torch.Tensor):
        return {
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_mean": float(output.float().mean().item()) if output.numel() else 0.0,
        }
    if isinstance(output, np.ndarray):
        return {
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_mean": float(output.mean()) if output.size else 0.0,
        }
    if isinstance(output, dict) or hasattr(output, "items"):
        payload: dict[str, Any] = {}
        for key, value in output.items():
            payload[str(key)] = _summarize_output(value)
        return payload
    return {"output_type": type(output).__name__, "output_repr": repr(output)[:500]}



# PI0.5 eager runner ---------------------------------------------------------

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


def _create_pi05_runner(args: Any, loaded: Any = None, source: Any = None, family: str | None = None):
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
        payload = _summarize_output(output)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    return run_once


# GR00T eager runner ---------------------------------------------------------

def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_path(path: str | None) -> str | None:
    """Resolve container paths when the adapter is run from the host checkout."""
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    if path.startswith("/workspace/"):
        scratch_candidate = Path("/mnt/scratch/workspace") / path.removeprefix("/workspace/")
        if scratch_candidate.exists():
            return str(scratch_candidate)
    return path


def _first_request(input_file: str | Path) -> dict[str, Any]:
    data = _load_json(input_file)
    requests = data.get("requests") or []
    if not requests:
        raise ValueError(f"No requests found in {input_file}")
    return requests[0]


def _prompt_and_image(request: dict[str, Any]) -> tuple[str, str | None]:
    """Extract the first text prompt and image path from an Edge request."""
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
    return prompt, _resolve_path(image_path)


def _load_rgb_image(path: str | None):
    """Load a PIL RGB image, or create a black image if the request has none."""
    from PIL import Image

    if path is None:
        return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    return Image.open(path).convert("RGB")


def _root_model_from_loaded(loaded: Any) -> Any:
    if loaded is None:
        raise ValueError("GR00T adapter requires loaded= from HFModelSource/load_vla_model")
    model = getattr(loaded, "model", loaded)
    if isinstance(model, dict):
        if "root" in model:
            return model["root"]
        if "model" in model:
            return model["model"]
        return next(iter(model.values()))
    return model


def _load_processor(loaded: Any, source: Any) -> Any:
    processor = getattr(loaded, "processor", None)
    if processor is not None and hasattr(processor, "collator"):
        return processor

    import gr00t.model  # noqa: F401  # registers GR00T AutoProcessor/AutoModel classes
    from transformers import AutoProcessor

    extra = getattr(loaded, "extra", {}) or {}
    model_dir = getattr(source, "model", None) or extra.get("model_dir")
    if not model_dir:
        raise ValueError("Could not resolve GR00T processor path from benchmark source")
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


def _request_embodiment_tag(request: dict[str, Any], processor: Any):
    """Choose an EmbodimentTag from explicit tag or numeric embodiment id."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    modality_configs = getattr(processor, "modality_configs", {}) or {}

    def resolve_if_supported(tag: str):
        try:
            member = EmbodimentTag.resolve(tag)
        except ValueError:
            return None
        if member.value in modality_configs:
            return member
        return None

    explicit_tag = request.get("embodiment_tag")
    if explicit_tag:
        tag = resolve_if_supported(str(explicit_tag))
        if tag is None:
            raise ValueError(f"GR00T embodiment_tag {explicit_tag!r} is not supported by this processor")
        return tag

    embodiment_id = request.get("embodiment_id")
    mapping = getattr(processor, "embodiment_id_mapping", {}) or {}
    if embodiment_id is not None:
        for tag, mapped_id in mapping.items():
            if int(mapped_id) == int(embodiment_id) and tag in modality_configs:
                resolved = resolve_if_supported(tag)
                if resolved is not None:
                    return resolved

    for fallback in (
        "real_g1_relative_eef_relative_joints",
        "xdof_relative_eef_relative_joint",
        "xdof_relative_eef_relative_joint_subtask",
        "oxe_droid_relative_eef_relative_joint",
        "real_r1_pro_sharpa_relative_eef",
        "real_r1_pro_sharpa_relative_eef_human",
        "real_r1_pro_sharpa_relative_eef_maxinsights",
        "real_r1_pro_sharpa_relative_eef_mecka",
        "simpler_env_google",
        "new_embodiment",
    ):
        resolved = resolve_if_supported(fallback)
        if resolved is not None:
            return resolved

    for tag in modality_configs:
        resolved = resolve_if_supported(str(tag))
        if resolved is not None:
            return resolved
    available = ", ".join(sorted(str(tag) for tag in modality_configs)[:8])
    raise ValueError(f"Could not infer a supported GR00T embodiment tag; processor tags include: {available}")


def _flat_state_values(request: dict[str, Any]) -> np.ndarray:
    raw_state = request.get("state", [])
    array = np.asarray(raw_state, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return array.reshape(-1)


def _norm_param_dim(processor: Any, embodiment_value: str, key: str, default: int) -> int:
    try:
        dim_value = processor.state_action_processor.norm_params[embodiment_value]["state"][key]["dim"]
        return int(np.asarray(dim_value).item())
    except Exception:
        return int(default)


def _split_state(request: dict[str, Any], processor: Any, embodiment_tag: Any) -> dict[str, np.ndarray]:
    """Split the Edge flat state vector into GR00T's configured state groups."""
    modality_cfg = processor.modality_configs[embodiment_tag.value]["state"]
    state_keys = list(modality_cfg.modality_keys)
    horizon = len(modality_cfg.delta_indices)
    flat_state = _flat_state_values(request)
    default_dim = max(1, int(np.ceil(max(1, flat_state.size) / max(1, len(state_keys)))))

    offset = 0
    states: dict[str, np.ndarray] = {}
    for key in state_keys:
        dim = _norm_param_dim(processor, embodiment_tag.value, key, default_dim)
        values = flat_state[offset : offset + dim]
        offset += dim
        if values.size < dim:
            values = np.pad(values, (0, dim - values.size))
        states[key] = np.tile(values.astype(np.float32, copy=False), (horizon, 1))
    return states


def _build_vla_step(request: dict[str, Any], processor: Any):
    from gr00t.data.types import VLAStepData

    prompt, image_path = _prompt_and_image(request)
    image = _load_rgb_image(image_path)
    embodiment_tag = _request_embodiment_tag(request, processor)

    video_cfg = processor.modality_configs[embodiment_tag.value]["video"]
    image_horizon = len(video_cfg.delta_indices)
    images = {key: [image.copy() for _ in range(image_horizon)] for key in video_cfg.modality_keys}
    states = _split_state(request, processor, embodiment_tag)

    return VLAStepData(
        images=images,
        states=states,
        actions={},
        text=prompt,
        embodiment=embodiment_tag,
    )


def _rec_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    """Match GR00T policy behavior: convert floating tensors to model dtype."""
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_dtype(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_dtype(item, dtype) for item in value]
    return value


def _model_dtype(model: Any) -> torch.dtype:
    try:
        return next(iter(model.parameters())).dtype
    except Exception:
        return torch.bfloat16


def _trajectory_from_array(value: Any) -> list[list[float]] | None:
    """Extract a first-batch ``[T, 2]`` trajectory from tensor/array action output."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        try:
            array = np.asarray(value, dtype=np.float32)
        except Exception:
            return None
    if array.size == 0:
        return None
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        if array.shape[0] < 2:
            return None
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[-1] < 2:
        return None
    return array[:, :2].astype(float).tolist()


def _trajectory_from_output(output: Any) -> list[list[float]] | None:
    """Find a trajectory-like action tensor in native eager model outputs."""
    direct = _trajectory_from_array(output)
    if direct is not None:
        return direct
    if isinstance(output, dict) or hasattr(output, "items"):
        for key in ("output_trajectory", "trajectory", "action_pred", "actions", "action", "output"):
            if key in output:
                trajectory = _trajectory_from_output(output[key])
                if trajectory is not None:
                    return trajectory
    return None


def _summarize_output(output: Any) -> dict[str, Any]:
    trajectory = _trajectory_from_output(output)

    def summarize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "mean": float(value.float().mean().item()) if value.numel() else 0.0,
            }
        if isinstance(value, np.ndarray):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "mean": float(value.mean()) if value.size else 0.0,
            }
        return repr(value)[:300]

    if isinstance(output, dict) or hasattr(output, "items"):
        payload = {str(key): summarize(value) for key, value in output.items()}
    else:
        payload = {"output": summarize(output)}
    if trajectory is not None:
        payload["output_trajectory"] = trajectory
    return payload

def _create_gr00t_runner(args: Any, loaded: Any = None, source: Any = None, family: str | None = None):
    """Create a full GR00T eager runner for benchmark iterations."""
    del family
    model = _root_model_from_loaded(loaded)
    processor = _load_processor(loaded, source)
    processor.eval()
    model.eval()

    request = _first_request(args.input_file)
    vla_step = _build_vla_step(request, processor)

    def make_inputs() -> Any:
        from gr00t.data.types import MessageType

        processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step}])
        collated = processor.collator([processed])
        return _rec_to_dtype(collated, _model_dtype(model))

    def run_once(*, iteration: int, phase: str, output_path: Path, seed: int, input_file: str):
        del iteration, phase, input_file
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inputs = make_inputs()
        with torch.inference_mode():
            output = model.get_action(**inputs)
        output_path.write_text(json.dumps(_summarize_output(output), indent=2), encoding="utf-8")

    return run_once

def _create_generic_runner(args: Any, loaded: Any, source: HFModelSource):
    """Create a fallback runner for processor-based VLA models."""
    model = _root_model_from_loaded(loaded)
    processor = _processor_from_loaded(loaded, source)
    device = _model_device(model)
    model.eval()
    inputs = _generic_vla_inputs(processor, args.input_file, device)

    def run_once(*, iteration: int, phase: str, output_path: Path, seed: int, input_file: str):
        del iteration, phase, input_file
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            if hasattr(model, "predict_action"):
                try:
                    output = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                except TypeError:
                    output = model.predict_action(**inputs, do_sample=False)
            elif hasattr(model, "generate"):
                output = model.generate(
                    **inputs,
                    max_new_tokens=int(getattr(args, "max_generate_length", 16) or 16),
                    do_sample=False,
                )
            else:
                output = model(**inputs)
        payload = _summarize_output(output)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    return run_once


def create_runner(args: Any, loaded: Any = None, source: Any = None, family: str | None = None):
    """Create a VLA eager runner by dispatching to the best model-specific path."""
    del family
    if loaded is None:
        raise ValueError("vla_eager_adapter requires loaded= from benchmark_edge_vs_eager.py")

    source_obj = _ensure_source(args, source)
    plan = _resolve_plan(source_obj)
    root_model = _root_model_from_loaded(loaded)
    adapter = _select_adapter(root_model, source_obj, plan)

    action_contract = plan.selected.get("action").contract if "action" in plan.selected else "unresolved"
    print(
        f"[vla_eager_adapter] family={plan.family} adapter={adapter} "
        f"action_contract={action_contract}"
    )

    if adapter == "pi05":
        return _create_pi05_runner(args=args, loaded=loaded, source=source_obj, family=plan.family)

    if adapter == "gr00t":
        return _create_gr00t_runner(args=args, loaded=loaded, source=source_obj, family=plan.family)

    return _create_generic_runner(args, loaded, source_obj)

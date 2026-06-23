# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Hugging Face source/strategy description for Edge exports.

This is the convergence point between the existing Edge component exporters and
the run_hf-style strategy model. It carries both: 
  * Hugging Face model retrieval / component-selection hints, and
  * generic strategy/compile/input knobs used by the upstream HF strategy tool.

The component exporters do not need every field today, but keeping the full
source context here prevents each exporter from growing a slightly different HF
configuration surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

from .detect import detect_family

Precision = Literal["FP16", "BF16", "FP32"]
Mode = Literal["export", "compile"]
KVCache = Literal["static_v1", "static_v2", "hf_static"]
EngineFormat = Literal["exported_program", "torchscript", "aot_inductor"]

HF_STRATEGY_FAMILIES = (
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
)


def _getattr_any(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-empty attribute from ``obj``.

    This lets one source object be built from several CLIs whose
    flags may use slightly different names, for example ``model``
    versus ``model_dir``.
    """
    for name in names:
        value = getattr(obj, name, default)
        if value not in (None, ""):
            return value
    return default


def _tuple_from_arg(value: Any) -> tuple[str, ...]:
    """Normalize comma-separated or repeated CLI role values.

    Argparse can produce ``None``, a string, or a sequence depending
    on how a flag is declared. Strategies only need a tuple.
    """
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence):
        items: list[str] = []
        for item in value:
            items.extend(part.strip() for part in str(item).split(",") if part.strip())
        return tuple(items)
    return (str(value),)


def _normalize_object_path(value: Any) -> Any:
    """Normalize object import hints to dotted paths.

    Some Torch/Python utilities accept ``module:Class`` while the legacy
    Edge-LLM exporters expect ``module.Class``. Keep the CLI tolerant and
    pass a single dotted form downstream.
    """
    if isinstance(value, str) and ":" in value:
        module_name, object_name = value.split(":", 1)
        if module_name and object_name:
            return f"{module_name}.{object_name}"
    return value


@dataclass(frozen=True)
class HFModelSource:
    """Reusable HF checkpoint plus run_hf-style strategy context.

    ``model`` is the HF model id or local model directory. The remaining fields
    are intentionally broad: the same object is used by direct-HF strategies and
    by the legacy component CLIs, so model loading and compile options stay in
    one schema.
    """

    # ---- Identity / family detection ----
    model: str
    task: Optional[str] = None
    family: Optional[str] = None
    roles: tuple[str, ...] = ()

    # ---- HF/custom model retrieval hints ----
    model_class: Optional[str] = None
    tokenizer: Optional[str] = None
    processor_model: Optional[str] = None
    trust_remote_code: bool = True
    low_cpu_mem_usage: Optional[bool] = None
    use_safetensors: Optional[bool] = None
    ignore_mismatched_sizes: Optional[bool] = None
    use_cache: Optional[bool] = None

    # ---- Edge component-selection hints ----
    language_module: Optional[str] = None
    lm_head_module: Optional[str] = None
    vision_module: Optional[str] = None
    projector_module: Optional[str] = None
    action_module: Optional[str] = None
    instantiate_from_config: bool = False
    no_processor: bool = False
    add_common_vlm_aliases: bool = False

    # ---- Strategy / precision / compile knobs from tools/hf RunConfig ----
    precision: Precision = "FP16"
    dtype: Optional[str] = None
    autocast: bool = False
    mode: Mode = "export"
    batch_size: int = 1
    iterations: int = 10
    min_block_size: int = 1
    optimization_level: Optional[int] = None
    offload_module_to_cpu: bool = False
    engine_cache_dir: Optional[str] = None
    debug: bool = False
    save_engine: Optional[str] = None
    engine_format: EngineFormat = "exported_program"
    save_trt_engine: Optional[str] = None
    save_exported_program: Optional[str] = None

    # ---- LLM-specific strategy knobs ----
    isl: int = 128
    num_tokens: int = 64
    cache: Optional[KVCache] = None
    prompt: str = "What is parallel programming?"
    attn_implementation: Optional[str] = "eager"

    # ---- Vision / diffusion / audio strategy knobs ----
    image_size: int = 512
    num_inference_steps: int = 20
    num_frames: int = 16
    audio_duration_s: float = 30.0
    visual_prompt: str = "Describe this image."
    vla_instruction: str = "pick up the red block and place it on the blue plate"

    # ---- Benchmark / generation / accuracy knobs ----
    benchmark: bool = False
    generate: bool = False
    accuracy: bool = False
    inductor: bool = False
    json_out: Optional[str] = None
    accuracy_atol: float = 1e-2
    accuracy_rtol: float = 1e-2
    accuracy_cos_sim_min: float = 0.99

    # ---- Edge output / runtime knobs ----
    output_root: Optional[str] = None
    llm_output_dir: Optional[str] = None
    visual_output_dir: Optional[str] = None
    action_output_dir: Optional[str] = None
    max_kv_cache_capacity: int = 4096
    action_batch_size: int = 2

    # ---- Escape hatch for strategy-specific fields not yet promoted ----
    extra_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: Any, *, model_attr: str = "model") -> "HFModelSource":
        """Build a source description from an argparse namespace.

        This is the boundary between command-line flags and the
        reusable HF/export schema. After this point, code should pass
        ``HFModelSource`` around instead of reading argparse fields.
        """
        model = _getattr_any(args, model_attr, "model", "model_dir")
        if not model:
            raise ValueError(f"Could not build HFModelSource: missing {model_attr!r}")

        return cls(
            model=str(model),
            task=_getattr_any(args, "task"),
            family=_getattr_any(args, "family"),
            roles=_tuple_from_arg(_getattr_any(args, "role", "roles")),
            model_class=_normalize_object_path(_getattr_any(args, "model_class")),
            tokenizer=_getattr_any(args, "tokenizer"),
            processor_model=_getattr_any(args, "processor_model"),
            trust_remote_code=bool(getattr(args, "trust_remote_code", True)),
            low_cpu_mem_usage=_getattr_any(args, "low_cpu_mem_usage"),
            use_safetensors=_getattr_any(args, "use_safetensors"),
            ignore_mismatched_sizes=_getattr_any(args, "ignore_mismatched_sizes"),
            use_cache=_getattr_any(args, "use_cache"),
            language_module=_getattr_any(args, "language_module"),
            lm_head_module=_getattr_any(args, "lm_head_module"),
            vision_module=_getattr_any(args, "vision_module"),
            projector_module=_getattr_any(args, "projector_module"),
            action_module=_getattr_any(args, "action_module"),
            instantiate_from_config=bool(getattr(args, "instantiate_from_config", False)),
            no_processor=bool(getattr(args, "no_processor", False)),
            add_common_vlm_aliases=bool(getattr(args, "add_common_vlm_aliases", False)),
            precision=_getattr_any(args, "precision", default="FP16"),
            dtype=_getattr_any(args, "dtype"),
            autocast=bool(getattr(args, "autocast", False)),
            mode=_getattr_any(args, "mode", default="export"),
            batch_size=int(_getattr_any(args, "batch_size", "batchSize", default=1)),
            iterations=int(_getattr_any(args, "iterations", default=10)),
            min_block_size=int(_getattr_any(args, "min_block_size", default=1)),
            optimization_level=_getattr_any(args, "optimization_level"),
            offload_module_to_cpu=bool(getattr(args, "offload_module_to_cpu", False)),
            engine_cache_dir=_getattr_any(args, "engine_cache_dir"),
            debug=bool(getattr(args, "debug", False)),
            save_engine=_getattr_any(args, "save_engine"),
            engine_format=_getattr_any(args, "engine_format", default="exported_program"),
            save_trt_engine=_getattr_any(args, "save_trt_engine"),
            save_exported_program=_getattr_any(args, "save_exported_program"),
            isl=int(_getattr_any(args, "isl", default=128)),
            num_tokens=int(_getattr_any(args, "num_tokens", default=64)),
            cache=_getattr_any(args, "cache"),
            prompt=_getattr_any(args, "prompt", default="What is parallel programming?"),
            attn_implementation=_getattr_any(args, "attn_implementation"),
            image_size=int(_getattr_any(args, "image_size", default=512) or 512),
            num_inference_steps=int(_getattr_any(args, "num_inference_steps", default=20)),
            num_frames=int(_getattr_any(args, "num_frames", default=16)),
            audio_duration_s=float(_getattr_any(args, "audio_duration_s", default=30.0)),
            visual_prompt=_getattr_any(args, "visual_prompt", "prompt", default="Describe this image."),
            vla_instruction=_getattr_any(
                args,
                "vla_instruction",
                default="pick up the red block and place it on the blue plate",
            ),
            benchmark=bool(getattr(args, "benchmark", False)),
            generate=bool(getattr(args, "generate", False)),
            accuracy=bool(getattr(args, "accuracy", False)),
            inductor=bool(getattr(args, "inductor", False)),
            json_out=_getattr_any(args, "json_out"),
            accuracy_atol=float(_getattr_any(args, "accuracy_atol", default=1e-2)),
            accuracy_rtol=float(_getattr_any(args, "accuracy_rtol", default=1e-2)),
            accuracy_cos_sim_min=float(_getattr_any(args, "accuracy_cos_sim_min", default=0.99)),
            output_root=_getattr_any(args, "output_root"),
            llm_output_dir=_getattr_any(args, "llm_output_dir"),
            visual_output_dir=_getattr_any(args, "visual_output_dir"),
            action_output_dir=_getattr_any(args, "action_output_dir"),
            max_kv_cache_capacity=int(_getattr_any(args, "max_kv_cache_capacity", default=4096)),
            action_batch_size=int(_getattr_any(args, "action_batch_size", default=2)),
        )

    def detected_family(self) -> str:
        """Return the requested family or run lightweight detection."""
        if self.family and self.family != "auto":
            return self.family
        return detect_family(self.model, task_override=self.task)

    def normalized_dtype(self) -> Optional[str]:
        """Map run_hf-style precision names to Edge exporter dtype names."""
        if self.dtype:
            return self.dtype
        if self.precision == "FP16":
            return "fp16"
        if self.precision == "BF16":
            return "bf16"
        if self.precision == "FP32":
            return "fp32"
        return None

    def run_config_kwargs(self) -> dict[str, Any]:
        """Return fields compatible with run_hf-style ``RunConfig``.

        Not every field is consumed by the Edge path today. Keeping
        them together makes it easier to converge this tool with the
        broader Torch-TensorRT HF strategy runner later.
        """
        return {
            "model": self.model,
            "task": self.task,
            "precision": self.precision,
            "autocast": self.autocast,
            "mode": self.mode,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "min_block_size": self.min_block_size,
            "optimization_level": self.optimization_level,
            "offload_module_to_cpu": self.offload_module_to_cpu,
            "engine_cache_dir": self.engine_cache_dir,
            "debug": self.debug,
            "save_engine": self.save_engine,
            "engine_format": self.engine_format,
            "save_trt_engine": self.save_trt_engine,
            "save_exported_program": self.save_exported_program,
            "isl": self.isl,
            "num_tokens": self.num_tokens,
            "cache": self.cache,
            "prompt": self.prompt,
            "image_size": self.image_size,
            "num_inference_steps": self.num_inference_steps,
            "num_frames": self.num_frames,
            "audio_duration_s": self.audio_duration_s,
            "inductor": self.inductor,
            "accuracy_atol": self.accuracy_atol,
            "accuracy_rtol": self.accuracy_rtol,
            "accuracy_cos_sim_min": self.accuracy_cos_sim_min,
            **dict(self.extra_options),
        }

    def language_kwargs(self) -> dict[str, Any]:
        """Return only the fields expected by the LLM exporter."""
        return {
            "model_dir": self.model,
            "model_class": self.model_class,
            "language_module": self.language_module,
            "lm_head_module": self.lm_head_module,
            "instantiate_from_config": self.instantiate_from_config,
            "tokenizer": self.tokenizer,
            "attn_implementation": self.attn_implementation,
        }

    def visual_kwargs(self) -> dict[str, Any]:
        """Return only the fields expected by the visual exporter."""
        return {
            "model_dir": self.model,
            "model_class": self.model_class,
            "vision_module": self.vision_module,
            "projector_module": self.projector_module,
            "processor_model": self.processor_model,
            "no_processor": self.no_processor,
            "add_common_vlm_aliases": self.add_common_vlm_aliases,
            "attn_implementation": self.attn_implementation,
        }

    def action_kwargs(self) -> dict[str, Any]:
        """Return only the fields expected by action-role exporters."""
        return {
            "model_dir": self.model,
            "model_class": self.model_class,
            "attn_implementation": self.attn_implementation,
        }

    def vla_loader_kwargs(self) -> dict[str, Any]:
        """Return model-loading fields for the VLA eager loader hook."""
        return {
            "model_dir": self.model,
            "model_class": self.model_class,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
        }

    def to_manifest_metadata(self) -> dict[str, Any]:
        """Serialize source information into manifest metadata.

        This makes generated manifests explain where they came from
        and which knobs were used, which is useful after engines have
        been copied away from the original export command.
        """
        data = asdict(self)
        data.pop("metadata", None)
        data.pop("extra_options", None)
        return {
            "hf_source": data,
            **dict(self.metadata),
        }

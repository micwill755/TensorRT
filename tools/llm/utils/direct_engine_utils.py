"""Shared helpers for direct serialized TensorRT engine workflows.

The helpers in this file are intentionally model-agnostic. Component-specific
exporters own model loading, contract selection, and example tensor creation;
this module owns the common TensorRT mechanics:

    ExportedProgram + input specs -> serialized engine bytes
    serialized engine bytes/path -> manifest metadata
    serialized engine path -> pure TensorRT Python runner
"""

from __future__ import annotations

import ctypes
import gc
import json
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resolve_trt_device(device: torch.device) -> Any:
    import torch_tensorrt

    if device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return torch_tensorrt.Device(f"cuda:{device_index}")
    return torch_tensorrt.Device(str(device))


def make_trt_input_specs(value: Any) -> Any:
    """Mirror a tensor tree with static Torch-TensorRT Input specs."""
    import torch_tensorrt

    if isinstance(value, torch.Tensor):
        return torch_tensorrt.Input(shape=tuple(value.shape), dtype=value.dtype)
    if isinstance(value, dict):
        return {key: make_trt_input_specs(child) for key, child in value.items()}
    if isinstance(value, list):
        return [make_trt_input_specs(child) for child in value]
    if isinstance(value, tuple):
        return tuple(make_trt_input_specs(child) for child in value)
    return value


@contextmanager
def patch_torchtrt_output_names(output_names: Optional[List[str]]):
    """Temporarily name TensorRT engine outputs for runtime ABI compatibility."""
    if not output_names:
        yield
        return

    import importlib

    trt_interpreter_module = importlib.import_module(
        "torch_tensorrt.dynamo.conversion._TRTInterpreter"
    )
    interpreter_cls = trt_interpreter_module.TRTInterpreter
    original_output = interpreter_cls.output

    def output_with_runtime_names(self: Any, target: str, args: Any, kwargs: Any) -> List[Any]:
        assert len(args) == 1
        if isinstance(args[0], tuple):
            outputs = args[0]
        elif isinstance(args[0], list):
            outputs = tuple(args[0])
        else:
            outputs = (args[0],)

        for output_idx in range(len(outputs)):
            output = outputs[output_idx]
            if not isinstance(output, trt_interpreter_module.trt.ITensor):
                new_output = trt_interpreter_module.get_trt_tensor(
                    self.ctx,
                    output,
                    target,
                )
                outputs = (
                    outputs[:output_idx]
                    + (new_output,)
                    + outputs[output_idx + 1 :]
                )

        if not all(
            isinstance(output, trt_interpreter_module.trt.ITensor)
            for output in outputs
        ):
            raise RuntimeError("TensorRT requires all outputs to be Tensor!")

        if self.output_dtypes is not None and len(self.output_dtypes) != len(outputs):
            raise RuntimeError(
                f"Specified output dtypes ({len(self.output_dtypes)}) differ "
                f"from number of outputs ({len(outputs)})"
            )

        marked_outputs_ids = []
        for i, output in enumerate(outputs):
            if id(output) in marked_outputs_ids:
                continue
            marked_outputs_ids.append(id(output))

            name = output_names[i] if i < len(output_names) else f"output{i}"

            if self.output_dtypes is not None:
                output_dtype = self.output_dtypes[i]
            elif any(
                op_name in output.name.split("_")
                for op_name in (
                    "eq",
                    "gt",
                    "lt",
                    "or",
                    "xor",
                    "and",
                    "not",
                    "ne",
                    "isinf",
                    "isnan",
                    "any",
                )
            ):
                output_dtype = trt_interpreter_module.dtype.b
            else:
                output_dtype = trt_interpreter_module.dtype.unknown

            if output_dtype is not trt_interpreter_module.dtype.unknown:
                trt_output_dtype = output_dtype.to(
                    trt_interpreter_module.trt.DataType,
                    use_default=True,
                )
                if output.dtype != trt_output_dtype:
                    if hasattr(self, "_cast_output_dtype"):
                        output = self._cast_output_dtype(output, trt_output_dtype, name)
                    else:
                        cast_layer = self.ctx.net.add_cast(output, trt_output_dtype)
                        cast_layer.name = (
                            f"Cast output {name} from {output.dtype} "
                            f"to {trt_output_dtype}"
                        )
                        output = cast_layer.get_output(0)

            output.name = name
            outputs = outputs[:i] + (output,) + outputs[i + 1 :]
            self.ctx.net.mark_output(output)
            self._output_names.append(name)

            logger = getattr(trt_interpreter_module, "_LOGGER", None)
            if logger is not None:
                logger.debug(
                    f"Marking output {name} "
                    f"[shape={output.shape}, dtype={output.dtype}]"
                )

        return list(outputs)

    interpreter_cls.output = output_with_runtime_names
    try:
        yield
    finally:
        interpreter_cls.output = original_output


def compile_exported_program_to_serialized_engine(
    exported_program: torch.export.ExportedProgram,
    *,
    arg_inputs: Any,
    kwarg_inputs: Optional[Dict[str, Any]] = None,
    device: torch.device,
    min_block_size: int = 1,
    workspace_size: int = 0,
    optimization_level: Optional[int] = None,
    require_full_compilation: bool = False,
    disable_tf32: bool = True,
    use_fp32_acc: bool = True,
    output_names: Optional[List[str]] = None,
    plugin_loader: Optional[Callable[[], None]] = None,
    use_input_specs: bool = True,
    **extra_compile_kwargs: Any,
) -> bytes:
    """Compile an ExportedProgram to serialized TensorRT engine bytes."""
    from torch_tensorrt.dynamo import convert_exported_program_to_serialized_trt_engine

    if plugin_loader is not None:
        plugin_loader()

    trt_arg_inputs = make_trt_input_specs(arg_inputs) if use_input_specs else arg_inputs
    trt_kwarg_inputs = (
        make_trt_input_specs(kwarg_inputs)
        if use_input_specs and kwarg_inputs is not None
        else kwarg_inputs
    )
    if use_input_specs:
        arg_inputs = None
        kwarg_inputs = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    compile_kwargs = {
        "arg_inputs": trt_arg_inputs,
        "kwarg_inputs": trt_kwarg_inputs or {},
        "device": resolve_trt_device(device),
        "min_block_size": min_block_size,
        "workspace_size": workspace_size,
        "optimization_level": optimization_level,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "use_fp32_acc": use_fp32_acc,
        "use_explicit_typing": True,
        "immutable_weights": True,
        "truncate_double": True,
        **extra_compile_kwargs,
    }
    supported_params = signature(convert_exported_program_to_serialized_trt_engine).parameters
    compile_kwargs = {
        name: value for name, value in compile_kwargs.items()
        if name in supported_params
    }

    with patch_torchtrt_output_names(output_names):
        return convert_exported_program_to_serialized_trt_engine(
            exported_program,
            **compile_kwargs,
        )


def add_direct_engine_manifest_fields(
    manifest: Dict[str, Any],
    *,
    engine_path: Path,
    output_dir: Path,
    engine_bytes: bytes,
    compile_options: Dict[str, Any],
    engine_info: Dict[str, Any],
    custom_op_module: Optional[str],
    plugin_path: Optional[str],
    runtime_requirements_extra: Optional[Dict[str, Any]] = None,
) -> None:
    manifest["artifacts"]["direct_tensorrt_engine"] = relative_or_absolute(
        engine_path,
        output_dir,
    )
    manifest["direct_tensorrt_engine"] = {
        "path": relative_or_absolute(engine_path, output_dir),
        "bytes": len(engine_bytes),
        **engine_info,
    }
    manifest["direct_tensorrt_compile"] = {
        "api": "torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine",
        **compile_options,
    }

    runtime_requirements = {
        "custom_op_module": custom_op_module,
        "tensorrt_plugin_path": plugin_path,
        "load_order": [
            f"import {custom_op_module}" if custom_op_module else None,
            "ctypes.CDLL(tensorrt_plugin_path)" if plugin_path else None,
            "tensorrt.Runtime(...).deserialize_cuda_engine(engine_bytes)",
        ],
    }
    runtime_requirements["load_order"] = [
        item for item in runtime_requirements["load_order"] if item is not None
    ]
    if runtime_requirements_extra:
        runtime_requirements.update(runtime_requirements_extra)
    manifest["runtime_requirements"] = runtime_requirements


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


_TRT_TO_TORCH_DTYPE: Dict[Any, torch.dtype] | None = None


def trt_dtype_to_torch(trt_dtype: Any) -> torch.dtype:
    import tensorrt as trt

    global _TRT_TO_TORCH_DTYPE
    if _TRT_TO_TORCH_DTYPE is None:
        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        if hasattr(trt.DataType, "BF16"):
            dtype_map[trt.DataType.BF16] = torch.bfloat16
        _TRT_TO_TORCH_DTYPE = dtype_map
    return _TRT_TO_TORCH_DTYPE.get(trt_dtype, torch.float32)


class DirectEngineRunner:
    """Pure TensorRT Python runner for a serialized engine file."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        plugin_path: Optional[str] = None,
        device: str | torch.device = "cuda",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        import tensorrt as trt

        if plugin_path:
            ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)

        self.engine_path = Path(engine_path)
        self.device = torch.device(device)
        self.stream = stream
        self._output_cache: Dict[str, torch.Tensor] = {}

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    @property
    def cuda_stream(self) -> int:
        if self.stream is not None:
            return int(self.stream.cuda_stream)
        return int(torch.cuda.current_stream(self.device).cuda_stream)

    def _output_tensor(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        cached = self._output_cache.get(name)
        if cached is None or tuple(cached.shape) != shape or cached.dtype != dtype:
            cached = torch.empty(shape, dtype=dtype, device=self.device)
            self._output_cache[name] = cached
        return cached

    def __call__(self, *inputs: torch.Tensor, **named_inputs: torch.Tensor) -> List[torch.Tensor]:
        if inputs and named_inputs:
            raise ValueError("Pass either positional inputs or named inputs, not both.")

        if named_inputs:
            missing = [name for name in self.input_names if name not in named_inputs]
            if missing:
                raise ValueError(f"Missing engine inputs: {missing}")
            ordered_inputs = [named_inputs[name] for name in self.input_names]
        else:
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs, got {len(inputs)}"
                )
            ordered_inputs = list(inputs)

        bound_inputs: List[torch.Tensor] = []
        for name, tensor in zip(self.input_names, ordered_inputs):
            tensor = tensor.contiguous().to(self.device)
            bound_inputs.append(tensor)
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            self.context.set_input_shape(name, tuple(tensor.shape))

        outputs: List[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            output = self._output_tensor(name, shape)
            self.context.set_tensor_address(name, int(output.data_ptr()))
            outputs.append(output)

        self.context.execute_async_v3(self.cuda_stream)
        torch.cuda.current_stream(self.device).synchronize()
        return outputs

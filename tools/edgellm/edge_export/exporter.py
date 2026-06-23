# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Edge export orchestration.

The command-line front doors should only decide where the model description
comes from: a user-authored eager manifest, or a Hugging Face strategy that
emits one. This object owns the common Edge export lifecycle after that point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tools.edgellm.eager_export.capture import (
    capture_exported_program,
    infer_input_names,
    load_exported_program,
    resolve_role_examples,
    save_exported_program,
)
from tools.edgellm.eager_export.loader import load_eager_model
from tools.edgellm.eager_export.manifest import EagerExportManifest, load_manifest
from tools.edgellm.eager_export.package import (
    call_packager_hook,
    resolve_engine_path,
    resolve_output_dir,
    write_eager_export_summary,
    write_role_contract_manifest,
)
from tools.edgellm.eager_export.roles import resolve_roles


@dataclass(frozen=True)
class EdgeExportOptions:
    """Runtime options for exporting Edge roles from a manifest.

    These are command-time choices, not model description. The
    manifest says what can be exported; options say which roles to
    run now and whether to capture, compile, or only dry-run.
    """

    selected_roles: Optional[Sequence[str]] = None
    output_root: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    dry_run: bool = False
    capture: bool = False
    compile_torchtrt: bool = False
    strict_export: bool = False


@dataclass(frozen=True)
class EdgeRoleExportResult:
    """Artifacts emitted for one role.

    Keeping results structured makes it easy for a CLI, test, or
    future orchestration layer to report where each generated file
    landed.
    """

    name: str
    component: str
    output_dir: Optional[Path] = None
    engine_path: Optional[Path] = None
    exported_program_path: Optional[Path] = None
    contract_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    source: str = ""


def _role_should_compile(role_compile: Optional[bool], global_compile: bool) -> bool:
    """Decide whether this role should compile to a TensorRT engine.

    A role can override the global CLI flag. ``None`` means "inherit
    the command-line default".
    """
    if role_compile is None:
        return global_compile
    return bool(role_compile)


def _capture_role_for_torchtrt(
    module: Any,
    examples: Any,
    *,
    input_names: list[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    strict: bool,
) -> tuple[Any, Any, Any]:
    """Capture a live role and build matching Torch-TRT input specs.

    This keeps live modules on the same ``ExportedProgram`` path as roles that
    arrive with a pre-captured ``.pt2`` file. Dynamic-shape/profile inference is
    still delegated to the Edge-LLM Torch-TRT spec helper so the generated
    engine keeps the runtime profile behavior used by the existing exporter.
    """
    import torch
    from tensorrt_edgellm.onnx_export.torch_trt_utils import (
        _make_export_specs,
        _replace_tensor_leaves,
    )

    export_args = tuple(examples.args)
    export_kwargs = dict(examples.kwargs or {})
    inferred_dynamic_shapes, flat_trt_inputs = _make_export_specs(
        module,
        export_args,
        export_kwargs,
        input_names,
        dynamic_axes=dynamic_axes,
    )
    dynamic_shapes = (
        examples.dynamic_shapes
        if examples.dynamic_shapes is not None
        else inferred_dynamic_shapes
    )

    if hasattr(module, "eval"):
        module.eval()
    with torch.inference_mode():
        try:
            exported_program = torch.export.export(
                module,
                export_args,
                kwargs=export_kwargs or None,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
        except Exception:
            from torch.export._trace import _export

            exported_program = _export(
                module,
                export_args,
                kwargs=export_kwargs or None,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                prefer_deferred_runtime_asserts_over_guards=True,
            )

    trt_input_iter = iter(flat_trt_inputs)
    trt_arg_inputs = _replace_tensor_leaves(export_args, trt_input_iter)
    trt_kwarg_inputs = _replace_tensor_leaves(export_kwargs, trt_input_iter)
    return exported_program, trt_arg_inputs, trt_kwarg_inputs


def _compile_exported_program_torchtrt(
    exported_program: Any,
    examples: Any,
    *,
    engine_path: Path,
    input_names: list[str],
    output_names: list[str],
    trt_arg_inputs: Any = None,
    trt_kwarg_inputs: Any = None,
) -> None:
    """Compile a previously saved ``ExportedProgram`` to TensorRT.

    This is the ExportedProgram-first path: the graph has already been
    captured by PyTorch export, and this function turns it into a
    serialized TensorRT engine with stable input/output names.
    """
    if not examples.args and not examples.kwargs:
        raise ValueError(
            "Compiling an existing ExportedProgram requires example_inputs so "
            "Torch-TRT can build input specs."
        )

    import torch
    import torch_tensorrt
    from torch_tensorrt.dynamo.conversion.edge_plugins import load_edge_plugin
    from tensorrt_edgellm.onnx_export.torch_trt_utils import (
        _make_trt_input_specs,
        _patch_torchtrt_network_names,
        _patch_torchtrt_output_names,
        _resolve_trt_device,
        _supported_compile_kwargs,
    )

    if trt_arg_inputs is None:
        arg_name_hints = list(input_names[:len(examples.args)])
        trt_arg_inputs = _make_trt_input_specs(examples.args, arg_name_hints)
    if trt_kwarg_inputs is None:
        trt_kwarg_inputs = _make_trt_input_specs(examples.kwargs)
    convert_fn = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compile_kwargs = _supported_compile_kwargs(convert_fn, {
        "arg_inputs": trt_arg_inputs,
        "kwarg_inputs": trt_kwarg_inputs or {},
        "device": _resolve_trt_device(device),
        "min_block_size": 1,
        "workspace_size": 0,
        "require_full_compilation": False,
        "disable_tf32": True,
        "use_fp32_acc": True,
        "use_explicit_typing": True,
        "immutable_weights": True,
        "truncate_double": True,
    })

    plugin_path = load_edge_plugin()
    print(f"Saving TensorRT engine to {engine_path}")
    with _patch_torchtrt_output_names(list(output_names)), _patch_torchtrt_network_names():
        trt_engine = convert_fn(exported_program, **compile_kwargs)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(trt_engine)
    print(f"Loaded plugin: {plugin_path}")
    print(f"Saved TensorRT engine to {engine_path}")


class EdgeExport:
    """Export Edge runtime artifacts from an eager-style role manifest.

    ``EdgeExport`` is the shared core used by both front doors:
    direct eager manifests and Hugging Face strategies that generate
    a manifest. It owns the common lifecycle: load model, resolve
    roles, create example inputs, capture/compile, write contracts,
    call packagers, and write summaries.
    """

    def __init__(self, manifest: EagerExportManifest, *, manifest_path: Optional[str | Path] = None):
        """Store the parsed manifest and its source path, if known."""
        self.manifest = manifest
        self.manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else None

    @classmethod
    def from_manifest_path(cls, manifest_path: str | Path) -> "EdgeExport":
        """Load a manifest from JSON and create an exporter."""
        path = Path(manifest_path).expanduser().resolve()
        return cls(load_manifest(path), manifest_path=path)

    def run(self, options: Optional[EdgeExportOptions] = None) -> dict[str, EdgeRoleExportResult]:
        """Run the full export lifecycle for selected roles.

        The same function handles dry-run, capture-only, compile-only,
        and capture+compile flows. Each role is processed independently
        so a manifest can grow from one component to many over time.
        """
        options = options or EdgeExportOptions()
        loaded = load_eager_model(
            self.manifest,
            manifest_path=self.manifest_path,
            device=options.device,
            dtype=options.dtype,
        )
        selected = list(options.selected_roles) if options.selected_roles is not None else None
        # Convert manifest role names like "action" into concrete
        # Python modules or ExportedProgram paths.
        resolved_roles = resolve_roles(
            loaded.model,
            self.manifest,
            selected=selected,
        )

        if loaded.model is None:
            print("No eager model loader configured; using ExportedProgram role(s).")
        else:
            print(f"Loaded eager model: {type(loaded.model).__name__}")
        print(f"Resolved {len(resolved_roles)} role(s): {', '.join(resolved_roles)}")

        if options.dry_run and not options.capture and not options.compile_torchtrt:
            results: dict[str, EdgeRoleExportResult] = {}
            for name, resolved in resolved_roles.items():
                role = resolved.spec
                source = (
                    f"exported_program={role.exported_program}"
                    if role.exported_program
                    else f"module={type(resolved.module).__name__}"
                )
                print(
                    f"  {name}: component={role.component}, "
                    f"{source}, contract={role.contract or '<none>'}"
                )
                results[name] = EdgeRoleExportResult(
                    name=name,
                    component=role.component,
                    input_names=tuple(role.input_names),
                    output_names=tuple(role.output_names),
                    source="exported_program" if role.exported_program else "eager_module",
                )
            return results

        results: dict[str, EdgeRoleExportResult] = {}
        for name, resolved in resolved_roles.items():
            role = resolved.spec
            output_dir = resolve_output_dir(
                self.manifest,
                role,
                output_root=options.output_root,
            )
            if output_dir is None:
                raise ValueError(
                    f"Role {name!r} needs output_dir, outputs.{name}, "
                    "outputs.<component>, or output_root"
                )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Example inputs define the graph signature that Torch export
            # and Torch-TensorRT will see. They are not real runtime data.
            examples = resolve_role_examples(
                self.manifest,
                role,
                resolved.module,
                loaded,
                device=options.device,
                dtype=options.dtype,
            )
            if role.input_names:
                input_names = list(role.input_names)
            elif resolved.module is not None:
                input_names = infer_input_names(resolved.module, examples)
            else:
                input_names = list(examples.input_names)
            output_names = list(role.output_names or examples.output_names)
            dynamic_axes = dict(role.dynamic_axes or examples.dynamic_axes or {})

            source_exported_program_path = (
                Path(role.exported_program).expanduser().resolve()
                if role.exported_program
                else None
            )
            exported_program_path: Optional[Path] = source_exported_program_path
            captured_exported_program: Any = None
            trt_arg_inputs: Any = None
            trt_kwarg_inputs: Any = None
            if source_exported_program_path is not None:
                print(f"Using existing {name} ExportedProgram: {source_exported_program_path}")
            elif options.capture and not _role_should_compile(role.compile_torchtrt, options.compile_torchtrt):
                captured_exported_program = capture_exported_program(
                    resolved.module,
                    examples,
                    strict=options.strict_export,
                )
                exported_program_path = save_exported_program(
                    captured_exported_program,
                    output_dir / f"{name}_exported_program.pt2",
                )
                print(f"Saved {name} ExportedProgram to {exported_program_path}")

            engine_path: Optional[Path] = None
            if _role_should_compile(role.compile_torchtrt, options.compile_torchtrt):
                engine_path = resolve_engine_path(output_dir, role)
                if source_exported_program_path is not None:
                    captured_exported_program = load_exported_program(source_exported_program_path)
                else:
                    captured_exported_program, trt_arg_inputs, trt_kwarg_inputs = _capture_role_for_torchtrt(
                        resolved.module,
                        examples,
                        input_names=input_names,
                        dynamic_axes=dynamic_axes,
                        strict=options.strict_export,
                    )

                if options.capture and source_exported_program_path is None:
                    exported_program_path = save_exported_program(
                        captured_exported_program,
                        output_dir / f"{name}_exported_program.pt2",
                    )
                    print(f"Saved {name} ExportedProgram to {exported_program_path}")
                _compile_exported_program_torchtrt(
                    captured_exported_program,
                    examples,
                    engine_path=engine_path,
                    input_names=input_names,
                    output_names=output_names,
                    trt_arg_inputs=trt_arg_inputs,
                    trt_kwarg_inputs=trt_kwarg_inputs,
                )

            # Contracts are the semantic ABI consumed by generic Edge
            # runners; they explain what engine tensors mean.
            contract_path = write_role_contract_manifest(
                output_dir,
                role,
                examples,
                input_names=input_names,
                output_names=output_names,
                artifacts={
                    "engine": str(engine_path) if engine_path is not None else None,
                    "exported_program": (
                        str(exported_program_path)
                        if exported_program_path is not None
                        else None
                    ),
                },
            )

            call_packager_hook(
                role,
                manifest=self.manifest,
                output_dir=output_dir,
                engine_path=engine_path,
                exported_program_path=exported_program_path,
                loaded=loaded,
                module=resolved.module,
                examples=examples,
                input_names=input_names,
                output_names=output_names,
            )

            summary_path = write_eager_export_summary(
                output_dir,
                {
                    "role": name,
                    "component": role.component,
                    "source": "exported_program" if role.exported_program else "eager_module",
                    "module_path": role.module_path,
                    "module_type": (
                        type(resolved.module).__name__
                        if resolved.module is not None
                        else "ExportedProgram"
                    ),
                    "contract": role.contract,
                    "input_names": input_names,
                    "output_names": output_names,
                    "engine": str(engine_path) if engine_path is not None else None,
                    "exported_program": (
                        str(exported_program_path)
                        if exported_program_path is not None
                        else None
                    ),
                    "contract_manifest": (
                        str(contract_path) if contract_path is not None else None
                    ),
                },
            )
            print(f"Wrote {name} eager export summary to {summary_path}")

            results[name] = EdgeRoleExportResult(
                name=name,
                component=role.component,
                output_dir=output_dir,
                engine_path=engine_path,
                exported_program_path=exported_program_path,
                contract_path=contract_path,
                summary_path=summary_path,
                input_names=tuple(input_names),
                output_names=tuple(output_names),
                source="exported_program" if role.exported_program else "eager_module",
            )

        return results

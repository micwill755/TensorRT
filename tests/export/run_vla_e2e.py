# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the full VLA export, smoke, and benchmark flow.

This script intentionally orchestrates existing entrypoints instead of
reimplementing them:

1. ``tools.edgellm.export_from_hf`` exports language, visual, and action roles.
2. Edge LLM ``action_inference`` runs a runtime smoke test.
3. ``tests/export/benchmark_edge_vs_eager.py`` compares Edge runtime latency
   with a PyTorch eager baseline.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path, print_only: bool) -> None:
    print("\n$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    if print_only:
        return
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        stderr_tail = "\n".join(proc.stderr.splitlines()[-60:])
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(command)}\n"
            f"Stderr tail:\n{stderr_tail}"
        )


def _default_engine_dir(args: argparse.Namespace) -> Path:
    return Path(args.engine_dir) if args.engine_dir else args.output_root / "language"


def _default_multimodal_dir(args: argparse.Namespace) -> Path:
    return Path(args.multimodal_engine_dir) if args.multimodal_engine_dir else args.output_root


def _split_roles(values: list[str] | None) -> list[str]:
    if not values:
        return ["language", "visual", "action"]
    roles: list[str] = []
    for value in values:
        roles.extend(item.strip() for item in value.split(",") if item.strip())
    return roles


def _export_command(args: argparse.Namespace, role: str) -> list[str]:
    command = [
        args.python,
        "-m",
        "tools.edgellm.export_from_hf",
        "--model",
        args.model,
        "--family",
        args.family,
        "--role",
        role,
        "--output_root",
        str(args.output_root),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max_kv_cache_capacity",
        str(args.max_kv_cache_capacity),
        "--action_batch_size",
        str(args.action_batch_size),
        "--compile_torchtrt",
    ]
    if args.capture:
        command.append("--capture")
    command.extend(args.export_arg)
    return command


def _smoke_command(args: argparse.Namespace) -> list[str]:
    output_file = args.smoke_output or (args.output_root / "runtime_smoke_output.json")
    profile_file = args.smoke_profile or (args.output_root / "runtime_smoke_profile.json")
    command = [
        args.edge_binary,
        f"--inputFile={args.input_file}",
        f"--engineDir={_default_engine_dir(args)}",
        f"--multimodalEngineDir={_default_multimodal_dir(args)}",
        f"--outputFile={output_file}",
        f"--noiseSeed={args.seed}",
        f"--maxGenerateLength={args.max_generate_length}",
    ]
    if args.batch_size is not None:
        command.append(f"--batchSize={args.batch_size}")
    if args.dump_profile:
        command.extend(["--dumpProfile", f"--profileOutputFile={profile_file}"])
    if args.dump_output:
        command.append("--dumpOutput")
    if args.edge_debug:
        command.append("--debug")
    command.extend(args.edge_arg)
    return command


def _benchmark_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "tests/export/benchmark_edge_vs_eager.py",
        "--model",
        args.model,
        "--input_file",
        str(args.input_file),
        "--output_dir",
        str(args.benchmark_output_dir),
        "--edge_binary",
        args.edge_binary,
        "--engine_dir",
        str(_default_engine_dir(args)),
        "--multimodal_engine_dir",
        str(_default_multimodal_dir(args)),
        "--max_generate_length",
        str(args.max_generate_length),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--edge_dump_profile",
    ]
    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.edge_debug:
        command.append("--edge_debug")
    if args.dump_output:
        command.append("--edge_dump_output")
    for edge_arg in args.edge_arg:
        command.extend(["--edge_arg", edge_arg])

    if args.eager_adapter:
        command.extend(["--eager_adapter", args.eager_adapter])
    elif args.eager_command:
        command.extend(["--eager_command", args.eager_command])
    else:
        command.append("--skip_eager")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--input_file", required=True, type=Path, help="Runtime VLA request JSON.")
    parser.add_argument("--output_root", required=True, type=Path, help="Root directory for exported role artifacts.")
    parser.add_argument("--edge_binary", required=True, help="Path to Edge LLM action_inference binary.")
    parser.add_argument("--benchmark_output_dir", type=Path, default=None)

    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--family", default="vla")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--max_kv_cache_capacity", type=int, default=8192)
    parser.add_argument("--action_batch_size", type=int, default=2)
    parser.add_argument("--max_generate_length", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)

    parser.add_argument("--engine_dir", default=None, help="Override language engine dir. Default: output_root/language.")
    parser.add_argument(
        "--multimodal_engine_dir",
        default=None,
        help="Override multimodal engine dir. Default: output_root.",
    )
    parser.add_argument("--smoke_output", type=Path, default=None)
    parser.add_argument("--smoke_profile", type=Path, default=None)

    parser.add_argument("--eager_adapter", default=None, help="Python eager adapter factory: module:function.")
    parser.add_argument("--eager_command", default=None, help="External eager benchmark command template.")
    parser.add_argument("--export_arg", action="append", default=[], help="Extra raw arg passed to export_from_hf.")
    parser.add_argument(
        "--export_role",
        action="append",
        default=None,
        help="Role(s) to export sequentially. May be repeated or comma-separated. Default: language,visual,action.",
    )
    parser.add_argument("--edge_arg", action="append", default=[], help="Extra raw arg passed to action_inference.")

    parser.add_argument("--capture", action="store_true", help="Capture ExportedPrograms during export when supported.")
    parser.add_argument("--dump_profile", action="store_true", help="Dump profile JSON during smoke test.")
    parser.add_argument("--dump_output", action="store_true", help="Print runtime outputs during smoke/benchmark.")
    parser.add_argument("--edge_debug", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--print_only", action="store_true", help="Print commands without running them.")

    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.input_file = args.input_file.expanduser().resolve()
    if args.benchmark_output_dir is None:
        args.benchmark_output_dir = args.output_root / "benchmark_edge_vs_eager"
    else:
        args.benchmark_output_dir = args.benchmark_output_dir.expanduser().resolve()
    if args.eager_adapter and args.eager_command:
        parser.error("Use only one of --eager_adapter or --eager_command")
    args.export_roles = _split_roles(args.export_role)
    return args


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_export:
        for role in args.export_roles:
            _run(_export_command(args, role), cwd=_REPO_ROOT, print_only=args.print_only)
    if not args.skip_smoke:
        _run(_smoke_command(args), cwd=_REPO_ROOT, print_only=args.print_only)
    if not args.skip_benchmark:
        _run(_benchmark_command(args), cwd=_REPO_ROOT, print_only=args.print_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Edge LLM runtime inference against a PyTorch eager baseline.

This is intentionally a single test-style entrypoint, similar to the Alpamayo
TRT plugin test. The Edge side is driven through the C++ ``action_inference``
binary. The eager side loads the Hugging Face model once using the same
``HFModelSource`` and VLA eager-loader path used by ``tools.edgellm.export_from_hf``,
then lets a small adapter prepare requests and call model inference.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import inspect
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_STAGE_TIMING_RE = re.compile(r"Stage timings - (?P<stage>[^:]+): (?P<ms>[0-9.]+) ms")


def _stats(times_ms: list[float]) -> dict[str, float | int | None]:
    if not times_ms:
        return {
            "count": 0,
            "min_ms": None,
            "avg_ms": None,
            "max_ms": None,
            "std_ms": None,
        }
    return {
        "count": len(times_ms),
        "min_ms": min(times_ms),
        "avg_ms": statistics.fmean(times_ms),
        "max_ms": max(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_command(template: str, *, args: argparse.Namespace, iteration: int, output_path: Path) -> list[str]:
    values = {
        "iteration": iteration,
        "seed": args.seed,
        "model": args.model,
        "family": args.family,
        "input_file": args.input_file,
        "output": str(output_path),
        "output_dir": str(args.output_dir),
    }
    return shlex.split(template.format(**values))


def _load_dotted_object(path: str) -> Any:
    module_name, sep, object_name = path.partition(":")
    if not sep:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call an adapter hook with only the kwargs it accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)
    return fn(**{name: value for name, value in kwargs.items() if name in params})


def _cuda_synchronize_if_available() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _run_command(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: str | None,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[float, int, str, str]:
    start = time.perf_counter()
    proc = subprocess.run(command, env=env, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _write_text(stdout_path, proc.stdout)
    _write_text(stderr_path, proc.stderr)
    return elapsed_ms, proc.returncode, proc.stdout, proc.stderr


def _extract_stage_timings(log_text: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for match in _STAGE_TIMING_RE.finditer(log_text):
        stage_name = match.group("stage").strip().lower().replace(" ", "_")
        timings[stage_name] = float(match.group("ms"))
    return timings


def _edge_command(args: argparse.Namespace, *, output_path: Path, profile_path: Path) -> list[str]:
    command = [
        args.edge_binary,
        f"--inputFile={args.input_file}",
        f"--engineDir={args.engine_dir}",
        f"--multimodalEngineDir={args.multimodal_engine_dir}",
        f"--outputFile={output_path}",
        f"--noiseSeed={args.seed}",
        f"--maxGenerateLength={args.max_generate_length}",
    ]
    if args.batch_size is not None:
        command.append(f"--batchSize={args.batch_size}")
    if args.edge_dump_profile:
        command.extend(["--dumpProfile", f"--profileOutputFile={profile_path}"])
    if args.edge_debug:
        command.append("--debug")
    if args.edge_dump_output:
        command.append("--dumpOutput")
    command.extend(args.edge_arg)
    return command


def _run_series(
    name: str,
    *,
    args: argparse.Namespace,
    build_command: Callable[[int, Path, Path], list[str]],
    output_root: Path,
) -> dict[str, Any]:
    total_runs = args.warmup + args.iterations
    timed_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    runs: list[dict[str, Any]] = []

    for run_idx in range(total_runs):
        phase = "warmup" if run_idx < args.warmup else "timed"
        output_path = output_root / phase / f"{name}_{run_idx:03d}.json"
        profile_path = output_root / phase / f"{name}_{run_idx:03d}_profile.json"
        stdout_path = output_root / phase / f"{name}_{run_idx:03d}.stdout.txt"
        stderr_path = output_root / phase / f"{name}_{run_idx:03d}.stderr.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "EDGELLM_BENCH_NAME": name,
                "EDGELLM_BENCH_PHASE": phase,
                "EDGELLM_BENCH_ITERATION": str(run_idx),
                "EDGELLM_BENCH_SEED": str(args.seed),
                "EDGELLM_BENCH_OUTPUT": str(output_path),
            }
        )

        command = build_command(run_idx, output_path, profile_path)
        elapsed_ms, returncode, stdout, stderr = _run_command(
            command,
            env=env,
            cwd=args.cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        if returncode != 0 and not args.keep_going:
            stderr_tail = "\n".join(stderr_path.read_text(encoding="utf-8").splitlines()[-40:])
            raise RuntimeError(
                f"{name} benchmark command failed on {phase} run {run_idx} with exit code {returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"Stderr tail:\n{stderr_tail}"
            )

        stage_timings = _extract_stage_timings(stdout + "\n" + stderr)
        if phase == "timed" and returncode == 0:
            timed_ms.append(elapsed_ms)
            for stage, latency_ms in stage_timings.items():
                stage_ms.setdefault(stage, []).append(latency_ms)

        runs.append(
            {
                "iteration": run_idx,
                "phase": phase,
                "command": command,
                "returncode": returncode,
                "latency_ms": elapsed_ms,
                "output_file": str(output_path),
                "profile_file": str(profile_path) if profile_path.exists() else None,
                "stdout_file": str(stdout_path),
                "stderr_file": str(stderr_path),
                "stage_timings_ms": stage_timings,
                "output": _load_json(output_path),
                "profile": _load_json(profile_path),
            }
        )

        stage_note = ""
        if "e2e" in stage_timings:
            stage_note = f"  runtime_e2e={stage_timings['e2e']:9.3f} ms"
        print(f"{name:>5} {phase:>6} {run_idx:03d}: {elapsed_ms:9.3f} ms{stage_note}")

    return {
        "latencies_ms": timed_ms,
        "stats": _stats(timed_ms),
        "stage_stats": {stage: _stats(values) for stage, values in sorted(stage_ms.items())},
        "runs": runs,
    }


def _load_hf_eager_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    """Load the eager HF model using the same source path as export_from_hf."""
    from tools.edgellm.hf_export.source import HFModelSource
    from tools.edgellm.hf_export.strategies.vla import load_vla_model

    source = HFModelSource.from_args(args, model_attr="model")
    family = source.detected_family()
    if family != "vla":
        raise NotImplementedError(
            "The in-process eager benchmark loader currently supports VLA models. "
            f"Detected family={family!r}; use --eager_command for non-VLA models."
        )
    loaded = load_vla_model(
        **source.vla_loader_kwargs(),
        device=args.device,
        dtype=args.dtype,
    )
    return loaded, source, family


def _run_eager_adapter_series(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    """Run eager iterations with a model loaded once through HFModelSource."""
    loaded, source, family = _load_hf_eager_model(args)
    factory = _load_dotted_object(args.eager_adapter)
    runner = _call_with_supported_kwargs(
        factory,
        args=args,
        loaded=loaded,
        source=source,
        family=family,
    )
    if not callable(runner):
        raise TypeError(f"Eager adapter {args.eager_adapter!r} did not return a callable runner")

    total_runs = args.warmup + args.iterations
    timed_ms: list[float] = []
    runs: list[dict[str, Any]] = []

    for run_idx in range(total_runs):
        phase = "warmup" if run_idx < args.warmup else "timed"
        output_path = output_root / phase / f"eager_{run_idx:03d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _cuda_synchronize_if_available()
        start = time.perf_counter()
        result = runner(
            iteration=run_idx,
            phase=phase,
            output_path=output_path,
            seed=args.seed,
            input_file=args.input_file,
        )
        _cuda_synchronize_if_available()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if result is not None and not output_path.exists():
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if phase == "timed":
            timed_ms.append(elapsed_ms)

        runs.append(
            {
                "iteration": run_idx,
                "phase": phase,
                "latency_ms": elapsed_ms,
                "output_file": str(output_path),
                "output": _load_json(output_path),
            }
        )
        print(f"eager {phase:>6} {run_idx:03d}: {elapsed_ms:9.3f} ms")

    return {"latencies_ms": timed_ms, "stats": _stats(timed_ms), "stage_stats": {}, "runs": runs}


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Benchmark Summary ===")
    for name in ("edge", "eager"):
        result = summary.get(name)
        if not result:
            continue
        stats = result["stats"]
        if stats["count"] == 0:
            print(f"{name:>5}: no successful timed runs")
            continue
        print(
            f"{name:>5}: "
            f"min={stats['min_ms']:9.3f} ms  "
            f"avg={stats['avg_ms']:9.3f} ms  "
            f"max={stats['max_ms']:9.3f} ms  "
            f"std={stats['std_ms']:9.3f} ms"
        )
        stage_stats = result.get("stage_stats", {})
        if "e2e" in stage_stats and stage_stats["e2e"]["count"]:
            print(f"      runtime_e2e_avg={stage_stats['e2e']['avg_ms']:9.3f} ms")
    speedup = summary.get("speedup_eager_over_edge")
    if speedup is not None:
        print(f"speedup eager/edge: {speedup:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id/path.")
    parser.add_argument("--task", default=None, help="Optional HF task/family override.")
    parser.add_argument("--family", default="auto", help="Optional HF family override.")
    parser.add_argument("--model_class", default=None, help="Optional fully-qualified HF model/policy class.")
    parser.add_argument("--attn_implementation", default="eager", help="Attention implementation hint.")
    parser.add_argument("--device", default=None, help="Device used for eager loading, default chosen by loader.")
    parser.add_argument("--dtype", default="fp16", help="Dtype used for eager loading.")
    parser.add_argument("--input_file", required=True, help="Edge runtime request JSON.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for benchmark outputs.")
    parser.add_argument("--iterations", type=int, default=10, help="Timed iterations, excluding warmup.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations excluded from stats.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--cwd", default=None, help="Optional working directory for child commands.")
    parser.add_argument("--keep_going", action="store_true", help="Continue after a failed child command.")

    edge = parser.add_argument_group("Edge runtime")
    edge.add_argument("--skip_edge", action="store_true")
    edge.add_argument("--edge_binary", help="Path to Edge LLM action_inference binary.")
    edge.add_argument("--engine_dir", help="Language engine directory.")
    edge.add_argument("--multimodal_engine_dir", help="Directory containing visual/action engine artifacts.")
    edge.add_argument("--batch_size", type=int, default=None)
    edge.add_argument("--max_generate_length", type=int, default=16)
    edge.add_argument("--edge_dump_profile", action="store_true", help="Ask action_inference to export profile JSON.")
    edge.add_argument("--edge_debug", action="store_true")
    edge.add_argument("--edge_dump_output", action="store_true")
    edge.add_argument("--edge_arg", action="append", default=[], help="Extra raw argument passed to action_inference.")

    eager = parser.add_argument_group("PyTorch eager")
    eager.add_argument("--skip_eager", action="store_true")
    eager.add_argument(
        "--eager_command",
        help=(
            "Command template for one eager inference iteration. Supported fields: "
            "{iteration}, {seed}, {model}, {family}, {input_file}, {output}, {output_dir}. "
            "The command should write JSON to {output} when possible."
        ),
    )
    eager.add_argument(
        "--eager_adapter",
        help=(
            "Python factory as module:function. The benchmark loads the HF model once through "
            "HFModelSource/load_vla_model and passes it to the factory as loaded=. The factory must return "
            "a callable runner accepting iteration, phase, output_path, seed, and input_file."
        ),
    )

    args = parser.parse_args()
    if not args.skip_edge:
        missing = [name for name in ("edge_binary", "engine_dir", "multimodal_engine_dir") if not getattr(args, name)]
        if missing:
            parser.error(f"missing Edge runtime arguments: {', '.join('--' + m for m in missing)}")
    if not args.skip_eager:
        if bool(args.eager_command) == bool(args.eager_adapter):
            parser.error("provide exactly one of --eager_command or --eager_adapter unless --skip_eager is set")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "model": args.model,
        "family": args.family,
        "input_file": args.input_file,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "seed": args.seed,
    }

    if not args.skip_edge:
        summary["edge"] = _run_series(
            "edge",
            args=args,
            output_root=args.output_dir / "edge",
            build_command=lambda _i, out, profile: _edge_command(args, output_path=out, profile_path=profile),
        )

    if not args.skip_eager and args.eager_command:
        summary["eager"] = _run_series(
            "eager",
            args=args,
            output_root=args.output_dir / "eager",
            build_command=lambda i, out, _profile: _format_command(
                args.eager_command, args=args, iteration=i, output_path=out
            ),
        )
    elif not args.skip_eager and args.eager_adapter:
        summary["eager"] = _run_eager_adapter_series(args, args.output_dir / "eager")

    edge_avg = summary.get("edge", {}).get("stats", {}).get("avg_ms")
    eager_avg = summary.get("eager", {}).get("stats", {}).get("avg_ms")
    if edge_avg and eager_avg:
        summary["speedup_eager_over_edge"] = eager_avg / edge_avg

    summary_path = args.output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"\nWrote benchmark report to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

import numpy as np
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.export.vla_test_data import load_vla_test_sample


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


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    """Unit-neutral stats for quality metrics such as meters and ADE."""
    stats = _stats(values)
    return {
        "count": stats["count"],
        "min": stats["min_ms"],
        "avg": stats["avg_ms"],
        "max": stats["max_ms"],
        "std": stats["std_ms"],
    }


def _points_array(value: Any) -> np.ndarray | None:
    """Convert nested JSON trajectory-like data to ``[T, D]`` points."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.ndim == 1:
        if array.shape[0] < 2:
            return None
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[-1] < 2:
        return None
    return array[:, :2]


def _candidate_arrays(value: Any) -> list[np.ndarray]:
    """Convert trajectory-like JSON to candidate ``[T, 2]`` arrays."""
    if value is None:
        return []
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception:
        return []
    if array.size == 0:
        return []
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        return [_points_array(array)] if array.shape[0] >= 2 else []
    if array.ndim == 2:
        points = _points_array(array)
        return [points] if points is not None else []
    if array.ndim == 3:
        if array.shape[-1] >= 2:
            return [candidate[:, :2] for candidate in array]
        if array.shape[1] >= 2:
            return [candidate[:2, :].T for candidate in array]
    if array.ndim > 3 and array.shape[-1] >= 2:
        reshaped = array.reshape(-1, array.shape[-2], array.shape[-1])
        return [candidate[:, :2] for candidate in reshaped]
    return []


def _candidate_trajectories(value: Any) -> list[np.ndarray]:
    """Extract candidate ``[T, 2]`` trajectories from benchmark JSON output."""
    if value is None:
        return []
    if isinstance(value, dict):
        candidates: list[np.ndarray] = []
        for key in ("output_trajectory", "trajectory", "pred_trajectory", "action_trajectory", "pred_xyz"):
            candidates.extend(_candidate_arrays(value.get(key)))
        for response in value.get("responses", []) or []:
            candidates.extend(_candidate_trajectories(response))
        for key in ("action_pred", "actions", "action", "output"):
            child = value.get(key)
            if isinstance(child, dict):
                candidates.extend(_candidate_arrays(child.get("values") or child.get("data") or child.get("trajectory")))
            else:
                candidates.extend(_candidate_arrays(child))
        return [candidate for candidate in candidates if candidate is not None]
    if isinstance(value, list):
        return _candidate_arrays(value)
    return []


def _compute_minade(pred_candidates: list[np.ndarray], gt_xy: np.ndarray) -> float | None:
    """Alpamayo-style minADE: average displacement per candidate, then take min."""
    if not pred_candidates or gt_xy is None or gt_xy.size == 0:
        return None
    ades: list[float] = []
    for pred_xy in pred_candidates:
        horizon = min(len(pred_xy), len(gt_xy))
        if horizon <= 0:
            continue
        diff = np.linalg.norm(pred_xy[:horizon, :2] - gt_xy[:horizon, :2], axis=1).mean()
        ades.append(float(diff))
    return min(ades) if ades else None


def _add_minade_metrics(summary: dict[str, Any], args: argparse.Namespace) -> None:
    sample = load_vla_test_sample(args)
    if sample is None or sample.ground_truth_trajectory is None:
        return
    gt_xy = sample.ground_truth_trajectory
    sample.write_ground_truth_trajectory(getattr(args, "write_ground_truth_trajectory", None))
    summary["ground_truth_trajectory"] = {
        "num_points": int(gt_xy.shape[0]),
        "dim": int(gt_xy.shape[1]),
        "source": sample.source,
        "metadata": sample.metadata,
    }
    for name in ("edge", "eager"):
        result = summary.get(name)
        if not result:
            continue
        minades: list[float] = []
        for run in result.get("runs", []):
            candidates = _candidate_trajectories(run.get("output"))
            minade = _compute_minade(candidates, gt_xy)
            if minade is None:
                continue
            run.setdefault("quality_metrics", {})["minade"] = minade
            if run.get("phase") == "timed":
                minades.append(minade)
        if minades:
            result.setdefault("quality_stats", {})["minade"] = _metric_stats(minades)


def _first_candidate(output: Any) -> np.ndarray | None:
    candidates = _candidate_trajectories(output)
    return candidates[0] if candidates else None


def _trajectory_diff(pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float] | None:
    """Alpamayo-style trajectory tensor diff for two candidate trajectories."""
    horizon = min(len(pred_a), len(pred_b))
    if horizon <= 0:
        return None
    a = pred_a[:horizon, :2]
    b = pred_b[:horizon, :2]
    abs_diff = np.abs(a - b)
    l2 = np.linalg.norm(a - b, axis=1)
    return {
        "num_points": int(horizon),
        "max_abs_m": float(abs_diff.max()),
        "mean_abs_m": float(abs_diff.mean()),
        "mean_l2_m": float(l2.mean()),
    }


def _timed_runs_by_order(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    return [run for run in result.get("runs", []) if run.get("phase") == "timed"]


def _add_edge_eager_comparison(summary: dict[str, Any], args: argparse.Namespace) -> None:
    """Compare Edge and eager outputs in the same spirit as Alpamayo TRT tests."""
    edge_runs = _timed_runs_by_order(summary.get("edge"))
    eager_runs = _timed_runs_by_order(summary.get("eager"))
    pair_count = min(len(edge_runs), len(eager_runs))
    if pair_count == 0:
        return

    trajectory_diffs: list[dict[str, Any]] = []
    ade_diffs: list[float] = []
    for idx in range(pair_count):
        edge_run = edge_runs[idx]
        eager_run = eager_runs[idx]
        pair: dict[str, Any] = {
            "pair_index": idx,
            "edge_iteration": edge_run.get("iteration"),
            "eager_iteration": eager_run.get("iteration"),
        }

        edge_traj = _first_candidate(edge_run.get("output"))
        eager_traj = _first_candidate(eager_run.get("output"))
        diff = _trajectory_diff(edge_traj, eager_traj) if edge_traj is not None and eager_traj is not None else None
        if diff is not None:
            pair["trajectory_diff"] = diff

        edge_minade = (edge_run.get("quality_metrics") or {}).get("minade")
        eager_minade = (eager_run.get("quality_metrics") or {}).get("minade")
        if edge_minade is not None and eager_minade is not None:
            ade_diff = abs(float(edge_minade) - float(eager_minade))
            pair["ade_diff_m"] = ade_diff
            ade_diffs.append(ade_diff)

        if "trajectory_diff" in pair or "ade_diff_m" in pair:
            trajectory_diffs.append(pair)

    if not trajectory_diffs:
        return

    mean_abs = [item["trajectory_diff"]["mean_abs_m"] for item in trajectory_diffs if "trajectory_diff" in item]
    max_abs = [item["trajectory_diff"]["max_abs_m"] for item in trajectory_diffs if "trajectory_diff" in item]
    mean_l2 = [item["trajectory_diff"]["mean_l2_m"] for item in trajectory_diffs if "trajectory_diff" in item]

    comparison: dict[str, Any] = {
        "timed_pair_count": pair_count,
        "pairs": trajectory_diffs,
        "thresholds": {
            "trajectory_mean_abs_m": args.trajectory_diff_threshold,
            "ade_diff_m": args.ade_diff_threshold,
        },
    }
    if mean_abs:
        comparison["trajectory_diff_stats"] = {
            "mean_abs_m": _metric_stats(mean_abs),
            "max_abs_m": _metric_stats(max_abs),
            "mean_l2_m": _metric_stats(mean_l2),
        }
        comparison["trajectory_mean_abs_pass"] = statistics.fmean(mean_abs) <= args.trajectory_diff_threshold
    if ade_diffs:
        comparison["ade_diff_stats"] = _metric_stats(ade_diffs)
        comparison["ade_diff_pass"] = statistics.fmean(ade_diffs) <= args.ade_diff_threshold

    pass_values = [value for key, value in comparison.items() if key.endswith("_pass")]
    if pass_values:
        comparison["passed"] = all(pass_values)
    summary["edge_vs_eager_comparison"] = comparison


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
    proc = subprocess.run(command, env=env, cwd=cwd, capture_output=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    _write_text(stdout_path, stdout)
    _write_text(stderr_path, stderr)
    return elapsed_ms, proc.returncode, stdout, stderr


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
        quality_stats = result.get("quality_stats", {})
        if "minade" in quality_stats and quality_stats["minade"]["count"]:
            print(f"      minADE_avg={quality_stats['minade']['avg']:9.4f} m")
    comparison = summary.get("edge_vs_eager_comparison", {})
    if comparison:
        print("\n=== Edge vs Eager Quality ===")
        traj_stats = comparison.get("trajectory_diff_stats", {})
        if traj_stats:
            print(
                "trajectory diff: "
                f"mean_abs_avg={traj_stats['mean_abs_m']['avg']:9.4f} m  "
                f"max_abs_avg={traj_stats['max_abs_m']['avg']:9.4f} m  "
                f"mean_l2_avg={traj_stats['mean_l2_m']['avg']:9.4f} m"
            )
        ade_stats = comparison.get("ade_diff_stats")
        if ade_stats and ade_stats["count"]:
            print(f"ADE diff avg={ade_stats['avg']:9.4f} m")
        if "passed" in comparison:
            print(f"quality thresholds passed: {comparison['passed']}")
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
    parser.add_argument(
        "--test_data_source",
        default="auto",
        choices=("auto", "json", "input_json", "alpamayo", "lerobot", "none", "off"),
        help="Source used to resolve future ground truth for minADE.",
    )
    parser.add_argument("--ground_truth_trajectory", help="Optional JSON future trajectory used to compute minADE.")
    parser.add_argument("--write_ground_truth_trajectory", help="Optional path to write the resolved ground-truth trajectory JSON.")
    parser.add_argument("--alpamayo_clip_id", help="Load ground-truth future trajectory from Alpamayo's physical_ai_av dataset clip.")
    parser.add_argument("--alpamayo_t0_us", type=int, default=5_100_000, help="Alpamayo sample timestamp in microseconds.")
    parser.add_argument("--alpamayo_num_history_steps", type=int, default=16, help="Number of Alpamayo history trajectory points to load.")
    parser.add_argument("--alpamayo_num_future_steps", type=int, default=64, help="Number of Alpamayo future trajectory points to load.")
    parser.add_argument("--alpamayo_time_step", type=float, default=0.1, help="Alpamayo trajectory timestep in seconds.")
    parser.add_argument("--alpamayo_num_frames", type=int, default=4, help="Number of Alpamayo camera frames per camera to load.")
    parser.add_argument("--alpamayo_no_stream", action="store_true", help="Disable streaming when loading Alpamayo dataset features.")
    parser.add_argument("--alpamayo_src", help="Optional path to Alpamayo src directory if not importable.")
    parser.add_argument("--lerobot_dataset_repo_id", help="Optional LeRobot dataset repo id used to load future action ground truth.")
    parser.add_argument("--lerobot_episode_index", type=int, default=0, help="LeRobot episode index used for future action ground truth.")
    parser.add_argument("--lerobot_frame_index", type=int, default=0, help="Frame offset inside the LeRobot episode.")
    parser.add_argument("--lerobot_future_steps", type=int, default=50, help="Number of LeRobot future action steps to load.")
    parser.add_argument("--lerobot_action_key", default="action", help="LeRobot item key containing action vectors.")
    parser.add_argument("--trajectory_diff_threshold", type=float, default=0.05, help="Pass threshold for Edge-vs-eager mean absolute trajectory diff in meters.")
    parser.add_argument("--ade_diff_threshold", type=float, default=0.15, help="Pass threshold for Edge-vs-eager minADE difference in meters.")
    parser.add_argument("--fail_on_quality_thresholds", action="store_true", help="Return nonzero if Edge-vs-eager quality thresholds fail.")
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

    _add_minade_metrics(summary, args)
    _add_edge_eager_comparison(summary, args)

    edge_avg = summary.get("edge", {}).get("stats", {}).get("avg_ms")
    eager_avg = summary.get("eager", {}).get("stats", {}).get("avg_ms")
    if edge_avg and eager_avg:
        summary["speedup_eager_over_edge"] = eager_avg / edge_avg

    summary_path = args.output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"\nWrote benchmark report to {summary_path}")
    comparison = summary.get("edge_vs_eager_comparison", {})
    if args.fail_on_quality_thresholds and comparison.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

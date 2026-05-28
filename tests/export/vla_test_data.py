# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-data sources for VLA export/runtime benchmarks.

The benchmark should not need to know whether future ground truth came from a
plain JSON file, Alpamayo physical_ai_av, or a LeRobot dataset. This module
keeps that data access behind one small interface and returns trajectory points
in a common ``[T, 2]`` numpy shape for minADE-style checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def points_array(value: Any) -> np.ndarray | None:
    """Convert nested trajectory-like data to ``[T, 2]`` points."""
    if value is None:
        return None
    try:
        array = _to_numpy(value)
    except Exception:
        return None
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
    return array[:, :2].astype(np.float32, copy=False)


def ground_truth_from_json(value: Any) -> np.ndarray | None:
    """Find an explicit future/ground-truth trajectory in JSON-like data."""
    if isinstance(value, dict):
        for key in (
            "ground_truth_trajectory",
            "future_trajectory",
            "ego_future_trajectory",
            "ego_future_xyz",
            "gt_trajectory",
        ):
            points = points_array(value.get(key))
            if points is not None:
                return points
        for request in value.get("requests", []) or []:
            points = ground_truth_from_json(request)
            if points is not None:
                return points
    return None


@dataclass
class VLATestSample:
    """Resolved test data used by benchmark quality metrics."""

    ground_truth_trajectory: np.ndarray | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def write_ground_truth_trajectory(self, output_path: str | Path | None) -> None:
        if not output_path or self.ground_truth_trajectory is None:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ground_truth_trajectory": self.ground_truth_trajectory.astype(float).tolist(),
            "source": self.source,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class VLATestDataSource:
    """Base class for benchmark test-data sources."""

    def load(self) -> VLATestSample:
        raise NotImplementedError

    @staticmethod
    def from_args(args: argparse.Namespace) -> "VLATestDataSource | None":
        requested = getattr(args, "test_data_source", "auto") or "auto"
        requested = requested.lower()

        if requested == "json" or (requested == "auto" and getattr(args, "ground_truth_trajectory", None)):
            return JsonTrajectoryDataSource(Path(args.ground_truth_trajectory))
        if requested == "alpamayo" or (requested == "auto" and getattr(args, "alpamayo_clip_id", None)):
            return AlpamayoDataSource.from_args(args)
        if requested == "lerobot" or (requested == "auto" and getattr(args, "lerobot_dataset_repo_id", None)):
            return LeRobotDataSource.from_args(args)
        if requested == "input_json" or requested == "auto":
            return InputJsonDataSource(Path(args.input_file))
        if requested in ("none", "off"):
            return None
        raise ValueError(f"Unsupported --test_data_source: {requested}")


@dataclass
class JsonTrajectoryDataSource(VLATestDataSource):
    path: Path

    def load(self) -> VLATestSample:
        payload = _load_json(self.path)
        points = ground_truth_from_json(payload)
        if points is None:
            points = points_array(payload)
        if points is None:
            raise ValueError(f"Could not parse {self.path} as a [T,2+] ground-truth trajectory")
        return VLATestSample(
            ground_truth_trajectory=points,
            source=str(self.path),
            metadata={"source_type": "json"},
            raw=payload,
        )


@dataclass
class InputJsonDataSource(VLATestDataSource):
    input_file: Path

    def load(self) -> VLATestSample:
        payload = _load_json(self.input_file)
        points = ground_truth_from_json(payload)
        return VLATestSample(
            ground_truth_trajectory=points,
            source=str(self.input_file),
            metadata={"source_type": "input_json"},
            raw=payload,
        )


@dataclass
class AlpamayoDataSource(VLATestDataSource):
    """Alpamayo physical_ai_av source, matching alpamayo_r1/test_inference.py."""

    clip_id: str
    t0_us: int = 5_100_000
    maybe_stream: bool = True
    num_history_steps: int = 16
    num_future_steps: int = 64
    time_step: float = 0.1
    num_frames: int = 4
    alpamayo_src: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AlpamayoDataSource":
        return cls(
            clip_id=args.alpamayo_clip_id,
            t0_us=int(args.alpamayo_t0_us),
            maybe_stream=not args.alpamayo_no_stream,
            num_history_steps=int(args.alpamayo_num_history_steps),
            num_future_steps=int(args.alpamayo_num_future_steps),
            time_step=float(args.alpamayo_time_step),
            num_frames=int(args.alpamayo_num_frames),
            alpamayo_src=getattr(args, "alpamayo_src", None),
        )

    def _ensure_importable(self) -> None:
        candidates: list[Path] = []
        if self.alpamayo_src:
            candidates.append(Path(self.alpamayo_src))
        candidates.extend(
            [
                _REPO_ROOT.parents[1] / "alpamayo" / "src",
                Path("/workspace/alpamayo/src"),
                Path("/mnt/scratch/workspace/alpamayo/src"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return

    def load(self) -> VLATestSample:
        self._ensure_importable()
        try:
            from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        except ImportError as exc:
            raise ImportError(
                "Could not import alpamayo_r1. Pass --alpamayo_src /path/to/alpamayo/src "
                "or set PYTHONPATH to include the Alpamayo src directory."
            ) from exc

        data = load_physical_aiavdataset(
            self.clip_id,
            t0_us=self.t0_us,
            maybe_stream=self.maybe_stream,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
            num_frames=self.num_frames,
        )
        gt_xyz = data["ego_future_xyz"]
        gt_xy = _to_numpy(gt_xyz)[0, 0, :, :2]
        return VLATestSample(
            ground_truth_trajectory=gt_xy.astype(np.float32, copy=False),
            source=f"alpamayo:{self.clip_id}:t0_us={self.t0_us}",
            metadata={
                "source_type": "alpamayo",
                "clip_id": self.clip_id,
                "t0_us": self.t0_us,
                "maybe_stream": self.maybe_stream,
                "num_history_steps": self.num_history_steps,
                "num_future_steps": self.num_future_steps,
                "time_step": self.time_step,
                "num_frames": self.num_frames,
            },
            raw=data,
        )


@dataclass
class LeRobotDataSource(VLATestDataSource):
    """LeRobot dataset source for robot-action future trajectories."""

    repo_id: str
    episode_index: int = 0
    frame_index: int = 0
    future_steps: int = 50
    action_key: str = "action"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LeRobotDataSource":
        return cls(
            repo_id=args.lerobot_dataset_repo_id,
            episode_index=int(args.lerobot_episode_index),
            frame_index=int(args.lerobot_frame_index),
            future_steps=int(args.lerobot_future_steps),
            action_key=args.lerobot_action_key,
        )

    def _episode_start(self, dataset: Any) -> int:
        episode_data_index = getattr(dataset, "episode_data_index", None)
        if episode_data_index is not None:
            starts = episode_data_index.get("from") if isinstance(episode_data_index, dict) else None
            if starts is not None:
                return int(_to_numpy(starts)[self.episode_index])
        for idx in range(len(dataset)):
            item = dataset[idx]
            episode = item.get("episode_index", item.get("episode", None))
            if episode is not None and int(_to_numpy(episode).reshape(-1)[0]) == self.episode_index:
                return idx
        raise ValueError(f"Could not find episode_index={self.episode_index} in {self.repo_id}")

    def _import_dataset_cls(self) -> Any:
        import importlib

        candidates = (
            "lerobot.common.datasets.lerobot_dataset",
            "lerobot.datasets.lerobot_dataset",
        )
        errors: list[str] = []
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, "LeRobotDataset")
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
        raise ImportError(
            "Could not import LeRobotDataset. Tried: " + "; ".join(errors)
        )

    def load(self) -> VLATestSample:
        LeRobotDataset = self._import_dataset_cls()
        dataset = LeRobotDataset(self.repo_id)
        start = self._episode_start(dataset) + self.frame_index
        actions: list[np.ndarray] = []
        for offset in range(self.future_steps):
            item = dataset[start + offset]
            if self.action_key not in item:
                raise KeyError(f"LeRobot item does not contain action key {self.action_key!r}")
            action = _to_numpy(item[self.action_key]).reshape(-1)
            actions.append(action)
        points = points_array(np.stack(actions, axis=0))
        return VLATestSample(
            ground_truth_trajectory=points,
            source=f"lerobot:{self.repo_id}:episode={self.episode_index}:frame={self.frame_index}",
            metadata={
                "source_type": "lerobot",
                "repo_id": self.repo_id,
                "episode_index": self.episode_index,
                "frame_index": self.frame_index,
                "future_steps": self.future_steps,
                "action_key": self.action_key,
            },
            raw=None,
        )


def load_vla_test_sample(args: argparse.Namespace) -> VLATestSample | None:
    source = VLATestDataSource.from_args(args)
    if source is None:
        return None
    return source.load()

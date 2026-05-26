# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base types for run_hf-style Edge export strategies."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class HFExportConfig:
    """Common configuration passed from the HF CLI into a strategy.

    The CLI parses many flags, but each family strategy should see
    one stable object. Strategies can then build eager manifests
    without depending directly on argparse.
    """
    model: str
    family: str
    roles: Sequence[str]
    output_root: Optional[str] = None
    action_output_dir: Optional[str] = None
    model_class: Optional[str] = None
    attn_implementation: Optional[str] = "eager"
    trust_remote_code: bool = True
    max_kv_cache_capacity: int = 4096
    action_batch_size: int = 2
    source: Any = None
    metadata: Mapping[str, object] | None = None


class EdgeHFStrategy(abc.ABC):
    """A family-level strategy that emits an eager-export manifest.

    A strategy owns family-specific discovery. For example, the VLA
    strategy decides whether an eager model looks like PI0.5-style
    prefix-KV action export or GR00T-style state-conditioned export.
    """

    def __init__(self, cfg: HFExportConfig):
        """Store the normalized configuration for this family strategy."""
        self.cfg = cfg

    @abc.abstractmethod
    def build_manifest(self) -> dict:
        """Return an eager-export manifest dictionary.

        The returned manifest is then consumed by ``EdgeExport`` just
        like a user-authored eager manifest.
        """

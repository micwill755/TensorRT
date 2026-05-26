# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for exporting already-loaded eager PyTorch model components.

The eager export path starts after model loading. A user-provided loader returns
an eager model that already runs in Python, and this package resolves/captures
named runtime roles from that model.
"""

from .manifest import EagerExportManifest, EagerExportRole, load_manifest

__all__ = [
    "EagerExportManifest",
    "EagerExportRole",
    "load_manifest",
]

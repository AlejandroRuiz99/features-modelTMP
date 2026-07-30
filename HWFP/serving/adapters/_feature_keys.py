"""Canonical numeric feature key ordering for the PyTorch adapter bridge.

Re-exported from `HWFP.core.domain.feature_keys` — the single source of
truth shared with the training layer. REQ-11 forbids direct
`HWFP.serving` <-> `HWFP.training` imports, so both layers import this list
from `HWFP.core` instead of from each other (see the module docstring in
`HWFP.core.domain.feature_keys` for the full rationale).
"""

from __future__ import annotations

from HWFP.core.domain.feature_keys import CANONICAL_FEATURE_KEYS

__all__ = ["CANONICAL_FEATURE_KEYS"]

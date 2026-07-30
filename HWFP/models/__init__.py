"""HWFP.models — absorbed foul-prediction ensemble (leaf library package).

This package must not import anything from HWFP.core, HWFP.serving, or
HWFP.training (REQ-12, architecture-boundaries). It is a pure library:
model definitions, training-time layers, and checkpoint path resolution.
"""

from __future__ import annotations

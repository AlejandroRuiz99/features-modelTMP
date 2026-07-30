"""HWFP.features — absorbed feature-generation pipeline (leaf library package).

This package must not import anything from HWFP.core, HWFP.serving, or
HWFP.training (REQ-12, architecture-boundaries). It is a pure library:
feature assembly, transformation, and shared statistical-state helpers.
"""

from __future__ import annotations

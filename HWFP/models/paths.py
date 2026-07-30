"""Package-relative checkpoint path resolution for HWFP.models.

Replaces the legacy pattern of traversing to the repo root
(``Path(__file__).parents[3]``) to locate ``prediction_models/checkpoints``.
The production ensemble checkpoint now lives inside this package, so its
location can be resolved relative to this file with no ``sys.path``
mutation and no repo-root assumption.
"""

from __future__ import annotations

from pathlib import Path


def default_checkpoints_dir() -> Path:
    """Return the package-relative path to the production ensemble checkpoint.

    Returns:
        Path to ``HWFP/models/checkpoints/ensemble`` resolved next to this
        module, regardless of the current working directory.
    """
    return Path(__file__).resolve().parent / "checkpoints" / "ensemble"

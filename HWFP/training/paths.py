"""Package-relative path resolution for the training layer.

Mirrors `HWFP.models.paths.default_checkpoints_dir()` — the default lives
next to the code (never repo-root traversal), and callers can override via
an environment variable.
"""

from __future__ import annotations

from pathlib import Path


def default_training_data_path() -> Path:
    """Default location of the production training Parquet file.

    Package-file-relative: `HWFP/training/data/training.parquet`. Never
    traverses to repo root — the training data lives inside the training
    layer, alongside the code that consumes it (design D2).
    """
    return Path(__file__).parent / "data" / "training.parquet"

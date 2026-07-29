from __future__ import annotations

from enum import Enum


class ModelStatus(Enum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"

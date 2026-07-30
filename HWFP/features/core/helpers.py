"""Helpers matematicos puros compartidos por los modulos de transformation."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import TypeVar

T = TypeVar("T")


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def decay_weight(match_date: date, reference: date, lam: float) -> float:
    """Peso exponencial segun antiguedad del partido."""
    days = max(0, (reference - match_date).days)
    return math.exp(-lam * days)


def safe(val: T | None, default: T = 0) -> T:  # type: ignore[assignment]
    """Devuelve val si no es None; default en caso contrario."""
    return val if val is not None else default


def clip(v: float, lo: float, hi: float) -> float:
    """Acota v al intervalo [lo, hi]."""
    return max(lo, min(hi, v))

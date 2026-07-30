"""Calibración isotónica post-ensemble para probabilidades Over/Under.

Después de entrenar el ensemble, la capa de calibración aprende una función
monotónica no-paramétrica que mapea probabilidades brutas predichas → probabilidades
calibradas, una por cada línea OU.

Uso:
    cal = OUCalibrationLayer()
    cal.fit(predictions, fouls_totals)          # lista de MatchPrediction + outcomes
    cal.calibrate_pmf_ou(pmf)                   # dict {line: (p_over_cal, p_under_cal)}

    # Guardar / cargar
    cal.save(directory / "calibration.npz")
    cal.load(directory / "calibration.npz")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from HWFP.models.utils.distributions import FoulPMF

logger = logging.getLogger(__name__)

_DEFAULT_LINES = [21.5, 23.5, 25.5, 27.5, 29.5, 31.5, 33.5]
_MIN_SAMPLES = 30  # mínimo para calibrar; si hay menos, se devuelve prob sin ajustar


class OUCalibrationLayer:
    """Calibración isotónica por línea OU usando IsotonicRegression de sklearn.

    Attributes:
        lines: líneas OU para calibrar.
        _calibrators: dict {line: (X_thresholds, y_thresholds)} con los
            puntos de interpolación del regresor isotónico (evita pickle).
    """

    def __init__(self, lines: list[float] | None = None) -> None:
        self.lines: list[float] = lines or _DEFAULT_LINES
        self._calibrators: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._is_fitted = False

    def fit(self, predictions: list, fouls_totals: list[int]) -> None:
        """Entrena un regresor isotónico por línea OU.

        Args:
            predictions: lista de MatchPrediction del ensemble.
            fouls_totals: lista de totales de faltas reales observados.
        """
        from sklearn.isotonic import IsotonicRegression

        if len(predictions) != len(fouls_totals):
            raise ValueError("predictions y fouls_totals deben tener el mismo largo")

        self._calibrators = {}
        for line in self.lines:
            probs = np.array([float(p.pmf_total.prob_over(line)) for p in predictions])
            outcomes = np.array([1.0 if y > line else 0.0 for y in fouls_totals])

            if len(probs) < _MIN_SAMPLES:
                logger.warning(
                    "Calibración para línea %.1f: solo %d muestras (mínimo %d). "
                    "Se omite — se devuelven probabilidades sin calibrar.",
                    line, len(probs), _MIN_SAMPLES,
                )
                continue

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs, outcomes)
            self._calibrators[line] = (
                np.array(ir.X_thresholds_),
                np.array(ir.y_thresholds_),
            )
            logger.debug(
                "Calibración línea %.1f: %d puntos de corte, "
                "rango calibrado [%.3f, %.3f]",
                line, len(ir.X_thresholds_), ir.y_thresholds_.min(), ir.y_thresholds_.max(),
            )

        self._is_fitted = bool(self._calibrators)
        logger.info(
            "Calibración lista: %d/%d líneas calibradas",
            len(self._calibrators), len(self.lines),
        )

    def calibrate(self, line: float, p_over_raw: float) -> float:
        """Aplica calibración isotónica a una probabilidad bruta."""
        if line not in self._calibrators:
            return p_over_raw
        X_thr, y_thr = self._calibrators[line]
        return float(np.interp(p_over_raw, X_thr, y_thr))

    def calibrate_ou_table(
        self, pmf: FoulPMF
    ) -> dict[float, tuple[float, float]]:
        """Devuelve {line: (p_over_calibrada, p_under_calibrada)} para todas las líneas."""
        result = {}
        for line in self.lines:
            p_raw = float(pmf.prob_over(line))
            p_cal = self.calibrate(line, p_raw)
            result[line] = (round(p_cal, 4), round(1.0 - p_cal, 4))
        return result

    # -------------------------------------------------------------------------
    # Persistencia (NPZ + JSON, sin pickle)
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Guarda la capa de calibración en un archivo .npz."""
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        meta: dict = {"lines": self.lines, "calibrated_lines": []}

        for line in self.lines:
            if line in self._calibrators:
                key = f"line_{line:.1f}".replace(".", "_")
                X, y = self._calibrators[line]
                arrays[f"{key}_X"] = X
                arrays[f"{key}_y"] = y
                meta["calibrated_lines"].append(line)

        np.savez(path, **arrays)
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(meta, f)
        logger.info("Calibración guardada: %s", path)

    def load(self, path: str | Path) -> None:
        """Carga la capa de calibración desde .npz."""
        path = Path(path)
        if not path.exists():
            logger.warning("Archivo de calibración no encontrado: %s", path)
            return

        data = np.load(path, allow_pickle=False)
        json_path = path.with_suffix(".json")
        with open(json_path) as f:
            meta = json.load(f)

        self.lines = meta["lines"]
        self._calibrators = {}
        for line in meta.get("calibrated_lines", []):
            key = f"line_{line:.1f}".replace(".", "_")
            X = data[f"{key}_X"]
            y = data[f"{key}_y"]
            self._calibrators[line] = (X, y)

        self._is_fitted = bool(self._calibrators)
        logger.info(
            "Calibración cargada: %d/%d líneas",
            len(self._calibrators), len(self.lines),
        )

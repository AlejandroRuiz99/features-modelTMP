"""
referee_gmm.py — Inferencia del modo arbitral en prediction_models.

RESPONSABILIDAD DE ESTA PIEZA:
  - RefereeProfile: dataclass con los parametros GMM precalculados (viene del feature dict).
  - ModeSelector: MLP que predice P(modo estricto) dado contexto del partido.
  - RefereeProfiler: gestor de perfiles en memoria + entrenamiento del ModeSelector.

NO ES RESPONSABILIDAD DE ESTA PIEZA:
  - Ajustar (fit) el GMM desde historico. Eso lo hace features_generator/model/core/referee_gmm.py.
  - Los parametros mu/sigma/peso del GMM llegan ya calculados en el feature dict
    bajo arbitro.estadisticas.

Flujo en prediction_models:
  1. El dataset (Parquet) aporta referee_mu_permisivo/estricto/peso_estricto.
  2. El ensemble usa esos valores directamente para calcular P(modo estricto) via ModeSelector.
  3. RefereeProfiler.load() carga el ModeSelector entrenado desde checkpoints/.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata

# Minimum matches for a referee to be considered standalone (not shrunk)
MIN_MATCHES_STANDALONE = 8


def compute_percentile_target(weight_strict_values: np.ndarray) -> np.ndarray:
    """Compute the league-percentile rank of each referee's weight_strict.

    Maps each value to its rank percentile among all training referees,
    ensuring a monotonic, variance-rich target in [0, 1].

    Args:
        weight_strict_values: 1-D array of weight_strict values for N referees.

    Returns:
        1-D array of percentile ranks in [0, 1] (average method for ties).
    """
    values = np.asarray(weight_strict_values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    # rankdata uses 'average' method → ties get the same rank
    ranks = rankdata(values, method="average")
    return ranks / n


def filter_training_rows_by_n(
    df: pd.DataFrame,
    n_col: str = "referee_n_partidos",
    min_n: int = MIN_MATCHES_STANDALONE,
) -> pd.DataFrame:
    """Filter a training DataFrame to only include rows where n >= min_n.

    Per REQ-8: referees with fewer than MIN_MATCHES_STANDALONE matches are
    excluded from the training split (they can remain in validation).

    Args:
        df: DataFrame with at least a column named ``n_col``.
        n_col: Column name holding the match count per referee.
        min_n: Minimum match count threshold (default: MIN_MATCHES_STANDALONE=8).

    Returns:
        Filtered DataFrame (subset of ``df``) preserving the original index.
    """
    return df[df[n_col] >= min_n]


@dataclass
class RefereeProfile:
    """
    Perfil bimodal de un arbitro. Los parametros llegan del feature dict
    (calculados por features_generator).
    """

    name: str
    n_matches: int = 0
    mu: np.ndarray = field(default_factory=lambda: np.array([22.0, 30.0]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0]))
    weights: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    is_shrunk: bool = False

    @property
    def mu_permissive(self) -> float:
        return float(self.mu[np.argmin(self.mu)])

    @property
    def mu_strict(self) -> float:
        return float(self.mu[np.argmax(self.mu)])

    @property
    def sigma_permissive(self) -> float:
        return float(self.sigma[np.argmin(self.mu)])

    @property
    def sigma_strict(self) -> float:
        return float(self.sigma[np.argmax(self.mu)])

    @property
    def weight_strict(self) -> float:
        return float(self.weights[np.argmax(self.mu)])

    def expected_fouls(self, strict_prob: float) -> float:
        permissive_prob = 1.0 - strict_prob
        idx_perm = int(np.argmin(self.mu))
        idx_strict = 1 - idx_perm
        return permissive_prob * self.mu[idx_perm] + strict_prob * self.mu[idx_strict]

    def sample_fouls(self, strict_prob: float, n: int = 1) -> np.ndarray:
        idx_perm = int(np.argmin(self.mu))
        idx_strict = 1 - idx_perm
        modes = np.random.binomial(1, strict_prob, size=n)
        samples = np.where(
            modes == 0,
            np.random.normal(self.mu[idx_perm], self.sigma[idx_perm], size=n),
            np.random.normal(self.mu[idx_strict], self.sigma[idx_strict], size=n),
        )
        return np.clip(np.round(samples), 0, 60).astype(int)

    @classmethod
    def from_contract(cls, nombre: str, arb_stats: dict) -> "RefereeProfile":
        """Construye un RefereeProfile desde estadisticas de arbitro."""
        mu_p = float(arb_stats.get("mu_permisivo", 22.0))
        mu_s = float(arb_stats.get("mu_estricto", 30.0))
        sig_p = float(arb_stats.get("sigma_permisivo", 4.0))
        sig_s = float(arb_stats.get("sigma_estricto", 4.0))
        w_s = float(arb_stats.get("peso_estricto", 0.5))
        n = int(arb_stats.get("partidos_arbitrados", 0))
        shrunk = bool(arb_stats.get("is_shrunk", True))
        return cls(
            name=nombre,
            n_matches=n,
            mu=np.array([mu_p, mu_s]),
            sigma=np.array([sig_p, sig_s]),
            weights=np.array([1.0 - w_s, w_s]),
            is_shrunk=shrunk,
        )


def _norm_ref_pair_delta_sum(x: float) -> float:
    """[-8, 8] aprox. → [0, 1] para el MLP."""
    c = max(-8.0, min(8.0, float(x)))
    return (c / 8.0 + 1.0) * 0.5


def _norm_pace_index(pace: float) -> float:
    """Ritmo esperado del partido → [0, 1] (típico ~18–42)."""
    return max(0.0, min(1.0, (float(pace) - 18.0) / 24.0))


class ModeSelector(nn.Module):
    """
    MLP que predice P(modo estricto) dado contexto del partido.

    Input (9 features):
        - is_derby (0/1)
        - rank_diff / 19
        - season_phase
        - home_is_top (0/1)
        - referee GMM weight_strict (tasa base)
        - aggressiveness_norm_total [0,1]
        - urgency_avg = (urgency_home + urgency_away) / 2
        - ref_pair_delta_sum normalizado [0,1]
        - pace_index normalizado [0,1]
    """

    N_CONTEXT_FEATURES = 9

    def __init__(self, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [24, 12]

        layers = []
        in_dim = self.N_CONTEXT_FEATURES
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RefereeProfiler:
    """
    Gestor de perfiles de arbitros para prediction_models.

    A diferencia de features_generator, este profiler NO ajusta GMMs desde historico.
    Recibe perfiles precalculados (desde features) y entrena/usa el ModeSelector.
    """

    def __init__(self):
        self.profiles: dict[str, RefereeProfile] = {}
        self.mode_selector = ModeSelector()
        self._optimizer: torch.optim.Adam | None = None
        # D3: training-time gate decision persisted in checkpoint.
        # When True, predict_mode() returns profile.weight_strict directly
        # without invoking the MLP (static passthrough / fallback mode).
        self.use_static_fallback: bool = False

    def register_profile(self, profile: RefereeProfile) -> None:
        """Registra un perfil precalculado."""
        self.profiles[profile.name] = profile

    def get_profile(self, referee_name: str) -> RefereeProfile:
        if referee_name in self.profiles:
            return self.profiles[referee_name]
        return RefereeProfile(name=referee_name)

    def build_context_vector(
        self,
        referee_name: str,
        is_derby: bool,
        rank_diff: float,
        season_phase: float,
        home_is_top: bool,
        *,
        aggressiveness_norm_total: float = 0.5,
        urgency_home: float = 0.5,
        urgency_away: float = 0.5,
        ref_pair_delta_sum: float = 0.0,
        pace_index_curr: float = 31.0,
    ) -> torch.Tensor:
        profile = self.get_profile(referee_name)
        u_h = float(urgency_home)
        u_a = float(urgency_away)
        urgency_avg = 0.5 * (u_h + u_a)
        return torch.tensor(
            [
                float(is_derby),
                float(rank_diff) / 19.0,
                float(season_phase),
                float(home_is_top),
                float(profile.weight_strict),
                max(0.0, min(1.0, float(aggressiveness_norm_total))),
                max(0.0, min(1.0, urgency_avg)),
                _norm_ref_pair_delta_sum(ref_pair_delta_sum),
                _norm_pace_index(pace_index_curr),
            ],
            dtype=torch.float32,
        )

    def predict_mode(
        self,
        referee_name: str,
        is_derby: bool,
        rank_diff: float,
        season_phase: float,
        home_is_top: bool,
        *,
        aggressiveness_norm_total: float = 0.5,
        urgency_home: float = 0.5,
        urgency_away: float = 0.5,
        ref_pair_delta_sum: float = 0.0,
        pace_index_curr: float = 31.0,
    ) -> float:
        # D3: static fallback — skip MLP entirely, return weight_strict directly.
        if self.use_static_fallback:
            return self.get_profile(referee_name).weight_strict

        ctx = self.build_context_vector(
            referee_name,
            is_derby,
            rank_diff,
            season_phase,
            home_is_top,
            aggressiveness_norm_total=aggressiveness_norm_total,
            urgency_home=urgency_home,
            urgency_away=urgency_away,
            ref_pair_delta_sum=ref_pair_delta_sum,
            pace_index_curr=pace_index_curr,
        ).unsqueeze(0)
        self.mode_selector.eval()
        with torch.no_grad():
            return float(self.mode_selector(ctx).item())

    def train_mode_selector(
        self,
        contexts: np.ndarray,
        actual_fouls: np.ndarray,
        referee_names: list[str],
        epochs: int = 100,
        lr: float = 0.001,
    ):
        """
        Entrena el ModeSelector usando datos historicos.
        Target: P(estricto) derivado de si las faltas reales fueron mas cercanas
        al modo estricto o al permisivo del perfil del arbitro.

        T16 ablation (2026-04-02): ModeSelector vs. static threshold
        ────────────────────────────────────────────────────────────
        After full retrain with 3-way split (train=2023-24, tune=2024-25,
        test=2025-26), the ModeSelector was evaluated against a static
        threshold (predict_mode returns referee_peso_estricto directly).

        Results on tune set (2024-25, n=377):
          NLL WITH ModeSelector  : 3.1500
          NLL WITHOUT ModeSelector: 3.1895
          NLL improvement        : 0.0395  ≥ 0.02 threshold

        Results on test set (2025-26, n=277):
          NLL WITH ModeSelector  : 3.1494
          NLL WITHOUT ModeSelector: 3.2111
          NLL improvement        : 0.0617  ≥ 0.02 threshold

        DECISION: **KEEP ModeSelector**.  The 100-epoch in-sample GMM-distance
        MLP provides a meaningful improvement above the 0.02 NLL threshold on
        both tune and test sets.  Simplifying to a static threshold would cost
        ~0.04–0.06 NLL.
        """
        targets = []
        for name, fouls in zip(referee_names, actual_fouls):
            profile = self.get_profile(name)
            dist_perm = abs(fouls - profile.mu_permissive)
            dist_strict = abs(fouls - profile.mu_strict)
            p_strict = dist_perm / (dist_perm + dist_strict + 1e-8)
            targets.append(p_strict)

        X = torch.tensor(contexts, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        self._optimizer = torch.optim.Adam(self.mode_selector.parameters(), lr=lr)

        self.mode_selector.train()
        for _ in range(epochs):
            pred = self.mode_selector(X)
            loss = nn.functional.binary_cross_entropy(pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        self.mode_selector.eval()

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        profiles_data = {}
        for name, p in self.profiles.items():
            profiles_data[name] = {
                "n_matches": p.n_matches,
                "mu": p.mu.tolist(),
                "sigma": p.sigma.tolist(),
                "weights": p.weights.tolist(),
                "is_shrunk": p.is_shrunk,
            }
        # D3: persist fallback flag alongside profiles so inference-time
        # load() can restore the gate decision without re-running gates.
        checkpoint = {
            "profiles": profiles_data,
            "use_static_fallback": self.use_static_fallback,
        }
        with open(path / "profiles.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

        torch.save(self.mode_selector.state_dict(), path / "mode_selector.pt")

    def load(self, path: str | Path):
        path = Path(path)

        profiles_path = path / "profiles.pkl"
        if profiles_path.exists():
            with open(profiles_path, "rb") as f:
                raw = pickle.load(f)

            # D8: backward compatibility — old format was a flat dict of profiles.
            # New format is {"profiles": {...}, "use_static_fallback": bool}.
            if isinstance(raw, dict) and "profiles" in raw:
                profiles_data = raw["profiles"]
                # Default to False for old checkpoints that lack the key (D8).
                self.use_static_fallback = bool(raw.get("use_static_fallback", False))
            else:
                # Legacy flat format: raw is profiles_data directly
                profiles_data = raw
                self.use_static_fallback = False

            for name, data in profiles_data.items():
                self.profiles[name] = RefereeProfile(
                    name=name,
                    n_matches=data["n_matches"],
                    mu=np.array(data["mu"]),
                    sigma=np.array(data["sigma"]),
                    weights=np.array(data["weights"]),
                    is_shrunk=data["is_shrunk"],
                )

        ms_path = path / "mode_selector.pt"
        if ms_path.exists():
            state = torch.load(ms_path, weights_only=True)
            first_w = state.get("net.0.weight")
            if first_w is None or first_w.shape[1] != ModeSelector.N_CONTEXT_FEATURES:
                raise ValueError(
                    "Checkpoint de ModeSelector incompatible con la arquitectura actual."
                )
            self.mode_selector.load_state_dict(state)

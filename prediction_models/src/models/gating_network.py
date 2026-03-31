"""
Meta-Layer: Dynamic Gating Network.

Learns context-dependent weights to combine PMFs from the 3 prediction layers.
The gating network receives:
    - Match context features (derby, rank diff, season phase, ...)
    - Uncertainty signals from each layer (entropy, variance, firing strength)
    - Referee data quality (n_matches for shrinkage awareness)

Outputs softmax weights [w_bayes, w_regression, w_fuzzy] that sum to 1.
The final PMF is: PMF_final = w1*PMF_bayes + w2*PMF_regression + w3*PMF_fuzzy.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn

from src.utils.distributions import FoulPMF, mixture_pmf, MAX_K


class GatingNetwork(nn.Module):
    """
    MLP that produces dynamic mixture weights for the 3 layers.

    Input features (17-dim):
      Bloque A — Contexto del partido:
        0: aggressiveness_matchup (normalized agresividad total)
        1: intensidad_esperada (0=baja, 0.5=media, 1=alta)
        2: riesgo_disciplinario (0=bajo, 0.5=medio, 1=alto)
        3: urgency_avg (average urgency of both teams)
        4: rank_diff_norm
        5: season_phase
        6: has_market_odds (0/1 — indica si hay cuotas reales disponibles)
      Bloque B — Fiabilidad de cada capa:
        7: referee_n_matches (normalized 0-1)
        8-10: entropy of Bayes / Regression / ANFIS
        11: std of Regression prediction
        12: max firing strength of ANFIS
        13: agreement score (similarity between layer outputs)
        14: foul_market_divergence (market vs model gap, 0 si sin cuotas faltas)
        15: xfouls_vs_raw_gap (expected vs raw fouls delta)
        16: extremism_index = |xfouls_total - 25| / 5
    """

    N_GATE_FEATURES = 17
    N_LAYERS = 3

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
        temperature: float = 1.0,
        min_weight: float = 0.15,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        self.temperature = temperature
        self.min_weight = float(min_weight)

        layers = []
        in_dim = self.N_GATE_FEATURES
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.N_LAYERS))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, N_GATE_FEATURES)
        returns: (batch, 3) softmax weights con piso minimo por capa
        """
        logits = self.net(x) / self.temperature
        raw = torch.softmax(logits, dim=-1)
        # Piso minimo: evita que una capa reciba 0% cuando el entrenamiento es pequenio
        min_w = float(np.clip(self.min_weight, 0.0, 1.0 / self.N_LAYERS - 1e-6))
        floored = raw * (1.0 - self.N_LAYERS * min_w) + min_w
        return floored


class DynamicEnsembleWeighter:
    """
    Manages the gating network training and inference.
    Combines outputs from Bayes, Regression, and ANFIS layers.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
        temperature: float = 1.0,
        lr: float = 0.001,
        epochs: int = 150,
        batch_size: int = 32,
        min_weight: float = 0.15,
        prior_mix: float = 0.35,
    ):
        self.gating = GatingNetwork(
            hidden_dims, dropout, temperature, min_weight=min_weight
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Prior global aprendido en validacion OOS (mezclado con pesos dinamicos)
        self._global_prior = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        self._prior_mix = float(prior_mix)

    def build_gate_features(
        self,
        match: dict,
        pmf_bayes: FoulPMF,
        pmf_regression: FoulPMF,
        pmf_anfis: FoulPMF,
        anfis_max_firing: float = 0.5,
    ) -> np.ndarray:
        """Build the 17-dim feature vector for the gating network."""

        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return dot / (norm + 1e-10)

        sim_br = cosine_sim(pmf_bayes.probs, pmf_regression.probs)
        sim_ba = cosine_sim(pmf_bayes.probs, pmf_anfis.probs)
        sim_ra = cosine_sim(pmf_regression.probs, pmf_anfis.probs)
        agreement = (sim_br + sim_ba + sim_ra) / 3.0

        ref_n = match.get("referee_n_matches", 50)

        agg_home = float(match.get("aggressiveness_volume_home", 0.5))
        agg_away = float(match.get("aggressiveness_volume_away", 0.5))
        agg_norm = match.get("aggressiveness_norm_total")
        if agg_norm is not None:
            aggressiveness_matchup = float(np.clip(agg_norm, 0.0, 1.0))
        else:
            aggressiveness_matchup = np.clip((agg_home + agg_away) / 2.0, 0.0, 1.0)

        _INTENSITY_MAP = {"baja": 0.0, "media": 0.5, "alta": 1.0}
        _RISK_MAP = {"bajo": 0.0, "medio": 0.5, "alto": 1.0}
        intensidad = _INTENSITY_MAP.get(
            str(match.get("intensidad_esperada", "media")).lower(), 0.5
        )
        riesgo = _RISK_MAP.get(
            str(match.get("riesgo_disciplinario", "medio")).lower(), 0.5
        )

        urgency_avg = (
            float(match.get("urgency_home", 0.5))
            + float(match.get("urgency_away", 0.5))
        ) / 2.0

        # foul_market_divergence: solo tiene señal cuando hay cuotas reales de faltas
        # (foul_market_implied_mean solo viene de odds_raw en prediccion live; en training es 25.0)
        has_odds = float(bool(match.get("has_market_odds", False)))
        foul_market_mean = float(match.get("foul_market_implied_mean", 25.0))
        model_mean = (pmf_bayes.mean + pmf_regression.mean + pmf_anfis.mean) / 3.0
        foul_market_divergence = (
            np.clip(abs(foul_market_mean - model_mean) / 10.0, 0.0, 1.0) * has_odds
        )

        xfouls_total = float(match.get("xfouls_home", 12.5)) + float(
            match.get("xfouls_away", 12.5)
        )
        raw_total = float(match.get("home_fouls_committed_curr", 12.0)) + float(
            match.get("away_fouls_committed_curr", 12.0)
        )
        xfouls_vs_raw = np.clip(abs(xfouls_total - raw_total) / 8.0, 0.0, 1.0)
        extremism_index = abs(xfouls_total - 25.0) / 5.0

        return np.array(
            [
                aggressiveness_matchup,
                intensidad,
                riesgo,
                urgency_avg,
                abs(match.get("rank_diff_norm", 0.0)),
                match.get("season_phase", 0.5),
                has_odds,  # sustituye foul_market_prob_over (era constante 0.5 en training)
                min(ref_n / 150.0, 1.0),
                pmf_bayes.entropy,
                pmf_regression.entropy,
                pmf_anfis.entropy,
                pmf_regression.std,
                anfis_max_firing,
                agreement,
                foul_market_divergence,
                xfouls_vs_raw,
                extremism_index,
            ],
            dtype=np.float64,
        )

    def predict_weights(
        self,
        match: dict,
        pmf_bayes: FoulPMF,
        pmf_regression: FoulPMF,
        pmf_anfis: FoulPMF,
        anfis_max_firing: float = 0.5,
    ) -> np.ndarray:
        """Returns [w_bayes, w_regression, w_anfis] weights."""
        features = self.build_gate_features(
            match, pmf_bayes, pmf_regression, pmf_anfis, anfis_max_firing
        )
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        self.gating.eval()
        with torch.no_grad():
            dyn = self.gating(x).squeeze(0).numpy()
        # Mezcla dinamico + prior global para mejorar estabilidad fuera de muestra
        mixed = (1.0 - self._prior_mix) * dyn + self._prior_mix * self._global_prior
        mixed = np.asarray(mixed, dtype=np.float64)
        mixed /= mixed.sum() + 1e-12
        return mixed

    def combine(
        self,
        match: dict,
        pmf_bayes: FoulPMF,
        pmf_regression: FoulPMF,
        pmf_anfis: FoulPMF,
        anfis_max_firing: float = 0.5,
    ) -> tuple[FoulPMF, np.ndarray]:
        """Combine PMFs using dynamic weights. Returns (combined_pmf, weights)."""
        weights = self.predict_weights(
            match, pmf_bayes, pmf_regression, pmf_anfis, anfis_max_firing
        )
        combined = mixture_pmf(
            [pmf_bayes, pmf_regression, pmf_anfis],
            weights,
        )
        return combined, weights

    def fit(
        self,
        matches: list[dict],
        pmfs_bayes: list[FoulPMF],
        pmfs_regression: list[FoulPMF],
        pmfs_anfis: list[FoulPMF],
        actual_fouls: list[int],
        anfis_firings: list[float] | None = None,
    ):
        """
        Train the gating network with a hybrid objective aligned with betting goals:
          - NLL (distribution sharpness/calibration)
          - MAE de la media esperada (precision del numero de faltas)
          - CRPS discreto (calidad global de la distribucion)
        """
        if anfis_firings is None:
            anfis_firings = [0.5] * len(matches)

        features_list = []
        pmf_probs_list = []  # (n_samples, 3, MAX_K)
        targets = []

        for i, m in enumerate(matches):
            feat = self.build_gate_features(
                m, pmfs_bayes[i], pmfs_regression[i], pmfs_anfis[i], anfis_firings[i]
            )
            features_list.append(feat)

            stacked = np.stack(
                [
                    pmfs_bayes[i].probs,
                    pmfs_regression[i].probs,
                    pmfs_anfis[i].probs,
                ]
            )
            pmf_probs_list.append(stacked)
            targets.append(actual_fouls[i])

        X = torch.tensor(np.array(features_list), dtype=torch.float32)
        P = torch.tensor(np.array(pmf_probs_list), dtype=torch.float32)  # (N, 3, MAX_K)
        y = torch.tensor(targets, dtype=torch.long)
        k = torch.arange(MAX_K, dtype=torch.float32).unsqueeze(0)  # (1, MAX_K)

        dataset = torch.utils.data.TensorDataset(X, P, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.gating.parameters(), lr=self.lr)
        smooth_l1 = nn.SmoothL1Loss(beta=2.0)
        self.gating.train()

        for epoch in range(self.epochs):
            for X_b, P_b, y_b in loader:
                weights = self.gating(X_b)  # (batch, 3)

                # Combined PMF: sum of weighted component PMFs
                # weights: (batch, 3, 1) * P_b: (batch, 3, MAX_K) -> sum over dim 1
                combined = (weights.unsqueeze(2) * P_b).sum(dim=1)  # (batch, MAX_K)

                # 1) NLL at the observed outcome
                probs_at_y = combined[torch.arange(len(y_b)), y_b]
                nll = -torch.log(probs_at_y + 1e-10).mean()

                # 2) MAE-like loss sobre la media esperada (alinea con error absoluto)
                expected = (combined * k).sum(dim=1)
                mae_mean = smooth_l1(expected, y_b.float())

                # 3) CRPS discreto
                cdf_pred = torch.cumsum(combined, dim=1)
                y_grid = y_b.unsqueeze(1).expand(-1, MAX_K)
                indicator = (k.expand(len(y_b), -1) >= y_grid).float()
                crps = torch.mean((cdf_pred - indicator) ** 2)

                # Entropy regularization: evita que el gating colapse a un solo layer
                gate_entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=1).mean()
                loss = 0.35 * nll + 0.40 * mae_mean + 0.25 * crps - 0.12 * gate_entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.gating.parameters(), max_norm=1.0)
                optimizer.step()

        # Prior global por desempeño individual OOS (menor error => mayor peso)
        maes = []
        nlls = []
        for comp in range(3):
            pmf_comp = P[:, comp, :]
            exp_comp = (pmf_comp * k).sum(dim=1)
            maes.append(torch.mean(torch.abs(exp_comp - y.float())).item())
            p_at_y = pmf_comp[torch.arange(len(y)), y]
            nlls.append(torch.mean(-torch.log(p_at_y + 1e-10)).item())

        inv_score = np.array(
            [1.0 / (maes[i] + 0.7 * nlls[i] + 1e-8) for i in range(3)],
            dtype=np.float64,
        )
        self._global_prior = inv_score / inv_score.sum()
        self.gating.eval()

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.gating.state_dict(),
                "global_prior": self._global_prior,
                "prior_mix": self._prior_mix,
            },
            path,
        )

    def load(self, path: str):
        payload = torch.load(path, weights_only=False)
        if not (isinstance(payload, dict) and "state_dict" in payload):
            raise ValueError("Formato de checkpoint de gating no soportado.")
        self.gating.load_state_dict(payload["state_dict"])
        gp = payload.get("global_prior")
        if gp is not None:
            self._global_prior = np.asarray(gp, dtype=np.float64)
            self._global_prior /= self._global_prior.sum() + 1e-12
        self._prior_mix = float(payload.get("prior_mix", self._prior_mix))

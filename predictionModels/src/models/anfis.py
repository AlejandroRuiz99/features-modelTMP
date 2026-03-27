"""
Layer 3: Adaptive Neuro-Fuzzy Inference System (ANFIS) - Takagi-Sugeno.

5-layer architecture implemented in PyTorch:
    Layer 1: Fuzzification (Gaussian membership functions, learnable)
    Layer 2: Rule activation (T-norm product)
    Layer 3: Normalization (firing strengths)
    Layer 4: Consequent (Sugeno first-order: linear functions of inputs)
    Layer 5: Defuzzification (weighted sum)

Linguistic variables:
    - Home aggressiveness (low/medium/high)
    - Away aggressiveness (low/medium/high)
    - Match intensity (low/medium/high)
    - Play style index (possession/direct/transition)
    - Referee mode (permissive/strict)

Output: expected foul count + estimated variance -> converted to PMF.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.distributions import FoulPMF, pmf_from_negbin, MAX_K


class GaussianMembershipFunction(nn.Module):
    """
    Parametric Gaussian membership function.
    mu(x) = exp(-(x - center)^2 / (2 * sigma^2))
    Both center and sigma are learnable.
    """

    def __init__(self, n_variables: int, n_mfs: int):
        """
        n_variables: number of input variables
        n_mfs: number of membership functions per variable
        """
        super().__init__()
        self.n_variables = n_variables
        self.n_mfs = n_mfs

        # Initialize centers spread across [0, 1] for each variable
        centers = torch.zeros(n_variables, n_mfs)
        for i in range(n_mfs):
            centers[:, i] = i / max(n_mfs - 1, 1)
        self.centers = nn.Parameter(centers)

        # Initialize sigmas
        initial_sigma = 1.0 / (2 * n_mfs)
        self.log_sigmas = nn.Parameter(
            torch.full((n_variables, n_mfs), np.log(initial_sigma))
        )

    @property
    def sigmas(self) -> torch.Tensor:
        return torch.exp(self.log_sigmas) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_variables) - normalized inputs in [0, 1]
        returns: (batch, n_variables, n_mfs) - membership degrees
        """
        x_expanded = x.unsqueeze(2)  # (batch, n_vars, 1)
        centers = self.centers.unsqueeze(0)  # (1, n_vars, n_mfs)
        sigmas = self.sigmas.unsqueeze(0)

        memberships = torch.exp(-0.5 * ((x_expanded - centers) / sigmas) ** 2)
        return memberships


class ANFISModel(nn.Module):
    """
    Full ANFIS architecture for Takagi-Sugeno fuzzy inference.
    
    For n_vars=5, n_mfs=3: total rules = 3^5 = 243 (full combinatorial).
    In practice, we use rule reduction to keep it manageable.
    """

    def __init__(
        self,
        n_variables: int = 5,
        n_mfs: int = 3,
        max_rules: int = 64,
        use_rule_dropout: bool = True,
        rule_dropout_rate: float = 0.1,
        output_center: float = 25.0,
        output_scale: float = 18.0,
        output_tanh_temp: float = 4.0,
    ):
        super().__init__()
        self.n_variables = n_variables
        self.n_mfs = n_mfs
        self.use_rule_dropout = use_rule_dropout
        self.rule_dropout_rate = rule_dropout_rate
        self.output_center = float(output_center)
        self.output_scale = float(output_scale)
        self.output_tanh_temp = float(output_tanh_temp)

        # Layer 1: Fuzzification
        self.membership = GaussianMembershipFunction(n_variables, n_mfs)

        # Generate rule indices (combinations of MF indices per variable)
        full_n_rules = n_mfs ** n_variables
        if full_n_rules <= max_rules:
            self.n_rules = full_n_rules
            self.rule_indices = self._generate_all_rules()
        else:
            self.n_rules = max_rules
            self.rule_indices = self._generate_sampled_rules(max_rules)

        # Layer 4: Consequent parameters (Sugeno order 1)
        # Bias = 0 => mu inicial = 26 + tanh(0)*15 = 26 (media La Liga)
        self.consequent_weights = nn.Parameter(
            torch.randn(self.n_rules, n_variables) * 0.1
        )
        self.consequent_bias = nn.Parameter(torch.zeros(self.n_rules))

        # Variance head for uncertainty estimation
        self.var_weights = nn.Parameter(torch.randn(self.n_rules, n_variables) * 0.05)
        self.var_bias = nn.Parameter(torch.full((self.n_rules,), 1.0))

    def _generate_all_rules(self) -> torch.Tensor:
        """Generate all possible rule combinations."""
        import itertools
        combos = list(itertools.product(range(self.n_mfs), repeat=self.n_variables))
        return torch.tensor(combos, dtype=torch.long)

    def _generate_sampled_rules(self, max_rules: int) -> torch.Tensor:
        """Sample a subset of rules using importance-based selection."""
        import itertools
        all_combos = list(itertools.product(range(self.n_mfs), repeat=self.n_variables))
        # Prioritize rules involving extreme MFs (0 and n_mfs-1)
        scores = []
        for combo in all_combos:
            score = sum(1 for c in combo if c == 0 or c == self.n_mfs - 1)
            scores.append(score)
        indices = np.argsort(scores)[::-1][:max_rules]
        selected = [all_combos[i] for i in indices]
        return torch.tensor(selected, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full ANFIS forward pass.
        
        x: (batch, n_variables) - normalized inputs
        returns: (mu, log_var) - predicted mean and log-variance of fouls
        """
        batch_size = x.shape[0]

        # Layer 1: Fuzzification
        memberships = self.membership(x)  # (batch, n_vars, n_mfs)

        # Layer 2: Rule activation (T-norm = product)
        # For each rule, multiply the selected memberships across variables
        rule_activations = torch.ones(batch_size, self.n_rules, device=x.device)
        for v in range(self.n_variables):
            mf_indices = self.rule_indices[:, v]  # (n_rules,)
            var_memberships = memberships[:, v, :]  # (batch, n_mfs)
            selected = var_memberships[:, mf_indices]  # (batch, n_rules)
            rule_activations = rule_activations * selected

        # Layer 3: Normalization
        activation_sum = rule_activations.sum(dim=1, keepdim=True) + 1e-10
        normalized_activations = rule_activations / activation_sum  # (batch, n_rules)

        if self.training and self.use_rule_dropout:
            mask = torch.bernoulli(
                torch.full_like(normalized_activations, 1 - self.rule_dropout_rate)
            )
            normalized_activations = normalized_activations * mask
            norm_sum = normalized_activations.sum(dim=1, keepdim=True) + 1e-10
            normalized_activations = normalized_activations / norm_sum

        # Layer 4: Consequent (Sugeno first-order)
        # y_r = bias_r + sum_v(w_rv * x_v)
        # (batch, n_vars) @ (n_vars, n_rules) -> (batch, n_rules) + (n_rules,)
        rule_outputs = x @ self.consequent_weights.t() + self.consequent_bias
        rule_var = x @ self.var_weights.t() + self.var_bias

        # Layer 5: Defuzzification (weighted sum)
        # Formulacion residual: mu = 25 + tanh(raw/4) * 18
        #   - Valor inicial (bias=0): mu = 25
        #   - Rango aprendible aproximado: [7, 43] para capturar extremos
        #   - tanh evita gradiente cero (a diferencia del clamp)
        raw_mu = (normalized_activations * rule_outputs).sum(dim=1)
        mu = self.output_center + torch.tanh(raw_mu / self.output_tanh_temp) * self.output_scale
        log_var = (normalized_activations * rule_var).sum(dim=1)

        return mu, log_var


class ANFISFoulPredictor:
    """
    High-level wrapper for ANFIS-based foul prediction.
    
    Handles feature extraction, normalization, training, and PMF conversion.
    """

    FEATURE_NAMES = [
        "aggressiveness_combined",
        "match_intensity",
        "play_style",
        "referee_discipline",
        "has_market_odds",
    ]

    def __init__(
        self,
        n_mfs: int = 3,
        max_rules: int = 64,
        lr: float = 0.005,
        epochs: int = 200,
        batch_size: int = 64,
        use_market_context: bool = True,
        output_center: float = 25.0,
        output_scale: float = 18.0,
        output_tanh_temp: float = 4.0,
        match_intensity_weights: Optional[list[float]] = None,
        play_style_weights: Optional[list[float]] = None,
    ):
        self.n_variables = len(self.FEATURE_NAMES)
        self.n_mfs = n_mfs
        self.use_market_context = use_market_context
        self.match_intensity_weights = tuple(
            match_intensity_weights or [0.25, 0.20, 0.25, 0.15, 0.15]
        )
        self.play_style_weights = tuple(play_style_weights or [0.40, 0.35, 0.25])
        self.model = ANFISModel(
            n_variables=self.n_variables,
            n_mfs=n_mfs,
            max_rules=max_rules,
            output_center=output_center,
            output_scale=output_scale,
            output_tanh_temp=output_tanh_temp,
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._feature_mins: Optional[np.ndarray] = None
        self._feature_maxs: Optional[np.ndarray] = None

    def _extract_features(self, match: dict) -> np.ndarray:
        """
        Extract 5 fuzzy linguistic variables from match data.
        Each captures a distinct dimension of match character.
        """
        agg_norm = match.get("aggressiveness_norm_total")
        if agg_norm is not None:
            aggressiveness_combined = float(np.clip(agg_norm, 0.0, 1.0))
        else:
            agg_home = float(match.get("aggressiveness_volume_home", 0.5))
            agg_away = float(match.get("aggressiveness_volume_away", 0.5))
            aggressiveness_combined = (agg_home + agg_away) / 2.0

        rank_diff = abs(match.get("home_rank_curr", 10) - match.get("away_rank_curr", 10))
        is_derby = float(match.get("is_derby", False))
        urgency_avg = (
            float(match.get("urgency_home", 0.5))
            + float(match.get("urgency_away", 0.5))
        ) / 2.0
        momentum_gap = abs(
            float(match.get("momentum_home", 0.5))
            - float(match.get("momentum_away", 0.5))
        )
        fatigue_avg = (
            float(match.get("fatigue_home", 0.2))
            + float(match.get("fatigue_away", 0.2))
        ) / 2.0
        w_derby, w_rank, w_urg, w_mom, w_rest = self.match_intensity_weights
        match_intensity = (
            is_derby * w_derby
            + rank_diff / 19.0 * w_rank
            + urgency_avg * w_urg
            + momentum_gap * w_mom
            + (1.0 - fatigue_avg) * w_rest
        )

        possession_home = float(match.get("home_possession", 0.5))
        pace_index = float(match.get("pace_index_curr", 30.0))
        pace_norm = np.clip((pace_index - 20.0) / 25.0, 0.0, 1.0)
        has_odds = float(bool(match.get("has_market_odds", False)))
        # Cuando hay cuotas reales, ou_over aporta señal; si no, se fija a 0.5 (neutro)
        ou_over_raw = float(match.get("market_ou25_over_prob", 0.5))
        ou_over = ou_over_raw * has_odds + 0.5 * (1.0 - has_odds)
        w_pos, w_pace, w_ou = self.play_style_weights
        play_style = np.clip(
            (1.0 - possession_home) * w_pos + pace_norm * w_pace + ou_over * w_ou,
            0.0, 1.0,
        )

        referee_discipline = float(match.get("referee_strict_prob", 0.5))

        return np.array([
            aggressiveness_combined,
            match_intensity,
            play_style,
            referee_discipline,
            has_odds,   # indica al ANFIS si el contexto de mercado es fiable
        ], dtype=np.float64)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._feature_mins = X.min(axis=0)
            self._feature_maxs = X.max(axis=0)
        ranges = self._feature_maxs - self._feature_mins
        ranges[ranges < 1e-8] = 1.0
        return (X - self._feature_mins) / ranges

    def fit(self, matches: list[dict]):
        """Train the ANFIS model."""
        X_raw = np.array([self._extract_features(m) for m in matches])
        X = self._normalize(X_raw, fit=True)
        y = np.array([m["fouls_total"] for m in matches], dtype=np.float64)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        huber = nn.HuberLoss(delta=5.0)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                mu, log_var = self.model(X_batch)
                # Huber loss sobre la media: mas estable que Gaussian NLL
                loss_mu = huber(mu, y_batch)
                # Penalizacion de varianza: evita que se colapse a cero
                var = torch.exp(log_var) + 1e-6
                loss_var = torch.relu(2.0 - var).mean()
                loss = loss_mu + 0.1 * loss_var
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        self.model.eval()

    def predict_params(self, match: dict) -> tuple[float, float]:
        """Returns (mu, variance) for a single match."""
        X_raw = self._extract_features(match).reshape(1, -1)
        X = self._normalize(X_raw)
        X_t = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            mu, log_var = self.model(X_t)
            return float(mu.item()), float(torch.exp(log_var).item())

    def predict_pmf(self, match: dict) -> FoulPMF:
        """Convert ANFIS output to a full PMF using NegBin approximation."""
        mu, var = self.predict_params(match)
        mu = max(mu, 7.0)   # rango minimo [7, 43] por la formulacion tanh
        var = max(var, mu + 0.1)

        # NegBin: var = mu + alpha * mu^2  =>  alpha = (var - mu) / mu^2
        alpha = max((var - mu) / (mu ** 2 + 1e-8), 1e-4)
        return pmf_from_negbin(mu, alpha)

    def predict(self, match: dict) -> dict:
        mu, var = self.predict_params(match)
        pmf = self.predict_pmf(match)
        return {
            "mu": mu,
            "variance": var,
            "pmf": pmf,
            "expected_fouls": pmf.mean,
            "std_fouls": pmf.std,
        }

    def get_rule_importance(self, match: dict) -> list[tuple[str, float]]:
        """
        Get the top activated rules and their firing strengths for interpretability.
        Returns list of (rule_description, activation_strength).
        """
        X_raw = self._extract_features(match).reshape(1, -1)
        X = self._normalize(X_raw)
        X_t = torch.tensor(X, dtype=torch.float32)

        labels = ["low", "med", "high"]
        self.model.eval()
        with torch.no_grad():
            memberships = self.model.membership(X_t)
            rule_acts = torch.ones(1, self.model.n_rules)
            for v in range(self.model.n_variables):
                mf_idx = self.model.rule_indices[:, v]
                selected = memberships[0, v, mf_idx]
                rule_acts[0] *= selected

            total = rule_acts.sum() + 1e-10
            normalized = (rule_acts / total).squeeze(0)

        results = []
        for r in range(self.model.n_rules):
            strength = float(normalized[r].item())
            if strength > 0.01:
                parts = []
                for v in range(self.model.n_variables):
                    mf_idx = int(self.model.rule_indices[r, v].item())
                    parts.append(f"{self.FEATURE_NAMES[v]}={labels[mf_idx]}")
                desc = " AND ".join(parts)
                results.append((desc, strength))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]

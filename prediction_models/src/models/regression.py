"""
Layer 2: Statistical Regression for Foul Count Prediction.

Uses Negative Binomial GLM (handles overdispersion common in foul counts).
Referee mode enters as a log-offset on the mean.
Outputs a full PMF P(F=k) for k=0..60.

Two sub-models:
  - Total fouls model: predicts F_total = F_home + F_away
  - Per-team models: predicts F_home and F_away separately (for team-level markets)
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
from scipy.stats import nbinom, poisson

from src.utils.distributions import FoulPMF, MAX_K


class NegBinRegressor(nn.Module):
    """
    Negative Binomial regression implemented in PyTorch for GPU compatibility
    and end-to-end gradient flow.

    Parameterization: mean-dispersion
        mu = exp(X @ beta + offset)
        var = mu + alpha * mu^2
    where alpha > 0 is the dispersion parameter.
    """

    def __init__(
        self,
        n_features: int,
        regularization: float = 0.01,
        intercept_init: float = 3.2,
    ):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.intercept = nn.Parameter(torch.tensor(float(intercept_init)))
        self._log_alpha = nn.Parameter(torch.tensor(0.0))
        self.regularization = regularization

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self._log_alpha) + 1e-6

    def forward(
        self,
        X: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns log(mu) for each observation.
        X: (batch, n_features)
        offset: (batch,) optional log-offset (e.g. from referee mode)
        """
        log_mu = X @ self.beta + self.intercept
        if offset is not None:
            log_mu = log_mu + offset
        return log_mu

    def neg_log_likelihood(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Negative log-likelihood of the Negative Binomial.
        y: (batch,) observed foul counts.
        """
        log_mu = self.forward(X, offset)
        mu = torch.exp(log_mu)
        alpha = self.alpha

        r = 1.0 / alpha
        p = r / (r + mu)

        nll = -(
            torch.lgamma(y + r)
            - torch.lgamma(r)
            - torch.lgamma(y + 1)
            + r * torch.log(p)
            + y * torch.log(1 - p + 1e-10)
        )

        reg = self.regularization * torch.sum(self.beta**2)
        return nll.mean() + reg


class FoulRegressionPredictor:
    """
    Wraps the NegBin regressor for training and prediction.

    Features (total model):
        0: home_avg_fouls_committed_hist
        1: home_avg_fouls_suffered_hist
        2: away_avg_fouls_committed_hist
        3: away_avg_fouls_suffered_hist
        4: home_avg_fouls_committed_curr
        5: away_avg_fouls_committed_curr
        6: rank_diff (normalized)
        7: is_derby (0/1)
        8: home_possession (if available, else 0.5)
        9: home_xg_diff (if available, else 0)
    """

    BASE_FEATURE_NAMES = [
        "home_fouls_committed_avg",
        "home_fouls_suffered_avg",
        "away_fouls_committed_avg",
        "away_fouls_suffered_avg",
        "home_fouls_committed_curr",
        "away_fouls_committed_curr",
        "rank_diff_norm",
        "is_derby",
        "home_possession",
        "xg_diff",
    ]

    REF_FEATURE_NAMES = [
        "ref_home_delta",
        "ref_away_delta",
        "ref_pair_delta_sum",
        "ref_pair_samples",
    ]

    EXTRA_FEATURE_NAMES = [
        # has_market_odds va primero — actua como multiplicador para las features siguientes
        "has_market_odds",
        "market_home_win_prob_masked",
        "market_draw_prob_masked",
        "market_away_win_prob_masked",
        "market_ou25_over_prob_masked",
        "market_favorite_prob_masked",
        "market_balance_masked",
        "pace_index_curr",
        "home_shots_curr",
        "away_shots_curr",
        "home_corners_curr",
        "away_corners_curr",
    ]

    CONTEXT_FEATURE_NAMES = [
        "xfouls_home",
        "xfouls_away",
        "fouls_provoked_home",
        "fouls_provoked_away",
        "forma_fouls_home",
        "forma_fouls_away",
        "urgency_home",
        "urgency_away",
        "momentum_home",
        "momentum_away",
        "days_rest_home",
        "days_rest_away",
        "xfouls_factor_home",
        "xfouls_factor_away",
    ]

    def __init__(
        self,
        lr: float = 0.005,
        epochs: int = 300,
        regularization: float = 0.01,
        intercept_init: float = 3.2,
        referee_offset_reference: float = 26.0,
        use_extended_features: bool = True,
        use_ref_team_features: bool = True,
        use_context_features: bool = True,
    ):
        self.use_extended_features = use_extended_features
        self.use_ref_team_features = use_ref_team_features
        self.use_context_features = use_context_features
        self.feature_names = list(self.BASE_FEATURE_NAMES)
        if self.use_ref_team_features:
            self.feature_names += list(self.REF_FEATURE_NAMES)
        if self.use_extended_features:
            self.feature_names += list(self.EXTRA_FEATURE_NAMES)
        if self.use_context_features:
            self.feature_names += list(self.CONTEXT_FEATURE_NAMES)
        self.n_features = len(self.feature_names)
        self.model = NegBinRegressor(
            self.n_features,
            regularization,
            intercept_init=intercept_init,
        )
        self.lr = lr
        self.epochs = epochs
        self.referee_offset_reference = float(referee_offset_reference)
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def _extract_features(self, match: dict) -> np.ndarray:
        """Extract feature vector from a match dict."""
        values = [
            match.get("home_fouls_committed_avg", 12.0),
            match.get("home_fouls_suffered_avg", 12.0),
            match.get("away_fouls_committed_avg", 12.0),
            match.get("away_fouls_suffered_avg", 12.0),
            match.get("home_fouls_committed_curr", 12.0),
            match.get("away_fouls_committed_curr", 12.0),
            match.get("rank_diff_norm", 0.0),
            float(match.get("is_derby", False)),
            match.get("home_possession", 0.5),
            match.get("xg_diff", 0.0),
        ]
        if self.use_ref_team_features:
            values += [
                match.get("ref_home_delta", 0.0),
                match.get("ref_away_delta", 0.0),
                match.get("ref_pair_delta_sum", 0.0),
                match.get("ref_pair_samples", 0.0),
            ]
        if self.use_extended_features:
            has_odds = float(bool(match.get("has_market_odds", False)))
            values += [
                has_odds,
                match.get("market_home_win_prob", 1 / 3) * has_odds,
                match.get("market_draw_prob", 1 / 3) * has_odds,
                match.get("market_away_win_prob", 1 / 3) * has_odds,
                match.get("market_ou25_over_prob", 0.5) * has_odds,
                match.get("market_favorite_prob", 1 / 3) * has_odds,
                match.get("market_balance", 1.0) * has_odds,
                match.get("pace_index_curr", 30.0),
                match.get("home_shots_curr", 11.0),
                match.get("away_shots_curr", 11.0),
                match.get("home_corners_curr", 4.5),
                match.get("away_corners_curr", 4.5),
            ]
        if self.use_context_features:
            values += [
                match.get("xfouls_home", 12.5),
                match.get("xfouls_away", 12.5),
                match.get("fouls_provoked_home", 12.0),
                match.get("fouls_provoked_away", 12.0),
                match.get("forma_fouls_home", 12.0),
                match.get("forma_fouls_away", 12.0),
                match.get("urgency_home", 0.5),
                match.get("urgency_away", 0.5),
                match.get("momentum_home", 0.5),
                match.get("momentum_away", 0.5),
                match.get("days_rest_home", 7.0),
                match.get("days_rest_away", 7.0),
                match.get("xfouls_factor_home", 1.0),
                match.get("xfouls_factor_away", 1.0),
            ]
        return np.array(values, dtype=np.float64)

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._feature_means = X.mean(axis=0)
            self._feature_stds = X.std(axis=0) + 1e-8
        return (X - self._feature_means) / self._feature_stds

    def fit(self, matches: list[dict]):
        """Train the NegBin regressor on historical match data."""
        X_raw = np.array([self._extract_features(m) for m in matches])
        X = self._standardize(X_raw, fit=True)
        y = np.array([m["fouls_total"] for m in matches], dtype=np.float64)

        ref_base = max(self.referee_offset_reference, 1.0)
        offsets = np.array(
            [
                np.log(max(m.get("referee_expected_fouls", ref_base), 1.0))
                - np.log(ref_base)
                for m in matches
            ]
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        off_t = torch.tensor(offsets, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for epoch in range(self.epochs):
            loss = self.model.neg_log_likelihood(X_t, y_t, off_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()

    def predict_params(self, match: dict) -> tuple[float, float]:
        """Returns (mu, alpha) of the NegBin for a single match."""
        X_raw = self._extract_features(match).reshape(1, -1)
        X = self._standardize(X_raw)
        X_t = torch.tensor(X, dtype=torch.float32)

        ref_base = max(self.referee_offset_reference, 1.0)
        ref_expected = match.get("referee_expected_fouls", ref_base)
        offset = torch.tensor(
            [np.log(max(ref_expected, 1.0)) - np.log(ref_base)], dtype=torch.float32
        )

        self.model.eval()
        with torch.no_grad():
            log_mu = self.model(X_t, offset)
            mu = float(torch.exp(log_mu).item())
            alpha = float(self.model.alpha.item())

        return mu, alpha

    def predict_pmf(self, match: dict) -> FoulPMF:
        """Predict full PMF over foul counts."""
        mu, alpha = self.predict_params(match)
        r = 1.0 / alpha
        p = r / (r + mu)
        probs = nbinom.pmf(np.arange(MAX_K), n=r, p=p)
        return FoulPMF(probs=probs)

    def predict(self, match: dict) -> dict:
        mu, alpha = self.predict_params(match)
        pmf = self.predict_pmf(match)
        return {
            "mu": mu,
            "alpha": alpha,
            "pmf": pmf,
            "expected_fouls": pmf.mean,
            "std_fouls": pmf.std,
        }


class TeamFoulRegressor:
    """
    Separate regressors for home and away fouls.
    Allows predicting team-level over/under markets.

    Uses the same NegBin framework but with team-specific targets.
    """

    def __init__(self, **kwargs):
        self.home_model = FoulRegressionPredictor(**kwargs)
        self.away_model = FoulRegressionPredictor(**kwargs)

    def fit(self, matches: list[dict]):
        home_matches = [{**m, "fouls_total": m["fouls_home"]} for m in matches]
        away_matches = [{**m, "fouls_total": m["fouls_away"]} for m in matches]
        self.home_model.fit(home_matches)
        self.away_model.fit(away_matches)

    def predict(self, match: dict) -> dict:
        home_pred = self.home_model.predict({**match, "fouls_total": 0})
        away_pred = self.away_model.predict({**match, "fouls_total": 0})
        return {
            "home": home_pred,
            "away": away_pred,
            "total_expected": home_pred["expected_fouls"] + away_pred["expected_fouls"],
        }

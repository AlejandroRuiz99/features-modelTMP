"""
Ensemble Orchestrator.

Coordinates all prediction layers and produces final output:
    1. Referee GMM profiling
    2. Layer 1: Naive Bayes interval prediction
    3. Layer 2: Negative Binomial regression
    4. Layer 3: ANFIS Takagi-Sugeno
    5. Dynamic gating network combination
    6. PMF -> market conversion (Over/Under, intervals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.referee_gmm import RefereeProfiler
from src.models.naive_bayes import NaiveBayesFoulPredictor
from src.models.regression import FoulRegressionPredictor, TeamFoulRegressor
from src.models.anfis import ANFISFoulPredictor
from src.models.gating_network import DynamicEnsembleWeighter
from src.utils.distributions import FoulPMF, scale_pmf_variance, tilt_pmf_to_mean


@dataclass
class MatchPrediction:
    """Complete prediction output for a single match."""

    match_info: dict
    pmf_total: FoulPMF
    pmf_bayes: FoulPMF
    pmf_regression: FoulPMF
    pmf_anfis: FoulPMF
    weights: np.ndarray
    expected_fouls: float
    over_under: dict[float, tuple[float, float]] = field(default_factory=dict)
    intervals: dict[str, float] = field(default_factory=dict)
    referee_strict_prob: float = 0.5
    layer_details: dict = field(default_factory=dict)


class FoulPredictionEnsemble:
    """
    Main orchestrator that ties all components together.
    
    Usage:
        ensemble = FoulPredictionEnsemble()
        ensemble.fit(training_matches, referee_data, ...)
        prediction = ensemble.predict(match_dict)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        rec_cfg = self.config.get("reconciliation", {})
        # <1.0 => ancla más el total (se mueve menos)
        # >1.0 => permite mover más el total
        self.reconcile_total_variance_scale = float(
            rec_cfg.get("total_variance_scale", 0.6)
        )

        self.referee_profiler = RefereeProfiler()

        self.layer_bayes = NaiveBayesFoulPredictor(
            n_foul_clusters=self.config.get("naive_bayes", {}).get("n_clusters", 5),
            foul_percentiles=self.config.get("naive_bayes", {}).get("percentiles"),
        )

        reg_cfg = self.config.get("regression", {})
        self.layer_regression = FoulRegressionPredictor(
            regularization=reg_cfg.get("regularization", 0.01),
            intercept_init=reg_cfg.get("intercept_init", 3.2),
            referee_offset_reference=reg_cfg.get("referee_offset_reference", 26.0),
            use_extended_features=reg_cfg.get("use_extended_features", True),
            use_ref_team_features=reg_cfg.get("use_ref_team_features", True),
            use_context_features=reg_cfg.get("use_context_features", True),
        )

        anfis_cfg = self.config.get("anfis", {})
        self.layer_anfis = ANFISFoulPredictor(
            n_mfs=anfis_cfg.get("n_membership_functions", 3),
            max_rules=64,
            lr=anfis_cfg.get("lr", 0.005),
            epochs=anfis_cfg.get("epochs", 200),
            batch_size=anfis_cfg.get("batch_size", 64),
            use_market_context=anfis_cfg.get("use_market_context", True),
            output_center=anfis_cfg.get("output_center", 25.0),
            output_scale=anfis_cfg.get("output_scale", 18.0),
            output_tanh_temp=anfis_cfg.get("output_tanh_temp", 4.0),
            match_intensity_weights=anfis_cfg.get("match_intensity_weights"),
            play_style_weights=anfis_cfg.get("play_style_weights"),
        )

        gate_cfg = self.config.get("gating_network", {})
        self.weighter = DynamicEnsembleWeighter(
            hidden_dims=gate_cfg.get("hidden_dims", [32, 16]),
            dropout=gate_cfg.get("dropout", 0.15),
            temperature=gate_cfg.get("temperature", 1.0),
            lr=gate_cfg.get("lr", 0.001),
            min_weight=gate_cfg.get("min_weight", 0.05),
            prior_mix=gate_cfg.get("prior_mix", 0.35),
        )

        self.team_regressor: Optional[TeamFoulRegressor] = None
        self._bias_correction: float = float(
            self.config.get("bias_correction", 0.0)
        )
        self._variance_posthoc_scale: float = float(
            self.config.get("variance_posthoc_scale", 1.0)
        )
        self._is_fitted = False

    def fit(
        self,
        matches: list[dict],
        team_avg_committed: dict[str, float],
        team_avg_suffered: dict[str, float],
        team_avg_rank: dict[str, float],
        fit_team_models: bool = True,
        gating_matches: Optional[list[dict]] = None,
    ):
        """
        Entrena todos los componentes del ensemble.

        Staged training:
            1. Registrar perfiles GMM de arbitros desde los feature dicts
               (los parametros GMM vienen pre-calculados en featuresGenerator)
            2. Enriquecer matches con datos del arbitro
            3. Entrenar cada capa independientemente
            4. Generar predicciones para el gating network:
               - Si se pasan gating_matches (OUT-OF-SAMPLE, e.g. temporada actual),
                 el gating se entrena con esos - evita sobreajuste in-sample.
               - Si no, usa las mismas matches (in-sample, solo debug/warmup).
            5. Entrenar el gating network
            6. Entrenar el ModeSelector del arbitro

        Args:
            matches: Feature dicts de partidos historicos con fouls_total como label.
                     Cargados desde Parquet o inyectados por API.
            gating_matches: Partidos out-of-sample para entrenar el gating network.
        """
        # Stage 1: Registrar perfiles GMM desde los feature dicts
        # Los parametros GMM llegan ya calculados desde featuresGenerator.
        self._register_profiles_from_features(matches)
        if gating_matches:
            self._register_profiles_from_features(gating_matches)

        # Stage 2: Enrich matches with referee info
        enriched = [self._enrich_match(m) for m in matches]

        # Stage 3: Fit individual layers
        self.layer_bayes.fit(enriched, team_avg_committed, team_avg_suffered, team_avg_rank)
        self.layer_regression.fit(enriched)
        self.layer_anfis.fit(enriched)

        if fit_team_models:
            self.team_regressor = TeamFoulRegressor()
            self.team_regressor.fit(enriched)

        # Stage 4: Elegir fuente de datos para el gating
        # Out-of-sample si se proporcionan, in-sample como fallback
        if gating_matches:
            gating_enriched = [self._enrich_match(m) for m in gating_matches]
        else:
            gating_enriched = enriched

        pmfs_b, pmfs_r, pmfs_a = [], [], []
        firings = []
        actual = []

        for m in gating_enriched:
            pmfs_b.append(self.layer_bayes.predict_pmf(m))
            pmfs_r.append(self.layer_regression.predict_pmf(m))
            pmfs_a.append(self.layer_anfis.predict_pmf(m))
            actual.append(m["fouls_total"])

            rules = self.layer_anfis.get_rule_importance(m)
            firings.append(rules[0][1] if rules else 0.5)

        # Stage 5: Fit gating network
        self.weighter.fit(
            gating_enriched, pmfs_b, pmfs_r, pmfs_a, actual, firings
        )

        # Train referee mode selector
        contexts = []
        ref_names = []
        ref_actuals = []
        for m in enriched:
            ctx = self.referee_profiler.build_context_vector(
                m.get("referee", "unknown"),
                m.get("is_derby", False),
                abs(m.get("home_rank_curr", 10) - m.get("away_rank_curr", 10)),
                m.get("season_phase", 0.5),
                m.get("home_rank_curr", 10) <= 5,
                aggressiveness_norm_total=float(
                    m.get("aggressiveness_norm_total", 0.5)
                ),
                urgency_home=float(m.get("urgency_home", 0.5)),
                urgency_away=float(m.get("urgency_away", 0.5)),
                ref_pair_delta_sum=float(m.get("ref_pair_delta_sum", 0.0)),
                pace_index_curr=float(m.get("pace_index_curr", 31.0)),
            )
            contexts.append(ctx.numpy())
            ref_names.append(m.get("referee", "unknown"))
            ref_actuals.append(m["fouls_total"])

        self.referee_profiler.train_mode_selector(
            np.array(contexts), np.array(ref_actuals), ref_names
        )

        # Stage 6: Calibración del bias.
        # Usamos el mismo set de gating (out-of-sample si se pasó, sino in-sample)
        # para medir la desviación sistemática y corregirla en predict().
        # Si el usuario ya pasó un bias_correction en config, se respeta.
        if self.config.get("bias_correction") is None:
            self._bias_correction = self._estimate_bias(gating_enriched)
        # else: conservar el valor de config

        if self.config.get("variance_posthoc_scale") is None:
            self._variance_posthoc_scale = self._estimate_variance_scale(gating_enriched)

        self._is_fitted = True

    def _estimate_bias(self, matches: list[dict]) -> float:
        """
        Calcula el bias medio (pred - real) sobre un conjunto de matches
        para usarlo como corrección post-hoc en predict().
        """
        if not matches:
            return 0.0
        diffs = []
        for m in matches:
            try:
                pmf_b = self.layer_bayes.predict_pmf(m)
                pmf_r = self.layer_regression.predict_pmf(m)
                pmf_a = self.layer_anfis.predict_pmf(m)
                rules = self.layer_anfis.get_rule_importance(m)
                max_firing = rules[0][1] if rules else 0.5
                pmf_combined, _ = self.weighter.combine(m, pmf_b, pmf_r, pmf_a, max_firing)
                diffs.append(pmf_combined.mean - m["fouls_total"])
            except Exception:
                continue
        return float(np.mean(diffs)) if diffs else 0.0

    def _estimate_variance_scale(self, matches: list[dict]) -> float:
        """
        Estima factor post-hoc de varianza para corregir compresión excesiva.
        Escala objetivo = var(real) / var(pred_mean).
        """
        if len(matches) < 10:
            return 1.0
        preds = []
        reals = []
        for m in matches:
            try:
                pmf_b = self.layer_bayes.predict_pmf(m)
                pmf_r = self.layer_regression.predict_pmf(m)
                pmf_a = self.layer_anfis.predict_pmf(m)
                rules = self.layer_anfis.get_rule_importance(m)
                max_firing = rules[0][1] if rules else 0.5
                pmf_combined, _ = self.weighter.combine(m, pmf_b, pmf_r, pmf_a, max_firing)
                preds.append(float(pmf_combined.mean))
                reals.append(float(m["fouls_total"]))
            except Exception:
                continue
        if len(preds) < 10:
            return 1.0
        var_pred = float(np.var(preds))
        var_real = float(np.var(reals))
        if var_pred <= 1e-8:
            return 1.0
        scale = var_real / var_pred
        return float(np.clip(scale, 0.6, 2.5))

    def _register_profiles_from_features(self, feature_dicts: list[dict]) -> None:
        """
        Registra perfiles GMM de arbitros desde los feature dicts.

        Los parametros GMM (mu_permisivo, sigma_permisivo, mu_estricto, sigma_estricto,
        peso_estricto) ya vienen calculados por featuresGenerator.
        """
        from src.models.referee_gmm import RefereeProfile
        import numpy as np

        for m in feature_dicts:
            ref_name = m.get("referee", "unknown")
            if not ref_name or ref_name in self.referee_profiler.profiles:
                continue
            mu_p = float(m.get("referee_mu_permisivo", 22.0))
            mu_s = float(m.get("referee_mu_estricto", 30.0))
            sig_p = float(m.get("referee_sigma_permisivo", 4.0))
            sig_s = float(m.get("referee_sigma_estricto", 4.0))
            w_s = float(m.get("referee_peso_estricto", 0.5))
            n = int(m.get("referee_n_partidos", 0))
            profile = RefereeProfile(
                name=ref_name,
                n_matches=n,
                mu=np.array([mu_p, mu_s]),
                sigma=np.array([sig_p, sig_s]),
                weights=np.array([1.0 - w_s, w_s]),
            )
            self.referee_profiler.register_profile(profile)

    def _enrich_match(self, match: dict) -> dict:
        """
        Enriquece el feature dict con datos del perfil GMM del arbitro.

        Los parametros GMM ya vienen en el feature dict.
        Solo calcula referee_strict_prob via el ModeSelector y los campos derivados.
        """
        m = dict(match)
        ref_name = m.get("referee", "unknown")
        profile = self.referee_profiler.get_profile(ref_name)

        is_derby = bool(m.get("is_derby", False))
        rank_diff = abs(m.get("home_rank_curr", 10) - m.get("away_rank_curr", 10))
        season_phase = float(m.get("season_phase", 0.5))
        home_is_top = m.get("home_rank_curr", 10) <= 5

        strict_prob = self.referee_profiler.predict_mode(
            ref_name,
            is_derby,
            rank_diff,
            season_phase,
            home_is_top,
            aggressiveness_norm_total=float(
                m.get("aggressiveness_norm_total", 0.5)
            ),
            urgency_home=float(m.get("urgency_home", 0.5)),
            urgency_away=float(m.get("urgency_away", 0.5)),
            ref_pair_delta_sum=float(m.get("ref_pair_delta_sum", 0.0)),
            pace_index_curr=float(m.get("pace_index_curr", 31.0)),
        )

        m["referee_strict_prob"] = strict_prob
        m["referee_expected_fouls"] = profile.expected_fouls(strict_prob)
        m["referee_mu_permissive"] = profile.mu_permissive
        m["referee_mu_strict"] = profile.mu_strict
        m["referee_n_matches"] = profile.n_matches
        m["referee_mode"] = 1 if strict_prob > 0.5 else 0
        m["rank_diff_norm"] = float(m.get("rank_diff_norm", rank_diff / 19.0))

        return m

    def predict(self, match: dict) -> MatchPrediction:
        """Generate a complete prediction for a single match."""
        enriched = self._enrich_match(match)

        # Layer predictions
        pred_bayes = self.layer_bayes.predict(enriched)
        pred_reg = self.layer_regression.predict(enriched)
        pred_anfis = self.layer_anfis.predict(enriched)

        pmf_b = pred_bayes["pmf"]
        pmf_r = pred_reg["pmf"]
        pmf_a = pred_anfis["pmf"]

        rules = self.layer_anfis.get_rule_importance(enriched)
        max_firing = rules[0][1] if rules else 0.5

        # Gating combination
        pmf_combined, weights = self.weighter.combine(
            enriched, pmf_b, pmf_r, pmf_a, max_firing
        )

        # Bias correction: desplazamos el PMF para eliminar sesgo sistemático
        if self._bias_correction != 0.0:
            pmf_combined = tilt_pmf_to_mean(
                pmf_combined, pmf_combined.mean - self._bias_correction
            )

        if abs(self._variance_posthoc_scale - 1.0) > 1e-6:
            pmf_combined = scale_pmf_variance(pmf_combined, self._variance_posthoc_scale)

        # Market conversions
        ou_table = pmf_combined.over_under_table()
        interval_table = pmf_combined.interval_table()

        return MatchPrediction(
            match_info=enriched,
            pmf_total=pmf_combined,
            pmf_bayes=pmf_b,
            pmf_regression=pmf_r,
            pmf_anfis=pmf_a,
            weights=weights,
            expected_fouls=pmf_combined.mean,
            over_under=ou_table,
            intervals=interval_table,
            referee_strict_prob=enriched["referee_strict_prob"],
            layer_details={
                "bayes": pred_bayes,
                "regression": pred_reg,
                "anfis": pred_anfis,
                "top_rules": rules[:5],
            },
        )

    def _heuristic_team_split(
        self, match: dict, total_pred: MatchPrediction,
    ) -> dict:
        """Proportional split based on available features when TeamFoulRegressor is unavailable."""
        total_mu = float(total_pred.expected_fouls)
        home_fouls = float(match.get("xfouls_home") or match.get("home_fouls_committed_avg", 12.5))
        away_fouls = float(match.get("xfouls_away") or match.get("away_fouls_committed_avg", 12.5))
        raw_sum = home_fouls + away_fouls
        if raw_sum < 1e-6:
            ratio_h = 0.5
        else:
            ratio_h = home_fouls / raw_sum
        ratio_h = max(0.30, min(0.70, ratio_h))
        home_mu = total_mu * ratio_h
        away_mu = total_mu * (1.0 - ratio_h)
        home_pmf = tilt_pmf_to_mean(total_pred.pmf_total, home_mu)
        away_pmf = tilt_pmf_to_mean(total_pred.pmf_total, away_mu)
        return {
            "total_expected": total_mu,
            "home_expected": round(float(home_pmf.mean), 2),
            "away_expected": round(float(away_pmf.mean), 2),
            "total_pmf": total_pred.pmf_total,
            "home_pmf": home_pmf,
            "away_pmf": away_pmf,
            "raw_home_expected": home_mu,
            "raw_away_expected": away_mu,
            "reconciled": False,
            "method": "heuristic_xfouls_split",
        }

    def predict_team_fouls(
        self,
        match: dict,
        total_prediction: Optional[MatchPrediction] = None,
        reconcile: bool = True,
    ) -> Optional[dict]:
        """Predict fouls per team (for team-level markets)."""
        total_pred = total_prediction or self.predict(match)
        if self.team_regressor is None:
            return self._heuristic_team_split(match, total_pred)
        enriched = self._enrich_match(match)
        result = self.team_regressor.predict(enriched)
        raw_home_mu = float(result["home"]["expected_fouls"])
        raw_away_mu = float(result["away"]["expected_fouls"])
        raw_home_pmf = result["home"]["pmf"]
        raw_away_pmf = result["away"]["pmf"]

        if not reconcile:
            return {
                "home_expected": raw_home_mu,
                "away_expected": raw_away_mu,
                "total_expected": float(raw_home_mu + raw_away_mu),
                "home_pmf": raw_home_pmf,
                "away_pmf": raw_away_pmf,
                "raw_home_expected": raw_home_mu,
                "raw_away_expected": raw_away_mu,
                "reconciled": False,
            }

        # Reconciliación anclada al total del ensemble:
        # mantenemos fijo el total y ajustamos solo local/visitante para cumplir
        # H + A = T, repartiendo el ajuste según su incertidumbre relativa.
        raw_total_mu = float(total_pred.expected_fouls)

        var_h = float(max(raw_home_pmf.variance, 1e-6))
        var_a = float(max(raw_away_pmf.variance, 1e-6))
        gap = raw_total_mu - raw_home_mu - raw_away_mu
        denom = var_h + var_a

        total_mu = raw_total_mu
        home_mu = raw_home_mu + (var_h / denom) * gap
        away_mu = raw_away_mu + (var_a / denom) * gap

        total_pmf = tilt_pmf_to_mean(total_pred.pmf_total, total_mu)
        home_pmf = tilt_pmf_to_mean(raw_home_pmf, home_mu)
        away_pmf = tilt_pmf_to_mean(raw_away_pmf, away_mu)

        return {
            "total_expected": float(total_pmf.mean),
            "home_expected": float(home_pmf.mean),
            "away_expected": float(away_pmf.mean),
            "total_pmf": total_pmf,
            "home_pmf": home_pmf,
            "away_pmf": away_pmf,
            "raw_total_expected": raw_total_mu,
            "raw_home_expected": raw_home_mu,
            "raw_away_expected": raw_away_mu,
            "coherent_constraint_error": float(total_pmf.mean - home_pmf.mean - away_pmf.mean),
            "reconciled": True,
        }

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.referee_profiler.save(directory / "referee")
        self.weighter.save(str(directory / "gating.pt"))
        torch.save(self.layer_regression.model.state_dict(), directory / "regression.pt")
        torch.save(self.layer_anfis.model.state_dict(), directory / "anfis.pt")

        if self.team_regressor is not None:
            torch.save(
                self.team_regressor.home_model.model.state_dict(),
                directory / "team_home.pt",
            )
            torch.save(
                self.team_regressor.away_model.model.state_dict(),
                directory / "team_away.pt",
            )

        import pickle
        with open(directory / "bayes.pkl", "wb") as f:
            pickle.dump({
                "class_priors": self.layer_bayes._class_priors,
                "cond_probs": self.layer_bayes._cond_probs,
                "referee_cond_probs": self.layer_bayes._referee_cond_probs,
                "breakpoints": self.layer_bayes.foul_discretizer.breakpoints,
                "foul_clusterer_committed_centroids": self.layer_bayes.foul_clusterer_committed.centroids,
                "foul_clusterer_suffered_centroids": self.layer_bayes.foul_clusterer_suffered.centroids,
                "rank_clusterer_centroids": self.layer_bayes.rank_clusterer.centroids,
                "xfouls_clusterer_centroids": self.layer_bayes.xfouls_clusterer.centroids,
                "agg_clusterer_centroids": self.layer_bayes.agg_clusterer.centroids,
                "ref_delta_clusterer_centroids": self.layer_bayes.ref_delta_clusterer.centroids,
            }, f)

        norm_data = {
            "reg_means": self.layer_regression._feature_means,
            "reg_stds": self.layer_regression._feature_stds,
            "anfis_mins": self.layer_anfis._feature_mins,
            "anfis_maxs": self.layer_anfis._feature_maxs,
            "bias_correction": self._bias_correction,
            "variance_posthoc_scale": self._variance_posthoc_scale,
        }
        if self.team_regressor is not None:
            norm_data["team_home_means"] = self.team_regressor.home_model._feature_means
            norm_data["team_home_stds"] = self.team_regressor.home_model._feature_stds
            norm_data["team_away_means"] = self.team_regressor.away_model._feature_means
            norm_data["team_away_stds"] = self.team_regressor.away_model._feature_stds

        with open(directory / "normalization.pkl", "wb") as f:
            pickle.dump(norm_data, f)

    def load(self, directory: str | Path):
        directory = Path(directory)

        self.referee_profiler.load(directory / "referee")
        self.weighter.load(str(directory / "gating.pt"))
        self.layer_regression.model.load_state_dict(
            torch.load(directory / "regression.pt", weights_only=True)
        )
        self.layer_anfis.model.load_state_dict(
            torch.load(directory / "anfis.pt", weights_only=True)
        )

        import pickle
        with open(directory / "bayes.pkl", "rb") as f:
            bayes_data = pickle.load(f)
            self.layer_bayes._class_priors = bayes_data["class_priors"]
            self.layer_bayes._cond_probs = bayes_data["cond_probs"]
            self.layer_bayes._referee_cond_probs = bayes_data["referee_cond_probs"]
            self.layer_bayes.foul_discretizer.breakpoints = bayes_data["breakpoints"]
            if "foul_clusterer_committed_centroids" in bayes_data:
                self.layer_bayes.foul_clusterer_committed.centroids = bayes_data["foul_clusterer_committed_centroids"]
            if "foul_clusterer_suffered_centroids" in bayes_data:
                self.layer_bayes.foul_clusterer_suffered.centroids = bayes_data["foul_clusterer_suffered_centroids"]
            if "rank_clusterer_centroids" in bayes_data:
                self.layer_bayes.rank_clusterer.centroids = bayes_data["rank_clusterer_centroids"]
            if "xfouls_clusterer_centroids" in bayes_data:
                self.layer_bayes.xfouls_clusterer.centroids = bayes_data["xfouls_clusterer_centroids"]
            if "agg_clusterer_centroids" in bayes_data:
                self.layer_bayes.agg_clusterer.centroids = bayes_data["agg_clusterer_centroids"]
            if "ref_delta_clusterer_centroids" in bayes_data:
                self.layer_bayes.ref_delta_clusterer.centroids = bayes_data["ref_delta_clusterer_centroids"]

        with open(directory / "normalization.pkl", "rb") as f:
            norm_data = pickle.load(f)
            self.layer_regression._feature_means = norm_data["reg_means"]
            self.layer_regression._feature_stds = norm_data["reg_stds"]
            self.layer_anfis._feature_mins = norm_data["anfis_mins"]
            self.layer_anfis._feature_maxs = norm_data["anfis_maxs"]
            if "bias_correction" in norm_data:
                self._bias_correction = norm_data["bias_correction"]
            if "variance_posthoc_scale" in norm_data:
                self._variance_posthoc_scale = norm_data["variance_posthoc_scale"]

        if (directory / "team_home.pt").exists() and (directory / "team_away.pt").exists():
            self.team_regressor = TeamFoulRegressor()
            self.team_regressor.home_model.model.load_state_dict(
                torch.load(directory / "team_home.pt", weights_only=True)
            )
            self.team_regressor.away_model.model.load_state_dict(
                torch.load(directory / "team_away.pt", weights_only=True)
            )
            with open(directory / "normalization.pkl", "rb") as f:
                tn = pickle.load(f)
            if "team_home_means" in tn:
                self.team_regressor.home_model._feature_means = tn["team_home_means"]
                self.team_regressor.home_model._feature_stds = tn["team_home_stds"]
                self.team_regressor.away_model._feature_means = tn["team_away_means"]
                self.team_regressor.away_model._feature_stds = tn["team_away_stds"]

        self._is_fitted = True

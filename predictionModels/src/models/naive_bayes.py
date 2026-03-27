"""
Layer 1: Naive Bayes Foul Predictor.

Replicates and extends the model from Pérez-Blanco & Salmerón (2025).
Key features:
  - Custom percentile-based discretization into 4 foul intervals
  - Relative distance to cluster centroids for team ranking (5 clusters, 4 teams each)
  - Extension: referee node as additional parent in the BN
  - Output: interval probabilities + converted full PMF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.distributions import FoulPMF, pmf_from_intervals


@dataclass
class ClusterAssignment:
    """
    Relative membership of a team to its two nearest clusters.
    Implements equations 9-10 from the paper.
    """

    cluster_idx: int
    w_lower: float  # weight towards lower cluster centroid
    w_upper: float  # weight towards upper cluster centroid

    @staticmethod
    def compute(value: float, centroids: np.ndarray) -> "ClusterAssignment":
        """
        Assign a value to its two nearest clusters with relative weights.
        
        centroids: sorted array of cluster centroids [c1, c2, ..., c5]
        """
        n = len(centroids)
        if value <= centroids[0]:
            return ClusterAssignment(0, 1.0, 0.0)
        if value >= centroids[-1]:
            return ClusterAssignment(n - 2, 0.0, 1.0)

        for i in range(n - 1):
            if centroids[i] <= value <= centroids[i + 1]:
                span = centroids[i + 1] - centroids[i]
                if span < 1e-10:
                    return ClusterAssignment(i, 0.5, 0.5)
                # Eq 9: w_i = |x - c_i| / (c_{i+1} - c_i)
                # Note: w_i is distance to c_i, so HIGHER w_i means CLOSER to c_{i+1}
                w_i = (value - centroids[i]) / span
                w_i1 = (centroids[i + 1] - value) / span
                return ClusterAssignment(i, w_i1, w_i)

        return ClusterAssignment(0, 0.5, 0.5)


class FoulDiscretizer:
    """
    Discretizes the total fouls into intervals based on historical percentiles.
    Default: P25, P50, P75 -> 4 intervals.
    """

    def __init__(self, percentiles: list[int] = None):
        self.percentiles = percentiles or [25, 50, 75]
        self.breakpoints: list[int] = []

    def fit(self, historical_fouls: np.ndarray):
        """Compute breakpoints from historical foul totals."""
        cuts = np.percentile(historical_fouls, self.percentiles)
        # Cota del intervalo abierto final al P97 real (evita inflar la media
        # con masa en valores extremos poco probables como 50-61 faltas).
        cap = int(round(np.percentile(historical_fouls, 97)))
        cap = max(cap, int(round(cuts[-1])) + 3)
        self.breakpoints = [0] + [int(round(c)) for c in cuts] + [cap]

    def discretize(self, fouls: int) -> int:
        """Return the interval index (0-based) for a foul count."""
        for i in range(len(self.breakpoints) - 1):
            if fouls < self.breakpoints[i + 1]:
                return i
        return len(self.breakpoints) - 2

    @property
    def n_classes(self) -> int:
        return len(self.breakpoints) - 1

    @property
    def interval_labels(self) -> list[str]:
        labels = []
        for i in range(len(self.breakpoints) - 1):
            low = self.breakpoints[i]
            high = self.breakpoints[i + 1]
            if i < len(self.breakpoints) - 2:
                labels.append(f"{low}-{high - 1}")
            else:
                labels.append(f"{low}+")
        return labels


class TeamClusterer:
    """
    Clusters teams into 5 groups based on average foul rates.
    4 teams per cluster (for a 20-team league).
    Uses equi-magnitude intervals rather than k-means.
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.centroids: np.ndarray = np.array([])

    def fit(self, team_averages: dict[str, float]):
        """
        Compute cluster centroids from team average foul rates.
        
        team_averages: {team_name: avg_fouls_per_match}
        """
        values = sorted(team_averages.values())
        n = len(values)
        teams_per_cluster = n // self.n_clusters

        centroids = []
        for i in range(self.n_clusters):
            start = i * teams_per_cluster
            end = start + teams_per_cluster if i < self.n_clusters - 1 else n
            cluster_vals = values[start:end]
            centroids.append(np.mean(cluster_vals))

        self.centroids = np.array(centroids)

    def assign(self, value: float) -> ClusterAssignment:
        return ClusterAssignment.compute(value, self.centroids)


class NaiveBayesFoulPredictor:
    """
    Naive Bayes classifier for foul interval prediction.

    Predictor variables:
        - Historical fouls committed (home/away) x 2
        - Historical fouls suffered (home/away) x 2
        - Historical rank (home/away) x 2
        - Current rank (home/away) x 2
        - xFouls expected (home/away) x 2
        - Aggressiveness norm total x 1 (treated as home=away=total for pair-feature)
        - Referee-pair delta (home/away) x 2
        - Referee mode (permissive/strict) x 1

    Each continuous predictor is discretized via relative cluster distance.
    """

    def __init__(
        self,
        n_foul_clusters: int = 5,
        n_rank_clusters: int = 5,
        foul_percentiles: Optional[list[int]] = None,
    ):
        self.foul_discretizer = FoulDiscretizer(percentiles=foul_percentiles)
        self.foul_clusterer_committed = TeamClusterer(n_foul_clusters)
        self.foul_clusterer_suffered = TeamClusterer(n_foul_clusters)
        self.rank_clusterer = TeamClusterer(n_rank_clusters)
        self.xfouls_clusterer = TeamClusterer(n_foul_clusters)
        self.agg_clusterer = TeamClusterer(n_foul_clusters)
        self.ref_delta_clusterer = TeamClusterer(n_foul_clusters)

        self._n_foul_classes: int = 4
        self._class_priors: np.ndarray = np.array([])
        # cond_probs[feature_name] = array of shape (n_clusters, n_foul_classes)
        self._cond_probs: dict[str, np.ndarray] = {}
        self._referee_cond_probs: np.ndarray = np.array([])  # shape (2, n_foul_classes)

    def fit(
        self,
        matches: list[dict],
        team_avg_committed: dict[str, float],
        team_avg_suffered: dict[str, float],
        team_avg_rank: dict[str, float],
    ):
        """
        Train the NB classifier on historical match data.
        """
        fouls_array = np.array([m["fouls_total"] for m in matches])
        self.foul_discretizer.fit(fouls_array)
        self._n_foul_classes = self.foul_discretizer.n_classes

        self.foul_clusterer_committed.fit(team_avg_committed)
        self.foul_clusterer_suffered.fit(team_avg_suffered)
        self.rank_clusterer.fit(team_avg_rank)

        # Build per-team averages for xFouls, aggressiveness, and ref_delta
        xfouls_avg: dict[str, list] = {}
        agg_avg: dict[str, list] = {}
        ref_delta_avg: dict[str, list] = {}
        for m in matches:
            home, away = m.get("home_team", ""), m.get("away_team", "")
            xfouls_avg.setdefault(home, []).append(float(m.get("xfouls_home", 12.5)))
            xfouls_avg.setdefault(away, []).append(float(m.get("xfouls_away", 12.5)))
            agg_val = float(m.get("aggressiveness_norm_total", 0.5))
            agg_avg.setdefault(home, []).append(agg_val)
            agg_avg.setdefault(away, []).append(agg_val)
            ref_delta_avg.setdefault(home, []).append(float(m.get("ref_home_delta", 0.0)))
            ref_delta_avg.setdefault(away, []).append(float(m.get("ref_away_delta", 0.0)))

        self.xfouls_clusterer.fit({t: np.mean(v) for t, v in xfouls_avg.items()})
        self.agg_clusterer.fit({t: np.mean(v) for t, v in agg_avg.items()})
        self.ref_delta_clusterer.fit({t: np.mean(v) for t, v in ref_delta_avg.items()})

        class_counts = np.zeros(self._n_foul_classes) + 1e-6  # Laplace smoothing
        for m in matches:
            cls = self.foul_discretizer.discretize(m["fouls_total"])
            class_counts[cls] += 1
        self._class_priors = class_counts / class_counts.sum()

        feature_specs = self._get_feature_specs()
        for feat_name, clusterer, match_key_home, match_key_away in feature_specs:
            n_clusters = clusterer.n_clusters
            counts = np.ones((n_clusters, self._n_foul_classes)) * 1e-6
            for m in matches:
                cls = self.foul_discretizer.discretize(m["fouls_total"])
                for val in [m[match_key_home], m[match_key_away]]:
                    assignment = clusterer.assign(val)
                    idx = assignment.cluster_idx
                    counts[idx, cls] += assignment.w_lower
                    if idx + 1 < n_clusters:
                        counts[idx + 1, cls] += assignment.w_upper

            for c in range(n_clusters):
                row_sum = counts[c].sum()
                if row_sum > 0:
                    counts[c] /= row_sum
            self._cond_probs[feat_name] = counts

        # Referee conditional: P(referee_mode | foul_class)
        ref_counts = np.ones((2, self._n_foul_classes)) * 1e-6
        for m in matches:
            cls = self.foul_discretizer.discretize(m["fouls_total"])
            mode = int(m.get("referee_mode", 0))
            ref_counts[mode, cls] += 1
        for mode in range(2):
            ref_counts[mode] /= ref_counts[mode].sum()
        self._referee_cond_probs = ref_counts

    def _get_feature_specs(self):
        """Returns (feat_name, clusterer, home_key, away_key) tuples."""
        return [
            ("fouls_committed", self.foul_clusterer_committed,
             "home_fouls_committed_avg", "away_fouls_committed_avg"),
            ("fouls_suffered", self.foul_clusterer_suffered,
             "home_fouls_suffered_avg", "away_fouls_suffered_avg"),
            ("rank_hist", self.rank_clusterer,
             "home_rank_hist", "away_rank_hist"),
            ("rank_curr", self.rank_clusterer,
             "home_rank_curr", "away_rank_curr"),
            ("xfouls", self.xfouls_clusterer,
             "xfouls_home", "xfouls_away"),
            ("aggressiveness", self.agg_clusterer,
             "aggressiveness_norm_total", "aggressiveness_norm_total"),
            ("ref_delta", self.ref_delta_clusterer,
             "ref_home_delta", "ref_away_delta"),
        ]

    def predict_interval_probs(self, match: dict) -> np.ndarray:
        """
        Predict foul interval probabilities for a match.
        Returns array of shape (n_foul_classes,).
        
        Uses Bayes' rule (Eq. 3 from paper):
        P(C=cj | x) ∝ P(C=cj) * prod_i P(xi | C=cj)
        
        With weighted cluster assignments (Eq. 11):
        P(x|M) = w_i * P(C_i ∧ M) + w_{i+1} * P(C_{i+1} ∧ M)
        """
        log_posteriors = np.log(self._class_priors + 1e-15)

        feature_specs = self._get_feature_specs()
        for feat_name, clusterer, home_key, away_key in feature_specs:
            cond = self._cond_probs[feat_name]

            for val in [match[home_key], match[away_key]]:
                assignment = clusterer.assign(val)
                idx = assignment.cluster_idx

                likelihood = np.zeros(self._n_foul_classes)
                likelihood += assignment.w_lower * cond[idx]
                if idx + 1 < cond.shape[0]:
                    likelihood += assignment.w_upper * cond[idx + 1]

                log_posteriors += np.log(likelihood + 1e-15)

        # Referee contribution
        referee_mode_prob = match.get("referee_strict_prob", 0.5)
        ref_likelihood = (
            (1 - referee_mode_prob) * self._referee_cond_probs[0]
            + referee_mode_prob * self._referee_cond_probs[1]
        )
        log_posteriors += np.log(ref_likelihood + 1e-15)

        posteriors = np.exp(log_posteriors - log_posteriors.max())
        posteriors /= posteriors.sum()

        return posteriors

    def predict_pmf(self, match: dict) -> FoulPMF:
        """Convert interval probabilities to a full PMF."""
        interval_probs = self.predict_interval_probs(match)
        return pmf_from_intervals(interval_probs, self.foul_discretizer.breakpoints)

    def predict(self, match: dict) -> dict:
        """Full prediction output."""
        interval_probs = self.predict_interval_probs(match)
        pmf = pmf_from_intervals(interval_probs, self.foul_discretizer.breakpoints)
        return {
            "interval_probs": interval_probs,
            "interval_labels": self.foul_discretizer.interval_labels,
            "pmf": pmf,
            "expected_fouls": pmf.mean,
            "breakpoints": self.foul_discretizer.breakpoints,
        }

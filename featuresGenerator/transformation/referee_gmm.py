"""
Perfiles GMM bimodales de arbitros para el Knowledge Engine.

Ajusta un Gaussian Mixture Model de 2 componentes (permisivo / estricto)
sobre el historico de faltas totales de cada arbitro y produce los parametros
GMM que se incluyen en el JSON contrato bajo arbitro.estadisticas.

Los partidos recientes pesan mas que los antiguos (sample_weight con decaimiento
exponencial por dias hasta la fecha mas reciente del dataset), alineado con
config.decay_lambda del resto del modelo (~0.003 / dia).

Separa la responsabilidad de GENERACION DE CONOCIMIENTO (fitting) de la de
PREDICCION (inferencia de modo en predictionModels).

Output por arbitro:
  mu_permisivo, sigma_permisivo  -- componente de baja media
  mu_estricto, sigma_estricto    -- componente de alta media
  peso_estricto                  -- peso de la componente estricta
  partidos_arbitrados            -- n de partidos usados para el fit
  is_shrunk                      -- True si se aplico shrinkage hacia el global
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from core.helpers import parse_date, safe

# Decaimiento exponencial por antigüedad del partido (días).
# λ=0.003 → un partido de hace ~8 meses pesa ~mitad que uno reciente.
REFEREE_GMM_DECAY_LAMBDA_PER_DAY = 0.003


# ---------------------------------------------------------------------------
# Defaults de poblacion cuando no hay datos suficientes
# ---------------------------------------------------------------------------

GLOBAL_MU_PERMISIVO = 22.0
GLOBAL_MU_ESTRICTO = 30.0
GLOBAL_SIGMA = 4.0
GLOBAL_PESO_ESTRICTO = 0.45

MIN_MATCHES_STANDALONE = 8
SHRINKAGE_STRENGTH = 0.35


@dataclass
class RefereeGMMParams:
    """Parametros del perfil bimodal de un arbitro, listos para el JSON contrato."""
    nombre: str
    partidos_arbitrados: int = 0
    mu_permisivo: float = GLOBAL_MU_PERMISIVO
    sigma_permisivo: float = GLOBAL_SIGMA
    mu_estricto: float = GLOBAL_MU_ESTRICTO
    sigma_estricto: float = GLOBAL_SIGMA
    peso_estricto: float = GLOBAL_PESO_ESTRICTO
    is_shrunk: bool = False

    def to_dict(self) -> dict:
        return {
            "partidos_arbitrados": self.partidos_arbitrados,
            "mu_permisivo": round(self.mu_permisivo, 2),
            "sigma_permisivo": round(self.sigma_permisivo, 2),
            "mu_estricto": round(self.mu_estricto, 2),
            "sigma_estricto": round(self.sigma_estricto, 2),
            "peso_estricto": round(self.peso_estricto, 3),
            "is_shrunk": self.is_shrunk,
        }


# ---------------------------------------------------------------------------
# Extraccion de series historicas desde partidos
# ---------------------------------------------------------------------------

def _extraer_series_arbitros(
    partidos: list[dict],
) -> dict[str, list[tuple[date, float]]]:
    """
    Agrupa (fecha, faltas totales) por árbitro, ordenado por fecha.
    La fecha se usa para ponderar partidos recientes más en el GMM.
    """
    series: dict[str, list[tuple[date, float]]] = {}
    for p in partidos:
        ref = (p.get("referee") or "").strip()
        if not ref:
            continue
        f_h = safe(p.get("home", {}).get("fouls"))
        f_a = safe(p.get("away", {}).get("fouls"))
        total = f_h + f_a
        if total == 0:
            continue
        try:
            d = parse_date(str(p.get("date", ""))[:10])
        except Exception:
            continue
        series.setdefault(ref, []).append((d, float(total)))

    for ref in series:
        series[ref].sort(key=lambda x: x[0])
    return series


def _sample_weights_for_dates(
    dates: list[date],
    ref_date: date,
    decay_lambda: float = REFEREE_GMM_DECAY_LAMBDA_PER_DAY,
) -> np.ndarray:
    """Pesos exp(-λ * días_desde_ref); ref_date = ancla más reciente."""
    if not dates:
        return np.array([])
    w = np.array(
        [np.exp(-decay_lambda * max(0, (ref_date - d).days)) for d in dates],
        dtype=np.float64,
    )
    s = w.sum()
    if s <= 1e-12:
        w = np.ones_like(w)
        s = float(len(w))
    # Escala para que la suma sea n (misma escala que muestras sin peso)
    return w * (len(w) / s)


# ---------------------------------------------------------------------------
# Fitting de un GMM bimodal para un arbitro
# ---------------------------------------------------------------------------

def _expand_weighted_samples(
    X: np.ndarray,
    sample_weight: np.ndarray,
    *,
    repeat_scale: float = 4.0,
    max_rep: int = 25,
) -> np.ndarray:
    """
    GaussianMixture.fit no acepta sample_weight en todas las versiones de sklearn.
    Duplicamos filas de forma determinista según peso (partidos recientes → más copias).
    """
    w = np.asarray(sample_weight, dtype=np.float64)
    w = w / max(w.sum(), 1e-12) * len(w) * repeat_scale
    rep = np.maximum(1, np.round(w)).astype(int)
    rep = np.minimum(rep, max_rep)
    flat = X.reshape(-1)
    return np.repeat(flat, rep).reshape(-1, 1)


def _fit_gmm_single(
    fouls: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[float, float, float, float, float]:
    """
    Ajusta GMM de 2 componentes sobre la serie de faltas.
    Devuelve (mu_perm, sig_perm, mu_strict, sig_strict, peso_strict).
    """
    X = np.asarray(fouls, dtype=np.float64).reshape(-1, 1)
    if sample_weight is not None and len(sample_weight) == len(X):
        X_fit = _expand_weighted_samples(X, sample_weight)
    else:
        X_fit = X
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="spherical",
        n_init=5,
        random_state=42,
    )
    gmm.fit(X_fit)

    mu = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    idx_perm = int(np.argmin(mu))
    idx_strict = 1 - idx_perm

    return (
        float(mu[idx_perm]),
        float(sigma[idx_perm]),
        float(mu[idx_strict]),
        float(sigma[idx_strict]),
        float(weights[idx_strict]),
    )


# ---------------------------------------------------------------------------
# Fitting con shrinkage hacia el perfil global
# ---------------------------------------------------------------------------

def _fit_with_shrinkage(
    dated_fouls: list[tuple[date, float]],
    global_params: dict,
    nombre: str,
) -> RefereeGMMParams:
    """
    Para arbitros con pocos partidos aplica Bayesian shrinkage hacia el global.
    Cuantos menos partidos, mas atraccion hacia el global.
    """
    if not dated_fouls:
        return RefereeGMMParams(
            nombre=nombre,
            partidos_arbitrados=0,
            mu_permisivo=global_params["mu_permisivo"],
            sigma_permisivo=global_params["sigma_permisivo"],
            mu_estricto=global_params["mu_estricto"],
            sigma_estricto=global_params["sigma_estricto"],
            peso_estricto=global_params["peso_estricto"],
            is_shrunk=True,
        )

    dates = [d for d, _ in dated_fouls]
    fouls_arr = np.array([f for _, f in dated_fouls], dtype=np.float64)
    ref_d = max(dates)
    sw = _sample_weights_for_dates(dates, ref_d)

    try:
        mu_p, sig_p, mu_s, sig_s, w_s = _fit_gmm_single(fouls_arr, sample_weight=sw)
    except Exception:
        mu_p, sig_p, mu_s, sig_s, w_s = (
            global_params["mu_permisivo"],
            global_params["sigma_permisivo"],
            global_params["mu_estricto"],
            global_params["sigma_estricto"],
            global_params["peso_estricto"],
        )

    alpha = SHRINKAGE_STRENGTH * (1 - len(dated_fouls) / MIN_MATCHES_STANDALONE)
    alpha = max(0.0, min(1.0, alpha))

    return RefereeGMMParams(
        nombre=nombre,
        partidos_arbitrados=len(dated_fouls),
        mu_permisivo=(1 - alpha) * mu_p + alpha * global_params["mu_permisivo"],
        sigma_permisivo=(1 - alpha) * sig_p + alpha * global_params["sigma_permisivo"],
        mu_estricto=(1 - alpha) * mu_s + alpha * global_params["mu_estricto"],
        sigma_estricto=(1 - alpha) * sig_s + alpha * global_params["sigma_estricto"],
        peso_estricto=(1 - alpha) * w_s + alpha * global_params["peso_estricto"],
        is_shrunk=True,
    )


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def calcular_perfiles_gmm(
    partidos: list[dict],
) -> dict[str, RefereeGMMParams]:
    """
    Calcula perfiles GMM bimodales para todos los arbitros con historico.

    Args:
        partidos: lista de partidos con campos home.fouls, away.fouls, referee.

    Returns:
        dict[nombre_arbitro -> RefereeGMMParams]
    """
    series = _extraer_series_arbitros(partidos)
    if not series:
        return {}

    # Perfil global: todos los partidos con peso temporal (ancla = fecha más reciente)
    all_dated: list[tuple[date, float]] = []
    for vals in series.values():
        all_dated.extend(vals)
    all_dated.sort(key=lambda x: x[0])
    global_ref = all_dated[-1][0]
    g_dates = [d for d, _ in all_dated]
    g_fouls = np.array([f for _, f in all_dated], dtype=np.float64)
    g_sw = _sample_weights_for_dates(g_dates, global_ref)

    try:
        g_mu_p, g_sig_p, g_mu_s, g_sig_s, g_w_s = _fit_gmm_single(
            g_fouls, sample_weight=g_sw
        )
    except Exception:
        g_mu_p, g_sig_p = GLOBAL_MU_PERMISIVO, GLOBAL_SIGMA
        g_mu_s, g_sig_s = GLOBAL_MU_ESTRICTO, GLOBAL_SIGMA
        g_w_s = GLOBAL_PESO_ESTRICTO

    global_params = {
        "mu_permisivo": g_mu_p,
        "sigma_permisivo": g_sig_p,
        "mu_estricto": g_mu_s,
        "sigma_estricto": g_sig_s,
        "peso_estricto": g_w_s,
    }

    perfiles: dict[str, RefereeGMMParams] = {}
    for nombre, dated in series.items():
        dates = [d for d, _ in dated]
        ref_d = max(dates)
        fouls_arr = np.array([f for _, f in dated], dtype=np.float64)
        sw = _sample_weights_for_dates(dates, ref_d)

        if len(dated) >= MIN_MATCHES_STANDALONE:
            try:
                mu_p, sig_p, mu_s, sig_s, w_s = _fit_gmm_single(
                    fouls_arr, sample_weight=sw
                )
                perfiles[nombre] = RefereeGMMParams(
                    nombre=nombre,
                    partidos_arbitrados=len(dated),
                    mu_permisivo=mu_p,
                    sigma_permisivo=sig_p,
                    mu_estricto=mu_s,
                    sigma_estricto=sig_s,
                    peso_estricto=w_s,
                    is_shrunk=False,
                )
            except Exception:
                perfiles[nombre] = _fit_with_shrinkage(dated, global_params, nombre)
        else:
            perfiles[nombre] = _fit_with_shrinkage(dated, global_params, nombre)

    return perfiles


def buscar_perfil_gmm(
    nombre_input: Optional[str],
    perfiles: dict[str, RefereeGMMParams],
) -> Optional[RefereeGMMParams]:
    """Busqueda fuzzy del perfil GMM de un arbitro por nombre."""
    if not nombre_input:
        return None
    q = nombre_input.lower().strip()

    for nombre, perfil in perfiles.items():
        if nombre.lower() == q:
            return perfil

    candidatos = [n for n in perfiles if q in n.lower()]
    if len(candidatos) == 1:
        return perfiles[candidatos[0]]
    if candidatos:
        return perfiles[min(candidatos, key=len)]

    candidatos = [n for n in perfiles if n.lower() in q]
    if candidatos:
        return perfiles[max(candidatos, key=len)]

    return None


def get_perfil_gmm_o_default(
    nombre_input: Optional[str],
    perfiles: dict,
) -> dict:
    """
    Devuelve el dict de estadisticas GMM del arbitro, o defaults globales
    si no se encuentra en el historico.
    """
    perfil = buscar_perfil_gmm(nombre_input, perfiles)
    if perfil:
        return perfil.to_dict() if hasattr(perfil, "to_dict") else dict(perfil)
    return {
        "partidos_arbitrados": 0,
        "mu_permisivo": GLOBAL_MU_PERMISIVO,
        "sigma_permisivo": GLOBAL_SIGMA,
        "mu_estricto": GLOBAL_MU_ESTRICTO,
        "sigma_estricto": GLOBAL_SIGMA,
        "peso_estricto": GLOBAL_PESO_ESTRICTO,
        "is_shrunk": True,
    }

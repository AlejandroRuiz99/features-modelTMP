"""
predict.py - Motor de prediccion de faltas desde flat features dict.

Flujo:
  1. Lee el flat dict de features (JSON producido por featuresGenerator)
  2. Carga el ensemble entrenado desde checkpoints/
  3. Predice: PMF + Over/Under por linea + desglose por equipo

Uso:
  python -m scripts.predict --input features.json
  python -m scripts.predict --input features.json --checkpoint checkpoints/ensemble/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.ensemble import FoulPredictionEnsemble


# ---------------------------------------------------------------------------
# Output de prediccion
# ---------------------------------------------------------------------------

def _print_prediction(ensemble: FoulPredictionEnsemble, feat: dict) -> None:
    home = feat["home_team"]
    away = feat["away_team"]
    referee = feat["referee"]
    matchday = feat["matchday"]
    season = feat["season"]

    pred = ensemble.predict(feat)
    team_pred = ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True)

    lines = [21.5, 23.5, 24.5, 25.5, 27.5, 29.5]

    print("=" * 65)
    print(f"  {home} vs {away} | J{matchday} {season}")
    print(f"  Arbitro: {referee}")
    print("=" * 65)

    print(f"\n--- Senales clave ---")
    print(f"  Arbitro GMM:        permisivo={feat['referee_mu_permisivo']:.1f} | estricto={feat['referee_mu_estricto']:.1f} | p(estricto)={feat['referee_peso_estricto']:.0%} | n={feat['referee_n_partidos']}")
    print(f"  xFaltas esperadas:  {feat['xfouls_home']:.1f} (L) + {feat['xfouls_away']:.1f} (V) = {feat['xfouls_home'] + feat['xfouls_away']:.1f}")
    print(f"  Mercado faltas OU:  total={feat['foul_market_implied_mean']:.1f} (prob_mas={feat['foul_market_prob_over']:.1%}) | local={feat['foul_market_local_implied_mean']:.1f} | visitante={feat['foul_market_vis_implied_mean']:.1f}")
    print(f"  Forma reciente:     {feat['forma_fouls_home']:.1f} faltas (L) / {feat['forma_fouls_away']:.1f} faltas (V)")
    print(f"  Agresividad (norm): {feat['aggressiveness_volume_home']:.2f} (L) / {feat['aggressiveness_volume_away']:.2f} (V)")
    print(f"  Urgencia:           {feat['urgency_home']:.2f} (L) / {feat['urgency_away']:.2f} (V)")
    print(f"  Momentum:           {feat['momentum_home']:.3f} (L) / {feat['momentum_away']:.3f} (V)")
    print(f"  Dias descanso:      {feat['days_rest_home']:.0f} (L) / {feat['days_rest_away']:.0f} (V)")
    print(f"  Delta arbitro:      {feat['ref_home_delta']:+.2f} (L) / {feat['ref_away_delta']:+.2f} (V)")
    print(f"  Derby: {feat['is_derby']} | Season phase: {feat['season_phase']:.2f} | Pace index: {feat['pace_index_curr']:.1f}")
    print(f"  H2H: {feat['h2h_faltas_media']:.1f} faltas media ({feat['h2h_partidos']} partidos previos)")
    print(f"  Intensidad: {feat['intensidad_esperada']} | Riesgo disciplinario: {feat['riesgo_disciplinario']}")

    print(f"\n--- Prediccion ensemble ---")
    print(f"  Faltas esperadas (raw):        {pred.expected_fouls:.2f}")
    print(f"  P(arbitro estricto):            {pred.referee_strict_prob*100:.1f}%")
    print(
        f"  Pesos gating [NB, Reg, ANFIS]: "
        f"[{pred.weights[0]:.3f}, {pred.weights[1]:.3f}, {pred.weights[2]:.3f}]"
    )

    ou_table = pred.over_under
    if team_pred and team_pred.get("reconciled") and team_pred.get("total_pmf") is not None:
        ou_table = team_pred["total_pmf"].over_under_table(lines)

    print(f"\n--- Over/Under ---")
    for line in lines:
        if line in ou_table:
            p_o, p_u = ou_table[line]
            tag = ""
            if abs(line - feat["foul_market_implied_mean"]) < 1.0:
                tag = "  <-- linea mercado"
            print(f"  OU {line:.1f}: Over={p_o*100:5.1f}% | Under={p_u*100:5.1f}%{tag}")

    if team_pred:
        print(f"\n--- Desglose por equipo ---")
        print(f"  Total reconciliado:  {team_pred['total_expected']:.2f}")
        print(f"  {home:20s}: {team_pred['home_expected']:.2f}")
        print(f"  {away:20s}: {team_pred['away_expected']:.2f}")
        if team_pred.get("reconciled"):
            print(
                f"  Reconciliacion: raw({team_pred['raw_total_expected']:.2f}) "
                f"-> coherent({team_pred['total_expected']:.2f} = "
                f"{team_pred['home_expected']:.2f} + {team_pred['away_expected']:.2f})"
            )
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    feat = json.loads(Path(args.input).read_text(encoding="utf-8"))

    print("Cargando features...")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {checkpoint}\n"
            f"Entrena primero con: python -m scripts.train"
        )

    print(f"Cargando ensemble desde {checkpoint}...")
    config_path = checkpoint / "config.json"
    cfg = {}
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(checkpoint)

    _print_prediction(ensemble, feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prediccion de faltas desde flat features dict"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Ruta del JSON de features (producido por featuresGenerator)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/ensemble/",
        help="Directorio del ensemble entrenado (default: checkpoints/ensemble/)."
    )
    args = parser.parse_args()
    main(args)

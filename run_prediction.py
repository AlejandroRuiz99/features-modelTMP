"""
run_prediction.py - Predice faltas para un partido de La Liga.

Uso (partido unico):
  python run_prediction.py --local "Real Madrid" --visitante "Ath Madrid" --jornada 29 --fecha 2026-03-22
  python run_prediction.py --local "Levante" --visitante "Oviedo" --jornada 29 --fecha 2026-03-21 --arbitro "Munuera Montero"

Modo batch (JSON o CSV):
  python run_prediction.py --batch-file partidos.json --output-json resultados.json

Flags adicionales:
  --features-profile  prediction | training | minimal
  --refresh-data      Fuerza recarga desde Supabase
  --output-json PATH  Guarda la prediccion en JSON
  --log-level         CRITICAL | ERROR | WARNING | INFO | DEBUG
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path: cada subdirectorio usa imports sin prefijo de paquete
# (ej: "from core.xxx" en features_generator, "from src.models.xxx" en prediction_models).
# Ambos directorios deben estar en sys.path ANTES de cualquier import del proyecto.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FEATURES_DIR = ROOT / "features_generator"
PRED_DIR = ROOT / "prediction_models"

sys.path.insert(0, str(PRED_DIR))  # habilita: from src.models.xxx, from src.utils.xxx
sys.path.insert(
    0, str(FEATURES_DIR)
)  # habilita: from core.xxx, from generate import, etc.

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
from generate import generate_features  # noqa: E402  (features_generator)
from src.models.ensemble import FoulPredictionEnsemble  # noqa: E402  (prediction_models)

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = PRED_DIR / "checkpoints" / "ensemble"


# ---------------------------------------------------------------------------
# _print_prediction: salida legible por pantalla
# ---------------------------------------------------------------------------


def _print_prediction(ensemble: FoulPredictionEnsemble, feat: dict[str, Any]) -> None:
    """Imprime la prediccion de faltas en formato legible."""
    home = feat["home_team"]
    away = feat["away_team"]
    referee = feat.get("referee", "Desconocido")
    matchday = feat.get("matchday", "?")
    season = feat.get("season", "")

    pred = ensemble.predict(feat)
    team_pred = (
        ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True) or {}
    )

    lines = [21.5, 23.5, 24.5, 25.5, 27.5, 29.5]

    print("=" * 65)
    print(f"  {home} vs {away} | J{matchday} {season}")
    print(f"  Arbitro: {referee}")
    print("=" * 65)

    print("\n--- Senales clave ---")
    print(
        f"  Arbitro GMM:        permisivo={feat['referee_mu_permisivo']:.1f} | "
        f"estricto={feat['referee_mu_estricto']:.1f} | "
        f"p(estricto)={feat['referee_peso_estricto']:.0%} | "
        f"n={feat['referee_n_partidos']}"
    )
    print(
        f"  xFaltas esperadas:  {feat['xfouls_home']:.1f} (L) + "
        f"{feat['xfouls_away']:.1f} (V) = "
        f"{feat['xfouls_home'] + feat['xfouls_away']:.1f}"
    )

    market_total = feat.get("foul_market_implied_mean", 0.0)
    market_prob_over = feat.get("foul_market_prob_over", 0.0)
    market_local = feat.get(
        "foul_market_local_implied_mean", market_total / 2 if market_total else 0.0
    )
    market_visitor = feat.get(
        "foul_market_vis_implied_mean", market_total / 2 if market_total else 0.0
    )
    print(
        f"  Mercado faltas OU:  total={market_total:.1f} "
        f"(prob_mas={market_prob_over:.1%}) | "
        f"local={market_local:.1f} | visitante={market_visitor:.1f}"
    )

    print(
        f"  Forma reciente:     {feat['forma_fouls_home']:.1f} faltas (L) / "
        f"{feat['forma_fouls_away']:.1f} faltas (V)"
    )
    print(
        f"  Agresividad (norm): {feat['aggressiveness_volume_home']:.2f} (L) / "
        f"{feat['aggressiveness_volume_away']:.2f} (V)"
    )
    print(
        f"  Urgencia:           {feat['urgency_home']:.2f} (L) / {feat['urgency_away']:.2f} (V)"
    )
    print(
        f"  Momentum:           {feat['momentum_home']:.3f} (L) / {feat['momentum_away']:.3f} (V)"
    )
    print(
        f"  Dias descanso:      {feat['days_rest_home']:.0f} (L) / {feat['days_rest_away']:.0f} (V)"
    )
    print(
        f"  Delta arbitro:      {feat['ref_home_delta']:+.2f} (L) / {feat['ref_away_delta']:+.2f} (V)"
    )
    print(
        f"  Derby: {feat['is_derby']} | "
        f"Season phase: {feat['season_phase']:.2f} | "
        f"Pace index: {feat['pace_index_curr']:.1f}"
    )
    print(
        f"  H2H: {feat['h2h_faltas_media']:.1f} faltas media ({feat['h2h_partidos']} partidos previos)"
    )
    print(
        f"  Intensidad: {feat['intensidad_esperada']} | "
        f"Riesgo disciplinario: {feat['riesgo_disciplinario']}"
    )

    print("\n--- Prediccion ensemble ---")
    print(f"  Faltas esperadas (raw):        {pred.expected_fouls:.2f}")
    print(f"  P(arbitro estricto):            {pred.referee_strict_prob * 100:.1f}%")
    print(
        f"  Pesos gating [NB, Reg, ANFIS]: "
        f"[{pred.weights[0]:.3f}, {pred.weights[1]:.3f}, {pred.weights[2]:.3f}]"
    )

    ou_table = pred.over_under
    if team_pred.get("reconciled") and team_pred.get("total_pmf") is not None:
        ou_table = team_pred["total_pmf"].over_under_table(lines)

    print("\n--- Over/Under ---")
    for line in lines:
        if line in ou_table:
            p_o, p_u = ou_table[line]
            tag = "  <-- linea mercado" if abs(line - market_total) < 1.0 else ""
            print(
                f"  OU {line:.1f}: Over={p_o * 100:5.1f}% | Under={p_u * 100:5.1f}%{tag}"
            )

    if team_pred:
        print("\n--- Desglose por equipo ---")
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
# Helpers
# ---------------------------------------------------------------------------


def _validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _fuzzy_match_team(name: str, canonical: list[str]) -> str | None:
    from core.utils import fuzzy_name_search, TEAM_ALIASES

    return fuzzy_name_search(name, canonical, TEAM_ALIASES) or None


def _load_state(refresh: bool = False) -> dict[str, Any]:
    from core.state_cache import get_state

    return get_state(refresh=refresh)


def _load_ensemble() -> FoulPredictionEnsemble:
    config_path = CHECKPOINT_DIR / "config.json"
    regression_path = CHECKPOINT_DIR / "regression.pt"
    if not config_path.exists() or not regression_path.exists():
        logger.error(
            f"Checkpoint no encontrado en {CHECKPOINT_DIR}. "
            "Entrena primero con: python -m scripts.train"
        )
        sys.exit(1)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ensemble = FoulPredictionEnsemble(config=cfg)
    ensemble.load(CHECKPOINT_DIR)
    return ensemble


def _validate_single(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Valida inputs de modo single-match; muta args con nombres canonicos si hay match."""
    errors: list[str] = []

    if not _validate_date(args.fecha):
        errors.append(f"Fecha invalida '{args.fecha}'. Usa formato YYYY-MM-DD.")

    if not (1 <= args.jornada <= 38):
        errors.append(f"Jornada debe estar entre 1 y 38, recibido: {args.jornada}.")

    canonical = list(state.get("scores", {}).keys())
    if canonical:
        home_match = _fuzzy_match_team(args.local, canonical)
        away_match = _fuzzy_match_team(args.visitante, canonical)
        if not home_match:
            errors.append(
                f"Equipo local '{args.local}' no reconocido. "
                f"Prueba con: {canonical[:5]}..."
            )
        if not away_match:
            errors.append(
                f"Equipo visitante '{args.visitante}' no reconocido. "
                f"Prueba con: {canonical[:5]}..."
            )
        args.local = home_match or args.local
        args.visitante = away_match or args.visitante
    else:
        logger.warning(
            "No se pudo obtener la lista de equipos; se omite validacion de nombres."
        )

    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)


def _prediction_to_dict(
    ensemble: FoulPredictionEnsemble,
    feat: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta la prediccion y devuelve un dict JSON-serializable."""
    pred = ensemble.predict(feat)
    team_pred = (
        ensemble.predict_team_fouls(feat, total_prediction=pred, reconcile=True) or {}
    )
    return {
        "match": f"{feat['home_team']} vs {feat['away_team']}",
        "date": feat.get("date") or "",
        "jornada": feat.get("matchday") or 0,
        "referee": feat.get("referee") or "",
        "expected_fouls": float(pred.expected_fouls),
        "referee_strict_prob": float(pred.referee_strict_prob),
        "weights": [float(w) for w in pred.weights],
        "home_expected": float(team_pred.get("home_expected", 0.0)),
        "away_expected": float(team_pred.get("away_expected", 0.0)),
        "total_expected": float(team_pred.get("total_expected", pred.expected_fouls)),
        "reconciled": bool(team_pred.get("reconciled", False)),
        "over_under": {
            str(k): {"over": float(v[0]), "under": float(v[1])}
            for k, v in (pred.over_under or {}).items()
        },
    }


# ---------------------------------------------------------------------------
# Modo batch
# ---------------------------------------------------------------------------


def _run_batch(
    args: argparse.Namespace,
    ensemble: FoulPredictionEnsemble,
) -> list[dict[str, Any]]:
    batch_path = Path(args.batch_file)
    if not batch_path.is_file():
        logger.error(f"Archivo batch no encontrado: {batch_path}")
        sys.exit(1)

    matches: list[dict[str, Any]] = []
    if batch_path.suffix.lower() == ".json":
        with open(batch_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("El JSON de batch debe ser una lista de partidos.")
            sys.exit(1)
        matches = data
    elif batch_path.suffix.lower() == ".csv":
        with open(batch_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                matches.append(
                    {
                        "local": row.get("local") or row.get("home_team"),
                        "visitante": row.get("visitante") or row.get("away_team"),
                        "fecha": row.get("fecha") or row.get("match_date"),
                        "jornada": int(row["jornada"]) if row.get("jornada") else None,
                        "arbitro": row.get("arbitro"),
                    }
                )
    else:
        logger.error("Formato no soportado. Usa .json o .csv.")
        sys.exit(1)

    results: list[dict[str, Any]] = []
    for idx, m in enumerate(matches, 1):
        logger.info(f"[{idx}/{len(matches)}] {m.get('local')} vs {m.get('visitante')}")
        try:
            feat = generate_features(
                equipo_local=m["local"],
                equipo_visitante=m["visitante"],
                jornada=m.get("jornada") or args.jornada,
                fecha_partido=m.get("fecha") or args.fecha,
                arbitro=m.get("arbitro") or args.arbitro,
                features_profile=args.features_profile,
                refresh_data=args.refresh_data,
            )
            results.append(_prediction_to_dict(ensemble, feat))
        except Exception as exc:
            logger.error(f"  Error en {m.get('local')} vs {m.get('visitante')}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predice faltas para partidos de La Liga usando el ensemble entrenado."
    )
    parser.add_argument("--local", default="Real Madrid", help="Equipo local")
    parser.add_argument("--visitante", default="Ath Madrid", help="Equipo visitante")
    parser.add_argument(
        "--jornada", type=int, default=29, help="Numero de jornada (1-38)"
    )
    parser.add_argument("--fecha", default="2026-03-22", help="Fecha YYYY-MM-DD")
    parser.add_argument("--arbitro", default=None, help="Nombre del arbitro (opcional)")

    parser.add_argument(
        "--features-profile",
        dest="features_profile",
        choices=["prediction", "training", "minimal"],
        default=None,
        help="Perfil de features a usar",
    )
    parser.add_argument(
        "--refresh-data",
        dest="refresh_data",
        action="store_true",
        help="Fuerza recarga del estado desde Supabase",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        metavar="PATH",
        default=None,
        help="Guarda la prediccion en este fichero JSON",
    )
    parser.add_argument(
        "--batch-file",
        dest="batch_file",
        metavar="PATH",
        default=None,
        help="Fichero JSON o CSV con lista de partidos para prediccion en lote",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Nivel de logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # -- Estado historico --
    logger.info("Cargando estado historico desde Supabase...")
    try:
        state = _load_state(refresh=args.refresh_data)
        logger.info(
            f"Estado cargado. Equipos conocidos: {len(state.get('scores', {}))}"
        )
    except Exception as exc:
        logger.error(f"No se pudo cargar el estado: {exc}")
        sys.exit(1)

    # -- Validacion (solo modo single) --
    if not args.batch_file:
        _validate_single(args, state)

    # -- Ensemble --
    logger.info(f"Cargando ensemble desde {CHECKPOINT_DIR}...")
    try:
        ensemble = _load_ensemble()
        logger.info("Ensemble cargado correctamente.")
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"No se pudo cargar el ensemble: {exc}")
        sys.exit(1)

    # -- Prediccion --
    if args.batch_file:
        results = _run_batch(args, ensemble)
        output_json = json.dumps(results, indent=2, ensure_ascii=False)
        if args.output_json:
            Path(args.output_json).write_text(output_json, encoding="utf-8")
            logger.info(f"Resultados guardados en {args.output_json}")
        else:
            print(output_json)
    else:
        logger.info(
            f"Generando features: {args.local} vs {args.visitante} "
            f"| J{args.jornada} | {args.fecha}"
        )
        try:
            feat = generate_features(
                equipo_local=args.local,
                equipo_visitante=args.visitante,
                jornada=args.jornada,
                fecha_partido=args.fecha,
                arbitro=args.arbitro,
                features_profile=args.features_profile,
                refresh_data=args.refresh_data,
            )
        except Exception as exc:
            logger.error(f"Error generando features: {exc}")
            sys.exit(1)

        # Salida legible siempre
        _print_prediction(ensemble, feat)

        # JSON opcional
        if args.output_json:
            try:
                result = _prediction_to_dict(ensemble, feat)
                Path(args.output_json).write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                logger.info(f"Prediccion guardada en {args.output_json}")
            except Exception as exc:
                logger.error(f"No se pudo escribir el JSON: {exc}")
                sys.exit(1)

    logger.info("Listo.")


if __name__ == "__main__":
    main()

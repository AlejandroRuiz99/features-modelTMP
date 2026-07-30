"""Procesado de cuotas de apuestas para el builder de features."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional

from HWFP.features.core.utils import (
    dedupe_dicts,
    extract_ou,
    no_vig,
    no_vig_3,
    market_entropy,
    norm_text,
    safe_float,
    team_match,
)


# ---------------------------------------------------------------------------
# Cuotas historicas (datos de entrenamiento — CSV Bet365)
# ---------------------------------------------------------------------------


def market_features_from_historical_odds(odds: dict) -> dict:
    """Convierte cuotas historicas Bet365 en features de mercado sin vigorish.

    Usado exclusivamente en el pipeline de training_data donde las cuotas
    vienen del CSV historico (b365_home, b365_draw, b365_away, ou25_*).

    has_market_odds=True  -> cuotas 1X2 reales disponibles.
    has_market_odds=False -> sin cuotas; valores neutros (1/3, 0.5).
    """
    c_h = odds.get("b365_home")
    c_d = odds.get("b365_draw")
    c_a = odds.get("b365_away")
    c_ov = odds.get("ou25_over")
    c_un = odds.get("ou25_under")

    has_odds = bool(c_h and c_d and c_a)

    if has_odds:
        p_h, p_d, p_a = no_vig_3(float(c_h), float(c_d), float(c_a))
    else:
        p_h, p_d, p_a = 1 / 3, 1 / 3, 1 / 3

    p_ov, p_un = no_vig(float(c_ov), float(c_un)) if (c_ov and c_un) else (0.5, 0.5)

    return {
        "has_market_odds": has_odds,
        "market_home_win_prob": p_h,
        "market_draw_prob": p_d,
        "market_away_win_prob": p_a,
        "market_favorite_prob": round(max(p_h, p_a), 4),
        "market_balance": round(1.0 - abs(p_h - p_a), 4),
        "market_entropy": market_entropy(p_h, p_d, p_a),
        "market_ou25_over_prob": p_ov,
        "market_ou25_under_prob": p_un,
    }


# ---------------------------------------------------------------------------
# Configuracion de mercados O/U
# ---------------------------------------------------------------------------

# (market_key, required_tokens, forbidden_tokens)
# Orden importa: entradas mas especificas primero para evitar falsos positivos.
_OU_MARKETS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("shots_on_target_ou", ("mas/menos tiros a puerta",), ()),
    ("goals_ou", ("mas/menos", "goles"), ()),
    ("fouls_ou", ("faltas", "mas/menos"), ()),
    ("cards_ou", ("tarjetas", "mas/menos"), ()),
    ("corners_ou", ("corner", "mas/menos"), ()),
    ("shots_ou", ("mas/menos tiros",), ("a puerta",)),
    ("offsides_ou", ("fuera de juego", "mas/menos"), ()),
]

_PREFERRED_LINES: dict[str, float] = {
    "goals_ou": 2.5,
    "fouls_ou": 24.5,
    "cards_ou": 4.5,
    "corners_ou": 9.5,
    "shots_ou": 24.5,
    "shots_on_target_ou": 8.5,
    "offsides_ou": 4.5,
}

_TOP20_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("1x2", ("1x2",)),
    ("goals_ou", ("mas/menos total goles",)),
    ("btts", ("marcan ambos equipos",)),
    ("draw_no_bet", ("apuesta sin empate",)),
    ("double_chance", ("doble oportunidad",)),
    ("asian_handicap", ("handicap asiatico",)),
    ("first_team_to_score", ("primer equipo en marcar",)),
    ("ht_or_ft", ("resultado al descanso o final",)),
    ("goals_ou_first_half", ("1 parte - mas/menos total goles",)),
    ("corners_ou_total", ("total de corner mas/menos",)),
    ("corners_more_team", ("equipo con mas corner",)),
    ("shots_ou_total", ("mas/menos tiros",)),
    ("shots_on_target_ou_total", ("mas/menos tiros a puerta",)),
    ("team_more_shots", ("equipo con mas tiros",)),
    ("team_more_shots_on_target", ("equipo con mas tiros a puerta",)),
    ("offsides_ou_total", ("total de fueras de juego mas/menos",)),
    ("throwins_ou_total", ("total de saques de banda mas/menos",)),
    ("penalty_yes_no", ("habra penalti",)),
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _classify_ou_market(nm: str) -> str | None:
    """Clasifica un nombre de mercado normalizado en su market_key O/U."""
    for key, required, forbidden in _OU_MARKETS:
        if all(t in nm for t in required) and not any(t in nm for t in forbidden):
            return key
    return None


def _parse_1x2(
    rows: list[dict[str, Any]],
    eq_local: str,
    eq_visit: str,
) -> dict[str, float]:
    """Extrae cuotas 1X2, resolviendo local/empate/visitante por nombre de equipo."""
    result: dict[str, float] = {}
    for r in rows:
        if norm_text(r.get("mercado")) != "1x2":
            continue
        odd = safe_float(r.get("cuota"))
        if odd is None:
            continue
        sel = r.get("selection")
        ns = norm_text(sel)

        if ns in {"x", "empate", "draw"}:
            result["empate"] = odd
        elif eq_local and team_match(sel, eq_local):
            result["local"] = odd
        elif eq_visit and team_match(sel, eq_visit):
            result["visitante"] = odd
        elif "local" in ns or "home" in ns:
            result["local"] = odd
        elif "visitante" in ns or "away" in ns:
            result["visitante"] = odd
    return result


def _pick_best_line(
    lines_by_value: dict[float, dict[str, float]],
    preferred: float,
) -> dict[str, Any] | None:
    """Elige la linea mas cercana a la preferida que tenga over (y preferiblemente under)."""
    complete = [
        (ln, odds)
        for ln, odds in lines_by_value.items()
        if "over" in odds and "under" in odds
    ]
    partial = [(ln, odds) for ln, odds in lines_by_value.items() if "over" in odds]
    candidates = complete or partial
    if not candidates:
        return None
    ln, odds = min(candidates, key=lambda x: abs(x[0] - preferred))
    return {"line": ln, "over": odds["over"], "under": odds.get("under")}


def _extract_odds_entries(
    rows: list[dict[str, Any]],
    filter_tokens: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Extrae entradas de cuotas cuyo nombre de mercado contiene todos los tokens."""
    entries = []
    for r in rows:
        nm = norm_text(r.get("mercado"))
        if not all(t in nm for t in filter_tokens):
            continue
        odd = safe_float(r.get("cuota"))
        if odd is None:
            continue
        side, line = extract_ou(r.get("selection"))
        entries.append(
            {
                "mercado": r.get("mercado"),
                "selection": r.get("selection"),
                "odd": odd,
                "side": side,
                "line": line,
            }
        )
    return dedupe_dicts(entries, ["mercado", "selection", "odd"])


# ---------------------------------------------------------------------------
# Market data source (composition-root DI point)
# ---------------------------------------------------------------------------
#
# D1 (architecture-boundaries, REQ-12/14): this leaf module must not depend
# on `selection.odds_client` (a legacy Supabase adapter never reachable as a
# top-level import outside of pytest's `pythonpath` config — see the module
# docstring on HWFP.features.core.state_cache for the identical pattern).
# The real market-odds fetcher is wired by the composition root via
# `set_market_data_source(fn)`; without one, `build_market_category()` fails
# with an explicit, actionable RuntimeError instead of a cryptic
# ModuleNotFoundError/network error reaching into a package that was never
# importable in production.

MarketFetchFn = Callable[
    ..., tuple[list[dict[str, Any]], Optional[str], Optional[str]]
]

_market_fetch_fn: MarketFetchFn | None = None


def set_market_data_source(fn: MarketFetchFn) -> None:
    """Injects the callable that fetches raw odds rows for a match.

    The composition root wires the real adapter here (e.g. a Supabase-backed
    fetcher); tests wire a stub. This keeps the leaf `HWFP.features` package
    free of any dependency on a concrete odds-fetching implementation.

    The callable's signature must match
    `(scores, eq_local, eq_visit, *, match_date=None) -> (rows, scraped_at, event_id)`.
    """
    global _market_fetch_fn
    _market_fetch_fn = fn


def _fetch_match_odds_rows(
    scores: dict[str, Any],
    eq_local: str,
    eq_visit: str,
    *,
    match_date: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if _market_fetch_fn is None:
        raise RuntimeError(
            "No market data source configured for "
            "HWFP.features.assembly.betting_odds. Call "
            "set_market_data_source(fn) from the composition root before "
            "build_market_category() is invoked with skip_market_fetch=False."
        )
    return _market_fetch_fn(scores, eq_local, eq_visit, match_date=match_date)


# ---------------------------------------------------------------------------
# Funciones publicas
# ---------------------------------------------------------------------------


def build_market_input_model(
    rows: list[dict[str, Any]],
    eq_local: str = "",
    eq_visit: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Parsea filas de cuotas en el modelo de input para el builder.

    Returns:
        (market_dict, mercados_usados)
    """
    one_x_two = _parse_1x2(rows, eq_local, eq_visit)

    ou_lines: dict[str, dict[float, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        nm = norm_text(r.get("mercado"))
        odd = safe_float(r.get("cuota"))
        if odd is None:
            continue
        market_key = _classify_ou_market(nm)
        if market_key is None:
            continue
        side, line = extract_ou(r.get("selection"))
        if side is None or line is None:
            continue
        ou_lines[market_key][line][side] = odd

    out: dict[str, Any] = {}
    used: list[str] = []

    if {"local", "empate", "visitante"}.issubset(one_x_two):
        out["1x2"] = one_x_two
        used.append("1x2")

    for market_key, lines in ou_lines.items():
        best = _pick_best_line(lines, _PREFERRED_LINES.get(market_key, 0.0))
        if best is not None:
            out[market_key] = best
            used.append(market_key)

    return out, sorted(set(used))


def build_fouls_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extrae todos los mercados de faltas disponibles."""
    return {"mercados_faltas": _extract_odds_entries(rows, ("falta",))}


def build_top20_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extrae las categorias de mercado mas relevantes."""
    out: dict[str, Any] = {}
    for key, tokens in _TOP20_SPECS:
        entries = _extract_odds_entries(rows, tokens)
        if entries:
            out[key] = entries
    return out


def build_market_category(
    *,
    scores: dict[str, Any],
    eq_local: str,
    eq_visit: str,
    model_market_signal: dict[str, Any],
    match_date: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], str | None]:
    """Orquesta la construccion completa de la categoria de mercado.

    Args:
        match_date: Fecha del partido ISO 'YYYY-MM-DD'. Si se pasa, el fetch
            de cuotas busca en una ventana de varios días alrededor (para
            capturar scrapes diarios del bookmaker). Si es None, solo se
            mira el último scrape global (comportamiento legacy).

    Returns:
        (market_category, market_input_model, mercados_usados, scraped_at)
    """
    rows, scraped_at, event_id = _fetch_match_odds_rows(
        scores, eq_local, eq_visit, match_date=match_date
    )

    available = sorted(
        {name for r in rows if (name := (r.get("mercado") or "").strip())}
    )
    market_input_model, used = build_market_input_model(
        rows,
        eq_local=eq_local,
        eq_visit=eq_visit,
    )

    market = {
        "source": "odds_raw",
        "catalogo_disponible": available,
        "por_categoria": {
            "faltas": build_fouls_category(rows),
            "contexto_top20": build_top20_category(rows),
            "modelo_alineacion": model_market_signal.get("markets", {}),
        },
        "resumen_modelo": {
            "global_alignment": model_market_signal.get("global_alignment", {}),
            "markets_used_for_model": used,
        },
        "traza": {
            "event_id": event_id,
            "rows_used": len(rows),
            "odds_scraped_at": scraped_at,
        },
    }
    return market, market_input_model, used, scraped_at

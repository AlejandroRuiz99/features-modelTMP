"""Pure message-formatting functions for Telegram output."""

from __future__ import annotations

from HWFP.core.domain.betting_decision import BettingDecision, Recommendation
from HWFP.core.domain.calibration import CalibrationStatus
from HWFP.core.domain.line_movement import LineMovement
from HWFP.core.domain.performance_snapshot import PerformanceSnapshot

_STATUS_EMOJI = {
    CalibrationStatus.GREEN: "🟢",
    CalibrationStatus.YELLOW: "🟡",
    CalibrationStatus.ORANGE: "🟠",
    CalibrationStatus.RED: "🔴",
}


def format_bet_decision(d: BettingDecision) -> str:
    conf = d.confidence
    lines = [
        f"*BET SIGNAL* {_STATUS_EMOJI.get(None, '🎯')}",  # noqa: B012 — placeholder
        f"Match: `{d.match_id}`",
        f"Market: {d.market} {d.side} {d.line}",
        f"Edge: +{d.edge:.1%}  |  p_model: {d.p_model:.1%}",
        f"Bookmaker: {d.best_bookmaker} @ {d.best_odds_value:.2f}",
        f"Stake: *€{d.stake_euros:.0f}*",
        f"Confidence: {conf.level.value.upper()} (H={conf.pmf_entropy:.2f} bits, ref_n={conf.referee_sample_size})",
    ]
    return "\n".join(lines)


def format_scan_summary(decisions: list[BettingDecision]) -> str:
    bets = [d for d in decisions if d.recommendation == Recommendation.BET]
    skips = [d for d in decisions if d.recommendation == Recommendation.SKIP]

    if not bets:
        return f"Scan complete — no value found ({len(skips)} matches skipped)."

    parts = [f"*SCAN RESULTS* — {len(bets)} BET / {len(skips)} skip\n"]
    for d in bets:
        parts.append(
            f"• `{d.match_id}` {d.side} {d.line} @ {d.best_odds_value:.2f} "
            f"({d.best_bookmaker}) edge={d.edge:.1%} stake=€{d.stake_euros:.0f}"
        )
    return "\n".join(parts)


def format_performance_snapshot(s: PerformanceSnapshot) -> str:
    emoji = _STATUS_EMOJI[s.status]
    lines = [
        f"{emoji} *Performance Snapshot*",
        f"Status: {s.status.value.upper()}",
        f"Bets total: {s.n_bets_total}",
        f"ROI (last 30): {s.roi_trailing_30:+.1%}",
        f"ECE (last 50): {s.ece_trailing_50:.4f}",
        f"Win rate HIGH: {s.win_rate_high_conf:.1%}",
        f"CLV avg: {s.clv_avg:+.1%}",
        f"Kelly reduction: {s.kelly_reduction:.0%}",
        f"As of: {s.as_of.strftime('%Y-%m-%d %H:%M')} UTC",
    ]
    return "\n".join(lines)


def format_line_movement(m: LineMovement) -> str:
    direction = "▲" if m.delta > 0 else "▼"
    return (
        f"*LINE MOVEMENT* {direction}\n"
        f"Match: `{m.match_id}` | {m.bookmaker}\n"
        f"Market: {m.market}\n"
        f"Line: {m.line_before} → {m.line_after} ({direction}{abs(m.delta):.1f})\n"
        f"Odds: {m.odds_before:.2f} → {m.odds_after:.2f}"
    )

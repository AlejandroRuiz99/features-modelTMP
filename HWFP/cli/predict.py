"""CLI entry point: predict a match outcome using the foul prediction pipeline."""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HWFP foul prediction CLI")
    parser.add_argument("--match-id", required=True, help="Match identifier (e.g. M1)")
    parser.add_argument(
        "--mode",
        choices=["fake", "production"],
        default="fake",
        help="Container mode (default: fake)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.mode == "fake":
        from HWFP.serving.composition.container import container_fakes

        use_case = container_fakes()
    else:
        from HWFP.serving.composition.container import container_production

        use_case = container_production()

    from HWFP.core.application.predict_match import PredictMatchInput

    inp = PredictMatchInput(
        match_id=args.match_id,
        market="fouls_over_under",
        line=22.5,
        side="over",
        bankroll=1000.0,
    )

    try:
        out = use_case.execute(inp)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    payload = {
        "match_id": out.prediction.match_id,
        "pmf": list(out.prediction.pmf.pmf),
        "ev": out.ev.ev,
        "fair_prob": out.ev.fair_prob,
        "stake": out.stake.stake,
        "kelly_fraction": out.stake.kelly_fraction,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())

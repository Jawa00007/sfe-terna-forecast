"""Phase 3 economic backtest (PRD §9, go/no-go gate).

Walk-forward, expanding window. Reports net margin EUR/MWh after charges vs the perfect-hedge
baseline, the charge-sensitivity band, and the ablation vs a rule-of-thumb scheduler.

    python -m sfe.scripts.run_backtest --from 2024-06-01 --to 2025-02-28 \\
        --min-train-days 120 --test-days 30
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import date

from sfe.backtest.ablation import run_naive_walkforward
from sfe.backtest.economic import go_no_go
from sfe.backtest.sensitivity import charge_sensitivity
from sfe.backtest.walkforward import run_walkforward
from sfe.models.registry import log_run

_TINY = {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, "n_jobs": 1}


def _fmt(d: dict) -> str:
    return "\n".join(
        f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}"
        for k, v in d.items()
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-run-backtest", description=__doc__)
    p.add_argument("--from", dest="date_from", required=True, type=date.fromisoformat)
    p.add_argument("--to", dest="date_to", required=True, type=date.fromisoformat)
    p.add_argument("--horizon", default="D-1")
    p.add_argument("--min-train-days", type=int, default=120)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--fast", action="store_true", help="smaller trees (smoke runs)")
    p.add_argument("--min-margin", type=float, default=0.0, help="go/no-go threshold EUR/MWh")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = p.parse_args(argv)

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8")

    params = _TINY if args.fast else None
    wf = run_walkforward(
        args.date_from,
        args.date_to,
        horizon=args.horizon,
        sign_params=params,
        spread_params=params,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
    )
    report = wf["report"]
    if not report:
        print("no recommendations produced - land data and build features first")
        return 1

    ok, msg = go_no_go(report, min_margin_eur_per_mwh=args.min_margin)
    sens = charge_sensitivity(wf["recs"])
    naive = run_naive_walkforward(
        args.date_from,
        args.date_to,
        horizon=args.horizon,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
    )
    naive_margin = naive["report"].get("net_margin_eur_per_mwh")
    beats = (
        naive_margin is not None
        and report.get("net_margin_eur_per_mwh") is not None
        and report["net_margin_eur_per_mwh"] > naive_margin
    )
    log_run(
        "backtest",
        params={
            "date_from": str(args.date_from),
            "date_to": str(args.date_to),
            "horizon": args.horizon,
            "min_train_days": args.min_train_days,
            "test_days": args.test_days,
        },
        metrics={k: v for k, v in report.items() if isinstance(v, (int, float))},
        tags={"go": str(ok), "gate_msg": msg, "beats_naive": str(beats)},
    )

    if args.json:
        print(
            json.dumps(
                {
                    "report": report,
                    "go": ok,
                    "gate_msg": msg,
                    "sensitivity": sens.to_dict(orient="records"),
                    "naive_net_margin_eur_per_mwh": naive_margin,
                },
                indent=2,
                default=str,
            )
        )
        return 0 if ok else 2

    print("# Economic backtest\n")
    print(f"folds: {len(wf['folds'])}  horizon={args.horizon}\n")
    print("## Model policy (net of charges)")
    print(_fmt(report))
    print(f"\n## Go / no-go gate\n  {'PASS' if ok else 'FAIL'} - {msg}\n")
    print("## Charge sensitivity (B-5)")
    print(sens.to_string(index=False))
    print("\n## Ablation vs rule-of-thumb scheduler (B-6)")
    print(f"  model net margin  {report.get('net_margin_eur_per_mwh')}")
    print(f"  naive net margin  {naive_margin}")
    print(f"  model beats naive: {beats}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

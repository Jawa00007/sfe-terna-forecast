"""Shadow-mode batch job (PRD Phase 4, NF-1): generate + log the D-1 forecast.

Run once per day before the MGP gate. Produces the bundle, records every recommendation to
the audit log, and prints a scheduler-facing summary. Nothing is acted on.

    python -m sfe.scripts.shadow_run --date 2025-01-16 --horizon D-1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import date, timedelta

from sfe.monitoring.audit import AuditLog
from sfe.monitoring.drift import staleness_report
from sfe.serve.inference import generate_bundle

_FEEDS = [
    "landing/terna/fees/daily-prices",
    "landing/terna/fees/daily-macrozonal-imbalance",
    "landing/terna/fees/macrozonal-no-arbitrage-prices",
    "landing/terna/market/load-forecast",
    "landing/terna/generation/wind-production-forecast",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-shadow-run", description=__doc__)
    p.add_argument("--date", type=date.fromisoformat, default=date.today() + timedelta(days=1))
    p.add_argument("--horizon", default="D-1")
    p.add_argument("--no-audit", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8")

    bundle = generate_bundle(args.date, args.horizon)
    stale = staleness_report(_FEEDS, as_of=bundle.as_of)
    ids = [] if args.no_audit or bundle.table.empty else AuditLog().record(bundle)

    tbl = bundle.table
    active = tbl[~tbl["abstain"]] if not tbl.empty else tbl
    summary = {
        "delivery_day": args.date.isoformat(),
        "horizon": args.horizon,
        "as_of": bundle.as_of.isoformat(),
        "degraded": bundle.degraded,
        "degraded_reasons": bundle.degraded_reasons,
        "n_mtu": int(len(tbl)),
        "n_active_positions": int(len(active)),
        "abstention_fraction": float(tbl["abstain"].mean()) if len(tbl) else None,
        "expected_net_margin_eur": (
            float(active["expected_net_margin_eur"].sum()) if len(active) else 0.0
        ),
        "sum_abs_delta_mwh": float(active["delta_mwh"].abs().sum()) if len(active) else 0.0,
        "stale_feeds": stale.loc[stale["stale"], "dataset"].tolist(),
        "audit_rows": len(ids),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"# Shadow forecast  {args.date}  ({args.horizon})\n")
    if bundle.degraded:
        print("!! DEGRADED: " + "; ".join(bundle.degraded_reasons) + "\n")
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    if not active.empty:
        print("\n  top positions by |expected net margin|:")
        top = active.reindex(
            active["expected_net_margin_eur"].abs().sort_values(ascending=False).index
        ).head(8)
        for _, r in top.iterrows():
            print(
                f"    {r['_area']:5s} {str(r['_mtu_start'])[:16]}  "
                f"Delta={r['delta_mwh']:+7.1f}  E[net]={r['expected_net_margin_eur']:+9.0f} EUR  "
                f"p(long)={r['sign_prob']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

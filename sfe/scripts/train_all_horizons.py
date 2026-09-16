"""Train + register the model set for every decision horizon (PRD F-9).

F-9 requires a separate model per decision point (D-1, each MI session) rather than one
model reused everywhere - horizons differ in how much of the imbalance is already knowable
(publication lags, intraday price discovery). This trains and registers all of them from a
common lookback window.

    python -m sfe.scripts.train_all_horizons --end 2025-01-25 --lookback-days 90
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from sfe.features.spec import HORIZON_OFFSET_H
from sfe.features.store import default_store
from sfe.serve.inference import ensure_models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-train-all-horizons", description=__doc__)
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--lookback-days", type=int, default=120)
    p.add_argument(
        "--horizons", default=",".join(HORIZON_OFFSET_H), help="comma-separated horizon labels"
    )
    args = p.parse_args(argv)

    store = default_store()
    results = {}
    for horizon in args.horizons.split(","):
        try:
            sign, spread = ensure_models(
                horizon,
                end=args.end,
                lookback_days=args.lookback_days,
                store=store,
                force_train=True,
            )
            results[horizon] = {
                "trained": True,
                "sign_features": len(sign.feature_names),
                "spread_features": len(spread.feature_names),
            }
        except RuntimeError as e:
            results[horizon] = {"trained": False, "error": str(e)}

    print(json.dumps(results, indent=2))
    return 0 if all(r.get("trained") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

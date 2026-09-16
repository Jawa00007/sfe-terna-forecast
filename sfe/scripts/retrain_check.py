"""Drift-triggered retraining check (PRD F-10).

For each horizon: retrain if the registered model is missing, older than ``--max-age-days``,
or enough features have drifted (PSI) between a reference and a current window. ``--apply``
actually retrains + re-registers the triggered horizons; without it, this only reports the
decision (safe to run on a schedule).

    python -m sfe.scripts.retrain_check --end 2025-01-25 --apply
    python -m sfe.scripts.retrain_check --horizons D-1 --force --apply   # manual override
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from sfe.features.spec import HORIZON_OFFSET_H
from sfe.features.store import default_store
from sfe.models.registry import log_run
from sfe.models.retrain import should_retrain
from sfe.serve.inference import ensure_models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-retrain-check", description=__doc__)
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument(
        "--horizons", default=",".join(HORIZON_OFFSET_H), help="comma-separated horizon labels"
    )
    p.add_argument("--max-age-days", type=float, default=14.0)
    p.add_argument("--drift-alert-fraction", type=float, default=0.2)
    p.add_argument("--reference-days", type=int, default=60)
    p.add_argument("--current-days", type=int, default=14)
    p.add_argument("--lookback-days", type=int, default=120, help="training window if applied")
    p.add_argument("--force", action="store_true", help="retrain regardless of age/drift")
    p.add_argument("--apply", action="store_true", help="actually retrain triggered horizons")
    args = p.parse_args(argv)

    store = default_store()
    out = {}
    for horizon in args.horizons.split(","):
        decision = should_retrain(
            horizon,
            max_age_days=args.max_age_days,
            drift_alert_fraction=args.drift_alert_fraction,
            reference_days=args.reference_days,
            current_days=args.current_days,
            end=args.end,
            store=store,
            force=args.force,
        )
        entry = {
            "retrain": decision.retrain,
            "reasons": decision.reasons,
            "model_age_days": decision.model_age_days,
            "n_features_checked": int(len(decision.drift)),
            "n_features_drift_alert": (
                int((decision.drift["level"] == "alert").sum()) if not decision.drift.empty else 0
            ),
        }
        if decision.retrain and args.apply:
            try:
                sign, spread = ensure_models(
                    horizon,
                    end=args.end,
                    lookback_days=args.lookback_days,
                    store=store,
                    force_train=True,
                )
                entry["applied"] = True
                entry["n_features"] = len(sign.feature_names)
                log_run(
                    f"retrain_{horizon}",
                    params={"end": str(args.end), "lookback_days": args.lookback_days},
                    metrics={},
                    tags={"reasons": ";".join(decision.reasons)},
                )
            except RuntimeError as e:
                entry["applied"] = False
                entry["apply_error"] = str(e)
        else:
            entry["applied"] = False
        out[horizon] = entry

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

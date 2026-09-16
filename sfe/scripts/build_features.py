"""Materialise the point-in-time feature store for a range of delivery days (PRD F-4).

Example (against data already landed from the mock)::

    python -m sfe.scripts.build_features --from 2025-01-08 --to 2025-01-10 --horizon D-1
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

from sfe.features.spec import HORIZON_OFFSET_H, DecisionContext, as_of_for_horizon
from sfe.features.store import default_store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-build-features", description=__doc__)
    p.add_argument("--from", dest="date_from", required=True, type=date.fromisoformat)
    p.add_argument("--to", dest="date_to", required=True, type=date.fromisoformat)
    p.add_argument("--horizon", default="D-1", choices=sorted(HORIZON_OFFSET_H))
    p.add_argument("--dataset", default="pit")
    args = p.parse_args(argv)

    store = default_store()
    contexts = []
    d = args.date_from
    while d <= args.date_to:
        contexts.append(
            DecisionContext(
                as_of=as_of_for_horizon(d, args.horizon),
                delivery_day=d,
                horizon=args.horizon,
            )
        )
        d += timedelta(days=1)

    stats = store.materialize(contexts, dataset=args.dataset)
    print(json.dumps({"contexts": len(contexts), **stats}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

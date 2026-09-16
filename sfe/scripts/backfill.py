"""Backfill / incremental pull entrypoint (PRD F-1, F-2).

Examples
--------
Dry run (list the planned pulls, no HTTP)::

    python -m sfe.scripts.backfill --targets --from 2021-01-01 --to 2025-12-31 --dry-run

Against the mock server::

    python -m sfe.testing.mock_terna &
    SFE_TERNA__BASE_URL=http://127.0.0.1:8900 \\
      SFE_TERNA__TOKEN_URL=http://127.0.0.1:8900/oauth/accessToken \\
      SFE_TERNA__CLIENT_ID=mock SFE_TERNA__CLIENT_SECRET=mock \\
      python -m sfe.scripts.backfill --endpoint daily-prices --from 2025-01-01 --to 2025-01-07
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from sfe.curate import regimes
from sfe.ingest.endpoints import get_endpoint, iter_endpoints
from sfe.ingest.flows import ingest_endpoint


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sfe-backfill", description=__doc__)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--endpoint", help="single endpoint name, e.g. daily-prices")
    sel.add_argument("--targets", action="store_true", help="all sign/price/charge endpoints")
    sel.add_argument("--group", help="every endpoint in a catalogue group, e.g. fees")
    p.add_argument("--from", dest="date_from", required=True, type=date.fromisoformat)
    p.add_argument("--to", dest="date_to", required=True, type=date.fromisoformat)
    p.add_argument("--data-type", choices=["Orario", "Quarto Orario"], default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan; make no requests")
    return p.parse_args(argv)


def _selected(args: argparse.Namespace) -> list[str]:
    if args.endpoint:
        return [get_endpoint(args.endpoint).name]
    if args.group:
        return [e.name for e in iter_endpoints(group=args.group)]
    return [e.name for e in iter_endpoints() if e.role in {"target_sign", "target_price", "charge"}]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    names = _selected(args)
    dt = args.data_type or regimes.terna_data_type(args.date_from)

    if args.dry_run:
        plan = [
            {
                "endpoint": n,
                "path": get_endpoint(n).path,
                "from": args.date_from.strftime("%d/%m/%Y"),
                "to": args.date_to.strftime("%d/%m/%Y"),
                "dataType": dt if get_endpoint(n).supports_dataType else None,
                "dataset": get_endpoint(n).dataset,
            }
            for n in names
        ]
        print(json.dumps({"dry_run": True, "count": len(plan), "plan": plan}, indent=2))
        return 0

    results = [
        ingest_endpoint(n, args.date_from, args.date_to, data_type=args.data_type) for n in names
    ]
    print(json.dumps({"dry_run": False, "results": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

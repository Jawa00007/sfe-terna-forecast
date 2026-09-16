"""EDA on the imbalance spread distribution S (PRD Phase 1 exit, §8 baselines).

Text/Markdown report - no plotting deps. Consumes whatever fees data is landed.

    python -m sfe.scripts.eda_spread            # from ./data (or SFE_STORAGE__ROOT)
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np
import pandas as pd

from sfe.curate.timebase import utc_to_rome
from sfe.curate.vintage import latest_view, pivot_fields
from sfe.features.targets import assemble_targets
from sfe.io.storage import get_storage


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.2f}"


def _dist_block(s: pd.Series, label: str) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return f"### {label}\n\n_no data_\n"
    qs = s.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    abs_sorted = s.abs().sort_values(ascending=False)
    top5_share = abs_sorted.head(max(1, len(s) // 20)).sum() / abs_sorted.sum()
    lines = [
        f"### {label}",
        "",
        f"- n = {len(s):,}",
        f"- mean = {_fmt(s.mean())}   std = {_fmt(s.std())}",
        f"- min / max = {_fmt(s.min())} / {_fmt(s.max())}",
        "- quantiles: " + ", ".join(f"p{int(q*100)}={_fmt(v)}" for q, v in qs.items()),
        f"- P(S > 0) = {_fmt((s > 0).mean() * 100)}%   (area short)",
        f"- top-5% |S| periods carry {_fmt(top5_share * 100)}% of total |S|  "
        "(heavy-tail check, PRD risk row)",
        "",
    ]
    return "\n".join(lines)


def _revision_block() -> str:
    """Preliminary -> definitive revision of unbalance_price (DQ-1 risk term)."""
    storage = get_storage()
    prelim = storage.read_dataset("landing/terna/fees/preliminary-prices")
    defin = storage.read_dataset("landing/terna/fees/daily-prices")
    if prelim.empty or defin.empty:
        return "### Preliminary -> definitive revision\n\n_need both vintages landed_\n"

    def _px(df: pd.DataFrame) -> pd.DataFrame:
        w = pivot_fields(latest_view(df))
        return w[["_area", "_mtu_start", "unbalance_price_EURxMWh"]]

    merged = _px(prelim).merge(
        _px(defin), on=["_area", "_mtu_start"], suffixes=("_prelim", "_def")
    )
    rev = (
        merged["unbalance_price_EURxMWh_def"] - merged["unbalance_price_EURxMWh_prelim"]
    ).dropna()
    if rev.empty:
        return "### Preliminary -> definitive revision\n\n_no overlapping keys_\n"
    changed = (rev.abs() > 1e-6).mean()
    return "\n".join(
        [
            "### Preliminary -> definitive revision",
            "",
            f"- n overlapping keys = {len(rev):,}",
            f"- fraction revised = {_fmt(changed * 100)}%",
            f"- revision mean = {_fmt(rev.mean())}   std = {_fmt(rev.std())}",
            f"- revision p5 / p50 / p95 = "
            f"{_fmt(rev.quantile(0.05))} / {_fmt(rev.quantile(0.5))} / {_fmt(rev.quantile(0.95))}",
            "- treat |revision| as a risk term in the decision layer (PRD DQ-1).",
            "",
        ]
    )


def build_report() -> str:
    definitive = assemble_targets(as_of=None)
    out = ["# Spread S EDA", ""]
    if definitive.empty:
        return "# Spread S EDA\n\n_No fees data landed. Run a backfill first._\n"

    definitive = definitive.assign(
        _mtu_start=pd.to_datetime(definitive["_mtu_start"], utc=True)
    )
    definitive["hour_rome"] = definitive["_mtu_start"].map(
        lambda t: utc_to_rome(t.to_pydatetime()).hour
    )

    out.append(_dist_block(definitive["spread_S"], "All areas / all periods"))
    for area, g in definitive.groupby("_area"):
        out.append(_dist_block(g["spread_S"], f"Area {area}"))

    # sign base rates (baseline to beat, §8)
    sign = pd.to_numeric(definitive["imb_sign"], errors="coerce").dropna()
    if not sign.empty:
        out += [
            "### Sign base rates",
            "",
            f"- P(sign = +1, area long) = {_fmt((sign > 0).mean() * 100)}%",
            f"- P(sign = -1, area short) = {_fmt((sign < 0).mean() * 100)}%",
            "- persistence (sign_t == sign_{t-1 day}) not computed here; see backtest",
            "",
        ]

    out.append(_revision_block())
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sfe-eda-spread", description=__doc__)
    p.add_argument("--out", help="write the report here instead of stdout")
    args = p.parse_args(argv)
    report = build_report()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"wrote {args.out}")
    else:
        with contextlib.suppress(AttributeError, ValueError):
            sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

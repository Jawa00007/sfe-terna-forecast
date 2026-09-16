"""Ablation vs a rule-of-thumb scheduler (PRD B-6).

The naive policy replaces the learned spread fan with a seasonal-climatology point estimate
(mean S by area / month / local hour, from data available before each decision) wrapped in
a fixed-width band, and the sign probability with the climatological base rate. Everything
downstream - charges, sizing, limits, scoring - is identical, so the comparison isolates
the value of the models.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from sfe.backtest.economic import economic_report, realize_pnl
from sfe.backtest.walkforward import _attach_load, _daterange, build_actuals, make_folds
from sfe.curate.timebase import utc_to_rome
from sfe.decision.policy import recommend_frame
from sfe.features.spec import DecisionContext, as_of_for_horizon
from sfe.features.store import default_store
from sfe.features.targets import assemble_targets
from sfe.models.spread import QUANTILES


def _climatology(as_of) -> pd.DataFrame:
    tgt = assemble_targets(as_of=as_of)
    if tgt.empty:
        return tgt
    t = tgt.copy()
    t["_mtu_start"] = pd.to_datetime(t["_mtu_start"], utc=True)
    loc = t["_mtu_start"].map(lambda x: utc_to_rome(x.to_pydatetime()))
    t["month"] = [d.month for d in loc]
    t["hour"] = [d.hour for d in loc]
    t["sign_pos"] = (pd.to_numeric(t["imb_sign"], errors="coerce") > 0).astype(float)
    g = t.groupby(["_area", "month", "hour"], dropna=False).agg(
        clim_S=("spread_S", "mean"), clim_sd=("spread_S", "std"), clim_sign=("sign_pos", "mean")
    )
    return g


def _clim_lookup(clim: pd.DataFrame, col: str, keys: list, n: int, default: float) -> np.ndarray:
    if clim.empty or col not in clim.columns:
        return np.full(n, default)
    return clim[col].reindex(keys).to_numpy()


def run_naive_walkforward(
    start: date,
    end: date,
    *,
    horizon: str = "D-1",
    store=None,
    storage=None,
    min_train_days: int = 90,
    test_days: int = 30,
) -> dict:
    store = store or default_store()
    folds = make_folds(start, end, min_train_days=min_train_days, test_days=test_days)
    recs = []
    for fold in folds:
        for day in _daterange(fold.test_start, fold.test_end):
            ctx = DecisionContext(
                as_of=as_of_for_horizon(day, horizon), delivery_day=day, horizon=horizon
            )
            feats = store.assemble(ctx)
            if feats.empty:
                continue
            clim = _climatology(ctx.as_of)
            frame = feats[["_area", "_mtu_start"]].copy()
            frame["_mtu_start"] = pd.to_datetime(frame["_mtu_start"], utc=True)
            loc = frame["_mtu_start"].map(lambda x: utc_to_rome(x.to_pydatetime()))
            n = len(frame)
            keys = list(
                zip(frame["_area"], [d.month for d in loc], [d.hour for d in loc], strict=True)
            )
            cS = np.nan_to_num(_clim_lookup(clim, "clim_S", keys, n, 0.0), nan=0.0)
            cSd = _clim_lookup(clim, "clim_sd", keys, n, 10.0)
            finite_sd = cSd[np.isfinite(cSd)]
            fill_sd = float(finite_sd.mean()) if finite_sd.size else 10.0
            cSd = np.nan_to_num(cSd, nan=fill_sd)
            cSg = _clim_lookup(clim, "clim_sign", keys, n, 0.5)

            frame["sign_prob"] = np.nan_to_num(cSg, nan=0.5)
            zscores = {0.1: -1.2816, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745, 0.9: 1.2816}
            for q in QUANTILES:
                frame[f"S_p{int(q * 100)}"] = cS + zscores[q] * cSd
            frame = _attach_load(frame, ctx.as_of, storage)
            recs.append(recommend_frame(frame, as_of=ctx.as_of, storage=storage))

    if not recs:
        return {"recs": pd.DataFrame(), "pnl": pd.DataFrame(), "report": {}}
    allr = pd.concat(recs, ignore_index=True)
    actual = build_actuals(allr[["_area", "_mtu_start"]].drop_duplicates(), storage=storage)
    pnl = realize_pnl(allr, actual)
    return {"recs": allr, "pnl": pnl, "report": economic_report(pnl)}

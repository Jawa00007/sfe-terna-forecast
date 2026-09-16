"""Walk-forward, expanding-window economic backtest (PRD B-1, B-2, B-3).

For each fold: fit the sign + spread models on ``[start, train_end]``, then for every
delivery day in the test window replay the as-of feature view, forecast, and run the
decision policy (charges point-in-time). Recommendations from all folds are scored once
against realised outcomes. No shuffling, no k-fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from sfe.decision.charges import ChargeModel
from sfe.decision.policy import recommend_frame
from sfe.features.spec import DecisionContext, as_of_for_horizon
from sfe.features.store import FeatureStore, default_store
from sfe.features.targets import assemble_targets
from sfe.models.dataset import build_training_frame, feature_matrix, time_split
from sfe.models.sign import SignModel
from sfe.models.spread import QUANTILES, SpreadQuantileModel

_LOAD_DATASET = "landing/terna/market/load-forecast"


@dataclass(frozen=True)
class Fold:
    train_start: date
    train_end: date        # inclusive
    test_start: date
    test_end: date         # inclusive


def make_folds(
    start: date, end: date, *, min_train_days: int = 90, test_days: int = 30
) -> list[Fold]:
    folds: list[Fold] = []
    test_start = start + timedelta(days=min_train_days)
    while test_start <= end:
        test_end = min(test_start + timedelta(days=test_days - 1), end)
        folds.append(Fold(start, test_start - timedelta(days=1), test_start, test_end))
        test_start = test_end + timedelta(days=1)
    return folds


def _daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def _load_lookup(as_of, storage) -> pd.Series:
    from sfe.curate.vintage import as_of_view, pivot_fields
    from sfe.io.storage import get_storage

    storage = storage or get_storage()
    raw = storage.read_dataset(_LOAD_DATASET)
    if raw.empty:
        return pd.Series(dtype="float64")
    wide = pivot_fields(as_of_view(raw, as_of) if as_of is not None else raw)
    if "forecast_load_MW" not in wide.columns:
        return pd.Series(dtype="float64")
    wide["_mtu_start"] = pd.to_datetime(wide["_mtu_start"], utc=True)
    return wide.set_index(["_area", "_mtu_start"])["forecast_load_MW"]


def _attach_load(frame: pd.DataFrame, as_of, storage) -> pd.DataFrame:
    lut = _load_lookup(as_of, storage)
    if lut.empty:
        return frame
    keys = list(zip(frame["_area"], pd.to_datetime(frame["_mtu_start"], utc=True), strict=True))
    frame = frame.copy()
    frame["load_mwh"] = pd.to_numeric(pd.Series(lut.reindex(keys).to_numpy()), errors="coerce")
    return frame


def build_actuals(
    keys: pd.DataFrame, *, storage=None, charge_multiplier: float = 1.0
) -> pd.DataFrame:
    """Realised spread, charge and volume for the given ``(_area, _mtu_start)`` keys."""
    tgt = assemble_targets(as_of=None, storage=storage)[["_area", "_mtu_start", "spread_S"]]
    tgt["_mtu_start"] = pd.to_datetime(tgt["_mtu_start"], utc=True)
    k = keys.copy()
    k["_mtu_start"] = pd.to_datetime(k["_mtu_start"], utc=True)
    out = k.merge(tgt, on=["_area", "_mtu_start"], how="left")

    charge = ChargeModel.from_config(multiplier=charge_multiplier).charge_per_mwh(
        out[["_area", "_mtu_start"]], as_of=None, storage=storage
    )
    out["charge_realized_eur_per_mwh"] = charge.to_numpy()

    lut = _load_lookup(None, storage)
    if not lut.empty:
        keyl = list(zip(out["_area"], out["_mtu_start"], strict=True))
        out["load_realized_mwh"] = pd.to_numeric(
            pd.Series(lut.reindex(keyl).to_numpy()), errors="coerce"
        )
    else:
        out["load_realized_mwh"] = float("nan")
    return out


def run_walkforward(
    start: date,
    end: date,
    *,
    horizon: str = "D-1",
    store: FeatureStore | None = None,
    storage=None,
    sign_params: dict | None = None,
    spread_params: dict | None = None,
    calibration: str = "isotonic",
    min_train_days: int = 90,
    test_days: int = 30,
    min_train_rows: int = 200,
) -> dict:
    store = store or default_store()
    folds = make_folds(start, end, min_train_days=min_train_days, test_days=test_days)
    all_recs: list[pd.DataFrame] = []
    fold_meta: list[dict] = []

    for fold in folds:
        tf = build_training_frame(
            fold.train_start, fold.train_end, horizon=horizon, store=store, storage=storage
        )
        if len(tf) < min_train_rows:
            fold_meta.append({"fold": str(fold.test_start), "skipped": True, "rows": len(tf)})
            continue
        tr, va, _ = time_split(tf, valid_frac=0.15, test_frac=0.0)

        sp = {**SignModel().params, **(sign_params or {})}
        qp = {**SpreadQuantileModel().params, **(spread_params or {})}
        sign = SignModel(params=sp, calibration=calibration).fit(tr, va)
        spread = SpreadQuantileModel(params=qp, quantiles=QUANTILES).fit(tr, va)

        for day in _daterange(fold.test_start, fold.test_end):
            ctx = DecisionContext(
                as_of=as_of_for_horizon(day, horizon), delivery_day=day, horizon=horizon
            )
            feats = store.assemble(ctx)
            if feats.empty:
                continue
            X = feature_matrix(feats, tf.feature_names)
            frame = feats[["_area", "_mtu_start"]].copy()
            frame["sign_prob"] = sign.predict_proba(X)
            for q, arr in spread.predict(X).items():
                frame[f"S_p{int(q * 100)}"] = arr
            frame = _attach_load(frame, ctx.as_of, storage)
            rec = recommend_frame(frame, as_of=ctx.as_of, storage=storage)
            rec["_fold"] = str(fold.test_start)
            all_recs.append(rec)

        fold_meta.append(
            {"fold": str(fold.test_start), "skipped": False, "train_rows": len(tf)}
        )

    if not all_recs:
        return {"recs": pd.DataFrame(), "pnl": pd.DataFrame(), "report": {}, "folds": fold_meta}

    recs = pd.concat(all_recs, ignore_index=True)
    actual = build_actuals(recs[["_area", "_mtu_start"]].drop_duplicates(), storage=storage)

    from sfe.backtest.economic import economic_report, realize_pnl

    pnl = realize_pnl(recs, actual)
    return {
        "recs": recs,
        "pnl": pnl,
        "report": economic_report(pnl),
        "folds": fold_meta,
    }

"""Shadow-mode inference pipeline (PRD Phase 4, F-9, F-14, NF-3).

Produces one *forecast bundle* per (delivery day, horizon): calibrated sign probability,
spread and derived-price quantile fans, the risk-limited recommended Delta with its limit
utilisation and drivers, and a ``degraded`` flag when an upstream feature family is stale
or missing (graceful degradation rather than silent failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from sfe.decision.policy import recommend_frame
from sfe.features.spec import DecisionContext, as_of_for_horizon
from sfe.features.store import FeatureStore, default_store
from sfe.models.dataset import build_training_frame, feature_matrix, time_split
from sfe.models.price import derive_price
from sfe.models.registry import load_model, save_model
from sfe.models.sign import SignModel
from sfe.models.spread import QUANTILES, SpreadQuantileModel

_GME_DATASET = "landing/gme/mgp-zonal-prices"
# Feature families whose total absence marks the bundle degraded.
_FAMILY_PREFIXES = {"calendar": "cal_", "residual_load": "rl_", "lag_target": "lag_target_"}


@dataclass
class ForecastBundle:
    delivery_day: date
    horizon: str
    as_of: pd.Timestamp
    model_version: dict
    degraded: bool
    degraded_reasons: list[str]
    table: pd.DataFrame
    drivers: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        return {
            "delivery_day": self.delivery_day.isoformat(),
            "horizon": self.horizon,
            "as_of": self.as_of.isoformat(),
            "model_version": self.model_version,
            "degraded": self.degraded,
            "degraded_reasons": self.degraded_reasons,
            "rows": self.table.assign(
                _mtu_start=self.table["_mtu_start"].astype(str)
            ).to_dict(orient="records"),
            "drivers": self.drivers.to_dict(orient="records"),
        }


def ensure_models(
    horizon: str = "D-1",
    *,
    lookback_days: int = 120,
    end: date | None = None,
    store: FeatureStore | None = None,
    storage=None,
    params: dict | None = None,
    force_train: bool = False,
) -> tuple[SignModel, SpreadQuantileModel]:
    """Load registered models for *horizon*, or train + register them from recent history."""
    if not force_train:
        try:
            return load_model(f"sign_{horizon}"), load_model(f"spread_{horizon}")
        except (FileNotFoundError, OSError):
            pass

    end = end or (date.today() - timedelta(days=1))
    tf = build_training_frame(
        end - timedelta(days=lookback_days), end, horizon=horizon, store=store, storage=storage
    )
    if len(tf) < 100:
        raise RuntimeError(
            f"not enough history to train {horizon!r} models ({len(tf)} rows); land more data"
        )
    tr, va, _ = time_split(tf, valid_frac=0.15, test_frac=0.0)
    sp = {**SignModel().params, **(params or {})}
    qp = {**SpreadQuantileModel().params, **(params or {})}
    sign = SignModel(params=sp).fit(tr, va)
    spread = SpreadQuantileModel(params=qp, quantiles=QUANTILES).fit(tr, va)
    meta = {
        "horizon": horizon,
        "train_start": (end - timedelta(days=lookback_days)).isoformat(),
        "train_end": end.isoformat(),
        "n_rows": len(tf),
        "n_features": len(tf.feature_names),
    }
    save_model(sign, f"sign_{horizon}", meta={**meta, "kind": "sign"})
    save_model(spread, f"spread_{horizon}", meta={**meta, "kind": "spread"})
    return sign, spread


def _p_mkt_estimate(feats: pd.DataFrame, as_of, storage) -> np.ndarray:
    """Reference market price for the price derivation (open question #5).

    Prefers the as-of GME MGP zonal price; falls back to the last observed
    (price - spread) carried in the lag-target features.
    """
    from sfe.curate.vintage import as_of_view, pivot_fields
    from sfe.io.storage import get_storage

    storage = storage or get_storage()
    raw = storage.read_dataset(_GME_DATASET)
    if not raw.empty:
        wide = pivot_fields(as_of_view(raw, as_of))
        if "mgp_price_EURxMWh" in wide.columns:
            wide["_mtu_start"] = pd.to_datetime(wide["_mtu_start"], utc=True)
            lut = wide.set_index(["_area", "_mtu_start"])["mgp_price_EURxMWh"]
            keys = list(
                zip(feats["_area"], pd.to_datetime(feats["_mtu_start"], utc=True), strict=True)
            )
            vals = pd.to_numeric(pd.Series(lut.reindex(keys).to_numpy()), errors="coerce")
            if vals.notna().any():
                return vals.to_numpy()
    last_px = pd.to_numeric(
        feats.get("lag_target_last_unbalance_price_EURxMWh"), errors="coerce"
    )
    last_s = pd.to_numeric(feats.get("lag_target_last_spread_S"), errors="coerce")
    return (last_px - last_s).to_numpy()


def _degradation(feats: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    for name, prefix in _FAMILY_PREFIXES.items():
        cols = [c for c in feats.columns if c.startswith(prefix)]
        if not cols:
            reasons.append(f"{name}: no columns")
            continue
        filled = feats[cols].notna().mean().mean()
        if filled < 0.5:
            reasons.append(f"{name}: {(1 - filled) * 100:.0f}% of feature cells missing")
    return reasons


def generate_bundle(
    delivery_day: date,
    horizon: str = "D-1",
    *,
    sign_model: SignModel | None = None,
    spread_model: SpreadQuantileModel | None = None,
    store: FeatureStore | None = None,
    storage=None,
) -> ForecastBundle:
    store = store or default_store()
    if sign_model is None or spread_model is None:
        sign_model, spread_model = ensure_models(horizon, store=store, storage=storage)

    ctx = DecisionContext(
        as_of=as_of_for_horizon(delivery_day, horizon), delivery_day=delivery_day, horizon=horizon
    )
    feats = store.assemble(ctx)
    version = {
        "sign_n_features": len(getattr(sign_model, "feature_names", [])),
        "spread_n_features": len(getattr(spread_model, "feature_names", [])),
        "spread_quantiles": list(getattr(spread_model, "quantiles", QUANTILES)),
        "horizon": horizon,
    }
    if feats.empty:
        return ForecastBundle(
            delivery_day, horizon, ctx.as_of, version, True, ["no features assembled"],
            pd.DataFrame(),
        )

    reasons = _degradation(feats)
    X = feature_matrix(feats, sign_model.feature_names)

    frame = feats[["_area", "_mtu_start"]].copy()
    frame["sign_prob"] = sign_model.predict_proba(X)
    q = spread_model.predict(X)
    for qq, arr in q.items():
        frame[f"S_p{int(qq * 100)}"] = arr

    p_mkt = _p_mkt_estimate(feats, ctx.as_of, storage)
    price_q = derive_price(p_mkt, q)
    for qq, arr in price_q.items():
        frame[f"Pimb_p{int(qq * 100)}"] = arr

    if "rl_load_forecast_MW" in feats.columns:
        frame["load_mwh"] = pd.to_numeric(feats["rl_load_forecast_MW"], errors="coerce")

    rec = recommend_frame(frame, as_of=ctx.as_of, storage=storage)

    from sfe.decision.limits import Limits

    lim = Limits.from_config()
    load = pd.to_numeric(rec.get("load_mwh"), errors="coerce")
    rec["util_pct_load_cap"] = (rec["delta_mwh"].abs() / load.map(lim.per_mtu_cap)).replace(
        [np.inf, -np.inf], np.nan
    )
    day = rec["_mtu_start"].map(lambda t: pd.Timestamp(t).tz_convert("UTC").date())
    daily_abs = rec.groupby(day)["delta_mwh"].transform(lambda s: s.abs().sum())
    rec["util_daily_cap"] = daily_abs / lim.daily_mwh_cap

    drivers = (
        sign_model.feature_importance()
        .head(15)
        .rename_axis("feature")
        .reset_index(name="importance")
    )
    return ForecastBundle(
        delivery_day, horizon, ctx.as_of, version, bool(reasons), reasons, rec, drivers
    )

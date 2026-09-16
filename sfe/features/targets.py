"""Label assembly: sign, imbalance price, and the signed spread S (PRD G-1..G-3).

``as_of=None`` gives the latest-known vintage (training labels). Passing a decision
timestamp gives the point-in-time view - used when a past imbalance outcome is consumed as
a *lagged feature* (see :mod:`sfe.features.lags`), never as a label.

Reference market price ``p_mkt_ref`` is a placeholder (``base_price_EURxMWh``) pending open
question #5 (MGP zonal vs volume-weighted MGP+MI vs realised procurement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfe.curate.normalize import parse_sign, sign_from_number
from sfe.curate.vintage import as_of_view, latest_view, pivot_fields
from sfe.io.storage import StorageBackend, get_storage

KEY = ["_area", "_mtu_start"]

_FEE_DATASETS = (
    "landing/terna/fees/daily-macrozonal-imbalance",
    "landing/terna/fees/daily-prices",
    "landing/terna/fees/preliminary-macrozonal-imbalance",
    "landing/terna/fees/preliminary-prices",
    "landing/terna/fees/macrozonal-no-arbitrage-prices",
)

P_MKT_REF_FIELD = "base_price_EURxMWh"

OUT_COLS = [
    "_area", "_mtu_start", "imb_sign", "zonal_aggregate_unbalance_MWh",
    "unbalance_price_EURxMWh", "base_price_EURxMWh", "no_arbitrage_price_EURxMWh",
    "p_mkt_ref", "spread_S", "_publication_ts",
]


def _read_fees(storage: StorageBackend) -> pd.DataFrame:
    frames = [storage.read_dataset(d) for d in _FEE_DATASETS]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def assemble_targets(
    as_of: object | None = None, *, storage: StorageBackend | None = None
) -> pd.DataFrame:
    storage = storage or get_storage()
    landing = _read_fees(storage)
    if landing.empty:
        return pd.DataFrame(columns=OUT_COLS)

    view = latest_view(landing) if as_of is None else as_of_view(landing, as_of)
    if view.empty:
        return pd.DataFrame(columns=OUT_COLS)

    # Text sign ("+"/"-") survives only in value_raw; value_num has dropped it.
    txt = view[view["field"] == "zonal_aggregate_sign"].copy()
    txt["s"] = txt["value_raw"].map(parse_sign)
    text_sign = txt.groupby(KEY, dropna=False)["s"].last()

    wide = pivot_fields(view).set_index(KEY)
    sign_from_vol = _col(wide, "zonal_aggregate_unbalance_MWh").map(sign_from_number)
    wide["imb_sign"] = text_sign.reindex(wide.index).fillna(sign_from_vol).astype("Int64")

    wide["p_mkt_ref"] = _col(wide, P_MKT_REF_FIELD)
    wide["spread_S"] = _col(wide, "unbalance_price_EURxMWh") - wide["p_mkt_ref"]

    pub = view.groupby(KEY, dropna=False)["_publication_ts"].max()
    wide["_publication_ts"] = pub.reindex(wide.index)

    wide = wide.reset_index()
    for c in OUT_COLS:
        if c not in wide.columns:
            wide[c] = pd.NA
    return wide[OUT_COLS].sort_values(KEY).reset_index(drop=True)

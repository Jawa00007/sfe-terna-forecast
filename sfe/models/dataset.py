"""Assemble model-ready matrices from the point-in-time feature store (PRD F-6..F-8).

One row per ``(_area, _mtu_start)`` decision at a fixed horizon. Features come only from
:mod:`sfe.features` builders (all PIT-correct); labels are attached from the latest-known
target view. Splitting is strictly chronological - never shuffle a time series (B-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from sfe.features.spec import DecisionContext, as_of_for_horizon
from sfe.features.store import FeatureStore, attach_labels, default_store

META_COLS = ["_area", "_mtu_start", "_as_of", "_horizon"]
LABEL_COLS = ["label_sign", "label_spread_S", "label_unbalance_price_EURxMWh"]


@dataclass
class TrainingFrame:
    X: pd.DataFrame
    y_sign_pos: pd.Series          # 1 = area long (sign > 0), 0 otherwise
    y_spread: pd.Series            # signed spread S
    y_price: pd.Series             # absolute imbalance price
    meta: pd.DataFrame
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)

    def loc(self, mask: pd.Series) -> TrainingFrame:
        return TrainingFrame(
            self.X.loc[mask].reset_index(drop=True),
            self.y_sign_pos.loc[mask].reset_index(drop=True),
            self.y_spread.loc[mask].reset_index(drop=True),
            self.y_price.loc[mask].reset_index(drop=True),
            self.meta.loc[mask].reset_index(drop=True),
            self.feature_names,
        )


def _daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def feature_matrix(df: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    """All-numeric feature matrix. ``_area`` -> deterministic ``area_code``; unknown columns
    in *feature_names* are added as NaN so train- and predict-time frames line up."""
    out = pd.DataFrame(index=df.index)
    if "_area" in df.columns:
        cats = sorted(map(str, df["_area"].dropna().unique()))
        out["area_code"] = pd.Categorical(df["_area"].astype(str), categories=cats).codes
    src = [c for c in df.columns if c not in META_COLS + LABEL_COLS and not c.startswith("_")]
    for c in src:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    if feature_names is not None:
        for c in feature_names:
            if c not in out.columns:
                out[c] = np.nan
        out = out[feature_names]
    return out.astype("float64")


def build_training_frame(
    date_from: date,
    date_to: date,
    *,
    horizon: str = "D-1",
    store: FeatureStore | None = None,
    storage=None,
) -> TrainingFrame:
    store = store or default_store()
    contexts = [
        DecisionContext(as_of=as_of_for_horizon(d, horizon), delivery_day=d, horizon=horizon)
        for d in _daterange(date_from, date_to)
    ]
    feats = store.assemble_many(contexts)
    df = attach_labels(feats, storage=storage)

    # need at least one usable label
    df = df.dropna(subset=["label_sign", "label_spread_S"], how="all").reset_index(drop=True)

    raw_names = [
        c
        for c in df.columns
        if c not in META_COLS + LABEL_COLS and not c.startswith("_")
    ]
    X = feature_matrix(df, ["area_code", *raw_names])
    feature_names = list(X.columns)

    y_sign_pos = (pd.to_numeric(df["label_sign"], errors="coerce") > 0).astype("float64")
    y_sign_pos[pd.to_numeric(df["label_sign"], errors="coerce").isna()] = np.nan

    return TrainingFrame(
        X=X,
        y_sign_pos=y_sign_pos,
        y_spread=pd.to_numeric(df["label_spread_S"], errors="coerce"),
        y_price=pd.to_numeric(df["label_unbalance_price_EURxMWh"], errors="coerce"),
        meta=df[META_COLS].copy(),
        feature_names=feature_names,
    )


def time_split(
    tf: TrainingFrame, *, valid_frac: float = 0.2, test_frac: float = 0.2
) -> tuple[TrainingFrame, TrainingFrame, TrainingFrame]:
    """Chronological train / valid / test by ``_mtu_start`` (no shuffling - B-1)."""
    order = np.argsort(tf.meta["_mtu_start"].values, kind="mergesort")
    n = len(order)
    n_test = int(n * test_frac)
    n_valid = int(n * valid_frac)
    cuts = {
        "train": order[: n - n_valid - n_test],
        "valid": order[n - n_valid - n_test : n - n_test],
        "test": order[n - n_test :],
    }

    def _take(pos) -> TrainingFrame:
        mask = pd.Series(False, index=tf.X.index)
        mask.iloc[pos] = True
        return tf.loc(mask)

    return _take(cuts["train"]), _take(cuts["valid"]), _take(cuts["test"])

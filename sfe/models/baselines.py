"""Naive baselines the models must beat (PRD §8, B-6).

- **persistence**: same MTU, previous day (realised value).
- **seasonal climatology**: mean outcome by (area, month, local hour) on the training split.

Persistence here uses the realised previous-day outcome; a stricter as-of variant (only
what was published by the decision time) is a TODO for the economic backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfe.curate.timebase import utc_to_rome
from sfe.models.dataset import TrainingFrame


def _local_parts(mtu: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(mtu, utc=True)
    loc = ts.map(lambda t: utc_to_rome(t.to_pydatetime()))
    return pd.DataFrame(
        {"month": [d.month for d in loc], "hour": [d.hour for d in loc]}, index=mtu.index
    )


def persistence(reference: TrainingFrame, eval_tf: TrainingFrame) -> dict[str, np.ndarray]:
    ref = reference.meta.copy()
    ref["sign_pos"] = reference.y_sign_pos.to_numpy()
    ref["spread"] = reference.y_spread.to_numpy()
    ref = ref.set_index(["_area", "_mtu_start"])

    prev_key = list(
        zip(
            eval_tf.meta["_area"],
            pd.to_datetime(eval_tf.meta["_mtu_start"], utc=True) - pd.Timedelta(days=1),
            strict=True,
        )
    )
    sign = ref["sign_pos"].reindex(prev_key).to_numpy()
    spread = ref["spread"].reindex(prev_key).to_numpy()

    sign = np.where(np.isfinite(sign), sign, np.nanmean(reference.y_sign_pos.to_numpy()))
    spread = np.where(np.isfinite(spread), spread, np.nanmean(reference.y_spread.to_numpy()))
    return {"p_sign_pos": np.clip(sign, 0.0, 1.0), "spread": spread}


def seasonal_climatology(train: TrainingFrame, eval_tf: TrainingFrame) -> dict[str, np.ndarray]:
    tr = train.meta[["_area"]].copy()
    tr = pd.concat([tr, _local_parts(train.meta["_mtu_start"])], axis=1)
    tr["sign_pos"] = train.y_sign_pos.to_numpy()
    tr["spread"] = train.y_spread.to_numpy()
    grp = tr.groupby(["_area", "month", "hour"], dropna=False)[["sign_pos", "spread"]].mean()

    ev = eval_tf.meta[["_area"]].copy()
    ev = pd.concat([ev, _local_parts(eval_tf.meta["_mtu_start"])], axis=1)
    keyed = list(zip(ev["_area"], ev["month"], ev["hour"], strict=True))
    sign = grp["sign_pos"].reindex(keyed).to_numpy()
    spread = grp["spread"].reindex(keyed).to_numpy()

    sign = np.where(np.isfinite(sign), sign, np.nanmean(train.y_sign_pos.to_numpy()))
    spread = np.where(np.isfinite(spread), spread, np.nanmean(train.y_spread.to_numpy()))
    return {"p_sign_pos": np.clip(sign, 0.0, 1.0), "spread": spread}

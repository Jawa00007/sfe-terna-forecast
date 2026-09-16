"""Price model (PRD F-8): derived, not modelled independently.

``P_imb_forecast = P_mkt_forecast + S_forecast`` so the two legs stay coherent. The
absolute imbalance price is reported for interpretability only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_price(
    p_mkt: np.ndarray | pd.Series, spread_preds_by_q: dict[float, np.ndarray]
) -> dict[float, np.ndarray]:
    p_mkt = np.asarray(p_mkt, dtype="float64")
    return {q: p_mkt + np.asarray(s, dtype="float64") for q, s in spread_preds_by_q.items()}


def derive_price_frame(p_mkt: np.ndarray | pd.Series, spread_frame: pd.DataFrame) -> pd.DataFrame:
    p_mkt = np.asarray(p_mkt, dtype="float64")
    out = {}
    for col in spread_frame.columns:
        out[col.replace("S_p", "Pimb_p")] = p_mkt + spread_frame[col].to_numpy()
    return pd.DataFrame(out)

"""Spread model (PRD F-7): quantile regression on S = P_imb - P_mkt.

One LightGBM regressor per quantile with the pinball objective. Predictions are sorted
across quantiles at inference so the fan never crosses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sfe.models.dataset import TrainingFrame

QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

_DEFAULT_PARAMS = {
    "objective": "quantile",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass
class SpreadQuantileModel:
    quantiles: tuple[float, ...] = QUANTILES
    params: dict = field(default_factory=lambda: dict(_DEFAULT_PARAMS))
    random_state: int = 0

    _boosters: dict = field(default_factory=dict, repr=False)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, train: TrainingFrame, valid: TrainingFrame | None = None) -> SpreadQuantileModel:
        import lightgbm as lgb

        m = train.y_spread.notna()
        X, y = train.X[m], train.y_spread[m]
        self.feature_names = list(train.feature_names)

        eval_kw: dict = {}
        if valid is not None:
            vm = valid.y_spread.notna()
            if vm.any():
                eval_kw = {
                    "eval_X": valid.X[vm],
                    "eval_y": valid.y_spread[vm],
                    "callbacks": [lgb.early_stopping(50, verbose=False)],
                }

        self._boosters = {}
        for q in self.quantiles:
            p = dict(self.params, alpha=q)
            gbm = lgb.LGBMRegressor(random_state=self.random_state, **p)
            gbm.fit(X, y, **eval_kw)
            self._boosters[q] = gbm
        return self

    def predict(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        cols = X[self.feature_names]
        raw = np.column_stack([self._boosters[q].predict(cols) for q in self.quantiles])
        raw.sort(axis=1)  # enforce non-crossing quantiles
        return {q: raw[:, i] for i, q in enumerate(self.quantiles)}

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = self.predict(X)
        return pd.DataFrame({f"S_p{int(q * 100)}": v for q, v in preds.items()})

    def feature_importance(self) -> pd.Series:
        if not self._boosters:
            return pd.Series(dtype="float64")
        imp = np.mean([b.feature_importances_ for b in self._boosters.values()], axis=0)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

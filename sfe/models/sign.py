"""Sign model (PRD F-6): gradient-boosted binary classifier + probability calibration.

Target: ``P(area long)`` = ``P(sign > 0)``, per area per MTU. Baseline learner is LightGBM;
calibration is isotonic (default) or Platt, fitted on a held-out slice and validated with a
reliability diagram (see :func:`sfe.models.metrics.reliability_table`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sfe.models.dataset import TrainingFrame

_DEFAULT_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
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
class SignModel:
    params: dict = field(default_factory=lambda: dict(_DEFAULT_PARAMS))
    calibration: str = "isotonic"          # "isotonic" | "platt" | "none"
    random_state: int = 0

    _booster: object = field(default=None, repr=False)
    _calibrator: object = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, train: TrainingFrame, valid: TrainingFrame | None = None) -> SignModel:
        import lightgbm as lgb

        m = train.y_sign_pos.notna()
        X, y = train.X[m], train.y_sign_pos[m].astype(int)
        self.feature_names = list(train.feature_names)

        self._booster = lgb.LGBMClassifier(random_state=self.random_state, **self.params)
        fit_kw = {}
        if valid is not None:
            vm = valid.y_sign_pos.notna()
            if vm.any():
                fit_kw["eval_X"] = valid.X[vm]
                fit_kw["eval_y"] = valid.y_sign_pos[vm].astype(int)
                fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        self._booster.fit(X, y, **fit_kw)

        self._fit_calibrator(valid if valid is not None else train)
        return self

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._booster.predict_proba(X[self.feature_names])[:, 1]

    def _fit_calibrator(self, cal: TrainingFrame) -> None:
        self._calibrator = None
        if self.calibration == "none":
            return
        m = cal.y_sign_pos.notna()
        if m.sum() < 50:
            return
        p = self._raw_proba(cal.X[m])
        y = cal.y_sign_pos[m].to_numpy()
        if self.calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p, y)
            self._calibrator = iso
        elif self.calibration == "platt":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression()
            lr.fit(p.reshape(-1, 1), y)
            self._calibrator = lr
        else:
            raise ValueError(f"unknown calibration {self.calibration!r}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._raw_proba(X)
        if self._calibrator is None:
            return p
        if hasattr(self._calibrator, "predict_proba"):
            return self._calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(self._calibrator.predict(p), 0.0, 1.0)

    def predict_sign(self, X: pd.DataFrame, thr: float = 0.5) -> np.ndarray:
        return np.where(self.predict_proba(X) >= thr, 1, -1)

    def feature_importance(self) -> pd.Series:
        if self._booster is None:
            return pd.Series(dtype="float64")
        return pd.Series(
            self._booster.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

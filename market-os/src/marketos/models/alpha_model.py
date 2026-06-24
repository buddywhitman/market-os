"""Alpha model — gradient-boosted trees, walk-forward trained, SHAP-explained.

We predict *distributions/probabilities*, never bare point forecasts dressed as truth.
Targets are things like P(large up move), 5d/20d return, forward volatility. Weights are
never hardcoded permanently — they are re-fit each walk-forward fold and audited with
SHAP. If a feature does not improve out-of-sample expectancy, it gets cut.

Falls back to sklearn's HistGradientBoosting if xgboost isn't installed, so the contract
holds on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marketos.backtest.walkforward import walk_forward_splits


@dataclass
class AlphaModel:
    feature_cols: list[str]
    target_col: str
    label_horizon: int = 5
    classification: bool = True   # P(move) by default
    params: dict = field(default_factory=dict)
    _model: object = None

    def _new_estimator(self):
        try:
            import xgboost as xgb

            common = dict(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=-1, **self.params,
            )
            return xgb.XGBClassifier(**common) if self.classification else xgb.XGBRegressor(**common)
        except Exception:
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
            )

            return (HistGradientBoostingClassifier if self.classification
                    else HistGradientBoostingRegressor)(
                max_depth=4, learning_rate=0.03, max_iter=300)

    def walk_forward_predict(
        self, df: pd.DataFrame, *, ts_col: str = "asof_ts",
        train_periods: int = 504, test_periods: int = 63,
    ) -> pd.DataFrame:
        """Generate strictly out-of-sample predictions via purged walk-forward.

        Returns the input rows that received an OOS prediction, with a `pred` column.
        This is the ONLY prediction series we allow into a backtest.
        """
        df = df.sort_values(ts_col).reset_index(drop=True)
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        preds = np.full(len(df), np.nan)

        for split in walk_forward_splits(
            df[ts_col], train_periods=train_periods, test_periods=test_periods,
            label_horizon=self.label_horizon,
        ):
            est = self._new_estimator()
            tr, te = split.train_idx, split.test_idx
            mask = ~np.isnan(y[tr])
            if mask.sum() < 30:
                continue
            est.fit(X[tr][mask], y[tr][mask])
            if self.classification and hasattr(est, "predict_proba"):
                preds[te] = est.predict_proba(X[te])[:, 1]
            else:
                preds[te] = est.predict(X[te])
            self._model = est

        out = df.copy()
        out["pred"] = preds
        return out.dropna(subset=["pred"])

    def shap_attributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP values for the last fitted model. No black boxes."""
        if self._model is None:
            raise RuntimeError("fit via walk_forward_predict first")
        try:
            import shap

            explainer = shap.TreeExplainer(self._model)
            vals = explainer.shap_values(df[self.feature_cols].values)
            vals = vals[1] if isinstance(vals, list) else vals
            return pd.DataFrame(vals, columns=self.feature_cols, index=df.index)
        except Exception:
            # Fallback: model feature_importances_ broadcast (still explainable, coarser).
            imp = getattr(self._model, "feature_importances_", None)
            if imp is None:
                raise
            return pd.DataFrame([imp], columns=self.feature_cols)

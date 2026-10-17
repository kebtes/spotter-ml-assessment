"""
Model definitions, training routines, and ensembling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.base import BaseEstimator, RegressorMixin
from src.config import CATEGORICAL_FEATURES, RANDOM_STATE


class DomainBaselineModel(BaseEstimator, RegressorMixin):
    """
    Domain-informed heuristic baseline.
    Estimates rate as distance * quote_signal, adjusted by equipment-specific median ratios.
    """

    def __init__(self):
        self.equipment_ratios_: dict[str, float] = {}
        self.global_ratio_: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DomainBaselineModel:
        base = X["distance"] * X["quote_signal"]
        ratios = y / np.clip(base, 1.0, None)
        self.global_ratio_ = float(ratios.median())

        df_temp = pd.DataFrame({"equipment": X["equipment"], "ratio": ratios})
        self.equipment_ratios_ = df_temp.groupby("equipment")["ratio"].median().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = (X["distance"] * X["quote_signal"]).values
        eq = X["equipment"].astype(str)
        multiplier = eq.map(self.equipment_ratios_).fillna(self.global_ratio_).values
        preds = base * multiplier
        return np.clip(preds, a_min=10.0, a_max=None)


class LightGBMRateModel:
    """LightGBM gradient boosting model configured for freight rate regression."""

    def __init__(
        self,
        n_estimators: int = 1200,
        learning_rate: float = 0.03,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 30,
        random_state: int = RANDOM_STATE,
    ):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            objective="regression_l1",  # Optimizes MAE directly
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMRateModel:
        X_clean = X.copy()
        for col in CATEGORICAL_FEATURES:
            X_clean[col] = X_clean[col].astype("category")
        self.model.fit(X_clean, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.copy()
        for col in CATEGORICAL_FEATURES:
            X_clean[col] = X_clean[col].astype("category")
        preds = self.model.predict(X_clean)
        return np.clip(preds, a_min=10.0, a_max=None)

    def get_feature_importances(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)


class CatBoostRateModel:
    """CatBoost gradient boosting model with native categorical feature support."""

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = RANDOM_STATE,
    ):
        self.model = cb.CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=random_seed,
            verbose=0,
            thread_count=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CatBoostRateModel:
        X_clean = X.copy()
        for col in CATEGORICAL_FEATURES:
            X_clean[col] = X_clean[col].astype(str)
        self.model.fit(X_clean, y, cat_features=CATEGORICAL_FEATURES)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.copy()
        for col in CATEGORICAL_FEATURES:
            X_clean[col] = X_clean[col].astype(str)
        preds = self.model.predict(X_clean)
        return np.clip(preds, a_min=10.0, a_max=None)


class FreightRateEnsemble:
    """Weighted ensemble combining LightGBM and CatBoost models."""

    def __init__(self, lgb_weight: float = 0.55, cb_weight: float = 0.45):
        self.lgb_weight = lgb_weight
        self.cb_weight = cb_weight
        self.lgb_model = LightGBMRateModel()
        self.cb_model = CatBoostRateModel()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FreightRateEnsemble:
        self.lgb_model.fit(X, y)
        self.cb_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lgb_preds = self.lgb_model.predict(X)
        cb_preds = self.cb_model.predict(X)
        blended = self.lgb_weight * lgb_preds + self.cb_weight * cb_preds
        return np.clip(blended, a_min=10.0, a_max=None)

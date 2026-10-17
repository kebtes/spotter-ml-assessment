"""
Evaluation metrics and validation reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    Formula: sum(|y_true - y_pred|) / sum(y_true) * 100%
    Avoids zero/small divisor distortion typical of MAPE in rate forecasting.
    """
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-9) * 100.0)


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate comprehensive suite of freight rate prediction metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Post-processing safeguard: rates must be positive
    y_p = np.clip(y_p, a_min=1.0, a_max=None)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    med_ae = float(median_absolute_error(y_t, y_p))
    wape = calculate_wape(y_t, y_p)

    # Clean MAPE (filtered of near-zeros if any)
    pct_err = np.abs((y_t - y_p) / np.clip(y_t, 1.0, None))
    mape = float(np.mean(pct_err) * 100.0)

    return {
        "MAE ($)": round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "Median AE ($)": round(med_ae, 2),
        "WAPE (%)": round(wape, 2),
        "MAPE (%)": round(mape, 2),
    }


def format_metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Format multiple model evaluation results into a clean comparison DataFrame."""
    df = pd.DataFrame.from_dict(results, orient="index")
    return df

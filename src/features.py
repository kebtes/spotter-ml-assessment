"""
Feature engineering transformations for freight rate prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.config import (
    EARTH_RADIUS_MILES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES,
)


def calculate_haversine_distance(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Calculate the Great Circle distance (in miles) between two coordinate points."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_MILES * c


def engineer_features(
    df_raw: pd.DataFrame,
    city_coords: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Transforms raw load attributes into an engineered feature matrix.
    Handles geospatial, physical cargo, calendar seasonality, and market pricing signals.
    """
    df = df_raw.copy()

    # 1. Geospatial & Coordinate Imputation
    if "pickup_lat" not in df.columns or df["pickup_lat"].isna().any():
        if city_coords is not None:
            df["pickup_lat"] = df["pickup_lat"].fillna(df["pickup"].map(city_coords["lat"]))
            df["pickup_lon"] = df["pickup_lon"].fillna(df["pickup"].map(city_coords["lon"]))
            df["delivery_lat"] = df["delivery_lat"].fillna(df["delivery"].map(city_coords["lat"]))
            df["delivery_lon"] = df["delivery_lon"].fillna(df["delivery"].map(city_coords["lon"]))

    df["haversine"] = calculate_haversine_distance(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    df["circuity"] = df["distance"] / (df["haversine"] + 1.0)
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]

    # 2. Cargo & Weight Profile
    equipment_weight_defaults = {
        "Dry Van": 31_000.0,
        "Reefer": 32_000.0,
        "Flatbed": 30_000.0,
    }
    df["weight_clean"] = (
        df["weight"]
        .fillna(df["equipment"].map(equipment_weight_defaults))
        .fillna(31_000.0)
    )
    df["weight_per_mile"] = df["weight_clean"] / (df["distance"] + 1.0)
    df["is_heavy"] = (df["weight_clean"] > 40_000.0).astype(int)

    # 3. Calendar & Seasonality Signals
    dt = pd.to_datetime(df["date"])
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["days_to_month_end"] = dt.dt.days_in_month - df["day"]

    # Proximity to major freight surge holidays (2025)
    thanksgiving = pd.to_datetime("2025-11-27")
    christmas = pd.to_datetime("2025-12-25")

    df["days_to_thanksgiving"] = (thanksgiving - dt).dt.days
    df["days_to_christmas"] = (christmas - dt).dt.days

    # Distinct holiday behavior windows
    df["near_christmas"] = (
        (df["days_to_christmas"] >= -2) & (df["days_to_christmas"] <= 10)
    ).astype(int)
    df["pre_christmas_rush"] = (
        (df["days_to_christmas"] > 2) & (df["days_to_christmas"] <= 14)
    ).astype(int)
    df["post_christmas"] = (
        (df["days_to_christmas"] <= 0) & (df["days_to_christmas"] >= -6)
    ).astype(int)

    # 4. Market & Pricing Interactions
    if "quote_signal" in df.columns:
        df["base_cost_estimate"] = df["distance"] * df["quote_signal"]
        df["mkt_index_clean"] = df["market_index"].fillna(1.0)
        df["mkt_x_quote"] = df["mkt_index_clean"] * df["quote_signal"]
        df["quote_per_mkt"] = df["quote_signal"] / (df["mkt_index_clean"] + 1e-4)
    else:
        # Fallback values if signals not provided
        df["base_cost_estimate"] = df["distance"] * 2.05
        df["mkt_index_clean"] = 1.0
        df["mkt_x_quote"] = 2.05
        df["quote_per_mkt"] = 2.05
        df["quote_signal"] = 2.05

    # 5. Lane representation
    df["lane"] = df["pickup"].astype(str) + " -> " + df["delivery"].astype(str)

    return df


def prepare_feature_matrices(
    df_engineered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Extracts features and target into standardized format.
    Ensures categorical columns are properly cast.
    """
    df = df_engineered.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    X = df[ALL_MODEL_FEATURES].copy()
    y = df["posted_rate"].copy() if "posted_rate" in df.columns else None

    return X, y

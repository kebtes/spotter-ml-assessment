"""
Data loading, validation, and preprocessing pipeline.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    DECEMBER_PATH,
    LEXINGTON_COORDS,
    FORT_WAYNE_COORDS,
)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and december input datasets."""
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_dec = pd.read_csv(DECEMBER_PATH)
    return df_train, df_val, df_dec


def extract_city_coordinates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Extract coordinate lookup maps for cities from pickup/delivery locations.
    """
    pickups = df.groupby("pickup")[["pickup_lat", "pickup_lon"]].mean()
    deliveries = df.groupby("delivery")[["delivery_lat", "delivery_lon"]].mean()

    lat_map: dict[str, float] = {}
    lon_map: dict[str, float] = {}

    for city, row in pickups.iterrows():
        lat_map[str(city)] = float(row["pickup_lat"])
        lon_map[str(city)] = float(row["pickup_lon"])

    for city, row in deliveries.iterrows():
        if str(city) not in lat_map:
            lat_map[str(city)] = float(row["delivery_lat"])
            lon_map[str(city)] = float(row["delivery_lon"])

    # Ensure fixed cities are present
    lat_map["Lexington"] = LEXINGTON_COORDS[0]
    lon_map["Lexington"] = LEXINGTON_COORDS[1]
    lat_map["Fort Wayne"] = FORT_WAYNE_COORDS[0]
    lon_map["Fort Wayne"] = FORT_WAYNE_COORDS[1]

    return {"lat": lat_map, "lon": lon_map}


def get_daily_december_signals(df_val: pd.DataFrame) -> pd.DataFrame:
    """
    Extract daily average market_index and quote_signal for December from validation data.
    """
    dec_val = df_val[df_val["date"].str.startswith("2025-12")].copy()
    daily_mkt = dec_val.groupby("date")["market_index"].mean()
    # For Dry Van specifically or general loads
    daily_dryvan_quote = (
        dec_val[dec_val["equipment"] == "Dry Van"]
        .groupby("date")["quote_signal"]
        .mean()
    )
    # Fallback to general quote if needed
    daily_gen_quote = dec_val.groupby("date")["quote_signal"].mean()
    daily_quote = daily_dryvan_quote.combine_first(daily_gen_quote)

    signals = pd.DataFrame({
        "date": daily_mkt.index,
        "market_index": daily_mkt.values,
        "quote_signal": daily_quote.values,
    })
    return signals


def prepare_december_inputs(df_dec: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich december-chart-inputs with required geographic and signal columns.
    """
    dec = df_dec.copy()
    dec["pickup_lat"] = LEXINGTON_COORDS[0]
    dec["pickup_lon"] = LEXINGTON_COORDS[1]
    dec["delivery_lat"] = FORT_WAYNE_COORDS[0]
    dec["delivery_lon"] = FORT_WAYNE_COORDS[1]

    dec_signals = get_daily_december_signals(df_val)
    dec = dec.merge(dec_signals, on="date", how="left")

    # If any remaining NaNs in signals, fill with month average
    dec["market_index"] = dec["market_index"].fillna(dec["market_index"].mean())
    dec["quote_signal"] = dec["quote_signal"].fillna(dec["quote_signal"].mean())

    return dec


def get_time_based_split(
    df: pd.DataFrame,
    split_date: str = "2025-09-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict out-of-time (OOT) train/validation split.
    Simulates predicting the last 2 months from the preceding 8 months.
    """
    train_split = df[df["date"] < split_date].copy()
    val_split = df[df["date"] >= split_date].copy()
    return train_split, val_split

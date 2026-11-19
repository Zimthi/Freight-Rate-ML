"""Cleaning + feature engineering for the freight rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0
EQUIP = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def daily_market(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date, pooled over every frame that has the column.

    market_index is a date-level signal plus ~0.025 per-load noise, so the daily mean is a
    good imputation for missing values and a good stand-in for the (missing) December inputs.
    Only features are used here, never labels.
    """
    allf = pd.concat([f[["date", "market_index"]].assign(date=lambda x: pd.to_datetime(x["date"])) for f in frames])
    return allf.groupby("date").market_index.mean()


def clean(df: pd.DataFrame, market_by_date: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    # negative weights are sign errors (|w| matches the positive distribution)
    d["weight_was_negative"] = (d["weight"] < 0).astype(int)
    d["weight_missing"] = d["weight"].isna().astype(int)
    d["weight"] = d["weight"].abs().clip(upper=WEIGHT_CAP)
    d["weight"] = d["weight"].fillna(d.groupby("equipment")["weight"].transform("median"))
    d["weight"] = d["weight"].fillna(31_000.0)
    # missing market_index -> daily mean for that date
    if "market_index" not in d:
        d["market_index"] = np.nan
    d["market_index_missing"] = d["market_index"].isna().astype(int)
    d["market_index"] = d["market_index"].fillna(d["date"].map(market_by_date))
    return d


def build_features(d: pd.DataFrame, use_quote_signal: bool = True) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["distance"] = d["distance"]
    X["log_distance"] = np.log(d["distance"])
    X["equipment"] = d["equipment"].map(EQUIP)
    X["weight"] = d["weight"]
    X["weight_missing"] = d["weight_missing"]
    X["market_index"] = d["market_index"]
    if use_quote_signal:
        X["quote_signal"] = d["quote_signal"]
    # geography (city names are NOT used: validation contains 8 unseen cities)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = d[c]
    hv = haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    X["circuity"] = d["distance"] / hv.clip(lower=1)
    X["d_lat"] = d.delivery_lat - d.pickup_lat
    X["d_lon"] = d.delivery_lon - d.pickup_lon
    # calendar
    X["dow"] = d["date"].dt.dayofweek
    X["dom"] = d["date"].dt.day
    X["doy"] = d["date"].dt.dayofyear
    return X

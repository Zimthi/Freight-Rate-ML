"""Freight rate model: validation, training, and prediction.

Usage:  python train.py            (reads ./data, writes validation_predictions.csv, december_predictions.csv, metrics.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import build_features, clean, daily_market  # noqa: E402

DATA = Path("data")
SEEDS = [0, 1, 2, 3, 4]
DROP = ["quote_signal"]  # hurts forward-chaining CV (see report); also absent for the December chart


def make_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",  # median regression on log(rate): robust to the ~0.7% rate spikes
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=[0], random_state=seed,
    )


def fit_predict(Xtr, ytr, Xte, seeds=SEEDS):
    cols = ["equipment"] + [c for c in Xtr.columns if c != "equipment"]  # categorical first
    preds = []
    for s in seeds:
        m = make_model(s).fit(Xtr[cols], np.log(ytr))
        preds.append(m.predict(Xte[cols]))
    return np.exp(np.mean(preds, axis=0))


def metrics(y, p):
    e = p - y
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "MAPE_%": float((np.abs(e) / y).mean() * 100), "MedAPE_%": float(np.median(np.abs(e) / y) * 100)}


def main():
    t = pd.read_csv(DATA / "train-test.csv")
    v = pd.read_csv(DATA / "validation.csv")
    dec = pd.read_csv(DATA / "december-chart-inputs.csv")
    market = daily_market(t, v)  # features only, no labels
    T, V = clean(t, market), clean(v, market)
    X = build_features(T).drop(columns=DROP)
    y = t["posted_rate"].values
    dates = T["date"]
    out = {}

    # 1) forward-chaining (expanding window) CV: always train on the past, test on the next month
    rows = []
    for a, b in [("2025-06-01", "2025-07-01"), ("2025-07-01", "2025-08-01"), ("2025-08-01", "2025-09-01"),
                 ("2025-09-01", "2025-10-01"), ("2025-10-01", "2025-11-01")]:
        tr, te = (dates < a).values, ((dates >= a) & (dates < b)).values
        rows.append({"test_month": a[:7], **metrics(y[te], fit_predict(X[tr], y[tr], X[te], seeds=[0]))})
    cv = pd.DataFrame(rows)
    print(cv.round(2).to_string(index=False))
    out["forward_cv_monthly"] = rows
    out["forward_cv_mean"] = cv.drop(columns="test_month").mean().to_dict()

    # 2) unseen-city check: validation contains 8 cities not in train -> hold out cities in time-CV test month
    rng = np.random.default_rng(0)
    cities = np.array(sorted(set(t.pickup) | set(t.delivery)))
    hold = set(rng.choice(cities, 8, replace=False))
    touches = t.pickup.isin(hold) | t.delivery.isin(hold)
    tr = ((dates < "2025-09-01") & ~touches).values
    te = ((dates >= "2025-09-01") & touches).values
    out["unseen_city_holdout"] = metrics(y[te], fit_predict(X[tr], y[tr], X[te], seeds=[0]))
    seen_te = ((dates >= "2025-09-01") & ~touches).values
    out["seen_city_same_period"] = metrics(y[seen_te], fit_predict(X[tr], y[tr], X[seen_te], seeds=[0]))
    print("unseen cities:", {k: round(x, 2) for k, x in out["unseen_city_holdout"].items()})
    print("seen cities  :", {k: round(x, 2) for k, x in out["seen_city_same_period"].items()})

    # 3) final fit on all labeled data -> validation predictions
    XV = build_features(V).drop(columns=DROP)
    pv = fit_predict(X, y, XV)
    sub = pd.read_csv(DATA / "validation-predictions-template.csv")[["load_id"]]
    assert (sub.load_id.values == v.load_id.values).all()
    sub["predicted_rate"] = np.round(pv, 2)
    sub.to_csv("validation_predictions.csv", index=False)

    # 4) fixed December lane: only the date changes. market_index = daily mean from validation.csv features.
    D = dec.copy()
    D["load_id"] = "DEC"
    D["pickup_lat"], D["pickup_lon"] = t.query("pickup=='Lexington'")[["pickup_lat", "pickup_lon"]].iloc[0]
    D["delivery_lat"], D["delivery_lon"] = t.query("delivery=='Fort Wayne'")[["delivery_lat", "delivery_lon"]].iloc[0]
    D["market_index"] = np.nan
    D["quote_signal"] = np.nan
    DC = clean(D, market)
    pd_ = fit_predict(X, y, build_features(DC).drop(columns=DROP))
    dec["predicted_rate"] = np.round(pd_, 2)
    dec.to_csv("december_predictions.csv", index=False)
    Path("metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

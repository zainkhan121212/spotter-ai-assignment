"""Freight rate prediction pipeline.

Steps:
1. Clean the development data (fix sign-flipped weights, drop corrupted
   rate outliers, impute missing market_index / weight).
2. Engineer features (calendar, geography, lane/city statistics).
3. Validate with a time-based split (train Jan-Aug, holdout Sep-Oct) that
   mimics the real task of predicting Nov-Dec.
4. Retrain on all cleaned data and write:
   - validation_predictions.csv          (12,000 loads, Nov-Dec)
   - data/december_chart_inputs.csv      (predicted_rate column filled)

Run:  python scripts/train_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA = Path("data")
RNG = 42

# thresholds from the EDA: normal loads sit between ~$1.7 and ~$3.2 per mile,
# rows far outside that band are corrupted labels
RPM_LOW, RPM_HIGH = 0.8, 6.0


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
def clean(frame: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = frame.copy()

    # 292 training rows carry negative weights whose absolute values look like
    # ordinary weights -> sign-flip entry errors
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(out["weight"].median())

    # market_index is a market-level daily series; fill gaps with the mean of
    # the same day (falls back to overall median)
    daily = out.groupby("date")["market_index"].transform("mean")
    out["market_index"] = out["market_index"].fillna(daily)
    out["market_index"] = out["market_index"].fillna(out["market_index"].median())

    if is_train:
        rpm = out["posted_rate"] / out["distance"]
        keep = (rpm >= RPM_LOW) & (rpm <= RPM_HIGH)
        dropped = (~keep).sum()
        print(f"dropped {dropped} rows with corrupted posted_rate "
              f"({dropped / len(out):.2%} of training data)")
        out = out.loc[keep].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# feature engineering
# --------------------------------------------------------------------------
def lane_city_stats(train: pd.DataFrame) -> dict:
    """Rate-per-mile statistics learned from the training rows only."""
    rpm = train["posted_rate"] / train["distance"]
    base = train.assign(rpm=rpm)
    return {
        "global_rpm": rpm.mean(),
        "lane_rpm": base.groupby(["pickup", "delivery"])["rpm"].mean(),
        "pickup_rpm": base.groupby("pickup")["rpm"].mean(),
        "delivery_rpm": base.groupby("delivery")["rpm"].mean(),
        "equip_rpm": base.groupby("equipment")["rpm"].mean(),
    }


def build_features(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=frame.index)
    date = frame["date"]

    f["distance"] = frame["distance"]
    f["log_distance"] = np.log1p(frame["distance"])
    f["weight"] = frame["weight"]
    f["market_index"] = frame["market_index"]
    f["quote_signal"] = frame["quote_signal"]

    f["month"] = date.dt.month
    f["day_of_week"] = date.dt.dayofweek
    doy = date.dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["is_weekend"] = (date.dt.dayofweek >= 5).astype(int)

    f["pickup_lat"] = frame["pickup_lat"]
    f["pickup_lon"] = frame["pickup_lon"]
    f["delivery_lat"] = frame["delivery_lat"]
    f["delivery_lon"] = frame["delivery_lon"]

    f["equipment"] = frame["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2})

    # learned rate-per-mile levels; unseen lanes/cities fall back gracefully
    g = stats["global_rpm"]
    lane_keys = pd.MultiIndex.from_frame(frame[["pickup", "delivery"]])
    f["lane_rpm"] = stats["lane_rpm"].reindex(lane_keys).to_numpy()
    f["pickup_rpm"] = frame["pickup"].map(stats["pickup_rpm"])
    f["delivery_rpm"] = frame["delivery"].map(stats["delivery_rpm"])
    f["equip_rpm"] = frame["equipment"].map(stats["equip_rpm"])
    city_fallback = f[["pickup_rpm", "delivery_rpm"]].mean(axis=1)
    f["lane_rpm"] = f["lane_rpm"].fillna(city_fallback).fillna(g)
    f["pickup_rpm"] = f["pickup_rpm"].fillna(g)
    f["delivery_rpm"] = f["delivery_rpm"].fillna(g)

    f["expected_rate"] = f["lane_rpm"] * frame["distance"]
    return f


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=600,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=RNG,
    )


def evaluate(name: str, y_true, y_pred) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
    print(f"{name:<28} MAE ${mae:8.2f}   RMSE ${rmse:8.2f}   MAPE {mape:6.2%}")


# --------------------------------------------------------------------------
def main() -> None:
    dev = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    dev = clean(dev, is_train=True)

    # ---- time-based validation: last two months held out ----
    cutoff = pd.Timestamp("2025-09-01")
    tr, ho = dev[dev["date"] < cutoff], dev[dev["date"] >= cutoff]
    print(f"\ntime split: train {len(tr):,} rows (Jan-Aug) | holdout {len(ho):,} rows (Sep-Oct)")

    stats = lane_city_stats(tr)
    x_tr, x_ho = build_features(tr, stats), build_features(ho, stats)
    y_tr, y_ho = tr["posted_rate"], ho["posted_rate"]

    print("\nholdout performance (predicting 2 unseen future months):")
    evaluate("baseline lane-rpm x miles", y_ho, x_ho["expected_rate"])

    ridge = Ridge(alpha=1.0).fit(x_tr.fillna(0), y_tr)
    evaluate("ridge regression", y_ho, ridge.predict(x_ho.fillna(0)))

    gbm = make_model().fit(x_tr, y_tr)
    evaluate("gradient boosting", y_ho, gbm.predict(x_ho))

    # ---- final model on all cleaned development data ----
    stats = lane_city_stats(dev)
    final = make_model().fit(build_features(dev, stats), dev["posted_rate"])

    # ---- 12,000 validation predictions ----
    val = clean(pd.read_csv(DATA / "validation.csv", parse_dates=["date"]), is_train=False)
    preds = final.predict(build_features(val, stats))
    preds = np.clip(preds, 1.0, None)

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    template["predicted_rate"] = (
        template["load_id"].map(dict(zip(val["load_id"], preds))).round(2)
    )
    template.to_csv("validation_predictions.csv", index=False)
    print(f"\nwrote validation_predictions.csv ({len(template):,} rows)")

    # ---- fixed December lane, one prediction per day ----
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    lex = dev[dev["pickup"] == "Lexington"].iloc[0]
    fw = dev[dev["delivery"] == "Fort Wayne"].iloc[0]
    dec_full = dec.assign(
        load_id="DEC",
        pickup_lat=lex["pickup_lat"], pickup_lon=lex["pickup_lon"],
        delivery_lat=fw["delivery_lat"], delivery_lon=fw["delivery_lon"],
    )
    # market_index / quote_signal are absent from the December file; use the
    # daily market averages observed in validation.csv for the same dates
    market = (
        val[val["date"].dt.month == 12]
        .groupby("date")[["market_index", "quote_signal"]]
        .mean()
    )
    dec_full = dec_full.merge(market, on="date", how="left")
    dec_full[["market_index", "quote_signal"]] = dec_full[
        ["market_index", "quote_signal"]
    ].ffill().bfill()

    dec["predicted_rate"] = np.round(
        np.clip(final.predict(build_features(dec_full, stats)), 1.0, None), 2
    )
    dec_out = dec.copy()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("filled predicted_rate for 31 December days "
          f"(range ${dec['predicted_rate'].min():.0f}-${dec['predicted_rate'].max():.0f})")


if __name__ == "__main__":
    main()

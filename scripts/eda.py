"""Step 1 - Exploratory Data Analysis for the freight rate assessment.

Reads data/train_test.csv and data/validation.csv, prints a full data-quality
report, and saves exploration plots to reports/eda/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("reports/eda")
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])

section("BASIC SHAPE / TYPES")
print("train_test:", train.shape, "| validation:", val.shape)
print(train.dtypes)
print(train.head(3))

section("DATE COVERAGE")
print("train:", train["date"].min(), "->", train["date"].max())
print("validation:", val["date"].min(), "->", val["date"].max())
print("\nloads per month (train):")
print(train.groupby(train["date"].dt.to_period("M")).size())

section("MISSING VALUES")
print("train:\n", train.isna().sum())
print("validation:\n", val.isna().sum())

section("DUPLICATES")
print("duplicate load_id (train):", train["load_id"].duplicated().sum())
print("fully duplicated rows (train):", train.duplicated().sum())
print("duplicated rows ignoring load_id (train):", train.drop(columns=["load_id"]).duplicated().sum())
print("duplicate load_id (validation):", val["load_id"].duplicated().sum())

section("CATEGORICAL COLUMNS")
for col in ["equipment", "pickup", "delivery"]:
    vals = train[col]
    print(f"\n{col}: {vals.nunique()} unique")
    print(vals.value_counts().head(12))
    ws = vals.astype(str).str.strip().ne(vals.astype(str)).sum()
    print(f"values with leading/trailing whitespace: {ws}")
print("\nequipment (validation):", sorted(val["equipment"].dropna().unique()))
print("pickups only in validation:", sorted(set(val["pickup"]) - set(train["pickup"])))
print("deliveries only in validation:", sorted(set(val["delivery"]) - set(train["delivery"])))

section("NUMERIC SUMMARIES (train)")
num_cols = ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
print(train[num_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T)
for col in num_cols:
    v = pd.to_numeric(train[col], errors="coerce")
    print(f"{col}: NaN={v.isna().sum()}  <=0: {(v <= 0).sum()}  negative: {(v < 0).sum()}")

section("TARGET SANITY: posted_rate and rate per mile")
rpm = train["posted_rate"] / train["distance"]
print(rpm.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
print("rate/mile < 0.5 $:", (rpm < 0.5).sum(), "| > 10 $:", (rpm > 10).sum())
print("\ncheapest 5 loads:")
print(train.nsmallest(5, "posted_rate")[["load_id", "distance", "posted_rate", "date"]])
print("most expensive 5 loads:")
print(train.nlargest(5, "posted_rate")[["load_id", "distance", "posted_rate", "date"]])

section("COORDINATE CONSISTENCY")
for name, frame in [("train", train), ("validation", val)]:
    g = frame.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique()
    print(f"{name}: pickup cities with >1 distinct lat/lon:", (g > 1).any(axis=1).sum(), "of", len(g))
hv = haversine_miles(train["pickup_lat"], train["pickup_lon"], train["delivery_lat"], train["delivery_lon"])
ratio = train["distance"] / hv
print("distance / straight-line-distance ratio:")
print(ratio.describe(percentiles=[0.01, 0.5, 0.99]))
bad_geo = ((ratio < 0.9) | (ratio > 3)).sum()
print("rows where stated distance disagrees badly with coordinates:", bad_geo)

section("WEIGHT SANITY")
w = train["weight"]
print("weight < 1000 lb:", (w < 1000).sum(), "| > 48000 lb:", (w > 48000).sum())

section("SIGNAL COLUMNS vs TARGET (correlations)")
print(train[["distance", "weight", "market_index", "quote_signal", "posted_rate"]].corr()["posted_rate"])

section("MARKET INDEX / QUOTE SIGNAL OVER TIME")
monthly = train.groupby(train["date"].dt.to_period("M")).agg(
    mean_rate=("posted_rate", "mean"),
    mean_rpm=("posted_rate", lambda s: np.nan),
    market_index=("market_index", "mean"),
    quote_signal=("quote_signal", "mean"),
    n=("load_id", "size"),
)
monthly["mean_rpm"] = (train["posted_rate"] / train["distance"]).groupby(train["date"].dt.to_period("M")).mean()
print(monthly)
print("\nvalidation month means: market_index=%.4f quote_signal=%.4f" % (val["market_index"].mean(), val["quote_signal"].mean()))

section("DECEMBER LANE CHECK (Lexington -> Fort Wayne)")
lane = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
print("rows on this exact lane in train:", len(lane))
if len(lane):
    print(lane[["distance", "equipment", "weight", "posted_rate", "date"]].describe(include="all").head(12))
print("Lexington as pickup anywhere:", (train["pickup"] == "Lexington").sum())
print("Fort Wayne as delivery anywhere:", (train["delivery"] == "Fort Wayne").sum())

# ---------------- plots ----------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes[0, 0].hist(train["posted_rate"], bins=80)
axes[0, 0].set_title("posted_rate distribution")
axes[0, 1].hist(rpm.clip(0, 12), bins=80)
axes[0, 1].set_title("rate per mile ($/mi, clipped at 12)")
axes[0, 2].scatter(train["distance"], train["posted_rate"], s=2, alpha=0.15)
axes[0, 2].set_title("distance vs posted_rate")
daily = train.groupby("date")["posted_rate"].mean()
axes[1, 0].plot(daily.index, daily.values, lw=0.8)
axes[1, 0].set_title("daily mean posted_rate over 2025")
axes[1, 1].plot(train.groupby("date")["market_index"].mean(), lw=0.8, label="market_index")
axes[1, 1].set_title("market_index over time")
by_dow = train.groupby(train["date"].dt.dayofweek)["posted_rate"].mean()
axes[1, 2].bar(by_dow.index, by_dow.values)
axes[1, 2].set_title("mean rate by day of week (0=Mon)")
for ax in axes.flat:
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "eda_overview.png", dpi=150)
print(f"\nSaved plots to {OUT / 'eda_overview.png'}")

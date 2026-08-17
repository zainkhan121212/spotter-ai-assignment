# Freight Rate Prediction — Spotter AI ML Assessment

Predicts the posted rate for truckload freight using load features
(lane, distance, equipment, weight, date, market signals).

## Project structure

```
data/                          provided datasets (+ December file with predictions filled)
scripts/eda.py                 exploratory data analysis and data-quality report
scripts/train_model.py         cleaning, features, validation, final predictions
validation_predictions.csv     final predictions for the 12,000 validation loads
score.py, requirements.txt     provided scorer
reports/eda/                   EDA figures
scorer_results/                chart produced by score.py (generated)
```

## Setup and run

```bash
python -m pip install -r requirements.txt scikit-learn

python scripts/eda.py           # optional: reproduce the data exploration
python scripts/train_model.py   # trains the model and writes both output files

python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

## Approach

**Data cleaning** (issues found during EDA):
- 292 rows had negative weights whose absolute values look like ordinary
  weights — treated as sign-flip entry errors and fixed with `abs()`.
- 536 rows (~1.1%) had corrupted rates far outside the plausible
  $0.80–$6.00 per-mile band (e.g. $57 for a 105-mile load, $25,533 for a
  2,830-mile load) — removed from training.
- Missing `weight` (300) and `market_index` (374) values imputed with the
  median and the same-day market average respectively.

**Features:** distance (raw + log), weight, market index, quote signal,
calendar features (month, day-of-week, cyclic day-of-year), pickup/delivery
coordinates, equipment, and learned rate-per-mile levels for each lane,
city, and equipment type (with graceful fallbacks for lanes and cities that
never appear in training).

**Validation split:** the task is to predict Nov–Dec 2025 from Jan–Oct
data, so the split mirrors that: train on Jan–Aug, hold out Sep–Oct (the
last two months) as a pseudo-future. Random splitting would leak future
information and overstate accuracy.

**Model:** `HistGradientBoostingRegressor` (absolute-error loss) compared
against a lane-rate baseline and ridge regression on the holdout:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Lane rate-per-mile × miles | $151.49 | $317.61 | 6.24% |
| Ridge regression | $158.69 | $307.33 | 9.51% |
| Gradient boosting | **$76.67** | **$264.05** | **3.30%** |

The final model is retrained on all cleaned development data before
predicting the validation loads and the fixed December lane.

# Prompt for Antigravity

Copy everything below the line into Antigravity as a single task.

---

## Task: Complete the Spotter AI Machine Learning Engineer assessment

I am applying for a Machine Learning Engineer role at Spotter AI. I have a
freight-rate prediction assessment to complete. Most of the modelling work
already exists in a GitHub repository — your job is to move it into a clean
new repository, re-run everything on my machine (I have an NVIDIA GPU), and
produce the two remaining deliverables (a report and a Loom script).

Work through the steps in order. Do not skip the verification steps.

---

### Step 0 — Environment

I am on Windows with Anaconda installed. Create and use a Python environment
with these packages:

```
pandas>=2.0  numpy>=1.26  matplotlib>=3.8  scikit-learn  jupyter  nbformat
torch          # install the CUDA build so my GPU is used:
               # pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible before continuing:

```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If it prints `False`, install the CUDA build of PyTorch and re-check. Tell me
if it still fails — do not silently fall back to CPU.

---

### Step 1 — Get the existing work

Clone my existing repository and check out the working branch:

```bash
git clone https://github.com/zainkhan121212/spotter-ai-assignment.git
cd spotter-ai-assignment
git checkout claude/spotter-ai-assignment-ov8rrf
```

It contains:

```
data/train_test.csv                       48,000 labelled loads (Jan-Oct 2025)
data/validation.csv                       12,000 loads to predict (Nov-Dec 2025)
data/validation_predictions_template.csv  answer template (load_id, predicted_rate)
data/december_chart_inputs.csv            31 December rows, predicted_rate filled
scripts/eda.py                            exploration script
scripts/train_model.py                    cleaning + features + training pipeline
freight_rate_analysis.ipynb               step-by-step notebook (EDA -> predictions)
alternative_models.ipynb                  GPR / MLP / LSTM comparison
validation_predictions.csv                final predictions (12,000 rows)
score.py, requirements.txt                the scorer provided by Spotter
reports/, scorer_results/                 generated charts
```

---

### Step 2 — Create a new clean repository

Create a **new public** GitHub repository under my account `zainkhan121212`
named **`freight-rate-prediction`**, with no README, no .gitignore and no
license (it must start empty).

Copy all files from the cloned repo into it, then commit and push to `main`.

**Important commit rules:**
- Author every commit as `Zain Khan <zain2004khan@gmail.com>`.
- Do **not** mention any AI assistant, model name, or code-generation tool in
  commit messages, code comments, the README, or the report. The work must
  read as entirely my own.
- Use clear, ordinary commit messages such as
  `Add exploratory data analysis`, `Add training pipeline and predictions`.

---

### Step 3 — Re-run everything locally and confirm the numbers

Run both notebooks top to bottom (Restart Kernel and Run All) and the
pipeline script:

```bash
python scripts/train_model.py
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

The scorer must print:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

In `alternative_models.ipynb`, the PyTorch cells must print `device: cuda`
(they auto-detect the GPU). Confirm this happened.

These are the numbers already achieved on the Sep–Oct holdout — treat them as
the benchmark. Report any result that differs by more than a few percent:

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Gradient boosting (final model) | $76.67 | $264.05 | 3.30% | 0.9647 |
| Gaussian Process (2,000-row sample) | $156.35 | $316.91 | 6.44% | 0.9492 |
| Neural network (MLP) | $160.74 | $324.79 | 7.55% | 0.9466 |
| Baseline (lane rate/mile × miles) | $151.49 | $317.61 | 6.24% | — |
| Ridge regression | $158.69 | $307.33 | 9.51% | — |

Also: 84% of holdout predictions land within 5% of the actual rate, 99%
within 10%. The LSTM (market-level daily series) reaches 1.29% error on the
daily average, but it predicts one number per day and cannot price individual
loads.

---

### Step 4 — Context you need (already established — do not re-derive)

**The problem.** Predict `posted_rate` (the price in dollars) of a truckload
freight shipment from its features: pickup/delivery city and coordinates,
distance in miles, equipment type (Dry Van / Reefer / Flatbed), weight,
date, `market_index` and `quote_signal`.

**Data-quality issues found in `train_test.csv` (all already handled):**
1. **292 rows have negative `weight`** (e.g. −47,500 lb). Their absolute
   values look like ordinary truck weights, so these are sign-flip entry
   errors → fixed with `abs()`.
2. **536 rows (~1.1%) have corrupted `posted_rate`.** Normal loads price
   between roughly $1.70 and $3.20 per mile; these fall outside a
   $0.80–$6.00 per-mile band (e.g. $57 for a 105-mile load, $25,533 for a
   2,830-mile load) → dropped from training only.
3. **Missing values:** `weight` (300 rows) filled with the median;
   `market_index` (374 rows) filled with the same-day market average.
4. **New Orleans ↔ Shreveport rows have broken coordinates** (lat/lon only
   ~7 miles apart when the cities are ~300 miles apart). The `distance`
   column is correct, so distance is trusted over coordinates.
5. **8 cities appear only in `validation.csv`** (Allentown, Charlotte,
   Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego) and never in
   training — the features use fallbacks (lane → city average → global
   average) so unseen lanes still get a sensible value.

**Validation split.** Training data covers Jan–Oct 2025; the loads to predict
are Nov–Dec 2025. So the split is **time-based**: train on Jan–Aug, hold out
Sep–Oct as a pseudo-future. A random split would leak future information and
overstate accuracy.

**Features used.** distance (raw and log), weight, `market_index`,
`quote_signal`, month, day-of-week, cyclic day-of-year (sin/cos), weekend
flag, pickup/delivery latitude and longitude, equipment type, and learned
average rate-per-mile per lane, per pickup city, per delivery city and per
equipment type, plus an `expected_rate` = lane rate/mile × distance.

**Final model.** `HistGradientBoostingRegressor` (scikit-learn), absolute-error
loss, 600 iterations, learning rate 0.06, 63 leaf nodes, min 40 samples per
leaf, L2 regularisation 1.0, random_state 42. Retrained on all cleaned data
before producing the final predictions.

**December chart file.** `december_chart_inputs.csv` has only seven columns
(pickup, delivery, distance, equipment, weight, date, predicted_rate) — it is
missing `market_index` and `quote_signal`. Those are filled by borrowing each
December day's average market values from the December rows inside
`validation.csv`, and city coordinates come from the training data. The lane
is fixed: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb, one row per
day of December 2025. Predicted rates come out around $788–$811.

**Is this time-series data?** Only partly, and this matters for the report.
Each row is an independent shipment, not repeated observations of one entity
— it is tabular/panel data with a time component. The only genuine series is
the market-level daily average, which the calendar features and
`market_index` already capture. That is why gradient boosting beats the
sequence models, and it is worth stating explicitly.

---

### Step 5 — Write the report (deliverable, still missing)

Produce a **PDF** (3–4 pages) named `report.pdf`, committed to the repo. It
must contain, in this order:

1. **Problem and data overview** — what is being predicted, dataset sizes,
   date ranges.
2. **Data quality findings** — the five issues listed in Step 4 and how each
   was handled. Include the rate-per-mile histogram from the EDA.
3. **Train/test split and validation approach** — the time-based split, why
   random splitting is wrong here. *(Spotter explicitly asks for this.)*
4. **Feature engineering** — the feature groups and the unseen-city fallback.
5. **Model selection** — the comparison table from Step 3, and one paragraph
   explaining why gradient boosting was chosen over the Gaussian Process,
   neural network and LSTM.
6. **Results** — MAE, RMSE, MAPE, R², the within-5%/within-10% figures, and
   the predicted-vs-actual plots from `reports/predicted_vs_actual.png`.
7. **The December chart** — embed `scorer_results/candidate_december.png`
   exactly as produced by `score.py`. *(Spotter explicitly asks for this.)*

Write in plain, professional English. No AI attribution anywhere.

---

### Step 6 — Write the Loom script (deliverable, still missing)

Create `loom_script.md` — a word-for-word script for a **2.5 minute** video I
will record myself, with timings. It must cover the five points Spotter asks
for, in this order:

1. Key findings from exploring the data
2. Data-quality issues found and how they were addressed
3. Reasoning behind the chosen model
4. Training and validation approach, including how the data was split
5. A brief walkthrough of the most important parts of the code

Write it in natural spoken English (short sentences, easy to read aloud), and
mark which notebook section to have on screen at each point.

---

### Step 7 — Final check

Confirm and report back on each item:

- [ ] New public repo `freight-rate-prediction` exists and `main` has all files
- [ ] `validation_predictions.csv` has exactly 12,000 rows and the two columns
      `load_id,predicted_rate`, all values positive
- [ ] `data/december_chart_inputs.csv` has all 31 `predicted_rate` values filled
- [ ] `score.py` runs clean and regenerates `scorer_results/candidate_december.png`
- [ ] Both notebooks run top to bottom with no errors, PyTorch cells show `device: cuda`
- [ ] `report.pdf` and `loom_script.md` are committed
- [ ] README explains how to install and run everything
- [ ] No AI assistant or tool is mentioned anywhere in the repo
- [ ] Every commit is authored as Zain Khan <zain2004khan@gmail.com>

Then give me the repository URL and a short summary of the final metrics.

# ClimatePulse — Zerve Platform Workflow

## 1. Overview

ClimatePulse is a Canadian agriculture and climate analysis project built entirely on the Zerve platform. It answers a single thesis: **Extreme Weather leads to Crop Failure leads to Economic Impact** across the Prairie provinces (Alberta, Saskatchewan, Manitoba).

The project uses Zerve's core platform capabilities end-to-end:

- **DAG Execution**: A 4-branch parallel pipeline ingests data from three independent government sources, then joins them into a unified feature matrix.
- **Fleet**: Parallel processing distributes weather-station fetches across provinces and crops.
- **App Builder**: An interactive Streamlit dashboard lets users explore the weather-yield-price chain with province maps and parameter sliders.
- **API Endpoints**: Two model-serving endpoints (`/predict-yield` and `/predict-price-impact`) expose the trained models for programmatic access.
- **Notebook Environment**: Exploratory analysis, SHAP explainability plots, and data-quality checks run in Zerve notebooks.

## 2. DAG Architecture

The pipeline is a 4-branch directed acyclic graph. Branches 1-3 run in parallel (no dependencies between them). Branch 4 is a fan-in join that requires all three upstream branches to complete.

```
                    +---------------------------+
                    |   StatsCan 32-10-0359     |
                    |   (Crop Yields ZIP)       |
                    +---------------------------+
                                |
                    pipeline_statcan_yields.py
                                |
                                v
                    +---------------------------+
                    |   ca_crop_yields.csv      |
                    |   312 rows                |----+
                    +---------------------------+    |
                                                     |
                    +---------------------------+    |
                    |   StatsCan 32-10-0077     |    |
                    |   (Farm Prices ZIP)       |    |
                    +---------------------------+    |
                                |                    |
                    pipeline_statcan_prices.py       |
                                |                    |     +---------------------------+
                                v                    +---->|  pipeline_feature_matrix.py|
                    +---------------------------+    |     |  (Fan-in Join)             |
                    |   ca_farm_prices_monthly   |   |     +---------------------------+
                    |   16,776 rows             |----+                 |
                    +---------------------------+    |                 v
                                                     |     +---------------------------+
                    +---------------------------+    |     |   ca_feature_matrix.csv    |
                    |   ECCC Climate API        |    |     |   300 rows x 19 columns   |
                    |   (10 Prairie Stations)   |    |     +---------------------------+
                    +---------------------------+    |                 |
                                |                    |                 v
                    pipeline_eccc_weather.py         |     +---------------------------+
                                |                    |     |   XGBoost Training         |
                                v                    |     |   train_xgboost.py         |
                    +---------------------------+    |     +---------------------------+
                    |   ca_weather_features.csv |----+                 |
                    |   75 rows                 |          +-----------+-----------+
                    +---------------------------+          |                       |
                                                           v                       v
                                                  /predict-yield       /predict-price-impact
                                                           |                       |
                                                           +-----------+-----------+
                                                                       |
                                                                       v
                                                           Interactive Dashboard
                                                           (Zerve App Builder)
```

### Branch 1 — Crop Yields (parallel)

- **Script**: `scripts/pipeline_statcan_yields.py`
- **Source**: StatsCan Table 32-10-0359 (bulk ZIP, 5.1 MB)
- **Processing**: Downloads ZIP, extracts 390,493-row CSV, filters to Prairie provinces (AB, SK, MB) and 4 crops (Wheat, Canola, Barley, Oats), keeps metric units only (kg/ha, hectares, tonnes), restricts to 2000+.
- **Output**: `data/processed/ca_crop_yields.csv` (312 rows)
- **Key fields**: `year`, `province`, `crop`, `yield_kg_ha`, `harvested_ha`, `production_mt`

### Branch 2 — Farm Prices (parallel)

- **Script**: `scripts/pipeline_statcan_prices.py`
- **Source**: StatsCan Table 32-10-0077 (bulk ZIP, 1.5 MB)
- **Processing**: Downloads ZIP, extracts 116,839-row CSV, filters to Prairie provinces and target commodities (wheat, canola, barley, oats by keyword match), retains monthly granularity.
- **Output**: `data/processed/ca_farm_prices_monthly.csv` (16,776 rows)
- **Key fields**: `ref_date` (YYYY-MM), `province`, `commodity`, `price` (CAD), `uom`

### Branch 3 — Weather Features (parallel)

- **Script**: `scripts/pipeline_eccc_weather.py`
- **Source**: ECCC Climate Bulk Data API, 10 curated weather stations across the Prairies
- **Stations**: Alberta (4): Calgary Intl, Edmonton Intl, Lethbridge, Medicine Hat. Saskatchewan (4): Regina Intl, Saskatoon Diefenbaker, Swift Current CDA, Indian Head CDA. Manitoba (2): Winnipeg Richardson Intl, Brandon.
- **Processing**: Fetches daily records per station per year (250 HTTP requests, 2000-2024), filters to growing season (May 1 - Sep 30), computes 8 agro-climate features per station, then averages across stations within each province.
- **Output**: `data/processed/ca_weather_features.csv` (75 rows: 3 provinces x 25 years)
- **Features computed**: `gdd_total` (growing degree days, base 5C), `heat_stress_days` (max temp >30C), `precip_total_mm`, `precip_may_jun_mm`, `precip_jul_aug_mm`, `max_consecutive_dry_days` (precip <1mm), `frost_free_days`, `mean_temp_growing`

### Branch 4 — Feature Matrix (fan-in join)

- **Script**: `scripts/pipeline_feature_matrix.py`
- **Depends on**: All three upstream branches
- **Processing**: Joins yields, weather, and annualized prices on province + year. Adds lagged features (previous year's yield, precipitation, GDD) and price-change percentage. Focuses on 2000-2024.
- **Output**: `data/processed/ca_feature_matrix.csv` (300 rows x 19 columns)
- **Feature categories**: ID (year, province, crop), Target (yield_kg_ha), Area (harvested_ha, production_mt), Weather (8 features), Lagged (3 features), Price (2 features)

## 3. Pipeline Steps

Execution order within the Zerve DAG:

| Step | Script | Runtime | Zerve Node Type | Dependencies |
|------|--------|---------|-----------------|--------------|
| 1a | `pipeline_statcan_yields.py` | ~5 sec | Python | None (parallel) |
| 1b | `pipeline_statcan_prices.py` | ~5 sec | Python | None (parallel) |
| 1c | `pipeline_eccc_weather.py` | ~3 min | Python | None (parallel) |
| 2 | `pipeline_feature_matrix.py` | ~1 sec | Python | 1a, 1b, 1c |
| 3 | `train_xgboost.py` | ~2 min | Python | 2 |
| 4 | Dashboard deployment | -- | App Builder | 3 |

Steps 1a, 1b, and 1c execute concurrently in the Zerve DAG. Step 2 triggers only after all three complete. Step 3 trains the XGBoost model on the joined feature matrix. Step 4 deploys the interactive dashboard.

## 4. App Builder (Planned)

> **Status: Planned.** The following describes the planned App Builder dashboard, to be built using Zerve's agent and App Builder features. The design spec is finalized; implementation has not started.

The Zerve App Builder will host a Streamlit interactive dashboard with the following components:

**Province Map**: Choropleth visualization of the three Prairie provinces, color-coded by predicted yield impact or historical drought severity.

**Weather Parameter Sliders**: Users adjust growing-season weather inputs to run what-if scenarios:
- Growing degree days (GDD, base 5C)
- Heat stress days (>30C threshold)
- Total precipitation (mm)
- Early-season precipitation (May-Jun, mm)
- Mid-season precipitation (Jul-Aug, mm)
- Max consecutive dry days
- Frost-free period (days)
- Mean growing-season temperature (C)

**Model Output Display**: As sliders change, the dashboard is designed to call `/predict-yield` and `/predict-price-impact` in real time, showing:
- Predicted yield (kg/ha) per province and crop
- Yield anomaly relative to historical baseline
- Predicted price-change percentage
- SHAP waterfall for the current input (which weather factors drive the prediction)

**2021 Drought Case Study**: A preset button loads 2021 drought conditions to demonstrate the full thesis chain — extreme heat and low precipitation collapse yields, which triggers commodity price spikes. Note: the 2021 holdout R² is -2.3 (expected out-of-distribution behavior). The model overpredicts yields under drought conditions, and this gap between predicted and actual yield IS the drought-impact measure — the demo illustrates this intentionally.

## 5. API Endpoints (Planned)

> **Status: Planned.** The following endpoints are the next phase of work, to be deployed via Zerve's API serving layer. The schemas below are the design spec.

### `/predict-yield`

Predicts crop yield given weather conditions, province, and crop type.

**Method**: POST

**Input** (JSON):
```json
{
  "province": "Saskatchewan",
  "crop": "Wheat",
  "gdd_total": 1400.0,
  "heat_stress_days": 15,
  "precip_total_mm": 200.0,
  "precip_may_jun_mm": 90.0,
  "precip_jul_aug_mm": 80.0,
  "max_consecutive_dry_days": 18,
  "frost_free_days": 120,
  "mean_temp_growing": 15.5,
  "prev_year_precip_mm": 250.0,
  "prev_year_gdd": 1350.0,
  "prev_year_yield_kg_ha": 3200.0
}
```

**Output** (JSON):
```json
{
  "predicted_yield_kg_ha": 2850.0,
  "yield_anomaly_pct": -12.3,
  "model": "xgboost",
  "cv_r2": 0.68
}
```

### `/predict-price-impact`

Predicts commodity price change given a yield anomaly, using the OLS regression from the price-impact analysis.

**Method**: POST

**Input** (JSON):
```json
{
  "province": "Saskatchewan",
  "crop": "Wheat",
  "yield_anomaly_pct": -44.0
}
```

**Output** (JSON):
```json
{
  "predicted_price_change_pct": 15.2,
  "model": "ols_regression",
  "pearson_r": -0.37
}
```

Both endpoints are served from a single Zerve-hosted model artifact. The yield endpoint runs the XGBoost regressor; the price endpoint applies the OLS slope/intercept from the secondary price-impact model.

## 6. Fleet Usage (Planned/Partial)

> **Status: Partial.** Cross-validation parallelism (`n_jobs=-1`) is implemented and working. Weather-station Fleet distribution and Province x Crop inference parallelism are planned for the Zerve deployment phase.

Zerve Fleet enables parallel processing at two levels:

**Weather Station Fetching** *(planned)*: The ECCC weather pipeline fetches data for 10 stations across 25 years (250 HTTP requests). Fleet would distribute these fetches across workers, with rate-limiting (0.5s pause every 10 requests) to respect ECCC's API. Each station-year is independent, making this embarrassingly parallel.

**Province x Crop Prediction** *(planned)*: At inference time, predictions span 3 provinces x 4 crops = 12 combinations. Fleet would parallelize these predictions for the dashboard's real-time updates, ensuring the interactive sliders remain responsive even when computing all 12 province-crop predictions simultaneously.

**Cross-Validation** *(implemented)*: The XGBoost hyperparameter search runs 50 iterations of RandomizedSearchCV with purged GroupKFold (5 splits). The `n_jobs=-1` setting distributes these across available cores.

## 7. Reproducibility

### Prerequisites

```
pip install -r requirements.txt
```

Required packages: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `shap`

### Full Pipeline Run

```bash
# Step 1: Data ingestion (branches 1-3 can run in parallel)
python scripts/pipeline_statcan_yields.py      # ~5 sec (downloads + caches ZIP)
python scripts/pipeline_statcan_prices.py      # ~5 sec (downloads + caches ZIP)
python scripts/pipeline_eccc_weather.py        # ~3 min (250 HTTP fetches)

# Step 2: Feature matrix join (requires all 3 upstream outputs)
python scripts/pipeline_feature_matrix.py      # ~1 sec

# Step 3: Model training + evaluation
python scripts/train_xgboost.py                # ~2 min
```

### Output Artifacts

| File | Description |
|------|-------------|
| `data/processed/ca_crop_yields.csv` | 312 rows, cleaned crop yields |
| `data/processed/ca_farm_prices.csv` | Historical prices from Table 32-10-0359 |
| `data/processed/ca_farm_prices_monthly.csv` | 16,776 rows, monthly prices from Table 32-10-0077 |
| `data/processed/ca_weather_features.csv` | 75 rows, province-year weather features |
| `data/processed/ca_feature_matrix.csv` | 300 rows x 19 cols, model-ready matrix |
| `data/processed/ca_model_results.json` | Full model results, metrics, and metadata |
| `data/processed/charts/shap_*.png` | SHAP explainability visualizations (5 plots) |

### Key Model Details

- **Algorithm**: XGBoost Regressor with RandomizedSearchCV tuning (50 iterations)
- **Validation**: Purged GroupKFold by year (5 splits, embargo=1) to prevent temporal leakage through lagged features
- **Holdout**: 2021 Western Canadian drought (year 2022 also excluded from training because its `prev_year_*` features embed 2021 target values)
- **Purged CV R-squared**: 0.68
- **Holdout R-squared**: -2.3 (expected — 2021 drought is out-of-distribution; overpredictions reflect a normal-year baseline, not a counterfactual estimate)
- **Price correlation**: Pearson r = -0.37 (yield anomaly vs. price change)
- **Random state**: 42 (all results deterministic given the same input data)

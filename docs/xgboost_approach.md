# XGBoost Crop Yield Model — Approach Document

**Script:** `scripts/train_xgboost.py`
**Input:** `data/processed/ca_feature_matrix.csv` (300 rows, 19 cols)
**Outputs:** `data/processed/ca_model_results.json`, 5 SHAP PNGs in `data/processed/charts/`

---

## 1. Data Splits

| Split | Rows | Years | Rationale |
|-------|------|-------|-----------|
| Drop | 12 | 2000 | Missing lagged features (`prev_year_*` are NaN) |
| Holdout | 12 | 2021 | External test -- Western Canadian drought |
| Drop | 12 | 2022 | `prev_year_*` features embed 2021 holdout yields (lag contamination) |
| Training pool | 264 | 2001-2020, 2023-2024 | Everything else |

**Why drop 2022?** The features `prev_year_yield_kg_ha`, `prev_year_gdd`, and `prev_year_precip_mm` for 2022 rows contain actual 2021 values. If 2022 were in training, the model would see 2021 holdout yields as input features, contaminating the holdout evaluation. This follows the purged cross-validation methodology (de Prado, 2018).

**Residual temporal contamination:** 2023-2024 remain in training. Their `prev_year_*` features reference 2022 and 2023 respectively -- not 2021 directly. However, 2022 yields may reflect drought recovery effects, creating indirect information flow. **We have not run a sensitivity comparison** (e.g., CV/holdout results with vs. without 2023-2024), so the practical magnitude of including these rows is assumed to be small rather than measured. A strict temporal holdout (train only on pre-2021 data) would reduce training to 240 rows and eliminate this concern entirely, at the cost of 24 fewer training rows.

## 2. Feature Selection (13 features)

**Included:**

| # | Feature | Category |
|---|---------|----------|
| 1 | gdd_total | Weather |
| 2 | heat_stress_days | Weather |
| 3 | precip_total_mm | Weather |
| 4 | precip_may_jun_mm | Weather |
| 5 | precip_jul_aug_mm | Weather |
| 6 | max_consecutive_dry_days | Weather |
| 7 | frost_free_days | Weather |
| 8 | mean_temp_growing | Weather |
| 9 | prev_year_precip_mm | Lagged |
| 10 | prev_year_gdd | Lagged |
| 11 | prev_year_yield_kg_ha | Lagged |
| 12 | province (LabelEncoded) | Categorical |
| 13 | crop (LabelEncoded) | Categorical |

**Excluded:**

| Column | Reason |
|--------|--------|
| `harvested_ha` | Derived from target -- leakage |
| `production_mt` | = yield x harvested_ha -- leakage |
| `price_cad_per_tonne` | Outcome variable, used in secondary model |
| `price_change_pct` | Outcome variable, used in secondary model |
| `year` | Temporal signal already captured by lagged features; including it risks overfitting to time trends |

**Target:** `yield_kg_ha`

**Why one unified model (not per-crop)?** 264 training rows / 4 crops = ~66 per crop, too few for XGBoost hyperparameter search. Crop identity is encoded as a categorical feature instead.

**Encoding limitation:** LabelEncoder assigns arbitrary ordinal integers to provinces and crops. XGBoost handles this via tree splits (no ordinal assumption in the split logic), but `crop_encoded` consistently ranks as the 2nd most important feature in both gain and SHAP. This means a substantial portion of the explanation layer is tied to arbitrary numeric coding. One-hot encoding would be cleaner but adds dimensionality on an already small dataset.

## 3. Hyperparameter Tuning

- **Method:** `RandomizedSearchCV`, 50 iterations
- **Scoring:** `neg_mean_absolute_error` (MAE)
  - *Why MAE over R^2?* MAE penalizes in yield units (kg/ha), directly interpretable. R^2 can be misleading with grouped data.
- **CV strategy:** Purged `GroupKFold(n_splits=5)`, grouped by `year`, embargo=1
  - *Why purged GroupKFold?* Standard GroupKFold has lag leakage: when year t is held out, year t+1 in training has `prev_year_*` features that embed year t's target values. The purge removes year t+1 from training when year t is in the test fold, following de Prado (2018).
  - **Not chronological:** Folds are not time-ordered -- early years can appear in test while later years are in training. This is a cross-sectional weather-yield model, not a forecasting model.
  - **Not nested:** Hyperparameters are tuned on the same purged fold structure used for reporting. The reported CV R^2 is an optimistic upper bound. The 2021 holdout is the true external test.
  - 5 folds over 22 training years, ~4-5 years per fold, minus ~4-5 embargo years per fold.

**Search space:**

| Parameter | Values | Notes |
|-----------|--------|-------|
| n_estimators | 100, 200, 300, 500 | Trees |
| max_depth | 3, 4, 5, 6, 7 | Complexity control |
| learning_rate | 0.01, 0.05, 0.1, 0.2 | Step size |
| subsample | 0.7, 0.8, 0.9, 1.0 | Row sampling |
| colsample_bytree | 0.7, 0.8, 0.9, 1.0 | Column sampling |
| min_child_weight | 1, 3, 5, 10 | Regularization |
| reg_alpha | 0, 0.01, 0.1, 1.0 | L1 |
| reg_lambda | 0.5, 1.0, 2.0, 5.0 | L2 |
| gamma | 0, 0.1, 0.5, 1.0 | Min split loss |

`random_state=42` everywhere for reproducibility.

## 4. Cross-Validation Reporting

From the 5-fold purged GroupKFold results, extract per-fold:
- R^2, MAE, RMSE
- Which years were held out
- Training set size (varies per fold due to embargo)

Print as table + overall mean +/- std.

## 5. 2021 Holdout Evaluation

- Retrain final model on all 264 training rows with best hyperparameters
- Predict the 12 held-out 2021 rows
- Print per-row: province, crop, actual yield, predicted yield, error
- Highlight Saskatchewan Wheat specifically (24% below the 25-year baseline from the drought)
- Overall holdout R^2, MAE, RMSE

**Expected result: the holdout test will fail.** The model was trained on normal-range years (2001-2020, 2023-2024) and has never seen drought conditions this extreme. It will systematically overpredict 2021 yields, producing a negative holdout R^2.

**What the failure tells us:** The pattern of overprediction (largest for SK, then AB, then MB) is consistent with the drought's known geographic gradient. This is an observation about the residuals, not a validated counterfactual estimate. The model cannot predict extreme drought years. Its overpredictions reflect a normal-year baseline, and the gap between that baseline and actual 2021 yields makes the drought shortfall visible by contrast -- but this is residual analysis, not a causal claim about what yields "would have been."

**What the model uses:** Feature importance is dominated by `prev_year_yield_kg_ha` and `crop_encoded`, not pure weather features. The model is a crop yield predictor with weather inputs, not a weather-only yield predictor.

## 6. Feature Importance + SHAP

**XGBoost native importance:** Gain-based ranking -> printed table.

**SHAP (TreeExplainer):** Global summaries computed on training data only (no holdout contamination). The 2021 waterfall is computed separately on holdout data.

5 saved plots:
1. `shap_summary_bee.png` -- beeswarm showing feature direction + magnitude
2. `shap_summary_bar.png` -- mean |SHAP| bar chart
3. `shap_2021_drought.png` -- waterfall for Saskatchewan Wheat 2021
4. `shap_dependence_heat_stress.png` -- heat_stress_days vs SHAP value
5. `shap_dependence_precip.png` -- precip_total_mm vs SHAP value

SHAP wrapped in try/except -- model results saved even if SHAP plotting fails (version incompatibilities are common). Uses `matplotlib.use("Agg")` for headless rendering.

## 7. Price Impact Model (Secondary)

This is a correlation/regression analysis to test the second thesis link: crop failure -> price spike. The evidence is weak.

**Steps:**
1. Compute yield anomalies per (crop, province) group from training pool:
   `yield_anomaly_pct = (actual - group_mean) / group_std * 100`
2. Correlate `yield_anomaly_pct` vs `price_change_pct` (Pearson + Spearman, with p-values)
3. Break down correlation per crop
4. Simple OLS via `scipy.stats.linregress`: `price_change_pct ~ yield_anomaly_pct`
5. 2021 chain test: compute yield anomalies from **model-predicted** 2021 yields (not actual), then predict price changes via OLS, compare to actual prices

**Why predicted yields for 2021?** Using actual 2021 yields would leak the holdout target into the price analysis. By using the XGBoost model's predicted yields, the full chain is tested: weather -> predicted yield -> yield anomaly -> price prediction.

**Temporal note:** The anomaly baselines (mean/std per crop-province group) are computed from the training pool, which includes 2023-2024 (post-holdout). This is temporal contamination in the normalization whose magnitude has not been quantified. We do not claim this is a clean out-of-sample test.

**Expected result:** Moderate negative correlation overall (OLS R^2 ~ 0.14, Pearson r ~ -0.37). Barley and canola drive the signal; wheat shows no useful correlation. The 2021 chain test fails because the yield model overpredicts 2021 (doesn't see the drought), so yield anomalies come out positive and predicted price changes are near zero -- missing the actual 20-47% price spikes entirely. This confirms the full weather->yield->price chain does not work for extreme events.

## 8. Output JSON Structure

`data/processed/ca_model_results.json` includes `run_metadata` with package versions, input file hash, and configuration for reproducibility tracking.

Key sections: `run_metadata`, `dataset`, `tuning`, `cv_results`, `holdout_2021`, `feature_importance`, `price_impact`.

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| 264 rows is small for XGBoost | Heavy regularization in search space; purged GroupKFold prevents lag leakage |
| LabelEncoder ordinal encoding | `crop_encoded` is 2nd most important feature in both gain and SHAP -- arbitrary integer coding materially shapes the explanation layer, so feature-importance rankings are qualified rather than cleanly agronomic. One-hot encoding would resolve this but adds dimensionality on a small dataset. |
| SHAP version incompatibility | try/except wrapper; model results saved regardless |
| Price link is moderate (OLS R^2 ~ 0.14) | Stated honestly; per-crop breakdown reveals wheat has no signal; framed as partial association, not validated link |
| 2021 holdout fails (negative R^2) | Stated honestly; framed as expected failure on out-of-distribution extreme event, not as validation success |
| Not nested CV | Reported CV R^2 is an optimistic upper bound; acknowledged in docs |
| 2023-2024 in training pool | Mild indirect temporal contamination via drought recovery effects; accepted to preserve dataset size |
| Reproducibility across environments | Output JSON includes package versions and input hash; `random_state=42` ensures same-environment determinism |

## 10. What This Does NOT Do

- No feature engineering beyond what's in the existing matrix
- No ensemble with other model types
- No time-series forecasting -- this is cross-sectional (year-province-crop rows)
- No dashboard integration (that's a separate step)
- No GDP/insurance impact modeling (potential future work)
- Does NOT confirm the thesis -- this is exploratory analysis, not confirmatory evidence

## 11. Dependencies

```
pip install xgboost shap
```

Both are standard, well-maintained packages. `shap` pulls in `numba` which can be slow to install.

## 12. Verification Checklist

1. `python scripts/train_xgboost.py` runs clean, exits 0
2. `ca_model_results.json` exists, valid JSON, all keys present (including `run_metadata`)
3. 5 SHAP PNGs exist in `data/processed/charts/`
4. 2021 holdout has negative R^2 (expected -- out-of-distribution)
5. 2021 overprediction is largest for SK, then AB, then MB
6. Feature importance shows `prev_year_yield_kg_ha` and `crop_encoded` dominating
7. Price correlation is negative overall, driven by barley and canola (not wheat)
8. 2021 price chain test shows near-zero predictions (model doesn't see drought)
9. Run twice -> identical JSON (determinism via `random_state=42`, same environment)
10. `harvested_ha` and `production_mt` are NOT in features
11. 2022 is NOT in training pool
12. `run_metadata` includes package versions and input file hash

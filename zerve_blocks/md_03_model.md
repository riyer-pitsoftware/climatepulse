# XGBoost Model: Weather → Yield

**Validation:** Purged GroupKFold (5 splits, embargo=1) prevents temporal leakage through lagged features. RandomizedSearchCV with 50 iterations selects hyperparameters.

**Results:**

| Metric | Value |
|--------|-------|
| CV R² (mean ± std) | 0.6551 ± 0.0932 |
| CV MAE | 347 kg/ha |
| Holdout R² (2021) | -1.208 |
| Holdout MAE | 679 kg/ha |

**Why the holdout fails:** The 2021 drought is out-of-distribution. The model systematically overpredicts yields because it learned normal-year patterns. Saskatchewan shows the worst overprediction, then Alberta, then Manitoba — matching the drought's actual geographic gradient.

The gap between predicted and actual yield IS the drought-impact measure.

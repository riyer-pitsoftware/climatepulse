# Zerve Canvas Blocks — ClimatePulse

Mirror of the Zerve canvas block-by-block. Each file = one Zerve block.

## Block Order (DAG left-to-right)

| # | File | Zerve Block | Status |
|---|------|------------|--------|
| 1 | `01_load_statcan_data.py` | load_statcan_data | Agent code ✓ |
| 2 | `02_clean_and_filter_data.py` | clean_and_filter_data | Agent + reconciliation filter appended |
| 3-6 | `03_06_weather_upload.py` | weather_upload | **REPLACES** agent blocks 03-06 in critical path |
| 3 | `03_fetch_eccc_stations.py` | fetch_eccc_stations | Agent code (exploration only) |
| 4 | `04_fetch_all_prairie_weather.py` | fetch_all_prairie_weather | Agent code (exploration only) |
| 5 | `05_build_weather_df.py` | build_weather_df | Agent code (exploration only) |
| 6 | `06_validate_and_visualize_weather.py` | validate_and_visualize_weather | Agent code (exploration only) |
| 7 | `07_build_feature_matrix.py` | build_feature_matrix | **REPLACED** with ground truth logic |
| 8 | `08_train_xgboost_model.py` | train_xgboost_model | **REPLACED** with ground truth logic |

## Critical Path (what must be connected in Zerve DAG)

```
01_load_statcan_data → 02_clean_and_filter_data ──┐
                                                    ├→ 07_build_feature_matrix → 08_train_xgboost_model
03_06_weather_upload ─────────────────────────────┘
```

Agent blocks 03-06 stay in canvas as exploration artifacts (visible to judges)
but are NOT connected to block 07.

## Reconciliation Summary

- **Block 02**: Appended 4-crop filter (Wheat/Canola/Barley/Oats), year >= 2000, full province names
- **Blocks 03-06**: Replaced by CSV upload (agent used monthly API, can't compute daily features)
- **Block 07**: Rewritten — 300×19 matrix, proper lag features, no rolling averages
- **Block 08**: Rewritten — XGBoost, purged GroupKFold embargo=1, 2021-only holdout, SHAP

## How to Use

1. Upload `data/processed/ca_weather_features.csv` to Zerve
2. Paste each block's code into the corresponding Zerve block
3. Wire DAG connections per critical path above
4. Run All — verify: 312 yield rows, 300 features, holdout R² ~ -2.3

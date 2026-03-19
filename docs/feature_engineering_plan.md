> **HISTORICAL — US Climate-AQI Thesis (FAILED)**
> This document records the original US feature engineering plan (12 features for AQI prediction).
> The model built from these features failed cross-validation. The project pivoted to
> Canadian Agriculture in Session 4. Preserved for audit trail.
> Current pipeline: see `docs/pipeline_architecture.md`

# Feature Engineering Plan — cp-1a3 (ARCHIVED)

**Input:** `data/processed/unified_analysis.csv` (expanded with baseline days via cp-9v9)
**Output:** `data/processed/feature_matrix.csv` + `data/processed/feature_metadata.json`
**Depends on:** cp-9v9 (Extend event windows with baseline days)
**Blocks:** cp-f5g (Train prediction model — Ridge/ElasticNet)

---

## Team Decisions (2026-03-18)

| Decision | Rationale | Owner |
|----------|-----------|-------|
| **Add baseline days to training data** (cp-9v9) | 59 rows is not viable; baseline days are "free" rows from existing data with no distribution shift. Target: 100-120 rows. | Suki + Kenji |
| **Cut features from 25 → 12** | 2.4:1 row-to-feature ratio overfits. 12 features at 100+ rows gives ~8-10:1. | Suki |
| **Use Ridge/ElasticNet, not XGBoost** | Linear models generalize better at small N; XGBoost only if rows exceed 100+ after baseline expansion. | Suki |
| **Skip Canadian data for model** (cp-9ik stays P4) | Different AQI scale (AQHI vs AQI), different grid reporting, distribution shift hurts model. Keep as stretch visualization-only. | Suki + Kenji |
| **Skip Hurricane Ida** | Different causal mechanism (diesel generators, not grid fuel-switching). Dilutes the product story. | Kenji |
| **Baseline expansion is a separate bead** | Decouple pipeline work from feature engineering. If join pipeline update breaks, feature code is unaffected. | Kenji |

---

## Data Summary

### Current (pre cp-9v9)

| Event | Event Rows | Region (BA) | Dates | Baseline Fossil % | Baseline Renewable % |
|-------|-----------|-------------|-------|-------------------|---------------------|
| Winter Storm Uri | 28 | ERCO (Texas) | Feb 1-28, 2021 | 58.68% | 28.33% |
| PNW Heat Dome | 21 | BPAT (Pacific NW) | Jun 20-Jul 10, 2021 | 0.14% | 93.05% |
| Winter Storm Elliott | 23 | PJM (Mid-Atlantic) | Dec 15, 2022-Jan 5, 2023 | 56.66% | 6.48% |

### After cp-9v9 (expected)

| Event | Event Rows | Baseline Rows | Total | Baseline Period |
|-------|-----------|--------------|-------|-----------------|
| Winter Storm Uri | 28 | ~14-21 | ~42-49 | Same-season non-event days used for baseline_fossil_pct |
| PNW Heat Dome | 21 | ~14-21 | ~35-42 | Same-season non-event days used for baseline_fossil_pct |
| Winter Storm Elliott | 23 | ~14-21 | ~37-44 | Same-season non-event days used for baseline_fossil_pct |
| **Total** | **72** | **~42-63** | **~114-135** | |

The expanded dataset will include an `is_event` column (1 = during extreme weather, 0 = baseline period). Baseline rows will have `fossil_pct_change ≈ 0` by definition and "normal" AQI values, giving the model counterfactual context.

### Known data gaps

- `mean_prcp`: NaN for all 21 Heat Dome rows (NOAA stations did not report precipitation)
- `pm25_aqi`, `ozone_aqi`, `pm25_mean`, `ozone_mean`: NaN for 6 Elliott rows (Jan 1-5, 2023 — EPA monitors offline over holidays)
- 2 rows flagged `suspect_low_generation` (Heat Dome Jul 10, Elliott Jan 5)

---

## Prediction Targets

### Target 1: `pm25_aqi_next` (Primary)

- **What:** PM2.5 Air Quality Index value for the *next calendar day*
- **How:** Shift `pm25_aqi` backward by 1 row *within each event group* so that row N's features predict row N+1's AQI
- **Why:** Statistical analysis confirmed the strongest signal is a 1-day lag: fossil_pct_change at day T predicts pm25_aqi at day T+1 (pooled r=0.378, p=0.0023; Uri-specific r=0.697, p=0.0001). Same-day correlation is weak (r=0.18, p=0.15). The physical mechanism — emissions take ~24 hours to disperse and register on ground-level PM2.5 monitors — supports this lag.
- **Range:** 16.2 to 79.8 (continuous, EPA AQI scale)
- **Missing after shift:** Last row of each event loses its target (no "next day" to predict). Net loss: 3 rows.

### Target 2: `ozone_aqi_next` (Secondary)

- **What:** Ozone AQI value for the next calendar day
- **How:** Same shift logic as pm25_aqi_next
- **Why:** Ozone is a secondary pollutant formed from NOx + VOCs under UV. Fossil generation emits NOx; AQI impact should also lag. Less statistically validated than PM2.5 in our analysis, but worth predicting for completeness.
- **Range:** 6.4 to 99.0
- **Missing:** Same 3 event-end rows + 6 rows of original NaN = up to 9 missing values

### Target 3: `aqi_category_next` (Classification fallback)

- **What:** Binary label: `good` (AQI 0-50), `not_good` (AQI 51+)
- **How:** Bin `pm25_aqi_next` using EPA breakpoint at 50
- **Why:** If regression R² is poor at ~100 rows, a 2-class logistic regression classifier may generalize better. Also more interpretable for a demo ("will tomorrow's air quality be Good or Not Good?").
- **Class distribution (estimated):** ~60% good, ~40% not_good (better balance than 3-class)
- **Note:** Simplified from original 3-class plan (good/moderate/unhealthy_sensitive) per Suki's recommendation — insufficient rows for a minority class at ~5%.

---

## Features (12 total)

### Group 1: Weather Severity (3 features)

#### 1. `temp_deviation`
- **Formula:** `abs(mean_tmax - 65.0)`
- **Unit:** Degrees Fahrenheit
- **Range:** ~0 to ~39 (65°F is the ASHRAE human comfort baseline where neither heating nor cooling demand spikes)
- **Why:** Both cold snaps and heat waves stress the grid, but in opposite directions. This symmetric measure captures distance from comfort regardless of direction. Statistical analysis found mean_tmax correlates with fossil_pct_change at r=-0.369, p=0.0015 — the negative sign is because colder temps (lower tmax) drive higher fossil use, so absolute deviation is more useful as a predictor than raw temperature.
- **Expected signal:** Higher deviation → larger fossil_pct_change → higher next-day AQI
- **Why kept (over temp_range):** temp_range (max_tmax - min_tmin) is correlated with temp_deviation but adds collinearity without independent signal. temp_deviation is the cleaner causal proxy.

#### 2. `cold_severity`
- **Formula:** `max(0, 32.0 - mean_tmin)`
- **Unit:** Degrees Fahrenheit below freezing
- **Range:** 0 to 34.1 (0 for Heat Dome, up to 34.1 for Uri)
- **Why:** Cold events and heat events stress the grid through different mechanisms. Cold snaps cause heating demand spikes + gas pipeline pressure drops + frozen wind turbines (Uri was defined by gas infrastructure failures). This asymmetric feature isolates the cold-specific pathway. The 32°F threshold is the freezing point — below this, gas infrastructure performance degrades non-linearly.
- **Expected signal:** Higher cold_severity in Uri/Elliott → higher fossil dependence

#### 3. `heat_severity`
- **Formula:** `max(0, mean_tmax - 95.0)`
- **Unit:** Degrees Fahrenheit above 95°F
- **Range:** 0 to 8.9 (0 for Uri/Elliott, up to 8.9 for Heat Dome)
- **Why:** Heat waves cause cooling demand spikes + thermal derating of power lines + reduced hydro output (Heat Dome dried up PNW rivers). 95°F is the threshold where grid operators typically trigger emergency protocols and peaker plants fire. This isolates the heat-specific pathway.
- **Expected signal:** Higher heat_severity in Heat Dome → massive fossil shift (+6.2pp avg)

**Dropped from this group:**
- `temp_range` — correlated with temp_deviation, no independent signal
- `precip_intensity` — correlated with cold_severity (snow drives it); Heat Dome NaN fill adds uncertainty
- `snow_flag` — correlated with cold_severity and is_cold_event; redundant

---

### Group 2: Lagged Grid Response (1 feature)

#### 4. `fossil_pct_change_lag1`
- **Formula:** Previous day's `fossil_pct_change`, *within the same event*
- **Unit:** Percentage points
- **Range:** -23.8 to +25.1
- **Why this is the most important feature:** Lag-1 correlation between fossil_pct_change and pm25_aqi is r=0.378 (p=0.0023) pooled across events, and r=0.697 (p=0.0001) for Uri alone. The regression slope from Uri is 0.472 AQI points per percentage-point of fossil shift. Granger causality test confirms: Uri lag-1 F=16.2, p=0.0005.
- **Event boundary handling:** First row of each event gets NaN (no "yesterday" exists). These rows are dropped from training.

**Dropped from this group:**
- `fossil_pct_change_lag2` — multicollinear with lag1 (autocorrelation in time series); adds NaN rows; lag-1 captures the primary signal
- `renewable_pct_change_lag1` — mirror of fossil shift (correlation ~-0.9); adds collinearity without new information because the `other` category is the only source of independence
- `fossil_shift_acceleration` — derived from lag1 (difference of a feature already included); linear model can learn this implicitly

---

### Group 3: Grid State (3 features)

#### 5. `fossil_pct_change`
- **Formula:** Already in data — `fossil_pct - baseline_fossil_pct`
- **Unit:** Percentage points
- **Range:** -23.8 to +25.1
- **Why:** The current-day fossil shift is the direct measure of grid stress response. It is the middle link in the causal chain (weather → **fossil shift** → AQI). Including both same-day and lag-1 lets the model weight immediate vs delayed effects.
- **Expected signal:** Higher fossil shift today → higher AQI tomorrow (via lag) AND potentially same-day ozone effects

#### 6. `fossil_dominance_ratio`
- **Formula:** `fossil_pct / max(renewable_pct, 1.0)` (floor of 1.0 to avoid division by zero)
- **Unit:** Dimensionless ratio
- **Range:** ~0.03 (Heat Dome — nearly all renewable) to ~28 (Elliott — fossil dominant)
- **Why:** Captures the structural cleanliness of the grid. A grid at 60% fossil / 30% renewable (ratio 2.0) is in a fundamentally different emission regime than one at 80% fossil / 5% renewable (ratio 16.0). The ratio is more informative than either percentage alone because it captures relative dominance.
- **Log-transform consideration:** The range is wide (0.03 to 28); `log(fossil_dominance_ratio)` may be used in training for better model behavior, but we store the raw ratio and let the model handle it.

#### 7. `generation_utilization`
- **Formula:** `total_generation_mwh / event_median_generation` where `event_median_generation` is the median `total_generation_mwh` for that event
- **Unit:** Dimensionless ratio (1.0 = typical day for this event)
- **Range:** ~0.01 to ~2.0
- **Why:** Captures grid stress in terms of demand. A day at 1.5x median generation means the grid is running hot, likely firing inefficient peaker plants that emit more pollutants per MWh. The suspect_low_generation days will show ratios well below 1.0.
- **Event-relative:** Using per-event median normalizes across regions (Texas ERCO generates far more MWh than Pacific NW BPAT).

**Dropped from this group:**
- `fossil_above_baseline` — redundant with fossil_pct_change (sign already encodes above/below); a continuous feature gives the linear model more to work with than a binary threshold

---

### Group 4: Temporal Controls (2 features)

#### 8. `event_day`
- **Formula:** Day index within the event (1, 2, 3, ... up to 28 for Uri). For baseline rows: negative values counting backward from event start (e.g., -14, -13, ..., -1) or a separate index — TBD based on cp-9v9 output structure.
- **Unit:** Integer
- **Range:** ~-21 to 28 (after baseline expansion)
- **Why:** Within each weather event, there's a temporal arc: conditions worsen, peak, then recover. Grid response and AQI follow the same arc with lag. `event_day` lets the model learn position-in-event effects: early days (ramp-up), peak days (worst impact), and recovery days (grid normalizing). Baseline days at negative values give the model a "before" reference.

#### 9. `is_weekend`
- **Formula:** `1 if date.dayofweek >= 5 else 0`
- **Unit:** Binary
- **Why:** Weekend electricity demand is typically 10-15% lower than weekday. Lower demand means fewer peaker plants running, potentially less fossil generation and lower emissions. Also, weekend traffic patterns differ, affecting ozone precursor (NOx) levels. This is a known confounder in air quality studies.

**Dropped from this group:**
- `phase_peak` / `phase_recovery` — correlated with event_day (coarser binning of the same signal); event_day is more informative for a linear model
- `month_sin` / `month_cos` — only 3 months represented (Feb, Jun-Jul, Dec-Jan); with 3 events, cyclical encoding is just a noisy proxy for is_cold_event + region. Wastes 2 degrees of freedom.

---

### Group 5: Event & Region Encoding (2 features)

#### 10. `is_cold_event`
- **Formula:** `1 if event in ('uri_2021', 'elliott_2022') else 0`
- **Unit:** Binary
- **Why:** Cold events and heat events have fundamentally different causal mechanisms: cold → heating demand + gas infrastructure failure; heat → cooling demand + hydro/wind reduction. One binary feature captures this regime difference. We use cold (not heat) as the positive class because 2 of 3 events are cold (Uri + Elliott), giving better class balance.

#### 11. `region_fossil_baseline`
- **Formula:** Copy of `baseline_fossil_pct` (already in data)
- **Unit:** Percent
- **Range:** 0.14% (BPAT) to 58.68% (ERCO)
- **Why:** Encodes regional grid structure as a continuous feature rather than a one-hot categorical. A region with 0.14% baseline fossil (PNW hydro-dominated) responds very differently to stress than one at 58.68% (Texas gas-dominated). Using the continuous baseline is more informative than one-hot dummies and avoids spending 2 degrees of freedom on 3 categories.

**Dropped from this group:**
- `region_renewable_baseline` — approximately the inverse of region_fossil_baseline (correlation ~-0.95 across our 3 regions because the `other` category is relatively stable). Including both adds collinearity with no new information.

---

### Group 6: Interaction (1 feature)

#### 12. `severity_x_fossil_shift`
- **Formula:** `temp_deviation * fossil_pct_change`
- **Unit:** (°F deviation) * (percentage points)
- **Why:** The alternate hypothesis analysis found that extreme temperatures (>93°F or <37°F from 65°F baseline, i.e., temp_deviation > 28) produce 3.5x stronger fossil shifts than moderate temperatures (+10.74pp vs -1.5pp). This interaction term lets the model learn that the combination of extreme weather AND high fossil shift is worse than either alone. Bin analysis was highly significant: Kruskal-Wallis H=14.0, p=0.0009.
- **Expected behavior:** Near-zero for baseline days (low temp_deviation AND low fossil_pct_change). Large positive for peak event days. This makes it a natural "event severity" composite.

**Dropped from this group:**
- `severity_x_lag1_aqi` — requires lag1, adds another NaN-row cost; the single interaction term above captures the threshold effect adequately
- `temp_deviation_squared` — with Ridge regression (L2 penalty), the model already handles non-linearity via the interaction term + regularization; a squared term at this N is a luxury

---

## Feature Summary Table

| # | Feature | Group | Type | Range | NaN Rows | Source Columns |
|---|---------|-------|------|-------|----------|----------------|
| 1 | temp_deviation | Weather | float | 0–39 | 0 | mean_tmax |
| 2 | cold_severity | Weather | float | 0–34.1 | 0 | mean_tmin |
| 3 | heat_severity | Weather | float | 0–8.9 | 0 | mean_tmax |
| 4 | fossil_pct_change_lag1 | Lag | float | -24 to 25 | 3 (first/event) | fossil_pct_change |
| 5 | fossil_pct_change | Grid | float | -24 to 25 | 0 | (already in data) |
| 6 | fossil_dominance_ratio | Grid | float | 0.03–28 | 0 | fossil_pct, renewable_pct |
| 7 | generation_utilization | Grid | float | 0.01–2.0 | 0 | total_generation_mwh |
| 8 | event_day | Temporal | int | ~-21 to 28 | 0 | date, event |
| 9 | is_weekend | Temporal | binary | 0/1 | 0 | date |
| 10 | is_cold_event | Event | binary | 0/1 | 0 | event |
| 11 | region_fossil_baseline | Event | float | 0.14–58.7 | 0 | baseline_fossil_pct |
| 12 | severity_x_fossil_shift | Interaction | float | ~0–800 | 0 | temp_deviation, fossil_pct_change |

**Total: 12 features + 3 targets (pm25_aqi_next, ozone_aqi_next, aqi_category_next)**

---

## Row Accounting

### Estimated (after cp-9v9 baseline expansion)

| Step | Rows Lost | Remaining | Reason |
|------|-----------|-----------|--------|
| Start (post cp-9v9) | — | ~114-135 | unified_analysis.csv with baseline days |
| Drop suspect_low_generation | -2 | ~112-133 | data_quality flag |
| Target shift (last row/event) | -3 | ~109-130 | No next-day AQI to predict |
| Lag-1 features (first row/event) | -3 | ~106-127 | No previous day for lagging |
| EPA NaN in target | -5 | ~101-122 | Elliott Jan 1-4 missing AQI |

**Final training set: ~101-122 rows with 12 features**
**Row-to-feature ratio: ~8.4:1 to 10.2:1** (viable for Ridge/ElasticNet)

### Fallback (if cp-9v9 is delayed or produces fewer rows)

The feature engineering script must work on whatever `unified_analysis.csv` contains. If baseline expansion doesn't land:

| Step | Rows Lost | Remaining |
|------|-----------|-----------|
| Start (current) | — | 72 |
| Drops (same as above) | -13 | 59 |

59 rows / 12 features = 4.9:1. Marginal but workable with strong regularization (high Ridge alpha). The 12-feature cut makes this survivable.

---

## Implementation Plan

### Dependency: cp-9v9 (Baseline Expansion)

cp-9v9 updates `scripts/join_datasets.py` to output baseline rows alongside event rows. The feature engineering script does NOT depend on any specific row count — it works on whatever the CSV contains. But the quality of the model depends on having ~100+ rows.

### File: `scripts/feature_engineering.py`

**Structure:**
```
load_data()                    → reads unified_analysis.csv
filter_quality(df)             → drops suspect_low_generation rows
build_weather_features(df)     → Group 1: temp_deviation, cold_severity, heat_severity
build_lag_features(df)         → Group 2: fossil_pct_change_lag1 (respects event boundaries)
build_grid_features(df)        → Group 3: fossil_pct_change (passthrough), fossil_dominance_ratio, generation_utilization
build_temporal_features(df)    → Group 4: event_day, is_weekend
build_event_features(df)       → Group 5: is_cold_event, region_fossil_baseline
build_interaction_features(df) → Group 6: severity_x_fossil_shift
build_targets(df)              → shift AQI by -1 within events, create binary category
assemble_matrix(df)            → select 12 feature cols + 3 targets + metadata, drop NaN rows
save_outputs(df, metadata)     → write CSV + JSON
main()                         → orchestrate all steps, print summary stats
```

**Each function:**
- Takes the full DataFrame as input
- Returns the DataFrame with new columns added
- Includes a docstring explaining the features it creates
- Is independently testable

### File: `data/processed/feature_matrix.csv`

Output CSV containing:
- All 12 feature columns
- 3 target columns
- `event`, `date`, `is_event` columns retained for cross-validation splits and traceability
- Rows with any NaN in features or primary target (pm25_aqi_next) are excluded

### File: `data/processed/feature_metadata.json`

```json
{
  "features": {
    "temp_deviation": {
      "group": "weather",
      "type": "float",
      "formula": "abs(mean_tmax - 65.0)",
      "source_columns": ["mean_tmax"],
      "description": "Temperature deviation from 65F comfort baseline"
    }
  },
  "targets": {
    "pm25_aqi_next": {
      "type": "float",
      "formula": "pm25_aqi shifted -1 within event",
      "description": "Next-day PM2.5 AQI (primary regression target)"
    },
    "ozone_aqi_next": {
      "type": "float",
      "formula": "ozone_aqi shifted -1 within event",
      "description": "Next-day Ozone AQI (secondary regression target)"
    },
    "aqi_category_next": {
      "type": "binary",
      "formula": "good if pm25_aqi_next <= 50 else not_good",
      "description": "Next-day AQI category (classification fallback target)"
    }
  },
  "row_accounting": {
    "input_rows": "variable (depends on cp-9v9)",
    "output_rows": "variable",
    "drops": {
      "suspect_low_generation": 2,
      "target_shift": 3,
      "lag_features": 3,
      "epa_nan": 5
    }
  },
  "feature_count": 12,
  "target_count": 3,
  "model_recommendation": "Ridge or ElasticNet regression",
  "decisions": {
    "date": "2026-03-18",
    "cut_from": 25,
    "cut_to": 12,
    "reason": "Row-to-feature ratio at original 25 was 2.4:1; cut to achieve 8-10:1"
  }
}
```

### File: `tests/test_features.py`

Test cases:
1. **Row count:** Output has fewer rows than input (drops applied correctly)
2. **No NaN in features:** All 12 feature columns are complete in output
3. **Event boundaries:** Lag features do not cross event boundaries (verify first row of each event is dropped, not filled with prior event's data)
4. **Value ranges:** Each feature falls within expected bounds (e.g., cold_severity >= 0, heat_severity >= 0, is_weekend in {0,1})
5. **Target alignment:** pm25_aqi_next at row N equals pm25_aqi at row N+1 (within same event)
6. **Category labels:** aqi_category_next values are only from {good, not_good}
7. **Determinism:** Running twice produces identical output
8. **Feature count:** Output has exactly 12 feature columns + 3 targets + 3 metadata columns

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| cp-9v9 produces fewer baseline rows than expected | Row-to-feature ratio stays below 8:1 | Script works at any N; 12-feature cut makes 59-row fallback survivable with high regularization |
| Heat Dome precipitation NaN | Cannot compute precip features | precip_intensity dropped from feature set — no longer relevant |
| Event and region are perfectly correlated | Cannot disentangle regional vs event effects | Use continuous region_fossil_baseline instead of one-hot; acknowledge limitation in writeup |
| Baseline days have low variance in targets | Model learns "normal = boring" but doesn't help predict event-day AQI | Baseline days provide the counterfactual; event_day + is_event in features let model distinguish regimes |
| Ridge regression too simple for non-linear effects | Misses threshold amplification | severity_x_fossil_shift interaction term captures the primary non-linearity; ElasticNet as backup; XGBoost if N > 120 |

---

## Dropped Features — Rationale Record

For auditability, here are the 13 features cut and why:

| Feature | Group | Why Dropped |
|---------|-------|-------------|
| temp_range | Weather | Correlated with temp_deviation (~0.8); no independent signal |
| precip_intensity | Weather | Correlated with cold_severity; Heat Dome NaN fill adds uncertainty |
| snow_flag | Weather | Correlated with cold_severity + is_cold_event; redundant binary |
| fossil_pct_change_lag2 | Lag | Multicollinear with lag1; costs 3 extra NaN rows per event |
| renewable_pct_change_lag1 | Lag | Mirror of fossil shift (r~-0.9); collinear |
| fossil_shift_acceleration | Lag | Derived from lag1; linear model learns implicitly |
| fossil_above_baseline | Grid | Redundant with continuous fossil_pct_change |
| phase_peak / phase_recovery | Temporal | Correlated with event_day (coarser binning) |
| month_sin / month_cos | Temporal | Only 3 months; noisy proxy for is_cold_event + region |
| region_renewable_baseline | Event | Inverse of region_fossil_baseline (r~-0.95) |
| severity_x_lag1_aqi | Interaction | Requires lag1; extra NaN cost; one interaction is enough |
| temp_deviation_squared | Interaction | Luxury at small N; interaction term handles non-linearity |

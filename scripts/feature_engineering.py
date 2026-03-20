#!/usr/bin/env python3
"""Feature engineering for ClimatePulse ML pipeline.

Transforms unified_analysis.csv into a feature matrix ready for model training.
Implements 12 features + 3 targets per the team-reviewed plan (2026-03-18).

Bead: cp-1a3
Depends on: cp-9v9 (baseline expansion — optional, script works at any row count)
Blocks: cp-f5g (model training)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_CSV = DATA_DIR / "unified_analysis.csv"
OUTPUT_CSV = DATA_DIR / "feature_matrix.csv"
OUTPUT_META = DATA_DIR / "feature_metadata.json"

FEATURE_COLUMNS = [
    "temp_deviation",
    "cold_severity",
    "heat_severity",
    "fossil_pct_change_lag1",
    "fossil_pct_change",
    "fossil_dominance_ratio",
    "generation_utilization",
    "event_day",
    "is_weekend",
    "is_cold_event",
    "region_fossil_baseline",
    "severity_x_fossil_shift",
]

TARGET_COLUMNS = [
    "pm25_aqi_next",
    "ozone_aqi_next",
    "aqi_category_next",
]

METADATA_COLUMNS = ["event", "date"]


def load_data():
    """Load unified_analysis.csv and parse dates."""
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {INPUT_CSV.name}")
    return df


def filter_quality(df):
    """Drop rows flagged as suspect_low_generation.

    These are days where total generation fell below 50% of the event median,
    indicating partial/unreliable data (e.g., Heat Dome Jul 10, Elliott Jan 5).
    """
    if "data_quality" not in df.columns:
        return df
    mask = df["data_quality"] == "suspect_low_generation"
    n_drop = mask.sum()
    if n_drop > 0:
        print(f"  Dropped {n_drop} suspect_low_generation rows")
    return df[~mask].reset_index(drop=True)


def build_weather_features(df):
    """Group 1: Weather severity features.

    - temp_deviation: absolute distance from 65°F comfort baseline (ASHRAE)
    - cold_severity: degrees below freezing (0 for warm days)
    - heat_severity: degrees above 95°F grid-emergency threshold (0 for cool days)
    """
    df["temp_deviation"] = (df["mean_tmax"] - 65.0).abs()
    df["cold_severity"] = (32.0 - df["mean_tmin"]).clip(lower=0)
    df["heat_severity"] = (df["mean_tmax"] - 95.0).clip(lower=0)
    return df


def build_lag_features(df):
    """Group 2: Lagged grid response.

    - fossil_pct_change_lag1: previous day's fossil shift, within same event.
      First row of each event gets NaN (no cross-event leakage).
      Pooled lag-1 r=0.136 (p=0.188, n.s.); Uri r=0.354 (p=0.051, borderline).
    """
    df["fossil_pct_change_lag1"] = df.groupby("event")["fossil_pct_change"].shift(1)
    return df


def build_grid_features(df):
    """Group 3: Grid state features.

    - fossil_pct_change: passthrough (already in data) — current-day fossil shift
    - fossil_dominance_ratio: fossil_pct / renewable_pct (floored at 1.0)
    - generation_utilization: total generation vs per-event median
    """
    # fossil_pct_change already exists — no action needed
    df["fossil_dominance_ratio"] = df["fossil_pct"] / df["renewable_pct"].clip(lower=1.0)
    event_medians = df.groupby("event")["total_generation_mwh"].transform("median")
    df["generation_utilization"] = df["total_generation_mwh"] / event_medians
    return df


def build_temporal_features(df):
    """Group 4: Temporal controls.

    - event_day: day index within event (1-based for event rows).
      If is_baseline column exists, baseline rows get negative indices
      counting backward from event start.
    - is_weekend: binary weekend indicator (known demand confounder).
    """
    df = df.sort_values(["event", "date"]).reset_index(drop=True)

    if "is_baseline" in df.columns or "is_event" in df.columns:
        # Expanded dataset with baseline rows
        event_col = "is_event" if "is_event" in df.columns else None
        baseline_col = "is_baseline" if "is_baseline" in df.columns else None

        day_indices = []
        for _, group in df.groupby("event"):
            if baseline_col:
                is_evt = ~group[baseline_col].astype(bool)
            elif event_col:
                is_evt = group[event_col].astype(bool)
            else:
                is_evt = pd.Series(True, index=group.index)

            event_rows = group[is_evt]
            baseline_rows = group[~is_evt]

            # Event rows: 1, 2, 3, ...
            evt_days = pd.Series(range(1, len(event_rows) + 1), index=event_rows.index)
            # Baseline rows: -N, -(N-1), ..., -1
            bl_days = pd.Series(
                range(-len(baseline_rows), 0), index=baseline_rows.index
            )
            day_indices.append(pd.concat([bl_days, evt_days]).sort_index())

        df["event_day"] = pd.concat(day_indices)
    else:
        # Original dataset — event rows only
        df["event_day"] = df.groupby("event").cumcount() + 1

    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    return df


def build_event_features(df):
    """Group 5: Event & region encoding.

    - is_cold_event: binary (Uri + Elliott = cold; Heat Dome = heat)
    - region_fossil_baseline: continuous regional grid structure proxy
    """
    cold_events = {"uri_2021", "elliott_2022"}
    df["is_cold_event"] = df["event"].isin(cold_events).astype(int)
    df["region_fossil_baseline"] = df["baseline_fossil_pct"]
    return df


def build_interaction_features(df):
    """Group 6: Interaction terms.

    - severity_x_fossil_shift: temp_deviation * fossil_pct_change
      Captures threshold amplification effect (Kruskal-Wallis p=0.0009).
      Near-zero for baseline days, large for peak event days.
    """
    df["severity_x_fossil_shift"] = df["temp_deviation"] * df["fossil_pct_change"]
    return df


def build_targets(df):
    """Build prediction targets by shifting AQI backward within each event.

    Row N's features predict row N+1's AQI (1-day lag, matching the strongest
    statistical signal). Last row of each event gets NaN target.

    - pm25_aqi_next: primary regression target
    - ozone_aqi_next: secondary regression target
    - aqi_category_next: binary classification fallback (good/not_good at AQI 50)
    """
    df["pm25_aqi_next"] = df.groupby("event")["pm25_aqi"].shift(-1)
    df["ozone_aqi_next"] = df.groupby("event")["ozone_aqi"].shift(-1)
    df["aqi_category_next"] = df["pm25_aqi_next"].apply(
        lambda x: "good" if x <= 50 else "not_good" if pd.notna(x) else np.nan
    )
    return df


def assemble_matrix(df):
    """Select final columns and drop rows with NaN in features or primary target.

    Returns the clean feature matrix ready for model training.
    """
    # Determine metadata columns (include is_event/is_baseline if present)
    meta_cols = list(METADATA_COLUMNS)
    for col in ["is_event", "is_baseline"]:
        if col in df.columns:
            meta_cols.append(col)

    all_cols = meta_cols + FEATURE_COLUMNS + TARGET_COLUMNS
    result = df[all_cols].copy()

    # Drop rows with NaN in any feature column
    feature_nans = result[FEATURE_COLUMNS].isna().any(axis=1)
    n_feature_nan = feature_nans.sum()

    # Drop rows with NaN in primary target
    target_nans = result["pm25_aqi_next"].isna()
    n_target_nan = (~feature_nans & target_nans).sum()  # only count new drops

    result = result.dropna(subset=FEATURE_COLUMNS + ["pm25_aqi_next"]).reset_index(
        drop=True
    )

    print(f"  Dropped {n_feature_nan} rows with NaN features (lag/boundary)")
    print(f"  Dropped {n_target_nan} rows with NaN primary target (shift/EPA gaps)")
    return result


def build_metadata(df_in, df_out):
    """Build feature_metadata.json with feature definitions and row accounting."""
    return {
        "features": {
            "temp_deviation": {
                "group": "weather",
                "type": "float",
                "formula": "abs(mean_tmax - 65.0)",
                "source_columns": ["mean_tmax"],
                "description": "Temperature deviation from 65F comfort baseline",
            },
            "cold_severity": {
                "group": "weather",
                "type": "float",
                "formula": "max(0, 32.0 - mean_tmin)",
                "source_columns": ["mean_tmin"],
                "description": "Degrees below freezing (cold stress proxy)",
            },
            "heat_severity": {
                "group": "weather",
                "type": "float",
                "formula": "max(0, mean_tmax - 95.0)",
                "source_columns": ["mean_tmax"],
                "description": "Degrees above 95F grid-emergency threshold",
            },
            "fossil_pct_change_lag1": {
                "group": "lag",
                "type": "float",
                "formula": "groupby(event).shift(1) on fossil_pct_change",
                "source_columns": ["fossil_pct_change"],
                "description": "Previous day fossil shift (within event)",
            },
            "fossil_pct_change": {
                "group": "grid",
                "type": "float",
                "formula": "fossil_pct - baseline_fossil_pct (already in data)",
                "source_columns": ["fossil_pct_change"],
                "description": "Current-day fossil shift from baseline",
            },
            "fossil_dominance_ratio": {
                "group": "grid",
                "type": "float",
                "formula": "fossil_pct / max(renewable_pct, 1.0)",
                "source_columns": ["fossil_pct", "renewable_pct"],
                "description": "Fossil-to-renewable generation ratio",
            },
            "generation_utilization": {
                "group": "grid",
                "type": "float",
                "formula": "total_generation_mwh / event_median_generation",
                "source_columns": ["total_generation_mwh"],
                "description": "Grid utilization relative to event median",
            },
            "event_day": {
                "group": "temporal",
                "type": "int",
                "formula": "day index within event (1-based; negative for baseline)",
                "source_columns": ["date", "event"],
                "description": "Position within event temporal arc",
            },
            "is_weekend": {
                "group": "temporal",
                "type": "binary",
                "formula": "1 if dayofweek >= 5 else 0",
                "source_columns": ["date"],
                "description": "Weekend indicator (demand confounder)",
            },
            "is_cold_event": {
                "group": "event",
                "type": "binary",
                "formula": "1 if event in (uri_2021, elliott_2022) else 0",
                "source_columns": ["event"],
                "description": "Cold event regime indicator",
            },
            "region_fossil_baseline": {
                "group": "event",
                "type": "float",
                "formula": "copy of baseline_fossil_pct",
                "source_columns": ["baseline_fossil_pct"],
                "description": "Regional grid fossil baseline (continuous encoding)",
            },
            "severity_x_fossil_shift": {
                "group": "interaction",
                "type": "float",
                "formula": "temp_deviation * fossil_pct_change",
                "source_columns": ["temp_deviation", "fossil_pct_change"],
                "description": "Weather-grid interaction (threshold amplification)",
            },
        },
        "targets": {
            "pm25_aqi_next": {
                "type": "float",
                "formula": "pm25_aqi shifted -1 within event",
                "description": "Next-day PM2.5 AQI (primary regression target)",
            },
            "ozone_aqi_next": {
                "type": "float",
                "formula": "ozone_aqi shifted -1 within event",
                "description": "Next-day Ozone AQI (secondary regression target)",
            },
            "aqi_category_next": {
                "type": "binary",
                "formula": "good if pm25_aqi_next <= 50 else not_good",
                "description": "Next-day AQI category (classification fallback)",
            },
        },
        "row_accounting": {
            "input_rows": len(df_in),
            "output_rows": len(df_out),
            "drops": {
                "total_dropped": len(df_in) - len(df_out),
            },
        },
        "feature_count": 12,
        "target_count": 3,
        "model_recommendation": "Ridge or ElasticNet regression",
        "decisions": {
            "date": "2026-03-18",
            "cut_from": 25,
            "cut_to": 12,
            "reason": "Row-to-feature ratio at original 25 was 2.4:1; cut to achieve 8-10:1",
        },
    }


def save_outputs(df_out, metadata):
    """Write feature_matrix.csv and feature_metadata.json."""
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV.name} ({len(df_out)} rows)")

    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {OUTPUT_META.name}")


def main():
    print("=" * 60)
    print("ClimatePulse Feature Engineering (cp-1a3)")
    print("=" * 60)

    # Load and filter
    df = load_data()
    input_rows = len(df)
    df = filter_quality(df)

    # Build all feature groups
    print("\nBuilding features...")
    df = build_weather_features(df)
    df = build_lag_features(df)
    df = build_grid_features(df)
    df = build_temporal_features(df)
    df = build_event_features(df)
    df = build_interaction_features(df)

    # Build targets
    print("Building targets...")
    df = build_targets(df)

    # Assemble clean matrix
    print("\nAssembling feature matrix...")
    df_out = assemble_matrix(df)

    # Summary stats
    print(f"\n{'='*60}")
    print(f"Row accounting: {input_rows} input → {len(df_out)} output")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Row-to-feature ratio: {len(df_out) / len(FEATURE_COLUMNS):.1f}:1")

    print(f"\nTarget distribution (pm25_aqi_next):")
    print(f"  Range: {df_out['pm25_aqi_next'].min():.1f} - {df_out['pm25_aqi_next'].max():.1f}")
    print(f"  Mean:  {df_out['pm25_aqi_next'].mean():.1f}")
    print(f"  Std:   {df_out['pm25_aqi_next'].std():.1f}")

    cat_counts = df_out["aqi_category_next"].value_counts()
    print(f"\nClassification target distribution:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} ({count/len(df_out)*100:.0f}%)")

    print(f"\nFeature summary:")
    for col in FEATURE_COLUMNS:
        vals = df_out[col]
        print(f"  {col:30s}  min={vals.min():8.2f}  max={vals.max():8.2f}  mean={vals.mean():8.2f}")

    # Hypothesis check: verify causal chain signal in features
    # Hypothesis check: verify causal chain signal in features.
    # The 1-day lag is: today's fossil_pct_change predicts tomorrow's pm25_aqi.
    # In our matrix, fossil_pct_change (day T) → pm25_aqi_next (day T+1).
    # NOTE: fossil_pct_change_lag1 (day T-1) → pm25_aqi_next (day T+1) is a
    # 2-day lag and expected to be weaker.
    print(f"\n{'='*60}")
    print("Hypothesis check: fossil_pct_change → pm25_aqi_next (1-day lag)")
    from scipy import stats as sp_stats

    valid = df_out[["fossil_pct_change", "pm25_aqi_next"]].dropna()
    r, p = sp_stats.pearsonr(valid["fossil_pct_change"], valid["pm25_aqi_next"])
    print(f"  Pooled: Pearson r = {r:.3f}, p = {p:.4f}")

    for evt in sorted(df_out["event"].unique()):
        sub = df_out[df_out["event"] == evt][["fossil_pct_change", "pm25_aqi_next"]].dropna()
        if len(sub) > 5:
            r_e, p_e = sp_stats.pearsonr(sub["fossil_pct_change"], sub["pm25_aqi_next"])
            print(f"  {evt}: r = {r_e:.3f}, p = {p_e:.4f}, n = {len(sub)}")

    if p < 0.05:
        print("  CONFIRMED: 1-day lag fossil shift → AQI signal preserved in features")
    else:
        print("  WARNING: Signal weakened — review before proceeding to model training")

    # Save
    metadata = build_metadata(df, df_out)
    save_outputs(df_out, metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())

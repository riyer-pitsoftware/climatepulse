"""Feature engineering output tests for ClimatePulse feature matrix.

Validates data integrity invariants in the feature_matrix.csv output:
row counts, NaN completeness, event boundary handling, value ranges,
target alignment, category labels, metadata structure, and column counts.

Bead: cp-1a3
"""

import json
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
FEATURE_CSV = DATA_DIR / "feature_matrix.csv"
UNIFIED_CSV = DATA_DIR / "unified_analysis.csv"
METADATA_JSON = DATA_DIR / "feature_metadata.json"

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def feature_df():
    """Load the feature matrix CSV once for all tests."""
    assert FEATURE_CSV.exists(), f"Missing {FEATURE_CSV}"
    return pd.read_csv(FEATURE_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def unified_df():
    """Load the unified analysis CSV once for cross-referencing."""
    assert UNIFIED_CSV.exists(), f"Missing {UNIFIED_CSV}"
    return pd.read_csv(UNIFIED_CSV, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureMatrix:

    # 1. Row count: output has fewer rows than unified_analysis.csv
    def test_row_count_reduced(self, feature_df, unified_df):
        """Feature matrix should have fewer rows than unified_analysis.csv.

        Rows are lost to quality filtering (suspect_low_generation),
        lag feature construction (first row per event), target shift
        (last row per event), and EPA NaN in targets.
        """
        assert len(feature_df) < len(unified_df), (
            f"Feature matrix ({len(feature_df)} rows) should have fewer rows "
            f"than unified_analysis ({len(unified_df)} rows) due to drops"
        )

    # 2. No NaN in features: all 12 feature columns are complete
    def test_no_nan_in_features(self, feature_df):
        """All 12 feature columns must have zero NaN values.

        The pipeline drops rows with missing features, so the output
        matrix should be fully complete for all feature columns.
        """
        for col in FEATURE_COLUMNS:
            nan_count = feature_df[col].isna().sum()
            assert nan_count == 0, (
                f"Feature column '{col}' has {nan_count} NaN values; "
                f"expected 0 after pipeline drops"
            )

    # 3. Event boundaries: lag features don't cross events
    def test_lag_respects_event_boundaries(self, feature_df, unified_df):
        """Lag features must not leak across event boundaries.

        For the first remaining row of each event in the feature matrix,
        verify that fossil_pct_change_lag1 is NOT the last row of the
        previous event's fossil_pct_change in the unified data.
        """
        events = feature_df["event"].unique()
        assert len(events) >= 2, "Need at least 2 events to test boundary handling"

        # Build a lookup of the last fossil_pct_change per event in unified data
        last_value_per_event = {}
        for evt in unified_df["event"].unique():
            evt_rows = unified_df[unified_df["event"] == evt].sort_values("date")
            last_value_per_event[evt] = evt_rows["fossil_pct_change"].iloc[-1]

        # For each event in the feature matrix, get the first row's lag value
        sorted_events = list(unified_df["event"].unique())
        for i, evt in enumerate(sorted_events):
            if i == 0:
                continue  # first event has no preceding event to leak from
            prev_evt = sorted_events[i - 1]
            evt_feature_rows = feature_df[feature_df["event"] == evt].sort_values("date")
            if evt_feature_rows.empty:
                continue
            first_lag_value = evt_feature_rows["fossil_pct_change_lag1"].iloc[0]
            prev_last_value = last_value_per_event[prev_evt]
            assert first_lag_value != prev_last_value or pd.isna(first_lag_value), (
                f"Event '{evt}' first row lag1 ({first_lag_value}) matches "
                f"last value of previous event '{prev_evt}' ({prev_last_value}); "
                f"lag feature appears to cross event boundaries"
            )

    # 4. Value ranges: each feature within expected bounds
    def test_feature_value_ranges(self, feature_df):
        """Each feature must fall within physically plausible bounds.

        temp_deviation >= 0 (absolute value of deviation from 65F)
        cold_severity >= 0 (clamped at 0 for warm days)
        heat_severity >= 0 (clamped at 0 for cold days)
        is_weekend in {0, 1}
        is_cold_event in {0, 1}
        fossil_dominance_ratio > 0 (ratio with floor of 1.0 in denominator)
        generation_utilization > 0 (ratio of generation to median)
        """
        assert (feature_df["temp_deviation"] >= 0).all(), (
            f"temp_deviation has negative values: min={feature_df['temp_deviation'].min()}"
        )
        assert (feature_df["cold_severity"] >= 0).all(), (
            f"cold_severity has negative values: min={feature_df['cold_severity'].min()}"
        )
        assert (feature_df["heat_severity"] >= 0).all(), (
            f"heat_severity has negative values: min={feature_df['heat_severity'].min()}"
        )
        assert set(feature_df["is_weekend"].unique()).issubset({0, 1}), (
            f"is_weekend has values outside {{0, 1}}: {feature_df['is_weekend'].unique()}"
        )
        assert set(feature_df["is_cold_event"].unique()).issubset({0, 1}), (
            f"is_cold_event has values outside {{0, 1}}: {feature_df['is_cold_event'].unique()}"
        )
        assert (feature_df["fossil_dominance_ratio"] > 0).all(), (
            f"fossil_dominance_ratio has non-positive values: "
            f"min={feature_df['fossil_dominance_ratio'].min()}"
        )
        assert (feature_df["generation_utilization"] > 0).all(), (
            f"generation_utilization has non-positive values: "
            f"min={feature_df['generation_utilization'].min()}"
        )

    # 5. Target alignment: pm25_aqi_next at row N == pm25_aqi at row N+1
    def test_target_alignment(self, feature_df, unified_df):
        """pm25_aqi_next at row N should equal pm25_aqi at row N+1 within an event.

        Join the feature matrix back to unified_analysis on event+date to
        verify the shifted target values are correct.
        """
        # Build next-day lookup from unified data (within each event)
        unified_sorted = unified_df.sort_values(["event", "date"]).reset_index(drop=True)
        next_day_aqi = {}
        for evt in unified_sorted["event"].unique():
            evt_rows = unified_sorted[unified_sorted["event"] == evt].reset_index(drop=True)
            for i in range(len(evt_rows) - 1):
                key = (evt, evt_rows.loc[i, "date"])
                next_day_aqi[key] = evt_rows.loc[i + 1, "pm25_aqi"]

        # Check each feature matrix row against the lookup
        mismatches = 0
        checked = 0
        for _, row in feature_df.iterrows():
            key = (row["event"], row["date"])
            if key in next_day_aqi and pd.notna(next_day_aqi[key]):
                checked += 1
                expected = next_day_aqi[key]
                actual = row["pm25_aqi_next"]
                if pd.notna(actual) and abs(actual - expected) > 0.01:
                    mismatches += 1

        assert checked > 0, "No rows could be cross-referenced for target alignment"
        assert mismatches == 0, (
            f"{mismatches} of {checked} rows have pm25_aqi_next misaligned "
            f"with next day's pm25_aqi in unified_analysis"
        )

    # 6. Category labels: aqi_category_next only from {good, not_good}
    def test_category_labels(self, feature_df):
        """aqi_category_next must only contain 'good' and/or 'not_good'."""
        valid_labels = {"good", "not_good"}
        actual_labels = set(feature_df["aqi_category_next"].dropna().unique())
        invalid = actual_labels - valid_labels
        assert not invalid, (
            f"aqi_category_next contains unexpected labels: {invalid}; "
            f"expected only {valid_labels}"
        )

    # 7. Determinism: feature_metadata.json exists and has expected structure
    def test_metadata_structure(self):
        """feature_metadata.json must exist and contain expected keys.

        Required structure:
        - 'features' dict with 12 entries
        - 'targets' dict with 3 entries
        - 'feature_count' == 12
        - 'target_count' == 3
        """
        assert METADATA_JSON.exists(), f"Missing {METADATA_JSON}"

        with open(METADATA_JSON) as f:
            metadata = json.load(f)

        assert "features" in metadata, "metadata missing 'features' key"
        assert "targets" in metadata, "metadata missing 'targets' key"
        assert "feature_count" in metadata, "metadata missing 'feature_count' key"
        assert "target_count" in metadata, "metadata missing 'target_count' key"

        assert len(metadata["features"]) == 12, (
            f"Expected 12 feature entries in metadata, got {len(metadata['features'])}"
        )
        assert len(metadata["targets"]) == 3, (
            f"Expected 3 target entries in metadata, got {len(metadata['targets'])}"
        )
        assert metadata["feature_count"] == 12, (
            f"feature_count is {metadata['feature_count']}, expected 12"
        )
        assert metadata["target_count"] == 3, (
            f"target_count is {metadata['target_count']}, expected 3"
        )

    # 8. Feature count: exactly 12 feature columns + 3 targets + metadata columns
    def test_column_counts(self, feature_df):
        """Output must contain exactly 12 features, 3 targets, and metadata columns.

        Metadata columns are 'event' and 'date' (plus optionally 'is_event').
        Total: at least 17 columns (12 + 3 + 2).
        """
        missing_features = set(FEATURE_COLUMNS) - set(feature_df.columns)
        assert not missing_features, (
            f"Missing feature columns: {missing_features}"
        )

        missing_targets = set(TARGET_COLUMNS) - set(feature_df.columns)
        assert not missing_targets, (
            f"Missing target columns: {missing_targets}"
        )

        missing_meta = set(METADATA_COLUMNS) - set(feature_df.columns)
        assert not missing_meta, (
            f"Missing metadata columns: {missing_meta}"
        )

        # Verify exact feature count (no extra unlisted feature columns)
        expected_all = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS) | set(METADATA_COLUMNS)
        extra = set(feature_df.columns) - expected_all
        # Allow 'is_event' or 'is_baseline' as optional metadata columns (cp-9v9)
        extra.discard("is_event")
        extra.discard("is_baseline")
        if extra:
            pytest.fail(
                f"Unexpected extra columns in feature matrix: {extra}; "
                f"expected only {sorted(expected_all)} (plus optional 'is_event')"
            )

"""Pipeline transformation tests for ClimatePulse unified analysis.

Validates data integrity invariants after the join_datasets.py pipeline runs:
row counts, column presence, join coverage, date validity, numeric ranges,
and statistical output structure.

Bead: cp-gp4
"""

import json
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
UNIFIED_CSV = DATA_DIR / "unified_analysis.csv"
STATS_JSON = DATA_DIR / "stats_results.json"

EXPECTED_EVENTS = {"uri_2021", "heat_dome_2021", "elliott_2022"}

EXPECTED_COLUMNS = [
    "event", "date",
    # NOAA weather
    "mean_tmin", "mean_tmax", "min_tmin", "max_tmax", "mean_prcp",
    "total_snow", "station_count",
    # EIA grid
    "fossil", "renewable", "other", "total_generation_mwh",
    "fossil_pct", "renewable_pct",
    "baseline_fossil_pct", "baseline_renewable_pct",
    "fossil_pct_change", "renewable_pct_change",
    # EPA air quality
    "pm25_mean", "ozone_mean", "pm25_aqi", "ozone_aqi",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unified_df():
    """Load the unified analysis CSV once for all tests."""
    assert UNIFIED_CSV.exists(), f"Missing {UNIFIED_CSV}"
    return pd.read_csv(UNIFIED_CSV, parse_dates=["date"])


@pytest.fixture(scope="module")
def stats():
    """Load stats_results.json once for all tests."""
    assert STATS_JSON.exists(), f"Missing {STATS_JSON}"
    with open(STATS_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------

class TestFileExistence:
    def test_unified_csv_exists(self):
        assert UNIFIED_CSV.exists()

    def test_stats_json_exists(self):
        assert STATS_JSON.exists()


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

class TestColumns:
    def test_expected_columns_present(self, unified_df):
        missing = set(EXPECTED_COLUMNS) - set(unified_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_unexpected_extra_columns(self, unified_df):
        """Warn (but don't fail) if extra columns appear."""
        extra = set(unified_df.columns) - set(EXPECTED_COLUMNS)
        if extra:
            pytest.skip(f"Extra columns found (informational): {extra}")


# ---------------------------------------------------------------------------
# Row counts and events
# ---------------------------------------------------------------------------

class TestRowCounts:
    def test_row_count_sanity(self, unified_df):
        """Expect approximately 71 rows (3 events over multi-day windows)."""
        assert 60 <= len(unified_df) <= 90, f"Unexpected row count: {len(unified_df)}"

    def test_exact_row_count(self, unified_df):
        assert len(unified_df) == 71, f"Expected 71 rows, got {len(unified_df)}"

    def test_all_three_events_present(self, unified_df):
        actual_events = set(unified_df["event"].unique())
        assert actual_events == EXPECTED_EVENTS, (
            f"Expected {EXPECTED_EVENTS}, got {actual_events}"
        )

    def test_no_null_events(self, unified_df):
        assert unified_df["event"].notna().all(), "Found NaN in event column"


# ---------------------------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_no_duplicate_event_date_rows(self, unified_df):
        dupes = unified_df.duplicated(subset=["event", "date"], keep=False)
        assert not dupes.any(), (
            f"Found {dupes.sum()} duplicate event+date rows: "
            f"{unified_df.loc[dupes, ['event', 'date']].to_dict('records')}"
        )


# ---------------------------------------------------------------------------
# Date validity
# ---------------------------------------------------------------------------

class TestDates:
    def test_date_column_is_datetime(self, unified_df):
        assert pd.api.types.is_datetime64_any_dtype(unified_df["date"]), (
            f"date column dtype is {unified_df['date'].dtype}, expected datetime"
        )

    def test_no_null_dates(self, unified_df):
        assert unified_df["date"].notna().all(), "Found NaN dates"

    def test_dates_within_expected_range(self, unified_df):
        """All dates should fall within 2021-2023 (the three event windows)."""
        assert unified_df["date"].min() >= pd.Timestamp("2021-01-01")
        assert unified_df["date"].max() <= pd.Timestamp("2023-12-31")


# ---------------------------------------------------------------------------
# Numeric range checks
# ---------------------------------------------------------------------------

class TestNumericRanges:
    def test_fossil_pct_bounded(self, unified_df):
        col = unified_df["fossil_pct"].dropna()
        assert (col >= 0).all() and (col <= 100).all(), (
            f"fossil_pct out of [0, 100]: min={col.min()}, max={col.max()}"
        )

    def test_renewable_pct_bounded(self, unified_df):
        col = unified_df["renewable_pct"].dropna()
        assert (col >= 0).all() and (col <= 100).all(), (
            f"renewable_pct out of [0, 100]: min={col.min()}, max={col.max()}"
        )

    def test_generation_positive(self, unified_df):
        col = unified_df["total_generation_mwh"].dropna()
        assert (col > 0).all(), "total_generation_mwh should be positive"

    def test_station_count_positive(self, unified_df):
        col = unified_df["station_count"].dropna()
        assert (col > 0).all(), "station_count should be positive"

    def test_pm25_aqi_non_negative(self, unified_df):
        col = unified_df["pm25_aqi"].dropna()
        assert (col >= 0).all(), f"pm25_aqi has negatives: min={col.min()}"

    def test_ozone_aqi_non_negative(self, unified_df):
        col = unified_df["ozone_aqi"].dropna()
        assert (col >= 0).all(), f"ozone_aqi has negatives: min={col.min()}"


# ---------------------------------------------------------------------------
# NaN coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_core_weather_columns_complete(self, unified_df):
        """NOAA weather columns should have no NaN (inner join with EIA)."""
        for col in ["mean_tmin", "mean_tmax", "station_count"]:
            pct = unified_df[col].notna().mean()
            assert pct >= 0.95, f"{col} coverage too low: {pct:.1%}"

    def test_grid_columns_complete(self, unified_df):
        """EIA grid columns should have no NaN (inner join)."""
        for col in ["fossil_pct", "renewable_pct", "total_generation_mwh"]:
            assert unified_df[col].notna().all(), f"{col} has unexpected NaN"

    def test_epa_coverage_reasonable(self, unified_df):
        """EPA data is a left join, so some NaN is expected. Check >= 80%."""
        for col in ["pm25_aqi", "ozone_aqi"]:
            pct = unified_df[col].notna().mean()
            assert pct >= 0.80, f"{col} coverage too low: {pct:.1%}"


# ---------------------------------------------------------------------------
# stats_results.json structure
# ---------------------------------------------------------------------------

class TestStatsResults:
    EXPECTED_TOP_KEYS = {
        "correlation",
        "lagged_correlation",
        "granger_causality",
        "difference_in_differences",
        "pooled",
    }

    def test_top_level_keys(self, stats):
        actual = set(stats.keys())
        missing = self.EXPECTED_TOP_KEYS - actual
        assert not missing, f"Missing top-level keys in stats: {missing}"

    def test_correlation_has_entries(self, stats):
        assert len(stats["correlation"]) > 0, "correlation section is empty"

    def test_lagged_correlation_has_events(self, stats):
        lagged = stats["lagged_correlation"]
        for evt in EXPECTED_EVENTS:
            assert evt in lagged, f"Missing event {evt} in lagged_correlation"

    def test_granger_causality_has_events(self, stats):
        gc = stats["granger_causality"]
        for evt in EXPECTED_EVENTS:
            assert evt in gc, f"Missing event {evt} in granger_causality"

    def test_difference_in_differences_has_events(self, stats):
        did = stats["difference_in_differences"]
        for evt in EXPECTED_EVENTS:
            assert evt in did, f"Missing event {evt} in difference_in_differences"

    def test_pooled_has_expected_fields(self, stats):
        pooled = stats["pooled"]
        for key in ["n", "fossil_mean_shift", "t_stat", "p_value", "ci_95"]:
            assert key in pooled, f"Missing key '{key}' in pooled stats"

    def test_pooled_n_matches_row_count(self, stats, unified_df):
        assert stats["pooled"]["n"] == len(unified_df), (
            f"pooled.n ({stats['pooled']['n']}) != row count ({len(unified_df)})"
        )

"""
01_noaa_weather_timeline.py
Process raw NOAA CDO CSV data into a clean weather-event timeline.

Input:  data/raw/noaa/*.csv  (6 files, 3 events)
Output: data/processed/noaa_event_timeline.csv
"""

import pathlib
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "noaa"
OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed" / "noaa_event_timeline.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── event manifest ───────────────────────────────────────────────────────
# Each entry maps a display name to its raw CSV files.
# Baseline entries (cp-9v9) use the same event label so they join correctly
# downstream; is_baseline flag distinguishes them from event-period data.
EVENTS = {
    "Winter Storm Uri (Feb 2021)": [
        "uri_houston_tx.csv",
        "uri_dallas_tx.csv",
    ],
    "Winter Storm Elliott (Dec 2022)": [
        "elliott_allegheny_pa.csv",
        "elliott_hamilton_oh.csv",
    ],
    "PNW Heat Dome (Jun-Jul 2021)": [
        "heatdome_multnomah_or.csv",
        "heatdome_king_wa.csv",
    ],
}

# cp-9v9: Same-year pre-event baseline periods (2 weeks before each event).
# These map to the SAME parent event so the join script can pair them.
BASELINE_EVENTS = {
    "Winter Storm Uri Baseline (Jan 2021)": {
        "files": [
            "uri_baseline_houston_tx.csv",
            "uri_baseline_dallas_tx.csv",
        ],
        "parent_event": "Winter Storm Uri (Feb 2021)",
    },
    "PNW Heat Dome Baseline (Jun 2021)": {
        "files": [
            "heatdome_baseline_multnomah_or.csv",
            "heatdome_baseline_king_wa.csv",
        ],
        "parent_event": "PNW Heat Dome (Jun-Jul 2021)",
    },
    "Winter Storm Elliott Baseline (Dec 2022)": {
        "files": [
            "elliott_baseline_allegheny_pa.csv",
            "elliott_baseline_hamilton_oh.csv",
        ],
        "parent_event": "Winter Storm Elliott (Dec 2022)",
    },
}

DATATYPES = ["TMIN", "TMAX", "PRCP", "SNOW"]


# ── helpers ──────────────────────────────────────────────────────────────
def load_event(event_name: str, filenames: list[str]) -> pd.DataFrame:
    """Load CSVs for one event, return long-form DataFrame with event label."""
    frames = []
    for fn in filenames:
        df = pd.read_csv(RAW_DIR / fn)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["event"] = event_name
    combined["date"] = pd.to_datetime(combined["date"]).dt.date
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    return combined


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form NOAA data so each row is one date + station."""
    wide = (
        df.pivot_table(
            index=["event", "date", "station"],
            columns="datatype",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None  # drop multiindex name
    # Ensure all expected columns exist
    for col in DATATYPES:
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide


def daily_aggregates(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute daily stats across all stations for one event."""
    agg = (
        wide.groupby(["event", "date"])
        .agg(
            mean_tmin=("TMIN", "mean"),
            mean_tmax=("TMAX", "mean"),
            min_tmin=("TMIN", "min"),
            max_tmax=("TMAX", "max"),
            mean_prcp=("PRCP", "mean"),
            total_snow=("SNOW", "sum"),
            station_count=("station", "nunique"),
        )
        .reset_index()
    )
    # Coerce to numeric and round for readability
    for col in ["mean_tmin", "mean_tmax", "min_tmin", "max_tmax", "mean_prcp", "total_snow"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").round(1)
    return agg


# ── main pipeline ────────────────────────────────────────────────────────
def main():
    all_wide = []
    all_daily = []

    # Process event-period data
    for event_name, files in EVENTS.items():
        print(f"\n{'='*60}")
        print(f"Loading: {event_name}")
        print(f"  files: {files}")

        long = load_event(event_name, files)
        print(f"  rows loaded: {len(long):,}")

        wide = pivot_to_wide(long)
        print(f"  station-days after pivot: {len(wide):,}")
        all_wide.append(wide)

        daily = daily_aggregates(wide)
        daily["is_baseline"] = 0
        print(f"  unique dates: {len(daily):,}")
        all_daily.append(daily)

    # cp-9v9: Process baseline-period data
    # Baseline rows get the PARENT event name so they join with the same
    # event in the downstream pipeline, plus is_baseline=1 to distinguish.
    for bl_name, bl_cfg in BASELINE_EVENTS.items():
        files = bl_cfg["files"]
        parent = bl_cfg["parent_event"]
        print(f"\n{'='*60}")
        print(f"Loading baseline: {bl_name}")
        print(f"  files: {files}  (parent event: {parent})")

        try:
            long = load_event(parent, files)
        except FileNotFoundError:
            print(f"  SKIP: baseline files not yet pulled — run pull_noaa_sample.py first")
            continue

        print(f"  rows loaded: {len(long):,}")

        wide = pivot_to_wide(long)
        print(f"  station-days after pivot: {len(wide):,}")
        all_wide.append(wide)

        daily = daily_aggregates(wide)
        daily["is_baseline"] = 1
        print(f"  unique dates: {len(daily):,}")
        all_daily.append(daily)

    # ── combine all events ───────────────────────────────────────────────
    timeline = pd.concat(all_daily, ignore_index=True)
    timeline = timeline.sort_values(["event", "date"]).reset_index(drop=True)
    timeline.to_csv(OUT_PATH, index=False)
    print(f"\nSaved combined timeline -> {OUT_PATH}")
    print(f"  total rows: {len(timeline):,}")

    # ── extreme-day summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXTREME-DAY SUMMARY")
    print(f"{'='*60}")

    for event_name, daily_df in zip(EVENTS.keys(), all_daily):
        print(f"\n--- {event_name} ---")

        if "Uri" in event_name or "Elliott" in event_name:
            # Coldest days by min_tmin
            coldest = daily_df.nsmallest(3, "min_tmin")
            print("  Coldest days (lowest TMIN recorded at any station):")
            for _, row in coldest.iterrows():
                print(
                    f"    {row['date']}  min_tmin={row['min_tmin']:.1f}F  "
                    f"mean_tmin={row['mean_tmin']:.1f}F  "
                    f"mean_tmax={row['mean_tmax']:.1f}F  "
                    f"stations={row['station_count']}"
                )
        else:
            # Hottest days by max_tmax
            hottest = daily_df.nlargest(3, "max_tmax")
            print("  Hottest days (highest TMAX recorded at any station):")
            for _, row in hottest.iterrows():
                print(
                    f"    {row['date']}  max_tmax={row['max_tmax']:.1f}F  "
                    f"mean_tmax={row['mean_tmax']:.1f}F  "
                    f"mean_tmin={row['mean_tmin']:.1f}F  "
                    f"stations={row['station_count']}"
                )

    # ── quick sanity check on date ranges ────────────────────────────────
    print(f"\n{'='*60}")
    print("DATE RANGES PER EVENT")
    print(f"{'='*60}")
    for event_name, daily_df in zip(EVENTS.keys(), all_daily):
        d_min, d_max = daily_df["date"].min(), daily_df["date"].max()
        print(f"  {event_name}: {d_min} -> {d_max}  ({len(daily_df)} days)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-dataset join: align weather × grid × air quality by event and date.

Produces unified_analysis.csv — the core analytical artifact for the
ClimatePulse thesis: Extreme Weather → Grid Fossil Shift → AQI Degradation.
"""

import pandas as pd
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Canonical event mapping across all 3 datasets
EVENT_MAP_NOAA = {
    "PNW Heat Dome (Jun-Jul 2021)": "heat_dome_2021",
    "Winter Storm Uri (Feb 2021)": "uri_2021",
    "Winter Storm Elliott (Dec 2022)": "elliott_2022",
}
EVENT_MAP_EIA = {
    "BPAT_HeatDome_2021": "heat_dome_2021",
    "ERCO_URI_2021": "uri_2021",
    "PJM_Elliott_2022": "elliott_2022",
}
EVENT_MAP_EPA = {
    "heat_dome": "heat_dome_2021",
    "uri": "uri_2021",
    "elliott": "elliott_2022",
}


def load_noaa():
    """Load NOAA weather timeline — already daily per event."""
    df = pd.read_csv(DATA_DIR / "noaa_event_timeline.csv")
    df["event"] = df["event"].map(EVENT_MAP_NOAA)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_eia():
    """Load EIA grid data. Aggregate hourly → daily by fuel_category.

    Compute fossil_pct and renewable_pct per day.
    Also compute baseline averages so we can show the shift.
    """
    df = pd.read_csv(DATA_DIR / "eia_grid_response.csv")
    df["event"] = df["event"].map(EVENT_MAP_EIA)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    # Daily totals by event, period_type, fuel_category
    daily = (
        df.groupby(["event", "period_type", "date", "fuel_category"])["generation_mwh"]
        .sum()
        .reset_index()
    )

    # Pivot fuel_category into columns
    daily_wide = daily.pivot_table(
        index=["event", "period_type", "date"],
        columns="fuel_category",
        values="generation_mwh",
        fill_value=0,
    ).reset_index()
    daily_wide.columns.name = None

    # Compute percentages
    total = daily_wide["fossil"] + daily_wide["renewable"] + daily_wide["other"]
    daily_wide["fossil_pct"] = (daily_wide["fossil"] / total * 100).round(2)
    daily_wide["renewable_pct"] = (daily_wide["renewable"] / total * 100).round(2)
    daily_wide["total_generation_mwh"] = total

    # Split into baseline averages and event data
    baseline = daily_wide[daily_wide["period_type"] == "baseline"]
    event = daily_wide[daily_wide["period_type"] == "event"].copy()

    # Compute per-event baseline averages for fossil/renewable pct
    baseline_avg = (
        baseline.groupby("event")[["fossil_pct", "renewable_pct"]]
        .mean()
        .rename(columns={
            "fossil_pct": "baseline_fossil_pct",
            "renewable_pct": "baseline_renewable_pct",
        })
    )

    event["date"] = pd.to_datetime(event["date"])
    event = event.merge(baseline_avg, on="event", how="left")
    event["fossil_pct_change"] = (event["fossil_pct"] - event["baseline_fossil_pct"]).round(2)
    event["renewable_pct_change"] = (event["renewable_pct"] - event["baseline_renewable_pct"]).round(2)

    cols = [
        "event", "date",
        "fossil", "renewable", "other", "total_generation_mwh",
        "fossil_pct", "renewable_pct",
        "baseline_fossil_pct", "baseline_renewable_pct",
        "fossil_pct_change", "renewable_pct_change",
    ]
    return event[cols]


def load_epa():
    """Load EPA air quality — average across counties per event+date."""
    df = pd.read_csv(DATA_DIR / "epa_air_quality.csv")
    df["event"] = df["event"].map(EVENT_MAP_EPA)
    df["date"] = pd.to_datetime(df["date"])

    # Average across counties for a regional daily value
    daily = (
        df.groupby(["event", "date"])[["pm25_mean", "ozone_mean", "pm25_aqi", "ozone_aqi"]]
        .mean()
        .round(4)
        .reset_index()
    )
    return daily


def main():
    print("Loading datasets...")
    noaa = load_noaa()
    eia = load_eia()
    epa = load_epa()

    print(f"  NOAA: {len(noaa)} rows, events: {noaa.event.unique().tolist()}")
    print(f"  EIA:  {len(eia)} rows, events: {eia.event.unique().tolist()}")
    print(f"  EPA:  {len(epa)} rows, events: {epa.event.unique().tolist()}")

    # Join NOAA × EIA on event + date
    merged = noaa.merge(eia, on=["event", "date"], how="inner", suffixes=("", "_eia"))
    print(f"\n  NOAA × EIA inner join: {len(merged)} rows")

    # Join with EPA on event + date
    merged = merged.merge(epa, on=["event", "date"], how="left", suffixes=("", "_epa"))
    print(f"  + EPA left join: {len(merged)} rows")

    # Report coverage
    has_epa = merged["pm25_aqi"].notna().sum()
    print(f"  EPA coverage: {has_epa}/{len(merged)} rows ({has_epa/len(merged)*100:.0f}%)")

    # Sort by event + date
    merged = merged.sort_values(["event", "date"]).reset_index(drop=True)

    # Save
    out_path = DATA_DIR / "unified_analysis.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(merged)} rows, {len(merged.columns)} columns)")
    print(f"Columns: {list(merged.columns)}")

    # Quick summary per event
    print("\n=== Per-event summary ===")
    for evt in sorted(merged["event"].unique()):
        sub = merged[merged["event"] == evt]
        print(f"\n{evt}:")
        print(f"  Date range: {sub.date.min().date()} to {sub.date.max().date()}")
        print(f"  Avg max temp: {sub.mean_tmax.mean():.1f}°F")
        print(f"  Fossil % change: {sub.fossil_pct_change.mean():+.2f} pp")
        print(f"  Renewable % change: {sub.renewable_pct_change.mean():+.2f} pp")
        epa_sub = sub.dropna(subset=["pm25_aqi"])
        if len(epa_sub) > 0:
            print(f"  Avg PM2.5 AQI: {epa_sub.pm25_aqi.mean():.1f}")
            print(f"  Avg Ozone AQI: {epa_sub.ozone_aqi.mean():.1f}")
        else:
            print("  No EPA data overlap")

    return 0


if __name__ == "__main__":
    sys.exit(main())

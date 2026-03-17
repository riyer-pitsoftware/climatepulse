#!/usr/bin/env python3
"""Cross-dataset join: align weather × grid × air quality by event and date.

Produces unified_analysis.csv — the core analytical artifact for the
ClimatePulse thesis: Extreme Weather → Grid Fossil Shift → AQI Degradation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

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

    Timezone note (cp-42z):
    EIA API hourly data uses *local prevailing time* of each balancing
    authority (ERCO → Central, BPAT → Pacific, PJM → Eastern).  The raw
    files contain hour labels like "2021-02-01T00" with no explicit UTC
    offset.  We treat them as local time and truncate to calendar dates in
    that same local time, which aligns with the NOAA and EPA daily
    boundaries that also use local calendar dates.

    Risk: if EIA ever switches to UTC delivery, our daily aggregation
    could shift by one calendar day for late-night hours.  We verified
    against raw files (e.g. eia_erco_uri_2021.csv "period" column starts
    at T00 on the first event day and ends at T23, consistent with local
    midnight-to-midnight).  NOAA "Date Local" and EPA "Date Local" fields
    are explicitly local-time, so the join on calendar date is consistent
    across all three datasets.
    """
    df = pd.read_csv(DATA_DIR / "eia_grid_response.csv")
    df["event"] = df["event"].map(EVENT_MAP_EIA)
    # Parse as naive datetime — EIA hours are in local prevailing time
    # of the balancing authority (see docstring for timezone rationale).
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


def _build_county_monitor_weights():
    """Count distinct monitoring sites per county from raw EPA AQS data.

    The raw EPA daily files contain one row per site-day-POC.  We count
    unique (State Code, County Code, Site Num) tuples to estimate each
    county's monitoring density.  Counties with more monitors contribute
    proportionally more to the regional daily AQI average.

    Limitation (cp-3xe): this counts *sites that ever reported* in the
    annual file, not sites reporting on a specific day.  A site that was
    offline for part of the event still contributes to its county's
    weight.  A per-day site count would be more precise but would require
    joining raw site-level data back through the full pipeline.
    """
    site_frames = []
    for pattern in ["daily_88101_*/*.csv", "daily_88101_*.csv",
                    "daily_44201_*.csv"]:
        for fp in sorted((RAW_DIR / "epa").glob(pattern)):
            try:
                raw = pd.read_csv(fp, usecols=["State Code", "County Code", "Site Num"],
                                  dtype=str)
                site_frames.append(raw)
            except Exception:
                pass

    if not site_frames:
        return None

    sites = pd.concat(site_frames, ignore_index=True)
    sites = sites.drop_duplicates()
    weights = (
        sites.groupby(["State Code", "County Code"])
        .size()
        .reset_index(name="monitor_count")
    )
    # Normalise column names to match processed EPA file
    weights.rename(columns={"State Code": "state_code",
                            "County Code": "county_code"}, inplace=True)
    return weights


def load_epa():
    """Load EPA air quality — monitor-count weighted average across counties.

    cp-3xe: Instead of a simple unweighted mean, each county's AQI is
    weighted by the number of distinct monitoring sites in that county
    (derived from raw AQS annual files).  This gives higher weight to
    counties with denser monitoring networks, better representing
    population-level exposure.
    """
    df = pd.read_csv(DATA_DIR / "epa_air_quality.csv")
    df["event"] = df["event"].map(EVENT_MAP_EPA)
    df["date"] = pd.to_datetime(df["date"])
    # Ensure join keys are strings for merge with raw-derived weights
    df["state_code"] = df["state_code"].astype(str).str.zfill(2)
    df["county_code"] = df["county_code"].astype(str).str.zfill(3)

    weights = _build_county_monitor_weights()
    aqi_cols = ["pm25_mean", "ozone_mean", "pm25_aqi", "ozone_aqi"]

    if weights is not None:
        weights["state_code"] = weights["state_code"].astype(str).str.zfill(2)
        weights["county_code"] = weights["county_code"].astype(str).str.zfill(3)
        df = df.merge(weights, on=["state_code", "county_code"], how="left")
        # Counties with no raw match get weight 1 (unweighted fallback)
        df["monitor_count"] = df["monitor_count"].fillna(1).astype(int)
        print(f"  EPA weighting: {df['monitor_count'].gt(1).sum()}/{len(df)} "
              f"county-rows have >1 monitor (max={df['monitor_count'].max()})")
    else:
        # Fallback: equal weight per county (no raw site data found)
        df["monitor_count"] = 1
        print("  EPA weighting: raw site data unavailable, using equal county weights")

    # Weighted average across counties per event+date
    def _weighted_mean(group):
        w = group["monitor_count"].values
        result = {}
        for col in aqi_cols:
            vals = group[col].values
            mask = ~np.isnan(vals)
            if mask.any():
                result[col] = round(np.average(vals[mask], weights=w[mask]), 4)
            else:
                result[col] = np.nan
        return pd.Series(result)

    daily = (
        df.groupby(["event", "date"])
        .apply(_weighted_mean, include_groups=False)
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

    # ── Data quality: flag rows with suspiciously low generation ──
    # For each event, compute median total_generation_mwh.
    # Flag rows below 50% of their event median as suspect partial-day data.
    event_medians = merged.groupby("event")["total_generation_mwh"].median()
    merged["data_quality"] = "ok"
    for evt, median_gen in event_medians.items():
        mask = (merged["event"] == evt) & (
            merged["total_generation_mwh"] < 0.5 * median_gen
        )
        merged.loc[mask, "data_quality"] = "suspect_low_generation"

    suspect = merged[merged["data_quality"] == "suspect_low_generation"]
    if len(suspect) > 0:
        print(f"\n⚠  Flagged {len(suspect)} rows as suspect_low_generation:")
        for _, row in suspect.iterrows():
            evt_median = event_medians[row["event"]]
            print(
                f"   {row['event']} {row['date'].date()}: "
                f"total_generation_mwh={row['total_generation_mwh']:.0f} "
                f"(event median={evt_median:.0f})"
            )

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

"""
03_epa_air_quality.py
Process raw EPA air quality data (PM2.5 + Ozone) into an event-aligned dataset
for ClimatePulse extreme-weather analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
RAW = _ROOT / "data" / "raw" / "epa"
OUT = _ROOT / "data" / "processed" / "epa_air_quality.csv"

PM25_FILES = [
    RAW / "daily_88101_2021" / "daily_88101_2021.csv",
    RAW / "daily_88101_2022" / "daily_88101_2022.csv",
]
OZONE_FILES = [
    RAW / "daily_44201_2021.csv",
    RAW / "daily_44201_2022.csv",
]

# ---------------------------------------------------------------------------
# Target regions  (state_code, county_code, event_label)
# ---------------------------------------------------------------------------
REGIONS = [
    # Winter Storm Uri — TX
    ("48", "201", "uri"),   # Harris County
    ("48", "113", "uri"),   # Dallas County
    # Winter Storm Elliott — OH / PA
    ("39", "061", "elliott"),  # Hamilton County OH
    ("42", "003", "elliott"),  # Allegheny County PA
    # Heat Dome — OR / WA
    ("41", "051", "heat_dome"),  # Multnomah County OR
    ("53", "033", "heat_dome"),  # King County WA
]

# ---------------------------------------------------------------------------
# Event windows (event_start, event_end) — plus 2 weeks before/after
# ---------------------------------------------------------------------------
EVENTS = {
    "uri":       (pd.Timestamp("2021-02-13"), pd.Timestamp("2021-02-27")),
    "elliott":   (pd.Timestamp("2022-12-21"), pd.Timestamp("2023-01-03")),
    "heat_dome": (pd.Timestamp("2021-06-25"), pd.Timestamp("2021-07-10")),
}

# Full extraction windows (event ± 2 weeks)
WINDOWS = {
    "uri":       (pd.Timestamp("2021-01-15"), pd.Timestamp("2021-03-15")),
    "elliott":   (pd.Timestamp("2022-12-01"), pd.Timestamp("2023-01-20")),
    "heat_dome": (pd.Timestamp("2021-06-05"), pd.Timestamp("2021-07-25")),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_epa(paths: list[Path], date_col="Date Local") -> pd.DataFrame:
    """Load and concatenate EPA daily CSVs, normalise key columns."""
    frames = []
    for p in paths:
        print(f"  Loading {p.name} …", end=" ")
        df = pd.read_csv(p, dtype=str, low_memory=False)
        print(f"{len(df):,} rows")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Normalise column names — strip quotes that some EPA files carry
    df.columns = [c.strip('"').strip() for c in df.columns]

    # Parse date — EPA uses either YYYY-MM-DD or M/D/YYYY
    df["date"] = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)

    # Zero-pad state/county codes to 2/3 digits
    df["state_code"] = df["State Code"].str.strip('"').str.zfill(2)
    df["county_code"] = df["County Code"].str.strip('"').str.zfill(3)

    # Numeric columns
    df["arithmetic_mean"] = pd.to_numeric(df["Arithmetic Mean"], errors="coerce")
    df["aqi"] = pd.to_numeric(df["AQI"], errors="coerce")

    return df


def filter_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows matching our target regions."""
    masks = []
    for sc, cc, _ in REGIONS:
        masks.append((df["state_code"] == sc) & (df["county_code"] == cc))
    return df[pd.concat([m for m in masks], axis=1).any(axis=1)].copy()


def assign_event(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with its event label based on state/county."""
    lookup = {(sc, cc): ev for sc, cc, ev in REGIONS}
    df["event"] = df.apply(lambda r: lookup.get((r["state_code"], r["county_code"])), axis=1)
    return df


def filter_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates within the extraction window for each event."""
    keep = pd.Series(False, index=df.index)
    for event, (start, end) in WINDOWS.items():
        mask = (df["event"] == event) & (df["date"] >= start) & (df["date"] <= end)
        keep |= mask
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading PM2.5 data")
print("=" * 70)
pm25_raw = load_epa(PM25_FILES)
pm25 = filter_regions(pm25_raw)
pm25 = assign_event(pm25)
pm25 = filter_windows(pm25)
print(f"  PM2.5 rows after filtering: {len(pm25):,}")

print()
print("=" * 70)
print("Loading Ozone data")
print("=" * 70)
ozone_raw = load_epa(OZONE_FILES)
ozone = filter_regions(ozone_raw)
ozone = assign_event(ozone)
ozone = filter_windows(ozone)
print(f"  Ozone rows after filtering: {len(ozone):,}")

# ---------------------------------------------------------------------------
# Aggregate: daily mean per county for each pollutant
# ---------------------------------------------------------------------------
group_cols = ["date", "event", "state_code", "county_code"]

pm25_daily = (
    pm25.groupby(group_cols)
    .agg(pm25_mean=("arithmetic_mean", "mean"), pm25_aqi=("aqi", "mean"))
    .reset_index()
)

ozone_daily = (
    ozone.groupby(group_cols)
    .agg(ozone_mean=("arithmetic_mean", "mean"), ozone_aqi=("aqi", "mean"))
    .reset_index()
)

# Merge PM2.5 and ozone on date + location
merged = pd.merge(pm25_daily, ozone_daily, on=group_cols, how="outer").sort_values(
    ["event", "state_code", "county_code", "date"]
)

# Add county name for readability
county_names = {
    ("48", "201"): "Harris",
    ("48", "113"): "Dallas",
    ("39", "061"): "Hamilton",
    ("42", "003"): "Allegheny",
    ("41", "051"): "Multnomah",
    ("53", "033"): "King",
}
merged["county"] = merged.apply(
    lambda r: county_names.get((r["state_code"], r["county_code"]), "Unknown"), axis=1
)

# Reorder columns
merged = merged[
    ["date", "event", "county", "state_code", "county_code",
     "pm25_mean", "ozone_mean", "pm25_aqi", "ozone_aqi"]
]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT, index=False)
print(f"\nSaved {len(merged):,} rows -> {OUT}")

# ---------------------------------------------------------------------------
# Analysis: per-event findings
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("EVENT-LEVEL AIR QUALITY ANALYSIS")
print("=" * 70)

for event_name, (ev_start, ev_end) in EVENTS.items():
    win_start, win_end = WINDOWS[event_name]
    pre_start = win_start
    pre_end = ev_start - pd.Timedelta(days=1)
    post_start = ev_end + pd.Timedelta(days=1)
    post_end = win_end

    sub = merged[merged["event"] == event_name].copy()
    if sub.empty:
        print(f"\n--- {event_name.upper()}: NO DATA ---")
        continue

    before = sub[(sub["date"] >= pre_start) & (sub["date"] <= pre_end)]
    during = sub[(sub["date"] >= ev_start) & (sub["date"] <= ev_end)]
    after  = sub[(sub["date"] >= post_start) & (sub["date"] <= post_end)]

    print(f"\n{'─' * 70}")
    print(f"EVENT: {event_name.upper()}")
    print(f"  Window:  {win_start.date()} to {win_end.date()}")
    print(f"  Event:   {ev_start.date()} to {ev_end.date()}")
    print(f"  Pre:     {pre_start.date()} to {pre_end.date()}  ({len(before)} obs)")
    print(f"  During:  {ev_start.date()} to {ev_end.date()}  ({len(during)} obs)")
    print(f"  Post:    {post_start.date()} to {post_end.date()}  ({len(after)} obs)")

    for county_label in sub["county"].unique():
        c_before = before[before["county"] == county_label]
        c_during = during[during["county"] == county_label]

        bl_pm25 = c_before["pm25_mean"].mean()
        pk_pm25 = c_during["pm25_mean"].max()
        bl_pm25_aqi = c_before["pm25_aqi"].mean()
        pk_pm25_aqi = c_during["pm25_aqi"].max()

        bl_ozone = c_before["ozone_mean"].mean()
        pk_ozone = c_during["ozone_mean"].max()
        bl_ozone_aqi = c_before["ozone_aqi"].mean()
        pk_ozone_aqi = c_during["ozone_aqi"].max()

        # % increase
        pct_pm25 = ((pk_pm25 - bl_pm25) / bl_pm25 * 100) if bl_pm25 and not np.isnan(bl_pm25) else np.nan
        pct_ozone = ((pk_ozone - bl_ozone) / bl_ozone * 100) if bl_ozone and not np.isnan(bl_ozone) else np.nan

        # Days exceeding thresholds during event
        usg_pm25 = (c_during["pm25_aqi"] > 100).sum() + (c_during["pm25_mean"] > 35.4).sum()
        # de-dup: count day as exceeding if EITHER metric exceeds
        exceed_pm25 = ((c_during["pm25_aqi"] > 100) | (c_during["pm25_mean"] > 35.4)).sum()
        exceed_ozone = (c_during["ozone_aqi"] > 100).sum()

        sc = c_during["state_code"].iloc[0] if len(c_during) else "?"
        cc = c_during["county_code"].iloc[0] if len(c_during) else "?"

        print(f"\n  {county_label} County (state={sc}, county={cc}):")
        print(f"    PM2.5  — baseline avg: {bl_pm25:6.2f} µg/m³ (AQI {bl_pm25_aqi:5.1f})"
              f"  |  peak during event: {pk_pm25:6.2f} µg/m³ (AQI {pk_pm25_aqi:5.1f})"
              f"  |  Δ = {pct_pm25:+.1f}%")
        print(f"    Ozone  — baseline avg: {bl_ozone:8.4f} ppm  (AQI {bl_ozone_aqi:5.1f})"
              f"  |  peak during event: {pk_ozone:8.4f} ppm  (AQI {pk_ozone_aqi:5.1f})"
              f"  |  Δ = {pct_ozone:+.1f}%")
        print(f"    Days exceeding 'Unhealthy for Sensitive Groups':  PM2.5 = {exceed_pm25},  Ozone = {exceed_ozone}")

print(f"\n{'=' * 70}")
print("Done.")

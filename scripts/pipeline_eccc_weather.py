#!/usr/bin/env python3
"""
Production pipeline: ECCC daily weather → growing season features by province.

Downloads daily data for 10 Prairie weather stations (2000-2024),
computes per-station growing season (May 1 – Sep 30) aggregates,
then averages to province level.

Output features per province-year:
  - gdd_total: Growing degree days (base 5°C)
  - heat_stress_days: Days with max temp > 30°C
  - precip_total_mm: Total growing season precipitation
  - precip_may_jun_mm: Early season precipitation
  - precip_jul_aug_mm: Mid-season precipitation
  - max_consecutive_dry_days: Longest dry spell
  - frost_free_days: Days between last spring frost and first fall frost
  - mean_temp_growing: Average growing season temperature

Output: data/processed/ca_weather_features.csv
"""

import csv
import io
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from collections import defaultdict

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw" / "eccc"
OUT_DIR = ROOT / "data" / "processed"

# Prairie weather stations — curated for long records and data quality
# Format: station_name: [(station_id, year_start, year_end), ...] to handle ID changes
STATIONS = {
    "AB": {
        "CALGARY INTL A": [(50430, 2012, 2025), (2205, 2000, 2012)],
        "EDMONTON INTL A": [(50149, 2012, 2025), (1865, 2000, 2012)],
        "LETHBRIDGE A": [(50430, 2012, 2025), (2263, 2000, 2012)],
        "MEDICINE HAT A": [(2273, 2000, 2025)],
    },
    "SK": {
        "REGINA INTL A": [(28011, 2000, 2025)],
        "SASKATOON DIEFENBAKER INTL A": [(47707, 2008, 2025), (3328, 2000, 2008)],
        "SWIFT CURRENT CDA": [(3185, 2000, 2025)],
        "INDIAN HEAD CDA": [(2925, 2000, 2025)],
    },
    "MB": {
        "WINNIPEG RICHARDSON INTL A": [(27174, 2000, 2025)],
        "BRANDON A": [(3471, 2000, 2025)],
    },
}

# Province name mapping
PROV_NAMES = {"AB": "Alberta", "SK": "Saskatchewan", "MB": "Manitoba"}

YEAR_START = 2000
YEAR_END = 2024


def fetch_daily(station_id, year):
    """Fetch one station-year of daily data from ECCC."""
    url = (
        f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&stationID={station_id}&Year={year}&Month=1&Day=14&timeframe=2"
        f"&submit=Download+Data"
    )
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse)"})
        resp = urlopen(req, timeout=30)
        content = resp.read().decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
    except Exception:
        return []


def safe_float(val):
    """Convert to float, return None if empty/invalid."""
    if not val or not val.strip():
        return None
    try:
        return float(val)
    except ValueError:
        return None


def compute_growing_season_features(daily_rows):
    """Compute growing season features from one station-year of daily data."""
    # Filter to growing season (May 1 – Sep 30)
    growing = []
    all_rows = []

    for row in daily_rows:
        month = safe_float(row.get("Month"))
        if month is None:
            continue
        month = int(month)

        mean_temp = safe_float(row.get("Mean Temp (\u00b0C)", row.get("Mean Temp (°C)")))
        max_temp = safe_float(row.get("Max Temp (\u00b0C)", row.get("Max Temp (°C)")))
        min_temp = safe_float(row.get("Min Temp (\u00b0C)", row.get("Min Temp (°C)")))
        precip = safe_float(row.get("Total Precip (mm)"))

        record = {
            "month": month,
            "day": int(safe_float(row.get("Day", 0)) or 0),
            "mean_temp": mean_temp,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "precip": precip,
        }
        all_rows.append(record)

        if 5 <= month <= 9:
            growing.append(record)

    if len(growing) < 100:  # Need at least ~100 days of growing season data
        return None

    # GDD: sum of max(0, mean_temp - 5) over growing season
    gdd_total = sum(
        max(0, r["mean_temp"] - 5.0) for r in growing if r["mean_temp"] is not None
    )

    # Heat stress days: max temp > 30°C
    heat_stress_days = sum(
        1 for r in growing if r["max_temp"] is not None and r["max_temp"] > 30.0
    )

    # Precipitation totals
    precip_total = sum(r["precip"] for r in growing if r["precip"] is not None)
    precip_may_jun = sum(
        r["precip"] for r in growing
        if r["precip"] is not None and r["month"] in (5, 6)
    )
    precip_jul_aug = sum(
        r["precip"] for r in growing
        if r["precip"] is not None and r["month"] in (7, 8)
    )

    # Max consecutive dry days (precip < 1mm)
    max_dry = 0
    current_dry = 0
    for r in growing:
        if r["precip"] is not None and r["precip"] < 1.0:
            current_dry += 1
            max_dry = max(max_dry, current_dry)
        else:
            current_dry = 0

    # Mean growing season temperature
    temps = [r["mean_temp"] for r in growing if r["mean_temp"] is not None]
    mean_temp_growing = sum(temps) / len(temps) if temps else None

    # Frost-free period: last spring frost (min < 0 in Jan-Jun) to first fall frost (min < 0 in Jul-Dec)
    last_spring_frost_day = 0
    first_fall_frost_day = 365
    for i, r in enumerate(all_rows):
        if r["min_temp"] is not None and r["min_temp"] < 0:
            if r["month"] <= 6:
                last_spring_frost_day = max(last_spring_frost_day, i)
            elif r["month"] >= 7:
                first_fall_frost_day = min(first_fall_frost_day, i)
                break

    frost_free_days = max(0, first_fall_frost_day - last_spring_frost_day)

    return {
        "gdd_total": round(gdd_total, 1),
        "heat_stress_days": heat_stress_days,
        "precip_total_mm": round(precip_total, 1),
        "precip_may_jun_mm": round(precip_may_jun, 1),
        "precip_jul_aug_mm": round(precip_jul_aug, 1),
        "max_consecutive_dry_days": max_dry,
        "frost_free_days": frost_free_days,
        "mean_temp_growing": round(mean_temp_growing, 2) if mean_temp_growing else None,
        "growing_season_days_with_data": len(growing),
    }


def main():
    print("=" * 60)
    print("Pipeline: ECCC Weather → Growing Season Features")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect station-level features
    station_features = []  # (year, province, station, features_dict)
    total_fetches = 0
    failed_fetches = 0

    for prov_code, stations in STATIONS.items():
        province = PROV_NAMES[prov_code]
        for station_name, id_ranges in stations.items():
            for year in range(YEAR_START, YEAR_END + 1):
                # Find the right station ID for this year
                station_id = None
                for sid, y_start, y_end in id_ranges:
                    if y_start <= year <= y_end:
                        station_id = sid
                        break

                if station_id is None:
                    continue

                total_fetches += 1
                rows = fetch_daily(station_id, year)

                if not rows:
                    failed_fetches += 1
                    continue

                features = compute_growing_season_features(rows)
                if features:
                    station_features.append((year, province, station_name, features))

                # Rate limiting — be nice to ECCC
                if total_fetches % 10 == 0:
                    print(f"  Fetched {total_fetches} station-years... ({len(station_features)} with features)")
                    time.sleep(0.5)

    print(f"\nTotal fetches: {total_fetches}, failed: {failed_fetches}")
    print(f"Station-years with features: {len(station_features)}")

    # Aggregate to province level (average across stations)
    prov_year_features = defaultdict(list)
    for year, province, station, features in station_features:
        prov_year_features[(year, province)].append(features)

    feature_keys = [
        "gdd_total", "heat_stress_days", "precip_total_mm",
        "precip_may_jun_mm", "precip_jul_aug_mm",
        "max_consecutive_dry_days", "frost_free_days", "mean_temp_growing",
    ]

    output_rows = []
    for (year, province), feat_list in sorted(prov_year_features.items()):
        row = {"year": year, "province": province, "n_stations": len(feat_list)}
        for key in feature_keys:
            values = [f[key] for f in feat_list if f.get(key) is not None]
            row[key] = round(sum(values) / len(values), 2) if values else ""
        output_rows.append(row)

    # Save
    out_path = OUT_DIR / "ca_weather_features.csv"
    fieldnames = ["year", "province", "n_stations"] + feature_keys
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✓ Saved {len(output_rows)} province-year records → {out_path}")

    # Show 2021 drought year as sanity check
    drought = [r for r in output_rows if r["year"] == 2021]
    if drought:
        print(f"\n  2021 drought year features:")
        for r in drought:
            print(f"    {r['province']}:")
            print(f"      GDD: {r['gdd_total']}, Heat stress days: {r['heat_stress_days']}")
            print(f"      Precip: {r['precip_total_mm']}mm, Max dry spell: {r['max_consecutive_dry_days']}d")

    # Show normal year for comparison
    normal = [r for r in output_rows if r["year"] == 2019]
    if normal:
        print(f"\n  2019 normal year for comparison:")
        for r in normal:
            print(f"    {r['province']}:")
            print(f"      GDD: {r['gdd_total']}, Heat stress days: {r['heat_stress_days']}")
            print(f"      Precip: {r['precip_total_mm']}mm, Max dry spell: {r['max_consecutive_dry_days']}d")

    print("\n✓ ECCC pipeline complete")


if __name__ == "__main__":
    main()

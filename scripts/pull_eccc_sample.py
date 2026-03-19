#!/usr/bin/env python3
"""
Data spike: Pull weather data from Environment and Climate Change Canada (ECCC).
Uses the ECCC climate data bulk download service.
Target: Prairie weather stations — daily temp/precip for 2000-2024.
"""

import csv
import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "eccc"

# ECCC Climate Data bulk download
# Docs: https://climate.weather.gc.ca/doc/Technical_Documentation.pdf
# Bulk URL pattern for daily data:
# https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=XXXX&Year=YYYY&Month=1&Day=14&timeframe=2
# timeframe=2 = daily

# Key prairie weather stations (high-quality, long records)
PRAIRIE_STATIONS = {
    # Station Name: (Station ID, Province)
    "CALGARY INTL A": (50430, "AB"),
    "EDMONTON INTL A": (50149, "AB"),
    "LETHBRIDGE A": (50430, "AB"),  # Will verify
    "REGINA INTL A": (28011, "SK"),
    "SASKATOON DIEFENBAKER INTL A": (47707, "SK"),
    "WINNIPEG RICHARDSON INTL A": (27174, "MB"),
    "BRANDON A": (3471, "MB"),
    "MEDICINE HAT A": (2273, "AB"),
    "SWIFT CURRENT CDA": (3185, "SK"),
    "INDIAN HEAD CDA": (2925, "SK"),
}

# We'll test with a few key stations first
TEST_STATIONS = {
    "REGINA INTL A": (28011, "SK"),
    "CALGARY INTL A": (50430, "AB"),
    "WINNIPEG RICHARDSON INTL A": (27174, "MB"),
}


def fetch_station_inventory():
    """Fetch the ECCC station inventory to validate station IDs."""
    print("Fetching ECCC station inventory...")

    url = "https://dd.weather.gc.ca/climate/observations/daily/csv/"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse Research)"})
        resp = urlopen(req, timeout=30)
        print(f"  Status: {resp.status}")
        # This is a directory listing — just confirms the endpoint works
        html = resp.read(5000).decode("utf-8", errors="replace")
        print(f"  Response length: {len(html)} bytes")
        print("  ✓ ECCC bulk download endpoint is accessible")
        return True
    except Exception as e:
        print(f"  ✗ Inventory fetch failed: {e}")
        return False


def fetch_daily_data(station_name, station_id, province, year):
    """Fetch daily climate data for a station and year."""
    # ECCC bulk CSV endpoint
    url = (
        f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        f"?format=csv&stationID={station_id}&Year={year}&Month=1&Day=14&timeframe=2"
        f"&submit=Download+Data"
    )

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse Research)"})
        resp = urlopen(req, timeout=30)

        if resp.status != 200:
            return None, f"HTTP {resp.status}"

        content = resp.read().decode("utf-8-sig", errors="replace")

        # Parse CSV
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            return None, "Empty response"

        return rows, None

    except Exception as e:
        return None, str(e)


def analyze_station_data(rows, station_name, year):
    """Analyze daily weather data quality."""
    if not rows:
        return None

    headers = list(rows[0].keys())

    # Key fields we need
    key_fields = {
        "Date/Time": 0,
        "Max Temp (°C)": 0,
        "Min Temp (°C)": 0,
        "Mean Temp (°C)": 0,
        "Total Precip (mm)": 0,
    }

    # Also check alternate header names
    alt_fields = {
        "Max Temp (\u00b0C)": "Max Temp (°C)",
        "Min Temp (\u00b0C)": "Min Temp (°C)",
        "Mean Temp (\u00b0C)": "Mean Temp (°C)",
    }

    total = len(rows)
    non_null = {k: 0 for k in key_fields}

    for row in rows:
        for field in key_fields:
            # Try exact match and alternate names
            val = row.get(field, "")
            if not val:
                for alt, canonical in alt_fields.items():
                    if canonical == field:
                        val = row.get(alt, "")
                        if val:
                            break
            if val and val.strip():
                non_null[field] += 1

    completeness = {k: (v / total * 100 if total > 0 else 0) for k, v in non_null.items()}

    return {
        "station": station_name,
        "year": year,
        "total_days": total,
        "headers": headers[:15],
        "completeness": completeness,
    }


def main():
    print("=" * 60)
    print("ClimatePulse Data Spike: ECCC Weather Data")
    print("Target: Prairie stations — daily temp/precip")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Check endpoint accessibility
    endpoint_ok = fetch_station_inventory()

    # Step 2: Pull sample data from test stations
    all_results = []
    all_sample_rows = []
    test_years = [2010, 2021]  # 2021 = drought year (demo climax)

    for station_name, (station_id, province) in TEST_STATIONS.items():
        for year in test_years:
            print(f"\nFetching {station_name} ({province}), {year}...")
            rows, error = fetch_daily_data(station_name, station_id, province, year)

            if error:
                print(f"  ✗ Error: {error}")
                all_results.append({
                    "station": station_name,
                    "province": province,
                    "year": year,
                    "status": "error",
                    "error": error,
                })
                continue

            print(f"  ✓ Got {len(rows)} rows")

            # Analyze quality
            analysis = analyze_station_data(rows, station_name, year)
            if analysis:
                analysis["province"] = province
                analysis["station_id"] = station_id
                analysis["status"] = "ok"
                all_results.append(analysis)
                print(f"  Headers: {analysis['headers'][:8]}")
                print(f"  Completeness:")
                for field, pct in analysis["completeness"].items():
                    status = "✓" if pct > 90 else "⚠" if pct > 50 else "✗"
                    print(f"    {status} {field}: {pct:.1f}%")

            # Save sample rows (add station info)
            for row in rows:
                row["_station"] = station_name
                row["_province"] = province
                row["_station_id"] = station_id
            all_sample_rows.extend(rows)

    # Step 3: Save results
    # Save sample data
    if all_sample_rows:
        sample_path = OUTPUT_DIR / "prairie_weather_sample.csv"
        fieldnames = list(all_sample_rows[0].keys())
        with open(sample_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_sample_rows)
        print(f"\n✓ Saved {len(all_sample_rows):,} daily records to {sample_path}")

    # Save analysis
    stats_path = OUTPUT_DIR / "eccc_spike_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"✓ Saved analysis to {stats_path}")

    # Step 4: Verdict
    print("\n" + "=" * 60)
    ok_count = sum(1 for r in all_results if r.get("status") == "ok")
    total_count = len(all_results)

    if ok_count >= 4:  # At least 4 of 6 station-year combos work
        print(f"✓ SPIKE PASSED: ECCC weather data is viable ({ok_count}/{total_count} fetches OK)")
        # Check data quality
        good_completeness = sum(
            1 for r in all_results
            if r.get("status") == "ok"
            and r.get("completeness", {}).get("Mean Temp (°C)", 0) > 80
        )
        print(f"  Temperature completeness: {good_completeness}/{ok_count} stations >80%")
    elif ok_count > 0:
        print(f"⚠ SPIKE PARTIAL: Some data available ({ok_count}/{total_count})")
    else:
        print("✗ SPIKE FAILED: No ECCC data retrieved")
    print("=" * 60)


if __name__ == "__main__":
    main()

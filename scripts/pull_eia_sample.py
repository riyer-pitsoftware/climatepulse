#!/usr/bin/env python3
"""
Pull EIA hourly electricity generation by fuel type for 3 target weather events
plus baseline comparison months. Saves CSVs to data/raw/eia/.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("EIA_API_KEY")
if not API_KEY:
    sys.exit("ERROR: EIA_API_KEY not found in .env")

BASE_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "eia"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 5000  # EIA page size cap

# Each entry: (label, filename, respondent, start, end)
EVENTS = [
    # --- Storm events ---
    ("Texas Winter Storm Uri",      "eia_erco_uri_2021.csv",           "ERCO", "2021-02-01T00", "2021-03-01T00"),
    ("Winter Storm Elliott",        "eia_pjm_elliott_2022.csv",        "PJM",  "2022-12-15T00", "2023-01-05T00"),
    ("PNW Heat Dome",              "eia_bpat_heatdome_2021.csv",      "BPAT", "2021-06-20T00", "2021-07-10T00"),
    # --- Baselines ---
    ("ERCO Baseline (Feb 2020)",   "eia_erco_baseline_2020.csv",      "ERCO", "2020-02-01T00", "2020-03-01T00"),
    ("PJM Baseline (Dec 2021)",    "eia_pjm_baseline_2021.csv",       "PJM",  "2021-12-15T00", "2022-01-05T00"),
    ("BPAT Baseline (Jun 2020)",   "eia_bpat_baseline_2020.csv",      "BPAT", "2020-06-20T00", "2020-07-10T00"),
]


def fetch_eia(respondent: str, start: str, end: str) -> pd.DataFrame:
    """Fetch all pages of hourly fuel-type generation for a respondent/window."""
    all_rows = []
    offset = 0

    while True:
        params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "start": start,
            "end": end,
            "length": MAX_LENGTH,
            "offset": offset,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }

        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        body = resp.json()

        data = body.get("response", {}).get("data", [])
        total = int(body.get("response", {}).get("total", 0))

        if not data:
            break

        all_rows.extend(data)
        offset += len(data)

        print(f"    fetched {offset}/{total} records ...", flush=True)

        if offset >= total:
            break

        # polite pause between pages
        time.sleep(0.5)

    return pd.DataFrame(all_rows)


def main():
    summary = []

    for label, filename, respondent, start, end in EVENTS:
        print(f"\n{'='*60}")
        print(f"  {label}  ({respondent}  {start} -> {end})")
        print(f"{'='*60}")

        df = fetch_eia(respondent, start, end)

        if df.empty:
            print(f"  ** No data returned for {label} **")
            summary.append((label, 0, []))
            continue

        outpath = OUT_DIR / filename
        df.to_csv(outpath, index=False)
        print(f"  Saved {len(df)} rows -> {outpath}")

        fuel_types = sorted(df["fueltype"].unique()) if "fueltype" in df.columns else []
        summary.append((label, len(df), fuel_types))

    # --- Summary ---
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for label, count, fuels in summary:
        print(f"  {label}: {count:,} records")
        if fuels:
            print(f"    fuel types: {', '.join(fuels)}")
    print()


if __name__ == "__main__":
    main()

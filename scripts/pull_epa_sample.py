#!/usr/bin/env python3
"""Download EPA AirData pre-generated CSV files for target years.

Downloads daily PM2.5 (param 88101) and Ozone (param 44201) data
for 2021 and 2022, plus the AQS site listing with lat/lon.

Target weather events:
  - Winter Storm Uri (Feb 2021, Texas)
  - Pacific NW Heat Dome (Jun-Jul 2021, Oregon)
  - Winter Storm Elliott (Dec 2022, Ohio/Eastern US)
"""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("/Users/riyer/code/zervehack/data/raw/epa")

FILES = [
    {
        "url": "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2021.zip",
        "label": "PM2.5 Daily 2021",
        "kind": "pm25",
    },
    {
        "url": "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2022.zip",
        "label": "PM2.5 Daily 2022",
        "kind": "pm25",
    },
    {
        "url": "https://aqs.epa.gov/aqsweb/airdata/daily_44201_2021.zip",
        "label": "Ozone Daily 2021",
        "kind": "ozone",
    },
    {
        "url": "https://aqs.epa.gov/aqsweb/airdata/daily_44201_2022.zip",
        "label": "Ozone Daily 2022",
        "kind": "ozone",
    },
    {
        "url": "https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip",
        "label": "AQS Site Listing",
        "kind": "sites",
    },
]

# States of interest for our 3 weather events
TARGET_STATES = {
    "48": "Texas",       # Winter Storm Uri
    "39": "Ohio",        # Winter Storm Elliott
    "41": "Oregon",      # Pacific NW Heat Dome
}


def download_and_extract(url: str, dest_dir: Path) -> Path:
    """Download a zip file and extract the CSV inside. Returns path to CSV."""
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in {url}")
        # Extract all CSVs (usually just one)
        for name in csv_names:
            zf.extract(name, dest_dir)
            print(f"  Extracted: {name} ({os.path.getsize(dest_dir / name) / 1e6:.1f} MB)")
        return dest_dir / csv_names[0]


def summarize_daily(csv_path: Path, label: str, kind: str) -> None:
    """Load a daily data CSV and print summary statistics."""
    print(f"\n{'='*70}")
    print(f"  {label}  —  {csv_path.name}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path, low_memory=False)

    print(f"  Rows:           {len(df):,}")
    print(f"  Columns:        {list(df.columns)}")

    if "Date Local" in df.columns:
        dates = pd.to_datetime(df["Date Local"])
        print(f"  Date range:     {dates.min().date()} → {dates.max().date()}")
    elif "Date of Last Change" in df.columns:
        # sites file fallback
        pass

    # Unique sites
    site_id_cols = ["State Code", "County Code", "Site Num"]
    if all(c in df.columns for c in site_id_cols):
        df["_site"] = (
            df["State Code"].astype(str).str.zfill(2)
            + "-"
            + df["County Code"].astype(str).str.zfill(3)
            + "-"
            + df["Site Num"].astype(str).str.zfill(4)
        )
        print(f"  Unique sites:   {df['_site'].nunique():,}")
    elif "AQS Site ID" in df.columns:
        print(f"  Unique sites:   {df['AQS Site ID'].nunique():,}")

    # State-level breakdown for PM2.5 files
    if kind == "pm25" and "State Code" in df.columns:
        print(f"\n  Rows per target state (PM2.5):")
        df["_state_code"] = df["State Code"].astype(str).str.zfill(2)
        for code, name in TARGET_STATES.items():
            n = (df["_state_code"] == code).sum()
            print(f"    {name} (State {code}): {n:,} rows")


def summarize_sites(csv_path: Path) -> None:
    """Summarize the AQS sites file."""
    print(f"\n{'='*70}")
    print(f"  AQS Site Listing  —  {csv_path.name}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Rows:           {len(df):,}")
    print(f"  Columns:        {list(df.columns)}")

    if "Latitude" in df.columns and "Longitude" in df.columns:
        print(f"  Lat range:      {df['Latitude'].min():.4f} → {df['Latitude'].max():.4f}")
        print(f"  Lon range:      {df['Longitude'].min():.4f} → {df['Longitude'].max():.4f}")

    # Count sites in target states
    if "State Code" in df.columns:
        df["_state_code"] = df["State Code"].astype(str).str.zfill(2)
        print(f"\n  Sites in target states:")
        for code, name in TARGET_STATES.items():
            n = (df["_state_code"] == code).sum()
            print(f"    {name} (State {code}): {n:,} sites")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to: {DATA_DIR}\n")

    for entry in FILES:
        csv_path = download_and_extract(entry["url"], DATA_DIR)

        if entry["kind"] == "sites":
            summarize_sites(csv_path)
        else:
            summarize_daily(csv_path, entry["label"], entry["kind"])

    print(f"\n{'='*70}")
    print("Done. All files saved to:", DATA_DIR)
    # List final contents
    for p in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

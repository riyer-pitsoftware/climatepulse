"""
ClimatePulse — Download EPA AirData bulk CSV files.
Downloads daily PM2.5 and Ozone data for target years.
No API key required.

Usage: python scripts/download_epa_data.py
"""

import os
import zipfile
from pathlib import Path

import requests

# Target years covering our weather events (2018-2024)
YEARS = range(2018, 2025)

# EPA AirData base URL
BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"

# Files to download per year
FILE_TEMPLATES = {
    "pm25": "daily_88101_{year}.zip",       # PM2.5 FRM/FEM Mass
    "ozone": "daily_44201_{year}.zip",      # Ozone
}

# Site listing (download once)
SITE_FILE = "aqs_sites.zip"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "epa"


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress indicator."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    print(f"  [downloading] {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  [{pct}%] {dest.name}", end="", flush=True)
        print(f"\r  [done] {dest.name} ({downloaded // 1024}KB)")
        return True
    except Exception as e:
        print(f"  [error] {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract a zip file."""
    print(f"  [extracting] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_dir = OUTPUT_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ClimatePulse — EPA AirData Download")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Years: {YEARS[0]}-{YEARS[-1]}")
    print("=" * 60)

    # Download site listing
    print("\n--- Site Listing ---")
    site_zip = OUTPUT_DIR / SITE_FILE
    if download_file(f"{BASE_URL}/{SITE_FILE}", site_zip):
        extract_zip(site_zip, csv_dir)

    # Download daily data files
    for year in YEARS:
        print(f"\n--- Year {year} ---")
        for param_name, template in FILE_TEMPLATES.items():
            filename = template.format(year=year)
            zip_path = OUTPUT_DIR / filename
            if download_file(f"{BASE_URL}/{filename}", zip_path):
                extract_zip(zip_path, csv_dir)

    print("\n" + "=" * 60)
    csv_count = len(list(csv_dir.glob("*.csv")))
    print(f"Done. {csv_count} CSV files in {csv_dir}")
    print("Next: run scripts/validate_apis.py to verify all data sources.")


if __name__ == "__main__":
    main()

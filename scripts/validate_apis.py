"""
ClimatePulse — API Validation Script
Run this after setting up .env to verify all 3 data sources are accessible.
Usage: python scripts/validate_apis.py
"""

import os
import json
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def validate_noaa():
    """Test NOAA CDO API v2 access."""
    token = os.getenv("NOAA_API_TOKEN")
    if not token:
        return False, "NOAA_API_TOKEN not set in .env"

    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets"
    headers = {"token": token}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            count = len(data.get("results", []))
            return True, f"OK — {count} datasets available"
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"Connection error: {e}"


def validate_eia():
    """Test EIA API v2 access."""
    key = os.getenv("EIA_API_KEY")
    if not key:
        return False, "EIA_API_KEY not set in .env"

    # Query electricity generation summary
    url = f"https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    params = {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "length": 5,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("response", {}).get("total", 0)
            return True, f"OK — fuel-type-data endpoint, {total} total records"
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"Connection error: {e}"


def validate_epa():
    """Test EPA AirData bulk CSV download availability (no API key needed)."""
    # Check that the pre-generated files page is accessible
    url = "https://aqs.epa.gov/aqsweb/airdata/download_files.html"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            # Try downloading the site listing to verify bulk access works
            site_url = "https://aqs.epa.gov/aqsweb/airdata/aqs_sites.zip"
            site_resp = requests.head(site_url, timeout=15)
            if site_resp.status_code == 200:
                return True, "OK — bulk CSV downloads accessible (no API key needed)"
            return True, "OK — download page accessible, but could not verify file download"
        return False, f"HTTP {resp.status_code}: download page not accessible"
    except Exception as e:
        return False, f"Connection error: {e}"


def main():
    print("=" * 60)
    print("ClimatePulse — API Validation")
    print("=" * 60)

    checks = [
        ("NOAA CDO", validate_noaa),
        ("EIA Open Data", validate_eia),
        ("EPA AQS", validate_epa),
    ]

    results = []
    for name, fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        results.append((name, ok, msg))
        print(f"\n[{status}] {name}")
        print(f"       {msg}")

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Result: {passed}/{len(results)} APIs validated")

    if passed == len(results):
        print("All APIs ready. Proceed to data collection.")
    else:
        print("Fix failing APIs before proceeding.")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""Pull NOAA GHCND daily weather data for three target extreme-weather events.
Uses specific county FIPS codes instead of entire states to keep data volume manageable."""

import csv
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")
if not NOAA_TOKEN:
    sys.exit("ERROR: NOAA_API_TOKEN not found in .env")

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
HEADERS = {"token": NOAA_TOKEN}
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "noaa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use county-level FIPS for targeted pulls
EVENTS = [
    {
        "name": "uri_houston_tx",
        "locationid": "FIPS:48201",  # Harris County (Houston)
        "startdate": "2021-02-01",
        "enddate": "2021-02-28",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "uri_dallas_tx",
        "locationid": "FIPS:48113",  # Dallas County
        "startdate": "2021-02-01",
        "enddate": "2021-02-28",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "elliott_allegheny_pa",
        "locationid": "FIPS:42003",  # Allegheny County (Pittsburgh)
        "startdate": "2022-12-15",
        "enddate": "2023-01-05",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "elliott_hamilton_oh",
        "locationid": "FIPS:39061",  # Hamilton County (Cincinnati)
        "startdate": "2022-12-15",
        "enddate": "2023-01-05",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "heatdome_multnomah_or",
        "locationid": "FIPS:41051",  # Multnomah County (Portland)
        "startdate": "2021-06-20",
        "enddate": "2021-07-10",
        "datatypeid": "TMIN,TMAX",
    },
    {
        "name": "heatdome_king_wa",
        "locationid": "FIPS:53033",  # King County (Seattle)
        "startdate": "2021-06-20",
        "enddate": "2021-07-10",
        "datatypeid": "TMIN,TMAX",
    },
    # ── Baseline periods (cp-9v9): same-year, 2 weeks before each event ──
    # These provide counterfactual "normal weather" data for the ML model.
    {
        "name": "uri_baseline_houston_tx",
        "locationid": "FIPS:48201",  # Harris County (Houston)
        "startdate": "2021-01-18",
        "enddate": "2021-01-31",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "uri_baseline_dallas_tx",
        "locationid": "FIPS:48113",  # Dallas County
        "startdate": "2021-01-18",
        "enddate": "2021-01-31",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "heatdome_baseline_multnomah_or",
        "locationid": "FIPS:41051",  # Multnomah County (Portland)
        "startdate": "2021-06-06",
        "enddate": "2021-06-19",
        "datatypeid": "TMIN,TMAX",
    },
    {
        "name": "heatdome_baseline_king_wa",
        "locationid": "FIPS:53033",  # King County (Seattle)
        "startdate": "2021-06-06",
        "enddate": "2021-06-19",
        "datatypeid": "TMIN,TMAX",
    },
    {
        "name": "elliott_baseline_allegheny_pa",
        "locationid": "FIPS:42003",  # Allegheny County (Pittsburgh)
        "startdate": "2022-12-01",
        "enddate": "2022-12-14",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
    {
        "name": "elliott_baseline_hamilton_oh",
        "locationid": "FIPS:39061",  # Hamilton County (Cincinnati)
        "startdate": "2022-12-01",
        "enddate": "2022-12-14",
        "datatypeid": "TMIN,TMAX,PRCP,SNOW",
    },
]

LIMIT = 1000


def fetch_event(event: dict) -> list[dict]:
    all_results = []
    offset = 1
    retries = 0

    while True:
        params = {
            "datasetid": "GHCND",
            "locationid": event["locationid"],
            "startdate": event["startdate"],
            "enddate": event["enddate"],
            "datatypeid": event["datatypeid"],
            "units": "standard",
            "limit": LIMIT,
            "offset": offset,
        }
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=60)
            if resp.status_code == 503:
                retries += 1
                if retries > 3:
                    print(f"  FAILED after 3 retries: 503 Service Unavailable")
                    break
                print(f"  503 — retry {retries}/3 after 5s...")
                time.sleep(5)
                continue
            resp.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            retries += 1
            if retries > 3:
                print(f"  FAILED after 3 retries: {e}")
                break
            print(f"  retry {retries}/3 after timeout...")
            time.sleep(3)
            continue
        retries = 0  # reset on success

        body = resp.json()
        results = body.get("results", [])
        if not results:
            break

        all_results.extend(results)
        meta = body.get("metadata", {}).get("resultset", {})
        total = meta.get("count", 0)
        print(f"  fetched {len(all_results)}/{total} records")

        if len(all_results) >= total:
            break

        offset += LIMIT
        time.sleep(0.25)

    return all_results


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  WARNING: no records for {path.name}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summary = []

    for event in EVENTS:
        name = event["name"]
        out_path = OUT_DIR / f"{name}.csv"

        if out_path.exists():
            print(f"\n=== {name} === SKIP (already exists: {out_path.name})")
            summary.append((name, -1))
            continue

        print(f"\n=== {name} ===")
        print(f"  {event['locationid']}  {event['startdate']} -> {event['enddate']}")

        rows = fetch_event(event)
        save_csv(rows, out_path)
        summary.append((name, len(rows)))
        print(f"  -> {len(rows)} records saved to {out_path.name}")
        time.sleep(0.5)

    print("\n========== SUMMARY ==========")
    for name, count in summary:
        print(f"  {name}: {count} records")
    print("==============================")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Production pipeline: StatsCan Table 32-10-0077-01 → monthly farm product prices.

Table 32-10-0359 discontinued prices after 2014.
Table 32-10-0077 has monthly farm product prices by province through recent years.

Output: data/processed/ca_farm_prices_monthly.csv
"""

import csv
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw" / "statcan"
OUT_DIR = ROOT / "data" / "processed"
PRODUCT_ID = "32100077"

PRAIRIE_PROVINCES = ["Alberta", "Saskatchewan", "Manitoba"]

# Target commodities — we'll inspect exact names after download
TARGET_KEYWORDS = ["wheat", "canola", "barley", "oats"]


def download_and_extract():
    """Download and extract StatsCan CSV."""
    csv_files = list(RAW_DIR.glob(f"{PRODUCT_ID}*.csv"))
    data_csvs = [f for f in csv_files if "MetaData" not in f.name]
    if data_csvs:
        print(f"Using cached {data_csvs[0]}")
        return data_csvs[0]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{PRODUCT_ID}-eng.zip"

    if not zip_path.exists():
        url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{PRODUCT_ID}-eng.zip"
        print(f"Downloading {url}...")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse)"})
        data = urlopen(req, timeout=120).read()
        zip_path.write_bytes(data)
        print(f"  Downloaded {len(data):,} bytes")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"  Zip contents: {names}")
        csv_name = [n for n in names if n.endswith(".csv") and "MetaData" not in n][0]
        zf.extract(csv_name, RAW_DIR)
        return RAW_DIR / csv_name


def parse_prices(csv_path):
    """Parse and filter price data."""
    print(f"\nParsing: {csv_path}")

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"  Columns: {headers[:10]}...")

        # First pass: discover column names and commodities
        commodities = set()
        geos = set()
        rows_all = []
        for row in reader:
            rows_all.append(row)
            geo = row.get("GEO", "")
            if any(p in geo for p in PRAIRIE_PROVINCES):
                geos.add(geo)
                # Try different column names for commodity
                commodity = row.get("Farm products", row.get("Type of product", row.get("Products", "")))
                if commodity:
                    commodities.add(commodity)

        print(f"  Total rows: {len(rows_all):,}")
        print(f"  Prairie GEOs: {sorted(geos)}")
        print(f"  Commodities ({len(commodities)}): {sorted(list(commodities))[:20]}")

        # Filter for target crops
        filtered = []
        for row in rows_all:
            geo = row.get("GEO", "")
            if geo not in PRAIRIE_PROVINCES:
                continue

            commodity = row.get("Farm products", row.get("Type of product", row.get("Products", "")))
            if not any(kw in commodity.lower() for kw in TARGET_KEYWORDS):
                continue

            value = row.get("VALUE", "").strip()
            if not value:
                continue

            ref_date = row.get("REF_DATE", "")
            uom = row.get("UOM", "")

            filtered.append({
                "ref_date": ref_date,
                "province": geo,
                "commodity": commodity,
                "price": float(value),
                "uom": uom,
            })

        return filtered


def main():
    print("=" * 60)
    print("Pipeline: StatsCan Farm Prices (Table 32-10-0077)")
    print("=" * 60)

    csv_path = download_and_extract()
    rows = parse_prices(csv_path)

    if not rows:
        print("\n⚠ No matching price data found")
        print("  This table may not cover the expected commodities.")
        print("  Check commodity names above and adjust TARGET_KEYWORDS.")
        return

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ca_farm_prices_monthly.csv"
    fieldnames = ["ref_date", "province", "commodity", "price", "uom"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Saved {len(rows):,} rows → {out_path}")

    # Summary
    years = set(r["ref_date"][:4] for r in rows)
    print(f"  Year range: {min(years)}–{max(years)}")
    commodities = set(r["commodity"] for r in rows)
    print(f"  Commodities: {sorted(commodities)}")

    # Show recent wheat prices
    wheat = [r for r in rows if "wheat" in r["commodity"].lower() and r["ref_date"] >= "2020"]
    if wheat:
        print(f"\n  Recent wheat prices (sample):")
        for r in wheat[:5]:
            print(f"    {r['ref_date']} | {r['province']} | {r['commodity']} | ${r['price']:.2f} {r['uom']}")


if __name__ == "__main__":
    main()

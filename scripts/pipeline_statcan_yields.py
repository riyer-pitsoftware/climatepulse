#!/usr/bin/env python3
"""
Production pipeline: StatsCan Table 32-10-0359-01 → cleaned crop yields + prices.

Extracts for Prairie provinces (AB, SK, MB):
  - Yield: kg/ha (metric)
  - Harvested area: hectares
  - Production: metric tonnes
  - Farm prices: $/tonne
For crops: wheat (all), canola, barley, oats

Output: data/processed/ca_crop_yields.csv
        data/processed/ca_farm_prices.csv
"""

import csv
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw" / "statcan"
OUT_DIR = ROOT / "data" / "processed"
PRODUCT_ID = "32100359"

PRAIRIE_PROVINCES = ["Alberta", "Saskatchewan", "Manitoba"]

# Crops we care about — map to canonical names
CROP_MAP = {
    "Wheat, all": "Wheat",
    "Canola (rapeseed)": "Canola",
    "Barley": "Barley",
    "Oats": "Oats",
}

# Dispositions we want (metric units only)
YIELD_DISP = "Average yield (kilograms per hectare)"
AREA_DISP = "Harvested area (hectares)"
PRODUCTION_DISP = "Production (metric tonnes)"
PRICE_DISP = "Average farm price (dollars per tonne)"


def download_if_needed():
    """Download and extract StatsCan CSV if not already present."""
    csv_path = RAW_DIR / f"{PRODUCT_ID}.csv"
    if csv_path.exists():
        print(f"Using cached {csv_path}")
        return csv_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{PRODUCT_ID}-eng.zip"

    if not zip_path.exists():
        url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{PRODUCT_ID}-eng.zip"
        print(f"Downloading {url}...")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse)"})
        data = urlopen(req, timeout=60).read()
        zip_path.write_bytes(data)
        print(f"  Downloaded {len(data):,} bytes")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(f"{PRODUCT_ID}.csv", RAW_DIR)

    return csv_path


def parse_and_clean(csv_path):
    """Parse raw StatsCan CSV into structured records."""
    yields = []  # (year, province, crop, yield_kg_ha)
    areas = []   # (year, province, crop, harvested_ha)
    prods = []   # (year, province, crop, production_mt)
    prices = []  # (year, province, crop, price_per_tonne)

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geo = row["GEO"]
            if geo not in PRAIRIE_PROVINCES:
                continue

            crop_raw = row["Type of crop"]
            if crop_raw not in CROP_MAP:
                continue

            crop = CROP_MAP[crop_raw]
            year = int(row["REF_DATE"])
            disp = row["Harvest disposition"]
            value_str = row["VALUE"].strip()

            if not value_str:
                continue

            try:
                value = float(value_str)
            except ValueError:
                continue

            record = (year, geo, crop, value)

            if disp == YIELD_DISP:
                yields.append(record)
            elif disp == AREA_DISP:
                areas.append(record)
            elif disp == PRODUCTION_DISP:
                prods.append(record)
            elif disp == PRICE_DISP:
                prices.append(record)

    return yields, areas, prods, prices


def build_yield_table(yields, areas, prods):
    """Merge yield, area, production into one table."""
    # Index by (year, province, crop)
    yield_map = {(y, p, c): v for y, p, c, v in yields}
    area_map = {(y, p, c): v for y, p, c, v in areas}
    prod_map = {(y, p, c): v for y, p, c, v in prods}

    all_keys = sorted(set(yield_map) | set(area_map) | set(prod_map))

    rows = []
    for key in all_keys:
        year, province, crop = key
        # Focus on 2000+ for model training
        if year < 2000:
            continue
        rows.append({
            "year": year,
            "province": province,
            "crop": crop,
            "yield_kg_ha": yield_map.get(key, ""),
            "harvested_ha": area_map.get(key, ""),
            "production_mt": prod_map.get(key, ""),
        })

    return rows


def build_price_table(prices):
    """Build price table. Note: StatsCan discontinued farm prices in this table
    after 2014 — values for 2015+ are empty. We include all available years."""
    rows = []
    for year, province, crop, price in prices:
        rows.append({
            "year": year,
            "province": province,
            "crop": crop,
            "price_cad_per_tonne": price,
        })
    return sorted(rows, key=lambda r: (r["year"], r["province"], r["crop"]))


def save_csv(rows, path, fieldnames):
    """Write rows to CSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows → {path}")


def main():
    print("=" * 60)
    print("Pipeline: StatsCan Crop Yields + Prices")
    print("=" * 60)

    csv_path = download_if_needed()
    yields, areas, prods, prices = parse_and_clean(csv_path)

    print(f"\nRaw counts (Prairie provinces, target crops):")
    print(f"  Yields:     {len(yields):,}")
    print(f"  Areas:      {len(areas):,}")
    print(f"  Production: {len(prods):,}")
    print(f"  Prices:     {len(prices):,}")

    # Build and save yield table
    yield_rows = build_yield_table(yields, areas, prods)
    save_csv(yield_rows,
             OUT_DIR / "ca_crop_yields.csv",
             ["year", "province", "crop", "yield_kg_ha", "harvested_ha", "production_mt"])

    # Build and save price table
    price_rows = build_price_table(prices)
    save_csv(price_rows,
             OUT_DIR / "ca_farm_prices.csv",
             ["year", "province", "crop", "price_cad_per_tonne"])

    # Summary stats
    print(f"\nYield table: {len(yield_rows)} rows")
    if yield_rows:
        years = [r["year"] for r in yield_rows]
        print(f"  Years: {min(years)}–{max(years)}")
        crops = set(r["crop"] for r in yield_rows)
        print(f"  Crops: {sorted(crops)}")
        provs = set(r["province"] for r in yield_rows)
        print(f"  Provinces: {sorted(provs)}")

        # Show 2021 drought year data as sanity check
        drought_wheat = [r for r in yield_rows if r["year"] == 2021 and r["crop"] == "Wheat"]
        if drought_wheat:
            print(f"\n  2021 drought year — Wheat yields:")
            for r in drought_wheat:
                print(f"    {r['province']}: {r['yield_kg_ha']} kg/ha")

    print(f"\nPrice table: {len(price_rows)} rows")
    if price_rows:
        # Show 2021-2022 price spike
        spike = [r for r in price_rows if r["year"] in (2020, 2021, 2022) and r["crop"] == "Wheat"]
        if spike:
            print(f"\n  Wheat price spike (2020-2022):")
            for r in spike:
                print(f"    {r['year']} {r['province']}: ${r['price_cad_per_tonne']:.0f}/tonne")

    print("\n✓ StatsCan pipeline complete")


if __name__ == "__main__":
    main()

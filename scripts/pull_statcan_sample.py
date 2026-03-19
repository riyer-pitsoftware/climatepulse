#!/usr/bin/env python3
"""
Data spike: Pull crop yield data from StatsCan Table 32-10-0359-01.
Uses the StatsCan Web Data Service (WDS) REST API.
Target: Prairie provinces (AB, SK, MB) — wheat, canola, barley yields 2000-2024.
"""

import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import csv
import io

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "statcan"

# StatsCan WDS API for Table 32-10-0359-01 (crop yields by province)
# Product ID: 32-10-0359-01
PRODUCT_ID = "32100359"

# WDS endpoint
BASE_URL = "https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadTbl/en"

# Try the bulk CSV approach first — more reliable than WDS for full tables
BULK_CSV_URL = f"https://www150.statcan.gc.ca/n1/tbl/csv/{PRODUCT_ID}-eng.zip"

# Alternative: WDS JSON API
WDS_URL = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action"

# Prairie province GEO codes in StatsCan
PRAIRIE_PROVINCES = {
    "Alberta": "48",
    "Saskatchewan": "47",
    "Manitoba": "46",
}

# Target crops
TARGET_CROPS = ["wheat", "canola", "barley"]


def try_wds_api():
    """Try StatsCan WDS REST API for JSON data."""
    print("Attempting StatsCan WDS API...")

    # The WDS getDataFromCubePidCoordAndLatestNPeriods endpoint
    wds_base = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action"

    # Try the direct CSV download endpoint
    # Format: https://www150.statcan.gc.ca/n1/tbl/csv/PIDID-eng.zip
    url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{PRODUCT_ID}-eng.zip"
    print(f"Trying bulk CSV: {url}")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse Research)"})
        resp = urlopen(req, timeout=30)
        print(f"  Status: {resp.status}")
        print(f"  Content-Type: {resp.headers.get('Content-Type')}")
        content_length = resp.headers.get("Content-Length", "unknown")
        print(f"  Content-Length: {content_length}")

        if resp.status == 200:
            import zipfile
            import tempfile

            # Save zip to temp, extract CSV
            zip_path = OUTPUT_DIR / f"{PRODUCT_ID}-eng.zip"
            data = resp.read()
            zip_path.write_bytes(data)
            print(f"  Downloaded {len(data):,} bytes to {zip_path}")

            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                print(f"  Zip contents: {names}")

                # Find the main data CSV (usually the largest file)
                csv_name = [n for n in names if n.endswith(".csv") and "MetaData" not in n]
                if csv_name:
                    csv_name = csv_name[0]
                    print(f"  Extracting: {csv_name}")
                    zf.extract(csv_name, OUTPUT_DIR)
                    csv_path = OUTPUT_DIR / csv_name
                    return csv_path

                # If all CSVs have MetaData, just extract the first CSV
                csv_name = [n for n in names if n.endswith(".csv")][0]
                zf.extract(csv_name, OUTPUT_DIR)
                return OUTPUT_DIR / csv_name

    except (URLError, HTTPError) as e:
        print(f"  Bulk CSV failed: {e}")

    return None


def try_wds_json_api():
    """Try the newer WDS JSON API."""
    print("\nAttempting WDS JSON API...")

    # getFullTableDownloadCSV endpoint
    url = f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={PRODUCT_ID}01"
    print(f"Trying: {url}")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ClimatePulse Research)"})
        resp = urlopen(req, timeout=30)
        print(f"  Status: {resp.status}")
        # This returns HTML — we just want to confirm the table exists
        html = resp.read(2000).decode("utf-8", errors="replace")
        if "32-10-0359" in html or "crop" in html.lower():
            print("  ✓ Table exists and is accessible")
            return True
    except Exception as e:
        print(f"  Failed: {e}")

    return False


def analyze_crop_data(csv_path):
    """Parse and filter the StatsCan CSV for prairie crop yields."""
    print(f"\nAnalyzing: {csv_path}")

    # StatsCan CSVs use UTF-8-BOM encoding
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"  Columns: {headers[:10]}...")

        rows = []
        crop_types = set()
        geos = set()
        years = set()
        total_rows = 0

        for row in reader:
            total_rows += 1
            geo = row.get("GEO", "")
            crop = row.get("Type of crop", row.get("Crop type", ""))
            ref_date = row.get("REF_DATE", "")

            # Track unique values
            if geo:
                geos.add(geo)
            if crop:
                crop_types.add(crop)
            if ref_date:
                years.add(ref_date)

            # Filter for prairie provinces and target crops
            is_prairie = any(p in geo for p in PRAIRIE_PROVINCES.keys())
            is_target_crop = any(c.lower() in crop.lower() for c in TARGET_CROPS) if crop else False

            if is_prairie and is_target_crop:
                rows.append(row)

        print(f"\n  Total rows in table: {total_rows:,}")
        print(f"  Unique GEOs: {len(geos)}")
        print(f"  Sample GEOs: {sorted(list(geos))[:10]}")
        print(f"  Unique crops: {len(crop_types)}")
        print(f"  Sample crops: {sorted(list(crop_types))[:15]}")
        print(f"  Year range: {min(years) if years else 'N/A'} — {max(years) if years else 'N/A'}")
        print(f"  Prairie + target crop rows: {len(rows):,}")

        return rows, headers, {
            "total_rows": total_rows,
            "unique_geos": len(geos),
            "unique_crops": len(crop_types),
            "year_range": f"{min(years) if years else 'N/A'}–{max(years) if years else 'N/A'}",
            "prairie_target_rows": len(rows),
            "sample_geos": sorted(list(geos))[:10],
            "sample_crops": sorted(list(crop_types))[:15],
        }


def save_filtered_data(rows, headers, stats):
    """Save filtered prairie crop data."""
    if not rows:
        print("\n⚠ No filtered rows — saving stats only")
        stats_path = OUTPUT_DIR / "statcan_spike_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved stats to {stats_path}")
        return

    out_path = OUTPUT_DIR / "prairie_crop_yields_sample.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  ✓ Saved {len(rows):,} filtered rows to {out_path}")

    # Also save stats
    stats_path = OUTPUT_DIR / "statcan_spike_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved stats to {stats_path}")

    # Print sample rows
    print("\n  Sample data (first 5 rows):")
    for row in rows[:5]:
        geo = row.get("GEO", "")
        crop = row.get("Type of crop", row.get("Crop type", ""))
        ref_date = row.get("REF_DATE", "")
        value = row.get("VALUE", "")
        uom = row.get("UOM", "")
        print(f"    {ref_date} | {geo} | {crop} | {value} {uom}")


def main():
    print("=" * 60)
    print("ClimatePulse Data Spike: StatsCan Table 32-10-0359-01")
    print("Target: Prairie crop yields (wheat, canola, barley)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download the data
    csv_path = try_wds_api()

    if csv_path is None:
        # Fallback: confirm table exists via HTML
        exists = try_wds_json_api()
        if exists:
            print("\n⚠ Table confirmed but bulk download blocked.")
            print("  Fallback: Use statscan package or manual download.")
            print("  pip install stats_can")
        else:
            print("\n✗ Could not access StatsCan table at all.")
        sys.exit(1)

    # Step 2: Analyze and filter
    rows, headers, stats = analyze_crop_data(csv_path)

    # Step 3: Save filtered data
    save_filtered_data(rows, headers, stats)

    # Step 4: Verdict
    print("\n" + "=" * 60)
    if stats["prairie_target_rows"] > 50:
        print("✓ SPIKE PASSED: StatsCan crop yield data is viable")
        print(f"  {stats['prairie_target_rows']:,} rows for prairie crops")
        print(f"  Year range: {stats['year_range']}")
    elif stats["total_rows"] > 0:
        print("⚠ SPIKE PARTIAL: Data exists but filtering needs adjustment")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Check crop names and GEO matching")
    else:
        print("✗ SPIKE FAILED: No data retrieved")
    print("=" * 60)


if __name__ == "__main__":
    main()

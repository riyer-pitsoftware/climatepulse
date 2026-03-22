# Zerve Block: fetch_all_prairie_weather
# DAG position: 4 of 4 (after 03_fetch_eccc_stations)
# Inputs: YEAR_START, YEAR_END from block 03 (does NOT use PRAIRIE_STATIONS)
# Outputs: raw_monthly_df (DataFrame — all ECCC monthly climate records for AB/SK/MB,
#          1990-2023, all stations in the GeoMet collection)
#
# WHAT THIS DOES: Queries the ECCC GeoMet OGC API (climate-monthly collection) for
#   every province+year combination (3 provinces x 34 years = 102 paginated fetches).
#   Collects all station-month records returned by the API into a single DataFrame.
#   No filtering to specific stations, no growing-season subsetting, no feature engineering.
#
# DIVERGENCES FROM GROUND TRUTH (scripts/pipeline_eccc_weather.py):
#   - Uses GeoMet OGC API (api.weather.gc.ca/collections/climate-monthly/items) returning
#     MONTHLY pre-aggregated data; ground truth uses bulk CSV download endpoint
#     (climate.weather.gc.ca/climate_data/bulk_data_e.html) returning DAILY data
#   - Fetches ALL stations province-wide; ground truth fetches only 10 curated stations
#   - Returns raw monthly records with no feature engineering; ground truth computes 8
#     derived features (GDD, heat stress days, precip splits, max dry spell, frost-free
#     days, mean growing temp) from daily data
#   - No growing season filter (May-Sep); ground truth filters to May 1 – Sep 30
#   - Does not use the station list from block 03 at all — only uses YEAR_START/YEAR_END;
#     ground truth stations dict drives the entire fetch loop
#   - Ignores station ID changes over time since it queries by province, not station
#   - Outputs province-wide raw monthly data; ground truth outputs province-level
#     annual feature vectors averaged across curated stations


import requests
import pandas as pd
import time

BASE_GEOMET = "https://api.weather.gc.ca/collections/climate-monthly/items"
PAGE_SIZE   = 500
PROVINCES   = ["AB", "SK", "MB"]

def fetch_province_year(province: str, year: int) -> list:
    """Fetch all monthly records for one province+year via pagination."""
    records = []
    offset  = 0
    while True:
        params = {
            "f":             "json",
            "limit":         PAGE_SIZE,
            "offset":        offset,
            "PROVINCE_CODE": province,
            "LOCAL_YEAR":    year,
        }
        resp = requests.get(BASE_GEOMET, params=params, timeout=60)
        resp.raise_for_status()
        _j = resp.json()
        features   = _j.get("features", [])
        n_returned = _j.get("numberReturned", 0)
        records.extend(f["properties"] for f in features)
        offset += n_returned
        if n_returned < PAGE_SIZE:
            break
        time.sleep(0.05)
    return records

# ── Full fetch: 3 provinces × 34 years ──────────────────────────────────────
prairie_records = []
total_combos    = len(PROVINCES) * (YEAR_END - YEAR_START + 1)
done            = 0

for prov in PROVINCES:
    prov_total = 0
    for yr in range(YEAR_START, YEAR_END + 1):
        rows         = fetch_province_year(prov, yr)
        prairie_records.extend(rows)
        prov_total  += len(rows)
        done        += 1
        if done % 10 == 0:
            pct = 100 * done / total_combos
            print(f"  [{pct:4.0f}%] {prov} {yr}: {len(rows)} recs | running total: {len(prairie_records):,}")
        time.sleep(0.1)
    print(f"✓ {prov}: {prov_total:,} station-month records")

# ── Build DataFrame ──────────────────────────────────────────────────────────
raw_monthly_df = pd.DataFrame(prairie_records)

print(f"\n{'='*55}")
print(f"Total raw records   : {len(raw_monthly_df):,}")
print(f"Provinces           : {sorted(raw_monthly_df['PROVINCE_CODE'].unique())}")
print(f"Year range          : {raw_monthly_df['LOCAL_YEAR'].min()} – {raw_monthly_df['LOCAL_YEAR'].max()}")
print(f"Unique stations     : {raw_monthly_df['CLIMATE_IDENTIFIER'].nunique():,}")
print(f"Shape               : {raw_monthly_df.shape}")

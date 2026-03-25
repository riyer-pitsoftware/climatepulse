# Zerve Block: fetch_all_prairie_weather  (Fleet version)
# DAG position: after fleet (markdown) / fetch_eccc_stations
# Inputs: YEAR_START, YEAR_END (from upstream blocks)
# Outputs: slice_records (list of dicts for one province+year)
#
# WHAT THIS DOES: Uses spread() to fan out 102 (province, year) combinations across
#   Fleet workers. Each execution fetches a single province+year via fetch_province_year().
#   An AGGREGATOR block downstream gathers all slices into raw_monthly_df.

import requests
import time

YEAR_START = 2000
YEAR_END = 2024

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
        time.sleep(0.1)
    return records


# ── Build full list of (province, year) combos and fan out via Fleet ─────────
all_combos = [
    (prov, yr)
    for prov in PROVINCES
    for yr in range(YEAR_START, YEAR_END + 1)
]  # 3 × 25 = 75 tuples

# spread() fans out one execution per element in the list
combo = spread(all_combos)

# Each execution receives a SlicedIterable — .data holds this slice's tuples
# With one element per slice, combo.data[0] is the single (province, year) tuple
province, year = combo.data[0]
slice_records = fetch_province_year(province, year)

print(f"[{province} {year}] fetched {len(slice_records):,} records")

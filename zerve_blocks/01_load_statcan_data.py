# Zerve Block: load_statcan_data
# DAG position: 1 of 4 (leftmost, parallel with 03_fetch_eccc_stations)
# Inputs: none — downloads two ZIP CSVs from StatsCan over HTTPS
# Outputs: yields_raw (DataFrame, full 32-10-0359) and prices_raw (DataFrame, full 32-10-0077)
#
# WHAT THIS DOES: Downloads the complete StatsCan tables 32-10-0359 (field crop
#   yields/production) and 32-10-0077 (monthly farm product prices) as ZIP CSVs,
#   extracts them in memory, and loads each into a pandas DataFrame. No filtering
#   or cleaning is done here — that happens in block 02.
#
# DIVERGENCES FROM GROUND TRUTH (scripts/):
#   - Downloads both tables (yields + prices) in one block; ground truth splits
#     these into two separate scripts (pipeline_statcan_yields.py, pipeline_statcan_prices.py)
#   - Reads ZIPs entirely in memory (BytesIO) instead of caching to data/raw/statcan/
#   - Uses pandas read_csv with latin-1 encoding; ground truth uses stdlib csv with utf-8-sig
#   - Returns raw DataFrames with all rows/columns; ground truth filters to 4 crops
#     (Wheat, Canola, Barley, Oats) and Prairie provinces during the parse step
#   - Does not extract harvested area or production dispositions (only done in block 02
#     for yields); ground truth extracts yield, area, production, and price in one pass


import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── FOUND IT! ─────────────────────────────────────────────────────────────────
# The correct download URL is:
#   https://www150.statcan.gc.ca/n1/tbl/csv/32100359-eng.zip
# (without the dashes in the table ID, just 8 digits + -eng.zip)
# This was found in the ZIP links on the tv.action page.

PRAIRIE_PROVINCES = {"Alberta", "Saskatchewan", "Manitoba"}
PROV_ABBR = {"Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB"}

def fetch_statscan_zip_csv(table_id_nodash: str) -> pd.DataFrame:
    """
    Download Statistics Canada table CSV zip.
    table_id_nodash: 8-digit table ID without dashes, e.g. '32100359'
    """
    url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{table_id_nodash}-eng.zip"
    print(f"  Downloading: {url}")
    r = requests.get(url, headers=HEADERS, timeout=180, stream=True)
    r.raise_for_status()
    
    content = BytesIO(r.content)
    z = ZipFile(content)
    print(f"  ZIP contents: {z.namelist()}")
    
    # Pick the data file (not MetaData)
    data_file = next(n for n in z.namelist() if "MetaData" not in n and n.endswith(".csv"))
    df = pd.read_csv(z.open(data_file), encoding="latin-1", low_memory=False)
    return df

print("=" * 60)
print("Fetching Table 32-10-0359 (Field Crop Yields & Production)")
print("=" * 60)
yields_raw = fetch_statscan_zip_csv("32100359")
print(f"Shape: {yields_raw.shape}")
print(f"Columns: {yields_raw.columns.tolist()}")
print(f"\nSample GEO values: {yields_raw['GEO'].unique()[:10] if 'GEO' in yields_raw.columns else 'N/A'}")
print()

print("=" * 60)
print("Fetching Table 32-10-0077 (Farm Prices of Agricultural Products)")
print("=" * 60)
prices_raw = fetch_statscan_zip_csv("32100077")
print(f"Shape: {prices_raw.shape}")
print(f"Columns: {prices_raw.columns.tolist()}")
print(f"\nSample GEO values: {prices_raw['GEO'].unique()[:10] if 'GEO' in prices_raw.columns else 'N/A'}")

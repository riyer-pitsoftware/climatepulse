# Zerve Block: clean_and_filter_data
# DAG position: 2 of 4 (after 01_load_statcan_data)
# Inputs: yields_raw, prices_raw (DataFrames from block 01)
# Outputs: crop_yields_df (DataFrame — year, province, crop_type, yield_kg_per_ha, ...),
#          farm_prices_df (DataFrame — date, year, month, province, crop_type, price_value, ...)
#
# WHAT THIS DOES: Filters both raw StatsCan DataFrames to Prairie provinces (AB, SK, MB).
#   For yields (32-10-0359): keeps only "Average yield (kilograms per hectare)" rows,
#   retains ~19 crop types (PRAIRIE_CROPS set), adds province abbreviation, drops NaN values.
#   For prices (32-10-0077): regex-matches ~15 crop keywords against "Farm products" column,
#   parses YYYY-MM REF_DATE into date/year/month, drops NaN prices.
#
# DIVERGENCES FROM GROUND TRUTH (scripts/):
#   - Retains ~19 crop types (all Prairie field crops); ground truth keeps only 4
#     (Wheat, Canola, Barley, Oats) via CROP_MAP with canonical name normalization
#   - Does not extract harvested area (hectares) or production (metric tonnes);
#     ground truth builds a merged yield+area+production table
#   - Does not filter to year >= 2000; keeps full history from StatsCan
#   - Prices: uses broad keyword regex matching (~15 keywords including lentil, mustard,
#     sunflower, etc.); ground truth uses only 4 keywords (wheat, canola, barley, oats)
#   - Prices: keeps monthly granularity with date column; ground truth pipeline_statcan_prices.py
#     also keeps monthly but uses different column names (ref_date, commodity, price, uom)
#   - Uses pandas throughout; ground truth uses stdlib csv with manual parsing


import pandas as pd
import numpy as np

# ── Prairie province config ───────────────────────────────────────────────────
PRAIRIE_PROVINCES = {"Alberta", "Saskatchewan", "Manitoba"}
PROV_ABBR = {"Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB"}

# Key crop types to retain (common Prairie field crops)
PRAIRIE_CROPS = {
    "Wheat, all", "Wheat", "Spring wheat", "Winter wheat",
    "Durum wheat", "Canola (rapeseed)", "Barley", "Oats",
    "Flaxseed", "Rye", "Lentils", "Peas, dry", "Canary seed",
    "Mustard seed", "Corn for grain", "Mixed grains",
    "Chick peas", "Faba beans", "Sunflower seed",
}

# Farm price product keywords to match (Table 32-10-0077 has coded names)
PRICE_CROP_KEYWORDS = [
    "wheat", "durum", "canola", "rapeseed", "barley", "oats",
    "flax", "rye", "lentil", "peas", "corn", "mustard",
    "canary seed", "chick pea", "sunflower",
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. CLEAN CROP YIELDS (Table 32-10-0359)
# ──────────────────────────────────────────────────────────────────────────────
yld = yields_raw.copy()
# Fix BOM in column names
yld.columns = [c.replace('ï»¿"', '').replace('"', '').strip() for c in yld.columns]

# Filter: Prairie provinces only
yld = yld[yld["GEO"].isin(PRAIRIE_PROVINCES)].copy()

# Filter: yield rows only (kg/ha is the preferred SI unit)
yld_kg_ha = yld[yld["Harvest disposition"] == "Average yield (kilograms per hectare)"].copy()

# Parse year
yld_kg_ha["year"] = yld_kg_ha["REF_DATE"].astype(int)

# Rename and select columns
crop_yields_df = yld_kg_ha.rename(columns={
    "GEO": "province_name",
    "Type of crop": "crop_type",
    "VALUE": "yield_kg_per_ha",
    "UOM": "yield_unit",
    "SCALAR_FACTOR": "scalar_factor",
    "STATUS": "status_flag",
})[["year", "province_name", "crop_type", "yield_kg_per_ha", "yield_unit", "status_flag"]].copy()

# Add province abbreviation
crop_yields_df["province"] = crop_yields_df["province_name"].map(PROV_ABBR)

# Drop rows where VALUE is NaN (suppressed data)
crop_yields_df_clean = crop_yields_df.dropna(subset=["yield_kg_per_ha"]).reset_index(drop=True)

# Final column order
crop_yields_df = crop_yields_df_clean[[
    "year", "province", "province_name", "crop_type", "yield_kg_per_ha", "yield_unit", "status_flag"
]]

print("=" * 65)
print("CROP YIELDS DATAFRAME — Table 32-10-0359")
print("=" * 65)
print(f"Shape: {crop_yields_df.shape}")
print(f"Year range: {crop_yields_df['year'].min()} – {crop_yields_df['year'].max()}")
print(f"Provinces: {sorted(crop_yields_df['province'].unique())}")
print(f"Crops ({crop_yields_df['crop_type'].nunique()}): {sorted(crop_yields_df['crop_type'].unique())[:10]} ...")
print(f"\nNull counts:\n{crop_yields_df.isnull().sum()}")
print(f"\nSample:")
print(crop_yields_df.head(8).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 2. CLEAN FARM PRICES (Table 32-10-0077)
# ──────────────────────────────────────────────────────────────────────────────
prx = prices_raw.copy()
prx.columns = [c.replace('ï»¿"', '').replace('"', '').strip() for c in prx.columns]

# Filter: Prairie provinces only
prx = prx[prx["GEO"].isin(PRAIRIE_PROVINCES)].copy()

# Filter: crop-related products only (many animal/livestock products exist)
mask_crops = prx["Farm products"].str.lower().str.contains(
    "|".join(PRICE_CROP_KEYWORDS), na=False
)
prx = prx[mask_crops].copy()

# Parse dates — REF_DATE is "YYYY-MM" format (monthly)
prx["date"] = pd.to_datetime(prx["REF_DATE"], format="%Y-%m", errors="coerce")
prx["year"] = prx["date"].dt.year
prx["month"] = prx["date"].dt.month

# Rename and select
farm_prices_df = prx.rename(columns={
    "GEO": "province_name",
    "Farm products": "crop_type",
    "VALUE": "price_value",
    "UOM": "price_unit",
    "STATUS": "status_flag",
})[["date", "year", "month", "province_name", "crop_type", "price_value", "price_unit", "status_flag"]].copy()

# Add province abbreviation
farm_prices_df["province"] = farm_prices_df["province_name"].map(PROV_ABBR)

# Drop rows where price is NaN
farm_prices_df = farm_prices_df.dropna(subset=["price_value"]).reset_index(drop=True)

# Final column order
farm_prices_df = farm_prices_df[[
    "date", "year", "month", "province", "province_name", "crop_type", "price_value", "price_unit", "status_flag"
]]

print("\n\n" + "=" * 65)
print("FARM PRICES DATAFRAME — Table 32-10-0077")
print("=" * 65)
print(f"Shape: {farm_prices_df.shape}")
print(f"Date range: {farm_prices_df['date'].min().date()} – {farm_prices_df['date'].max().date()}")
print(f"Year range: {farm_prices_df['year'].min()} – {farm_prices_df['year'].max()}")
print(f"Provinces: {sorted(farm_prices_df['province'].unique())}")
print(f"Products ({farm_prices_df['crop_type'].nunique()}): {sorted(farm_prices_df['crop_type'].unique())[:10]} ...")

# ══════════════════════════════════════════════════════════════════════
# RECONCILIATION FILTER — align with ground truth (4 crops, 2000-2024)
# ══════════════════════════════════════════════════════════════════════
# Ground truth uses exactly 4 canonical crops; the agent kept ~19.
# We filter yields here and normalize column names for downstream blocks.

CANONICAL_CROPS = {
    "Wheat, all": "Wheat",
    "Canola (rapeseed)": "Canola",
    "Barley": "Barley",
    "Oats": "Oats",
}

# Filter yields to 4 crops and rename
crop_yields_df = crop_yields_df[crop_yields_df["crop_type"].isin(CANONICAL_CROPS.keys())].copy()
crop_yields_df["crop"] = crop_yields_df["crop_type"].map(CANONICAL_CROPS)

# Filter to 2000+ (model needs 2000-2024 window)
crop_yields_df = crop_yields_df[crop_yields_df["year"] >= 2000].reset_index(drop=True)

# Use full province names (ground truth convention)
ABBR_TO_FULL = {"AB": "Alberta", "SK": "Saskatchewan", "MB": "Manitoba"}
crop_yields_df["province"] = crop_yields_df["province"].map(ABBR_TO_FULL)

# Rename yield column to match ground truth
crop_yields_df = crop_yields_df.rename(columns={"yield_kg_per_ha": "yield_kg_ha"})

# Build the df_yields that downstream blocks expect
df_yields = crop_yields_df[["year", "province", "crop", "yield_kg_ha"]].copy()

print(f"\n{'=' * 65}")
print(f"RECONCILED df_yields (4 crops, 2000+)")
print(f"{'=' * 65}")
print(f"Shape: {df_yields.shape}  (expect ~312 rows)")
print(f"Crops: {sorted(df_yields['crop'].unique())}")
print(f"Years: {df_yields['year'].min()}–{df_yields['year'].max()}")
print(f"Provinces: {sorted(df_yields['province'].unique())}")

# Reconcile prices — keep monthly, normalize to df_prices_monthly
farm_prices_df["province"] = farm_prices_df["province"].map(ABBR_TO_FULL)
df_prices_monthly = farm_prices_df.rename(columns={
    "date": "ref_date",
    "crop_type": "commodity",
    "price_value": "price",
    "price_unit": "uom",
})[["ref_date", "year", "province", "commodity", "price", "uom"]].copy()

# Filter to relevant years
df_prices_monthly = df_prices_monthly[df_prices_monthly["year"] >= 2000].reset_index(drop=True)

print(f"\n✓ df_prices_monthly: {len(df_prices_monthly):,} rows")
print(f"  Commodities: {sorted(df_prices_monthly['commodity'].unique())[:8]}...")
print(f"\nNull counts:\n{farm_prices_df.isnull().sum()}")
print(f"\nSample:")
print(farm_prices_df.head(8).to_string(index=False))


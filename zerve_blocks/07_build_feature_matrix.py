# Zerve Block: build_feature_matrix (RECONCILED — replaces agent version)
# DAG position: fan-in join (after 02_clean_and_filter + 03_06_weather_upload)
# Inputs: df_yields (~312 rows), df_prices_monthly, df_weather (75 rows) from upstream
# Outputs: df_features (300 rows × 19 columns)
#
# WHAT THIS DOES: Three-way join of yields + annualized prices + weather on
#   (province, year). Adds lagged features (prev_year_yield, prev_year_precip,
#   prev_year_gdd) and price_change_pct. This matches the ground truth
#   scripts/pipeline_feature_matrix.py.
#
# RECONCILIATION APPLIED (vs agent version):
#   - Uses only 4 crops (from reconciled df_yields in block 02)
#   - 300 rows × 19 cols (not 2,960 × 28)
#   - No yield_roll3/roll5 (potential leakage via rolling windows)
#   - Lag features via groupby shift (prev_year_*), not rolling averages
#   - Price commodity mapping matches ground truth exactly
#   - No imputation — blanks are left as-is for the model to handle

import pandas as pd
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════
# 1. ANNUALIZE MONTHLY PRICES
# ══════════════════════════════════════════════════════════════════════
PRICE_COMMODITY_MAP = {
    "Wheat (except durum wheat) [1121111]": "Wheat",
    "Canola (including rapeseed) [113111]": "Canola",
    "Barley [1151141]": "Barley",
    "Oats [115113111]": "Oats",
}

df_pm = df_prices_monthly.copy()
df_pm["crop"] = df_pm["commodity"].map(PRICE_COMMODITY_MAP)
df_pm = df_pm.dropna(subset=["crop"])
# Handle ref_date as string or datetime
if df_pm["ref_date"].dtype == object:
    df_pm["year"] = pd.to_datetime(df_pm["ref_date"]).dt.year
price_annual = df_pm.groupby(["year", "province", "crop"])["price"].mean().round(2).reset_index()
price_annual.columns = ["year", "province", "crop", "price_cad_per_tonne"]

print(f"Annualized prices: {len(price_annual)} rows")

# ══════════════════════════════════════════════════════════════════════
# 2. MERGE YIELDS + WEATHER
# ══════════════════════════════════════════════════════════════════════
wx_cols = [c for c in df_weather.columns if c != "n_stations"]
df = df_yields.merge(df_weather[wx_cols], on=["year", "province"], how="left")

# ══════════════════════════════════════════════════════════════════════
# 3. MERGE PRICES
# ══════════════════════════════════════════════════════════════════════
df = df.merge(price_annual, on=["year", "province", "crop"], how="left")

# ══════════════════════════════════════════════════════════════════════
# 4. LAGGED FEATURES (prev_year_*)
# ══════════════════════════════════════════════════════════════════════
df = df.sort_values(["province", "crop", "year"]).reset_index(drop=True)

df["prev_year_yield_kg_ha"] = df.groupby(["province", "crop"])["yield_kg_ha"].shift(1)
df["prev_year_precip_mm"] = df.groupby(["province", "crop"])["precip_total_mm"].shift(1)
df["prev_year_gdd"] = df.groupby(["province", "crop"])["gdd_total"].shift(1)

# ══════════════════════════════════════════════════════════════════════
# 5. PRICE CHANGE (year-over-year %)
# ══════════════════════════════════════════════════════════════════════
df["price_change_pct"] = (
    df.groupby(["province", "crop"])["price_cad_per_tonne"].pct_change() * 100
).round(2)

# ══════════════════════════════════════════════════════════════════════
# 6. FILTER TO FINAL WINDOW
# ══════════════════════════════════════════════════════════════════════
df_features = df[(df["year"] >= 2000) & (df["year"] <= 2024)].copy()

# ══════════════════════════════════════════════════════════════════════
# 7. VALIDATION
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print(f"FEATURE MATRIX — VALIDATION")
print(f"{'=' * 65}")
print(f"  Shape: {df_features.shape}  (expect ~300 × 19)")
print(f"  Years: {df_features['year'].min()}–{df_features['year'].max()}")
print(f"  Provinces: {sorted(df_features['province'].unique())}")
print(f"  Crops: {sorted(df_features['crop'].unique())}")

total = len(df_features)
has_yield = df_features["yield_kg_ha"].notna().sum()
has_weather = df_features["gdd_total"].notna().sum()
has_price = df_features["price_cad_per_tonne"].notna().sum()
print(f"\n  Completeness:")
print(f"    Has yield:   {has_yield}/{total} ({has_yield/total*100:.0f}%)")
print(f"    Has weather: {has_weather}/{total} ({has_weather/total*100:.0f}%)")
print(f"    Has price:   {has_price}/{total} ({has_price/total*100:.0f}%)")

drought = df_features[(df_features["year"] == 2021) & (df_features["crop"] == "Wheat")]
if len(drought):
    print(f"\n  2021 Drought — Wheat:")
    for _, r in drought.iterrows():
        print(f"    {r['province']}: yield={r['yield_kg_ha']:.0f}kg/ha, "
              f"GDD={r['gdd_total']:.0f}, heat={r['heat_stress_days']:.0f}d, "
              f"precip={r['precip_total_mm']:.0f}mm")

print(f"\n  Columns: {list(df_features.columns)}")
print(f"\n✅ df_features ready for modelling")

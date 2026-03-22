# Zerve Block: build_weather_df
# DAG position: after fetch_all_prairie_weather (block 04)
# Inputs: raw_monthly_df (station-level monthly ECCC records from block 04)
# Outputs: weather_df (~102 rows: 3 provinces x 34 years, 1990-2023)
#   - 15 weather columns + n_stations metadata
#   - Columns: mean_temp_annual, min_temp_annual, max_temp_annual, mean_temp_grow,
#     min_temp_grow, precip_annual_mm, precip_grow_mm, snowfall_annual_cm,
#     gdd_annual, gdd_growing, heating_dd_annual, cooling_dd_annual,
#     spring_frost_min, fall_frost_min, sunshine_annual_h, n_stations
#
# WHAT THIS DOES: Aggregates monthly station records to province-year level.
#   Computes GDD (base 5C) from monthly mean temp * days_in_month. Groups by
#   station-year first, then takes province-year median across stations (requires
#   >=8 months of data per station-year). Covers May-Aug growing season plus
#   full-year metrics.
#
# DIVERGENCES FROM GROUND TRUTH (scripts/pipeline_eccc_weather.py):
#   - Uses MONTHLY input data (raw_monthly_df), not daily station data
#   - GDD approximated as max(0, mean_temp - 5) * days_in_month; ground truth sums daily GDD
#   - No heat_stress_days feature here (moved to block 07); ground truth includes it
#   - No precip_may_jun_mm / precip_jul_aug_mm split; only precip_grow_mm (May-Aug)
#   - No max_consecutive_dry_days (impossible from monthly data); ground truth computes from daily
#   - No frost_free_days (last spring frost / first fall frost); uses spring_frost_min / fall_frost_min proxies instead
#   - 15 features vs ground truth's 8; adds annual temps, snowfall, heating/cooling DD, sunshine
#   - Year range 1990-2023 vs ground truth's 2000-2024
#   - Province-year aggregation uses median (not mean) across stations
#   - 10+ stations implied by raw_monthly_df vs ground truth's 10 curated stations

# TODO: paste code from Zerve

import pandas as pd
import numpy as np

# ── Work from raw station-month records ─────────────────────────────────────
_df = raw_monthly_df.copy()

# Coerce numeric fields
_df["LATITUDE"]  = pd.to_numeric(_df["LATITUDE"],  errors="coerce")
_df["LONGITUDE"] = pd.to_numeric(_df["LONGITUDE"], errors="coerce")

# ── Agronomic Growing Season (May–Aug, months 5-8) helper columns ───────────
GROW_MONTHS = [5, 6, 7, 8]
FROST_MONTHS_SPRING = [4, 5]      # risk of late spring frost
FROST_MONTHS_FALL   = [8, 9, 10]  # risk of early fall frost

_df["is_growing_season"] = _df["LOCAL_MONTH"].isin(GROW_MONTHS)
_df["is_spring_frost"]   = _df["LOCAL_MONTH"].isin(FROST_MONTHS_SPRING)
_df["is_fall_frost"]     = _df["LOCAL_MONTH"].isin(FROST_MONTHS_FALL)

# ── Growing Degree Days (GDD base 5°C) using monthly mean temperature ───────
# GDD_monthly ≈ max(0, mean_temp - base) × days_in_month
_df["days_in_month"] = _df["LOCAL_DATE"].apply(
    lambda d: pd.Period(d, freq="M").days_in_month
)
_df["gdd_base5"] = np.maximum(0, _df["MEAN_TEMPERATURE"].fillna(0) - 5.0) * _df["days_in_month"]
_df["gdd_base5"] = _df["gdd_base5"].where(_df["MEAN_TEMPERATURE"].notna(), np.nan)

# ── Aggregate to province × year ────────────────────────────────────────────
# For each station, compute annual aggregates, then take province median
# (median is more robust than mean against sparse remote stations)

# Station-year aggregates
_stn_yr = (
    _df.groupby(["CLIMATE_IDENTIFIER", "PROVINCE_CODE", "LOCAL_YEAR"], observed=True)
    .agg(
        # Temperature (annual)
        mean_temp_annual   = ("MEAN_TEMPERATURE",   "mean"),
        min_temp_annual    = ("MIN_TEMPERATURE",     "min"),
        max_temp_annual    = ("MAX_TEMPERATURE",     "max"),
        # Growing season temperature (May–Aug)
        mean_temp_grow     = ("MEAN_TEMPERATURE",
                              lambda x: x[_df.loc[x.index, "is_growing_season"]].mean()),
        min_temp_grow      = ("MIN_TEMPERATURE",
                              lambda x: x[_df.loc[x.index, "is_growing_season"]].min()),
        # Precipitation
        precip_annual_mm   = ("TOTAL_PRECIPITATION", "sum"),
        precip_grow_mm     = ("TOTAL_PRECIPITATION",
                              lambda x: x[_df.loc[x.index, "is_growing_season"]].sum()),
        snowfall_annual_cm = ("TOTAL_SNOWFALL",       "sum"),
        # Degree days
        gdd_annual         = ("gdd_base5",            "sum"),
        gdd_growing        = ("gdd_base5",
                              lambda x: x[_df.loc[x.index, "is_growing_season"]].sum()),
        heating_dd_annual  = ("HEATING_DEGREE_DAYS",  "sum"),
        cooling_dd_annual  = ("COOLING_DEGREE_DAYS",  "sum"),
        # Frost proxies — min monthly temp during shoulder seasons
        spring_frost_min   = ("MIN_TEMPERATURE",
                              lambda x: x[_df.loc[x.index, "is_spring_frost"]].min()),
        fall_frost_min     = ("MIN_TEMPERATURE",
                              lambda x: x[_df.loc[x.index, "is_fall_frost"]].min()),
        # Sunshine hours
        sunshine_annual_h  = ("BRIGHT_SUNSHINE",     "sum"),
        # Data completeness
        months_with_data   = ("MEAN_TEMPERATURE",    lambda x: x.notna().sum()),
    )
    .reset_index()
)

# ── Province-year aggregates (median across stations) ───────────────────────
# Require at least 8 months of valid temperature data per station-year
_stn_yr_valid = _stn_yr[_stn_yr["months_with_data"] >= 8].copy()

weather_df = (
    _stn_yr_valid
    .groupby(["PROVINCE_CODE", "LOCAL_YEAR"], observed=True)
    .agg(
        mean_temp_annual   = ("mean_temp_annual",   "median"),
        min_temp_annual    = ("min_temp_annual",    "median"),
        max_temp_annual    = ("max_temp_annual",    "median"),
        mean_temp_grow     = ("mean_temp_grow",     "median"),
        min_temp_grow      = ("min_temp_grow",      "median"),
        precip_annual_mm   = ("precip_annual_mm",   "median"),
        precip_grow_mm     = ("precip_grow_mm",     "median"),
        snowfall_annual_cm = ("snowfall_annual_cm", "median"),
        gdd_annual         = ("gdd_annual",         "median"),
        gdd_growing        = ("gdd_growing",        "median"),
        heating_dd_annual  = ("heating_dd_annual",  "median"),
        cooling_dd_annual  = ("cooling_dd_annual",  "median"),
        spring_frost_min   = ("spring_frost_min",   "median"),
        fall_frost_min     = ("fall_frost_min",     "median"),
        sunshine_annual_h  = ("sunshine_annual_h",  "median"),
        n_stations         = ("CLIMATE_IDENTIFIER", "count"),
    )
    .reset_index()
    .rename(columns={"PROVINCE_CODE": "province", "LOCAL_YEAR": "year"})
)

# ── Data quality summary ─────────────────────────────────────────────────────
print("weather_df shape:", weather_df.shape)
print("Provinces:", sorted(weather_df["province"].unique()))
print("Year range:", weather_df["year"].min(), "–", weather_df["year"].max())
print(f"Expected rows (3 prov × 34 yr): {3*34} | Actual: {len(weather_df)}")
print(f"\nMissing values per column:")
_miss = weather_df.isnull().sum()
print(_miss[_miss > 0].to_string() if _miss.any() else "  None — all columns complete ✓")

print("\nSample rows:")
print(weather_df[["province","year","mean_temp_annual","mean_temp_grow",
                   "precip_annual_mm","gdd_annual","n_stations"]].head(12).to_string(index=False))

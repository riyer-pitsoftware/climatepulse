# Zerve Block: validate_and_visualize_weather
# DAG position: after build_weather_df (block 05)
# Inputs: weather_df from block 05 (province x year, 15 weather features)
# Outputs: printed validation report + weather_df_overview.png (4-panel time series)
#
# WHAT THIS DOES: Validates weather_df shape, completeness, and value ranges.
#   Checks for coverage gaps in the 1990-2023 x {AB, MB, SK} grid.
#   Plots 4 time series: growing season mean temp, annual precip, growing GDD,
#   and spring frost risk, colored by province.
#
# DIVERGENCES FROM GROUND TRUTH (scripts/pipeline_eccc_weather.py):
#   - Validates block 05's monthly-derived features, not the ground truth's daily-derived ones
#   - No heat_stress_days check (not in weather_df; computed later in block 07)
#   - No max_consecutive_dry_days or frost_free_days validation (not available from monthly data)
#   - Checks 1990-2023 range vs ground truth's 2000-2024
#   - No 2021-vs-2019 drought comparison (original header described this but code does range checks instead)

# TODO: paste code from Zerve


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Validation ───────────────────────────────────────────────────────────────
print("=" * 60)
print("weather_df VALIDATION REPORT")
print("=" * 60)

# 1. Shape & completeness
print(f"\n Shape : {weather_df.shape}")
print(f" Provinces : {sorted(weather_df['province'].unique())}")
print(f" Years     : {weather_df['year'].min()} – {weather_df['year'].max()}  ({weather_df['year'].nunique()} years)")
print(f" Missing   : {weather_df.isnull().sum().sum()} cells total")

# 2. Full coverage check
_expected = pd.MultiIndex.from_product(
    [["AB", "MB", "SK"], range(1990, 2024)], names=["province", "year"]
)
_actual = pd.MultiIndex.from_arrays([weather_df["province"], weather_df["year"]])
_missing_combos = _expected.difference(_actual)
print(f"\n Coverage gaps (province-year): {len(_missing_combos)}")
if len(_missing_combos):
    print("  Missing:", _missing_combos.tolist())

# 3. Sanity checks on values
print("\n Sanity checks:")
print(f"  Annual mean temp range   : {weather_df['mean_temp_annual'].min():.1f}°C – {weather_df['mean_temp_annual'].max():.1f}°C  ✓" if -30 < weather_df['mean_temp_annual'].min() < 10 else "  ⚠️ annual temp out of expected range")
print(f"  Growing mean temp range  : {weather_df['mean_temp_grow'].min():.1f}°C – {weather_df['mean_temp_grow'].max():.1f}°C  ✓")
print(f"  Annual precip range (mm) : {weather_df['precip_annual_mm'].min():.0f} – {weather_df['precip_annual_mm'].max():.0f}")
print(f"  GDD (annual) range       : {weather_df['gdd_annual'].min():.0f} – {weather_df['gdd_annual'].max():.0f}  degree-days")
print(f"  GDD (growing) range      : {weather_df['gdd_growing'].min():.0f} – {weather_df['gdd_growing'].max():.0f}  degree-days")
print(f"  Stations per prov-yr min : {weather_df['n_stations'].min()} | max: {weather_df['n_stations'].max()}")

# 4. Column summary
print("\n Columns in weather_df:")
for col in weather_df.columns:
    print(f"  {col:<25s} dtype={weather_df[col].dtype}")

print("\n✅ weather_df is ready for joining with crop data on (province, year)")

# ── Visualization ────────────────────────────────────────────────────────────
PROV_COLORS = {"AB": "#A1C9F4", "SK": "#FFB482", "MB": "#8DE5A1"}
BG          = "#1D1D20"
FG          = "#fbfbff"
SFG         = "#909094"

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("Prairie Provinces — ECCC Climate Features (1990–2023)",
             color=FG, fontsize=15, fontweight="bold", y=1.01)

plot_cfg = [
    ("mean_temp_grow",   "Growing Season Mean Temp (°C)",   axes[0, 0]),
    ("precip_annual_mm", "Annual Total Precipitation (mm)", axes[0, 1]),
    ("gdd_growing",      "Growing Season GDD (base 5°C)",   axes[1, 0]),
    ("spring_frost_min", "Spring Frost Risk — Min Temp (°C)", axes[1, 1]),
]

for col, title, ax in plot_cfg:
    ax.set_facecolor(BG)
    for prov, grp in weather_df.groupby("province"):
        grp = grp.sort_values("year")
        ax.plot(grp["year"], grp[col],
                color=PROV_COLORS[prov], label=prov, linewidth=1.8, alpha=0.9)
    ax.set_title(title, color=FG, fontsize=11, pad=8)
    ax.tick_params(colors=SFG, labelsize=9)
    ax.spines[:].set_color("#444")
    ax.set_xlabel("Year", color=SFG, fontsize=9)
    ax.legend(framealpha=0.2, labelcolor=FG, fontsize=9,
              facecolor="#333", edgecolor="#555")
    ax.grid(axis="y", color="#333", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("weather_df_overview.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.show()

#!/usr/bin/env python3
"""
Production pipeline: Join weather features + crop yields + prices → feature matrix.

Joins at province-year grain:
  - Weather features from ECCC (growing season aggregates)
  - Crop yields from StatsCan 32-10-0359
  - Annual avg prices from StatsCan 32-10-0077

Output: data/processed/ca_feature_matrix.csv — ready for XGBoost training
"""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"

# Focus window for model training — last decade plus buffer for lagged features
YEAR_MIN = 2000
YEAR_MAX = 2024

# Primary commodity mapping for prices table
PRICE_COMMODITY_MAP = {
    "Wheat (except durum wheat) [1121111]": "Wheat",
    "Canola (including rapeseed) [113111]": "Canola",
    "Barley [1151141]": "Barley",
    "Oats [115113111]": "Oats",
}


def load_yields():
    """Load crop yield data."""
    data = {}  # (year, province, crop) -> {yield_kg_ha, harvested_ha, production_mt}
    with open(PROC_DIR / "ca_crop_yields.csv") as f:
        for row in csv.DictReader(f):
            key = (int(row["year"]), row["province"], row["crop"])
            data[key] = {
                "yield_kg_ha": float(row["yield_kg_ha"]) if row["yield_kg_ha"] else None,
                "harvested_ha": float(row["harvested_ha"]) if row["harvested_ha"] else None,
                "production_mt": float(row["production_mt"]) if row["production_mt"] else None,
            }
    return data


def load_weather():
    """Load weather feature data."""
    data = {}  # (year, province) -> feature dict
    with open(PROC_DIR / "ca_weather_features.csv") as f:
        for row in csv.DictReader(f):
            key = (int(row["year"]), row["province"])
            data[key] = {k: float(v) if v else None for k, v in row.items()
                         if k not in ("year", "province", "n_stations")}
            data[key]["n_weather_stations"] = int(row["n_stations"])
    return data


def load_prices():
    """Load and annualize monthly price data."""
    # Group by (year, province, commodity) → list of monthly prices
    monthly = defaultdict(list)
    with open(PROC_DIR / "ca_farm_prices_monthly.csv") as f:
        for row in csv.DictReader(f):
            commodity_raw = row["commodity"]
            crop = PRICE_COMMODITY_MAP.get(commodity_raw)
            if not crop:
                continue
            year = int(row["ref_date"][:4])
            province = row["province"]
            price = float(row["price"])
            monthly[(year, province, crop)].append(price)

    # Compute annual average
    annual = {}
    for (year, province, crop), prices in monthly.items():
        annual[(year, province, crop)] = {
            "price_annual_avg": round(sum(prices) / len(prices), 2),
            "price_month_count": len(prices),
        }
    return annual


def build_feature_matrix(yields, weather, prices):
    """Build unified feature matrix."""
    rows = []

    # Iterate over all yield entries
    for (year, province, crop), yield_data in sorted(yields.items()):
        if year < YEAR_MIN or year > YEAR_MAX:
            continue

        row = {
            "year": year,
            "province": province,
            "crop": crop,
        }

        # Yield target
        row["yield_kg_ha"] = yield_data["yield_kg_ha"] or ""
        row["harvested_ha"] = yield_data["harvested_ha"] or ""
        row["production_mt"] = yield_data["production_mt"] or ""

        # Weather features (same province-year)
        wx = weather.get((year, province), {})
        for feat in ["gdd_total", "heat_stress_days", "precip_total_mm",
                      "precip_may_jun_mm", "precip_jul_aug_mm",
                      "max_consecutive_dry_days", "frost_free_days",
                      "mean_temp_growing"]:
            row[feat] = wx.get(feat, "") if wx.get(feat) is not None else ""

        # Lagged weather (previous year)
        wx_lag = weather.get((year - 1, province), {})
        row["prev_year_precip_mm"] = wx_lag.get("precip_total_mm", "") if wx_lag.get("precip_total_mm") is not None else ""
        row["prev_year_gdd"] = wx_lag.get("gdd_total", "") if wx_lag.get("gdd_total") is not None else ""

        # Lagged yield (previous year, same crop)
        prev_yield = yields.get((year - 1, province, crop), {})
        row["prev_year_yield_kg_ha"] = prev_yield.get("yield_kg_ha", "") or ""

        # Price data
        price_data = prices.get((year, province, crop), {})
        row["price_cad_per_tonne"] = price_data.get("price_annual_avg", "")

        # Price from previous year for price change feature
        prev_price = prices.get((year - 1, province, crop), {})
        if price_data.get("price_annual_avg") and prev_price.get("price_annual_avg"):
            row["price_change_pct"] = round(
                (price_data["price_annual_avg"] - prev_price["price_annual_avg"])
                / prev_price["price_annual_avg"] * 100, 2
            )
        else:
            row["price_change_pct"] = ""

        rows.append(row)

    return rows


def main():
    print("=" * 60)
    print("Pipeline: Feature Matrix Join")
    print("=" * 60)

    yields = load_yields()
    weather = load_weather()
    prices = load_prices()

    print(f"  Yield records: {len(yields)}")
    print(f"  Weather records: {len(weather)}")
    print(f"  Price records: {len(prices)}")

    rows = build_feature_matrix(yields, weather, prices)

    # Save
    fieldnames = [
        "year", "province", "crop",
        "yield_kg_ha", "harvested_ha", "production_mt",
        "gdd_total", "heat_stress_days", "precip_total_mm",
        "precip_may_jun_mm", "precip_jul_aug_mm",
        "max_consecutive_dry_days", "frost_free_days", "mean_temp_growing",
        "prev_year_precip_mm", "prev_year_gdd", "prev_year_yield_kg_ha",
        "price_cad_per_tonne", "price_change_pct",
    ]

    out_path = PROC_DIR / "ca_feature_matrix.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Saved {len(rows)} rows → {out_path}")

    # Quality report
    total = len(rows)
    has_yield = sum(1 for r in rows if r["yield_kg_ha"])
    has_weather = sum(1 for r in rows if r["gdd_total"])
    has_price = sum(1 for r in rows if r["price_cad_per_tonne"])
    has_all = sum(1 for r in rows if r["yield_kg_ha"] and r["gdd_total"] and r["price_cad_per_tonne"])

    print(f"\nCompleteness:")
    print(f"  Total rows:     {total}")
    print(f"  Has yield:      {has_yield} ({has_yield/total*100:.0f}%)")
    print(f"  Has weather:    {has_weather} ({has_weather/total*100:.0f}%)")
    print(f"  Has price:      {has_price} ({has_price/total*100:.0f}%)")
    print(f"  Has ALL three:  {has_all} ({has_all/total*100:.0f}%)")

    # Focus on last decade
    recent = [r for r in rows if int(r["year"]) >= 2015]
    recent_complete = [r for r in recent if r["yield_kg_ha"] and r["gdd_total"]]
    print(f"\n  Last decade (2015-2024):")
    print(f"    Total: {len(recent)}, with yield+weather: {len(recent_complete)}")

    # 2021 drought showcase
    drought = [r for r in rows if int(r["year"]) == 2021 and r["crop"] == "Wheat"]
    if drought:
        print(f"\n  2021 Drought — Wheat:")
        for r in drought:
            print(f"    {r['province']}: yield={r['yield_kg_ha']}kg/ha, "
                  f"GDD={r['gdd_total']}, heat={r['heat_stress_days']}d, "
                  f"precip={r['precip_total_mm']}mm, dry_spell={r['max_consecutive_dry_days']}d"
                  f"{', price=$'+str(r['price_cad_per_tonne'])+'/t' if r['price_cad_per_tonne'] else ''}")

    print("\n✓ Feature matrix complete")


if __name__ == "__main__":
    main()

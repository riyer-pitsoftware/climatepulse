#!/usr/bin/env python3
"""
export_visualizations.py
Generate high-quality data visualizations for the ClimatePulse demo video.
Reads processed data from data/processed/ and writes PNGs to data/processed/charts/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
CHARTS = DATA / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

FEATURE_MATRIX = DATA / "ca_feature_matrix.csv"
CROP_YIELDS = DATA / "ca_crop_yields.csv"
WEATHER_FEATURES = DATA / "ca_weather_features.csv"
MODEL_RESULTS = DATA / "ca_model_results.json"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
DPI = 180

# Consistent colour palette
PROVINCE_COLORS = {
    "Alberta": "#2563EB",       # blue
    "Manitoba": "#16A34A",      # green
    "Saskatchewan": "#DC2626",  # red
}
CROP_COLORS = {
    "Barley": "#6366F1",   # indigo
    "Canola": "#F59E0B",   # amber
    "Oats": "#10B981",     # emerald
    "Wheat": "#EF4444",    # red
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
fm = pd.read_csv(FEATURE_MATRIX)
cy = pd.read_csv(CROP_YIELDS)
wf = pd.read_csv(WEATHER_FEATURES)
with open(MODEL_RESULTS) as f:
    mr = json.load(f)

print(f"Loaded feature_matrix={len(fm)}, crop_yields={len(cy)}, "
      f"weather_features={len(wf)} rows")

# ===================================================================
# 1. pipeline_dag.png
# ===================================================================
def draw_pipeline_dag():
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    box_kw = dict(boxstyle="round,pad=0.4", linewidth=1.5)

    def draw_box(x, y, w, h, label, color, fontsize=11):
        """Draw a rounded box. Returns dict with edge midpoints for arrows."""
        box = FancyBboxPatch((x, y), w, h, **box_kw,
                             facecolor=color, edgecolor="#334155", alpha=0.92)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white")
        return {
            "left":  (x,         y + h / 2),
            "right": (x + w,     y + h / 2),
        }

    def draw_arrow(start, end, color="#475569"):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    # Source boxes (left column)
    b_weather = draw_box(0.4, 4.5, 2.4, 0.8, "ECCC Weather\nStations", "#2563EB")
    b_yields  = draw_box(0.4, 2.9, 2.4, 0.8, "StatsCan\nCrop Yields", "#16A34A")
    b_prices  = draw_box(0.4, 1.3, 2.4, 0.8, "StatsCan\nCrop Prices", "#DC2626")

    # Join box (center-left)
    b_join  = draw_box(4.2, 2.9, 2.4, 1.0, "Feature\nMatrix Join", "#7C3AED")

    # Model box (center-right)
    b_model = draw_box(8.0, 2.9, 2.4, 1.0, "XGBoost\nModel", "#0891B2")

    # Output boxes (right column, well separated)
    b_api  = draw_box(11.8, 4.5, 1.8, 0.7, "REST API", "#0D9488", fontsize=10)
    b_dash = draw_box(11.8, 3.3, 1.8, 0.7, "Dashboard", "#0D9488", fontsize=10)
    b_csv  = draw_box(11.8, 2.1, 1.8, 0.7, "CSV Export", "#0D9488", fontsize=10)

    # Arrows: sources -> join (right edge of source -> left edge of join)
    for src in [b_weather, b_yields, b_prices]:
        draw_arrow(src["right"], b_join["left"])

    # Arrow: join -> model
    draw_arrow(b_join["right"], b_model["left"])

    # Arrows: model -> outputs
    for out in [b_api, b_dash, b_csv]:
        draw_arrow(b_model["right"], out["left"])

    ax.set_title("ClimatePulse Pipeline DAG", fontsize=16, fontweight="bold",
                 pad=10, color="#1E293B")

    fig.tight_layout()
    fig.savefig(CHARTS / "pipeline_dag.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [1/6] pipeline_dag.png")


# ===================================================================
# 2. drought_2021_yields.png
# ===================================================================
def draw_drought_2021():
    # Compute 25-year baseline (2000-2024 excl 2021) for each province x crop
    baseline = (cy[cy["year"] != 2021]
                .groupby(["province", "crop"])["yield_kg_ha"]
                .mean()
                .reset_index()
                .rename(columns={"yield_kg_ha": "baseline"}))

    actual = cy[cy["year"] == 2021][["province", "crop", "yield_kg_ha"]].copy()
    merged = actual.merge(baseline, on=["province", "crop"])
    merged["label"] = merged["province"].str[:2] + " " + merged["crop"]
    merged["pct_drop"] = (merged["yield_kg_ha"] - merged["baseline"]) / merged["baseline"] * 100
    merged = merged.sort_values("pct_drop")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    x = np.arange(len(merged))
    w = 0.35
    bars_base = ax.bar(x - w / 2, merged["baseline"], w, label="25-yr Baseline",
                       color="#94A3B8", edgecolor="white", zorder=3)
    bars_2021 = ax.bar(x + w / 2, merged["yield_kg_ha"], w, label="2021 Actual",
                       color="#EF4444", edgecolor="white", zorder=3)

    # Highlight SK Wheat (worst drop)
    worst_idx = merged["pct_drop"].idxmin()
    worst_pos = merged.index.get_loc(worst_idx) if hasattr(merged.index, "get_loc") else list(merged.index).index(worst_idx)
    # We need the positional index in sorted order
    worst_pos_sorted = list(merged.index).index(worst_idx)
    bars_2021[worst_pos_sorted].set_edgecolor("#000000")
    bars_2021[worst_pos_sorted].set_linewidth(2.5)

    # Annotate worst
    worst_row = merged.iloc[worst_pos_sorted]
    ax.annotate(f"{worst_row['pct_drop']:.0f}%",
                xy=(worst_pos_sorted + w / 2, worst_row["yield_kg_ha"]),
                xytext=(worst_pos_sorted + w / 2 + 0.6, worst_row["yield_kg_ha"] + 350),
                fontsize=11, fontweight="bold", color="#991B1B",
                arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Yield (kg/ha)", fontsize=12)
    ax.set_title("2021 Drought: Actual Yields vs 25-Year Baseline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, merged["baseline"].max() * 1.25)

    fig.tight_layout()
    fig.savefig(CHARTS / "drought_2021_yields.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [2/6] drought_2021_yields.png")


# ===================================================================
# 3. yield_timeseries.png
# ===================================================================
def draw_yield_timeseries():
    wheat = cy[cy["crop"] == "Wheat"].copy()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    for prov, color in PROVINCE_COLORS.items():
        sub = wheat[wheat["province"] == prov].sort_values("year")
        ax.plot(sub["year"], sub["yield_kg_ha"], marker="o", markersize=5,
                linewidth=2, color=color, label=prov, zorder=4)

    # Highlight 2021
    ax.axvspan(2020.5, 2021.5, color="#FCA5A5", alpha=0.35, zorder=1,
               label="2021 Drought")
    ax.axvline(2021, color="#DC2626", linestyle="--", linewidth=1, alpha=0.6, zorder=2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Wheat Yield (kg/ha)", fontsize=12)
    ax.set_title("Wheat Yields Across Prairie Provinces (2000-2024)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlim(1999.5, 2024.5)

    fig.tight_layout()
    fig.savefig(CHARTS / "yield_timeseries.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [3/6] yield_timeseries.png")


# ===================================================================
# 4. weather_yield_scatter.png
# ===================================================================
def draw_weather_yield_scatter():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    for prov, color in PROVINCE_COLORS.items():
        sub = fm[fm["province"] == prov]
        ax.scatter(sub["heat_stress_days"], sub["yield_kg_ha"],
                   c=color, label=prov, alpha=0.6, edgecolors="white",
                   linewidths=0.5, s=50, zorder=3)

    # Overall trend line
    mask = fm["heat_stress_days"].notna() & fm["yield_kg_ha"].notna()
    x_vals = fm.loc[mask, "heat_stress_days"]
    y_vals = fm.loc[mask, "yield_kg_ha"]
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, p(x_line), "--", color="#475569", linewidth=2, alpha=0.8,
            label=f"Trend (slope={z[0]:.1f})", zorder=5)

    ax.set_xlabel("Heat Stress Days (Tmax > 30 C)", fontsize=12)
    ax.set_ylabel("Yield (kg/ha)", fontsize=12)
    ax.set_title("Heat Stress vs Crop Yield", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(CHARTS / "weather_yield_scatter.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [4/6] weather_yield_scatter.png")


# ===================================================================
# 5. price_impact.png
# ===================================================================
def draw_price_impact():
    # Use the feature matrix rows that have both yield anomaly and price change
    # We need to compute yield_anomaly_pct ourselves from the raw data
    # Baseline per province x crop
    baseline = (fm.groupby(["province", "crop"])["yield_kg_ha"]
                .mean()
                .reset_index()
                .rename(columns={"yield_kg_ha": "baseline"}))

    df = fm[fm["price_change_pct"].notna()].copy()
    df = df.merge(baseline, on=["province", "crop"])
    df["yield_anomaly_pct"] = (df["yield_kg_ha"] - df["baseline"]) / df["baseline"] * 100

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    for crop, color in CROP_COLORS.items():
        sub = df[df["crop"] == crop]
        ax.scatter(sub["yield_anomaly_pct"], sub["price_change_pct"],
                   c=color, label=crop, alpha=0.6, edgecolors="white",
                   linewidths=0.5, s=50, zorder=3)

    # OLS fit line from model results
    ols = mr["price_impact"]["ols"]
    x_range = np.linspace(df["yield_anomaly_pct"].min(), df["yield_anomaly_pct"].max(), 100)
    y_fit = ols["slope"] * x_range + ols["intercept"]
    ax.plot(x_range, y_fit, "--", color="#1E293B", linewidth=2, alpha=0.8,
            label=f"OLS (R2={ols['r2']:.2f}, p<0.001)", zorder=5)

    ax.axhline(0, color="#94A3B8", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#94A3B8", linewidth=0.8, zorder=1)

    ax.set_xlabel("Yield Anomaly (%)", fontsize=12)
    ax.set_ylabel("Year-over-Year Price Change (%)", fontsize=12)
    ax.set_title("Yield Anomaly vs Commodity Price Change", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(CHARTS / "price_impact.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [5/6] price_impact.png")


# ===================================================================
# 6. holdout_2021_comparison.png
# ===================================================================
def draw_holdout_2021():
    preds = pd.DataFrame(mr["holdout_2021"]["predictions"])
    preds["label"] = preds["province"].str[:2] + "\n" + preds["crop"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    x = np.arange(len(preds))
    w = 0.35

    bars_actual = ax.bar(x - w / 2, preds["actual"], w, label="Actual (2021)",
                         color="#EF4444", edgecolor="white", zorder=3)
    bars_pred   = ax.bar(x + w / 2, preds["predicted"], w, label="Predicted (XGBoost)",
                         color="#3B82F6", edgecolor="white", zorder=3)

    # Annotate error_pct on each pair
    for i, row in preds.iterrows():
        idx = list(preds.index).index(i)
        ax.text(idx + w / 2, row["predicted"] + 50,
                f"+{row['error_pct']:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#1E40AF",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(preds["label"], fontsize=9)
    ax.set_ylabel("Yield (kg/ha)", fontsize=12)
    ax.set_title("2021 Holdout: Actual vs Predicted Yields\n"
                 f"(Overall R2 = {mr['holdout_2021']['overall']['r2']:.2f}  --  "
                 f"model trained on normal years cannot predict drought)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, preds["predicted"].max() * 1.2)

    fig.tight_layout()
    fig.savefig(CHARTS / "holdout_2021_comparison.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  [6/6] holdout_2021_comparison.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating ClimatePulse demo charts...")
    draw_pipeline_dag()
    draw_drought_2021()
    draw_yield_timeseries()
    draw_weather_yield_scatter()
    draw_price_impact()
    draw_holdout_2021()

    generated = list(CHARTS.glob("*.png"))
    print(f"\nDone. {len(generated)} charts saved to {CHARTS}/")
    for p in sorted(generated):
        print(f"  {p.name}")

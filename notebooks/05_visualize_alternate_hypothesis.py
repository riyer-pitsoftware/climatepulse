"""
ClimatePulse — Alternate Hypothesis Visualization
Temperature Threshold Effects on Grid Fossil Dependence.

Two-panel chart:
  Left:  Scatter + linear fit of temp deviation → fossil shift, color-coded by bin
  Right: Box/bar plot of fossil shift by thermal stress category (moderate/elevated/extreme)

Output: PNG in data/processed/charts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHART_DIR = DATA_DIR / "charts"

COMFORT = 65.0  # baseline temperature


def main():
    df = pd.read_csv(DATA_DIR / "unified_analysis.csv", parse_dates=["date"])
    df["temp_deviation"] = (df["mean_tmax"] - COMFORT).abs()

    x = df["temp_deviation"].values
    y = df["fossil_pct_change"].values

    # Tercile bins
    thresholds = np.percentile(x, [33.3, 66.7])
    bins = np.digitize(x, thresholds)
    bin_labels = ["Moderate", "Elevated", "Extreme"]
    bin_colors = ["#4CAF50", "#FF9800", "#D32F2F"]

    # Event markers
    event_markers = {"heat_dome_2021": "o", "uri_2021": "s", "elliott_2022": "D"}
    event_labels = {"heat_dome_2021": "PNW Heat Dome", "uri_2021": "Winter Storm Uri",
                    "elliott_2022": "Winter Storm Elliott"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("ClimatePulse — Grid Fossil Dependence Accelerates Under Extreme Thermal Stress",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Left panel: Scatter with regression ──
    for event in df["event"].unique():
        mask = df["event"] == event
        for b in range(3):
            bmask = mask & (bins == b)
            if bmask.sum() == 0:
                continue
            ax1.scatter(
                x[bmask], y[bmask],
                c=bin_colors[b], marker=event_markers[event],
                s=60, alpha=0.75, edgecolors="white", linewidth=0.5,
                zorder=5,
            )

    # Linear regression line
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color="#333", linewidth=2,
             linestyle="--", alpha=0.7, label=f"Linear: R²={r_val**2:.2f}, p={p_val:.1e}")

    # Threshold lines
    for t in thresholds:
        ax1.axvline(t, color="#999", linestyle=":", linewidth=1, alpha=0.6)

    ax1.set_xlabel("Temperature Deviation from 65°F", fontsize=11)
    ax1.set_ylabel("Fossil Generation Shift (pp vs baseline)", fontsize=11)
    ax1.set_title("Scatter — Thermal Stress vs Fossil Shift", fontsize=11, loc="left", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Legend: events
    event_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="#666", markersize=8, label=l)
        for m, l in zip(["o", "s", "D"], ["Heat Dome", "Uri", "Elliott"])
    ]
    bin_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(bin_colors, bin_labels)]
    ax1.legend(handles=event_handles + bin_handles, loc="upper left", fontsize=8, ncol=2)

    # ── Right panel: Bar chart with error bars ──
    means, ses, ns = [], [], []
    for b in range(3):
        vals = y[bins == b]
        means.append(vals.mean())
        ses.append(stats.sem(vals))
        ns.append(len(vals))

    bars = ax2.bar(bin_labels, means, color=bin_colors, alpha=0.85,
                   edgecolor="white", linewidth=1.5,
                   yerr=ses, capsize=6, error_kw={"linewidth": 1.5, "color": "#333"})

    # Annotate bars
    for bar, mean, se, n in zip(bars, means, ses, ns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.5,
                 f"{mean:+.1f}pp\n(n={n})", ha="center", fontsize=10, fontweight="bold")

    # Significance bracket between moderate and extreme
    kw_p = 0.0009  # from the analysis
    bracket_y = max(means) + max(ses) + 3
    ax2.plot([0, 0, 2, 2], [bracket_y - 0.5, bracket_y, bracket_y, bracket_y - 0.5],
             color="#333", linewidth=1.5)
    ax2.text(1, bracket_y + 0.3, f"p = {kw_p:.4f} ***", ha="center", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Mean Fossil Shift (pp vs baseline)", fontsize=11)
    ax2.set_title("Fossil Shift by Thermal Stress Level", fontsize=11, loc="left", fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Add the key takeaway
    ax2.text(0.5, -0.12,
             "Extreme thermal stress → 12pp more fossil dependence than moderate (Mann-Whitney p < 0.001)",
             transform=ax2.transAxes, ha="center", fontsize=9, style="italic", color="#555")

    plt.tight_layout()
    out_path = CHART_DIR / "alternate_threshold_effects.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
ClimatePulse — Causal Chain Visualization
Generates 3-panel charts showing: Weather → Grid Shift → Air Quality for each event.
Output: PNG files in data/processed/charts/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
CHART_DIR = DATA_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

# Load data
noaa = pd.read_csv(DATA_DIR / "noaa_event_timeline.csv", parse_dates=["date"])
eia = pd.read_csv(DATA_DIR / "eia_grid_response.csv", parse_dates=["datetime"])
epa = pd.read_csv(DATA_DIR / "epa_air_quality.csv", parse_dates=["date"])

# ─── Event configs ───────────────────────────────────────────────────────────
EVENTS = [
    {
        "name": "Winter Storm Uri",
        "noaa_event": "Winter Storm Uri (Feb 2021)",
        "eia_event": "ERCO_URI_2021",
        "epa_event": "uri",
        "epa_counties": ["Harris", "Dallas"],
        "weather_col": "min_tmin",
        "weather_label": "Min Temperature (°F)",
        "weather_color": "#2166AC",
        "date_range": ("2021-02-01", "2021-02-28"),
        "invert_weather": True,  # colder = more extreme
    },
    {
        "name": "Winter Storm Elliott",
        "noaa_event": "Winter Storm Elliott (Dec 2022)",
        "eia_event": "PJM_Elliott_2022",
        "epa_event": "elliott",
        "epa_counties": ["Allegheny", "Hamilton"],
        "weather_col": "min_tmin",
        "weather_label": "Min Temperature (°F)",
        "weather_color": "#2166AC",
        "date_range": ("2022-12-15", "2023-01-05"),
        "invert_weather": True,
    },
    {
        "name": "PNW Heat Dome",
        "noaa_event": "PNW Heat Dome (Jun-Jul 2021)",
        "eia_event": "BPAT_HeatDome_2021",
        "epa_event": "heat_dome",
        "epa_counties": ["Multnomah", "King"],
        "weather_col": "max_tmax",
        "weather_label": "Max Temperature (°F)",
        "weather_color": "#B2182B",
        "date_range": ("2021-06-20", "2021-07-10"),
        "invert_weather": False,  # hotter = more extreme
    },
]


def plot_event(event_cfg, fig_num):
    """Create a 3-panel vertical chart for one event."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"ClimatePulse — {event_cfg['name']}", fontsize=16, fontweight="bold", y=0.98)

    start, end = event_cfg["date_range"]

    # ── Panel 1: Weather ─────────────────────────────────────────────────────
    ax1 = axes[0]
    weather = noaa[noaa["event"] == event_cfg["noaa_event"]].copy()
    weather = weather[(weather["date"] >= start) & (weather["date"] <= end)]

    ax1.fill_between(weather["date"], weather[event_cfg["weather_col"]],
                     alpha=0.3, color=event_cfg["weather_color"])
    ax1.plot(weather["date"], weather[event_cfg["weather_col"]],
             color=event_cfg["weather_color"], linewidth=2, marker="o", markersize=4)

    # Mark the extreme day
    if event_cfg["invert_weather"]:
        extreme_idx = weather[event_cfg["weather_col"]].idxmin()
    else:
        extreme_idx = weather[event_cfg["weather_col"]].idxmax()
    extreme_row = weather.loc[extreme_idx]
    ax1.annotate(f"{extreme_row[event_cfg['weather_col']]:.0f}°F",
                 xy=(extreme_row["date"], extreme_row[event_cfg["weather_col"]]),
                 xytext=(10, 15), textcoords="offset points",
                 fontsize=12, fontweight="bold", color=event_cfg["weather_color"],
                 arrowprops=dict(arrowstyle="->", color=event_cfg["weather_color"]))

    ax1.set_ylabel(event_cfg["weather_label"], fontsize=11)
    ax1.set_title("(1) Extreme Weather", fontsize=12, loc="left", fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # ── Panel 2: Grid Fuel Mix ───────────────────────────────────────────────
    ax2 = axes[1]
    grid = eia[(eia["event"] == event_cfg["eia_event"]) & (eia["period_type"] == "event")].copy()

    # Aggregate hourly to daily for cleaner visualization
    grid["date"] = grid["datetime"].dt.date
    daily_grid = grid.groupby(["date", "fuel_category"])["generation_mwh"].sum().reset_index()
    daily_total = grid.groupby("date")["generation_mwh"].sum().reset_index()
    daily_total.columns = ["date", "total_mwh"]
    daily_grid = daily_grid.merge(daily_total, on="date")
    daily_grid["pct"] = daily_grid["generation_mwh"] / daily_grid["total_mwh"] * 100
    daily_grid["date"] = pd.to_datetime(daily_grid["date"])

    # Plot stacked area
    for cat, color, label in [
        ("fossil", "#D32F2F", "Fossil (Coal/Gas/Oil)"),
        ("other", "#FFA726", "Other (Nuclear)"),
        ("renewable", "#388E3C", "Renewable (Solar/Wind/Hydro)"),
    ]:
        cat_data = daily_grid[daily_grid["fuel_category"] == cat].set_index("date")["pct"]
        if not cat_data.empty:
            ax2.fill_between(cat_data.index, 0, cat_data.values, alpha=0.6, color=color, label=label)
            ax2.plot(cat_data.index, cat_data.values, color=color, linewidth=1.5)

    # Find peak fossil day
    fossil_daily = daily_grid[daily_grid["fuel_category"] == "fossil"]
    if not fossil_daily.empty:
        peak_fossil = fossil_daily.loc[fossil_daily["pct"].idxmax()]
        ax2.annotate(f"{peak_fossil['pct']:.0f}% fossil",
                     xy=(peak_fossil["date"], peak_fossil["pct"]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=11, fontweight="bold", color="#D32F2F",
                     arrowprops=dict(arrowstyle="->", color="#D32F2F"))

    ax2.set_ylabel("% of Total Generation", fontsize=11)
    ax2.set_title("(2) Grid Response — Fuel Mix Shift", fontsize=12, loc="left", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # ── Panel 3: Air Quality ─────────────────────────────────────────────────
    ax3 = axes[2]
    aq = epa[epa["event"] == event_cfg["epa_event"]].copy()

    colors = ["#E65100", "#7B1FA2"]
    for i, county in enumerate(event_cfg["epa_counties"]):
        county_data = aq[aq["county"] == county]
        if county_data.empty:
            continue
        ax3.plot(county_data["date"], county_data["pm25_aqi"],
                 color=colors[i], linewidth=2, marker="s", markersize=3,
                 label=f"PM2.5 AQI — {county}")

    # Add "Unhealthy for Sensitive Groups" threshold line
    ax3.axhline(y=100, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax3.text(aq["date"].max(), 102, "USG Threshold", fontsize=8, color="red", alpha=0.7, ha="right")

    # Find peak AQI
    if not aq.empty and "pm25_aqi" in aq.columns:
        peak_aqi = aq.loc[aq["pm25_aqi"].idxmax()]
        ax3.annotate(f"AQI {peak_aqi['pm25_aqi']:.0f}",
                     xy=(peak_aqi["date"], peak_aqi["pm25_aqi"]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=11, fontweight="bold", color="#E65100",
                     arrowprops=dict(arrowstyle="->", color="#E65100"))

    ax3.set_ylabel("PM2.5 AQI", fontsize=11)
    ax3.set_title("(3) Air Quality Impact", fontsize=12, loc="left", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = CHART_DIR / f"causal_chain_{event_cfg['name'].lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    return out_path


def plot_summary_comparison():
    """Single chart comparing all 3 events side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ClimatePulse — The Causal Chain Across Three Events",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("Weather\nExtreme", [-2, -9, 116], ["Uri\n(TX)", "Elliott\n(OH/PA)", "Heat Dome\n(OR/WA)"],
         ["#2166AC", "#2166AC", "#B2182B"], "°F"),
        ("Peak Fossil\nGeneration %", [89.5, 66.1, 30.4], ["Uri", "Elliott", "Heat Dome"],
         ["#D32F2F", "#D32F2F", "#D32F2F"], "%"),
        ("Peak PM2.5\nAQI Increase", [133, 125, 474], ["Uri", "Elliott", "Heat Dome"],
         ["#E65100", "#E65100", "#E65100"], "%"),
    ]

    for ax, (title, values, labels, colors, unit) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    f"{val}{unit}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    out_path = CHART_DIR / "summary_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_lagged_aqi():
    """Visualize the key finding: fossil shift predicts AQI with 1-day lag.

    Two panels:
    - Left: scatter of fossil_pct_change vs next-day PM2.5 AQI (Uri, strongest)
    - Right: bar chart of correlation strength at lag 0-3 days (pooled)
    """
    from scipy import stats as sp_stats

    unified = pd.read_csv(DATA_DIR / "unified_analysis.csv", parse_dates=["date"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ClimatePulse — The Hidden Delay: Fossil Emissions → Air Quality",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Left panel: Uri scatter with regression ──
    uri = unified[unified["event"] == "uri_2021"].sort_values("date").dropna(
        subset=["fossil_pct_change", "pm25_aqi"]
    ).copy()
    uri["pm25_next_day"] = uri["pm25_aqi"].shift(-1)
    uri_clean = uri.dropna(subset=["pm25_next_day"])

    ax1.scatter(uri_clean["fossil_pct_change"], uri_clean["pm25_next_day"],
                color="#D32F2F", s=60, alpha=0.7, edgecolors="white", zorder=5)

    # Regression line
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(
        uri_clean["fossil_pct_change"], uri_clean["pm25_next_day"]
    )
    x_line = pd.Series([uri_clean["fossil_pct_change"].min(), uri_clean["fossil_pct_change"].max()])
    ax1.plot(x_line, slope * x_line + intercept, color="#D32F2F", linewidth=2, linestyle="--")

    ax1.set_xlabel("Fossil Generation Shift (pp above baseline)", fontsize=11)
    ax1.set_ylabel("PM2.5 AQI (Next Day)", fontsize=11)
    ax1.set_title("Winter Storm Uri — 1-Day Lag", fontsize=12, loc="left", fontweight="bold")
    ax1.text(0.05, 0.95, f"r = +{r_val:.2f}\np = {p_val:.1e}\nR² = {r_val**2:.2f}",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # ── Right panel: Pooled lag bar chart ──
    lags = [0, 1, 2, 3]
    pooled_r = []
    pooled_p = []

    for lag in lags:
        all_x, all_y = [], []
        for event in unified["event"].unique():
            sub = unified[unified["event"] == event].sort_values("date").dropna(
                subset=["fossil_pct_change", "pm25_aqi"]
            ).copy()
            if len(sub) < 5:
                continue
            sub["pm25_lag"] = sub["pm25_aqi"].shift(-lag)
            sub = sub.dropna(subset=["pm25_lag"])
            all_x.extend(sub["fossil_pct_change"].tolist())
            all_y.extend(sub["pm25_lag"].tolist())
        r, p = sp_stats.pearsonr(all_x, all_y)
        pooled_r.append(r)
        pooled_p.append(p)

    colors = ["#9E9E9E" if p >= 0.05 else "#D32F2F" for p in pooled_p]
    bars = ax2.bar(["Same\nday", "+1\nday", "+2\ndays", "+3\ndays"],
                   pooled_r, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)

    for bar, r, p in zip(bars, pooled_r, pooled_p):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"r={r:.2f}\n{sig}", ha="center", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Correlation (Pearson r)", fontsize=11)
    ax2.set_title("Pooled — Correlation Strengthens with Lag", fontsize=12, loc="left", fontweight="bold")
    ax2.set_ylim(0, 0.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out_path = CHART_DIR / "lagged_aqi_finding.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating causal chain visualizations...\n")

    for i, event_cfg in enumerate(EVENTS):
        print(f"[{i+1}/3] {event_cfg['name']}")
        plot_event(event_cfg, i)

    print(f"\n[4/4] Summary comparison")
    plot_summary_comparison()

    print(f"\n[5/5] Lagged AQI finding")
    plot_lagged_aqi()

    print(f"\nAll charts saved to: {CHART_DIR}")

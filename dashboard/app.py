"""ClimatePulse — Interactive Dashboard

Interactive causal-chain explorer: Extreme Weather → Grid Fossil Shift → AQI
Complements app_explore.py with geographic context, event filtering, and
a prediction interface placeholder.
"""

import json
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sp_stats
from streamlit_folium import st_folium

st.set_page_config(page_title="ClimatePulse Dashboard", layout="wide", page_icon="🌡️")

# ── Paths ──
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ── Colors (shared with app_explore.py) ──
FOSSIL_COLOR = "#d62728"
RENEWABLE_COLOR = "#2ca02c"
TEMP_HIGH_COLOR = "#ff7f0e"
TEMP_LOW_COLOR = "#1f77b4"
AQI_COLOR = "#9467bd"
BASELINE_COLOR = "#999999"

# ── Event metadata ──
EVENT_META = {
    "heat_dome_2021": {
        "label": "PNW Heat Dome",
        "period": "June–July 2021",
        "region": "Pacific Northwest (BPA)",
        "icon": "🔥",
        "lat": 45.52,
        "lon": -122.68,
        "grid": "BPAT",
        "color": "red",
        "weather_col": "max_tmax",
        "weather_label": "Peak High (°F)",
        "weather_dir": "up",
    },
    "uri_2021": {
        "label": "Winter Storm Uri",
        "period": "February 2021",
        "region": "Texas (ERCOT)",
        "icon": "🧊",
        "lat": 29.76,
        "lon": -95.37,
        "grid": "ERCO",
        "color": "blue",
        "weather_col": "min_tmin",
        "weather_label": "Record Low (°F)",
        "weather_dir": "down",
    },
    "elliott_2022": {
        "label": "Winter Storm Elliott",
        "period": "December 2022",
        "region": "Eastern US (PJM)",
        "icon": "❄️",
        "lat": 40.44,
        "lon": -79.99,
        "grid": "PJM",
        "color": "darkblue",
        "weather_col": "min_tmin",
        "weather_label": "Record Low (°F)",
        "weather_dir": "down",
    },
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "unified_analysis.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_stats():
    fp = DATA_DIR / "stats_results.json"
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


df_all = load_data()
stats_data = load_stats()

# ── Sidebar ──
st.sidebar.markdown("# ClimatePulse")
st.sidebar.markdown("Interactive causal-chain explorer")
st.sidebar.divider()

# Event selector
event_options = {meta["label"]: key for key, meta in EVENT_META.items()}
selected_label = st.sidebar.radio(
    "Select Event",
    list(event_options.keys()),
    index=0,
)
selected_event = event_options[selected_label]
meta = EVENT_META[selected_event]

# Baseline toggle
show_baseline = st.sidebar.checkbox(
    "Show baseline days", value=True,
    help="Include pre-event baseline rows if available",
)

st.sidebar.divider()
st.sidebar.markdown("**Data sources**")
st.sidebar.caption("NOAA GHCN-D · EIA Hourly Grid · EPA AQS")
st.sidebar.caption("2021–2023 · 3 events · ~70–135 daily obs")

# ── Filter data ──
if show_baseline:
    df = df_all[df_all["event"] == selected_event].copy()
else:
    df = df_all[(df_all["event"] == selected_event) & (df_all.get("is_baseline", 0) == 0)].copy()

event_rows = df[df.get("is_baseline", pd.Series(dtype=int)).eq(0) | ~df.columns.isin(["is_baseline"])]
if "is_baseline" in df.columns:
    event_only = df[df["is_baseline"] == 0]
    baseline_only = df[df["is_baseline"] == 1]
else:
    event_only = df
    baseline_only = pd.DataFrame()

# ── Header ──
st.markdown(f"# {meta['icon']} {meta['label']} — {meta['period']}")
st.markdown(f"*{meta['region']}*")
st.divider()

# ══════════════════════════════════════════════════════════════
# Section 1 — Geographic Context (Map)
# ══════════════════════════════════════════════════════════════
st.markdown("## Geographic Context")

map_col, info_col = st.columns([2, 1])

with map_col:
    m = folium.Map(location=[meta["lat"], meta["lon"]], zoom_start=5, tiles="CartoDB positron")

    for key, ev in EVENT_META.items():
        is_selected = key == selected_event
        folium.CircleMarker(
            location=[ev["lat"], ev["lon"]],
            radius=14 if is_selected else 8,
            color=ev["color"],
            fill=True,
            fill_color=ev["color"],
            fill_opacity=0.8 if is_selected else 0.3,
            popup=folium.Popup(
                f"<b>{ev['label']}</b><br>{ev['period']}<br>{ev['region']}",
                max_width=200,
            ),
            tooltip=ev["label"],
        ).add_to(m)

        if is_selected:
            folium.Marker(
                location=[ev["lat"], ev["lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:24px;text-align:center">{ev["icon"]}</div>',
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                ),
            ).add_to(m)

    st_folium(m, height=350, use_container_width=True, returned_objects=[])

with info_col:
    st.markdown(f"### {meta['label']}")
    st.markdown(f"**Grid region:** {meta['grid']}")
    st.markdown(f"**Period:** {meta['period']}")

    if meta["weather_dir"] == "up":
        extreme_val = event_only[meta["weather_col"]].max()
    else:
        extreme_val = event_only[meta["weather_col"]].min()
    st.metric(meta["weather_label"], f"{extreme_val:.0f}°F")

    avg_fossil = event_only["fossil_pct_change"].mean()
    st.metric("Avg Fossil Shift", f"{avg_fossil:+.1f}pp")

    epa = event_only.dropna(subset=["pm25_aqi"])
    if len(epa) > 0:
        st.metric("Avg PM2.5 AQI", f"{epa['pm25_aqi'].mean():.0f}")
    else:
        st.metric("Avg PM2.5 AQI", "—")

    st.metric("Days observed", f"{len(event_only)}" + (
        f" + {len(baseline_only)} baseline" if len(baseline_only) > 0 else ""
    ))

st.divider()

# ══════════════════════════════════════════════════════════════
# Section 2 — Causal Chain: Before / During / After
# ══════════════════════════════════════════════════════════════
st.markdown("## Causal Chain — Weather → Grid → Air Quality")

col1, col2, col3 = st.columns(3)

# Determine color assignment for baseline vs event
has_baseline = "is_baseline" in df.columns and len(baseline_only) > 0
if has_baseline:
    df["period_label"] = df["is_baseline"].map({0: "Event", 1: "Baseline"})

with col1:
    st.markdown("### 1. The Storm")
    temp_cols = ["mean_tmin", "mean_tmax"]
    chart_df = df.rename(columns={"mean_tmin": "Daily Low", "mean_tmax": "Daily High"})

    temp_chart = (
        alt.Chart(chart_df)
        .transform_fold(["Daily Low", "Daily High"], as_=["measure", "temp"])
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
            y=alt.Y("temp:Q", title="Temperature (°F)"),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(
                    domain=["Daily Low", "Daily High"],
                    range=[TEMP_LOW_COLOR, TEMP_HIGH_COLOR],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            strokeDash=alt.condition(
                alt.datum.is_baseline == 1,
                alt.value([4, 4]),
                alt.value([0]),
            ) if has_baseline else alt.value([0]),
        )
        .properties(height=280)
    )
    st.altair_chart(temp_chart, use_container_width=True)

with col2:
    st.markdown("### 2. Grid Response")
    grid_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
            y=alt.Y("fossil_pct_change:Q", title="Fossil shift (pp)"),
            color=alt.condition(
                alt.datum.fossil_pct_change > 0,
                alt.value(FOSSIL_COLOR),
                alt.value(RENEWABLE_COLOR),
            ),
            opacity=alt.condition(
                alt.datum.is_baseline == 1,
                alt.value(0.4),
                alt.value(0.9),
            ) if has_baseline else alt.value(0.9),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("fossil_pct_change:Q", title="Fossil shift (pp)", format="+.1f"),
                alt.Tooltip("fossil_pct:Q", title="Fossil %", format=".1f"),
                alt.Tooltip("baseline_fossil_pct:Q", title="Baseline %", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], color=BASELINE_COLOR)
        .encode(y="y:Q")
    )
    st.altair_chart(grid_chart + zero_rule, use_container_width=True)
    st.caption("Red = more fossil than baseline. Green = less.")

with col3:
    st.markdown("### 3. Air Quality")
    epa_df = df.dropna(subset=["pm25_aqi"])
    if len(epa_df) > 0:
        aqi_chart = (
            alt.Chart(epa_df)
            .mark_area(opacity=0.3, color=AQI_COLOR)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                y=alt.Y("pm25_aqi:Q", title="PM2.5 AQI"),
            )
            .properties(height=280)
        )
        aqi_line = (
            alt.Chart(epa_df)
            .mark_line(strokeWidth=2, color=AQI_COLOR)
            .encode(
                x="date:T",
                y="pm25_aqi:Q",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("pm25_aqi:Q", title="PM2.5 AQI", format=".0f"),
                    alt.Tooltip("ozone_aqi:Q", title="Ozone AQI", format=".0f"),
                ],
            )
        )
        st.altair_chart(aqi_chart + aqi_line, use_container_width=True)
    else:
        st.info("No EPA monitoring data for this date range.")

st.divider()

# ══════════════════════════════════════════════════════════════
# Section 3 — Key Findings
# ══════════════════════════════════════════════════════════════
st.markdown("## Key Findings")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("#### Weather → Fossil Shift")
    pooled = stats_data.get("pooled", {})
    st.success(
        f"**Pooled fossil shift: +{pooled.get('fossil_mean_shift', 4.0):.1f}pp** "
        f"(p={pooled.get('p_value', 0.002)}, d={pooled.get('cohens_d', 0.39)})\n\n"
        "All three events show increased fossil generation during extreme weather."
    )

with f2:
    st.markdown("#### Fossil → AQI (Lagged)")
    lagged = stats_data.get("lagged_correlation", {}).get("pooled", {})
    lag1 = lagged.get("lag_1", {})
    st.warning(
        f"**1-day lag: r={lag1.get('pearson_r', 0.14):+.2f}** "
        f"(p={lag1.get('p_value_bh', lag1.get('p_value_raw', 0.19))})\n\n"
        "Same-day and lagged correlations are weak at the pooled level. "
        "Uri shows the strongest case-study signal (r=+0.35, p=0.051)."
    )

with f3:
    st.markdown("#### Limitations")
    st.info(
        "**Exploratory analysis** of 3 case studies (~70 event-days). "
        "Spatial unit mismatch, wildfire confounding (Heat Dome), and small sample "
        "limit causal inference. Associations generate hypotheses, not conclusions."
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# Section 4 — Prediction Interface (Placeholder)
# ══════════════════════════════════════════════════════════════
st.markdown("## Prediction Explorer")
st.markdown(
    "Estimate next-day air quality impact based on thermal stress and grid conditions. "
    "*(Model training in progress — using linear approximation from observed data.)*"
)

pred_col1, pred_col2 = st.columns([1, 1])

with pred_col1:
    thermal_stress = st.slider(
        "Temperature deviation from 65°F",
        min_value=0, max_value=60, value=25, step=1,
        help="How far from comfortable 65°F (captures both heat and cold stress)",
    )
    grid_region = st.selectbox(
        "Grid region",
        ["ERCOT (Texas)", "PJM (Eastern US)", "BPA (Pacific NW)"],
    )

with pred_col2:
    # Simple linear prediction based on observed stats
    # From thermal stress analysis: moderate(~10°F)->+0pp, extreme(>24°F)->+10.7pp
    if thermal_stress < 12:
        predicted_fossil_shift = thermal_stress * 0.05
        stress_label = "Moderate"
        stress_color = "#4CAF50"
    elif thermal_stress < 25:
        predicted_fossil_shift = thermal_stress * 0.22
        stress_label = "Elevated"
        stress_color = "#FF9800"
    else:
        predicted_fossil_shift = thermal_stress * 0.40
        stress_label = "Extreme"
        stress_color = "#D32F2F"

    # Lag-1 regression slope from Uri: ~0.5 AQI per 1pp fossil
    predicted_aqi_delta = predicted_fossil_shift * 0.5

    st.markdown(f"**Thermal stress level:** :{stress_label.lower()}[{stress_label}]")
    st.metric("Predicted fossil shift", f"+{predicted_fossil_shift:.1f}pp")
    st.metric("Est. next-day PM2.5 AQI change", f"+{predicted_aqi_delta:.1f}")
    st.caption(
        "Based on observed thermal-stress thresholds and Uri lag-1 regression slope. "
        "Not a validated forecast model."
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# Section 5 — Raw Data
# ══════════════════════════════════════════════════════════════
with st.expander(f"Raw data — {meta['label']} ({len(df)} rows)"):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ──
st.divider()
st.caption(
    "ClimatePulse — ZerveHack 2026 · Climate & Energy Track · "
    "Data: NOAA GHCN-D, EIA Hourly Grid Monitor, EPA AQS"
)

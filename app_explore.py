"""ClimatePulse — Thesis Explorer

Exploratory visualization: Extreme Weather → Grid Fossil Shift → AQI Association
"""

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="ClimatePulse", layout="wide")

# --- Theme colors ---
FOSSIL_COLOR = "#d62728"     # red — danger, fossil
RENEWABLE_COLOR = "#2ca02c"  # green — renewable
TEMP_HIGH_COLOR = "#ff7f0e"  # orange — heat
TEMP_LOW_COLOR = "#1f77b4"   # blue — cold
AQI_COLOR = "#9467bd"        # purple — air quality
OZONE_COLOR = "#8c564b"      # brown — ozone
BASELINE_COLOR = "#999999"   # grey — baseline reference

EVENT_META = {
    "heat_dome_2021": {
        "label": "PNW Heat Dome",
        "period": "June–July 2021",
        "region": "Pacific Northwest (BPA)",
        "icon": "🔥",
        "description": "Record-shattering heat across Oregon and Washington. "
            "Portland hit 116°F. Power demand surged as AC units ran nonstop.",
    },
    "uri_2021": {
        "label": "Winter Storm Uri",
        "period": "February 2021",
        "region": "Texas (ERCOT)",
        "icon": "🧊",
        "description": "Unprecedented freeze collapsed Texas grid infrastructure. "
            "Wind turbines froze, natural gas pipelines failed, millions lost power.",
    },
    "elliott_2022": {
        "label": "Winter Storm Elliott",
        "period": "December 2022",
        "region": "Eastern US (PJM)",
        "icon": "❄️",
        "description": "Bomb cyclone brought extreme cold from the Great Lakes to the Southeast. "
            "Heating demand spiked during the holiday weekend.",
    },
}


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/unified_analysis.csv", parse_dates=["date"])
    return df


df = load_data()

# --- Header ---
st.markdown("# ClimatePulse")
st.markdown(
    "### When extreme weather hits, power grids burn more fossil fuel — "
    "and the air we breathe may get worse."
)
st.markdown(
    "We tracked three major US weather disasters and measured what happened to the power grid "
    "and local air quality during each one. The exploratory evidence suggests a pattern: "
    "**extreme weather** → **more fossil fuel burned** → **associated AQI changes**. "
    "The strength of evidence varies by event and by link in the chain — see findings below."
)
st.divider()


def render_event(event_key, data):
    """Render the three-act story for one event."""
    meta = EVENT_META[event_key]
    epa = data.dropna(subset=["pm25_aqi"])

    # Compute headline stats
    avg_fossil_shift = data["fossil_pct_change"].mean()
    avg_renewable_shift = data["renewable_pct_change"].mean()
    peak_fossil_shift = data["fossil_pct_change"].max()
    peak_fossil_date = data.loc[data["fossil_pct_change"].idxmax(), "date"]
    avg_pm25 = epa["pm25_aqi"].mean() if len(epa) > 0 else None
    peak_pm25 = epa["pm25_aqi"].max() if len(epa) > 0 else None

    # --- Event header ---
    st.markdown(f"## {meta['icon']} {meta['label']} — {meta['period']}")
    st.markdown(f"*{meta['region']}* · {meta['description']}")

    # --- Headline verdict in plain language ---
    # Fossil shift narrative
    if avg_fossil_shift > 0:
        fossil_narrative = (
            f"During this event, the grid burned **{avg_fossil_shift:.1f} percentage points more "
            f"fossil fuel** than it normally does — peaking at **{peak_fossil_shift:+.1f}pp** "
            f"on {peak_fossil_date.strftime('%b %d')}."
        )
    else:
        fossil_narrative = (
            f"Fossil fuel generation stayed roughly at baseline levels "
            f"(shift: {avg_fossil_shift:+.1f}pp)."
        )

    # Renewable shift narrative
    if avg_renewable_shift < -1:
        renewable_narrative = (
            f"Meanwhile, clean energy's share of the grid **fell {abs(avg_renewable_shift):.1f} "
            f"percentage points** — meaning fossil plants had to pick up the slack."
        )
    elif avg_renewable_shift > 1:
        renewable_narrative = (
            f"Renewable energy's share actually rose {avg_renewable_shift:.1f}pp, "
            f"but fossil still grew faster."
        )
    else:
        renewable_narrative = "Renewable generation held roughly steady."

    # AQI narrative
    if avg_pm25 is not None:
        if avg_pm25 > 50:
            aqi_quality = "unhealthy for sensitive groups"
        elif avg_pm25 > 35:
            aqi_quality = "moderate — noticeable for people with respiratory conditions"
        else:
            aqi_quality = "within normal range, though elevated on peak days"
        aqi_narrative = (
            f"Air quality in the region averaged a PM2.5 index of **{avg_pm25:.0f}** "
            f"({aqi_quality}), peaking at **{peak_pm25:.0f}**. "
            f"PM2.5 measures fine particulate matter — the kind produced by burning fossil fuels "
            f"that lodges deep in lungs."
        )
    else:
        aqi_narrative = ""

    st.markdown(fossil_narrative)
    st.markdown(renewable_narrative)
    if aqi_narrative:
        st.markdown(aqi_narrative)

    # --- Three-act cascade: side by side ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### The Storm")
        temp_data = data.rename(columns={
            "mean_tmin": "Daily Low",
            "mean_tmax": "Daily High",
        })
        temp_chart = (
            alt.Chart(temp_data)
            .transform_fold(["Daily Low", "Daily High"], as_=["measure", "temp"])
            .mark_area(opacity=0.3)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                y=alt.Y("temp:Q", title="Temperature (°F)"),
                y2=alt.value(0),
                color=alt.Color(
                    "measure:N",
                    scale=alt.Scale(
                        domain=["Daily Low", "Daily High"],
                        range=[TEMP_LOW_COLOR, TEMP_HIGH_COLOR],
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
            )
            .properties(height=250)
        )
        temp_lines = (
            alt.Chart(temp_data)
            .transform_fold(["Daily Low", "Daily High"], as_=["measure", "temp"])
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("temp:Q"),
                color=alt.Color(
                    "measure:N",
                    scale=alt.Scale(
                        domain=["Daily Low", "Daily High"],
                        range=[TEMP_LOW_COLOR, TEMP_HIGH_COLOR],
                    ),
                    legend=None,
                ),
            )
        )
        st.altair_chart(temp_chart + temp_lines, use_container_width=True)

    with col2:
        st.markdown("#### Grid's Fossil Surge")
        # Show fossil % change as bars — the delta IS the story
        grid_chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                y=alt.Y("fossil_pct_change:Q", title="More fossil than normal (pp)"),
                color=alt.condition(
                    alt.datum.fossil_pct_change > 0,
                    alt.value(FOSSIL_COLOR),
                    alt.value(RENEWABLE_COLOR),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("fossil_pct_change:Q", title="Fossil shift (pp)", format="+.1f"),
                    alt.Tooltip("fossil_pct:Q", title="Fossil %", format=".1f"),
                    alt.Tooltip("baseline_fossil_pct:Q", title="Baseline %", format=".1f"),
                ],
            )
            .properties(height=250)
        )
        # Baseline reference line at 0
        baseline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            strokeDash=[4, 4], color=BASELINE_COLOR
        ).encode(y="y:Q")
        st.altair_chart(grid_chart + baseline, use_container_width=True)
        st.caption(
            f"Red bars = grid burning more fossil than usual. "
            f"Green = less than usual."
        )

    with col3:
        st.markdown("#### What We Breathe")
        if len(epa) > 0:
            aqi_chart = (
                alt.Chart(epa)
                .mark_area(opacity=0.3, color=AQI_COLOR)
                .encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                    y=alt.Y("pm25_aqi:Q", title="AQI"),
                )
                .properties(height=250)
            )
            aqi_line = (
                alt.Chart(epa)
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
            ozone_line = (
                alt.Chart(epa)
                .mark_line(strokeWidth=1.5, strokeDash=[4, 4], color=OZONE_COLOR)
                .encode(x="date:T", y=alt.Y("ozone_aqi:Q"))
            )
            st.altair_chart(aqi_chart + aqi_line + ozone_line, use_container_width=True)
            st.caption(
                f"Purple = fine particulate matter (PM2.5). "
                f"Dashed = ozone. Higher = worse for lungs."
            )
        else:
            st.info("EPA monitoring data not available for this date range.")

    st.divider()

# --- Tabs ---
tab_causal, tab_threshold = st.tabs([
    "Event Analysis — Exploratory Evidence",
    "Alternate Hypothesis — Thermal Threshold Effects",
])

with tab_causal:
    # --- Render each event as a story ---
    for event_key in ["heat_dome_2021", "uri_2021", "elliott_2022"]:
        event_data = df[df["event"] == event_key].copy()
        render_event(event_key, event_data)

    # --- Cross-event comparison ---
    st.markdown("## Observed Patterns")
    st.markdown(
        "Across three weather disasters in three grid regions, exploratory evidence suggests "
        "extreme weather is associated with increased fossil fuel generation. However, the "
        "strength and consistency of the air quality link varies across events."
    )

    comparison_data = []
    for event_key, meta in EVENT_META.items():
        sub = df[df["event"] == event_key]
        epa = sub.dropna(subset=["pm25_aqi"])
        fossil_shift = sub["fossil_pct_change"].mean()
        renewable_shift = sub["renewable_pct_change"].mean()
        comparison_data.append({
            "Event": f"{meta['icon']} {meta['label']}",
            "Grid Region": meta["region"],
            "More fossil than normal": f"{fossil_shift:+.1f}%",
            "Renewable change": f"{renewable_shift:+.1f}%",
            "Worst fossil spike (single day)": f"{sub['fossil_pct_change'].max():+.1f}%",
            "Avg air quality (PM2.5)": f"{epa['pm25_aqi'].mean():.0f}" if len(epa) > 0 else "—",
        })

    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "> *Data sources: NOAA GHCN-D (weather), EIA Hourly Grid Monitor (generation), "
        "EPA AQS (air quality). Analysis covers 2021–2023.*"
    )

    # --- Statistical Evidence ---
    st.markdown("## Exploratory Evidence")

    st.markdown(
        "We ran correlation, mean-shift, lagged correlation, and Granger causality tests "
        "across all three events. The evidence strength varies by link in the proposed chain."
    )

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("#### Finding 1 — SUPPORTED")
        st.success(
            "**Extreme weather is associated with more fossil fuel generation**\n\n"
            "Pooled across all events: fossil generation was **+4.0 percentage points** "
            "above baseline (p=0.002). The PNW Heat Dome alone showed a "
            "+6.2pp shift (p<0.0001, Cohen's d = 3.31). This link is the strongest "
            "in the chain and is supported across all three events."
        )

    with col_f2:
        st.markdown("#### Finding 2 — TENTATIVE")
        st.warning(
            "**Fossil generation associated with next-day AQI (case-study finding)**\n\n"
            "The same-day correlation is weak (r=0.18, p=0.15), but with a **1-day lag** "
            "the pooled association strengthens: r=+0.38 (p=0.002). Winter Storm Uri shows "
            "the strongest case-study signal at r=+0.70 (p=0.0001). "
            "However, this association is not consistent across all events and does not "
            "establish causation."
        )

    with col_f3:
        st.markdown("#### Finding 3 — NOT ESTABLISHED")
        st.info(
            "**Cross-event causal chain not confirmed**\n\n"
            "While each event shows some association between grid changes and AQI, "
            "we cannot confirm a consistent causal chain across all three events. "
            "Confounding factors (wildfire smoke, meteorological dispersion patterns, "
            "spatial mismatches in data) limit causal inference. "
            "The ~24-hour lag pattern is suggestive but not definitive."
        )

    # --- The Hidden Delay: Lagged AQI Visualization ---
    st.markdown("## Lagged Association — Fossil Generation and Next-Day AQI")
    st.markdown(
        "Same-day correlations between fossil generation and air quality are weak. "
        "When we shift the air quality data forward by one day — testing whether today's "
        "fossil spike is associated with **tomorrow's** PM2.5 — a stronger association appears. "
        "This is suggestive of an emissions-to-AQI lag, but other explanations "
        "(e.g., weather persistence, confounding pollutant sources) cannot be ruled out."
    )

    import numpy as np
    from scipy import stats as sp_stats

    lag_col1, lag_col2 = st.columns(2)

    with lag_col1:
        st.markdown("#### Winter Storm Uri — 1-Day Lag")
        st.markdown(
            "Each point shows one day's fossil shift (x) against the **next day's** "
            "PM2.5 air quality index (y). Uri shows the strongest association of the three events."
        )
        uri = df[df["event"] == "uri_2021"].sort_values("date").dropna(
            subset=["fossil_pct_change", "pm25_aqi"]
        ).copy()
        uri["pm25_next_day"] = uri["pm25_aqi"].shift(-1)
        uri_clean = uri.dropna(subset=["pm25_next_day"])

        scatter_uri = (
            alt.Chart(uri_clean)
            .mark_circle(size=80, color=FOSSIL_COLOR, opacity=0.7,
                         stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X("fossil_pct_change:Q",
                         title="Fossil Shift (pp above baseline)"),
                y=alt.Y("pm25_next_day:Q", title="PM2.5 AQI (Next Day)"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("fossil_pct_change:Q", title="Fossil Shift", format="+.1f"),
                    alt.Tooltip("pm25_next_day:Q", title="Next-day PM2.5", format=".0f"),
                ],
            )
            .properties(height=300)
        )

        # Regression line
        slope_u, intercept_u, r_u, p_u, _ = sp_stats.linregress(
            uri_clean["fossil_pct_change"], uri_clean["pm25_next_day"]
        )
        x_rng = pd.DataFrame({
            "fossil_pct_change": np.linspace(
                uri_clean["fossil_pct_change"].min(),
                uri_clean["fossil_pct_change"].max(), 50
            )
        })
        x_rng["pm25_next_day"] = slope_u * x_rng["fossil_pct_change"] + intercept_u

        reg_uri = (
            alt.Chart(x_rng)
            .mark_line(strokeDash=[6, 4], color=FOSSIL_COLOR, strokeWidth=2, opacity=0.6)
            .encode(x="fossil_pct_change:Q", y="pm25_next_day:Q")
        )

        st.altair_chart(scatter_uri + reg_uri, use_container_width=True)
        st.caption(
            f"r = +{r_u:.2f}, p = {p_u:.1e}, R² = {r_u**2:.2f}. "
            f"Each 1pp fossil increase predicts +{slope_u:.1f} PM2.5 AQI the next day."
        )

    with lag_col2:
        st.markdown("#### Pooled — Association Strengthens with Lag")
        st.markdown(
            "The same-day correlation is weak. Adding a 1-day delay strengthens the "
            "association to statistical significance — consistent with, but not proof of, "
            "an emissions-to-AQI mechanism."
        )

        # Compute pooled lagged correlations
        lag_results = []
        for lag in range(4):
            all_x, all_y = [], []
            for event in df["event"].unique():
                sub = df[df["event"] == event].sort_values("date").dropna(
                    subset=["fossil_pct_change", "pm25_aqi"]
                ).copy()
                if len(sub) < 5:
                    continue
                sub["pm25_lag"] = sub["pm25_aqi"].shift(-lag)
                sub = sub.dropna(subset=["pm25_lag"])
                all_x.extend(sub["fossil_pct_change"].tolist())
                all_y.extend(sub["pm25_lag"].tolist())
            r, p = sp_stats.pearsonr(all_x, all_y)
            sig_label = "n.s." if p >= 0.05 else ("***" if p < 0.001 else ("**" if p < 0.01 else "*"))
            lag_results.append({
                "Lag": f"+{lag} day{'s' if lag != 1 else ''}",
                "Correlation": r,
                "p-value": p,
                "Significant": p < 0.05,
                "Label": f"r={r:.2f} {sig_label}",
            })

        lag_df = pd.DataFrame(lag_results)

        lag_bar = (
            alt.Chart(lag_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Lag:N", sort=[r["Lag"] for r in lag_results],
                         axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Correlation:Q", title="Pearson r",
                         scale=alt.Scale(domain=[0, 0.5])),
                color=alt.condition(
                    alt.datum.Significant,
                    alt.value(FOSSIL_COLOR),
                    alt.value("#9E9E9E"),
                ),
                tooltip=[
                    alt.Tooltip("Lag:N"),
                    alt.Tooltip("Correlation:Q", format=".3f"),
                    alt.Tooltip("p-value:Q", format=".4f"),
                    alt.Tooltip("Label:N", title="Significance"),
                ],
            )
            .properties(height=300)
        )

        lag_text = (
            alt.Chart(lag_df)
            .mark_text(dy=-12, fontSize=11, fontWeight="bold")
            .encode(
                x=alt.X("Lag:N", sort=[r["Lag"] for r in lag_results]),
                y="Correlation:Q",
                text="Label:N",
                color=alt.condition(
                    alt.datum.Significant,
                    alt.value(FOSSIL_COLOR),
                    alt.value("#666"),
                ),
            )
        )

        st.altair_chart(lag_bar + lag_text, use_container_width=True)
        st.caption(
            "Grey = not significant. Red = statistically significant (p < 0.05). "
            "The 1-day lag shows the strongest signal."
        )

    st.divider()

    # --- Cross-event summary comparison ---
    st.markdown("## Side-by-Side — Three Events Compared")
    st.markdown(
        "Different regions, different seasons, different grid configurations. "
        "The fossil generation shift is consistent; the air quality association varies by event."
    )

    summary_metrics = []
    for event_key, meta in EVENT_META.items():
        sub = df[df["event"] == event_key]
        epa_sub = sub.dropna(subset=["pm25_aqi"])

        if event_key == "heat_dome_2021":
            weather_extreme = sub["max_tmax"].max()
            weather_label = f"{weather_extreme:.0f}°F (record high)"
        else:
            weather_extreme = sub["min_tmin"].min()
            weather_label = f"{weather_extreme:.0f}°F (record low)"

        summary_metrics.append({
            "Event": meta["label"],
            "Weather Extreme": weather_label,
            "Avg Fossil Shift": sub["fossil_pct_change"].mean(),
            "Peak Fossil Shift": sub["fossil_pct_change"].max(),
            "Avg PM2.5 AQI": epa_sub["pm25_aqi"].mean() if len(epa_sub) > 0 else None,
        })

    summary_df = pd.DataFrame(summary_metrics)

    scol1, scol2, scol3 = st.columns(3)
    metric_configs = [
        ("Avg Fossil Shift", "pp above baseline", FOSSIL_COLOR, "+.1f"),
        ("Peak Fossil Shift", "pp (worst single day)", "#B71C1C", "+.1f"),
        ("Avg PM2.5 AQI", "air quality index", AQI_COLOR, ".0f"),
    ]

    for col, (metric, unit, color, fmt) in zip([scol1, scol2, scol3], metric_configs):
        with col:
            chart_data = summary_df.dropna(subset=[metric])
            bar = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color, opacity=0.8)
                .encode(
                    x=alt.X("Event:N", sort=[m["label"] for m in EVENT_META.values()],
                             axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y(f"{metric}:Q", title=f"{metric} ({unit})"),
                    tooltip=[
                        alt.Tooltip("Event:N"),
                        alt.Tooltip(f"{metric}:Q", format=fmt),
                    ],
                )
                .properties(height=250)
            )
            text = (
                alt.Chart(chart_data)
                .mark_text(dy=-10, fontSize=12, fontWeight="bold", color=color)
                .encode(
                    x="Event:N",
                    y=f"{metric}:Q",
                    text=alt.Text(f"{metric}:Q", format=fmt),
                )
            )
            st.altair_chart(bar + text, use_container_width=True)

    st.divider()

    with st.expander("Detailed statistics per event"):
        for event_key in ["heat_dome_2021", "uri_2021", "elliott_2022"]:
            meta = EVENT_META[event_key]
            sub = df[df["event"] == event_key]
            fossil = sub["fossil_pct_change"]
            renewable = sub["renewable_pct_change"]

            t_f, p_f = sp_stats.ttest_1samp(fossil.dropna(), 0)
            t_r, p_r = sp_stats.ttest_1samp(renewable.dropna(), 0)
            d_f = fossil.mean() / fossil.std() if fossil.std() > 0 else 0

            sig_f = "significant" if p_f < 0.05 else "not significant"
            st.markdown(f"**{meta['icon']} {meta['label']}** ({len(fossil)} days)")
            st.markdown(
                f"- Fossil shift: {fossil.mean():+.2f}pp (p={p_f:.4f}, {sig_f}, d={d_f:.2f})\n"
                f"- Renewable shift: {renewable.mean():+.2f}pp (p={p_r:.4f})"
            )

    # --- Raw data tucked away ---
    with st.expander("Raw data (71 rows)"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Limitations and Methodology Notes ---
    st.divider()
    st.markdown("## Limitations & Methodology Notes")

    st.warning(
        "**Spatial unit mismatch.** This analysis joins three datasets with incompatible "
        "geographic footprints: station-level NOAA weather observations, county-level EPA "
        "air quality monitors, and balancing-authority-wide EIA grid generation data. "
        "These are matched on event and date, but a county's AQI monitoring station may not "
        "represent the same geographic area as the grid balancing authority's generation fleet. "
        "This mismatch introduces ecological inference risk — aggregate patterns may not "
        "reflect local-level relationships."
    )

    st.warning(
        "**PNW Heat Dome wildfire confounding.** The elevated AQI observed during the "
        "June–July 2021 Pacific Northwest heat dome is likely confounded by concurrent "
        "wildfire smoke. The extreme heat triggered wildfires across Oregon and Washington, "
        "and our pipeline does not separate wildfire-sourced particulate matter from "
        "power-sector emissions. The AQI signal for this event should not be attributed "
        "solely to changes in grid fossil fuel generation."
    )

    st.info(
        "**Interpretation guidance.** This is an exploratory analysis of three case studies. "
        "The sample size (three events, 71 total days) is too small to establish general "
        "causal claims. Associations observed here generate hypotheses for further "
        "investigation with larger datasets and causal identification strategies."
    )

with tab_threshold:
    import numpy as np
    from scipy import stats as sp_stats

    st.markdown("## Temperature Threshold Effects on Fossil Dependence")
    st.markdown(
        "Does fossil fuel dependence increase **linearly** with temperature stress, "
        "or does the grid hit a **threshold** beyond which fossil reliance accelerates? "
        "If the latter, climate change will disproportionately worsen air quality as "
        "extreme events intensify."
    )
    st.divider()

    # Prepare data
    COMFORT = 65.0
    df_alt = df.copy()
    df_alt["temp_deviation"] = (df_alt["mean_tmax"] - COMFORT).abs()

    thresholds = np.percentile(df_alt["temp_deviation"].values, [33.3, 66.7])
    df_alt["stress_level"] = pd.cut(
        df_alt["temp_deviation"],
        bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
        labels=["Moderate", "Elevated", "Extreme"],
    )

    # --- Scatter: thermal stress vs fossil shift ---
    st.markdown("### Thermal Stress vs Fossil Generation Shift")
    st.markdown(
        "Each point is one day during a weather disaster. The x-axis measures how far "
        "temperatures deviated from a comfortable 65°F — capturing both freezing cold and "
        "scorching heat on a single 'thermal stress' axis."
    )

    event_label_map = {k: v["label"] for k, v in EVENT_META.items()}
    df_alt["event_label"] = df_alt["event"].map(event_label_map)

    scatter = (
        alt.Chart(df_alt)
        .mark_circle(size=70, opacity=0.75, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("temp_deviation:Q", title="Temperature Deviation from 65°F",
                     scale=alt.Scale(zero=True)),
            y=alt.Y("fossil_pct_change:Q", title="Fossil Shift (pp vs baseline)"),
            color=alt.Color(
                "stress_level:N",
                scale=alt.Scale(
                    domain=["Moderate", "Elevated", "Extreme"],
                    range=["#4CAF50", "#FF9800", "#D32F2F"],
                ),
                legend=alt.Legend(title="Stress Level"),
            ),
            shape=alt.Shape("event_label:N", legend=alt.Legend(title="Event")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("event_label:N", title="Event"),
                alt.Tooltip("mean_tmax:Q", title="Max Temp (°F)", format=".1f"),
                alt.Tooltip("temp_deviation:Q", title="Deviation (°F)", format=".1f"),
                alt.Tooltip("fossil_pct_change:Q", title="Fossil Shift (pp)", format="+.1f"),
                alt.Tooltip("stress_level:N", title="Stress Level"),
            ],
        )
        .properties(height=400)
    )

    # Regression line
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(
        df_alt["temp_deviation"], df_alt["fossil_pct_change"]
    )
    x_range = pd.DataFrame({
        "temp_deviation": np.linspace(df_alt["temp_deviation"].min(),
                                       df_alt["temp_deviation"].max(), 50)
    })
    x_range["fossil_pct_change"] = slope * x_range["temp_deviation"] + intercept

    reg_line = (
        alt.Chart(x_range)
        .mark_line(strokeDash=[6, 4], color="#333", strokeWidth=2, opacity=0.6)
        .encode(
            x="temp_deviation:Q",
            y="fossil_pct_change:Q",
        )
    )

    # Zero reference
    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], color=BASELINE_COLOR)
        .encode(y="y:Q")
    )

    # Threshold reference lines
    thresh_data = pd.DataFrame({"x": thresholds})
    thresh_rules = (
        alt.Chart(thresh_data)
        .mark_rule(strokeDash=[2, 2], color="#999", opacity=0.5)
        .encode(x="x:Q")
    )

    st.altair_chart(scatter + reg_line + zero_rule + thresh_rules, use_container_width=True)
    st.caption(
        f"Linear fit: R² = {r_val**2:.2f}, p = {p_val:.1e}. "
        f"Dotted verticals mark tercile boundaries ({thresholds[0]:.0f}°F, {thresholds[1]:.0f}°F deviation)."
    )

    # --- Bar chart: fossil shift by stress level ---
    st.markdown("### Fossil Shift by Thermal Stress Category")
    st.markdown(
        "Grouping days into three thermal stress categories reveals a dramatic staircase: "
        "moderate conditions show **no fossil increase**, while extreme stress drives "
        "**+10.7pp more fossil fuel** on the grid."
    )

    bin_stats = (
        df_alt.groupby("stress_level", observed=True)["fossil_pct_change"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    bin_stats.columns = ["Stress Level", "Mean Shift", "SE", "Days"]
    bin_stats["CI_lower"] = bin_stats["Mean Shift"] - 1.96 * bin_stats["SE"]
    bin_stats["CI_upper"] = bin_stats["Mean Shift"] + 1.96 * bin_stats["SE"]

    bar_chart = (
        alt.Chart(bin_stats)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Stress Level:N",
                     sort=["Moderate", "Elevated", "Extreme"],
                     axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Mean Shift:Q", title="Mean Fossil Shift (pp)"),
            color=alt.Color(
                "Stress Level:N",
                scale=alt.Scale(
                    domain=["Moderate", "Elevated", "Extreme"],
                    range=["#4CAF50", "#FF9800", "#D32F2F"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Stress Level:N"),
                alt.Tooltip("Mean Shift:Q", title="Mean Shift (pp)", format="+.1f"),
                alt.Tooltip("Days:Q", title="Days"),
            ],
        )
        .properties(height=350, width=400)
    )

    error_bars = (
        alt.Chart(bin_stats)
        .mark_errorbar(ticks=True)
        .encode(
            x=alt.X("Stress Level:N", sort=["Moderate", "Elevated", "Extreme"]),
            y=alt.Y("CI_lower:Q", title=""),
            y2="CI_upper:Q",
        )
    )

    st.altair_chart(bar_chart + error_bars, use_container_width=True)

    # Stats summary
    kw_groups = [
        df_alt[df_alt["stress_level"] == lvl]["fossil_pct_change"].values
        for lvl in ["Moderate", "Elevated", "Extreme"]
    ]
    h_stat, kw_p = sp_stats.kruskal(*kw_groups)
    mod_vals = kw_groups[0]
    ext_vals = kw_groups[2]
    u_stat, mw_p = sp_stats.mannwhitneyu(mod_vals, ext_vals, alternative="less")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Kruskal-Wallis p-value", f"{kw_p:.4f}",
                   delta="Significant" if kw_p < 0.05 else "Not significant")
    with col_s2:
        st.metric("Moderate vs Extreme gap",
                   f"+{ext_vals.mean() - mod_vals.mean():.1f}pp",
                   delta=f"Mann-Whitney p = {mw_p:.4f}")

    st.divider()
    st.markdown("### What This Means for Climate Change")
    st.markdown(
        "If the grid's fossil dependence accelerates non-linearly under thermal stress, "
        "then the **air quality penalty of climate change is worse than linear models predict**. "
        "As extreme weather events become more frequent and intense, each additional degree "
        "of warming pushes the grid further into fossil territory — and the air quality "
        "consequences compound."
    )
    st.info(
        "**Key takeaway:** Under moderate thermal stress, fossil generation stays near baseline. "
        "Under extreme stress (>24°F deviation from 65°F), fossil generation jumps **+10.7pp** "
        "above normal. This non-linear response means climate adaptation must prioritize "
        "grid resilience to prevent air quality cascades."
    )

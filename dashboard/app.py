"""ClimatePulse -- Canadian Agriculture + Climate Dashboard

Causal-chain explorer: Extreme Weather (ECCC) -> Crop Failure (StatsCan) -> Economic Impact (prices)
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ClimatePulse", layout="wide")

# -- Paths --
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHARTS_DIR = DATA_DIR / "charts"

# -- Colors --
DROUGHT_COLOR = "#d62728"
NORMAL_COLOR = "#2ca02c"
HIGHLIGHT_COLOR = "#ff7f0e"
MUTED_COLOR = "#999999"
PROVINCE_COLORS = {"Alberta": "#1f77b4", "Saskatchewan": "#d62728", "Manitoba": "#2ca02c"}
CROP_COLORS = {"Wheat": "#e6ab02", "Barley": "#7570b3", "Canola": "#1b9e77", "Oats": "#d95f02"}


# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_feature_matrix():
    return pd.read_csv(DATA_DIR / "ca_feature_matrix.csv")


@st.cache_data
def load_crop_yields():
    return pd.read_csv(DATA_DIR / "ca_crop_yields.csv")


@st.cache_data
def load_weather():
    return pd.read_csv(DATA_DIR / "ca_weather_features.csv")


@st.cache_data
def load_prices():
    df = pd.read_csv(DATA_DIR / "ca_farm_prices_monthly.csv")
    df["ref_date"] = pd.to_datetime(df["ref_date"])
    return df


@st.cache_data
def load_model_results():
    fp = DATA_DIR / "ca_model_results.json"
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


fm = load_feature_matrix()
yields = load_crop_yields()
weather = load_weather()
prices = load_prices()
model = load_model_results()


# ══════════════════════════════════════════════════════════════
# Section 1 -- Header
# ══════════════════════════════════════════════════════════════
st.markdown("# ClimatePulse: Canadian Agriculture and Climate")
st.markdown(
    "**Thesis:** Extreme Weather (ECCC) -> Crop Failure (StatsCan) -> Economic Impact (prices)"
)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Provinces", "3")
h2.metric("Crops", "4")
h3.metric("Years", "25 (2000-2024)")
h4.metric("Observations", f"{len(fm):,}")

st.divider()


# ══════════════════════════════════════════════════════════════
# Section 2 -- 2021 Drought Spotlight
# ══════════════════════════════════════════════════════════════
st.markdown("## 2021 Western Canada Drought -- The Focal Event")

holdout = model.get("holdout_2021", {})
predictions = holdout.get("predictions", [])
holdout_overall = holdout.get("overall", {})

spot_col1, spot_col2 = st.columns([1, 1])

with spot_col1:
    st.markdown("### Saskatchewan Wheat Collapse")

    # Pull SK wheat from holdout
    sk_wheat = next(
        (p for p in predictions if p["province"] == "Saskatchewan" and p["crop"] == "Wheat"),
        None,
    )
    if sk_wheat:
        s1, s2, s3 = st.columns(3)
        s1.metric("Actual Yield", f"{sk_wheat['actual']:,.0f} kg/ha")
        s2.metric("Model Predicted", f"{sk_wheat['predicted']:,.0f} kg/ha")
        s3.metric("Overprediction", f"+{sk_wheat['error_pct']:.0f}%")

    st.markdown(
        "The model, trained on normal years (purged CV R^2 = 0.68), predicted "
        "~3,041 kg/ha for Saskatchewan wheat. Actual: **1,890 kg/ha** -- a 24% drop "
        "below the historical baseline (~2,480 kg/ha). The 2021 drought was "
        "out-of-distribution: heat stress days doubled (22 vs ~9), max dry spell +40% "
        "(21 vs ~15 days)."
    )

with spot_col2:
    st.markdown("### Geographic Gradient of Overprediction")

    if predictions:
        pred_df = pd.DataFrame(predictions)
        # Heatmap: province x crop showing error_pct
        heatmap = (
            alt.Chart(pred_df)
            .mark_rect()
            .encode(
                x=alt.X("crop:N", title="Crop", sort=["Barley", "Canola", "Oats", "Wheat"]),
                y=alt.Y(
                    "province:N",
                    title="Province",
                    sort=["Saskatchewan", "Alberta", "Manitoba"],
                ),
                color=alt.Color(
                    "error_pct:Q",
                    title="Overprediction %",
                    scale=alt.Scale(scheme="reds", domain=[15, 75]),
                ),
                tooltip=[
                    alt.Tooltip("province:N"),
                    alt.Tooltip("crop:N"),
                    alt.Tooltip("actual:Q", title="Actual (kg/ha)", format=",.0f"),
                    alt.Tooltip("predicted:Q", title="Predicted (kg/ha)", format=",.0f"),
                    alt.Tooltip("error_pct:Q", title="Overprediction %", format=".1f"),
                ],
            )
            .properties(height=220)
        )
        text = (
            alt.Chart(pred_df)
            .mark_text(fontSize=12, fontWeight="bold")
            .encode(
                x=alt.X("crop:N", sort=["Barley", "Canola", "Oats", "Wheat"]),
                y=alt.Y(
                    "province:N", sort=["Saskatchewan", "Alberta", "Manitoba"]
                ),
                text=alt.Text("error_pct:Q", format=".0f"),
                color=alt.condition(
                    alt.datum.error_pct > 50,
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )
        st.altair_chart(heatmap + text, use_container_width=True)
        st.caption(
            "Saskatchewan shows the worst overprediction (53-72%), consistent with "
            "the epicentre of the 2021 heat dome. Manitoba fared best (17-36%)."
        )

st.divider()


# ══════════════════════════════════════════════════════════════
# Section 3 -- Causal Chain: Weather -> Yield
# ══════════════════════════════════════════════════════════════
st.markdown("## Causal Chain: Weather -> Yield")

cc_left, cc_right = st.columns([1, 3])

with cc_left:
    provinces = sorted(fm["province"].unique())
    crops = sorted(fm["crop"].unique())
    sel_province = st.selectbox("Province", provinces, index=provinces.index("Saskatchewan"))
    sel_crop = st.selectbox("Crop", crops, index=crops.index("Wheat"))
    weather_overlay = st.selectbox(
        "Weather overlay",
        ["heat_stress_days", "precip_total_mm", "max_consecutive_dry_days", "frost_free_days"],
        format_func=lambda x: {
            "heat_stress_days": "Heat Stress Days",
            "precip_total_mm": "Total Precipitation (mm)",
            "max_consecutive_dry_days": "Max Dry Spell (days)",
            "frost_free_days": "Frost-Free Days",
        }.get(x, x),
    )

with cc_right:
    subset = fm[(fm["province"] == sel_province) & (fm["crop"] == sel_crop)].copy()
    subset = subset.sort_values("year")

    # Yield timeseries
    base = alt.Chart(subset).encode(
        x=alt.X("year:O", title="Year"),
    )

    yield_line = base.mark_line(
        strokeWidth=2.5, color=PROVINCE_COLORS.get(sel_province, "#333")
    ).encode(
        y=alt.Y("yield_kg_ha:Q", title="Yield (kg/ha)"),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("yield_kg_ha:Q", title="Yield (kg/ha)", format=",.0f"),
        ],
    )

    yield_points = base.mark_circle(size=50).encode(
        y=alt.Y("yield_kg_ha:Q"),
        color=alt.condition(
            alt.datum.year == 2021,
            alt.value(DROUGHT_COLOR),
            alt.value(PROVINCE_COLORS.get(sel_province, "#333")),
        ),
        size=alt.condition(
            alt.datum.year == 2021,
            alt.value(120),
            alt.value(50),
        ),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("yield_kg_ha:Q", title="Yield (kg/ha)", format=",.0f"),
        ],
    )

    # Weather overlay on secondary axis
    weather_label = {
        "heat_stress_days": "Heat Stress Days",
        "precip_total_mm": "Precip (mm)",
        "max_consecutive_dry_days": "Max Dry Spell (days)",
        "frost_free_days": "Frost-Free Days",
    }.get(weather_overlay, weather_overlay)

    weather_bars = base.mark_bar(opacity=0.25, color=HIGHLIGHT_COLOR).encode(
        y=alt.Y(f"{weather_overlay}:Q", title=weather_label),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip(f"{weather_overlay}:Q", title=weather_label, format=".1f"),
        ],
    )

    weather_highlight = base.mark_bar(opacity=0.6, color=DROUGHT_COLOR).encode(
        y=alt.Y(f"{weather_overlay}:Q"),
    ).transform_filter(alt.datum.year == 2021)

    chart_yield = alt.layer(yield_line, yield_points).properties(
        height=300, title=f"{sel_province} -- {sel_crop}: Yield (kg/ha)"
    )
    chart_weather = alt.layer(weather_bars, weather_highlight).properties(
        height=180, title=f"{weather_label} (2021 highlighted in red)"
    )

    combined = alt.vconcat(chart_yield, chart_weather).resolve_scale(x="shared")
    st.altair_chart(combined, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════
# Section 4 -- Model Performance
# ══════════════════════════════════════════════════════════════
st.markdown("## Model Performance")

cv_results = model.get("cv_results", {})
cv_overall = cv_results.get("overall", {})

mp1, mp2 = st.columns([1, 1])

with mp1:
    st.markdown("### Cross-Validation (Normal Years)")
    st.markdown(
        f"Purged GroupKFold(5) by year with 1-year embargo. "
        f"Trained on {model.get('dataset', {}).get('training_rows', 264)} rows."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean R^2", f"{cv_overall.get('r2_mean', 0):.3f}")
    m2.metric("Mean MAE", f"{cv_overall.get('mae_mean', 0):.0f} kg/ha")
    m3.metric("Mean RMSE", f"{cv_overall.get('rmse_mean', 0):.0f} kg/ha")

    # Fold-level results table
    folds = cv_results.get("folds", [])
    if folds:
        fold_df = pd.DataFrame(folds)
        fold_df["years_held_out"] = fold_df["years_held_out"].apply(
            lambda x: ", ".join(str(y) for y in x)
        )
        fold_df = fold_df.rename(columns={
            "fold": "Fold",
            "years_held_out": "Held-Out Years",
            "n_train": "N Train",
            "r2": "R2",
            "mae": "MAE",
            "rmse": "RMSE",
        })
        st.dataframe(
            fold_df[["Fold", "Held-Out Years", "N Train", "R2", "MAE", "RMSE"]],
            use_container_width=True,
            hide_index=True,
        )

with mp2:
    st.markdown("### Holdout 2021 (Drought Year)")
    st.markdown(
        "The model was never trained on 2021. Systematic overprediction confirms "
        "2021 was out-of-distribution."
    )

    ho1, ho2, ho3 = st.columns(3)
    ho1.metric("Holdout R^2", f"{holdout_overall.get('r2', 0):.2f}")
    ho2.metric("Holdout MAE", f"{holdout_overall.get('mae', 0):.0f} kg/ha")
    ho3.metric("Holdout RMSE", f"{holdout_overall.get('rmse', 0):.0f} kg/ha")

    if predictions:
        ho_df = pd.DataFrame(predictions)
        ho_df = ho_df.rename(columns={
            "province": "Province",
            "crop": "Crop",
            "actual": "Actual (kg/ha)",
            "predicted": "Predicted (kg/ha)",
            "error": "Error (kg/ha)",
            "error_pct": "Error %",
        })
        st.dataframe(ho_df, use_container_width=True, hide_index=True)

# SHAP charts
st.markdown("### Feature Importance (SHAP)")
shap_col1, shap_col2 = st.columns(2)

shap_bar = CHARTS_DIR / "shap_summary_bar.png"
shap_bee = CHARTS_DIR / "shap_summary_bee.png"

with shap_col1:
    if shap_bar.exists():
        st.image(str(shap_bar), caption="SHAP Feature Importance (bar)", use_container_width=True)
    else:
        st.info("shap_summary_bar.png not found.")

with shap_col2:
    if shap_bee.exists():
        st.image(str(shap_bee), caption="SHAP Beeswarm Plot", use_container_width=True)
    else:
        st.info("shap_summary_bee.png not found.")

# Additional SHAP charts
shap_row2_col1, shap_row2_col2, shap_row2_col3 = st.columns(3)

shap_2021 = CHARTS_DIR / "shap_2021_drought.png"
shap_heat = CHARTS_DIR / "shap_dependence_heat_stress.png"
shap_precip = CHARTS_DIR / "shap_dependence_precip.png"

with shap_row2_col1:
    if shap_2021.exists():
        st.image(str(shap_2021), caption="SHAP -- 2021 Drought", use_container_width=True)

with shap_row2_col2:
    if shap_heat.exists():
        st.image(str(shap_heat), caption="SHAP Dependence -- Heat Stress", use_container_width=True)

with shap_row2_col3:
    if shap_precip.exists():
        st.image(str(shap_precip), caption="SHAP Dependence -- Precipitation", use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════
# Section 5 -- Price Impact
# ══════════════════════════════════════════════════════════════
st.markdown("## Price Impact: Yield Anomaly -> Price Change")

price_impact = model.get("price_impact", {})
ols = price_impact.get("ols", {})
pearson = price_impact.get("pearson", {})

pi1, pi2 = st.columns([2, 1])

with pi1:
    # Build scatter from feature matrix rows that have price data
    price_df = fm.dropna(subset=["price_change_pct", "yield_kg_ha"]).copy()

    if len(price_df) > 0:
        # Compute yield anomaly as % deviation from crop-province mean
        mean_yields = price_df.groupby(["province", "crop"])["yield_kg_ha"].transform("mean")
        price_df["yield_anomaly_pct"] = (
            (price_df["yield_kg_ha"] - mean_yields) / mean_yields
        ) * 100

        scatter = (
            alt.Chart(price_df)
            .mark_circle(size=50, opacity=0.6)
            .encode(
                x=alt.X("yield_anomaly_pct:Q", title="Yield Anomaly (% from mean)"),
                y=alt.Y("price_change_pct:Q", title="Price Change (%)"),
                color=alt.Color(
                    "crop:N",
                    title="Crop",
                    scale=alt.Scale(
                        domain=list(CROP_COLORS.keys()),
                        range=list(CROP_COLORS.values()),
                    ),
                ),
                tooltip=[
                    alt.Tooltip("year:O", title="Year"),
                    alt.Tooltip("province:N"),
                    alt.Tooltip("crop:N"),
                    alt.Tooltip("yield_anomaly_pct:Q", title="Yield Anomaly %", format=".1f"),
                    alt.Tooltip("price_change_pct:Q", title="Price Change %", format=".1f"),
                ],
            )
            .properties(height=380)
        )

        # OLS regression line
        x_range = pd.DataFrame({
            "yield_anomaly_pct": np.linspace(
                price_df["yield_anomaly_pct"].min(),
                price_df["yield_anomaly_pct"].max(),
                50,
            )
        })
        slope = ols.get("slope", -0.079)
        intercept = ols.get("intercept", 3.92)
        x_range["price_change_pct"] = slope * x_range["yield_anomaly_pct"] + intercept

        ols_line = (
            alt.Chart(x_range)
            .mark_line(strokeDash=[6, 3], color="#333", strokeWidth=2)
            .encode(
                x="yield_anomaly_pct:Q",
                y="price_change_pct:Q",
            )
        )

        st.altair_chart(scatter + ols_line, use_container_width=True)
    else:
        st.info("No price data available for scatter plot.")

with pi2:
    st.markdown("### Overall Correlation")
    st.metric("Pearson r", f"{pearson.get('r', 0):.3f}")
    st.metric("OLS R^2", f"{ols.get('r2', 0):.3f}")
    st.metric("OLS slope", f"{ols.get('slope', 0):.3f}")
    st.metric("p-value", f"< 0.001")

    st.markdown("### Per-Crop Breakdown")
    per_crop = price_impact.get("per_crop", {})
    crop_rows = []
    for crop_name, crop_data in per_crop.items():
        crop_pearson = crop_data.get("pearson", {})
        crop_rows.append({
            "Crop": crop_name,
            "n": crop_data.get("n", 0),
            "r": crop_pearson.get("r", 0),
            "p": crop_pearson.get("p", 1),
        })
    if crop_rows:
        crop_corr_df = pd.DataFrame(crop_rows)
        st.dataframe(crop_corr_df, use_container_width=True, hide_index=True)

    st.markdown(
        "Barley and canola drive the price signal. "
        "Wheat shows no useful correlation (r = -0.07, p = 0.72)."
    )

st.divider()


# ══════════════════════════════════════════════════════════════
# Section 6 -- Data Explorer
# ══════════════════════════════════════════════════════════════
st.markdown("## Data Explorer")

tab_fm, tab_yields, tab_weather, tab_prices = st.tabs([
    "Feature Matrix", "Crop Yields", "Weather Features", "Farm Prices"
])

with tab_fm:
    exp_provinces = st.multiselect(
        "Filter by province", sorted(fm["province"].unique()), default=[], key="fm_prov"
    )
    exp_crops = st.multiselect(
        "Filter by crop", sorted(fm["crop"].unique()), default=[], key="fm_crop"
    )
    filtered_fm = fm.copy()
    if exp_provinces:
        filtered_fm = filtered_fm[filtered_fm["province"].isin(exp_provinces)]
    if exp_crops:
        filtered_fm = filtered_fm[filtered_fm["crop"].isin(exp_crops)]
    st.dataframe(filtered_fm, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered_fm)} rows")

with tab_yields:
    st.dataframe(yields, use_container_width=True, hide_index=True)
    st.caption(f"{len(yields)} rows")

with tab_weather:
    st.dataframe(weather, use_container_width=True, hide_index=True)
    st.caption(f"{len(weather)} rows")

with tab_prices:
    price_commodity = st.text_input("Filter commodity (contains)", "", key="price_filter")
    filtered_prices = prices.copy()
    if price_commodity:
        filtered_prices = filtered_prices[
            filtered_prices["commodity"].str.contains(price_commodity, case=False, na=False)
        ]
    st.dataframe(filtered_prices, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered_prices)} rows")


# -- Footer --
st.divider()
st.caption(
    "ClimatePulse -- ZerveHack 2026 -- Climate & Energy Track -- "
    "Data: StatsCan, ECCC"
)

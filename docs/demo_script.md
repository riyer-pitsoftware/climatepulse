# ClimatePulse Demo Script — 3 Minutes

**ZerveHack 2026 | Climate & Energy Track**
**Thesis:** Extreme Weather → Crop Failure → Economic Impact on the Canadian Prairies

---

## [0:00–0:20] HOOK — The 2021 Prairie Drought

**Screen:** Full-screen still image of parched Saskatchewan farmland (2021 drought footage). Fade to a single stat card: "Saskatchewan Wheat Yield: -24% below baseline."

**Speaker:**

> In the summer of 2021, a heat dome parked over western Canada for weeks. Saskatchewan wheat yields collapsed to 1,890 kilograms per hectare — 24 percent below the 25-year baseline. Wheat prices surged past $300 a tonne. Farmers lost crops. Consumers paid more. The question we asked: can we trace this chain from weather to harvest to price — and can data tell us when it's coming?

---

## [0:20–0:50] THE CAUSAL CHAIN

**Screen:** Animated three-link chain diagram, each link appearing in sequence:
1. Extreme Weather (heat stress, drought)
2. Crop Failure (yield collapse)
3. Economic Impact (price spikes)

Beneath the chain, a timeline showing 2021 data points: heat stress days doubling from ~9 to 22, max dry spell rising from ~15 to 21 days, wheat prices crossing $300/tonne.

**Speaker:**

> ClimatePulse models a three-link causal chain. First, extreme weather — heat stress days more than doubled in 2021, and maximum dry spells increased 40 percent. Second, crop failure — yields fell across all four major Prairie crops: wheat, canola, barley, and oats. Third, economic impact — commodity prices spiked as supply contracted. We built separate models for each link and tested whether the full chain holds together.

---

## [0:50–1:30] DATA AND ZERVE WORKFLOW

**Screen:** Switch to the Zerve workspace. Show the 4-branch parallel DAG:
- Branch 1: StatsCan yields (Table 32-10-0359)
- Branch 2: StatsCan prices (Table 32-10-0077)
- Branch 3: ECCC weather (10 stations, daily obs)
- Branch 4: Feature matrix join → XGBoost training

Briefly hover over each branch. Then show the output: the 300-row feature matrix table scrolling in a dataframe view.

**Speaker:**

> We built this entirely on Zerve using a four-branch DAG. Three branches pull raw data: crop yields from Statistics Canada Table 32-10-0359, farm product prices from Table 32-10-0077, and daily weather observations from 10 Environment Canada stations across Alberta, Saskatchewan, and Manitoba. The fourth branch joins everything into a 300-row feature matrix — three provinces, four crops, 25 years. Thirteen engineered features: eight weather variables like growing degree days, heat stress days, and precipitation windows; three lagged features capturing the previous year's conditions; and province and crop identifiers. All processing runs in parallel on Zerve, reproducible end to end.

---

## [1:30–2:10] KEY FINDINGS

**Screen:** Dashboard view. Show three panels in sequence:
1. Bar chart: heat stress days by year, 2021 spike highlighted in red
2. Scatter plot: yield vs. heat stress days, 2021 cluster clearly separated
3. Line chart: wheat price per tonne over time, 2021 spike annotated

**Speaker:**

> Here is what the data shows. Heat stress days — days above 30 degrees Celsius during the growing season — more than doubled in 2021 compared to the historical average. Maximum consecutive dry days jumped 40 percent. The yield impact was immediate: Saskatchewan wheat fell to 1,890 kilograms per hectare. And the price response followed — wheat crossed $300 a tonne across all three provinces.
>
> But the story is not uniform across crops. Barley and canola show the strongest yield-to-price signal. Wheat, despite being the most visible crop, has a surprisingly weak price correlation. That nuance matters — and it is something you only see when you look at the data crop by crop.

---

## [2:10–2:40] MODEL AND HONESTY

**Screen:** Split view. Left panel: XGBoost cross-validation results table (purged CV R^2 = 0.68, per-fold breakdown). Right panel: SHAP beeswarm plot showing feature importance. Then transition to the 2021 holdout results table showing predicted vs. actual yields with large overprediction errors.

**Speaker:**

> Our XGBoost yield model achieves a purged cross-validated R-squared of 0.68 on normal years, using grouped K-fold with embargo to prevent lag leakage. SHAP analysis shows previous year's yield and crop type dominate, with heat stress and precipitation as the key weather drivers.
>
> But here is the honest part. On the 2021 drought holdout, the model fails — R-squared of negative 2.3. It systematically overpredicts yields because the drought is out of distribution. The model has never seen conditions that extreme. Saskatchewan has the worst overprediction, then Alberta, then Manitoba — matching the drought's actual geographic gradient. The price correlation is moderate overall — Pearson r of negative 0.37 — but explains only 14 percent of price variance.
>
> A brief note on our journey: we originally investigated US extreme weather and fossil fuel generation. That model scored an R-squared of 0.047. We pivoted to Canadian agriculture because the data was stronger and the story more compelling. We think that pivot — and being honest about what works and what does not — is part of good data science.

---

## [2:40–3:00] STAKES AND CLOSE

**Screen:** Return to the three-link chain diagram, now overlaid with a forward-looking climate projection arrow. Final card: "As extreme weather intensifies, these patterns will worsen."

**Speaker:**

> Climate change is a food security problem. The patterns we found — heat stress driving yield loss, yield loss associated with price spikes — are directionally consistent with the thesis, even if the evidence is partial. The 2021 drought is not an anomaly. It is a preview. As extreme weather intensifies on the Canadian Prairies, the chain from heat to harvest to household cost will tighten. ClimatePulse gives us a framework to watch it happen — and the honesty to say where the data stops and the uncertainty begins. Thank you.

---

## Production Notes

| Element | Source / Asset |
|---------|---------------|
| Drought imagery | Public domain, Government of Saskatchewan 2021 drought documentation |
| Chain diagram | Custom graphic, 3-link animated SVG |
| DAG walkthrough | Live Zerve workspace, 4-branch view |
| Dashboard panels | Zerve App Builder or Streamlit, connected to `ca_feature_matrix.csv` |
| CV results table | From `ca_model_results.json` → `cv_results` |
| SHAP beeswarm | `data/processed/charts/shap_summary_bee.png` |
| 2021 holdout table | From `ca_model_results.json` → `holdout_2021` |
| SHAP waterfall | `data/processed/charts/shap_2021_drought.png` (Saskatchewan Wheat) |

**Total runtime target:** 2:55–3:00. Do not exceed 3:00.

**Tone:** Serious, data-driven, accessible. No hype. Let the numbers speak. The intellectual honesty — about model failure, about the pivot, about partial evidence — is the differentiator.

**Rehearsal priority:** The Model and Honesty section (2:10–2:40) is the densest. Practice delivering the negative R-squared and pivot narrative smoothly without rushing. The judges weight Technical Depth at 35% and Innovation at 35% — this section serves both.

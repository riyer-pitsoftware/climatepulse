ClimatePulse: When the Heat Hits the Harvest

When extreme weather strikes the Canadian Prairies, what happens to the food supply — and what does that mean for commodity prices? ClimatePulse investigates a three-link causal chain: extreme weather damages crop yields, yield collapses drive commodity price spikes, and those price shocks ripple through the agricultural economy.

We study three Prairie provinces (Alberta, Saskatchewan, Manitoba) across 25 years of data (2000–2024), joining daily weather station observations from Environment and Climate Change Canada (ECCC), annual crop yield statistics from Statistics Canada (Table 32-10-0359), and monthly farm product prices from Statistics Canada (Table 32-10-0077) into a unified 300-row feature matrix covering wheat, canola, barley, and oats.

The signal is clear. During the 2021 Western Canadian drought — the worst in living memory — Saskatchewan wheat yields collapsed 44% below normal (1,890 kg/ha vs ~3,400 typical). Heat stress days tripled (22 vs ~8 normal), maximum dry spells nearly doubled (21 days vs ~12), and growing season precipitation dropped 20%. Wheat prices surged past $300/tonne across all three provinces.

Our XGBoost model trained on growing season weather features (growing degree days, heat stress days, precipitation timing, consecutive dry days, frost-free period) predicts crop yields with strong accuracy when validated against the held-out 2021 drought year. SHAP analysis reveals which weather features matter most — and when — giving agricultural planners actionable early warning signals.

This is not just a historical analysis. As climate change intensifies extreme weather events on the Prairies, ClimatePulse demonstrates that food security is a climate security problem. The same drought patterns that devastated 2021 harvests are projected to become more frequent. The question is not whether the next crop failure will happen, but when.

## Technical Architecture

- **Data sources:** StatsCan (yields, prices), ECCC (10 weather stations, daily observations)
- **Pipeline:** 4-stage DAG — weather | yields | prices → join → feature matrix
- **Model:** XGBoost regressor (weather → yield), price impact model (yield anomaly → price spike)
- **Features:** 19 columns including growing degree days, heat stress, precipitation timing, lagged yields
- **Deployment:** Zerve 4-branch DAG with /predict-yield and /predict-price-impact API endpoints

## Historical Note

ClimatePulse originally investigated US extreme weather events (Heat Dome, Winter Storm Uri, Winter Storm Elliott) and their impact on grid fossil fuel generation and air quality. That approach produced strong statistical correlations (lag-1 fossil→AQI r=+0.38, p=0.002) but the ML model failed cross-validation (R²=0.09). The Canadian Agriculture pivot was adopted after team evaluation showed stronger data, better model fundamentals, and a more compelling story. The original US analysis is preserved in `docs/model_training_guide.md` and `docs/feature_engineering_plan.md` as an honest failure narrative (Act 1 of the demo).

ClimatePulse: When the Heat Hits the Harvest

When extreme weather strikes the Canadian Prairies, what happens to the food supply -- and what does that mean for commodity prices? ClimatePulse investigates the relationship between extreme weather, crop yields, and commodity prices across the Prairie provinces.

We study three Prairie provinces (Alberta, Saskatchewan, Manitoba) across 25 years of data (2000-2024), joining daily weather station observations from Environment and Climate Change Canada (ECCC), annual crop yield statistics from Statistics Canada (Table 32-10-0359), and monthly farm product prices from Statistics Canada (Table 32-10-0077) into a unified 300-row feature matrix covering wheat, canola, barley, and oats.

The 2021 Western Canadian drought is our focal event. Saskatchewan wheat yields dropped 24% below the 25-year baseline (1,890 kg/ha vs ~2,480 typical). Heat stress days more than doubled (22 vs ~9 normal), and maximum dry spells increased 40% (21 days vs ~15). Wheat prices surged past $300/tonne across all three provinces.

Our XGBoost model trained on weather and lagged yield features achieves purged cross-validated R^2 = 0.68 on normal years (purged GroupKFold with embargo=1 to prevent lag leakage through prev_year features). On the held-out 2021 drought year, the model fails: it systematically overpredicts yields (holdout R^2 = -2.3), because the drought is out-of-distribution. The pattern of overprediction (Saskatchewan worst, then Alberta, then Manitoba) is consistent with the drought's geographic gradient, but this is an observation about residuals, not a validated prediction. A secondary price-impact analysis finds a moderate negative correlation between yield anomalies and price changes (overall Pearson r = -0.37, p < 0.001, OLS R^2 = 0.14), strongest for barley and canola; wheat shows no useful signal.

This is an exploratory analysis, not a confirmatory one. The yield model works on normal years but cannot predict extreme drought events. The price link is statistically significant but explains only ~14% of price variance. As climate change intensifies extreme weather on the Prairies, the patterns we observe -- heat stress driving yield loss, yield loss associated with price spikes -- are directionally consistent with the thesis that food security is tied to climate outcomes, but the evidence is partial.

## Technical Architecture

- **Data sources:** StatsCan (yields, prices), ECCC (10 weather stations, daily observations)
- **Pipeline:** 4-stage DAG -- weather | yields | prices -> join -> feature matrix
- **Model:** XGBoost regressor (weather -> yield), price impact model (yield anomaly -> price change)
- **Features:** 13 model features from 19 columns: weather (8), lagged (3), categorical (2)
- **Deployment:** Zerve 4-branch DAG with /predict-yield and /predict-price-impact API endpoints

## Historical Note

ClimatePulse originally investigated US extreme weather events (Heat Dome, Winter Storm Uri, Winter Storm Elliott) and their impact on grid fossil fuel generation and air quality. That approach found a confirmed fossil surge during extreme weather (+3.44pp, p=0.0003) but the fossil->AQI link was weak (pooled lag-1 r=+0.14, p=0.19; Uri-specific r=+0.35, p=0.051) and the ML model failed cross-validation (Ridge R^2=0.047, ElasticNet R^2=-0.14). The Canadian Agriculture pivot was adopted after team evaluation showed stronger data, better model fundamentals, and a more compelling story. The original US analysis is preserved in `docs/model_training_guide.md` and `docs/feature_engineering_plan.md` as an honest failure narrative (Act 1 of the demo).

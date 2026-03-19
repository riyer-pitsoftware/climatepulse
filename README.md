# ClimatePulse

**Extreme Weather → Crop Failure → Economic Impact**

ClimatePulse analyzes how extreme weather events on the Canadian Prairies drive crop yield collapse and commodity price spikes. Built for the [ZerveHack hackathon](https://zervehack.devpost.com/) (Climate & Energy track).

## Quick Start

```bash
pip install -r requirements.txt

# Run data pipelines (order matters for join step)
python scripts/pipeline_statcan_yields.py     # StatsCan crop yields
python scripts/pipeline_statcan_prices.py     # StatsCan farm prices
python scripts/pipeline_eccc_weather.py       # ECCC weather (10 Prairie stations, ~3 min)
python scripts/pipeline_feature_matrix.py     # Join → feature matrix
```

## Data Sources

| Source | What | Coverage |
|--------|------|----------|
| [StatsCan 32-10-0359](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210035901) | Crop yields, area, production | Prairie provinces, 1908–2025 |
| [StatsCan 32-10-0077](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210007701) | Monthly farm product prices | Prairie provinces, 1980–2026 |
| [ECCC Climate Data](https://climate.weather.gc.ca/) | Daily temp, precip (10 stations) | 2000–2024 |

## Architecture

See [docs/pipeline_architecture.md](docs/pipeline_architecture.md) for full pipeline DAG with Mermaid diagrams.

## License

See [LICENSE](LICENSE).

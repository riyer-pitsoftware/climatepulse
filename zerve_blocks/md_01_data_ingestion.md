# Branch 1–3: Parallel Data Ingestion

Three independent government data sources, fetched in parallel:

| Branch | Source | Volume | Key Fields |
|--------|--------|--------|------------|
| Yields | StatsCan Table 32-10-0359 | 312 rows | yield_kg_ha, harvested_ha, production_mt |
| Prices | StatsCan Table 32-10-0077 | 16,776 rows | monthly price (CAD), commodity, province |
| Weather | ECCC Climate API (10 stations) | 75 rows | GDD, heat stress days, precip windows, frost-free days |

**Coverage:** Alberta, Saskatchewan, Manitoba — 2000 to 2024. Four crops: wheat, canola, barley, oats.

No branch depends on another. The DAG runs all three concurrently.

# Zerve Fleet: Parallel Weather Ingestion

## Why Fleet?

The ECCC weather pipeline makes 102 independent HTTP requests (3 provinces × 34 years). Each request fetches monthly climate records for a single province-year combination. There are no dependencies between requests — this is embarrassingly parallel.

Running sequentially, this takes approximately 3 minutes due to API response times and rate-limiting pauses. Fleet distributes these fetches across parallel workers.

## Design

```
                    Fleet Workers (10 concurrent)
                    ┌─────────────────────────────┐
(AB, 1990) ────────>│  Worker 1: fetch_province_year  │───> records
(AB, 1991) ────────>│  Worker 2: fetch_province_year  │───> records
(SK, 1990) ────────>│  Worker 3: fetch_province_year  │───> records
   ...              │       ...                       │     ...
(MB, 2023) ────────>│  Worker 10: fetch_province_year │───> records
                    └─────────────────────────────┘
                                  │
                                  v
                         raw_monthly_df
                    (same schema as sequential)
```

**Concurrency = 10 workers.** Rate-limited to respect ECCC's API — each worker pauses 0.1s between paginated requests within a single fetch.

## What This Demonstrates

- **Zerve Fleet integration:** Not just running code in notebooks, but leveraging Fleet's parallel execution infrastructure
- **Practical parallelism:** 102 independent I/O-bound tasks distributed across workers with controlled concurrency
- **API-respectful design:** Rate limiting preserved even under parallel execution — responsible data sourcing from government APIs

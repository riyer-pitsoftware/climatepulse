# Fan-In Join: Feature Matrix

The three upstream branches merge on **(province, year)** into a single 300-row × 19-column feature matrix.

**13 model features:**
- 8 weather: GDD, heat stress days, total precip, early-season precip (May–Jun), mid-season precip (Jul–Aug), max consecutive dry days, frost-free days, mean growing-season temp
- 3 lagged: previous year's yield, precipitation, and GDD
- 2 categorical: province and crop (label-encoded)

**Key design decisions:**
- Year 2000 dropped (no lag features available)
- Year 2022 excluded from training (its `prev_year_*` features embed 2021 holdout target values)
- Year 2021 reserved as the drought holdout — never seen during training

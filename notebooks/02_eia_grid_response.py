"""
02_eia_grid_response.py — Process raw EIA hourly generation data into
a grid-response analysis dataset for ClimatePulse.

Compares fuel-mix shifts during three extreme-weather events against
their baselines to quantify how grids lean on fossil fuels under stress.
"""

import pandas as pd
import pathlib
import sys

# ── paths ──────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = _ROOT / "data" / "raw" / "eia"
OUT = _ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── event definitions ──────────────────────────────────────────────────
EVENTS = {
    "ERCO_URI_2021": {
        "event_file": RAW / "eia_erco_uri_2021.csv",
        "baseline_file": RAW / "eia_erco_baseline_2020.csv",
        "label": "Winter Storm Uri (ERCOT, Feb 2021)",
    },
    "PJM_Elliott_2022": {
        "event_file": RAW / "eia_pjm_elliott_2022.csv",
        "baseline_file": RAW / "eia_pjm_baseline_2021.csv",
        "label": "Winter Storm Elliott (PJM, Dec 2022)",
    },
    "BPAT_HeatDome_2021": {
        "event_file": RAW / "eia_bpat_heatdome_2021.csv",
        "baseline_file": RAW / "eia_bpat_baseline_2020.csv",
        "label": "Pacific NW Heat Dome (BPA, Jun 2021)",
    },
}

# cp-9v9: Same-year pre-event baseline files (2 weeks before each event).
# These are processed identically to event data and tagged with
# period_type="pre_event_baseline" so the join script can include them
# as counterfactual "normal grid" rows for the ML model.
PRE_EVENT_BASELINES = {
    "ERCO_URI_2021": {
        "file": RAW / "eia_erco_pre_event_2021.csv",
        "label": "ERCO Pre-Event Baseline (Jan 2021)",
    },
    "BPAT_HeatDome_2021": {
        "file": RAW / "eia_bpat_pre_event_2021.csv",
        "label": "BPAT Pre-Event Baseline (Jun 2021)",
    },
    "PJM_Elliott_2022": {
        "file": RAW / "eia_pjm_pre_event_2022.csv",
        "label": "PJM Pre-Event Baseline (Dec 2022)",
    },
}

# ── fuel-type categorization ──────────────────────────────────────────
FUEL_CATEGORIES = {
    "SUN": "renewable",
    "WND": "renewable",
    "WAT": "renewable",
    "COL": "fossil",
    "NG":  "fossil",
    "OIL": "fossil",
    "NUC": "other",
    "OTH": "other",
}


def load_and_tag(path: pathlib.Path, event_name: str, period_type: str) -> pd.DataFrame:
    """Load a CSV, parse timestamps, tag with event name and period type."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
    df["event"] = event_name
    df["period_type"] = period_type  # 'event' or 'baseline'
    df["fuel_type"] = df["fueltype"]
    df["fuel_category"] = df["fueltype"].map(FUEL_CATEGORIES).fillna("other")
    df["generation_mwh"] = df["value"]
    return df


def compute_hourly_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_generation_mwh and pct_of_total per hour."""
    hourly_total = (
        df.groupby(["datetime", "event", "period_type"])["generation_mwh"]
        .sum()
        .rename("total_generation_mwh")
    )
    df = df.merge(hourly_total, on=["datetime", "event", "period_type"])
    df["pct_of_total"] = (df["generation_mwh"] / df["total_generation_mwh"] * 100).round(2)
    return df


def category_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate generation by fuel_category per hour."""
    return (
        df.groupby(["datetime", "event", "period_type", "fuel_category"])
        .agg(generation_mwh=("generation_mwh", "sum"))
        .reset_index()
    )


def analyze_event(event_df: pd.DataFrame, baseline_df: pd.DataFrame, label: str):
    """Print key findings for one event vs its baseline."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Aggregate by category per hour
    ev_cat = category_hourly(event_df)
    bl_cat = category_hourly(baseline_df)

    # ── renewable % ────────────────────────────────────────────────
    def avg_pct(cat_df, category):
        """Average hourly % of total for a fuel category."""
        hourly_total = cat_df.groupby("datetime")["generation_mwh"].sum()
        cat_gen = (
            cat_df[cat_df["fuel_category"] == category]
            .groupby("datetime")["generation_mwh"]
            .sum()
        )
        # align indices
        pct = (cat_gen / hourly_total * 100).dropna()
        return pct

    bl_renew_pct = avg_pct(bl_cat, "renewable")
    ev_renew_pct = avg_pct(ev_cat, "renewable")
    bl_fossil_pct = avg_pct(bl_cat, "fossil")
    ev_fossil_pct = avg_pct(ev_cat, "fossil")

    bl_renew_avg = bl_renew_pct.mean()
    ev_renew_avg = ev_renew_pct.mean()
    bl_fossil_avg = bl_fossil_pct.mean()
    ev_fossil_avg = ev_fossil_pct.mean()

    renew_change = ev_renew_avg - bl_renew_avg
    fossil_change = ev_fossil_avg - bl_fossil_avg

    print(f"\n  Baseline avg renewable share:  {bl_renew_avg:6.1f}%")
    print(f"  Event avg renewable share:     {ev_renew_avg:6.1f}%")
    print(f"  >> Renewable share change:     {renew_change:+6.1f} pp")

    print(f"\n  Baseline avg fossil share:     {bl_fossil_avg:6.1f}%")
    print(f"  Event avg fossil share:        {ev_fossil_avg:6.1f}%")
    print(f"  >> Fossil share change:        {fossil_change:+6.1f} pp")

    # ── peak fossil generation ─────────────────────────────────────
    ev_fossil_gen = (
        ev_cat[ev_cat["fuel_category"] == "fossil"]
        .groupby("datetime")["generation_mwh"]
        .sum()
    )
    peak_fossil_mwh = ev_fossil_gen.max()
    peak_fossil_hour = ev_fossil_gen.idxmax()
    print(f"\n  Peak fossil generation:        {peak_fossil_mwh:,.0f} MWh")
    print(f"    at hour:                     {peak_fossil_hour}")

    # ── peak fossil % of total ─────────────────────────────────────
    peak_fossil_pct = ev_fossil_pct.max()
    peak_fossil_pct_hour = ev_fossil_pct.idxmax()
    print(f"  Peak fossil % of total:        {peak_fossil_pct:6.1f}%")
    print(f"    at hour:                     {peak_fossil_pct_hour}")

    # ── minimum renewable generation ───────────────────────────────
    ev_renew_gen = (
        ev_cat[ev_cat["fuel_category"] == "renewable"]
        .groupby("datetime")["generation_mwh"]
        .sum()
    )
    min_renew_mwh = ev_renew_gen.min()
    min_renew_hour = ev_renew_gen.idxmin()
    print(f"\n  Min renewable generation:      {min_renew_mwh:,.0f} MWh")
    print(f"    at hour:                     {min_renew_hour}")
    min_renew_pct = ev_renew_pct.min()
    min_renew_pct_hour = ev_renew_pct.idxmin()
    print(f"  Min renewable % of total:      {min_renew_pct:6.1f}%")
    print(f"    at hour:                     {min_renew_pct_hour}")

    # ── absolute generation comparison ─────────────────────────────
    bl_fossil_avg_mwh = (
        bl_cat[bl_cat["fuel_category"] == "fossil"]
        .groupby("datetime")["generation_mwh"]
        .sum()
        .mean()
    )
    ev_fossil_avg_mwh = ev_fossil_gen.mean()
    fossil_mwh_change = (ev_fossil_avg_mwh - bl_fossil_avg_mwh) / bl_fossil_avg_mwh * 100

    bl_renew_avg_mwh = (
        bl_cat[bl_cat["fuel_category"] == "renewable"]
        .groupby("datetime")["generation_mwh"]
        .sum()
        .mean()
    )
    ev_renew_avg_mwh = ev_renew_gen.mean()
    renew_mwh_change = (ev_renew_avg_mwh - bl_renew_avg_mwh) / bl_renew_avg_mwh * 100

    print(f"\n  Avg fossil gen — baseline:     {bl_fossil_avg_mwh:,.0f} MWh/hr")
    print(f"  Avg fossil gen — event:        {ev_fossil_avg_mwh:,.0f} MWh/hr")
    print(f"  >> Fossil generation change:   {fossil_mwh_change:+.1f}%")

    print(f"\n  Avg renewable gen — baseline:  {bl_renew_avg_mwh:,.0f} MWh/hr")
    print(f"  Avg renewable gen — event:     {ev_renew_avg_mwh:,.0f} MWh/hr")
    print(f"  >> Renewable generation change: {renew_mwh_change:+.1f}%")

    # ── hours of maximum stress (top-5 by fossil %) ────────────────
    top_stress = ev_fossil_pct.nlargest(5)
    print(f"\n  Top 5 hours of grid stress (highest fossil %):")
    for dt, pct in top_stress.items():
        print(f"    {dt}  —  fossil {pct:.1f}%")


# ── main ───────────────────────────────────────────────────────────────
def main():
    all_frames = []

    for ename, cfg in EVENTS.items():
        print(f"Loading {ename} ...")
        ev_raw = load_and_tag(cfg["event_file"], ename, "event")
        bl_raw = load_and_tag(cfg["baseline_file"], ename, "baseline")

        print(f"  event rows: {len(ev_raw):,}   baseline rows: {len(bl_raw):,}")
        print(f"  event period: {ev_raw['datetime'].min()} → {ev_raw['datetime'].max()}")
        print(f"  baseline period: {bl_raw['datetime'].min()} → {bl_raw['datetime'].max()}")

        # Compute mix columns
        ev = compute_hourly_mix(ev_raw)
        bl = compute_hourly_mix(bl_raw)

        # Analysis printout
        analyze_event(ev, bl, cfg["label"])

        all_frames.append(ev)
        all_frames.append(bl)

    # cp-9v9: Load and process pre-event baseline files (same-year, 2 weeks before event).
    # These are tagged as period_type="pre_event_baseline" and use the SAME event name
    # so the join script can pair them with the right event.
    for ename, bl_cfg in PRE_EVENT_BASELINES.items():
        bl_file = bl_cfg["file"]
        if not bl_file.exists():
            print(f"\n  SKIP pre-event baseline for {ename}: {bl_file.name} not yet pulled")
            continue
        print(f"\nLoading pre-event baseline: {bl_cfg['label']} ...")
        bl_raw = load_and_tag(bl_file, ename, "pre_event_baseline")
        print(f"  rows: {len(bl_raw):,}")
        print(f"  period: {bl_raw['datetime'].min()} → {bl_raw['datetime'].max()}")
        bl = compute_hourly_mix(bl_raw)
        all_frames.append(bl)

    # ── build combined output DataFrame ────────────────────────────
    combined = pd.concat(all_frames, ignore_index=True)
    out_cols = [
        "datetime", "event", "period_type",
        "fuel_type", "fuel_category",
        "generation_mwh", "total_generation_mwh", "pct_of_total",
    ]
    combined = combined[out_cols].sort_values(
        ["event", "period_type", "datetime", "fuel_category", "fuel_type"]
    ).reset_index(drop=True)

    out_path = OUT / "eia_grid_response.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"  Saved combined dataset: {out_path}")
    print(f"  Shape: {combined.shape[0]:,} rows x {combined.shape[1]} columns")
    print(f"  Events: {combined['event'].nunique()}")
    print(f"  Date range: {combined['datetime'].min()} → {combined['datetime'].max()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

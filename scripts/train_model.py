#!/usr/bin/env python3
"""Model training for ClimatePulse ML pipeline.

Trains Ridge, ElasticNet, and Logistic Regression models on the 12-feature
matrix using leave-one-event-out cross-validation.  Produces auditable metrics,
feature importance rankings, ablation comparisons, and event-specific refits.

Bead: cp-bvq
Depends on: cp-1a3 (feature engineering → feature_matrix.csv)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV,
    LogisticRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_CSV = DATA_DIR / "feature_matrix.csv"
INPUT_META = DATA_DIR / "feature_metadata.json"
OUTPUT_RESULTS = DATA_DIR / "model_results.json"
OUTPUT_FOLDS = DATA_DIR / "fold_predictions.json"

FEATURE_COLUMNS = [
    "temp_deviation",
    "cold_severity",
    "heat_severity",
    "fossil_pct_change_lag1",
    "fossil_pct_change",
    "fossil_dominance_ratio",
    "generation_utilization",
    "event_day",
    "is_weekend",
    "is_cold_event",
    "region_fossil_baseline",
    "severity_x_fossil_shift",
]

TARGET = "pm25_aqi_next"
AQI_THRESHOLD = 50  # good vs not_good boundary

EVENT_NAMES = ["elliott_2022", "heat_dome_2021", "uri_2021"]
EVENT_LABELS = {"elliott_2022": "Elliott", "heat_dome_2021": "Heat Dome", "uri_2021": "Uri"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def load_data():
    """Load feature matrix and metadata."""
    df = pd.read_csv(INPUT_CSV)
    with open(INPUT_META) as f:
        meta = json.load(f)
    print(f"Loaded {len(df)} rows, {len(FEATURE_COLUMNS)} features from {INPUT_CSV.name}")
    print(f"Events: {sorted(df['event'].unique())}")
    print(f"Target: {TARGET}  range [{df[TARGET].min():.1f}, {df[TARGET].max():.1f}]  "
          f"mean={df[TARGET].mean():.1f}  std={df[TARGET].std():.1f}")
    return df, meta


def make_event_folds(df):
    """Build leave-one-event-out fold indices.

    Returns list of (event_name, train_idx, test_idx) tuples.
    """
    folds = []
    for event in EVENT_NAMES:
        test_mask = df["event"] == event
        train_idx = df.index[~test_mask].tolist()
        test_idx = df.index[test_mask].tolist()
        folds.append((event, train_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# Step 1 — Standardization demo (before/after ranges, informational only)
# ---------------------------------------------------------------------------

def show_scaling_ranges(df):
    """Print feature ranges before and after standardization (full dataset, demo only)."""
    section("STEP 1 — STANDARDIZATION (informational)")
    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Before scaling — selected feature ranges:")
    for name in ["severity_x_fossil_shift", "fossil_pct_change", "generation_utilization"]:
        idx = FEATURE_COLUMNS.index(name)
        col = X[:, idx]
        print(f"  {name:35s}  [{col.min():8.2f}, {col.max():8.2f}]  (range={col.max()-col.min():.2f})")

    print("\nAfter scaling — same features:")
    for name in ["severity_x_fossil_shift", "fossil_pct_change", "generation_utilization"]:
        idx = FEATURE_COLUMNS.index(name)
        col = X_scaled[:, idx]
        print(f"  {name:35s}  [{col.min():.2f}, {col.max():.2f}]")

    print("\n  NOTE: Scaler is fit INSIDE each CV fold during actual training.")


# ---------------------------------------------------------------------------
# Step 2 — Ridge Regression (in-sample)
# ---------------------------------------------------------------------------

def train_ridge_insample(X, y):
    """Fit RidgeCV on full dataset (in-sample baseline)."""
    section("STEP 2 — RIDGE REGRESSION (in-sample)")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    alphas = np.logspace(-2, 4, 200)
    model = RidgeCV(alphas=alphas, scoring="r2")
    model.fit(X_s, y)

    y_pred = model.predict(X_s)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"Best alpha: {model.alpha_:.2f}")
    print(f"R² = {r2:.3f}   MAE = {mae:.1f} AQI points")

    # Coefficients (standardized)
    coefs = dict(zip(FEATURE_COLUMNS, model.coef_))
    ranked = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\nTop coefficients (standardized):")
    for name, c in ranked[:6]:
        print(f"  {c:+.3f}  {name}")

    return model, scaler, coefs


# ---------------------------------------------------------------------------
# Step 3 — ElasticNet (in-sample)
# ---------------------------------------------------------------------------

def train_elasticnet_insample(X, y):
    """Fit ElasticNetCV on full dataset (in-sample baseline)."""
    section("STEP 3 — ELASTICNET (in-sample)")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=np.logspace(-3, 2, 100),
        cv=3,
        max_iter=10000,
        random_state=42,
    )
    model.fit(X_s, y)

    y_pred = model.predict(X_s)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    n_kept = np.sum(np.abs(model.coef_) > 1e-8)

    print(f"Best alpha: {model.alpha_:.2f}, best l1_ratio: {model.l1_ratio_:.2f}")
    print(f"R² = {r2:.3f}   MAE = {mae:.1f} AQI points")
    print(f"Features kept: {n_kept}/{len(FEATURE_COLUMNS)}")

    coefs = dict(zip(FEATURE_COLUMNS, model.coef_))
    return model, scaler, coefs


# ---------------------------------------------------------------------------
# Step 4 — Leave-One-Event-Out Cross-Validation
# ---------------------------------------------------------------------------

def run_loeo_cv(df, X, y, model_class, model_name, **model_kwargs):
    """Run leave-one-event-out CV for a regression model.

    Fits StandardScaler inside each fold (no leakage).
    Returns per-fold and overall metrics plus fold predictions.
    """
    folds = make_event_folds(df)
    fold_results = []
    all_actual = []
    all_pred = []
    fold_predictions = []

    for event, train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler inside fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit model
        if model_class == RidgeCV:
            model = model_class(alphas=np.logspace(-2, 4, 200), scoring="r2")
        elif model_class == ElasticNetCV:
            model = model_class(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                alphas=np.logspace(-3, 2, 100),
                cv=3,
                max_iter=10000,
                random_state=42,
            )
        else:
            model = model_class(**model_kwargs)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        label = EVENT_LABELS[event]
        fold_results.append({"event": event, "label": label, "r2": r2, "mae": mae})
        all_actual.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        fold_predictions.append({
            "event": event,
            "label": label,
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist(),
        })

        print(f"  Hold out {label:12s}  R² = {r2:+.3f}   MAE = {mae:.1f}")

    overall_r2 = r2_score(all_actual, all_pred)
    overall_mae = mean_absolute_error(all_actual, all_pred)
    print(f"  {'OVERALL':12s}       R² = {overall_r2:.3f}   MAE = {overall_mae:.1f}")

    return {
        "model": model_name,
        "folds": fold_results,
        "overall_r2": overall_r2,
        "overall_mae": overall_mae,
    }, fold_predictions


# ---------------------------------------------------------------------------
# Step 5 — Classification Fallback
# ---------------------------------------------------------------------------

def run_classification(df, X, y):
    """Logistic regression with leave-one-event-out CV for good/not_good."""
    section("STEP 5 — CLASSIFICATION FALLBACK (good vs not_good)")

    y_bin = (y > AQI_THRESHOLD).astype(int)  # 1 = not_good
    n_good = np.sum(y_bin == 0)
    n_bad = np.sum(y_bin == 1)
    print(f"Class balance: good={n_good} ({n_good/len(y_bin)*100:.0f}%), "
          f"not_good={n_bad} ({n_bad/len(y_bin)*100:.0f}%)")
    print(f"Baseline accuracy: {n_good/len(y_bin)*100:.0f}% (always predict 'good')\n")

    folds = make_event_folds(df)
    all_actual = []
    all_pred = []
    fold_results = []

    for event, train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(max_iter=5000, random_state=42, solver="lbfgs")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        label = EVENT_LABELS[event]
        fold_results.append({
            "event": event, "label": label,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        })
        all_actual.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"  Hold out {label:12s}  Acc={acc:.0%}  Prec={prec:.0%}  "
              f"Rec={rec:.0%}  F1={f1:.2f}")

    overall_acc = accuracy_score(all_actual, all_pred)
    overall_prec = precision_score(all_actual, all_pred, zero_division=0)
    overall_rec = recall_score(all_actual, all_pred, zero_division=0)
    overall_f1 = f1_score(all_actual, all_pred, zero_division=0)
    print(f"  {'OVERALL':12s}       Acc={overall_acc:.0%}  Prec={overall_prec:.0%}  "
          f"Rec={overall_rec:.0%}  F1={overall_f1:.2f}")

    return {
        "folds": fold_results,
        "overall_accuracy": overall_acc,
        "overall_precision": overall_prec,
        "overall_recall": overall_rec,
        "overall_f1": overall_f1,
    }


# ---------------------------------------------------------------------------
# Step 6 — Feature Importance
# ---------------------------------------------------------------------------

def print_feature_importance(coefs, meta):
    """Print feature importance table ranked by absolute coefficient value."""
    section("STEP 6 — FEATURE IMPORTANCE (Ridge, standardized)")

    # Build group lookup from metadata
    groups = {}
    for name, info in meta.get("features", {}).items():
        groups[name] = info.get("group", "")

    thesis_map = {
        "fossil_dominance_ratio": "YES (strong)",
        "severity_x_fossil_shift": "YES (strong)",
        "fossil_pct_change": "YES (weak)",
        "fossil_pct_change_lag1": "CONTRARY (moderate)",
    }

    ranked = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"  {'Feature':35s} {'Group':12s} {'Coef':>8s}  Thesis?")
    print(f"  {'-'*35} {'-'*12} {'-'*8}  {'-'*20}")
    for name, c in ranked:
        group = groups.get(name, "")
        thesis = thesis_map.get(name, "")
        print(f"  {name:35s} {group:12s} {c:+8.3f}  {thesis}")

    return [{"feature": n, "group": groups.get(n, ""), "coefficient": c}
            for n, c in ranked]


# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------

def run_ablation(df, X_full, y, feature_names):
    """Rerun Ridge CV dropping (a) is_cold_event, (b) is_cold_event + region_fossil_baseline."""
    section("ABLATION STUDY")

    configs = [
        ("Full (12 features)", feature_names),
        ("Drop is_cold_event", [f for f in feature_names if f != "is_cold_event"]),
        ("Drop is_cold_event + region_fossil_baseline",
         [f for f in feature_names if f not in ("is_cold_event", "region_fossil_baseline")]),
    ]

    results = []
    for label, features in configs:
        col_idx = [feature_names.index(f) for f in features]
        X = X_full[:, col_idx]

        folds = make_event_folds(df)
        all_actual = []
        all_pred = []

        for event, train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = RidgeCV(alphas=np.logspace(-2, 4, 200), scoring="r2")
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            all_actual.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())

        r2 = r2_score(all_actual, all_pred)
        mae = mean_absolute_error(all_actual, all_pred)
        results.append({"config": label, "n_features": len(features), "r2": r2, "mae": mae})

    print(f"  {'Configuration':50s} {'N':>3s}  {'R²':>7s}  {'MAE':>6s}")
    print(f"  {'-'*50} {'-'*3}  {'-'*7}  {'-'*6}")
    for r in results:
        print(f"  {r['config']:50s} {r['n_features']:3d}  {r['r2']:+7.3f}  {r['mae']:6.1f}")

    return results


# ---------------------------------------------------------------------------
# Event-Specific Refit
# ---------------------------------------------------------------------------

def run_event_refit(df, X, y):
    """For each event, 70/30 temporal train/test split with Ridge."""
    section("EVENT-SPECIFIC REFIT (70/30 temporal split)")

    results = []
    for event in EVENT_NAMES:
        mask = df["event"] == event
        X_evt = X[mask.values]
        y_evt = y[mask.values]
        n = len(y_evt)
        split = int(n * 0.7)

        if split < 3 or (n - split) < 2:
            print(f"  {EVENT_LABELS[event]:12s}  SKIP (too few rows)")
            continue

        X_train, X_test = X_evt[:split], X_evt[split:]
        y_train, y_test = y_evt[:split], y_evt[split:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RidgeCV(alphas=np.logspace(-2, 4, 200), scoring="r2")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        label = EVENT_LABELS[event]

        results.append({
            "event": event, "label": label,
            "n_train": split, "n_test": n - split,
            "r2": r2, "mae": mae,
        })
        print(f"  {label:12s}  train={split}  test={n-split}  "
              f"R² = {r2:+.3f}   MAE = {mae:.1f}")

    return results


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(ridge_cv, enet_cv, classification, coef_table,
                 ablation, event_refit, ridge_coefs, enet_coefs):
    """Write model_results.json with all metrics and coefficients."""
    results = {
        "bead": "cp-bvq",
        "dataset": {"rows": 91, "features": 12, "target": TARGET},
        "ridge_cv": ridge_cv,
        "elasticnet_cv": enet_cv,
        "classification": classification,
        "feature_importance": coef_table,
        "ridge_coefficients": {k: round(v, 6) for k, v in ridge_coefs.items()},
        "elasticnet_coefficients": {k: round(v, 6) for k, v in enet_coefs.items()},
        "ablation": ablation,
        "event_refit": event_refit,
    }

    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_RESULTS.name}")


def save_fold_predictions(ridge_folds, enet_folds):
    """Write fold_predictions.json with actual vs predicted arrays."""
    output = {
        "ridge": ridge_folds,
        "elasticnet": enet_folds,
    }
    with open(OUTPUT_FOLDS, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {OUTPUT_FOLDS.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ClimatePulse Model Training (cp-bvq)")
    print("=" * 60)

    # Load data
    df, meta = load_data()
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET].values

    # Step 1 — Scaling demo
    show_scaling_ranges(df)

    # Step 2 — Ridge in-sample
    ridge_model, ridge_scaler, ridge_coefs = train_ridge_insample(X, y)

    # Step 3 — ElasticNet in-sample
    enet_model, enet_scaler, enet_coefs = train_elasticnet_insample(X, y)

    # Step 4 — Leave-one-event-out CV
    section("STEP 4 — LEAVE-ONE-EVENT-OUT CROSS-VALIDATION")

    print("Ridge:")
    ridge_cv, ridge_fold_preds = run_loeo_cv(df, X, y, RidgeCV, "Ridge")

    print("\nElasticNet:")
    enet_cv, enet_fold_preds = run_loeo_cv(df, X, y, ElasticNetCV, "ElasticNet")

    # Step 5 — Classification
    cls_results = run_classification(df, X, y)

    # Step 6 — Feature importance
    coef_table = print_feature_importance(ridge_coefs, meta)

    # Ablation
    ablation_results = run_ablation(df, X, y, FEATURE_COLUMNS)

    # Event-specific refit
    refit_results = run_event_refit(df, X, y)

    # Save outputs
    section("SAVING OUTPUTS")
    save_results(ridge_cv, enet_cv, cls_results, coef_table,
                 ablation_results, refit_results, ridge_coefs, enet_coefs)
    save_fold_predictions(ridge_fold_preds, enet_fold_preds)

    section("DONE")
    print("  Regression cross-validation R² ~ 0.09 — does not generalize across events.")
    print("  Classification accuracy = baseline (76%) — no predictive lift.")
    print("  Coefficients are exploratory only; not confirmatory evidence for thesis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

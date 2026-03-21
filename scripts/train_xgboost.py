#!/usr/bin/env python3
"""XGBoost crop yield model + price impact analysis for ClimatePulse.

Trains a unified XGBoost regressor on the Canadian agriculture feature matrix
(3 provinces × 4 crops × 25 years).  Validates on the 2021 Western Canadian
drought holdout, generates SHAP explainability plots, and runs a secondary
price-impact correlation/regression analysis.

Temporal-leakage controls:
  - 2022 excluded from training (prev_year_* embeds 2021 holdout yields)
  - Purged GroupKFold: when year t is held out, year t+1 is also excluded
    from training (its prev_year_* embeds year t's target values)

Bead: cp-a7w
Depends on: ca_feature_matrix.csv (300 rows × 19 cols)
"""

import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_CSV = DATA_DIR / "ca_feature_matrix.csv"
OUTPUT_RESULTS = DATA_DIR / "ca_model_results.json"
CHARTS_DIR = DATA_DIR / "charts"

TARGET = "yield_kg_ha"

FEATURE_COLS_NUMERIC = [
    "gdd_total",
    "heat_stress_days",
    "precip_total_mm",
    "precip_may_jun_mm",
    "precip_jul_aug_mm",
    "max_consecutive_dry_days",
    "frost_free_days",
    "mean_temp_growing",
    "prev_year_precip_mm",
    "prev_year_gdd",
    "prev_year_yield_kg_ha",
]

CATEGORICAL_COLS = ["province", "crop"]

EXCLUDED = {
    "harvested_ha": "derived from target (leakage)",
    "production_mt": "yield * harvested_ha (leakage)",
    "price_cad_per_tonne": "outcome variable, used in secondary model",
    "price_change_pct": "outcome variable, used in secondary model",
    "year": "temporal info captured by lagged features",
}

RANDOM_STATE = 42
LAG_ORDER = 1  # prev_year_* features look back 1 year

SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "gamma": [0, 0.1, 0.5, 1.0],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def purged_group_kfold(X, y, groups, n_splits, embargo=1):
    """GroupKFold with embargo to prevent lag-based temporal leakage.

    When year t is in the test fold, years t+1..t+embargo are purged from
    training because their prev_year_* features embed year t's target values.
    """
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups):
        test_groups = set(groups[test_idx])
        # Purge: drop years within embargo distance AFTER each test year
        purge_groups = set()
        for tg in test_groups:
            for offset in range(1, embargo + 1):
                purge_groups.add(tg + offset)
        purged_train_idx = np.array([i for i in train_idx
                                     if groups[i] not in purge_groups])
        yield purged_train_idx, test_idx


def file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# STEP 1 — Data Loading & Preparation
# ---------------------------------------------------------------------------

def load_and_prepare():
    section("STEP 1 — DATA LOADING & PREPARATION")

    df = pd.read_csv(INPUT_CSV)
    input_hash = file_sha256(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {INPUT_CSV.name}")
    print(f"  SHA-256 (first 16): {input_hash}")

    # Encode categoricals
    le_province = LabelEncoder()
    le_crop = LabelEncoder()
    df["province_encoded"] = le_province.fit_transform(df["province"])
    df["crop_encoded"] = le_crop.fit_transform(df["crop"])

    province_map = dict(zip(le_province.classes_.tolist(),
                            le_province.transform(le_province.classes_).tolist()))
    crop_map = dict(zip(le_crop.classes_.tolist(),
                        le_crop.transform(le_crop.classes_).tolist()))
    print(f"Province encoding: {province_map}")
    print(f"Crop encoding:     {crop_map}")

    # Drop year 2000 (missing lagged features)
    n_before = len(df)
    df_clean = df[df["year"] != 2000].copy()
    print(f"Dropped year 2000: {n_before - len(df_clean)} rows (missing lagged features)")

    # Split holdout
    holdout = df_clean[df_clean["year"] == 2021].copy()

    # Drop 2022 from training: its prev_year_* features embed 2021 holdout yields
    train_pool = df_clean[(df_clean["year"] != 2021) &
                          (df_clean["year"] != 2022)].copy()
    n_2022 = len(df_clean[df_clean["year"] == 2022])
    print(f"Holdout 2021: {len(holdout)} rows")
    print(f"Dropped 2022: {n_2022} rows (prev_year_* embeds 2021 holdout yields)")
    print(f"Training pool: {len(train_pool)} rows (2001-2020, 2023-2024)")

    # Feature list
    feature_names = FEATURE_COLS_NUMERIC + ["province_encoded", "crop_encoded"]
    print(f"\n{len(feature_names)} features: {feature_names}")

    print(f"\nExcluded columns:")
    for col, reason in EXCLUDED.items():
        print(f"  {col:25s} -- {reason}")

    print(f"\nTarget: {TARGET}")
    print(f"  Training range: [{train_pool[TARGET].min():.0f}, {train_pool[TARGET].max():.0f}] kg/ha")
    print(f"  Training mean:  {train_pool[TARGET].mean():.0f} kg/ha")

    return df_clean, train_pool, holdout, feature_names, province_map, crop_map, input_hash


# ---------------------------------------------------------------------------
# STEP 2 — Hyperparameter Tuning (purged CV)
# ---------------------------------------------------------------------------

def tune_model(train_pool, feature_names):
    section("STEP 2 — HYPERPARAMETER TUNING")

    X_train = train_pool[feature_names].values
    y_train = train_pool[TARGET].values
    groups = train_pool["year"].values

    base_model = XGBRegressor(
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    # Build purged fold indices for RandomizedSearchCV
    purged_folds = list(purged_group_kfold(
        X_train, y_train, groups, n_splits=5, embargo=LAG_ORDER))

    search = RandomizedSearchCV(
        base_model,
        param_distributions=SEARCH_SPACE,
        n_iter=50,
        scoring="neg_mean_absolute_error",
        cv=purged_folds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    print(f"Running RandomizedSearchCV (50 iterations, purged GroupKFold by year, "
          f"embargo={LAG_ORDER})...")
    search.fit(X_train, y_train)

    best = search.best_params_
    print(f"\nBest MAE (CV): {-search.best_score_:.1f} kg/ha")
    print(f"Best params:")
    for k, v in sorted(best.items()):
        print(f"  {k:25s} = {v}")

    return search.best_params_


# ---------------------------------------------------------------------------
# STEP 3 — Cross-Validation Results (purged)
# ---------------------------------------------------------------------------

def run_cv(train_pool, feature_names, best_params):
    section("STEP 3 — CROSS-VALIDATION RESULTS (purged)")

    X = train_pool[feature_names].values
    y = train_pool[TARGET].values
    groups = train_pool["year"].values
    years = train_pool["year"].values

    fold_results = []

    print(f"  Purged GroupKFold: when year t is held out, year t+1 is excluded")
    print(f"  from training (its prev_year_* embeds year t's target).\n")
    print(f"  {'Fold':>4s}  {'Years held out':30s}  {'N_train':>7s}  {'R²':>6s}  {'MAE':>8s}  {'RMSE':>8s}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}")

    all_r2 = []
    all_mae = []
    all_rmse = []

    for i, (train_idx, test_idx) in enumerate(
            purged_group_kfold(X, y, groups, n_splits=5, embargo=LAG_ORDER)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = XGBRegressor(
            **best_params,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        fold_r2 = r2_score(y_te, y_pred)
        fold_mae = mean_absolute_error(y_te, y_pred)
        fold_rmse = rmse(y_te, y_pred)

        held_years = sorted(int(y) for y in set(years[test_idx]))
        held_str = ", ".join(str(y) for y in held_years)

        fold_results.append({
            "fold": i + 1,
            "years_held_out": held_years,
            "n_train": len(train_idx),
            "r2": round(fold_r2, 4),
            "mae": round(fold_mae, 1),
            "rmse": round(fold_rmse, 1),
        })

        all_r2.append(fold_r2)
        all_mae.append(fold_mae)
        all_rmse.append(fold_rmse)

        print(f"  {i+1:4d}  {held_str:30s}  {len(train_idx):7d}  "
              f"{fold_r2:6.3f}  {fold_mae:8.1f}  {fold_rmse:8.1f}")

    overall = {
        "r2_mean": round(float(np.mean(all_r2)), 4),
        "r2_std": round(float(np.std(all_r2)), 4),
        "mae_mean": round(float(np.mean(all_mae)), 1),
        "mae_std": round(float(np.std(all_mae)), 1),
        "rmse_mean": round(float(np.mean(all_rmse)), 1),
        "rmse_std": round(float(np.std(all_rmse)), 1),
    }

    print(f"\n  Overall: R² = {overall['r2_mean']:.3f} +/- {overall['r2_std']:.3f}   "
          f"MAE = {overall['mae_mean']:.1f} +/- {overall['mae_std']:.1f}   "
          f"RMSE = {overall['rmse_mean']:.1f} +/- {overall['rmse_std']:.1f}")

    return {"folds": fold_results, "overall": overall}


# ---------------------------------------------------------------------------
# STEP 4 — 2021 Holdout Validation
# ---------------------------------------------------------------------------

def validate_holdout(train_pool, holdout, feature_names, best_params):
    section("STEP 4 — 2021 HOLDOUT VALIDATION")

    X_train = train_pool[feature_names].values
    y_train = train_pool[TARGET].values
    X_hold = holdout[feature_names].values
    y_hold = holdout[TARGET].values

    print(f"  Training on {len(train_pool)} rows (2022 excluded — lag contamination)")
    print(f"  Predicting {len(holdout)} holdout rows (2021)\n")

    model = XGBRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_hold)

    predictions = []
    print(f"  {'Province':15s} {'Crop':10s} {'Actual':>8s} {'Predicted':>10s} "
          f"{'Error':>8s} {'Error%':>8s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for idx, (_, row) in enumerate(holdout.iterrows()):
        actual = y_hold[idx]
        pred = y_pred[idx]
        err = pred - actual
        err_pct = (err / actual * 100) if actual != 0 else 0

        marker = " <<<" if (row["province"] == "Saskatchewan"
                            and row["crop"] == "Wheat") else ""
        print(f"  {row['province']:15s} {row['crop']:10s} {actual:8.0f} {pred:10.0f} "
              f"{err:+8.0f} {err_pct:+7.1f}%{marker}")

        predictions.append({
            "province": row["province"],
            "crop": row["crop"],
            "actual": round(float(actual), 1),
            "predicted": round(float(pred), 1),
            "error": round(float(err), 1),
            "error_pct": round(float(err_pct), 1),
        })

    hold_r2 = r2_score(y_hold, y_pred)
    hold_mae = mean_absolute_error(y_hold, y_pred)
    hold_rmse = rmse(y_hold, y_pred)

    print(f"\n  Holdout overall: R² = {hold_r2:.3f}   MAE = {hold_mae:.0f}   "
          f"RMSE = {hold_rmse:.0f}")
    print(f"\n  NOTE: Negative R² is expected — 2021 drought is out-of-distribution.")
    print(f"  Overpredictions reflect a normal-year baseline, not a counterfactual estimate.")
    print(f"  Overprediction magnitude traces the drought's geographic gradient.")

    overall = {
        "r2": round(float(hold_r2), 4),
        "mae": round(float(hold_mae), 1),
        "rmse": round(float(hold_rmse), 1),
    }

    return model, y_pred, {"predictions": predictions, "overall": overall}


# ---------------------------------------------------------------------------
# STEP 5 — Feature Importance
# ---------------------------------------------------------------------------

def compute_importance(model, feature_names):
    section("STEP 5 — FEATURE IMPORTANCE")

    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"  {'Rank':>4s}  {'Feature':30s}  {'Gain':>10s}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*10}")
    for i, (name, imp) in enumerate(ranked):
        print(f"  {i+1:4d}  {name:30s}  {imp:10.4f}")

    return [{"feature": n, "importance": round(float(v), 6)} for n, v in ranked]


# ---------------------------------------------------------------------------
# STEP 6 — SHAP Visualizations
# ---------------------------------------------------------------------------

def generate_shap(model, train_pool, holdout, feature_names):
    section("STEP 6 — SHAP VISUALIZATIONS")

    try:
        import shap

        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        # Global SHAP from training data only (no holdout contamination)
        X_train = train_pool[feature_names].copy()
        X_train.columns = feature_names

        explainer = shap.TreeExplainer(model)
        shap_values_train = explainer.shap_values(X_train)

        # 1. Beeswarm (training data only)
        print("  Saving shap_summary_bee.png (training data only) ...")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values_train, X_train, show=False)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "shap_summary_bee.png", dpi=150)
        plt.close("all")

        # 2. Bar chart (training data only)
        print("  Saving shap_summary_bar.png (training data only) ...")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values_train, X_train, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "shap_summary_bar.png", dpi=150)
        plt.close("all")

        # 3. 2021 SK Wheat waterfall (holdout — computed separately)
        print("  Saving shap_2021_drought.png ...")
        X_hold = holdout[feature_names].copy()
        X_hold.columns = feature_names
        shap_values_hold = explainer.shap_values(X_hold)

        sk_wheat_mask = (
            (holdout["province"] == "Saskatchewan") &
            (holdout["crop"] == "Wheat")
        )
        if sk_wheat_mask.any():
            sk_idx = holdout.index.get_loc(holdout[sk_wheat_mask].index[0])
            explanation = shap.Explanation(
                values=shap_values_hold[sk_idx],
                base_values=explainer.expected_value,
                data=X_hold.iloc[sk_idx].values,
                feature_names=feature_names,
            )
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.plots.waterfall(explanation, show=False)
            plt.title("SK Wheat 2021 — Drought Impact")
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / "shap_2021_drought.png", dpi=150)
            plt.close("all")
        else:
            print("    WARNING: SK Wheat 2021 row not found, skipping waterfall")

        # 4. Dependence — heat_stress_days (training data)
        print("  Saving shap_dependence_heat_stress.png ...")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.dependence_plot("heat_stress_days", shap_values_train, X_train,
                             show=False, ax=ax)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "shap_dependence_heat_stress.png", dpi=150)
        plt.close("all")

        # 5. Dependence — precip_total_mm (training data)
        print("  Saving shap_dependence_precip.png ...")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.dependence_plot("precip_total_mm", shap_values_train, X_train,
                             show=False, ax=ax)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "shap_dependence_precip.png", dpi=150)
        plt.close("all")

        # Compute mean |SHAP| from training data only
        mean_abs_shap = np.mean(np.abs(shap_values_train), axis=0)
        shap_ranked = sorted(zip(feature_names, mean_abs_shap),
                             key=lambda x: x[1], reverse=True)
        shap_importance = [{"feature": n, "importance": round(float(v), 2)}
                           for n, v in shap_ranked]

        print(f"\n  All 5 SHAP plots saved to {CHARTS_DIR}/")
        return shap_importance

    except Exception as e:
        print(f"  WARNING: SHAP failed -- {e}")
        print("  Model results will be saved without SHAP importance.")
        return None


# ---------------------------------------------------------------------------
# STEP 7 — Price Impact Model
# ---------------------------------------------------------------------------

def price_impact(train_pool, holdout, holdout_yield_preds):
    section("STEP 7 — PRICE IMPACT MODEL")

    # Compute yield anomalies from training pool stats
    group_stats = train_pool.groupby(["crop", "province"])[TARGET].agg(["mean", "std"])
    group_stats.columns = ["yield_mean", "yield_std"]

    def anomaly_from_yield(yield_val, crop, province):
        key = (crop, province)
        if key in group_stats.index:
            m, s = group_stats.loc[key]
            if s > 0:
                return (yield_val - m) / s * 100
        return np.nan

    # Training anomalies use actual yields
    train_anom = train_pool.copy()
    train_anom["yield_anomaly_pct"] = train_anom.apply(
        lambda r: anomaly_from_yield(r[TARGET], r["crop"], r["province"]), axis=1)

    # Filter to rows with price data
    price_mask = train_anom["price_change_pct"].notna()
    corr_df = train_anom[price_mask].dropna(
        subset=["yield_anomaly_pct", "price_change_pct"])

    n_price_rows = len(corr_df)
    print(f"Training rows with price data: {n_price_rows}")
    print(f"  NOTE: Anomaly baselines computed from training pool (2001-2020,")
    print(f"  2023-2024, excl. 2022). This includes post-2021 data in")
    print(f"  normalization, which is a mild temporal contamination.")

    # Overall correlations
    pearson_r, pearson_p = stats.pearsonr(corr_df["yield_anomaly_pct"],
                                          corr_df["price_change_pct"])
    spearman_r, spearman_p = stats.spearmanr(corr_df["yield_anomaly_pct"],
                                              corr_df["price_change_pct"])

    print(f"\nOverall correlations (yield_anomaly vs price_change):")
    print(f"  Pearson:  r = {pearson_r:+.3f}  (p = {pearson_p:.4f})")
    print(f"  Spearman: r = {spearman_r:+.3f}  (p = {spearman_p:.4f})")

    # Per-crop breakdown
    print(f"\nPer-crop breakdown:")
    per_crop = {}
    for crop_name in sorted(corr_df["crop"].unique()):
        sub = corr_df[corr_df["crop"] == crop_name]
        if len(sub) >= 5:
            pr, pp = stats.pearsonr(sub["yield_anomaly_pct"],
                                    sub["price_change_pct"])
            sr, sp = stats.spearmanr(sub["yield_anomaly_pct"],
                                     sub["price_change_pct"])
            print(f"  {crop_name:10s}  n={len(sub):3d}  "
                  f"Pearson r={pr:+.3f} (p={pp:.3f})  "
                  f"Spearman r={sr:+.3f} (p={sp:.3f})")
            per_crop[crop_name] = {
                "n": int(len(sub)),
                "pearson": {"r": round(float(pr), 4), "p": round(float(pp), 4)},
                "spearman": {"r": round(float(sr), 4), "p": round(float(sp), 4)},
            }

    # OLS regression
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        corr_df["yield_anomaly_pct"], corr_df["price_change_pct"])
    print(f"\nOLS: price_change = {slope:+.3f} * yield_anomaly + {intercept:+.3f}")
    print(f"  R² = {r_val**2:.3f}   p = {p_val:.4f}   SE = {std_err:.3f}")

    # 2021 holdout — use MODEL-PREDICTED yields (end-to-end chain, no target leakage)
    print(f"\n2021 holdout — price predictions from MODEL-PREDICTED yields:")
    print(f"  (Full chain: weather -> predicted yield -> yield anomaly -> price)")
    hold_anom = holdout.copy()
    hold_anom["predicted_yield"] = holdout_yield_preds
    hold_anom["yield_anomaly_pct"] = hold_anom.apply(
        lambda r: anomaly_from_yield(r["predicted_yield"], r["crop"],
                                     r["province"]), axis=1)
    hold_anom["predicted_price_change"] = (slope * hold_anom["yield_anomaly_pct"]
                                           + intercept)

    holdout_predictions = []
    print(f"  {'Province':15s} {'Crop':10s} {'Pred Yield':>10s} {'Yield Anom%':>12s} "
          f"{'Pred Price%':>12s} {'Actual Price%':>14s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*14}")

    for _, row in hold_anom.iterrows():
        actual_price = row["price_change_pct"]
        actual_str = (f"{actual_price:+.1f}" if pd.notna(actual_price)
                      else "N/A")
        anom_str = (f"{row['yield_anomaly_pct']:+.1f}"
                    if pd.notna(row["yield_anomaly_pct"]) else "N/A")
        pred_str = (f"{row['predicted_price_change']:+.1f}"
                    if pd.notna(row["predicted_price_change"]) else "N/A")
        pred_yield_str = f"{row['predicted_yield']:.0f}"

        print(f"  {row['province']:15s} {row['crop']:10s} {pred_yield_str:>10s} "
              f"{anom_str:>12s} {pred_str:>12s} {actual_str:>14s}")

        holdout_predictions.append({
            "province": row["province"],
            "crop": row["crop"],
            "predicted_yield": round(float(row["predicted_yield"]), 1),
            "yield_anomaly_pct": (round(float(row["yield_anomaly_pct"]), 1)
                                  if pd.notna(row["yield_anomaly_pct"])
                                  else None),
            "predicted_price_change": (round(float(row["predicted_price_change"]), 1)
                                       if pd.notna(row["predicted_price_change"])
                                       else None),
            "actual_price_change": (round(float(actual_price), 1)
                                    if pd.notna(actual_price) else None),
        })

    print(f"\n  NOTE: The model overpredicts 2021 yields (doesn't see the drought),")
    print(f"  so yield anomalies are positive and predicted price changes are near")
    print(f"  zero. The full weather->yield->price chain fails to predict 2021")
    print(f"  price spikes because the first link (yield prediction) does not")
    print(f"  capture the drought's out-of-distribution severity.")

    return {
        "n_rows_with_price": int(n_price_rows),
        "pearson": {"r": round(float(pearson_r), 4),
                    "p": round(float(pearson_p), 4)},
        "spearman": {"r": round(float(spearman_r), 4),
                     "p": round(float(spearman_p), 4)},
        "per_crop": per_crop,
        "ols": {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "r2": round(float(r_val ** 2), 4),
            "p": round(float(p_val), 4),
            "std_err": round(float(std_err), 4),
        },
        "holdout_2021": holdout_predictions,
    }


# ---------------------------------------------------------------------------
# STEP 8 — Save Outputs
# ---------------------------------------------------------------------------

def save_outputs(feature_names, best_params, cv_results, holdout_results,
                 xgb_importance, shap_importance, price_results,
                 province_map, crop_map, input_hash, train_rows):
    section("STEP 8 — SAVING OUTPUTS")

    import xgboost
    import sklearn
    try:
        import shap as shap_mod
        shap_version = shap_mod.__version__
    except ImportError:
        shap_version = "not installed"

    results = {
        "run_metadata": {
            "input_file": INPUT_CSV.name,
            "input_sha256_16": input_hash,
            "random_state": RANDOM_STATE,
            "lag_embargo": LAG_ORDER,
            "versions": {
                "xgboost": xgboost.__version__,
                "scikit_learn": sklearn.__version__,
                "shap": shap_version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        "dataset": {
            "total_rows": 300,
            "training_rows": train_rows,
            "holdout_rows": 12,
            "dropped_rows_2000": 12,
            "dropped_rows_2022": 12,
            "drop_2022_reason": "prev_year_* features embed 2021 holdout yields",
            "features": feature_names,
            "excluded": EXCLUDED,
            "target": TARGET,
            "encoding": {
                "province": province_map,
                "crop": crop_map,
            },
        },
        "tuning": {
            "method": "RandomizedSearchCV",
            "n_iter": 50,
            "cv": "purged GroupKFold(5) by year, embargo=1",
            "scoring": "neg_mean_absolute_error",
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                               float(v) if isinstance(v, (np.floating,)) else v)
                           for k, v in best_params.items()},
        },
        "cv_results": cv_results,
        "holdout_2021": holdout_results,
        "feature_importance": {
            "xgboost_gain": xgb_importance,
        },
        "price_impact": price_results,
    }

    if shap_importance is not None:
        results["feature_importance"]["shap_mean_abs"] = shap_importance

    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT_RESULTS.name}")

    # List chart files
    chart_files = sorted(CHARTS_DIR.glob("shap_*.png"))
    if chart_files:
        print(f"Charts ({len(chart_files)}):")
        for p in chart_files:
            print(f"  {p.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ClimatePulse — XGBoost Crop Yield Model (cp-a7w)")
    print("=" * 60)

    # Step 1
    (df_clean, train_pool, holdout, feature_names,
     province_map, crop_map, input_hash) = load_and_prepare()

    # Step 2
    best_params = tune_model(train_pool, feature_names)

    # Step 3
    cv_results = run_cv(train_pool, feature_names, best_params)

    # Step 4
    model, holdout_yield_preds, holdout_results = validate_holdout(
        train_pool, holdout, feature_names, best_params)

    # Step 5
    xgb_importance = compute_importance(model, feature_names)

    # Step 6
    shap_importance = generate_shap(model, train_pool, holdout, feature_names)

    # Step 7 — pass model-predicted yields for holdout (no target leakage)
    price_results = price_impact(train_pool, holdout, holdout_yield_preds)

    # Step 8
    save_outputs(feature_names, best_params, cv_results, holdout_results,
                 xgb_importance, shap_importance, price_results,
                 province_map, crop_map, input_hash, len(train_pool))

    section("DONE")

    return 0


if __name__ == "__main__":
    sys.exit(main())

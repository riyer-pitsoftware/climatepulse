# Zerve Block: train_xgboost_model (RECONCILED — replaces agent version)
# DAG position: after build_feature_matrix (block 07)
# Inputs: df_features from block 07 (300 rows × 19 cols)
# Outputs: model, model_results, df_holdout_results, SHAP plots
#
# RECONCILIATION: Full replacement with ground truth methodology from
#   scripts/train_xgboost.py. Fixes: XGBoost (not HistGBR), purged CV
#   with embargo, 2021-only holdout, 2022 excluded, LabelEncoder, SHAP.

import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TARGET = "yield_kg_ha"

FEATURE_COLS_NUMERIC = [
    "gdd_total", "heat_stress_days", "precip_total_mm", "precip_may_jun_mm",
    "precip_jul_aug_mm", "max_consecutive_dry_days", "frost_free_days",
    "mean_temp_growing", "prev_year_precip_mm", "prev_year_gdd",
    "prev_year_yield_kg_ha",
]
CATEGORICAL_COLS = ["province", "crop"]

# --- Prepare data ---
df = df_features.copy()
le_prov = LabelEncoder().fit(df["province"])
le_crop = LabelEncoder().fit(df["crop"])
df["province_encoded"] = le_prov.transform(df["province"])
df["crop_encoded"] = le_crop.transform(df["crop"])

feature_cols = FEATURE_COLS_NUMERIC + ["province_encoded", "crop_encoded"]
df_model = df.dropna(subset=[TARGET] + FEATURE_COLS_NUMERIC)

# Splits: drop 2000 (no lag), holdout 2021, drop 2022 (lag contamination)
holdout = df_model[df_model["year"] == 2021]
train_pool = df_model[
    (df_model["year"] != 2021) &
    (df_model["year"] != 2022) &
    (df_model["year"] != 2000)
]

X_train = train_pool[feature_cols].values
y_train = train_pool[TARGET].values
X_hold = holdout[feature_cols].values
y_hold = holdout[TARGET].values
groups = train_pool["year"].values

print(f"Training: {len(train_pool)} rows, Holdout: {len(holdout)} rows")
print(f"Training years: {sorted(train_pool['year'].unique())}")
print(f"Excluded: 2000 (no lag), 2021 (holdout), 2022 (lag contamination)")

# --- Purged GroupKFold with embargo ---
class PurgedGroupKFold:
    def __init__(self, n_splits=5, embargo=1):
        self.n_splits = n_splits
        self.embargo = embargo
    def split(self, X, y, groups):
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, test_idx in gkf.split(X, y, groups):
            test_years = set(groups[test_idx])
            embargo_years = set()
            for ty in test_years:
                for e in range(1, self.embargo + 1):
                    embargo_years.add(ty + e)
            purged_train = [i for i in train_idx if groups[i] not in embargo_years]
            yield np.array(purged_train), test_idx
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# --- Hyperparameter search ---
search_space = {
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

cv = PurgedGroupKFold(n_splits=5, embargo=1)
search = RandomizedSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, tree_method="hist"),
    search_space, n_iter=50, cv=cv.split(X_train, y_train, groups),
    scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1,
)
search.fit(X_train, y_train)
best_params = search.best_params_
print(f"\nBest MAE: {-search.best_score_:.0f} kg/ha")
print(f"Best params: {best_params}")

# --- Retrain on full training pool ---
model = XGBRegressor(**best_params, random_state=RANDOM_STATE, tree_method="hist")
model.fit(X_train, y_train)

# --- Holdout evaluation ---
y_pred = model.predict(X_hold)
hold_r2 = r2_score(y_hold, y_pred)
hold_mae = mean_absolute_error(y_hold, y_pred)
hold_rmse = float(np.sqrt(((y_hold - y_pred) ** 2).mean()))

print(f"\nHoldout R²: {hold_r2:.3f}  MAE: {hold_mae:.0f}  RMSE: {hold_rmse:.0f}")
print("NOTE: Negative R² expected — 2021 drought is out-of-distribution.")
print("Overpredictions reflect a normal-year baseline, not a counterfactual estimate.")

df_holdout_results = holdout[["year", "province", "crop", TARGET]].copy()
df_holdout_results["predicted"] = y_pred
df_holdout_results["error"] = y_pred - holdout[TARGET].values
print(f"\n{df_holdout_results.to_string(index=False)}")

# --- Price impact ---
corr_df = train_pool.dropna(subset=["price_change_pct"]).copy()
group_stats = corr_df.groupby(["province", "crop"])[TARGET].agg(["mean", "std"])
corr_df = corr_df.join(group_stats, on=["province", "crop"])
corr_df["yield_anomaly_pct"] = (corr_df[TARGET] - corr_df["mean"]) / corr_df["std"] * 100
pearson_r, pearson_p = stats.pearsonr(corr_df["yield_anomaly_pct"], corr_df["price_change_pct"])
slope, intercept, _, _, _ = stats.linregress(corr_df["yield_anomaly_pct"], corr_df["price_change_pct"])
print(f"\nPrice impact: Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), OLS R²={pearson_r**2:.3f}")

model_results = {
    "cv_best_mae": round(-search.best_score_, 1),
    "holdout_r2": round(hold_r2, 4),
    "holdout_mae": round(hold_mae, 1),
    "holdout_rmse": round(hold_rmse, 1),
    "price_pearson_r": round(pearson_r, 4),
    "price_ols_r2": round(pearson_r**2, 4),
    "best_params": best_params,
    "feature_cols": feature_cols,
}

# --- SHAP ---
try:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
    plt.title("SHAP Feature Importance — Training Data")
    plt.tight_layout()
    plt.show()

    # SK Wheat 2021 waterfall
    sk_wheat = holdout[(holdout["province"] == "Saskatchewan") & (holdout["crop"] == "Wheat")]
    if len(sk_wheat):
        idx = list(holdout.index).index(sk_wheat.index[0])
        hold_shap = explainer.shap_values(X_hold)
        shap.waterfall_plot(shap.Explanation(
            values=hold_shap[idx], base_values=explainer.expected_value,
            data=X_hold[idx], feature_names=feature_cols
        ), show=False)
        plt.title("SHAP Waterfall — SK Wheat 2021 (Drought)")
        plt.tight_layout()
        plt.show()

    print("✓ SHAP plots generated")
except Exception as e:
    print(f"⚠ SHAP failed (non-fatal): {e}")

print(f"\n✅ Model training complete.")
print(f"   CV R² ~ 0.68 expected | Holdout R² ~ -2.3 expected")
print(f"   Actual: holdout R² = {hold_r2:.3f}")

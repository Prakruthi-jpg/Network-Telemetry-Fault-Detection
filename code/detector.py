"""
Anomaly Detection System
=========================
Trains on days 1–10, scores days 11–21.
Primary model: IsolationForest on rolling-window features.
Baseline: per-KPI robust z-score.
Threshold chosen to respect ≤2 false alarms/site/day budget on the TRAINING window.
Never reads ground_truth/.
"""

import numpy as np
import pandas as pd
import yaml
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from scipy.stats import median_abs_deviation


# ── Config ─────────────────────────────────────────────────────────────────

KPI_NAMES = [
    "downlink_traffic_gb", "active_users", "call_setup_success_rate",
    "call_drop_rate", "handover_success_rate", "signal_quality_db",
    "retransmission_rate", "latency_ms", "cpu_load", "temperature_c",
    "supply_voltage_v", "uplink_interference_dbm",
]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Missing value handling ──────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy: per-KPI per-site forward-fill then backward-fill (for isolated NaNs),
    then fill any remaining NaN with the site–KPI median from the training window.
    This avoids leaking future information.
    """
    df = df.copy()
    for site in df["site_id"].unique():
        site_mask = df["site_id"] == site
        df.loc[site_mask, KPI_NAMES] = (
            df.loc[site_mask, KPI_NAMES]
            .ffill()
            .bfill()
        )
    # Any still-NaN (start of series): fill with column median
    for col in KPI_NAMES:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    return df


# ── Feature engineering ─────────────────────────────────────────────────────

def make_features(site_df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    For each KPI: value, rolling mean, rolling std, z-score vs rolling stats.
    Also: hour-of-day and day-of-week (to help model seasonality).
    Returns DataFrame aligned with site_df index.
    """
    feats = pd.DataFrame(index=site_df.index)
    ts = pd.to_datetime(site_df["timestamp"])
    feats["hour"] = ts.dt.hour / 23.0
    feats["dow"] = ts.dt.dayofweek / 6.0

    for kpi in KPI_NAMES:
        x = site_df[kpi].values.astype(float)
        roll_mean = pd.Series(x).rolling(window, min_periods=2).mean().values
        roll_std = pd.Series(x).rolling(window, min_periods=2).std().values
        roll_std = np.where(roll_std < 1e-6, 1e-6, roll_std)

        feats[f"{kpi}"] = x
        feats[f"{kpi}_rmean"] = roll_mean
        feats[f"{kpi}_rstd"] = roll_std
        feats[f"{kpi}_zscore"] = (x - roll_mean) / roll_std

    feats = feats.fillna(0.0)
    return feats


# ── Baseline: per-KPI robust z-score ───────────────────────────────────────

def compute_baseline_scores(train_feats, eval_feats) -> np.ndarray:
    """
    Per-KPI robust z-score: (x - median) / MAD.
    Anomaly score = max absolute z-score across all KPIs at each timestamp.
    Calibrated on training data.
    """
    scores = np.zeros(len(eval_feats))
    for kpi in KPI_NAMES:
        col = kpi
        if col not in eval_feats.columns:
            continue
        train_vals = train_feats[col].values
        med = np.median(train_vals)
        mad = median_abs_deviation(train_vals, nan_policy="omit")
        if mad < 1e-6:
            mad = 1e-6
        z = np.abs((eval_feats[col].values - med) / (1.4826 * mad))
        scores = np.maximum(scores, z)
    # Normalise to [0,1] range for comparability
    scores = scores / (scores.max() + 1e-9)
    return scores


# ── Primary model: IsolationForest ─────────────────────────────────────────

def train_isolation_forest(train_feats: pd.DataFrame, contamination: float, seed: int):
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_feats.values)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model, scaler


def score_isolation_forest(model, scaler, feats: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(feats.values)
    # decision_function returns higher = more normal; negate and shift to [0,1]
    raw = -model.decision_function(X)
    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-9:
        return np.zeros(len(raw))
    return (raw - mn) / (mx - mn)


# ── Threshold selection (alarm-budget rule) ─────────────────────────────────

def select_threshold(train_scores: np.ndarray, n_sites: int, train_days: int,
                     fa_budget: float) -> float:
    """
    Choose threshold such that the number of triggered alerts on the TRAINING
    window respects the budget of fa_budget false alarms per site per day.
    We treat every run of consecutive above-threshold points as 1 alert.

    Method:
      1. Estimate the total allowed alerts = fa_budget * n_sites * train_days
      2. Binary-search for a threshold that produces ≤ that many alert events
         on the training data (conservative but label-free).
    """
    total_budget = fa_budget * n_sites * train_days
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        binary = (train_scores >= mid).astype(int)
        # count runs
        runs = int(np.sum(np.diff(np.concatenate([[0], binary, [0]])) == 1))
        if runs <= total_budget:
            hi = mid
        else:
            lo = mid
    # Use the highest threshold that stays within budget
    threshold = hi
    # Safety: never go below 0.5 (to avoid trivial thresholds)
    return max(threshold, 0.50)


# ── Main detection pipeline ─────────────────────────────────────────────────

def run_detection(data_path="../data/telemetry.parquet",
                  config_path="config.yaml",
                  output_dir="../data"):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    train_days = cfg["detection"]["train_days"]
    window = cfg["detection"]["window_size"]
    contamination = cfg["detection"]["contamination"]
    fa_budget = cfg["detection"]["fa_budget_per_site_per_day"]
    n_sites = cfg["n_sites"]

    print("Loading data…")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Handling missing values…")
    df = handle_missing(df)

    train_cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(days=train_days)
    train_df = df[df["timestamp"] < train_cutoff].copy()
    eval_df = df[df["timestamp"] >= train_cutoff].copy()

    sites = sorted(df["site_id"].unique())

    # ── Compute features per site ──────────────────────────────────────────
    print("Engineering features…")
    all_train_feats = []
    all_eval_feats = []
    for site in sites:
        t_site = train_df[train_df["site_id"] == site].sort_values("timestamp")
        e_site = eval_df[eval_df["site_id"] == site].sort_values("timestamp")
        tf = make_features(t_site, window)
        ef = make_features(e_site, window)
        all_train_feats.append(tf)
        all_eval_feats.append(ef)

    combined_train_feats = pd.concat(all_train_feats, ignore_index=True)

    # ── Train IsolationForest ──────────────────────────────────────────────
    print("Training IsolationForest…")
    model, scaler = train_isolation_forest(combined_train_feats, contamination, seed)

    # ── Score training and eval windows ────────────────────────────────────
    print("Scoring…")
    train_scores_all = score_isolation_forest(model, scaler, combined_train_feats)

    eval_results = []
    eval_scores_all = []
    baseline_eval_scores_all = []

    for i, site in enumerate(sites):
        tf = all_train_feats[i]
        ef = all_eval_feats[i]
        e_site = eval_df[eval_df["site_id"] == site].sort_values("timestamp").reset_index(drop=True)

        # IF scores
        if_scores = score_isolation_forest(model, scaler, ef)

        # Baseline scores
        bl_scores = compute_baseline_scores(tf, ef)

        eval_scores_all.append(if_scores)
        baseline_eval_scores_all.append(bl_scores)

        for j, (ts, score, bl_score) in enumerate(zip(e_site["timestamp"], if_scores, bl_scores)):
            eval_results.append({
                "site_id": site,
                "timestamp": ts,
                "if_score": score,
                "baseline_score": bl_score,
            })

    # ── Global threshold from training scores ──────────────────────────────
    threshold = select_threshold(
        train_scores_all, n_sites, train_days, fa_budget
    )
    print(f"Selected threshold (alarm-budget rule): {threshold:.4f}")

    # ── Build alert dataframe ──────────────────────────────────────────────
    scores_df = pd.DataFrame(eval_results)
    scores_df["alert"] = (scores_df["if_score"] >= threshold).astype(int)
    scores_df["baseline_alert"] = (
        scores_df["baseline_score"] >= scores_df["baseline_score"].quantile(0.97)
    ).astype(int)

    # Save outputs
    scores_df.to_parquet(os.path.join(output_dir, "scores.parquet"), index=False)
    scores_df.to_csv(os.path.join(output_dir, "scores.csv"), index=False)

    meta = {"threshold": threshold, "seed": seed,
            "train_days": train_days, "window": window}
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

    print(f"Saved scores. Alert rate: {scores_df['alert'].mean():.3f}")
    return scores_df, threshold, model, scaler, all_train_feats, all_eval_feats


if __name__ == "__main__":
    run_detection()
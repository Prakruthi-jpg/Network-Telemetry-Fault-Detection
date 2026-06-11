"""
Root Cause Attribution
=======================
For each detected alert event:
  1. Ranks top-3 KPIs by their z-score contribution.
  2. Maps the KPI pattern to a predicted fault archetype (F1–F6)
     using rule-based logic built on domain knowledge — NO test labels used.
"""

import numpy as np
import pandas as pd
import yaml
from scipy.stats import median_abs_deviation


KPI_NAMES = [
    "downlink_traffic_gb", "active_users", "call_setup_success_rate",
    "call_drop_rate", "handover_success_rate", "signal_quality_db",
    "retransmission_rate", "latency_ms", "cpu_load", "temperature_c",
    "supply_voltage_v", "uplink_interference_dbm",
]

# KPI groups for archetype rule matching
ARCHETYPE_RULES = {
    # Priority ordering matters: most specific first
    "F4": {  # Sleeping cell — traffic/users collapse, quality stays ok
        "must_high": [],
        "must_low": ["downlink_traffic_gb", "active_users"],
        "quality_ok": True,
    },
    "F3": {  # Power instability — voltage oscillates, temp/cpu erratic
        "must_high": ["supply_voltage_v", "temperature_c"],
        "must_low": [],
        "quality_ok": False,
    },
    "F5": {  # Interference burst — uplink interference high
        "must_high": ["uplink_interference_dbm"],
        "must_low": [],
        "quality_ok": False,
    },
    "F6": {  # Capacity overload — traffic/users high AND quality bad
        "must_high": ["downlink_traffic_gb", "active_users", "cpu_load"],
        "must_low": [],
        "quality_ok": False,
    },
    "F1": {  # RF degradation — signal quality drops, retx/drop rise
        "must_high": ["call_drop_rate", "retransmission_rate"],
        "must_low": ["signal_quality_db"],
        "quality_ok": False,
    },
    "F2": {  # Clock sync — handover degrades, rest near normal
        "must_high": ["handover_success_rate"],  # treated as delta (worsening)
        "must_low": [],
        "quality_ok": True,
    },
}


def compute_kpi_scores(event_window: pd.DataFrame, train_stats: dict) -> dict:
    """
    Compute per-KPI anomaly magnitude as standardised deviation from training stats.
    Returns dict kpi -> score (higher = more anomalous).
    """
    scores = {}
    for kpi in KPI_NAMES:
        if kpi not in event_window.columns:
            scores[kpi] = 0.0
            continue
        vals = event_window[kpi].dropna().values
        if len(vals) == 0:
            scores[kpi] = 0.0
            continue
        mu = train_stats[kpi]["median"]
        mad = train_stats[kpi]["mad"]
        if mad < 1e-9:
            mad = 1e-9
        z = np.abs((vals.mean() - mu) / (1.4826 * mad))
        scores[kpi] = float(z)
    return scores


def get_top3_kpis(kpi_scores: dict) -> list:
    """Return top-3 KPIs ranked by anomaly magnitude."""
    return sorted(kpi_scores, key=kpi_scores.get, reverse=True)[:3]


def predict_archetype(kpi_scores: dict, event_window: pd.DataFrame,
                       train_stats: dict) -> str:
    """
    Rule-based archetype prediction.
    Uses normalised KPI deviations and directional signals.
    """
    # Compute directional z-scores (signed, not absolute)
    signed_z = {}
    for kpi in KPI_NAMES:
        vals = event_window[kpi].dropna().values
        if len(vals) == 0:
            signed_z[kpi] = 0.0
            continue
        mu = train_stats[kpi]["median"]
        mad = train_stats[kpi]["mad"]
        if mad < 1e-9:
            mad = 1e-9
        signed_z[kpi] = float((vals.mean() - mu) / (1.4826 * mad))

    top3 = set(get_top3_kpis(kpi_scores))

    # --- F4: sleeping cell (traffic/users drop sharply) ---
    traffic_drop = signed_z["downlink_traffic_gb"] < -2.0
    users_drop = signed_z["active_users"] < -2.0
    signal_ok = abs(signed_z["signal_quality_db"]) < 1.5
    if traffic_drop and users_drop and signal_ok:
        return "F4"

    # --- F3: power instability (voltage anomalous) ---
    if "supply_voltage_v" in top3 and kpi_scores["supply_voltage_v"] > 1.5:
        return "F3"

    # --- F5: interference burst (uplink interference high) ---
    interf_z = signed_z["uplink_interference_dbm"]
    if interf_z > 2.0 and "uplink_interference_dbm" in top3:
        return "F5"

    # --- F6: capacity overload (traffic high AND quality bad) ---
    traffic_high = signed_z["downlink_traffic_gb"] > 2.0
    cpu_high = signed_z["cpu_load"] > 2.0
    drop_high = signed_z["call_drop_rate"] > 1.5
    if traffic_high and (cpu_high or drop_high):
        return "F6"

    # --- F1: RF degradation (signal quality drops, retx/drop high) ---
    signal_bad = signed_z["signal_quality_db"] < -1.5
    retx_high = signed_z["retransmission_rate"] > 1.5
    drop_high2 = signed_z["call_drop_rate"] > 1.5
    if signal_bad and (retx_high or drop_high2):
        return "F1"

    # --- F2: clock sync (handover degrades, rest near normal) ---
    ho_bad = signed_z["handover_success_rate"] < -1.5
    others_ok = all(abs(signed_z[k]) < 2.0 for k in KPI_NAMES
                    if k != "handover_success_rate")
    if ho_bad:
        return "F2"

    # Fallback: pick the archetype whose top-3 KPIs overlap most with detected top-3
    FAULT_KPIS_MAP = {
        "F1": {"signal_quality_db", "call_drop_rate", "retransmission_rate"},
        "F2": {"handover_success_rate", "latency_ms"},
        "F3": {"supply_voltage_v", "temperature_c", "cpu_load"},
        "F4": {"downlink_traffic_gb", "active_users"},
        "F5": {"uplink_interference_dbm", "signal_quality_db", "call_setup_success_rate"},
        "F6": {"downlink_traffic_gb", "active_users", "cpu_load", "call_drop_rate"},
    }
    best = max(FAULT_KPIS_MAP, key=lambda ft: len(top3 & FAULT_KPIS_MAP[ft]))
    return best


def compute_train_stats(train_df: pd.DataFrame) -> dict:
    stats = {}
    for kpi in KPI_NAMES:
        vals = train_df[kpi].dropna().values
        stats[kpi] = {
            "median": float(np.median(vals)),
            "mad": float(median_abs_deviation(vals, nan_policy="omit")),
        }
    return stats


def extract_alert_events(scores_df: pd.DataFrame, threshold: float) -> list:
    """
    Group consecutive alert timestamps (per site) into events.
    Returns list of dicts with site_id, start_ts, end_ts, timestamps.
    """
    events = []
    for site in sorted(scores_df["site_id"].unique()):
        site_df = scores_df[scores_df["site_id"] == site].sort_values("timestamp").reset_index(drop=True)
        in_event = False
        start_ts = None
        event_ts = []
        for _, row in site_df.iterrows():
            if row["alert"] == 1:
                if not in_event:
                    in_event = True
                    start_ts = row["timestamp"]
                    event_ts = [row["timestamp"]]
                else:
                    event_ts.append(row["timestamp"])
            else:
                if in_event:
                    events.append({
                        "site_id": site,
                        "start_ts": start_ts,
                        "end_ts": event_ts[-1],
                        "timestamps": event_ts,
                    })
                    in_event = False
                    event_ts = []
        if in_event and event_ts:
            events.append({
                "site_id": site,
                "start_ts": start_ts,
                "end_ts": event_ts[-1],
                "timestamps": event_ts,
            })
    return events


def run_attribution(scores_df: pd.DataFrame, telemetry_df: pd.DataFrame,
                    threshold: float, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    For each detected alert event: compute top-3 KPIs and predicted archetype.
    Returns attribution DataFrame.
    """
    cfg = yaml.safe_load(open(config_path))
    train_cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(days=cfg["detection"]["train_days"])
    train_df = telemetry_df[telemetry_df["timestamp"] < train_cutoff].copy()

    # Fill missing in training data for stats
    for kpi in KPI_NAMES:
        train_df[kpi] = train_df[kpi].fillna(train_df[kpi].median())

    train_stats = compute_train_stats(train_df)
    events = extract_alert_events(scores_df, threshold)

    results = []
    for ev in events:
        site = ev["site_id"]
        ts_list = ev["timestamps"]
        # Extract KPI values during alert window from telemetry
        site_tele = telemetry_df[telemetry_df["site_id"] == site].set_index("timestamp")
        event_window = site_tele.reindex(ts_list)[KPI_NAMES].ffill()

        kpi_scores = compute_kpi_scores(event_window, train_stats)
        top3 = get_top3_kpis(kpi_scores)
        archetype = predict_archetype(kpi_scores, event_window, train_stats)

        results.append({
            "site_id": site,
            "event_start": ev["start_ts"],
            "event_end": ev["end_ts"],
            "n_alerts": len(ts_list),
            "top1_kpi": top3[0] if len(top3) > 0 else "",
            "top2_kpi": top3[1] if len(top3) > 1 else "",
            "top3_kpi": top3[2] if len(top3) > 2 else "",
            "predicted_archetype": archetype,
            "top_kpi_scores": str({k: round(kpi_scores[k], 2) for k in top3}),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import json
    scores_df = pd.read_parquet("../data/scores.parquet")
    tele_df = pd.read_parquet("../data/telemetry.parquet")
    tele_df["timestamp"] = pd.to_datetime(tele_df["timestamp"])
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"])
    with open("../data/model_meta.json") as f:
        meta = json.load(f)
    attr_df = run_attribution(scores_df, tele_df, meta["threshold"])
    attr_df.to_csv("../data/attribution.csv", index=False)
    print(attr_df.to_string())
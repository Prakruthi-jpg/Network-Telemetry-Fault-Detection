"""
Evaluation Protocol
====================
This is the ONLY script that reads ground_truth/labels.csv.
Computes:
  - Event-level precision, recall, F1 (with stated overlap rule)
  - Point-level F1 with AND without point-adjust convention
  - Mean detection delay per archetype
  - False alarms per site per day
  - Per-archetype breakdown
  - Attribution hit-rate
  - Baseline vs primary model comparison
"""

import numpy as np
import pandas as pd
import yaml
import os
import json


KPI_NAMES = [
    "downlink_traffic_gb", "active_users", "call_setup_success_rate",
    "call_drop_rate", "handover_success_rate", "signal_quality_db",
    "retransmission_rate", "latency_ms", "cpu_load", "temperature_c",
    "supply_voltage_v", "uplink_interference_dbm",
]

OVERLAP_THRESHOLD = 0.25  # fraction of fault window that must be covered


def load_ground_truth(gt_path: str) -> pd.DataFrame:
    gt = pd.read_csv(gt_path)
    gt["start_ts"] = pd.to_datetime(gt["start_ts"])
    gt["end_ts"] = pd.to_datetime(gt["end_ts"])
    return gt


def load_scores(scores_path: str) -> pd.DataFrame:
    df = pd.read_parquet(scores_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Event-level metrics ────────────────────────────────────────────────────

def compute_event_metrics(gt: pd.DataFrame, scores_df: pd.DataFrame,
                          alert_col: str = "alert",
                          overlap_frac: float = OVERLAP_THRESHOLD):
    """
    Overlap rule:
      A ground-truth fault event [start, end] is "detected" if the fraction
      of its 15-min intervals that have alert=1 is >= overlap_frac.
      A detected run of alerts is a False Positive if it does not overlap with
      any ground-truth fault window.

    Returns: tp_events, fp_events, fn_events, precision, recall, f1
    """
    tp_events, fn_events = 0, 0
    matched_alert_intervals = set()

    per_archetype = {}

    for _, fault in gt.iterrows():
        site = fault["site_id"]
        start, end = fault["start_ts"], fault["end_ts"]
        ftype = fault["fault_type"]

        # Get eval-window alert flags for this site in the fault window
        site_scores = scores_df[
            (scores_df["site_id"] == site) &
            (scores_df["timestamp"] >= start) &
            (scores_df["timestamp"] <= end)
        ]
        if len(site_scores) == 0:
            fn_events += 1
            per_archetype.setdefault(ftype, {"tp": 0, "fn": 0, "fp": 0, "delays": []})
            per_archetype[ftype]["fn"] += 1
            continue

        n_fault_pts = len(site_scores)
        n_alerted = site_scores[alert_col].sum()
        coverage = n_alerted / n_fault_pts if n_fault_pts > 0 else 0.0

        per_archetype.setdefault(ftype, {"tp": 0, "fn": 0, "fp": 0, "delays": []})

        if coverage >= overlap_frac:
            tp_events += 1
            per_archetype[ftype]["tp"] += 1
            # Record detection delay
            first_alert = site_scores[site_scores[alert_col] == 1]["timestamp"].min()
            delay_min = (first_alert - start).total_seconds() / 60.0
            per_archetype[ftype]["delays"].append(max(0.0, delay_min))
            # Track which alert timestamps are matched
            for ts in site_scores[site_scores[alert_col] == 1]["timestamp"]:
                matched_alert_intervals.add((site, ts))
        else:
            fn_events += 1
            per_archetype[ftype]["fn"] += 1

    # False positives: alert runs not matched to any fault
    fp_events = 0
    for site in scores_df["site_id"].unique():
        site_alerts = scores_df[
            (scores_df["site_id"] == site) & (scores_df[alert_col] == 1)
        ].sort_values("timestamp")

        in_run = False
        run_matched = False
        for _, row in site_alerts.iterrows():
            if not in_run:
                in_run = True
                run_matched = (site, row["timestamp"]) in matched_alert_intervals
            else:
                if (site, row["timestamp"]) in matched_alert_intervals:
                    run_matched = True
            # Check if run ended (simplified: count each disconnected run)

        # Proper run counting
        if len(site_alerts) == 0:
            continue
        ts_series = site_alerts["timestamp"].values
        runs = []
        run_start = ts_series[0]
        prev_ts = ts_series[0]
        run_ts = [ts_series[0]]
        for ts in ts_series[1:]:
            gap = (pd.Timestamp(ts) - pd.Timestamp(prev_ts)).total_seconds() / 60.0
            if gap <= 30:  # same run if within 2 intervals
                run_ts.append(ts)
            else:
                runs.append(run_ts)
                run_ts = [ts]
            prev_ts = ts
        runs.append(run_ts)

        # Build GT fault windows for this site
        site_gt = gt[gt["site_id"] == site]
        for run in runs:
            run_set = set(run)
            matched = False
            for _, fault in site_gt.iterrows():
                fault_ts_range = pd.date_range(fault["start_ts"], fault["end_ts"], freq="15min")
                overlap = len(run_set & set(fault_ts_range.values))
                if overlap > 0:
                    matched = True
                    break
            if not matched:
                fp_events += 1
                for ftype in per_archetype:
                    pass  # fp doesn't have an archetype
                # Count towards a global FP pool
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp_events, "fp": fp_events, "fn": fn_events,
        "precision": precision, "recall": recall, "f1": f1,
        "per_archetype": per_archetype,
    }


# ── Point-level F1 ─────────────────────────────────────────────────────────

def compute_point_f1(gt: pd.DataFrame, scores_df: pd.DataFrame,
                     alert_col: str = "alert"):
    """
    Without point-adjust:
      label each timestamp individually as 1 if inside a fault window, else 0.
    With point-adjust:
      if ANY point in a fault window is detected, mark ALL points in that window as detected.
    """
    # Build per-timestamp ground truth
    eval_df = scores_df.copy()
    eval_df["gt_label"] = 0

    for _, fault in gt.iterrows():
        mask = (
            (eval_df["site_id"] == fault["site_id"]) &
            (eval_df["timestamp"] >= fault["start_ts"]) &
            (eval_df["timestamp"] <= fault["end_ts"])
        )
        eval_df.loc[mask, "gt_label"] = 1

    pred = eval_df[alert_col].values
    truth = eval_df["gt_label"].values

    # Without point-adjust
    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_raw = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # With point-adjust: if any alert fires in a fault window, credit all points
    pa_pred = pred.copy()
    for _, fault in gt.iterrows():
        mask = (
            (eval_df["site_id"] == fault["site_id"]) &
            (eval_df["timestamp"] >= fault["start_ts"]) &
            (eval_df["timestamp"] <= fault["end_ts"])
        ).values
        if pred[mask].sum() > 0:
            pa_pred[mask] = 1

    tp_pa = int(np.sum((pa_pred == 1) & (truth == 1)))
    fp_pa = int(np.sum((pa_pred == 1) & (truth == 0)))
    fn_pa = int(np.sum((pa_pred == 0) & (truth == 1)))
    prec_pa = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
    rec_pa = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
    f1_pa = 2 * prec_pa * rec_pa / (prec_pa + rec_pa) if (prec_pa + rec_pa) > 0 else 0.0

    return {
        "raw": {"tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1_raw},
        "point_adjust": {"tp": tp_pa, "fp": fp_pa, "fn": fn_pa,
                         "precision": prec_pa, "recall": rec_pa, "f1": f1_pa},
    }


# ── False alarm rate ────────────────────────────────────────────────────────

def compute_fa_rate(scores_df: pd.DataFrame, gt: pd.DataFrame,
                    alert_col: str = "alert", eval_days: int = 11) -> float:
    """
    Count alert events that do not overlap any ground-truth window.
    FA rate = total FP events / (n_sites * eval_days).
    """
    fa_count = 0
    for site in scores_df["site_id"].unique():
        site_alerts = scores_df[
            (scores_df["site_id"] == site) & (scores_df[alert_col] == 1)
        ].sort_values("timestamp")
        if len(site_alerts) == 0:
            continue

        site_gt = gt[gt["site_id"] == site]
        ts_list = site_alerts["timestamp"].values

        # Identify runs
        runs = []
        if len(ts_list) == 0:
            continue
        run_ts = [ts_list[0]]
        for ts in ts_list[1:]:
            gap = (pd.Timestamp(ts) - pd.Timestamp(run_ts[-1])).total_seconds() / 60.0
            if gap <= 30:
                run_ts.append(ts)
            else:
                runs.append(run_ts)
                run_ts = [ts]
        runs.append(run_ts)

        for run in runs:
            run_set = set(run)
            matched = False
            for _, fault in site_gt.iterrows():
                ft_range = set(pd.date_range(
                    fault["start_ts"], fault["end_ts"], freq="15min"
                ).values)
                if len(run_set & ft_range) > 0:
                    matched = True
                    break
            if not matched:
                fa_count += 1

    n_sites = scores_df["site_id"].nunique()
    return fa_count / (n_sites * eval_days)


# ── Attribution hit-rate ────────────────────────────────────────────────────

def compute_attribution_hitrate(attr_df: pd.DataFrame, gt: pd.DataFrame) -> float:
    """
    For each detected event (row in attr_df), check if any of its top-3 KPIs
    appears in the true affected KPIs for any ground-truth fault it overlaps.
    Hit-rate = fraction of detected events with at least 1 true KPI in top-3.
    """
    if len(attr_df) == 0:
        return 0.0

    hits = 0
    for _, ev in attr_df.iterrows():
        top3 = {ev["top1_kpi"], ev["top2_kpi"], ev["top3_kpi"]} - {""}
        site = ev["site_id"]
        ev_start = pd.Timestamp(ev["event_start"])
        ev_end = pd.Timestamp(ev["event_end"])

        # Find overlapping ground-truth faults
        site_gt = gt[
            (gt["site_id"] == site) &
            (gt["start_ts"] <= ev_end) &
            (gt["end_ts"] >= ev_start)
        ]
        if len(site_gt) == 0:
            continue  # False positive — don't count

        for _, fault in site_gt.iterrows():
            true_kpis = set(fault["affected_kpis"].split(", "))
            if len(top3 & true_kpis) > 0:
                hits += 1
                break

    # Denominator: detected events that overlap a ground-truth fault (TPs only)
    tp_count = 0
    for _, ev in attr_df.iterrows():
        site = ev["site_id"]
        ev_start = pd.Timestamp(ev["event_start"])
        ev_end = pd.Timestamp(ev["event_end"])
        site_gt = gt[
            (gt["site_id"] == site) &
            (gt["start_ts"] <= ev_end) &
            (gt["end_ts"] >= ev_start)
        ]
        if len(site_gt) > 0:
            tp_count += 1

    return hits / tp_count if tp_count > 0 else 0.0


# ── Archetype-level F1 ─────────────────────────────────────────────────────

def compute_archetype_f1(per_arch: dict) -> dict:
    result = {}
    for arch, counts in per_arch.items():
        tp = counts["tp"]
        fn = counts["fn"]
        fp = counts.get("fp", 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        delays = counts.get("delays", [])
        result[arch] = {
            "tp": tp, "fn": fn,
            "recall": round(rec, 3),
            "mean_delay_min": round(np.mean(delays), 1) if delays else None,
        }
    return result


# ── Main evaluation ────────────────────────────────────────────────────────

def run_evaluation(gt_path: str = "../ground_truth/labels.csv",
                   scores_path: str = "../data/scores.parquet",
                   attr_path: str = "../data/attribution.csv",
                   meta_path: str = "../data/model_meta.json",
                   output_dir: str = "../data"):
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    gt = load_ground_truth(gt_path)
    scores_df = load_scores(scores_path)

    with open(meta_path) as f:
        meta = json.load(f)
    threshold = meta["threshold"]

    n_sites = scores_df["site_id"].nunique()
    eval_days = 11

    # ── Event-level metrics (IF model) ─────────────────────────────────────
    event_m = compute_event_metrics(gt, scores_df, "alert")
    print(f"\nOVERLAP RULE: an event is detected if ≥{OVERLAP_THRESHOLD*100:.0f}% of its")
    print("              timestamps fall under an active alert.\n")
    print("Event-level (IsolationForest model):")
    print(f"  TP={event_m['tp']}  FP={event_m['fp']}  FN={event_m['fn']}")
    print(f"  Precision={event_m['precision']:.3f}  Recall={event_m['recall']:.3f}  F1={event_m['f1']:.3f}")

    # ── Event-level metrics (baseline z-score) ─────────────────────────────
    event_bl = compute_event_metrics(gt, scores_df, "baseline_alert")
    print("\nEvent-level (Baseline robust z-score):")
    print(f"  TP={event_bl['tp']}  FP={event_bl['fp']}  FN={event_bl['fn']}")
    print(f"  Precision={event_bl['precision']:.3f}  Recall={event_bl['recall']:.3f}  F1={event_bl['f1']:.3f}")

    # ── Point-level F1 ─────────────────────────────────────────────────────
    pt_m = compute_point_f1(gt, scores_df, "alert")
    pt_bl = compute_point_f1(gt, scores_df, "baseline_alert")
    print(f"\nPoint-level F1 (IF model):")
    print(f"  Without point-adjust: {pt_m['raw']['f1']:.3f}  "
          f"(P={pt_m['raw']['precision']:.3f}, R={pt_m['raw']['recall']:.3f})")
    print(f"  With    point-adjust: {pt_m['point_adjust']['f1']:.3f}  "
          f"(P={pt_m['point_adjust']['precision']:.3f}, R={pt_m['point_adjust']['recall']:.3f})")
    print(f"\nPoint-level F1 (Baseline z-score):")
    print(f"  Without point-adjust: {pt_bl['raw']['f1']:.3f}")
    print(f"  With    point-adjust: {pt_bl['point_adjust']['f1']:.3f}")

    # ── False alarm rate ───────────────────────────────────────────────────
    fa_rate = compute_fa_rate(scores_df, gt, "alert", eval_days)
    fa_rate_bl = compute_fa_rate(scores_df, gt, "baseline_alert", eval_days)
    print(f"\nFalse alarms per site per day:")
    print(f"  IsolationForest: {fa_rate:.3f}  (budget: ≤2.0)")
    print(f"  Baseline z-score: {fa_rate_bl:.3f}")

    # ── Per-archetype breakdown ────────────────────────────────────────────
    arch_f1 = compute_archetype_f1(event_m["per_archetype"])
    print("\nPer-archetype breakdown (IsolationForest):")
    print(f"{'Arch':6} {'TP':>4} {'FN':>4} {'Recall':>8} {'Mean delay (min)':>18}")
    for arch in sorted(arch_f1.keys()):
        d = arch_f1[arch]
        delay_str = f"{d['mean_delay_min']:.1f}" if d['mean_delay_min'] is not None else "N/A"
        print(f"  {arch:4} {d['tp']:>4} {d['fn']:>4} {d['recall']:>8.3f} {delay_str:>18}")

    # ── Attribution hit-rate ───────────────────────────────────────────────
    if os.path.exists(attr_path):
        attr_df = pd.read_csv(attr_path)
        attr_df["event_start"] = pd.to_datetime(attr_df["event_start"])
        attr_df["event_end"] = pd.to_datetime(attr_df["event_end"])
        hitrate = compute_attribution_hitrate(attr_df, gt)
        print(f"\nAttribution hit-rate (≥1 true KPI in top-3): {hitrate:.3f}")

        # Archetype prediction accuracy (for detected TPs)
        correct_arch = 0
        total_matched = 0
        for _, ev in attr_df.iterrows():
            site_gt = gt[
                (gt["site_id"] == ev["site_id"]) &
                (gt["start_ts"] <= pd.Timestamp(ev["event_end"])) &
                (gt["end_ts"] >= pd.Timestamp(ev["event_start"]))
            ]
            if len(site_gt) > 0:
                total_matched += 1
                if ev["predicted_archetype"] == site_gt.iloc[0]["fault_type"]:
                    correct_arch += 1
        arch_acc = correct_arch / total_matched if total_matched > 0 else 0.0
        print(f"Archetype classification accuracy: {arch_acc:.3f} ({correct_arch}/{total_matched})")
    else:
        hitrate = 0.0
        attr_df = pd.DataFrame()

    # ── Compile results dict ───────────────────────────────────────────────
    results = {
        "threshold": threshold,
        "event_metrics_if": event_m,
        "event_metrics_baseline": event_bl,
        "point_f1_if": pt_m,
        "point_f1_baseline": pt_bl,
        "fa_rate_if": fa_rate,
        "fa_rate_baseline": fa_rate_bl,
        "per_archetype": arch_f1,
        "attribution_hitrate": hitrate,
    }

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}/evaluation_results.json")

    return results, gt, scores_df, attr_df


if __name__ == "__main__":
    run_evaluation()
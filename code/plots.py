"""
Plotting Module
================
Generates all required figures:
  1. Generator verification plots (seasonality, heterogeneity, cross-KPI correlation)
  2. Anomaly scores — all sites with ground truth overlay
  3. Detection by archetype (TP/FN bar chart)
  4. Failure case study
  5. Point-adjust F1 comparison
  6. Attribution top-KPI heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import json
import os
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "lines.linewidth": 1.0,
})

ARCHETYPE_COLORS = {
    "F1": "#E07B39", "F2": "#9467bd", "F3": "#d62728",
    "F4": "#2ca02c", "F5": "#1f77b4", "F6": "#8c564b",
}

KPI_NAMES = [
    "downlink_traffic_gb", "active_users", "call_setup_success_rate",
    "call_drop_rate", "handover_success_rate", "signal_quality_db",
    "retransmission_rate", "latency_ms", "cpu_load", "temperature_c",
    "supply_voltage_v", "uplink_interference_dbm",
]


# ── Helper ─────────────────────────────────────────────────────────────────

def shade_faults(ax, gt_site, ftype_filter=None, alpha=0.15):
    """Shade ground-truth fault windows on an axis."""
    for _, fault in gt_site.iterrows():
        if ftype_filter and fault["fault_type"] != ftype_filter:
            continue
        color = ARCHETYPE_COLORS.get(fault["fault_type"], "gray")
        ax.axvspan(fault["start_ts"], fault["end_ts"],
                   color=color, alpha=alpha, zorder=0)


# ── Fig 1: Generator verification ─────────────────────────────────────────

def plot_generator_verification(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Generator verification — data quality checks", fontsize=11)

    # 1. Daily seasonality — aggregate across all sites, first 10 days
    train_df = df[df["timestamp"] < pd.Timestamp("2024-01-11")].copy()
    train_df["hour"] = train_df["timestamp"].dt.hour
    hourly = train_df.groupby("hour")["downlink_traffic_gb"].mean()
    ax = axes[0, 0]
    ax.plot(hourly.index, hourly.values, color="#1f77b4", linewidth=1.5)
    ax.fill_between(hourly.index, 0, hourly.values, alpha=0.2, color="#1f77b4")
    ax.set_title("Daily seasonality (avg downlink traffic, training window)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg traffic (GB)")
    ax.set_xticks(range(0, 24, 3))

    # 2. Weekly pattern
    train_df["dow"] = train_df["timestamp"].dt.dayofweek
    dow_mean = train_df.groupby("dow")["downlink_traffic_gb"].mean()
    ax2 = axes[0, 1]
    colors_dow = ["#1f77b4"] * 5 + ["#ff7f0e"] * 2
    ax2.bar(dow_mean.index, dow_mean.values, color=colors_dow)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax2.set_title("Weekly pattern — weekdays vs weekends")
    ax2.set_ylabel("Avg traffic (GB)")
    blue_p = mpatches.Patch(color="#1f77b4", label="Weekday")
    ora_p = mpatches.Patch(color="#ff7f0e", label="Weekend")
    ax2.legend(handles=[blue_p, ora_p], fontsize=8)

    # 3. Site heterogeneity — box plots per site for downlink traffic
    ax3 = axes[1, 0]
    site_data = [df[df["site_id"] == f"S{i}"]["downlink_traffic_gb"].dropna().values
                 for i in range(10)]
    ax3.boxplot(site_data, labels=[f"S{i}" for i in range(10)],
                patch_artist=True,
                boxprops=dict(facecolor="#AEC6CF", alpha=0.7),
                medianprops=dict(color="red"))
    ax3.set_title("Site heterogeneity — downlink traffic distribution")
    ax3.set_xlabel("Site")
    ax3.set_ylabel("Traffic (GB)")

    # 4. Cross-KPI correlation heatmap (training window, one site)
    ax4 = axes[1, 1]
    site0_train = train_df[train_df["site_id"] == "S0"][KPI_NAMES].dropna()
    corr = site0_train.corr()
    short_names = [k[:10] for k in KPI_NAMES]
    im = ax4.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax4.set_xticks(range(len(short_names)))
    ax4.set_yticks(range(len(short_names)))
    ax4.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6.5)
    ax4.set_yticklabels(short_names, fontsize=6.5)
    ax4.set_title("Cross-KPI correlation (S0, training window)")
    plt.colorbar(im, ax=ax4, fraction=0.04, pad=0.02)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig1_generator_verification.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Fig 2: Anomaly scores all sites ──────────────────────────────────────

def plot_all_sites_scores(scores_df: pd.DataFrame, gt: pd.DataFrame,
                          threshold: float, output_dir: str):
    sites = sorted(scores_df["site_id"].unique())
    n = len(sites)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Anomaly scores (all sites) — red=truth, green=detected", fontsize=10)

    for i, site in enumerate(sites):
        ax = axes[i]
        site_scores = scores_df[scores_df["site_id"] == site].sort_values("timestamp")
        ax.plot(site_scores["timestamp"], site_scores["if_score"],
                color="#1f77b4", linewidth=0.6, alpha=0.9)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

        site_gt = gt[gt["site_id"] == site]
        # Shade ground truth windows (red)
        for _, fault in site_gt.iterrows():
            ax.axvspan(fault["start_ts"], fault["end_ts"],
                       color=ARCHETYPE_COLORS[fault["fault_type"]], alpha=0.2)
        # Shade detected alert windows (green)
        alerts = site_scores[site_scores["alert"] == 1]
        if len(alerts) > 0:
            ts = alerts["timestamp"].values
            if len(ts) > 0:
                runs = []
                run = [ts[0]]
                for t in ts[1:]:
                    if (pd.Timestamp(t) - pd.Timestamp(run[-1])).total_seconds() <= 30 * 60:
                        run.append(t)
                    else:
                        runs.append(run)
                        run = [t]
                runs.append(run)
                for run in runs:
                    ax.axvspan(pd.Timestamp(run[0]), pd.Timestamp(run[-1]),
                               color="green", alpha=0.18)

        ax.set_ylabel(site, fontsize=8, labelpad=2)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, "fig2_overview_all_sites.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Fig 3: Archetype breakdown bar chart ─────────────────────────────────

def plot_archetype_breakdown(eval_results: dict, output_dir: str):
    per_arch = eval_results["per_archetype"]
    archetypes = ["F1", "F2", "F3", "F4", "F5", "F6"]
    tps = [per_arch.get(a, {}).get("tp", 0) for a in archetypes]
    fns = [per_arch.get(a, {}).get("fn", 0) for a in archetypes]

    x = np.arange(len(archetypes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    bars_tp = ax.bar(x - width / 2, tps, width, label="Detected (TP)", color="#2ca02c")
    bars_fn = ax.bar(x + width / 2, fns, width, label="Missed (FN)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(archetypes)
    ax.set_ylabel("Event count")
    ax.set_title("Detection by fault archetype")
    ax.legend()
    ax.yaxis.get_major_locator().set_params(integer=True)

    # Annotate with recall
    for a, tp, fn in zip(archetypes, tps, fns):
        total = tp + fn
        rec = tp / total if total > 0 else 0
        idx = archetypes.index(a)
        ax.text(idx, max(tp, fn) + 0.08, f"R={rec:.0%}", ha="center", fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3_archetype_breakdown.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Fig 4: Failure case study ─────────────────────────────────────────────

def plot_failure_case(df: pd.DataFrame, scores_df: pd.DataFrame,
                      gt: pd.DataFrame, threshold: float, output_dir: str):
    """
    Show a missed fault with the KPI signals and anomaly score.
    Picks the first F3 missed event.
    """
    # Find a missed fault: loop over all F3, find one with 0 alert coverage
    missed = None
    for _, fault in gt[gt["fault_type"] == "F3"].iterrows():
        site = fault["site_id"]
        site_sc = scores_df[
            (scores_df["site_id"] == site) &
            (scores_df["timestamp"] >= fault["start_ts"]) &
            (scores_df["timestamp"] <= fault["end_ts"])
        ]
        if len(site_sc) > 0 and site_sc["alert"].sum() == 0:
            missed = fault
            break

    if missed is None:
        # Fallback: pick any missed event from hard archetypes first
        for prefer_type in ["F2", "F4", "F1", "F5", "F6"]:
            for _, fault in gt[gt["fault_type"] == prefer_type].iterrows():
                site = fault["site_id"]
                site_sc = scores_df[
                    (scores_df["site_id"] == site) &
                    (scores_df["timestamp"] >= fault["start_ts"]) &
                    (scores_df["timestamp"] <= fault["end_ts"])
                ]
                if len(site_sc) > 0 and site_sc["alert"].sum() == 0:
                    missed = fault
                    break
            if missed is not None:
                break

    if missed is None:
        # Last resort: pick the fault with lowest coverage
        best_cov = 1.0
        for _, fault in gt.iterrows():
            site = fault["site_id"]
            site_sc = scores_df[
                (scores_df["site_id"] == site) &
                (scores_df["timestamp"] >= fault["start_ts"]) &
                (scores_df["timestamp"] <= fault["end_ts"])
            ]
            if len(site_sc) > 0:
                cov = site_sc["alert"].sum() / len(site_sc)
                if cov < best_cov:
                    best_cov = cov
                    missed = fault

    if missed is None:
        print("No missed event found for failure case study — skipping.")
        return

    site = missed["site_id"]
    start = missed["start_ts"]
    end = missed["end_ts"]
    ftype = missed["fault_type"]

    # 12-hour context window around the fault
    ctx_start = start - pd.Timedelta(hours=12)
    ctx_end = end + pd.Timedelta(hours=12)

    site_df = df[
        (df["site_id"] == site) &
        (df["timestamp"] >= ctx_start) &
        (df["timestamp"] <= ctx_end)
    ].sort_values("timestamp")

    site_sc = scores_df[
        (scores_df["site_id"] == site) &
        (scores_df["timestamp"] >= ctx_start) &
        (scores_df["timestamp"] <= ctx_end)
    ].sort_values("timestamp")

    # Show top affected KPIs for this archetype
    display_kpis_map = {
        "F3": ["supply_voltage_v", "cpu_load", "temperature_c", "downlink_traffic_gb"],
        "F2": ["handover_success_rate", "latency_ms", "call_setup_success_rate", "signal_quality_db"],
        "F1": ["signal_quality_db", "call_drop_rate", "retransmission_rate", "downlink_traffic_gb"],
        "F4": ["downlink_traffic_gb", "active_users", "call_setup_success_rate", "cpu_load"],
        "F5": ["uplink_interference_dbm", "signal_quality_db", "call_drop_rate", "retransmission_rate"],
        "F6": ["downlink_traffic_gb", "active_users", "cpu_load", "call_drop_rate"],
    }
    display_kpis = display_kpis_map.get(ftype, KPI_NAMES[:4])

    n_kpis = len(display_kpis)
    fig, axes = plt.subplots(n_kpis + 1, 1, figsize=(12, 2.0 * (n_kpis + 1)),
                             sharex=True)
    fig.suptitle(
        f"Failure case study — {site} {ftype} missed\n"
        f"({start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%H:%M')})",
        fontsize=10
    )

    for i, kpi in enumerate(display_kpis):
        ax = axes[i]
        if kpi in site_df.columns:
            ax.plot(site_df["timestamp"], site_df[kpi],
                    color="#1f77b4", linewidth=0.9)
        ax.axvspan(start, end, color="salmon", alpha=0.3)
        ax.set_ylabel(kpi.replace("_", "\n"), fontsize=7.5, labelpad=2)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    # Score panel
    ax_s = axes[-1]
    ax_s.plot(site_sc["timestamp"], site_sc["if_score"],
              color="#d62728", linewidth=1.0)
    ax_s.axhline(threshold, color="black", linestyle="--", linewidth=0.8,
                 label=f"threshold={threshold:.3f}")
    ax_s.axvspan(start, end, color="salmon", alpha=0.3, label=f"True {ftype}")
    ax_s.set_ylabel("Anomaly\nscore", fontsize=7.5)
    ax_s.set_ylim(0, 1.05)
    ax_s.legend(fontsize=7.5, loc="upper right")
    ax_s.tick_params(axis="x", labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "fig4_failure_case_study.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return missed


# ── Fig 5: Point-adjust comparison ───────────────────────────────────────

def plot_point_adjust_comparison(eval_results: dict, output_dir: str):
    categories = ["Without\npoint-adjust", "With\npoint-adjust"]
    if_f1 = [
        eval_results["point_f1_if"]["raw"]["f1"],
        eval_results["point_f1_if"]["point_adjust"]["f1"],
    ]
    bl_f1 = [
        eval_results["point_f1_baseline"]["raw"]["f1"],
        eval_results["point_f1_baseline"]["point_adjust"]["f1"],
    ]

    x = np.arange(len(categories))
    width = 0.3
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, if_f1, width, label="IsolationForest", color="#1f77b4")
    ax.bar(x + width / 2, bl_f1, width, label="Baseline z-score", color="#ff7f0e")

    for xi, v in zip(x - width / 2, if_f1):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, bl_f1):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Point-level F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Point-level F1 — with vs without point-adjust")
    ax.legend()
    ax.axhline(0.0, color="black", linewidth=0.5)

    inflate = eval_results["point_f1_if"]["point_adjust"]["f1"] - eval_results["point_f1_if"]["raw"]["f1"]
    ax.text(0.5, 0.05,
            f"Point-adjust inflation for IF model: +{inflate:.3f}",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig5_point_adjust_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Fig 6: Attribution heatmap ────────────────────────────────────────────

def plot_attribution_heatmap(attr_df: pd.DataFrame, output_dir: str):
    if len(attr_df) == 0:
        return
    kpi_counts = {}
    for col in ["top1_kpi", "top2_kpi", "top3_kpi"]:
        for kpi in attr_df[col].dropna():
            if kpi:
                kpi_counts[kpi] = kpi_counts.get(kpi, 0) + 1

    if not kpi_counts:
        return

    sorted_kpis = sorted(kpi_counts, key=kpi_counts.get, reverse=True)[:10]
    counts = [kpi_counts[k] for k in sorted_kpis]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(sorted_kpis[::-1], counts[::-1], color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Frequency in top-3 rankings")
    ax.set_title("Most frequently attributed KPIs (top-3 rankings)")
    for bar, count in zip(bars, counts[::-1]):
        ax.text(count + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig6_attribution_kpis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Fig 7: Detection delay by archetype ──────────────────────────────────

def plot_detection_delay(eval_results: dict, output_dir: str):
    per_arch = eval_results["per_archetype"]
    archetypes = sorted(per_arch.keys())
    delays = [per_arch[a].get("mean_delay_min") for a in archetypes]
    colors = [ARCHETYPE_COLORS.get(a, "gray") for a in archetypes]

    # Filter to archetypes with actual detections
    arch_with_detect = [(a, d) for a, d in zip(archetypes, delays) if d is not None]
    if not arch_with_detect:
        return
    arches_plot, delays_plot = zip(*arch_with_detect)
    colors_plot = [ARCHETYPE_COLORS.get(a, "gray") for a in arches_plot]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(arches_plot, delays_plot, color=colors_plot, alpha=0.85)
    ax.set_ylabel("Mean detection delay (minutes)")
    ax.set_title("Mean detection delay by fault archetype")
    for i, (a, d) in enumerate(zip(arches_plot, delays_plot)):
        ax.text(i, d + 1, f"{d:.0f} min", ha="center", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig7_detection_delay.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Master plot runner ────────────────────────────────────────────────────

def run_all_plots(data_dir: str = "../data", plot_dir: str = "../plots",
                  gt_path: str = "../ground_truth/labels.csv"):
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(data_dir, "telemetry.parquet"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    scores_df = pd.read_parquet(os.path.join(data_dir, "scores.parquet"))
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"])

    with open(os.path.join(data_dir, "model_meta.json")) as f:
        meta = json.load(f)
    threshold = meta["threshold"]

    gt = pd.read_csv(gt_path)
    gt["start_ts"] = pd.to_datetime(gt["start_ts"])
    gt["end_ts"] = pd.to_datetime(gt["end_ts"])

    with open(os.path.join(data_dir, "evaluation_results.json")) as f:
        eval_results = json.load(f)

    attr_path = os.path.join(data_dir, "attribution.csv")
    attr_df = pd.read_csv(attr_path) if os.path.exists(attr_path) else pd.DataFrame()

    print("Generating all plots…")
    plot_generator_verification(df, plot_dir)
    plot_all_sites_scores(scores_df, gt, threshold, plot_dir)
    plot_archetype_breakdown(eval_results, plot_dir)
    plot_failure_case(df, scores_df, gt, threshold, plot_dir)
    plot_point_adjust_comparison(eval_results, plot_dir)
    plot_attribution_heatmap(attr_df, plot_dir)
    plot_detection_delay(eval_results, plot_dir)
    print("All plots saved.")


if __name__ == "__main__":
    run_all_plots()
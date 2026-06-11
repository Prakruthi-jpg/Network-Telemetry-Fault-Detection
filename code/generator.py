"""
Synthetic Network Telemetry Generator
======================================
Generates realistic 15-minute KPI telemetry for 10 cell sites over 21 days,
with injected fault events drawn from 6 archetypes (F1–F6).
"""

import numpy as np
import pandas as pd
import yaml
import os
import json
from datetime import datetime, timedelta

# ── KPI names ───────────────────────────────────────────────────────────────
KPI_NAMES = [
    "downlink_traffic_gb",
    "active_users",
    "call_setup_success_rate",
    "call_drop_rate",
    "handover_success_rate",
    "signal_quality_db",
    "retransmission_rate",
    "latency_ms",
    "cpu_load",
    "temperature_c",
    "supply_voltage_v",
    "uplink_interference_dbm",
]

# Affected KPIs per fault archetype (ground-truth attribution)
FAULT_KPIS = {
    "F1": ["signal_quality_db", "call_drop_rate", "retransmission_rate"],
    "F2": ["handover_success_rate"],
    "F3": ["supply_voltage_v", "temperature_c", "cpu_load"],
    "F4": ["downlink_traffic_gb", "active_users"],
    "F5": ["uplink_interference_dbm", "signal_quality_db", "call_setup_success_rate"],
    "F6": ["downlink_traffic_gb", "active_users", "cpu_load", "call_drop_rate"],
}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Normal baseline generators ───────────────────────────────────────────────

def diurnal_pattern(timestamps, profile):
    """Generate hourly load pattern: high in daytime, low at night."""
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    # Double-humped Gaussian: morning peak ~9h, evening peak ~20h
    pattern = (
        0.7 * np.exp(-0.5 * ((hours - 9.0) / 2.5) ** 2) +
        0.9 * np.exp(-0.5 * ((hours - 20.0) / 2.5) ** 2) +
        0.15
    )
    # Weekend dampening
    weekdays = np.array([ts.weekday() for ts in timestamps])
    weekend_mask = weekdays >= 5
    pattern[weekend_mask] *= 0.65
    return pattern * profile["traffic_scale"]


def generate_normal_kpis(timestamps, profile, rng):
    """Generate correlated normal-state KPIs for one site."""
    n = len(timestamps)
    load = diurnal_pattern(timestamps, profile)  # 0..~2.5

    # Heteroscedastic noise: more noise at high load
    noise_std = profile["noise_scale"]

    # --- traffic & users (base load) ---
    traffic = np.maximum(
        load * 10 + rng.normal(0, 0.5 * noise_std, n) * load,
        0.05
    )
    users = np.maximum(
        load * 120 * (profile["user_scale"] / profile["traffic_scale"])
        + rng.normal(0, 4 * noise_std, n) * load,
        1
    ).astype(float)

    # --- KPIs that degrade under load ---
    # Call setup success: degrades ~1% per unit load
    cssr = np.clip(
        98.0 - 0.8 * load + rng.normal(0, 0.3 * noise_std, n),
        85.0, 100.0
    )
    # Drop rate: increases under load + interference
    cdr = np.clip(
        0.5 + 0.3 * load + rng.normal(0, 0.1 * noise_std, n),
        0.0, 8.0
    )
    # Handover success
    hosr = np.clip(
        97.0 - 0.5 * load + rng.normal(0, 0.4 * noise_std, n),
        85.0, 100.0
    )
    # Signal quality (dBm): less noise at low load, more at high
    sig = np.clip(
        -75.0 + 5.0 * (1 - load / 2.5) + rng.normal(0, 1.5 * noise_std, n),
        -110.0, -40.0
    )
    # Retransmission rate
    retr = np.clip(
        2.0 + 1.0 * load + rng.normal(0, 0.2 * noise_std, n),
        0.0, 20.0
    )
    # Latency
    latency = np.clip(
        20.0 + 15.0 * load + rng.normal(0, 2.0 * noise_std, n),
        5.0, 300.0
    )
    # CPU load
    cpu = np.clip(
        30.0 + 25.0 * load + rng.normal(0, 2.0 * noise_std, n),
        5.0, 100.0
    )
    # Temperature (correlated with CPU)
    temp = np.clip(
        35.0 + 0.4 * (cpu - 30) + rng.normal(0, 1.0 * noise_std, n),
        20.0, 80.0
    )
    # Supply voltage (should be stable around 48V)
    voltage = np.clip(
        48.0 + rng.normal(0, 0.3 * noise_std, n),
        42.0, 54.0
    )
    # Uplink interference (worse at busy hours and weekends)
    interf = np.clip(
        -100.0 + 5.0 * load + rng.normal(0, 1.5 * noise_std, n),
        -120.0, -60.0
    )

    df = pd.DataFrame({
        "downlink_traffic_gb": traffic,
        "active_users": users,
        "call_setup_success_rate": cssr,
        "call_drop_rate": cdr,
        "handover_success_rate": hosr,
        "signal_quality_db": sig,
        "retransmission_rate": retr,
        "latency_ms": latency,
        "cpu_load": cpu,
        "temperature_c": temp,
        "supply_voltage_v": voltage,
        "uplink_interference_dbm": interf,
    })
    return df


# ── Fault injection functions ─────────────────────────────────────────────────

def inject_F1(df, mask, ramp, intensity):
    """RF unit degradation: gradual signal decay → higher drop+retx."""
    df.loc[mask, "signal_quality_db"] -= ramp * 18 * intensity
    df.loc[mask, "call_drop_rate"] += ramp * 4.0 * intensity
    df.loc[mask, "retransmission_rate"] += ramp * 8.0 * intensity
    df.loc[mask, "call_setup_success_rate"] -= ramp * 5.0 * intensity
    df.loc[mask, "handover_success_rate"] -= ramp * 3.0 * intensity
    return df


def inject_F2(df, mask, ramp, intensity):
    """Clock sync drift: slow HO degradation, near-normal otherwise."""
    df.loc[mask, "handover_success_rate"] -= ramp * 10.0 * intensity
    df.loc[mask, "latency_ms"] += ramp * 8.0 * intensity
    return df


def inject_F3(df, mask, ramp, rng, intensity, n_osc):
    """Power instability: oscillatory voltage, erratic temp/CPU."""
    t = np.linspace(0, n_osc * 2 * np.pi, mask.sum())
    osc = np.sin(t) * 4.5 * intensity
    df.loc[mask, "supply_voltage_v"] += osc
    df.loc[mask, "cpu_load"] += np.abs(np.sin(t * 1.3)) * 25 * intensity
    df.loc[mask, "temperature_c"] += np.abs(np.sin(t * 0.9)) * 8 * intensity
    # intermittent KPI dips
    dip_idx = np.random.choice(np.where(mask)[0], size=max(1, mask.sum() // 4), replace=False)
    df.loc[dip_idx, "downlink_traffic_gb"] *= 0.4
    df.loc[dip_idx, "active_users"] *= 0.4
    return df


def inject_F4(df, mask, ramp, intensity):
    """Sleeping cell: traffic & users collapse, quality KPIs stay healthy."""
    df.loc[mask, "downlink_traffic_gb"] *= (1 - 0.98 * intensity)
    df.loc[mask, "active_users"] *= (1 - 0.97 * intensity)
    # Quality KPIs stay near normal (deceptive)
    return df


def inject_F5(df, mask, ramp, intensity):
    """Interference burst: uplink jumps, quality and success degrade."""
    df.loc[mask, "uplink_interference_dbm"] += 18.0 * intensity
    df.loc[mask, "signal_quality_db"] -= 12.0 * intensity
    df.loc[mask, "call_setup_success_rate"] -= 8.0 * intensity
    df.loc[mask, "call_drop_rate"] += 3.0 * intensity
    df.loc[mask, "retransmission_rate"] += 5.0 * intensity
    return df


def inject_F6(df, mask, ramp, intensity):
    """Capacity overload: legitimate surge — traffic/users high, KPIs bad."""
    df.loc[mask, "downlink_traffic_gb"] *= (1 + 1.8 * intensity)
    df.loc[mask, "active_users"] *= (1 + 1.5 * intensity)
    df.loc[mask, "cpu_load"] = np.clip(df.loc[mask, "cpu_load"] + 40 * intensity, 0, 100)
    df.loc[mask, "call_drop_rate"] += 2.5 * intensity
    df.loc[mask, "call_setup_success_rate"] -= 6.0 * intensity
    df.loc[mask, "latency_ms"] += 40 * intensity
    df.loc[mask, "retransmission_rate"] += 4.0 * intensity
    return df


def apply_fault(df, timestamps, fault_cfg, rng):
    """Compute mask, ramp, and dispatch to the correct injection function."""
    start_dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=fault_cfg["day"] - 1, hours=fault_cfg["hour"])
    end_dt = start_dt + pd.Timedelta(hours=fault_cfg["duration_hours"])
    mask = (timestamps >= start_dt) & (timestamps < end_dt)

    if mask.sum() == 0:
        return df, mask

    n_pts = mask.sum()
    ftype = fault_cfg["type"]
    intensity = fault_cfg.get("intensity", 1.0)

    # Ramp: gradual onset for F1/F2, step for others
    if ftype in ("F1", "F2"):
        ramp = np.linspace(0.0, 1.0, n_pts)
    else:
        ramp = np.ones(n_pts)

    if ftype == "F1":
        df = inject_F1(df, mask, ramp, intensity)
    elif ftype == "F2":
        df = inject_F2(df, mask, ramp, intensity)
    elif ftype == "F3":
        inject_F3(df, mask, ramp, rng, intensity, n_osc=4)
    elif ftype == "F4":
        df = inject_F4(df, mask, ramp, intensity)
    elif ftype == "F5":
        df = inject_F5(df, mask, ramp, intensity)
    elif ftype == "F6":
        df = inject_F6(df, mask, ramp, intensity)

    return df, mask


# ── Missing value injection ───────────────────────────────────────────────────

def inject_missing(df, rng, rate_low=0.005, rate_high=0.02):
    """Inject benign missing values at 0.5–2% per KPI."""
    df = df.copy()
    for col in KPI_NAMES:
        rate = rng.uniform(rate_low, rate_high)
        missing_idx = rng.choice(len(df), size=int(len(df) * rate), replace=False)
        df.loc[missing_idx, col] = np.nan
    return df


# ── Clip KPIs to physical bounds ─────────────────────────────────────────────

BOUNDS = {
    "downlink_traffic_gb":       (0.0,  500.0),
    "active_users":               (0.0, 5000.0),
    "call_setup_success_rate":   (0.0,  100.0),
    "call_drop_rate":             (0.0,  100.0),
    "handover_success_rate":     (0.0,  100.0),
    "signal_quality_db":        (-120.0, -30.0),
    "retransmission_rate":        (0.0,  100.0),
    "latency_ms":                 (1.0, 5000.0),
    "cpu_load":                   (0.0,  100.0),
    "temperature_c":              (0.0,  100.0),
    "supply_voltage_v":          (30.0,   60.0),
    "uplink_interference_dbm":  (-130.0, -40.0),
}


def clip_kpis(df):
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


# ── Main generation function ──────────────────────────────────────────────────

def generate(config_path="config.yaml", output_dir="../data", gt_dir="../ground_truth"):
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    start_date = pd.Timestamp("2024-01-01")
    end_date = start_date + pd.Timedelta(days=cfg["n_days"])
    timestamps = pd.date_range(start=start_date, end=end_date,
                               freq=f"{cfg['interval_minutes']}min", inclusive="left")
    n_ts = len(timestamps)

    profiles = cfg["site_profiles"]
    # Assign profiles round-robin across sites
    site_profiles = [profiles[i % len(profiles)] for i in range(cfg["n_sites"])]

    all_site_dfs = []
    fault_labels = []

    for site_id in range(cfg["n_sites"]):
        site_rng = np.random.default_rng(cfg["seed"] + site_id * 1000)
        profile = site_profiles[site_id]

        # Generate normal baseline
        df = generate_normal_kpis(timestamps, profile, site_rng)
        df.insert(0, "timestamp", timestamps)
        df.insert(0, "site_id", f"S{site_id}")
        df = df.reset_index(drop=True)

        # Inject faults for this site
        for fault in cfg["faults"]:
            if fault["site"] != site_id:
                continue
            df_kpi = df[KPI_NAMES].copy()
            df_kpi, mask = apply_fault(df_kpi, timestamps, fault, site_rng)
            df[KPI_NAMES] = df_kpi

            if mask.sum() > 0 and fault.get("label") != "train_faint":
                fault_labels.append({
                    "site_id": f"S{site_id}",
                    "fault_type": fault["type"],
                    "start_ts": str(timestamps[mask][0]),
                    "end_ts": str(timestamps[mask][-1]),
                    "affected_kpis": ", ".join(FAULT_KPIS[fault["type"]]),
                    "intensity": fault.get("intensity", 1.0),
                })

        # Clip to physical bounds
        df[KPI_NAMES] = clip_kpis(df[KPI_NAMES])

        # Inject missing values
        df_with_missing = inject_missing(df, site_rng)
        all_site_dfs.append(df_with_missing)

    # Combine all sites
    full_df = pd.concat(all_site_dfs, ignore_index=True)
    full_df.to_parquet(os.path.join(output_dir, "telemetry.parquet"), index=False)
    full_df.to_csv(os.path.join(output_dir, "telemetry.csv"), index=False)
    print(f"Generated {len(full_df)} rows across {cfg['n_sites']} sites.")

    # Save ground truth (never read by detection code)
    gt_df = pd.DataFrame(fault_labels)
    gt_df.to_csv(os.path.join(gt_dir, "labels.csv"), index=False)
    print(f"Saved {len(gt_df)} fault events to ground_truth/labels.csv")

    return full_df, gt_df


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    generate(config_path)
# Network Telemetry Fault Detection & Root Cause Attribution
### Nomiso Inc — AI/ML Engineering Internship — Assignment A

---

## Project Overview

An end-to-end **unsupervised anomaly detection** system for mobile network telemetry.  
Generates synthetic 15-minute KPI data for 10 cell sites over 21 days, detects fault events using IsolationForest, performs root cause attribution (top-3 KPIs + fault archetype), and produces a full evaluation report with 7 figures.

---

## Folder Structure

```
nomiso/
├── README.md                   ← You are here
├── report.pdf                  ← Final 8-page PDF report
│
├── code/
│   ├── config.yaml             ← All hyperparameters, site profiles, fault schedule
│   ├── generator.py            ← Synthetic telemetry + fault injection (F1–F6)
│   ├── detector.py             ← IsolationForest model + z-score baseline
│   ├── attribution.py          ← Top-3 KPI scoring + rule-based archetype classifier
│   ├── evaluate.py             ← All metrics: event F1, point F1, FA rate, hit-rate
│   ├── plots.py                ← Generates all 7 figures
│   ├── report.py               ← Builds the PDF report using ReportLab
│   └── run_all.py              ← Master script — runs ALL steps in order
│
├── data/                       ← Auto-generated (created when you run the project)
│   ├── telemetry.parquet       ← 201,600 rows of KPI telemetry (10 sites × 21 days)
│   ├── telemetry.csv           ← Same data in CSV format
│   ├── scores.parquet          ← Anomaly scores for evaluation window (days 11–21)
│   ├── scores.csv              ← Same scores in CSV format
│   ├── attribution.csv         ← Per-event top-3 KPIs + predicted archetype
│   ├── model_meta.json         ← Threshold, seed, train config
│   └── evaluation_results.json ← All numeric metrics
│
├── ground_truth/               ← ONLY read by evaluate.py (never by detector)
│   └── labels.csv              ← 19 fault events with site, type, timestamps, KPIs
│
└── plots/                      ← Auto-generated figures
    ├── fig1_generator_verification.png   ← Seasonality, heterogeneity, correlation
    ├── fig2_overview_all_sites.png       ← Anomaly scores all 10 sites
    ├── fig3_archetype_breakdown.png      ← TP/FN per fault archetype
    ├── fig4_failure_case_study.png       ← Missed fault deep-dive
    ├── fig5_point_adjust_comparison.png  ← Point F1 with vs without point-adjust
    ├── fig6_attribution_kpis.png         ← Most frequently attributed KPIs
    └── fig7_detection_delay.png          ← Mean detection delay by archetype
```

---

## Quick Start — Run Everything in One Command

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy pyyaml pyarrow scipy matplotlib reportlab
```

### 2. Run the full pipeline
```bash
cd nomiso/code
python run_all.py
```

This single command runs all 5 steps automatically:
1. Generates synthetic telemetry data
2. Trains the detector and scores the evaluation window
3. Runs root cause attribution
4. Evaluates all metrics (reads ground truth **only here**)
5. Generates all 7 plots and builds `report.pdf`

### 3. Run with a different random seed
```bash
python run_all.py --seed 999
```

---

## Running Individual Steps

If you want to run steps one at a time:

```bash
cd nomiso/code

# Step 1 — Generate data
python generator.py config.yaml

# Step 2 — Train detector + score
python detector.py

# Step 3 — Root cause attribution
python attribution.py

# Step 4 — Evaluate (reads ground_truth/labels.csv)
python evaluate.py

# Step 5 — Generate all plots
python plots.py

# Step 6 — Build PDF report
python report.py
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `config.yaml` | Single source of truth for all settings: seed, site profiles, fault injection schedule, detection hyperparameters |
| `generator.py` | Creates 10-site telemetry with diurnal patterns, weekend dampening, cross-KPI correlations, and 6 fault archetypes (F1–F6) injected in the evaluation window |
| `detector.py` | Trains IsolationForest on days 1–10 (no labels). Threshold chosen by binary search to respect ≤2 false alarms/site/day. Also computes per-KPI robust z-score baseline |
| `attribution.py` | For each detected alert event: computes KPI contribution scores (MAD-normalised deviation), ranks top-3 KPIs, predicts fault archetype using rule-based logic |
| `evaluate.py` | **Only script that reads `ground_truth/`**. Computes event-level P/R/F1 (with overlap rule), point-level F1 with and without point-adjust, FA rate, per-archetype breakdown, attribution hit-rate |
| `plots.py` | Generates all 7 required figures |
| `report.py` | Builds the 8-page PDF with all tables, figures, and written analysis |
| `run_all.py` | Orchestrates all steps in order with timing and progress output |

---

## Fault Archetypes

| ID | Name | Affected KPIs | Difficulty |
|----|------|---------------|------------|
| F1 | RF Unit Degradation | signal_quality_db, call_drop_rate, retransmission_rate | Medium |
| F2 | Clock Sync Drift | handover_success_rate | **Hard** — single KPI, slow ramp |
| F3 | Power Instability | supply_voltage_v, temperature_c, cpu_load | Hard — intermittent/oscillatory |
| F4 | Sleeping Cell | downlink_traffic_gb, active_users | **Hard** — indistinguishable from normal low-traffic |
| F5 | Interference Burst | uplink_interference_dbm, signal_quality_db, call_setup_success_rate | Easy — sharp spike |
| F6 | Capacity Overload | downlink_traffic_gb, active_users, cpu_load, call_drop_rate | Easy — large magnitude |

---

## Key Results (seed=42)

| Metric | IsolationForest | Baseline z-score |
|--------|----------------|-----------------|
| Event Precision | 0.121 | 0.112 |
| Event Recall | 0.579 | 0.684 |
| Event F1 | 0.200 | 0.193 |
| Point F1 (raw) | 0.298 | 0.405 |
| Point F1 (point-adjust) | 0.600 | 0.866 |
| False alarms / site / day | **0.727** ✅ | 0.936 |
| Attribution hit-rate | **0.933** | — |

**Per-archetype recall:** F1=100%, F5=100%, F6=100%, F2=33%, F3=0%, F4=0%

---

## Design Decisions

- **No test labels used anywhere except `evaluate.py`** — the threshold is chosen purely from training-window alarm rate
- **Point-adjust inflation explained** — raw F1 (0.298) vs point-adjusted (0.600) differ by +0.302; the report explains why the raw number is the one to trust
- **F2 and F4 are intentionally hard** — single-KPI drift and traffic collapse are within the training distribution, by design
- **Missing values** — 0.5–2% NaN injected per KPI per site, handled by forward-fill then median imputation without leaking future values

---

## Requirements

```
scikit-learn >= 1.2
pandas >= 2.0
numpy >= 1.24
pyyaml >= 6.0
pyarrow >= 12.0
scipy >= 1.10
matplotlib >= 3.7
reportlab >= 4.0
```

---

## Reproducibility

All results are fully reproducible with `seed=42` (set in `config.yaml`).  
Re-run at any time: `cd nomiso/code && python run_all.py`

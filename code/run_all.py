"""
run_all.py — Single-command reproduction script
=================================================
Run:  python run_all.py
      python run_all.py --seed 999      (re-run with new seed)
      python run_all.py --config config.yaml

Steps:
  1. Generate synthetic telemetry + ground truth
  2. Train detector + score evaluation window
  3. Run root-cause attribution
  4. Evaluate (reads ground truth ONLY here)
  5. Generate all plots
  6. Build PDF report
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DATA_DIR    = os.path.join(BASE_DIR, "../data")
GT_DIR      = os.path.join(BASE_DIR, "../ground_truth")
PLOT_DIR    = os.path.join(BASE_DIR, "../plots")
REPORT_DIR  = os.path.join(BASE_DIR, "..")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def step(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed in config")
    parser.add_argument("--config", default=CONFIG_PATH,
                        help="Path to config.yaml")
    args = parser.parse_args()

    config_path = args.config

    # Optionally override seed
    if args.seed is not None:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["seed"] = args.seed
        tmp_config = os.path.join(BASE_DIR, "_tmp_config.yaml")
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)
        config_path = tmp_config
        print(f"Using seed={args.seed}")

    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Step 1: Generate data ─────────────────────────────────────────────
    step("Step 1/5 — Generating synthetic telemetry")
    from generator import generate
    full_df, gt_df = generate(
        config_path=config_path,
        output_dir=DATA_DIR,
        gt_dir=GT_DIR,
    )
    print(f"  Dataset: {len(full_df):,} rows, {full_df['site_id'].nunique()} sites")
    print(f"  Fault events: {len(gt_df)}")

    # ── Step 2: Train detector + score ───────────────────────────────────
    step("Step 2/5 — Training detector and scoring evaluation window")
    from detector import run_detection
    scores_df, threshold, model, scaler, train_feats, eval_feats = run_detection(
        data_path=os.path.join(DATA_DIR, "telemetry.parquet"),
        config_path=config_path,
        output_dir=DATA_DIR,
    )
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Alert rate on eval window: {scores_df['alert'].mean():.3f}")

    # ── Step 3: Root cause attribution ───────────────────────────────────
    step("Step 3/5 — Root cause attribution")
    import pandas as pd
    import json
    from attribution import run_attribution
    tele_df = pd.read_parquet(os.path.join(DATA_DIR, "telemetry.parquet"))
    tele_df["timestamp"] = pd.to_datetime(tele_df["timestamp"])
    scores_df["timestamp"] = pd.to_datetime(scores_df["timestamp"])

    with open(os.path.join(DATA_DIR, "model_meta.json")) as f:
        meta = json.load(f)

    attr_df = run_attribution(scores_df, tele_df, meta["threshold"], config_path)
    attr_df.to_csv(os.path.join(DATA_DIR, "attribution.csv"), index=False)
    print(f"  Attributed {len(attr_df)} detected events")

    # ── Step 4: Evaluation (ONLY place that reads ground_truth/) ─────────
    step("Step 4/5 — Evaluation (reading ground truth)")
    from evaluate import run_evaluation
    eval_results, gt, scores_df_eval, attr_df_eval = run_evaluation(
        gt_path=os.path.join(GT_DIR, "labels.csv"),
        scores_path=os.path.join(DATA_DIR, "scores.parquet"),
        attr_path=os.path.join(DATA_DIR, "attribution.csv"),
        meta_path=os.path.join(DATA_DIR, "model_meta.json"),
        output_dir=DATA_DIR,
    )

    # ── Step 5: Plots ─────────────────────────────────────────────────────
    step("Step 5/5 — Generating plots")
    from plots import run_all_plots
    run_all_plots(
        data_dir=DATA_DIR,
        plot_dir=PLOT_DIR,
        gt_path=os.path.join(GT_DIR, "labels.csv"),
    )

    # ── Build PDF report ──────────────────────────────────────────────────
    step("Building PDF report")
    try:
        from report import build_report
        report_path = build_report(
            eval_results=eval_results,
            threshold=threshold,
            data_dir=DATA_DIR,
            plot_dir=PLOT_DIR,
            output_dir=REPORT_DIR,
            gt_path=os.path.join(GT_DIR, "labels.csv"),
        )
        print(f"  Report saved: {report_path}")
    except Exception as e:
        print(f"  Report generation error: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  Data:      {DATA_DIR}")
    print(f"  Plots:     {PLOT_DIR}")
    print(f"  GT:        {GT_DIR}")
    print(f"  Report:    {REPORT_DIR}/report.pdf")


if __name__ == "__main__":
    main()
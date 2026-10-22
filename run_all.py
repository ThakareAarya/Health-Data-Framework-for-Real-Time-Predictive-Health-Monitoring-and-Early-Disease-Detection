"""Master reproduction script: Wearable-Integrated Preventive Health Analytics System.

Executes the entire research pipeline with one command:
1. Simulates 5 independent longitudinal cohorts (300 subjects x 90 days = 27,000 rows each).
2. Performs leakage-safe preprocessing, subject-wise splits, and causal baseline modeling.
3. Tunes and evaluates all 12 methods (HRI, Tuned HRI, PHRI, NEWS2, LR, SVM, RF, XGBoost on Sets A & B).
4. Conducts early detection analysis on High-reaching episodes.
5. Runs feature ablation (Set A vs Set Z vs Set B) and robustness tests (noise x1/x2/x3, missing 5%/20%/40%).
6. Computes paired t-tests comparing Set B vs Set A across all classifiers.
7. Saves all benchmark tables (CSV and Markdown) and high-resolution figures to results/.
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np

from src.config import COHORT_SEEDS
from src.data_gen import generate_all_cohorts
from src.evaluate import (
    aggregate_cohort_metrics,
    evaluate_single_cohort,
    plot_and_save_confusion_matrices,
    run_paired_t_tests,
    run_robustness_experiments,
)
from src.explain import ModelExplainer


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end Wearable Health Risk Pipeline"
    )
    parser.add_argument(
        "--num_cohorts",
        type=int,
        default=5,
        help="Number of cohorts to evaluate (default: 5)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast mode with reduced cohort size for quick testing",
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print(" WEARABLE-INTEGRATED PREVENTIVE HEALTH ANALYTICS SYSTEM", flush=True)
    print(" Causal Baseline Modeling, Multimodal Risk Engines, & Explainability", flush=True)
    print("=" * 80, flush=True)
    print("NOTICE: All data is synthetic. Not a medical diagnostic tool.\n", flush=True)

    t_start = time.time()
    seeds = COHORT_SEEDS[: args.num_cohorts]

    print(f"Step 1/5: Simulating {len(seeds)} independent cohorts...", flush=True)
    cohorts_dict = generate_all_cohorts(seeds=seeds, output_dir=data_dir)
    for c_id, (df_c, _) in cohorts_dict.items():
        print(f"  - Cohort {c_id}: {df_c.shape[0]} rows ({df_c['subject_id'].nunique()} subjects x {df_c['day'].nunique()} days)", flush=True)

    print("\nStep 2/5: Evaluating models & risk indices on held-out test subjects...", flush=True)
    all_cohort_results = []
    for idx, seed in enumerate(seeds, start=1):
        print(f"  - Processing Cohort {idx}/{len(seeds)} (seed={seed})...", flush=True)
        t0 = time.time()
        c_res = evaluate_single_cohort(cohort_id=idx, seed=seed)
        all_cohort_results.append(c_res)
        print(f"    Completed in {time.time() - t0:.1f}s", flush=True)

    print("\nStep 3/5: Aggregating performance tables across cohorts...", flush=True)
    df_table_vi, df_table_vii, df_table_viii = aggregate_cohort_metrics(all_cohort_results)

    # Save Table VI
    df_table_vi.to_csv(os.path.join(results_dir, "table_vi_performance.csv"), index=False)
    with open(os.path.join(results_dir, "table_vi_performance.md"), "w") as f:
        f.write("# Table VI: Performance on Held-Out Test Subjects (Mean ± SD over 5 Cohorts)\n\n")
        f.write(df_table_vi.to_markdown(index=False))

    # Save Table VII
    df_table_vii.to_csv(os.path.join(results_dir, "table_vii_early_detection.csv"), index=False)
    with open(os.path.join(results_dir, "table_vii_early_detection.md"), "w") as f:
        f.write("# Table VII: Early Detection (First 3 Days of High-Reaching Episodes) and False Alerts\n\n")
        f.write(df_table_vii.to_markdown(index=False))

    # Save Table VIII
    df_table_viii.to_csv(os.path.join(results_dir, "table_viii_ablation.csv"), index=False)
    with open(os.path.join(results_dir, "table_viii_ablation.md"), "w") as f:
        f.write("# Table VIII: Feature-Group Ablation for XGBoost\n\n")
        f.write(df_table_viii.to_markdown(index=False))

    # Step 4: Paired t-tests (Set B vs Set A)
    print("\nStep 4/5: Running paired t-tests comparing Set B vs Set A...")
    df_ttest = run_paired_t_tests(all_cohort_results)
    df_ttest.to_csv(os.path.join(results_dir, "paired_t_test_b_vs_a.csv"), index=False)
    with open(os.path.join(results_dir, "paired_t_test_b_vs_a.md"), "w") as f:
        f.write("# Paired t-tests: Feature Set B vs Feature Set A across 5 Cohorts\n\n")
        f.write(df_ttest.to_markdown(index=False))

    # Step 5: Robustness experiments (Table IX)
    print("\nStep 5/5: Conducting robustness experiments (Noise x1/x2/x3, Missing 5%/20%/40%)...", flush=True)
    df_table_ix = run_robustness_experiments(seeds=COHORT_SEEDS[:3], nominal_results=all_cohort_results[:3])
    df_table_ix.to_csv(os.path.join(results_dir, "table_ix_robustness.csv"), index=False)
    with open(os.path.join(results_dir, "table_ix_robustness.md"), "w") as f:
        f.write("# Table IX: Macro-F1 Under Degraded Input (Mean ± SD over 3 Cohorts)\n\n")
        f.write(df_table_ix.to_markdown(index=False))

    # Save confusion matrices figure
    print("\nGenerating evaluation charts and SHAP global importance plots...")
    rep_res = all_cohort_results[0]
    cm_path = os.path.join(results_dir, "confusion_matrices.png")
    plot_and_save_confusion_matrices(rep_res["y_test"], rep_res["preds_dict"], cm_path)

    # SHAP global importance
    best_xgb_bundle = rep_res["models_B"]["XGB"]
    explainer = ModelExplainer(best_xgb_bundle)
    shap_path = os.path.join(results_dir, "shap_summary.png")
    explainer.plot_global_importance(rep_res["df_test"], target_class=2, top_k=15, save_path=shap_path)

    elapsed = time.time() - t_start
    print(f"\nPipeline completed in {elapsed:.1f} seconds! All tables and figures saved to {results_dir}")

    print("\n" + "=" * 80)
    print(" TABLE VI: PERFORMANCE ON HELD-OUT SUBJECTS (5 COHORTS)")
    print("=" * 80)
    print(df_table_vi.to_string(index=False))

    print("\n" + "=" * 80)
    print(" TABLE VII: EARLY DETECTION & FALSE ALERTS")
    print("=" * 80)
    print(df_table_vii.to_string(index=False))

    print("\n" + "=" * 80)
    print(" TABLE VIII: FEATURE-GROUP ABLATION FOR XGBOOST")
    print("=" * 80)
    print(df_table_viii.to_string(index=False))

    print("\n" + "=" * 80)
    print(" TABLE IX: MACRO-F1 UNDER DEGRADED INPUT")
    print("=" * 80)
    print(df_table_ix.to_string(index=False))

    print("\n" + "=" * 80)
    print(" PAIRED T-TESTS (SET B VS SET A)")
    print("=" * 80)
    print(df_ttest.to_string(index=False))


if __name__ == "__main__":
    main()

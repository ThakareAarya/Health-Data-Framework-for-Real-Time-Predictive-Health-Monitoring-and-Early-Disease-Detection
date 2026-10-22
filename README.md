# Wearable-Integrated Preventive Health Analytics System (`wearable-health-risk`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!CAUTION]
> **CRITICAL MEDICAL DISCLAIMER**: All datasets generated, preprocessed, and analyzed by this system are **SYNTHETIC COMPUTER SIMULATIONS**. This software is strictly an academic research prototype and **DOES NOT CONSTITUTE A MEDICAL DIAGNOSTIC TOOL OR CLINICAL DECISION SUPPORT SYSTEM**. It is not cleared or approved by any medical regulatory authority and must never replace clinical judgment or professional medical care.

---

## 1. System Overview

Modern preventive medicine requires identifying physiological transitions before acute decompensation occurs. While commercial wearables capture rich longitudinal telemetry (heart rate, blood pressure, oxygenation, temperature, respiration, sleep, step activity, and heart rate variability), static population-wide thresholds struggle with between-person physiological variability.

The **Wearable-Integrated Preventive Health Analytics System** provides an end-to-end software framework that:
1. **Models Causal Personal Baselines**: Derives individualized normal operating ranges using strictly backward-looking windows ($W=28$ days, minimum 7 days) without future temporal contamination.
2. **Guarantees Leakage-Safe Evaluation**: Implements subject-disjoint cohort partitioning (60% train, 15% validation, 25% test), fitting all scalers, median imputers, and index thresholds exclusively on training subjects.
3. **Decouples Labels from Observations**: Ground truth health risks (Low, Moderate, High) depend solely on a latent disease severity process $e(t) \in [0, 1]$ generated from overlapping health episodes (infection-like, cardiovascular strain, and sleep/stress loss).
4. **Benchmarks Multi-Model Predictions**: Compares rule-based scores (Population HRI, Tuned HRI, Personalised HRI, NEWS2) against machine learning ensembles (Logistic Regression, RBF-SVM, Random Forest, XGBoost) across raw population features (Set A, 14 dims), baseline deviations (Set Z, 20 dims), and combined personalized features (Set B, 34 dims).
5. **Provides Explainability & Interactive Collaboration**: SHAP tree-based risk factor attributions paired with a Streamlit clinical inspection dashboard.

---

## 2. Directory Structure

```
wearable-health-risk/
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Clinical limits, plausibility bounds, weights, constants
│   ├── data_gen.py         # Stage 1: Longitudinal synthetic cohort simulator (300 subjects x 90 days)
│   ├── preprocess.py       # Stage 2: Plausibility filtering, causal ffill, subject-wise splits
│   ├── baseline.py         # Stage 3: Rolling causal median/IQR, z-scores, 3d-avg z, Sets A/Z/B
│   ├── hri.py              # Stage 4: Composite HRI, Tuned HRI, PHRI, and NEWS2 clinical scores
│   ├── models.py           # Stage 5: LR, RBF-SVM (8k subsample), RF (250 trees), XGBoost + validation tuning
│   ├── evaluate.py         # Stage 6: Multi-cohort evaluation, early detection, ablation, robustness
│   └── explain.py          # Stage 7: SHAP TreeExplainer & local vital factor decomposition
│
├── app/
│   └── dashboard.py        # Stage 8: Interactive Streamlit analytics dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_data_gen.py    # Unit tests for cohort shapes, parameters, and latent label integrity
│   ├── test_preprocess.py  # Unit tests for causal forward-fill and training-set median imputation
│   ├── test_baseline.py    # Unit tests for backward-only baseline windows and feature set dimensions
│   ├── test_leakage.py     # Rigorous verification: disjoint subject sets & backward-only baseline
│   ├── test_hri.py         # Unit tests for HRI, PHRI, and NEWS2 scoring formulas
│   └── test_models.py      # Unit tests for model training and validation hyperparameter tuning
│
├── results/                # Generated benchmark tables (CSV, Markdown) & figures (PNG)
│   ├── table_vi_performance.csv / .md
│   ├── table_vii_early_detection.csv / .md
│   ├── table_viii_ablation.csv / .md
│   ├── table_ix_robustness.csv / .md
│   ├── paired_t_test_b_vs_a.csv / .md
│   ├── confusion_matrices.png
│   └── shap_summary.png
│
├── data/                   # Generated synthetic cohorts (cohort_1.csv .. cohort_5.csv)
├── run_all.py              # Single command reproduction of the full benchmark pipeline
├── pytest.ini              # Pytest configuration ensuring project root is in PYTHONPATH
├── requirements.txt        # Pinned project dependencies
└── README.md               # Complete project documentation
```

---

## 3. Installation & Setup

Ensure Python 3.11+ is installed. Clone the repository and install requirements:

```bash
# Navigate to the workspace directory
cd "wearable-health-risk"

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running Unit Tests

The test suite validates data generation, preprocessing, causal rolling baselines, strict leakage prevention, clinical scoring indices, and machine learning models:

```bash
# Run the complete test suite (22 unit tests)
python -m pytest
```

---

## 5. End-to-End Pipeline Reproduction

Execute the entire benchmark experiment across 5 independent cohorts ($300 \times 90 = 27,000$ person-days each) with a single command:

```bash
python run_all.py
```

This command will:
1. Generate and persist cohorts in `data/` using fixed random seeds (`[42, 101, 202, 303, 404]`).
2. Execute subject-wise train/val/test splitting and causal imputation.
3. Compute rolling baselines and construct Feature Sets A, Z, and B.
4. Tune and evaluate all 12 methods on held-out test subjects.
5. Perform early detection analysis on high-severity health episodes.
6. Conduct feature ablation for XGBoost and robustness stress tests.
7. Perform paired $t$-tests comparing Feature Set B vs Set A.
8. Export all tables (CSV and Markdown) and PNG figures to `results/`.

---

## 6. Launching the Interactive Streamlit Dashboard

Explore longitudinal vitals, personal baseline bands, HRI progression, anomaly flags, and SHAP risk factor attributions in your browser:

```bash
streamlit run app/dashboard.py
```

### Dashboard Capabilities:
- **Persistent Disclaimer**: Visible disclaimer highlighting that the tool is for simulation research only.
- **Cohort & Subject Selector**: Switch seamlessly across any subject in the 5 cohorts.
- **Dynamic Baseline Bands**: View daily readings alongside the subject's causal baseline ($\text{median} \pm 1.96 \cdot \text{scale}$).
- **Anomaly Detection**: Instant red scatter markers on days with $|z| > 2.5$.
- **HRI Trajectories**: Track population HRI and baseline-relative PHRI over 90 days.
- **SHAP Day Inspector**: Select any day to view the exact local risk decomposition.

---

## 7. Experimental Methodology & Rigorous Safeguards

### Leakage Prevention Rules
1. **Split by Person**: Cohorts are partitioned strictly by `subject_id` (60% train, 15% val, 25% test). No individual ever appears in more than one split partition.
2. **Causal Backward-Looking Baselines**: Day $t$'s baseline is computed strictly using preceding days $[t-W, t-1]$. Day $t$ or future days never enter the baseline calculation.
3. **Training-Only Imputation**: Remaining missing values are imputed using medians computed solely from training subjects.
4. **Independent Threshold Calibration**: HRI and NEWS2 cutoffs are tuned solely on training subjects to maximize macro-F1.
5. **Independent Testing**: Test subjects are used only once for reporting final performance.

### Latent Severity Label Formulation
Labels are never computed from observed vitals or from the HRI. An unobserved daily latent severity $e(t) = \min(1.0, \sum_k e_k(t))$ governs health state:
- **Low Risk**: $e(t) < 0.25$ (~71% prevalence)
- **Moderate Risk**: $0.25 \le e(t) < 0.60$ (~14% prevalence)
- **High Risk**: $e(t) \ge 0.60$ (~15% prevalence)
Observed vitals reflect baseline + lifestyle load + episode effects + noise.

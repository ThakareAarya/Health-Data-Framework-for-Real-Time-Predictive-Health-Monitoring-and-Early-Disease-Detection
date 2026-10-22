"""Dedicated verification tests for data leakage prevention:
1. Proves no person appears in more than one split (Train, Val, Test are completely disjoint).
2. Proves that personal baseline calculations strictly use ONLY past days (causal, backward-looking only).
3. Proves preprocessing imputations use ONLY statistics computed from training subjects.
"""

import numpy as np
import pandas as pd
import pytest

from src.baseline import compute_all_baselines, compute_subject_baselines
from src.config import BASELINE_MEASURES, DAILY_VITALS
from src.data_gen import simulate_cohort
from src.preprocess import get_subject_split_ids, preprocess_cohort


def test_leakage_rule_1_strictly_disjoint_splits():
    """PROOFS NO PERSON APPEARS IN MORE THAN ONE SPLIT.
    
    Verifies that the sets of Subject IDs in Train, Validation, and Test
    have pairwise zero intersection and collectively cover all subjects.
    """
    for seed in [42, 101, 202, 303, 404]:
        train_ids, val_ids, test_ids = get_subject_split_ids(300, seed=seed)

        set_train = set(train_ids)
        set_val = set(val_ids)
        set_test = set(test_ids)

        # Pairwise intersections MUST be strictly empty
        assert (
            len(set_train.intersection(set_val)) == 0
        ), f"Leakage detected between Train and Val for seed {seed}"
        assert (
            len(set_train.intersection(set_test)) == 0
        ), f"Leakage detected between Train and Test for seed {seed}"
        assert (
            len(set_val.intersection(set_test)) == 0
        ), f"Leakage detected between Val and Test for seed {seed}"

        # Union must exactly equal all 300 subjects
        assert len(set_train) == 180
        assert len(set_val) == 45
        assert len(set_test) == 75
        assert set_train.union(set_val).union(set_test) == set(range(300))


def test_leakage_rule_2_baselines_use_only_past_data():
    """PROOFS BASELINES USE ONLY PAST DAYS (NO FUTURE DATA LEAKAGE).
    
    If future data (day t+1 onwards) is modified or perturbed,
    the computed baseline for day t MUST remain strictly unchanged.
    """
    n_days = 40
    rng = np.random.default_rng(12345)
    base_hr = rng.normal(70.0, 3.0, size=n_days)

    df_original = pd.DataFrame(
        {
            "subject_id": np.zeros(n_days, dtype=int),
            "day": np.arange(n_days),
            "hr": base_hr.copy(),
            "sbp": np.full(n_days, 120.0),
            "dbp": np.full(n_days, 80.0),
            "spo2": np.full(n_days, 98.0),
            "temp": np.full(n_days, 36.6),
            "rr": np.full(n_days, 16.0),
            "log_hrv": np.full(n_days, 3.8),
            "sleep": np.full(n_days, 7.5),
            "log_steps": np.full(n_days, 9.0),
            "stress": np.full(n_days, 30.0),
        }
    )

    # Compute baseline on original sequence
    res_orig = compute_subject_baselines(df_original, window_size=28, min_days=7)

    # Now create perturbed sequence where days 20 to 39 have catastrophic changes
    df_perturbed = df_original.copy()
    df_perturbed.loc[20:, "hr"] = 200.0  # Huge future spike

    res_pert = compute_subject_baselines(df_perturbed, window_size=28, min_days=7)

    # For all days t <= 20, the baseline median, scale, and z-score for day t
    # MUST BE IDENTICAL between original and perturbed!
    for t in range(7, 20):
        assert res_orig.loc[t, "baseline_med_hr"] == pytest.approx(
            res_pert.loc[t, "baseline_med_hr"]
        ), f"Future data leaked into day {t} baseline median!"
        assert res_orig.loc[t, "baseline_scale_hr"] == pytest.approx(
            res_pert.loc[t, "baseline_scale_hr"]
        ), f"Future data leaked into day {t} baseline scale!"
        assert res_orig.loc[t, "z_hr"] == pytest.approx(
            res_pert.loc[t, "z_hr"]
        ), f"Future data leaked into day {t} z-score!"


def test_leakage_rule_3_imputer_uses_only_training_subjects():
    """PROOFS IMPUTATION MEDIANS ARE COMPUTED STRICTLY FROM TRAINING SUBJECTS.
    
    Perturbing test subjects' values MUST have 0 effect on imputation medians.
    """
    df_raw, _ = simulate_cohort(cohort_id=1, n_subjects=50, n_days=15, seed=42)
    _, artifacts_1 = preprocess_cohort(df_raw, seed=42)

    # Now artificially corrupt test subjects in raw data with extreme values
    df_raw_corrupted = df_raw.copy()
    test_mask = df_raw_corrupted["subject_id"].isin(artifacts_1.test_subject_ids)
    df_raw_corrupted.loc[test_mask, "hr"] = 199.0

    _, artifacts_2 = preprocess_cohort(df_raw_corrupted, seed=42)

    # Imputer training medians MUST NOT change
    for vital in DAILY_VITALS:
        assert artifacts_1.train_medians[vital] == pytest.approx(
            artifacts_2.train_medians[vital]
        ), f"Test set corruption leaked into training median for {vital}!"

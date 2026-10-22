"""Unit tests for Stage 3: Personal Baseline Modeling and Feature Sets.
"""

import numpy as np
import pandas as pd
import pytest

from src.baseline import (
    compute_all_baselines,
    compute_sleep_deficit,
    compute_subject_baselines,
    get_feature_matrices,
)
from src.config import (
    BASELINE_MEASURES,
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_Z,
)
from src.data_gen import simulate_cohort
from src.preprocess import preprocess_cohort


def test_sleep_deficit():
    """Checks sleep deficit calculation."""
    df = pd.DataFrame({"sleep": [8.0, 7.5, 6.0, 4.5]})
    res = compute_sleep_deficit(df)
    assert res["sleep_deficit"].tolist() == [0.0, 0.0, 1.5, 3.0]


def test_causal_baseline_window():
    """Proves that baseline calculation strictly uses only past days [t-W, t)."""
    # Create artificial subject with sudden spike at day 15
    days = 25
    hr_vals = np.full(days, 70.0)
    hr_vals[15] = 120.0  # Massive spike at day 15

    subj_df = pd.DataFrame(
        {
            "subject_id": np.zeros(days, dtype=int),
            "day": np.arange(days),
            "hr": hr_vals,
            "sbp": np.full(days, 120.0),
            "dbp": np.full(days, 80.0),
            "spo2": np.full(days, 98.0),
            "temp": np.full(days, 36.6),
            "rr": np.full(days, 16.0),
            "log_hrv": np.full(days, 3.8),
            "sleep": np.full(days, 7.5),
            "log_steps": np.full(days, 9.0),
            "stress": np.full(days, 30.0),
        }
    )

    res = compute_subject_baselines(subj_df, window_size=28, min_days=7)

    # Days 0-6 should be NaN (min_days = 7)
    for d in range(7):
        assert np.isnan(res.loc[d, "baseline_med_hr"])

    # Day 14 baseline should be 70.0
    assert res.loc[14, "baseline_med_hr"] == pytest.approx(70.0)

    # Day 15 baseline must STILL be 70.0 because day 15 reading cannot leak into day 15 baseline!
    assert res.loc[15, "baseline_med_hr"] == pytest.approx(70.0)

    # Day 15 z-score must reflect the deviation
    assert res.loc[15, "z_hr"] > 10.0


def test_feature_sets_dimensions():
    """Validates feature sets A (14), Z (20), and B (34)."""
    df_raw, _ = simulate_cohort(cohort_id=1, n_subjects=10, n_days=20, seed=42)
    df_prep, _ = preprocess_cohort(df_raw, seed=42)
    df_base = compute_all_baselines(df_prep, window_size=28, min_days=7, drop_first_days=True)

    # Check dropped first 7 days: 20 days -> 13 days evaluated per subject
    assert df_base["day"].min() == 7
    assert len(df_base) == 10 * 13

    matrices = get_feature_matrices(df_base)

    X_A, y_A = matrices["A"]
    X_Z, y_Z = matrices["Z"]
    X_B, y_B = matrices["B"]

    assert X_A.shape[1] == 14
    assert X_Z.shape[1] == 20
    assert X_B.shape[1] == 34
    assert len(X_A) == len(X_B) == len(y_A)

    # Verify no NaNs in feature matrix B
    assert X_B.isna().sum().sum() == 0

"""Unit tests for Stage 2: Preprocessing and Leakage-Safe Splitting.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_gen import simulate_cohort
from src.preprocess import (
    add_log_transforms,
    apply_plausibility_filter,
    compute_missing_count,
    get_subject_split_ids,
    preprocess_cohort,
)


def test_plausibility_filter():
    """Checks that extreme artefacts outside physiological ranges become NaN."""
    df_test = pd.DataFrame(
        {
            "hr": [25.0, 75.0, 240.0],
            "spo2": [50.0, 98.0, 105.0],
            "temp": [32.0, 36.8, 44.0],
        }
    )
    filtered = apply_plausibility_filter(df_test)
    assert np.isnan(filtered.loc[0, "hr"])
    assert filtered.loc[1, "hr"] == 75.0
    assert np.isnan(filtered.loc[2, "hr"])

    assert np.isnan(filtered.loc[0, "spo2"])
    assert filtered.loc[1, "spo2"] == 98.0
    assert np.isnan(filtered.loc[2, "spo2"])

    assert np.isnan(filtered.loc[0, "temp"])
    assert filtered.loc[1, "temp"] == 36.8
    assert np.isnan(filtered.loc[2, "temp"])


def test_missing_count():
    """Verifies daily missingness count calculation."""
    df = pd.DataFrame(
        {
            "hr": [np.nan, 70.0, np.nan],
            "sbp": [np.nan, 120.0, 130.0],
            "dbp": [80.0, np.nan, np.nan],
            "spo2": [98.0, 99.0, 97.0],
            "temp": [36.5, 36.6, 36.7],
            "rr": [16.0, 18.0, 20.0],
            "hrv": [40.0, 45.0, 50.0],
            "sleep": [7.0, 8.0, 6.5],
            "steps": [8000.0, 9000.0, 7500.0],
            "stress": [30.0, 40.0, 35.0],
        }
    )
    df_res = compute_missing_count(df)
    assert df_res["missing_count"].tolist() == [2, 1, 2]


def test_subject_split_disjoint():
    """Ensures subject splits are strictly disjoint with exact 60/15/25 distribution."""
    train_ids, val_ids, test_ids = get_subject_split_ids(300, seed=42)
    assert len(train_ids) == 180
    assert len(val_ids) == 45
    assert len(test_ids) == 75

    s_train = set(train_ids)
    s_val = set(val_ids)
    s_test = set(test_ids)

    assert len(s_train.intersection(s_val)) == 0
    assert len(s_train.intersection(s_test)) == 0
    assert len(s_val.intersection(s_test)) == 0
    assert len(s_train.union(s_val).union(s_test)) == 300


def test_complete_preprocessing_pipeline():
    """Tests full preprocessing and confirms zero NaNs remain in daily vitals."""
    df_raw, _ = simulate_cohort(cohort_id=1, n_subjects=50, n_days=20, seed=42)
    df_prep, artifacts = preprocess_cohort(df_raw, seed=42)

    assert len(df_prep) == len(df_raw)
    assert "missing_count" in df_prep.columns
    assert "log_hrv" in df_prep.columns
    assert "log_steps" in df_prep.columns
    assert "split" in df_prep.columns

    # Check no NaNs left in processed vitals
    from src.config import DAILY_VITALS
    for col in DAILY_VITALS:
        assert df_prep[col].isna().sum() == 0, f"NaNs found in {col}"

    assert len(artifacts.train_medians) == len(DAILY_VITALS)

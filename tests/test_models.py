"""Unit tests for Stage 5: Machine Learning Models and Validation Tuning.
"""

import numpy as np
import pandas as pd
import pytest

from src.baseline import compute_all_baselines, get_feature_matrices
from src.config import FEATURE_SET_A, FEATURE_SET_B
from src.data_gen import simulate_cohort
from src.models import (
    train_and_tune_logistic_regression,
    train_and_tune_random_forest,
    train_and_tune_svm,
    train_and_tune_xgboost,
)
from src.preprocess import preprocess_cohort


@pytest.fixture(scope="module")
def prepared_cohort_data():
    """Generates a small cohort for fast unit testing of models."""
    df_raw, _ = simulate_cohort(cohort_id=1, n_subjects=30, n_days=25, seed=42)
    df_prep, artifacts = preprocess_cohort(df_raw, seed=42)
    df_base = compute_all_baselines(df_prep, window_size=28, min_days=7, drop_first_days=True)

    train_mask = df_base["split"] == "train"
    val_mask = df_base["split"] == "val"
    test_mask = df_base["split"] == "test"

    matrices = get_feature_matrices(df_base)
    X_A, y = matrices["A"]
    X_B, _ = matrices["B"]

    return {
        "X_train_A": X_A[train_mask],
        "y_train": y[train_mask],
        "X_val_A": X_A[val_mask],
        "y_val": y[val_mask],
        "X_test_A": X_A[test_mask],
        "y_test": y[test_mask],
        "X_train_B": X_B[train_mask],
        "X_val_B": X_B[val_mask],
        "X_test_B": X_B[test_mask],
    }


def test_logistic_regression_training(prepared_cohort_data):
    """Tests LR training and valid probability outputs."""
    d = prepared_cohort_data
    bundle = train_and_tune_logistic_regression(
        d["X_train_A"],
        d["y_train"],
        d["X_val_A"],
        d["y_val"],
        feature_set_name="A",
        feature_names=FEATURE_SET_A,
    )
    preds = bundle.predict(d["X_test_A"])
    probs = bundle.predict_proba(d["X_test_A"])

    assert len(preds) == len(d["X_test_A"])
    assert probs.shape == (len(d["X_test_A"]), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert bundle.val_macro_f1 >= 0.0


def test_random_forest_training(prepared_cohort_data):
    """Tests RF training with 250 trees and leaf size tuning."""
    d = prepared_cohort_data
    bundle = train_and_tune_random_forest(
        d["X_train_A"],
        d["y_train"],
        d["X_val_A"],
        d["y_val"],
        feature_set_name="A",
        feature_names=FEATURE_SET_A,
        n_estimators=50,  # Fast for unit test
    )
    preds = bundle.predict(d["X_test_A"])
    assert len(preds) == len(d["X_test_A"])
    assert bundle.best_param["min_samples_leaf"] in [1, 5]


def test_svm_training(prepared_cohort_data):
    """Tests RBF-SVM training with subsampling and probability outputs."""
    d = prepared_cohort_data
    bundle = train_and_tune_svm(
        d["X_train_A"],
        d["y_train"],
        d["X_val_A"],
        d["y_val"],
        feature_set_name="A",
        feature_names=FEATURE_SET_A,
        subsample_size=1000,
    )
    preds = bundle.predict(d["X_test_A"])
    probs = bundle.predict_proba(d["X_test_A"])
    assert len(preds) == len(d["X_test_A"])
    assert probs.shape == (len(d["X_test_A"]), 3)


def test_xgboost_training(prepared_cohort_data):
    """Tests gradient boosting / XGBoost with depth tuning."""
    d = prepared_cohort_data
    bundle = train_and_tune_xgboost(
        d["X_train_A"],
        d["y_train"],
        d["X_val_A"],
        d["y_val"],
        feature_set_name="A",
        feature_names=FEATURE_SET_A,
    )
    preds = bundle.predict(d["X_test_A"])
    probs = bundle.predict_proba(d["X_test_A"])
    assert len(preds) == len(d["X_test_A"])
    assert probs.shape == (len(d["X_test_A"]), 3)
    assert bundle.best_param["max_depth"] in [3, 5]

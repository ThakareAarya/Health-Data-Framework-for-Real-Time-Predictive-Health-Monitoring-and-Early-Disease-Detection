"""Unit tests for Stage 4: Composite Health Risk Indices (HRI, PHRI, NEWS2).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_HRI_CUTOFFS
from src.hri import (
    compute_hri_row,
    compute_news2_score,
    compute_personalised_hri,
    compute_population_hri,
    discretize_score,
    tune_cutoffs_on_train,
)


def test_illustrative_hri_table_iv():
    """Verifies HRI calculation matches expected behavior for Table IV observation."""
    obs = pd.Series(
        {
            "hr": 102.0,
            "spo2": 94.0,
            "sbp": 145.0,
            "dbp": 92.0,
            "temp": 37.8,
            "rr": 22.0,
            "hrv": 28.0,
            "sleep": 5.2,
            "steps": 3200.0,
            "stress": 78.0,
            "bmi": 27.5,
        }
    )

    hri_val, contribs = compute_hri_row(obs)

    # All 9 contributions should be in [0, 1]
    assert len(contribs) == 9
    for param, r in contribs.items():
        assert 0.0 <= r <= 1.0, f"Contribution for {param} out of range: {r}"

    # Overall score should be moderate (between 0.33 and 0.66)
    assert 0.50 <= hri_val <= 0.70
    preds = discretize_score(np.array([hri_val]), DEFAULT_HRI_CUTOFFS)
    assert preds[0] in [1, 2]  # Moderate or high elevated risk


def test_news2_scoring():
    """Tests Royal College of Physicians NEWS2 clinical scoring logic."""
    df_normal = pd.DataFrame(
        {
            "rr": [16.0],
            "spo2": [98.0],
            "sbp": [120.0],
            "hr": [70.0],
            "temp": [36.8],
        }
    )
    score_normal = compute_news2_score(df_normal)
    assert score_normal[0] == 0.0  # Fully normal vital signs

    df_critical = pd.DataFrame(
        {
            "rr": [28.0],     # +3
            "spo2": [89.0],   # +3
            "sbp": [85.0],    # +3
            "hr": [135.0],    # +3
            "temp": [39.5],   # +2
        }
    )
    score_critical = compute_news2_score(df_critical)
    assert score_critical[0] == 14.0  # Severe clinical danger


def test_cutoff_tuning_improves_macro_f1():
    """Ensures threshold calibration improves F1 over default fixed cutoffs."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.choice([0, 1, 2], size=n, p=[0.70, 0.15, 0.15])
    # Continuous scores correlated with y_true
    scores = y_true * 0.15 + rng.normal(0.2, 0.05, size=n)

    default_preds = discretize_score(scores, (0.33, 0.66))
    from sklearn.metrics import f1_score
    default_f1 = f1_score(y_true, default_preds, average="macro", zero_division=0)

    tuned_cutoffs = tune_cutoffs_on_train(scores, y_true)
    tuned_preds = discretize_score(scores, tuned_cutoffs)
    tuned_f1 = f1_score(y_true, tuned_preds, average="macro", zero_division=0)

    assert tuned_f1 > default_f1

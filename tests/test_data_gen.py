"""Unit tests for Stage 1: Synthetic Data Generator.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DAILY_VITALS,
    DEMOGRAPHICS,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MODERATE_THRESHOLD,
)
from src.data_gen import HealthEpisode, simulate_cohort


def test_health_episode_envelope():
    """Verifies HealthEpisode ramp-up, plateau, and decay envelope."""
    ep = HealthEpisode(
        episode_type="infection",
        onset_day=10,
        ramp_days=2,
        plateau_days=4,
        decay_days=3,
        peak_severity=0.8,
    )
    assert ep.total_duration == 9
    assert ep.severity_at(9) == 0.0  # Before onset
    assert ep.severity_at(10) > 0.0  # Ramp day 1
    assert ep.severity_at(11) == pytest.approx(0.8)  # Ramp day 2 (reaches peak)
    assert ep.severity_at(12) == pytest.approx(0.8)  # Plateau
    assert ep.severity_at(15) == pytest.approx(0.8)  # Plateau last day
    assert 0.0 < ep.severity_at(16) < 0.8  # Decay
    assert ep.severity_at(19) == 0.0  # Finished


def test_cohort_shape_and_columns():
    """Tests that cohort generation yields exact 27,000 rows and required columns."""
    df, episodes = simulate_cohort(cohort_id=1, n_subjects=300, n_days=90, seed=42)
    assert len(df) == 27000
    assert df["subject_id"].nunique() == 300
    assert df["day"].nunique() == 90

    for col in DEMOGRAPHICS + DAILY_VITALS:
        assert col in df.columns

    assert "latent_severity" in df.columns
    assert "label" in df.columns
    assert "target" in df.columns
    assert len(episodes) == 300


def test_latent_severity_label_consistency():
    """Proves labels are derived strictly and solely from latent severity e(t)."""
    df, _ = simulate_cohort(cohort_id=1, n_subjects=50, n_days=30, seed=123)

    low_mask = df["latent_severity"] < SEVERITY_MODERATE_THRESHOLD
    mod_mask = (df["latent_severity"] >= SEVERITY_MODERATE_THRESHOLD) & (
        df["latent_severity"] < SEVERITY_HIGH_THRESHOLD
    )
    high_mask = df["latent_severity"] >= SEVERITY_HIGH_THRESHOLD

    assert (df.loc[low_mask, "label"] == "Low").all()
    assert (df.loc[mod_mask, "label"] == "Moderate").all()
    assert (df.loc[high_mask, "label"] == "High").all()


def test_reproducibility():
    """Verifies that fixing random seed reproduces exact same data."""
    df1, _ = simulate_cohort(cohort_id=1, n_subjects=20, n_days=20, seed=999)
    df2, _ = simulate_cohort(cohort_id=1, n_subjects=20, n_days=20, seed=999)
    pd.testing.assert_frame_range_equal = True
    pd.testing.assert_frame_equal(df1, df2)


def test_class_balance_expectation():
    """Verifies expected class balance (~70% Low, ~14% Moderate, ~15% High)."""
    df, _ = simulate_cohort(cohort_id=1, n_subjects=300, n_days=90, seed=42)
    props = df["label"].value_counts(normalize=True)
    # Tolerances around target distribution
    assert 0.65 <= props["Low"] <= 0.77
    assert 0.10 <= props["Moderate"] <= 0.18
    assert 0.10 <= props["High"] <= 0.20

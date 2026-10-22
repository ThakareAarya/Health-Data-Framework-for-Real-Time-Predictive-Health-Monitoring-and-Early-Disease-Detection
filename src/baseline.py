"""Stage 3: Personal Baseline Modeling and Feature Engineering.

Implements causal personalized baseline modeling:
- Strict backward-looking window W=28 days (requiring >= 7 past days):
    m_k(t) = median(x_k(t-i))_{i=1..W}
    s_k(t) = max(IQR(x_k(t-i))_{i=1..W} / 1.349, phi_k)
    z_k(t) = (x_k(t) - m_k(t)) / s_k(t)
- Sustained deviation: 3-day moving average z_bar_k(t).
- Behavioral feature: sleep_deficit = max(0, 7.5 - sleep).
- Excludes each person's first 7 days (days 0-6) for all methods.
- Constructs Feature Sets:
    Set A (14 features): 10 raw vitals + age + bmi + missing_count + sleep_deficit
    Set Z (20 features): 10 z-scores + 10 3-day-average z-scores
    Set B (34 features): Set A + Set Z
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    BASELINE_MEASURES,
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_Z,
    IQR_FLOOR,
    SLEEP_TARGET_HOURS,
)


def compute_sleep_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """Computes sleep deficit: max(0, 7.5 - sleep)."""
    df_out = df.copy()
    if "sleep" in df_out.columns:
        df_out["sleep_deficit"] = np.maximum(0.0, SLEEP_TARGET_HOURS - df_out["sleep"])
    return df_out


def compute_subject_baselines(
    subject_df: pd.DataFrame,
    window_size: int = 28,
    min_days: int = 7,
) -> pd.DataFrame:
    """Computes causal baseline median, scale, z-score, and 3-day avg z
    for a single subject ordered chronologically by day.
    
    Guarantees that day t uses ONLY days t-window_size to t-1.
    """
    df_subj = subject_df.sort_values("day").copy().reset_index(drop=True)
    n_days = len(df_subj)

    # Initialize new columns
    for col in BASELINE_MEASURES:
        df_subj[f"baseline_med_{col}"] = np.nan
        df_subj[f"baseline_scale_{col}"] = np.nan
        df_subj[f"z_{col}"] = np.nan
        df_subj[f"z_avg3_{col}"] = np.nan

    # For each day t, compute baseline from previous days [max(0, t - window_size), t)
    for col in BASELINE_MEASURES:
        if col not in df_subj.columns:
            continue

        phi = IQR_FLOOR.get(col, 0.1)
        vals = df_subj[col].to_numpy()
        meds = np.full(n_days, np.nan)
        scales = np.full(n_days, np.nan)
        z_scores = np.full(n_days, np.nan)

        for t in range(n_days):
            start_idx = max(0, t - window_size)
            past_vals = vals[start_idx:t]

            if len(past_vals) >= min_days:
                m = float(np.median(past_vals))
                q75, q25 = np.percentile(past_vals, [75, 25])
                iqr = q75 - q25
                s = max(float(iqr / 1.349), phi)
                z = (vals[t] - m) / s

                meds[t] = m
                scales[t] = s
                z_scores[t] = z

        df_subj[f"baseline_med_{col}"] = meds
        df_subj[f"baseline_scale_{col}"] = scales
        df_subj[f"z_{col}"] = z_scores

        # 3-day moving average of z-scores (causal rolling mean over days t-2, t-1, t)
        z_series = pd.Series(z_scores)
        df_subj[f"z_avg3_{col}"] = (
            z_series.rolling(window=3, min_periods=1).mean().to_numpy()
        )

    return df_subj


def compute_all_baselines(
    df: pd.DataFrame,
    window_size: int = 28,
    min_days: int = 7,
    drop_first_days: bool = True,
) -> pd.DataFrame:
    """Computes causal personal baselines for all subjects in the cohort.

    If drop_first_days is True, drops each subject's first min_days (days 0-6)
    as required by the evaluation spec.
    """
    # First compute sleep deficit if not present
    df_ready = compute_sleep_deficit(df)

    # Process per subject
    processed_subjs = []
    for _, group in df_ready.groupby("subject_id", sort=False):
        subj_res = compute_subject_baselines(
            group, window_size=window_size, min_days=min_days
        )
        processed_subjs.append(subj_res)

    df_all = pd.concat(processed_subjs, ignore_index=True)

    if drop_first_days:
        # Exclude days 0 to min_days - 1
        df_all = df_all[df_all["day"] >= min_days].copy().reset_index(drop=True)

    return df_all


def get_feature_matrices(
    df: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Extracts feature matrices for Feature Sets A, Z, and B along with target."""
    y = df["target"].astype(int)

    # Ensure all columns exist
    X_A = df[FEATURE_SET_A].copy()
    X_Z = df[FEATURE_SET_Z].copy()
    X_B = df[FEATURE_SET_B].copy()

    return {
        "A": (X_A, y),
        "Z": (X_Z, y),
        "B": (X_B, y),
    }

"""Stage 4: Composite Health Risk Index (HRI), Personalised HRI (PHRI), and NEWS2.

Implements:
1. Population HRI: Weighted sum of 9 normalized parameters using a priori expert limits.
2. Tuned HRI: Thresholds optimized strictly on training subjects to maximize macro-F1.
3. Personalised HRI (PHRI): Baseline-relative risk contributions r_j^P = clip((delta_j * z_j - 1)/3, 0, 1).
4. NEWS2-style clinical score: Standard Royal College of Physicians bands with tuned cutoffs.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import (
    DEFAULT_HRI_CUTOFFS,
    HRI_CONFIG,
    PHRI_HARM_DELTA,
)


def compute_hri_row(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """Calculates population HRI and individual risk contributions for a single row."""
    total_score = 0.0
    contributions = {}

    for param, cfg in HRI_CONFIG.items():
        val = float(row[param])
        if cfg.harm == "high":
            r_j = (val - cfg.lower) / (cfg.upper - cfg.lower)
        else:  # 'low' harm
            r_j = (cfg.upper - val) / (cfg.upper - cfg.lower)

        r_j = float(np.clip(r_j, 0.0, 1.0))
        contributions[param] = r_j
        total_score += cfg.weight * r_j

    hri_val = float(np.clip(total_score, 0.0, 1.0))
    return hri_val, contributions


def compute_population_hri(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Computes continuous population HRI for all rows in df.

    Returns:
        (hri_scores, df_contributions)
    """
    scores = np.zeros(len(df))
    contrib_dicts = []

    for idx, (_, row) in enumerate(df.iterrows()):
        hri, contrib = compute_hri_row(row)
        scores[idx] = hri
        contrib_dicts.append(contrib)

    df_contrib = pd.DataFrame(contrib_dicts)
    return scores, df_contrib


def compute_personalised_hri(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Computes continuous Personalised HRI (PHRI) using baseline z-scores:
    r_j^P = clip((delta_j * z_j - 1) / 3, 0, 1)
    PHRI = sum_j w_j * r_j^P
    """
    scores = np.zeros(len(df))
    contrib_dicts = []

    # Map parameter names to z-score column names
    # Note: hrv uses z_log_hrv
    param_to_z = {
        "hr": "z_hr",
        "sbp": "z_sbp",
        "dbp": "z_dbp",
        "spo2": "z_spo2",
        "temp": "z_temp",
        "rr": "z_rr",
        "hrv": "z_log_hrv",
        "sleep": "z_sleep",
        "stress": "z_stress",
    }

    for idx, (_, row) in enumerate(df.iterrows()):
        total = 0.0
        c_dict = {}
        for param, cfg in HRI_CONFIG.items():
            z_col = param_to_z[param]
            z_val = float(row[z_col]) if z_col in row else 0.0
            delta = PHRI_HARM_DELTA[param]

            # Linear ramp from 1 to 4 standard deviations in harmful direction
            r_p = float(np.clip((delta * z_val - 1.0) / 3.0, 0.0, 1.0))
            c_dict[param] = r_p
            total += cfg.weight * r_p

        phri_val = float(np.clip(total, 0.0, 1.0))
        scores[idx] = phri_val
        contrib_dicts.append(c_dict)

    df_contrib = pd.DataFrame(contrib_dicts)
    return scores, df_contrib


def compute_news2_score(df: pd.DataFrame) -> np.ndarray:
    """Computes standard National Early Warning Score 2 (NEWS2) points.
    Published bands for RR, SpO2 (Scale 1), SBP, HR, Temp.
    """
    news_scores = np.zeros(len(df), dtype=float)

    for idx, (_, row) in enumerate(df.iterrows()):
        score = 0

        # 1. Respiration Rate (/min)
        rr = float(row["rr"])
        if rr <= 8 or rr >= 25:
            score += 3
        elif 21 <= rr <= 24:
            score += 2
        elif 9 <= rr <= 11:
            score += 1

        # 2. SpO2 (%) - Scale 1
        spo2 = float(row["spo2"])
        if spo2 <= 91:
            score += 3
        elif 92 <= spo2 <= 93:
            score += 2
        elif 94 <= spo2 <= 95:
            score += 1

        # 3. Systolic Blood Pressure (mmHg)
        sbp = float(row["sbp"])
        if sbp <= 90 or sbp >= 220:
            score += 3
        elif 91 <= sbp <= 100:
            score += 2
        elif 101 <= sbp <= 110:
            score += 1

        # 4. Heart Rate (bpm)
        hr = float(row["hr"])
        if hr <= 40 or hr >= 131:
            score += 3
        elif 111 <= hr <= 130:
            score += 2
        elif 41 <= hr <= 50 or 91 <= hr <= 110:
            score += 1

        # 5. Temperature (°C)
        temp = float(row["temp"])
        if temp <= 35.0:
            score += 3
        elif temp >= 39.1:
            score += 2
        elif (35.1 <= temp <= 36.0) or (38.1 <= temp <= 39.0):
            score += 1

        news_scores[idx] = float(score)

    return news_scores


def discretize_score(
    scores: np.ndarray, cutoffs: Tuple[float, float]
) -> np.ndarray:
    """Discretizes continuous scores into classes 0 (Low), 1 (Moderate), 2 (High)."""
    c1, c2 = cutoffs
    preds = np.zeros(len(scores), dtype=int)
    preds[(scores >= c1) & (scores < c2)] = 1
    preds[scores >= c2] = 2
    return preds


def tune_cutoffs_on_train(
    train_scores: np.ndarray,
    train_y: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    n_steps: int = 40,
) -> Tuple[float, float]:
    """Finds cutoffs (theta_1, theta_2) maximizing macro-F1 on training subjects."""
    if min_val is None:
        min_val = float(np.percentile(train_scores, 2))
    if max_val is None:
        max_val = float(np.percentile(train_scores, 98))

    threshold_grid = np.linspace(min_val, max_val, n_steps)
    best_f1 = -1.0
    best_cutoffs = (0.33, 0.66)

    # Grid search for theta_1 < theta_2
    for i in range(len(threshold_grid)):
        c1 = threshold_grid[i]
        for j in range(i + 1, len(threshold_grid)):
            c2 = threshold_grid[j]
            preds = discretize_score(train_scores, (c1, c2))
            f1 = f1_score(train_y, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_cutoffs = (float(c1), float(c2))

    return best_cutoffs

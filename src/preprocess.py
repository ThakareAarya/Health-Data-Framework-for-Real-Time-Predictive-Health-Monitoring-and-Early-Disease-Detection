"""Stage 2: Preprocessing and Leakage-Safe Data Splitting.

Implements:
1. Plausibility filtering: readings outside physiological limits become NaN.
2. Missingness counter: 'missing_count' recorded per day.
3. Subject-wise cohort splitting (60% train, 15% val, 25% test).
4. Causal forward-fill: maximum 3 days within each person's historical trajectory.
5. Training-set median imputation: remaining gaps filled using training subjects' medians only.
6. Log transformations for skewed parameters (HRV and step count).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DAILY_VITALS,
    DEFAULT_NUM_SUBJECTS,
    PLAUSIBLE_RANGES,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


@dataclass
class PreprocessingArtifacts:
    """Stores fitted parameters from the training set to prevent data leakage."""
    train_medians: Dict[str, float]
    train_subject_ids: List[int]
    val_subject_ids: List[int]
    test_subject_ids: List[int]


def apply_plausibility_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Sets values outside physiologically plausible ranges to NaN."""
    df_clean = df.copy()
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col in df_clean.columns:
            invalid_mask = (df_clean[col] < low) | (df_clean[col] > high)
            df_clean.loc[invalid_mask, col] = np.nan
    return df_clean


def compute_missing_count(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates number of missing daily vitals per person-day."""
    df_res = df.copy()
    # Count NaNs across the 10 daily vitals
    vitals_present = [v for v in DAILY_VITALS if v in df_res.columns]
    df_res["missing_count"] = df_res[vitals_present].isna().sum(axis=1)
    return df_res


def get_subject_split_ids(
    n_subjects: int = DEFAULT_NUM_SUBJECTS,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Splits subject IDs strictly into disjoint train, val, and test groups."""
    rng = np.random.Generator(np.random.PCG64(seed))
    all_subjects = np.arange(n_subjects)
    shuffled_subjects = rng.permutation(all_subjects).tolist()

    n_train = int(round(n_subjects * train_ratio))
    n_val = int(round(n_subjects * val_ratio))
    
    train_ids = sorted(shuffled_subjects[:n_train])
    val_ids = sorted(shuffled_subjects[n_train : n_train + n_val])
    test_ids = sorted(shuffled_subjects[n_train + n_val :])

    return train_ids, val_ids, test_ids


def causal_impute(
    df: pd.DataFrame,
    train_ids: List[int],
    ffill_limit: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Performs causal forward-fill within subject (max 3 days), then fills
    remaining NaNs using TRAINING-SET medians only to prevent leakage.
    """
    df_imputed = df.copy()
    # Sort by subject and day to guarantee causal chronological order
    df_imputed = df_imputed.sort_values(["subject_id", "day"]).reset_index(drop=True)

    # 1. Causal forward-fill per subject up to ffill_limit days
    for col in DAILY_VITALS:
        if col in df_imputed.columns:
            df_imputed[col] = (
                df_imputed.groupby("subject_id")[col]
                .transform(lambda s: s.ffill(limit=ffill_limit))
            )

    # 2. Compute medians strictly on training subjects
    train_mask = df_imputed["subject_id"].isin(train_ids)
    train_df = df_imputed[train_mask]
    
    train_medians: Dict[str, float] = {}
    for col in DAILY_VITALS:
        if col in df_imputed.columns:
            med_val = float(train_df[col].median(skipna=True))
            # Fallback if entirely empty (rare)
            if np.isnan(med_val):
                med_val = float(df_imputed[col].median(skipna=True))
            train_medians[col] = med_val
            # Impute remaining NaNs across all splits using the training median
            df_imputed[col] = df_imputed[col].fillna(med_val)

    return df_imputed, train_medians


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Adds natural log transformations for skewed vitals (HRV and step count)."""
    df_out = df.copy()
    if "hrv" in df_out.columns:
        df_out["log_hrv"] = np.log(np.maximum(df_out["hrv"], 1e-3))
    if "steps" in df_out.columns:
        df_out["log_steps"] = np.log(np.maximum(df_out["steps"], 1.0))
    return df_out


def preprocess_cohort(
    df_raw: pd.DataFrame,
    seed: int = 42,
    existing_artifacts: Optional[PreprocessingArtifacts] = None,
) -> Tuple[pd.DataFrame, PreprocessingArtifacts]:
    """Executes complete Stage 2 preprocessing pipeline on a cohort.

    Returns:
        (preprocessed_df, artifacts)
    """
    # 1. Range validation / plausibility filtering
    df_filtered = apply_plausibility_filter(df_raw)

    # 2. Count missing values before imputation
    df_with_missing = compute_missing_count(df_filtered)

    # 3. Subject-wise train/val/test splitting
    n_subjects = int(df_with_missing["subject_id"].nunique())
    if existing_artifacts is None:
        train_ids, val_ids, test_ids = get_subject_split_ids(
            n_subjects=n_subjects, seed=seed
        )
    else:
        train_ids = existing_artifacts.train_subject_ids
        val_ids = existing_artifacts.val_subject_ids
        test_ids = existing_artifacts.test_subject_ids

    # 4. Causal forward-fill + training median imputation
    df_imputed, train_medians = causal_impute(
        df_with_missing, train_ids=train_ids, ffill_limit=3
    )

    # 5. Log transforms
    df_transformed = add_log_transforms(df_imputed)

    # 6. Mark split partition
    df_transformed["split"] = "unknown"
    df_transformed.loc[df_transformed["subject_id"].isin(train_ids), "split"] = "train"
    df_transformed.loc[df_transformed["subject_id"].isin(val_ids), "split"] = "val"
    df_transformed.loc[df_transformed["subject_id"].isin(test_ids), "split"] = "test"

    artifacts = PreprocessingArtifacts(
        train_medians=train_medians,
        train_subject_ids=train_ids,
        val_subject_ids=val_ids,
        test_subject_ids=test_ids,
    )

    return df_transformed, artifacts

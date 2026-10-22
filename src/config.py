"""Configuration constants, reference ranges, clinical thresholds, and weights.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Ten daily physiological and behavioral vitals
DAILY_VITALS: List[str] = [
    "hr",
    "sbp",
    "dbp",
    "spo2",
    "temp",
    "rr",
    "hrv",
    "sleep",
    "steps",
    "stress",
]

# Static demographics
DEMOGRAPHICS: List[str] = ["age", "bmi"]

# 12 Raw observed parameters
RAW_PARAMETERS: List[str] = DEMOGRAPHICS + DAILY_VITALS

# Physiologically plausible limits for data validation/cleaning
PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "hr": (30.0, 220.0),       # bpm
    "sbp": (60.0, 260.0),      # mmHg
    "dbp": (40.0, 150.0),      # mmHg
    "spo2": (60.0, 100.0),     # %
    "temp": (33.0, 43.0),      # °C
    "rr": (6.0, 50.0),         # breaths/min
    "hrv": (5.0, 250.0),       # ms (RMSSD)
    "sleep": (0.0, 20.0),      # hours
    "steps": (0.0, 60000.0),   # count
    "stress": (0.0, 100.0),    # scale 0-100
}

# Table I: Episode types and effects at severity e = 1
# Note: For HRV and Steps, the effect is on the natural log scale!
EPISODE_EFFECTS: Dict[str, Dict[str, float]] = {
    "infection": {
        "hr": 14.0,
        "sbp": 0.0,
        "dbp": 0.0,
        "spo2": -2.5,
        "temp": 1.3,
        "rr": 4.0,
        "hrv_log": -0.45,
        "steps_log": -0.60,
        "sleep": 0.0,
        "stress": 8.0,
    },
    "cardiovascular": {
        "hr": 10.0,
        "sbp": 28.0,
        "dbp": 14.0,
        "spo2": -1.5,
        "temp": 0.0,
        "rr": 2.0,
        "hrv_log": -0.25,
        "steps_log": -0.15,
        "sleep": -0.3,
        "stress": 8.0,
    },
    "stress_sleep": {
        "hr": 8.0,
        "sbp": 8.0,
        "dbp": 4.0,
        "spo2": 0.0,
        "temp": 0.05,
        "rr": 1.0,
        "hrv_log": -0.40,
        "steps_log": -0.30,
        "sleep": -2.2,
        "stress": 30.0,
    },
}

# Table II: HRI Reference limits and weights (fixed a priori)
# 9 parameters included in HRI (all daily vitals except steps)
@dataclass(frozen=True)
class ParameterHRIConfig:
    lower: float
    upper: float
    harm: str  # 'high' or 'low'
    weight: float

HRI_CONFIG: Dict[str, ParameterHRIConfig] = {
    "hr": ParameterHRIConfig(lower=70.0, upper=120.0, harm="high", weight=0.15),
    "sbp": ParameterHRIConfig(lower=120.0, upper=160.0, harm="high", weight=0.15),
    "dbp": ParameterHRIConfig(lower=80.0, upper=100.0, harm="high", weight=0.05),
    "spo2": ParameterHRIConfig(lower=90.0, upper=98.0, harm="low", weight=0.15),
    "temp": ParameterHRIConfig(lower=36.8, upper=38.5, harm="high", weight=0.15),
    "rr": ParameterHRIConfig(lower=16.0, upper=24.0, harm="high", weight=0.10),
    "hrv": ParameterHRIConfig(lower=20.0, upper=60.0, harm="low", weight=0.10),
    "sleep": ParameterHRIConfig(lower=3.5, upper=7.5, harm="low", weight=0.05),
    "stress": ParameterHRIConfig(lower=40.0, upper=90.0, harm="high", weight=0.10),
}

# Harmful direction sign delta_j for PHRI: +1 if high is harmful, -1 if low is harmful
PHRI_HARM_DELTA: Dict[str, float] = {
    "hr": 1.0,
    "sbp": 1.0,
    "dbp": 1.0,
    "spo2": -1.0,
    "temp": 1.0,
    "rr": 1.0,
    "hrv": -1.0,
    "sleep": -1.0,
    "stress": 1.0,
}

# Small floor phi_k for baseline IQR spread to prevent division by near-zero spread
IQR_FLOOR: Dict[str, float] = {
    "hr": 1.0,
    "sbp": 1.5,
    "dbp": 1.0,
    "spo2": 0.2,
    "temp": 0.05,
    "rr": 0.5,
    "log_hrv": 0.05,
    "sleep": 0.2,
    "log_steps": 0.05,
    "stress": 1.0,
}

# Sleep target duration (hours) for sleep deficit feature
SLEEP_TARGET_HOURS: float = 7.5

# Latent episode severity cutoffs for class labels: Low, Moderate, High
SEVERITY_MODERATE_THRESHOLD: float = 0.25
SEVERITY_HIGH_THRESHOLD: float = 0.60

CLASS_LABELS: List[str] = ["Low", "Moderate", "High"]
CLASS_TO_INT: Dict[str, int] = {"Low": 0, "Moderate": 1, "High": 2}
INT_TO_CLASS: Dict[int, str] = {0: "Low", 1: "Moderate", 2: "High"}

# Default HRI fixed cutoffs (Table III)
DEFAULT_HRI_CUTOFFS: Tuple[float, float] = (0.33, 0.66)

# Feature Set names
FEATURE_SET_A: List[str] = [
    "hr",
    "sbp",
    "dbp",
    "spo2",
    "temp",
    "rr",
    "hrv",
    "sleep",
    "steps",
    "stress",
    "age",
    "bmi",
    "missing_count",
    "sleep_deficit",
]

# Set Z: 10 z-scores and 10 3-day-average z-scores
BASELINE_MEASURES: List[str] = [
    "hr",
    "sbp",
    "dbp",
    "spo2",
    "temp",
    "rr",
    "log_hrv",
    "sleep",
    "log_steps",
    "stress",
]

FEATURE_SET_Z: List[str] = [f"z_{m}" for m in BASELINE_MEASURES] + [
    f"z_avg3_{m}" for m in BASELINE_MEASURES
]

FEATURE_SET_B: List[str] = FEATURE_SET_A + FEATURE_SET_Z

# Default seeds for 5 independent cohorts
COHORT_SEEDS: List[int] = [42, 101, 202, 303, 404]

# Cohort generation dimensions
DEFAULT_NUM_SUBJECTS: int = 300
DEFAULT_NUM_DAYS: int = 90

# Subject-wise split ratios: 60% train, 15% validation, 25% test
TRAIN_RATIO: float = 0.60
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.25

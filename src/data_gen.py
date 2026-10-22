"""Stage 1: Synthetic Data Generator.

Generates longitudinal wearable health datasets for cohorts of 300 subjects
monitored over 90 days (27,000 person-days per cohort).
Each subject possesses individualized baselines, benign daily lifestyle load,
latent health episodes (infection, cardiovascular strain, stress/sleep loss),
and realistic sensor corruptions (artefacts, missingness, device-off days).
Ground-truth labels depend strictly on the latent severity e(t) and NEVER on
observed features.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    CLASS_LABELS,
    DAILY_VITALS,
    DEFAULT_NUM_DAYS,
    DEFAULT_NUM_SUBJECTS,
    EPISODE_EFFECTS,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MODERATE_THRESHOLD,
)


@dataclass
class HealthEpisode:
    """Represents a discrete latent health episode."""
    episode_type: str
    onset_day: int
    ramp_days: int
    plateau_days: int
    decay_days: int
    peak_severity: float

    @property
    def total_duration(self) -> int:
        return self.ramp_days + self.plateau_days + self.decay_days

    def severity_at(self, day: int) -> float:
        """Computes latent severity on a given day."""
        tau = day - self.onset_day
        if tau < 0 or tau >= self.total_duration:
            return 0.0

        if tau < self.ramp_days:
            # Linear ramp-up from 0 to peak_severity
            return self.peak_severity * ((tau + 1) / self.ramp_days)
        elif tau < self.ramp_days + self.plateau_days:
            # Plateau at peak_severity
            return self.peak_severity
        else:
            # Linear decay from peak_severity to 0
            decay_idx = tau - (self.ramp_days + self.plateau_days)
            remaining = 1.0 - ((decay_idx + 1) / (self.decay_days + 1))
            return max(0.0, float(self.peak_severity * remaining))


def generate_subject_baselines(
    n_subjects: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Generates unique physiological baselines for n_subjects."""
    # Demographics
    age = rng.integers(22, 73, size=n_subjects)
    bmi = np.clip(rng.normal(26.2, 4.2, size=n_subjects), 18.5, 42.0)

    age_diff = age - 45.0
    bmi_diff = bmi - 25.0

    # Individualized resting baselines with realistic between-person spread
    hr_base = rng.normal(68.0 + 0.08 * age_diff + 0.35 * bmi_diff, 7.0)
    sbp_base = rng.normal(118.0 + 0.35 * age_diff + 0.60 * bmi_diff, 9.0)
    dbp_base = rng.normal(78.0 + 0.15 * age_diff + 0.40 * bmi_diff, 6.5)

    # SpO2 baseline (subtle decline with age/obesity)
    spo2_base = np.clip(
        rng.normal(
            97.8 - 0.02 * np.maximum(0.0, age_diff) - 0.03 * np.maximum(0.0, bmi_diff),
            0.55,
        ),
        95.0,
        99.4,
    )

    temp_base = rng.normal(36.65, 0.16)
    rr_base = rng.normal(15.5 + 0.03 * bmi_diff, 1.4)

    # HRV and steps baselines (log scale)
    log_hrv_base = rng.normal(
        np.log(48.0) - 0.008 * age_diff - 0.010 * bmi_diff, 0.25
    )
    sleep_base = np.clip(rng.normal(7.3 - 0.010 * age_diff, 0.65), 5.5, 9.2)
    log_steps_base = rng.normal(
        np.log(7800.0) - 0.006 * age_diff - 0.015 * bmi_diff, 0.28
    )

    stress_base = np.clip(rng.normal(35.0, 7.5), 15.0, 60.0)

    return pd.DataFrame(
        {
            "subject_id": np.arange(n_subjects),
            "age": age,
            "bmi": bmi,
            "hr_base": hr_base,
            "sbp_base": sbp_base,
            "dbp_base": dbp_base,
            "spo2_base": spo2_base,
            "temp_base": temp_base,
            "rr_base": rr_base,
            "log_hrv_base": log_hrv_base,
            "sleep_base": sleep_base,
            "log_steps_base": log_steps_base,
            "stress_base": stress_base,
        }
    )


def simulate_cohort(
    cohort_id: int = 1,
    n_subjects: int = DEFAULT_NUM_SUBJECTS,
    n_days: int = DEFAULT_NUM_DAYS,
    seed: int = 42,
    noise_mult: float = 1.0,
    missing_rate: float = 0.05,
    device_off_rate: float = 0.02,
    artefact_rate: float = 0.01,
) -> Tuple[pd.DataFrame, List[List[HealthEpisode]]]:
    """Simulates a complete longitudinal cohort of subjects over n_days.

    Returns:
        (df_cohort, episodes_per_subject)
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    baselines_df = generate_subject_baselines(n_subjects, rng)

    rows = []
    all_episodes: List[List[HealthEpisode]] = []

    episode_types = ["infection", "cardiovascular", "stress_sleep"]

    for subj_idx in range(n_subjects):
        subj_base = baselines_df.iloc[subj_idx]

        # 1. Sample latent health episodes for this subject
        # Arrival rate: lambda = 0.035 per day
        subject_episodes: List[HealthEpisode] = []
        for day in range(n_days):
            if rng.random() < 0.035:
                ep_type = rng.choice(episode_types)
                ramp = int(rng.integers(1, 4))      # 1, 2, or 3 days
                plateau = int(rng.integers(4, 11))  # 4 to 10 days
                decay = 3                          # 3 days
                peak_sev = float(rng.uniform(0.25, 1.0))
                subject_episodes.append(
                    HealthEpisode(
                        episode_type=str(ep_type),
                        onset_day=day,
                        ramp_days=ramp,
                        plateau_days=plateau,
                        decay_days=decay,
                        peak_severity=peak_sev,
                    )
                )
        all_episodes.append(subject_episodes)

        # 2. Benign lifestyle load: AR(1) process with rho = 0.5
        rho = 0.5
        u_t = 0.0
        lifestyle_loads = np.zeros(n_days)
        for day in range(n_days):
            u_t = rho * u_t + np.sqrt(1.0 - rho**2) * rng.normal(0.0, 1.0)
            lifestyle_loads[day] = u_t

        # 3. Simulate day-by-day readings
        for day in range(n_days):
            # Weekend effect: day % 7 in {5, 6}
            is_weekend = (day % 7) in (5, 6)
            load = lifestyle_loads[day]

            # Sum of active episode severities and effects
            total_severity = 0.0
            effect_deltas: Dict[str, float] = {
                "hr": 0.0,
                "sbp": 0.0,
                "dbp": 0.0,
                "spo2": 0.0,
                "temp": 0.0,
                "rr": 0.0,
                "hrv_log": 0.0,
                "steps_log": 0.0,
                "sleep": 0.0,
                "stress": 0.0,
            }

            for ep in subject_episodes:
                s_ep = ep.severity_at(day)
                if s_ep > 0:
                    total_severity += s_ep
                    eff = EPISODE_EFFECTS[ep.episode_type]
                    for k, val in eff.items():
                        effect_deltas[k] += val * s_ep

            # Cap latent severity at 1.0
            e_t = float(min(1.0, total_severity))

            # Ground truth label: Low (<0.25), Moderate (0.25..0.60), High (>=0.60)
            if e_t < SEVERITY_MODERATE_THRESHOLD:
                label_str = "Low"
                label_int = 0
            elif e_t < SEVERITY_HIGH_THRESHOLD:
                label_str = "Moderate"
                label_int = 1
            else:
                label_str = "High"
                label_int = 2

            # Compute nominal daily physiological values
            # Baseline + Lifestyle load + Weekend shifts + Episode effects
            hr_val = (
                subj_base["hr_base"]
                + 2.5 * load
                + effect_deltas["hr"]
                + rng.normal(0.0, 1.8 * noise_mult)
            )

            sbp_val = (
                subj_base["sbp_base"]
                + 3.5 * load
                + effect_deltas["sbp"]
                + rng.normal(0.0, 2.5 * noise_mult)
            )

            dbp_val = (
                subj_base["dbp_base"]
                + 2.0 * load
                + effect_deltas["dbp"]
                + rng.normal(0.0, 1.8 * noise_mult)
            )

            spo2_val = (
                subj_base["spo2_base"]
                + effect_deltas["spo2"]
                + rng.normal(0.0, 0.35 * noise_mult)
            )
            spo2_val = min(100.0, spo2_val)

            temp_val = (
                subj_base["temp_base"]
                + effect_deltas["temp"]
                + rng.normal(0.0, 0.08 * noise_mult)
            )

            rr_val = (
                subj_base["rr_base"]
                + effect_deltas["rr"]
                + rng.normal(0.0, 0.7 * noise_mult)
            )

            # HRV (modeled on log scale)
            log_hrv = (
                subj_base["log_hrv_base"]
                - 0.08 * load
                + effect_deltas["hrv_log"]
                + rng.normal(0.0, 0.06 * noise_mult)
            )
            hrv_val = float(np.exp(log_hrv))

            # Sleep hours
            sleep_weekend = 0.6 if is_weekend else 0.0
            sleep_val = (
                subj_base["sleep_base"]
                - 0.35 * load
                + sleep_weekend
                + effect_deltas["sleep"]
                + rng.normal(0.0, 0.25 * noise_mult)
            )
            sleep_val = max(0.0, sleep_val)

            # Steps (modeled on log scale)
            steps_weekend = 0.12 if is_weekend else 0.0
            log_steps = (
                subj_base["log_steps_base"]
                - 0.08 * load
                + steps_weekend
                + effect_deltas["steps_log"]
                + rng.normal(0.0, 0.08 * noise_mult)
            )
            steps_val = float(np.exp(log_steps))

            # Stress level (0-100)
            stress_weekend = -3.0 if is_weekend else 0.0
            stress_val = (
                subj_base["stress_base"]
                + 5.0 * load
                + stress_weekend
                + effect_deltas["stress"]
                + rng.normal(0.0, 2.5 * noise_mult)
            )
            stress_val = np.clip(stress_val, 0.0, 100.0)

            rows.append(
                {
                    "cohort_id": cohort_id,
                    "subject_id": int(subj_idx),
                    "day": int(day),
                    "age": int(subj_base["age"]),
                    "bmi": float(subj_base["bmi"]),
                    "hr": float(hr_val),
                    "sbp": float(sbp_val),
                    "dbp": float(dbp_val),
                    "spo2": float(spo2_val),
                    "temp": float(temp_val),
                    "rr": float(rr_val),
                    "hrv": float(hrv_val),
                    "sleep": float(sleep_val),
                    "steps": float(steps_val),
                    "stress": float(stress_val),
                    "latent_severity": float(e_t),
                    "label": label_str,
                    "target": int(label_int),
                }
            )

    df = pd.DataFrame(rows)

    # 4. Inject Sensor Corruptions (Noise, Artefacts, Missingness, Device-off)
    # 4a. 2% Device-off days (all 10 daily vitals missing)
    if device_off_rate > 0.0:
        device_off_mask = rng.random(len(df)) < device_off_rate
        for v in DAILY_VITALS:
            df.loc[device_off_mask, v] = np.nan

    # 4b. 5% Missing at random on remaining readings
    if missing_rate > 0.0:
        for v in DAILY_VITALS:
            mar_mask = (rng.random(len(df)) < missing_rate) & (~df[v].isna())
            df.loc[mar_mask, v] = np.nan

    # 4c. 1% Large sensor artefacts (implausible spikes)
    if artefact_rate > 0.0:
        for v in DAILY_VITALS:
            artefact_mask = (rng.random(len(df)) < artefact_rate) & (~df[v].isna())
            if v == "hr":
                df.loc[artefact_mask, v] = rng.choice([25.0, 245.0, 280.0], size=artefact_mask.sum())
            elif v == "sbp":
                df.loc[artefact_mask, v] = rng.choice([40.0, 290.0, 310.0], size=artefact_mask.sum())
            elif v == "dbp":
                df.loc[artefact_mask, v] = rng.choice([20.0, 180.0, 200.0], size=artefact_mask.sum())
            elif v == "spo2":
                df.loc[artefact_mask, v] = rng.choice([45.0, 52.0, 115.0], size=artefact_mask.sum())
            elif v == "temp":
                df.loc[artefact_mask, v] = rng.choice([31.0, 43.5, 45.0], size=artefact_mask.sum())
            elif v == "rr":
                df.loc[artefact_mask, v] = rng.choice([3.0, 58.0, 65.0], size=artefact_mask.sum())
            elif v == "hrv":
                df.loc[artefact_mask, v] = rng.choice([1.0, 350.0, 500.0], size=artefact_mask.sum())
            elif v == "sleep":
                df.loc[artefact_mask, v] = rng.choice([24.0, 30.0], size=artefact_mask.sum())
            elif v == "steps":
                df.loc[artefact_mask, v] = rng.choice([75000.0, 95000.0], size=artefact_mask.sum())
            elif v == "stress":
                df.loc[artefact_mask, v] = rng.choice([-15.0, 125.0, 150.0], size=artefact_mask.sum())

    return df, all_episodes


def generate_all_cohorts(
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> Dict[int, Tuple[pd.DataFrame, List[List[HealthEpisode]]]]:
    """Generates 5 independent cohorts with reproducible random seeds."""
    from src.config import COHORT_SEEDS
    if seeds is None:
        seeds = COHORT_SEEDS

    cohorts = {}
    for idx, seed in enumerate(seeds, start=1):
        df, episodes = simulate_cohort(cohort_id=idx, seed=seed)
        cohorts[idx] = (df, episodes)
        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, f"cohort_{idx}.csv"), index=False)

    return cohorts


if __name__ == "__main__":
    df_test, eps = simulate_cohort(cohort_id=1, seed=42)
    print(f"Generated cohort 1: {df_test.shape}")
    print("Class label distribution:")
    print(df_test["label"].value_counts(normalize=True))

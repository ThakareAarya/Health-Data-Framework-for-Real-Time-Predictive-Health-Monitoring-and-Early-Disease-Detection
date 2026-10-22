"""Stage 6: Cohort Evaluation, Early Detection, Ablation, and Robustness Experiments.

Generates:
- Table VI: Performance on Held-Out Test Subjects (Mean +/- SD over 5 cohorts).
- Table VII: Early Detection (First 3 Days of High-Reaching Episodes) and False Alerts.
- Table VIII: Feature-Group Ablation for XGBoost (Set A vs Set Z vs Set B).
- Table IX: Macro-F1 Under Degraded Input (Noise x1, x2, x3; Missing 5%, 20%, 40%).
- Paired t-tests comparing Feature Set B vs Set A.
- Exports tables to results/ (CSV & Markdown) and figures (PNG).
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.baseline import compute_all_baselines, get_feature_matrices
from src.config import (
    CLASS_LABELS,
    COHORT_SEEDS,
    DEFAULT_HRI_CUTOFFS,
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_Z,
)
from src.data_gen import HealthEpisode, simulate_cohort
from src.hri import (
    compute_news2_score,
    compute_personalised_hri,
    compute_population_hri,
    discretize_score,
    tune_cutoffs_on_train,
)
from src.models import (
    TrainedModelBundle,
    train_all_models_for_feature_set,
    train_and_tune_xgboost,
)
from src.preprocess import preprocess_cohort


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    continuous_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates accuracy, macro-F1, High-class recall, High-class precision,
    multiclass macro AUROC, and binary alert AUROC (Moderate/High vs Low).
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Class 2 is High risk
    rec_high = float(recall_score(y_true, y_pred, labels=[2], average="micro", zero_division=0))
    prec_high = float(precision_score(y_true, y_pred, labels=[2], average="micro", zero_division=0))

    # Multiclass Macro AUROC
    macro_auroc = np.nan
    if probs is not None and probs.shape[1] == 3:
        try:
            macro_auroc = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
        except Exception:
            macro_auroc = np.nan

    # Binary Alert AUROC (Moderate/High [1, 2] vs Low [0])
    y_alert_true = (y_true >= 1).astype(int)
    alert_auroc = np.nan

    if probs is not None and probs.shape[1] == 3:
        # Alert score is P(Moderate) + P(High) = 1 - P(Low)
        alert_score = 1.0 - probs[:, 0]
        try:
            alert_auroc = float(roc_auc_score(y_alert_true, alert_score))
        except Exception:
            alert_auroc = np.nan
    elif continuous_score is not None:
        try:
            alert_auroc = float(roc_auc_score(y_alert_true, continuous_score))
        except Exception:
            alert_auroc = np.nan

    return {
        "macro_f1": macro_f1,
        "acc": acc,
        "recall_high": rec_high,
        "prec_high": prec_high,
        "macro_auroc": macro_auroc,
        "alert_auroc": alert_auroc,
    }


def compute_early_detection_metrics(
    df_test: pd.DataFrame,
    test_episodes: List[List[HealthEpisode]],
    test_subject_ids: List[int],
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculates early detection rate in first 3 days of High-reaching episodes,
    chance rate on episode-free 3-day windows, and daily false-alert rate.
    """
    df_eval = df_test.copy()
    df_eval["pred_alert"] = (y_pred >= 1).astype(int)

    # 1. False alert rate per episode-free day (where latent_severity < 0.25)
    free_mask = df_eval["latent_severity"] < 0.25
    free_days = df_eval[free_mask]
    false_alert_rate = (
        float(free_days["pred_alert"].mean()) if len(free_days) > 0 else 0.0
    )

    # 2. Chance level: probability of alert in consecutive 3-day window of episode-free days
    # Empirically sampled across episode-free stretches
    chance_alerts = []
    for _, group in free_days.groupby("subject_id"):
        days = group["day"].to_numpy()
        alerts = group["pred_alert"].to_numpy()
        for i in range(len(days) - 2):
            if days[i+2] == days[i] + 2:  # 3 consecutive episode-free days
                has_alert = int(np.any(alerts[i:i+3]))
                chance_alerts.append(has_alert)
    
    if len(chance_alerts) > 0:
        chance_rate = float(np.mean(chance_alerts))
    else:
        # Theoretical chance: 1 - (1 - FAR)^3
        chance_rate = float(1.0 - (1.0 - false_alert_rate)**3)

    # 3. Early detection: High-reaching episodes (peak >= 0.60)
    detected_count = 0
    total_high_episodes = 0

    # Map subject_id to rows in df_eval
    subj_group_map = {sid: grp for sid, grp in df_eval.groupby("subject_id")}

    for sid in test_subject_ids:
        if sid >= len(test_episodes):
            continue
        ep_list = test_episodes[sid]
        subj_data = subj_group_map.get(sid)
        if subj_data is None:
            continue

        subj_day_alerts = dict(zip(subj_data["day"], subj_data["pred_alert"]))

        for ep in ep_list:
            if ep.peak_severity >= 0.60:
                total_high_episodes += 1
                # Check first 3 days from onset: [onset, onset+1, onset+2]
                onset = ep.onset_day
                flagged = False
                for d in range(onset, onset + 3):
                    if subj_day_alerts.get(d, 0) == 1:
                        flagged = True
                        break
                if flagged:
                    detected_count += 1

    detect_pct = (
        (detected_count / total_high_episodes) if total_high_episodes > 0 else 0.0
    )

    return {
        "detect_le_3d": float(detect_pct * 100.0),
        "chance_pct": float(chance_rate * 100.0),
        "false_alert_rate_pct": float(false_alert_rate * 100.0),
    }


def evaluate_single_cohort(
    cohort_id: int,
    seed: int,
    noise_mult: float = 1.0,
    missing_rate: float = 0.05,
) -> Dict[str, Any]:
    """Runs complete end-to-end evaluation on one synthetic cohort."""
    df_raw, episodes = simulate_cohort(
        cohort_id=cohort_id,
        seed=seed,
        noise_mult=noise_mult,
        missing_rate=missing_rate,
    )
    df_prep, artifacts = preprocess_cohort(df_raw, seed=seed)
    df_base = compute_all_baselines(
        df_prep, window_size=28, min_days=7, drop_first_days=True
    )

    train_mask = df_base["split"] == "train"
    val_mask = df_base["split"] == "val"
    test_mask = df_base["split"] == "test"

    df_train = df_base[train_mask].reset_index(drop=True)
    df_val = df_base[val_mask].reset_index(drop=True)
    df_test = df_base[test_mask].reset_index(drop=True)

    y_train = df_train["target"]
    y_val = df_val["target"]
    y_test = df_test["target"].to_numpy()

    matrices = get_feature_matrices(df_base)
    X_A, _ = matrices["A"]
    X_B, _ = matrices["B"]
    X_Z, _ = matrices["Z"]

    # 1. Train ML models on Set A
    models_A = train_all_models_for_feature_set(
        X_train=X_A[train_mask].reset_index(drop=True),
        y_train=y_train,
        X_val=X_A[val_mask].reset_index(drop=True),
        y_val=y_val,
        feature_set_name="A",
        feature_names=FEATURE_SET_A,
        random_state=seed,
    )

    # 2. Train ML models on Set B
    models_B = train_all_models_for_feature_set(
        X_train=X_B[train_mask].reset_index(drop=True),
        y_train=y_train,
        X_val=X_B[val_mask].reset_index(drop=True),
        y_val=y_val,
        feature_set_name="B",
        feature_names=FEATURE_SET_B,
        random_state=seed,
    )

    # 3. Rule-based scores
    # Population HRI
    hri_train_scores, _ = compute_population_hri(df_train)
    hri_test_scores, _ = compute_population_hri(df_test)
    hri_tuned_cutoffs = tune_cutoffs_on_train(hri_train_scores, y_train.to_numpy())

    # Personalised HRI
    phri_train_scores, _ = compute_personalised_hri(df_train)
    phri_test_scores, _ = compute_personalised_hri(df_test)
    phri_tuned_cutoffs = tune_cutoffs_on_train(phri_train_scores, y_train.to_numpy())

    # NEWS2
    news_train_scores = compute_news2_score(df_train)
    news_test_scores = compute_news2_score(df_test)
    news_tuned_cutoffs = tune_cutoffs_on_train(news_train_scores, y_train.to_numpy())

    # Compile predictions and metrics for Table VI & Table VII
    eval_results = {}
    early_results = {}
    preds_dict = {}

    # Define all 12 evaluated methods
    methods_config = [
        ("HRI, fixed cut-offs (0.33/0.66)", "rule", hri_test_scores, DEFAULT_HRI_CUTOFFS),
        ("HRI, tuned cut-offs", "rule", hri_test_scores, hri_tuned_cutoffs),
        ("NEWS2-style score, tuned", "rule", news_test_scores, news_tuned_cutoffs),
        ("Personalised HRI, tuned", "rule", phri_test_scores, phri_tuned_cutoffs),
        ("LR (A: population features)", "model", models_A["LR"], None),
        ("SVM (A: population features)", "model", models_A["SVM"], None),
        ("RF (A: population features)", "model", models_A["RF"], None),
        ("XGB (A: population features)", "model", models_A["XGB"], None),
        ("LR (B: + personal baseline)", "model", models_B["LR"], None),
        ("SVM (B: + personal baseline)", "model", models_B["SVM"], None),
        ("RF (B: + personal baseline)", "model", models_B["RF"], None),
        ("XGB (B: + personal baseline)", "model", models_B["XGB"], None),
    ]

    for m_name, m_type, m_obj, cutoffs in methods_config:
        if m_type == "rule":
            preds = discretize_score(m_obj, cutoffs)
            probs = None
            cont_score = m_obj
        else:
            bundle = m_obj
            X_test_input = X_A[test_mask] if bundle.feature_set_name == "A" else X_B[test_mask]
            preds = bundle.predict(X_test_input)
            probs = bundle.predict_proba(X_test_input)
            cont_score = None

        preds_dict[m_name] = preds
        metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=preds,
            probs=probs,
            continuous_score=cont_score,
        )
        eval_results[m_name] = metrics

        early = compute_early_detection_metrics(
            df_test=df_test,
            test_episodes=episodes,
            test_subject_ids=artifacts.test_subject_ids,
            y_pred=preds,
        )
        early_results[m_name] = early

    # 4. Feature ablation for XGBoost (Set A, Set Z, Set B)
    xgb_Z = train_and_tune_xgboost(
        X_train=X_Z[train_mask].reset_index(drop=True),
        y_train=y_train,
        X_val=X_Z[val_mask].reset_index(drop=True),
        y_val=y_val,
        feature_set_name="Z",
        feature_names=FEATURE_SET_Z,
        learning_rate=0.08,
        random_state=seed,
    )
    preds_Z = xgb_Z.predict(X_Z[test_mask])
    probs_Z = xgb_Z.predict_proba(X_Z[test_mask])
    metrics_Z = compute_classification_metrics(y_true=y_test, y_pred=preds_Z, probs=probs_Z)

    ablation_results = {
        "A: raw + demographics (population)": eval_results["XGB (A: population features)"],
        "Z: personal-baseline deviations only": metrics_Z,
        "B: A U Z (full)": eval_results["XGB (B: + personal baseline)"],
    }

    return {
        "cohort_id": cohort_id,
        "eval_results": eval_results,
        "early_results": early_results,
        "ablation_results": ablation_results,
        "y_test": y_test,
        "preds_dict": preds_dict,
        "models_B": models_B,
        "df_test": df_test,
        "hri_test_scores": hri_test_scores,
        "phri_test_scores": phri_test_scores,
    }


def aggregate_cohort_metrics(
    all_cohort_results: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes Mean +/- SD across 5 cohorts for Tables VI, VII, and VIII."""
    # 1. Aggregate Table VI
    methods = list(all_cohort_results[0]["eval_results"].keys())
    metric_keys = ["macro_f1", "acc", "recall_high", "prec_high", "macro_auroc", "alert_auroc"]

    t6_rows = []
    for m in methods:
        row = {"Method": m}
        for k in metric_keys:
            vals = [res["eval_results"][m][k] for res in all_cohort_results]
            mean_val = float(np.nanmean(vals))
            std_val = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            if np.isnan(mean_val):
                row[k] = "–"
            else:
                row[k] = f"{mean_val:.3f}±{std_val:.3f}"
        t6_rows.append(row)

    df_table_vi = pd.DataFrame(t6_rows).rename(
        columns={
            "macro_f1": "Macro-F1",
            "acc": "Acc.",
            "recall_high": "Recall_H",
            "prec_high": "Prec_H",
            "macro_auroc": "AUROC",
            "alert_auroc": "AUROC_alert",
        }
    )

    # 2. Aggregate Table VII
    t7_rows = []
    early_keys = ["detect_le_3d", "chance_pct", "false_alert_rate_pct"]
    for m in methods:
        row = {"Method": m}
        for k in early_keys:
            vals = [res["early_results"][m][k] for res in all_cohort_results]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[k] = f"{mean_val:.1f}±{std_val:.1f}"
        t7_rows.append(row)

    df_table_vii = pd.DataFrame(t7_rows).rename(
        columns={
            "detect_le_3d": "Detect<=3d (%)",
            "chance_pct": "Chance (%)",
            "false_alert_rate_pct": "False alerts/day (%)",
        }
    )

    # 3. Aggregate Table VIII (Ablation)
    ablation_names = list(all_cohort_results[0]["ablation_results"].keys())
    t8_rows = []
    for a in ablation_names:
        row = {"Feature set (XGBoost)": a}
        for k in ["macro_f1", "recall_high", "macro_auroc"]:
            vals = [res["ablation_results"][a][k] for res in all_cohort_results]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[k] = f"{mean_val:.3f}±{std_val:.3f}"
        t8_rows.append(row)

    df_table_viii = pd.DataFrame(t8_rows).rename(
        columns={
            "macro_f1": "Macro-F1",
            "recall_high": "Recall_H",
            "macro_auroc": "AUROC",
        }
    )

    return df_table_vi, df_table_vii, df_table_viii


def run_paired_t_tests(
    all_cohort_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Computes paired t-tests over the 5 cohorts comparing Set B vs Set A."""
    models = ["LR", "SVM", "RF", "XGB"]
    records = []

    for mod in models:
        a_name = f"{mod} (A: population features)"
        b_name = f"{mod} (B: + personal baseline)"

        f1_a = [res["eval_results"][a_name]["macro_f1"] for res in all_cohort_results]
        f1_b = [res["eval_results"][b_name]["macro_f1"] for res in all_cohort_results]

        diff = np.array(f1_b) - np.array(f1_a)
        res_ttest = stats.ttest_rel(f1_b, f1_a)

        records.append(
            {
                "Model": mod,
                "Mean Set A": float(np.mean(f1_a)),
                "Mean Set B": float(np.mean(f1_b)),
                "Mean Gain (B - A)": float(np.mean(diff)),
                "t-statistic": float(res_ttest.statistic),
                "p-value": float(res_ttest.pvalue),
            }
        )

    return pd.DataFrame(records)


def evaluate_robustness_single_cohort(
    cohort_id: int,
    seed: int,
    noise_mult: float = 1.0,
    missing_rate: float = 0.05,
) -> Dict[str, float]:
    """Lightweight evaluation strictly for Table IX: HRI tuned, Pers. HRI, XGB-A, XGB-B."""
    df_raw, _ = simulate_cohort(
        cohort_id=cohort_id,
        seed=seed,
        noise_mult=noise_mult,
        missing_rate=missing_rate,
    )
    df_prep, _ = preprocess_cohort(df_raw, seed=seed)
    df_base = compute_all_baselines(
        df_prep, window_size=28, min_days=7, drop_first_days=True
    )

    train_mask = df_base["split"] == "train"
    val_mask = df_base["split"] == "val"
    test_mask = df_base["split"] == "test"

    df_train = df_base[train_mask].reset_index(drop=True)
    df_val = df_base[val_mask].reset_index(drop=True)
    df_test = df_base[test_mask].reset_index(drop=True)

    y_train = df_train["target"]
    y_val = df_val["target"]
    y_test = df_test["target"].to_numpy()

    matrices = get_feature_matrices(df_base)
    X_A, _ = matrices["A"]
    X_B, _ = matrices["B"]

    # 1. Tuned HRI
    hri_train, _ = compute_population_hri(df_train)
    hri_test, _ = compute_population_hri(df_test)
    hri_cut = tune_cutoffs_on_train(hri_train, y_train.to_numpy())
    preds_hri = discretize_score(hri_test, hri_cut)
    f1_hri = float(f1_score(y_test, preds_hri, average="macro", zero_division=0))

    # 2. Tuned PHRI
    phri_train, _ = compute_personalised_hri(df_train)
    phri_test, _ = compute_personalised_hri(df_test)
    phri_cut = tune_cutoffs_on_train(phri_train, y_train.to_numpy())
    preds_phri = discretize_score(phri_test, phri_cut)
    f1_phri = float(f1_score(y_test, preds_phri, average="macro", zero_division=0))

    # 3. XGB-A
    xgb_a = train_and_tune_xgboost(
        X_train=X_A[train_mask].reset_index(drop=True),
        y_train=y_train,
        X_val=X_A[val_mask].reset_index(drop=True),
        y_val=y_val,
        feature_set_name="A",
        feature_names=FEATURE_SET_A,
        learning_rate=0.08,
        random_state=seed,
    )
    preds_xgba = xgb_a.predict(X_A[test_mask])
    f1_xgba = float(f1_score(y_test, preds_xgba, average="macro", zero_division=0))

    # 4. XGB-B
    xgb_b = train_and_tune_xgboost(
        X_train=X_B[train_mask].reset_index(drop=True),
        y_train=y_train,
        X_val=X_B[val_mask].reset_index(drop=True),
        y_val=y_val,
        feature_set_name="B",
        feature_names=FEATURE_SET_B,
        learning_rate=0.08,
        random_state=seed,
    )
    preds_xgbb = xgb_b.predict(X_B[test_mask])
    f1_xgbb = float(f1_score(y_test, preds_xgbb, average="macro", zero_division=0))

    return {
        "HRI tuned": f1_hri,
        "Pers. HRI": f1_phri,
        "XGB-A": f1_xgba,
        "XGB-B": f1_xgbb,
    }


def run_robustness_experiments(
    seeds: Optional[List[int]] = None,
    nominal_results: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Runs Table IX input degradation tests on 3 cohorts per condition:
    - nominal (noise x1, 5% missing)
    - sensor noise x2
    - sensor noise x3
    - 20% missing
    - 40% missing
    """
    if seeds is None:
        seeds = COHORT_SEEDS[:3]

    conditions = [
        ("nominal (noise x1, 5% missing)", 1.0, 0.05),
        ("sensor noise x2", 2.0, 0.05),
        ("sensor noise x3", 3.0, 0.05),
        ("20% missing", 1.0, 0.20),
        ("40% missing", 1.0, 0.40),
    ]

    table_ix_rows = []

    for cond_name, noise, miss in conditions:
        hri_f1s = []
        phri_f1s = []
        xgba_f1s = []
        xgbb_f1s = []

        if cond_name.startswith("nominal") and nominal_results is not None:
            for res in nominal_results[:len(seeds)]:
                ev = res["eval_results"]
                hri_f1s.append(ev["HRI, tuned cut-offs"]["macro_f1"])
                phri_f1s.append(ev["Personalised HRI, tuned"]["macro_f1"])
                xgba_f1s.append(ev["XGB (A: population features)"]["macro_f1"])
                xgbb_f1s.append(ev["XGB (B: + personal baseline)"]["macro_f1"])
        else:
            for c_idx, s in enumerate(seeds, start=1):
                res = evaluate_robustness_single_cohort(
                    cohort_id=c_idx, seed=s, noise_mult=noise, missing_rate=miss
                )
                hri_f1s.append(res["HRI tuned"])
                phri_f1s.append(res["Pers. HRI"])
                xgba_f1s.append(res["XGB-A"])
                xgbb_f1s.append(res["XGB-B"])

        table_ix_rows.append(
            {
                "Condition": cond_name,
                "HRI tuned": f"{np.mean(hri_f1s):.3f}±{np.std(hri_f1s, ddof=1):.3f}",
                "Pers. HRI": f"{np.mean(phri_f1s):.3f}±{np.std(phri_f1s, ddof=1):.3f}",
                "XGB-A": f"{np.mean(xgba_f1s):.3f}±{np.std(xgba_f1s, ddof=1):.3f}",
                "XGB-B": f"{np.mean(xgbb_f1s):.3f}±{np.std(xgbb_f1s, ddof=1):.3f}",
            }
        )

    return pd.DataFrame(table_ix_rows)


def plot_and_save_confusion_matrices(
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    save_path: str,
):
    """Generates a comparison grid of confusion matrices."""
    methods_to_plot = [
        "HRI, fixed cut-offs (0.33/0.66)",
        "HRI, tuned cut-offs",
        "XGB (A: population features)",
        "XGB (B: + personal baseline)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    for idx, name in enumerate(methods_to_plot):
        ax = axes[idx]
        y_p = preds_dict[name]
        cm = confusion_matrix(y_true, y_p, labels=[0, 1, 2])
        # Normalized by row (recall per class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_LABELS, fontsize=8)
        ax.set_yticklabels(CLASS_LABELS, fontsize=8)
        ax.set_xlabel("Predicted Class", fontsize=9)
        if idx == 0:
            ax.set_ylabel("True Class", fontsize=9)

        # Annotate counts
        for r in range(3):
            for c in range(3):
                val = cm[r, c]
                rate = cm_norm[r, c]
                color = "white" if rate > 0.5 else "black"
                ax.text(c, r, f"{val}\n({rate:.1%})", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

"""Stage 8: Interactive Streamlit Health Risk Analytics Dashboard.

Features:
- Prominent Non-Diagnostic Health Disclaimer Banner.
- Cohort and Subject Selector.
- Longitudinal vital signs time-series with personal baseline normal bands (median +/- 1.96*scale).
- Anomaly markers on days with significant deviations (|z| > 2.5).
- Composite Health Risk Index (HRI & PHRI) trajectories over time.
- Day-specific inspector with live SHAP risk factor decomposition.
- Early warning alerts triggered by elevated model risk (Moderate/High) or severe baseline deviation.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline import compute_all_baselines, get_feature_matrices
from src.config import (
    CLASS_LABELS,
    DAILY_VITALS,
    DEFAULT_HRI_CUTOFFS,
    FEATURE_SET_A,
    FEATURE_SET_B,
    HRI_CONFIG,
)
from src.data_gen import simulate_cohort
from src.explain import ModelExplainer
from src.hri import compute_personalised_hri, compute_population_hri, discretize_score
from src.models import train_and_tune_xgboost
from src.preprocess import preprocess_cohort

# Page layout and configuration
st.set_page_config(
    page_title="Wearable Preventive Health Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium medical-grade styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background: linear-gradient(90deg, #fff3cd, #ffeeba);
        border-left: 6px solid #ffc107;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #856404;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .risk-badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .risk-badge-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .risk-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Mandatory Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>RESEARCH & SIMULATION NOTICE — NOT A MEDICAL DIAGNOSIS TOOL</strong><br>
        All telemetry presented in this system is synthetically simulated. This software does NOT provide clinical diagnoses, 
        medical prescriptions, or clinical triage. Always consult qualified healthcare professionals for medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Wearable Preventive Health Analytics System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Causal Personalized Baseline Tracking, Multimodal Risk Scoring & Explainability</div>', unsafe_allow_html=True)


@st.cache_data
def load_processed_cohort(cohort_id: int, seed: int):
    """Loads, preprocesses, and trains reference XGBoost model on cached cohort data."""
    df_raw, episodes = simulate_cohort(cohort_id=cohort_id, seed=seed)
    df_prep, artifacts = preprocess_cohort(df_raw, seed=seed)
    df_base = compute_all_baselines(df_prep, window_size=28, min_days=7, drop_first_days=False)

    # Compute population and personalised HRI
    hri_scores, df_hri_contrib = compute_population_hri(df_base)
    phri_scores, df_phri_contrib = compute_personalised_hri(df_base)

    df_base["hri"] = hri_scores
    df_base["phri"] = phri_scores
    df_base["hri_risk"] = discretize_score(hri_scores, DEFAULT_HRI_CUTOFFS)

    # Train model on training split for live inference
    train_mask = df_base["split"] == "train"
    val_mask = df_base["split"] == "val"
    y_train = df_base.loc[train_mask, "target"]
    y_val = df_base.loc[val_mask, "target"]

    model_xgb = train_and_tune_xgboost(
        X_train=df_base[train_mask],
        y_train=y_train,
        X_val=df_base[val_mask],
        y_val=y_val,
        feature_set_name="B",
        feature_names=FEATURE_SET_B,
        learning_rate=0.08,
        random_state=seed,
    )

    # Evaluate predictions for all rows where day >= 7
    eval_mask = df_base["day"] >= 7
    all_preds = np.zeros(len(df_base), dtype=int)
    all_probs = np.zeros((len(df_base), 3), dtype=float)

    if eval_mask.sum() > 0:
        preds_eval = model_xgb.predict(df_base[eval_mask])
        probs_eval = model_xgb.predict_proba(df_base[eval_mask])
        all_preds[eval_mask] = preds_eval
        all_probs[eval_mask] = probs_eval

    df_base["pred_risk"] = all_preds
    df_base["prob_low"] = all_probs[:, 0]
    df_base["prob_moderate"] = all_probs[:, 1]
    df_base["prob_high"] = all_probs[:, 2]

    return df_base, episodes, model_xgb


# Sidebar: Subject and Cohort Selection
st.sidebar.header("Cohort & Subject Selector")
cohort_choice = st.sidebar.selectbox("Cohort ID", options=[1, 2, 3, 4, 5], index=0)
seed_choice = [42, 101, 202, 303, 404][cohort_choice - 1]

with st.spinner("Loading synthetic telemetry and running inference..."):
    df_cohort, episodes, model_bundle = load_processed_cohort(cohort_choice, seed_choice)

all_subject_ids = sorted(df_cohort["subject_id"].unique().tolist())
selected_subject_id = st.sidebar.selectbox(
    "Select Subject ID",
    options=all_subject_ids,
    index=0,
)

subject_data = df_cohort[df_cohort["subject_id"] == selected_subject_id].sort_values("day").reset_index(drop=True)
subj_age = int(subject_data["age"].iloc[0])
subj_bmi = float(subject_data["bmi"].iloc[0])
subj_split = subject_data["split"].iloc[0].upper()

# Sidebar summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Age:** {subj_age} years")
st.sidebar.markdown(f"**BMI:** {subj_bmi:.1f} kg/m²")
st.sidebar.markdown(f"**Split:** `{subj_split}`")

# Day selector for deep inspection
st.sidebar.markdown("---")
available_days = subject_data["day"].tolist()
valid_days = [d for d in available_days if d >= 7]
selected_day = st.sidebar.slider(
    "Inspection Day (t >= 7)",
    min_value=7,
    max_value=max(available_days),
    value=min(45, max(available_days)),
)

current_day_data = subject_data[subject_data["day"] == selected_day].iloc[0]

# Metrics row
c1, c2, c3, c4, c5 = st.columns(5)

risk_int = int(current_day_data["pred_risk"])
risk_label = CLASS_LABELS[risk_int]
badge_class = f"risk-badge-{risk_label.lower()}"

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b;">PREDICTED RISK (Day {selected_day})</span><br>
            <span class="{badge_class}" style="font-size:1.15rem;">{risk_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b;">HIGH RISK PROBABILITY</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">{current_day_data['prob_high']:.1%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b;">COMPOSITE HRI</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">{current_day_data['hri']:.3f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b;">PERSONALISED HRI (PHRI)</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">{current_day_data['phri']:.3f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <span style="font-size:0.85rem; color:#64748b;">LATENT SEVERITY e(t)</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#1e293b;">{current_day_data['latent_severity']:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Alerts section
alerts = []
if risk_int in [1, 2]:
    alerts.append(f"Model Alert: Predicted health risk level is **{risk_label}**.")

# Baseline deviations
for param in DAILY_VITALS:
    z_col = "z_log_hrv" if param == "hrv" else ("z_log_steps" if param == "steps" else f"z_{param}")
    if z_col in current_day_data:
        z_val = current_day_data[z_col]
        if abs(z_val) > 2.5:
            direction = "elevated" if z_val > 0 else "suppressed"
            alerts.append(f"Telemetry Alert: **{param.upper()}** is {direction} at **{abs(z_val):.1f} standard deviations** from baseline.")

if alerts:
    with st.expander("🚨 Active Health Alerts & Anomaly Warnings", expanded=True):
        for alt in alerts:
            st.warning(alt)
else:
    st.success("✅ Normal Telemetry: No elevated risk triggers or large baseline deviations detected on this day.")

# Tabs for detailed views
tab_timeseries, tab_hri, tab_explain = st.tabs(
    ["📈 Vital Signs & Personal Baseline Bands", "📊 HRI & Risk Progression", "🔍 Day-Level XAI Feature Attribution"]
)

with tab_timeseries:
    st.subheader("Longitudinal Vital Telemetry vs Causal Personal Baselines")
    st.caption("Solid blue line: Observed daily reading. Shaded blue region: Causal personal baseline band (median ± 1.96 * scale). Red points: Anomaly markers (|z| > 2.5).")

    vital_to_plot = st.selectbox(
        "Choose Vital Sign to Inspect",
        options=[
            ("hr", "Heart Rate (bpm)"),
            ("sbp", "Systolic Blood Pressure (mmHg)"),
            ("dbp", "Diastolic Blood Pressure (mmHg)"),
            ("spo2", "Oxygen Saturation SpO2 (%)"),
            ("temp", "Body Temperature (°C)"),
            ("rr", "Respiratory Rate (/min)"),
            ("hrv", "Heart Rate Variability (ms)"),
            ("sleep", "Sleep Duration (hours)"),
            ("steps", "Daily Step Count"),
            ("stress", "Stress Level (0-100)"),
        ],
        format_func=lambda x: x[1],
    )[0]

    eval_data = subject_data[subject_data["day"] >= 7].copy()
    days_arr = eval_data["day"].to_numpy()
    obs_vals = eval_data[vital_to_plot].to_numpy()

    med_col = "baseline_med_log_hrv" if vital_to_plot == "hrv" else ("baseline_med_log_steps" if vital_to_plot == "steps" else f"baseline_med_{vital_to_plot}")
    scale_col = "baseline_scale_log_hrv" if vital_to_plot == "hrv" else ("baseline_scale_log_steps" if vital_to_plot == "steps" else f"baseline_scale_{vital_to_plot}")
    z_col = "z_log_hrv" if vital_to_plot == "hrv" else ("z_log_steps" if vital_to_plot == "steps" else f"z_{vital_to_plot}")

    med_vals = eval_data[med_col].to_numpy()
    scale_vals = eval_data[scale_col].to_numpy()
    z_vals = eval_data[z_col].to_numpy()

    # Convert log-scale baselines back to physical units for display if needed
    if vital_to_plot in ["hrv", "steps"]:
        med_display = np.exp(med_vals)
        low_display = np.exp(med_vals - 1.96 * scale_vals)
        high_display = np.exp(med_vals + 1.96 * scale_vals)
    else:
        med_display = med_vals
        low_display = med_vals - 1.96 * scale_vals
        high_display = med_vals + 1.96 * scale_vals

    fig_vital, ax_v = plt.subplots(figsize=(10, 4), dpi=150)
    ax_v.plot(days_arr, obs_vals, label="Observed", color="#1f77b4", linewidth=1.8)
    ax_v.plot(days_arr, med_display, label="Personal Baseline Median", color="#333333", linestyle="--", linewidth=1.2)
    ax_v.fill_between(days_arr, low_display, high_display, color="#1f77b4", alpha=0.15, label="Normal Operating Band (±1.96 SD)")

    # Highlight anomalies
    anomaly_mask = np.abs(z_vals) > 2.5
    if np.any(anomaly_mask):
        ax_v.scatter(days_arr[anomaly_mask], obs_vals[anomaly_mask], color="#d9534f", s=40, zorder=5, label="Anomaly (|z| > 2.5)")

    # Vertical line at currently inspected day
    ax_v.axvline(selected_day, color="#d9534f", linestyle=":", linewidth=1.5, label=f"Selected Day {selected_day}")

    ax_v.set_xlabel("Day", fontsize=10)
    ax_v.set_ylabel(vital_to_plot.upper(), fontsize=10)
    ax_v.legend(loc="upper right", fontsize=8)
    ax_v.grid(True, linestyle="--", alpha=0.4)
    fig_vital.tight_layout()
    st.pyplot(fig_vital)

with tab_hri:
    st.subheader("Composite Health Risk Index (HRI & PHRI) Trajectories")
    st.caption("Comparison of continuous population HRI (fixed reference limits) and Personalised HRI (baseline deviation based).")

    eval_data = subject_data[subject_data["day"] >= 7].copy()
    days_arr = eval_data["day"].to_numpy()

    fig_hri, ax_h = plt.subplots(figsize=(10, 4), dpi=150)
    ax_h.plot(days_arr, eval_data["hri"], label="Population HRI", color="#ff7f0e", linewidth=1.8)
    ax_h.plot(days_arr, eval_data["phri"], label="Personalised HRI (PHRI)", color="#2ca02c", linewidth=1.8)

    # Reference cutoffs
    ax_h.axhline(0.33, color="gray", linestyle=":", alpha=0.7, label="Default Moderate Cutoff (0.33)")
    ax_h.axhline(0.66, color="red", linestyle=":", alpha=0.7, label="Default High Cutoff (0.66)")
    ax_h.axvline(selected_day, color="#d9534f", linestyle=":", linewidth=1.5)

    ax_h.set_ylim(-0.05, 1.05)
    ax_h.set_xlabel("Day", fontsize=10)
    ax_h.set_ylabel("Risk Index [0, 1]", fontsize=10)
    ax_h.legend(loc="upper right", fontsize=8)
    ax_h.grid(True, linestyle="--", alpha=0.4)
    fig_hri.tight_layout()
    st.pyplot(fig_hri)

with tab_explain:
    st.subheader(f"Explainable AI: Risk Factor Breakdown (Day {selected_day})")
    st.caption("SHAP feature contributions decomposing model predictions. Red bars elevate risk towards High; Green bars suppress risk towards Low.")

    explainer = ModelExplainer(model_bundle)
    fig_xai = explainer.plot_instance_attribution(current_day_data, target_class=2, top_k=10)
    st.pyplot(fig_xai)

    # Table of local parameter readings vs personal baseline
    st.markdown("#### Parameter Comparison Table")
    table_rows = []
    for param in DAILY_VITALS:
        z_col = "z_log_hrv" if param == "hrv" else ("z_log_steps" if param == "steps" else f"z_{param}")
        med_col = "baseline_med_log_hrv" if param == "hrv" else ("baseline_med_log_steps" if param == "steps" else f"baseline_med_{param}")
        
        obs = current_day_data[param]
        z_score = current_day_data.get(z_col, np.nan)
        med_val = current_day_data.get(med_col, np.nan)
        if param in ["hrv", "steps"]:
            med_val = np.exp(med_val)

        table_rows.append(
            {
                "Parameter": param.upper(),
                "Observed Reading": f"{obs:.1f}",
                "Personal Baseline": f"{med_val:.1f}" if not np.isnan(med_val) else "–",
                "Deviation (z-score)": f"{z_score:+.2f}" if not np.isnan(z_score) else "–",
                "Status": "⚠️ Abnormal" if abs(z_score) > 2.5 else "Normal",
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

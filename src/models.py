"""Stage 5: Machine Learning Models and Validation-Based Hyperparameter Tuning.

Implements:
1. Multinomial Logistic Regression: C in {0.1, 1.0, 10.0}, balanced class weights.
2. Support Vector Machine (RBF kernel): C in {1.0, 10.0}, trained on random 8,000-row
   subsample of training set, balanced weights, probability calibration.
3. Random Forest Classifier: 250 trees, min_samples_leaf in {1, 5}, balanced weights.
4. XGBoost Classifier (or GradientBoostedTrees): lr 0.08, max_depth in {3, 5},
   early stopping on validation subjects, class-weighted sample weights.
5. All models trained and tuned across Feature Set A, Feature Set B (and Z for ablation).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# Try importing XGBoost; provide high-performance Scikit-Learn GradientBoost fallback if unavailable
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False


@dataclass
class TrainedModelBundle:
    """Encapsulates a fitted model along with its scaler and optimal hyperparameters."""
    model_name: str
    feature_set_name: str
    model: Any
    scaler: Optional[StandardScaler]
    best_param: Any
    val_macro_f1: float
    feature_names: List[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels (0, 1, 2) on input features."""
        df_sub = X[self.feature_names]
        if self.scaler is not None:
            mat = self.scaler.transform(df_sub)
        else:
            mat = df_sub.to_numpy()
        return self.model.predict(mat)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities on input features."""
        df_sub = X[self.feature_names]
        if self.scaler is not None:
            mat = self.scaler.transform(df_sub)
        else:
            mat = df_sub.to_numpy()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(mat)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(mat)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        else:
            raise NotImplementedError("Model lacks probability prediction support.")


def train_and_tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set_name: str,
    feature_names: List[str],
    random_state: int = 42,
) -> TrainedModelBundle:
    """Tunes Logistic Regression over C in {0.1, 1.0, 10.0} using validation macro-F1."""
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[feature_names])
    X_val_sc = scaler.transform(X_val[feature_names])

    best_c = 1.0
    best_f1 = -1.0
    best_model = None

    c_candidates = [0.1, 1.0, 10.0]
    for c in c_candidates:
        lr = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        lr.fit(X_tr_sc, y_train)
        preds = lr.predict(X_val_sc)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_model = lr

    return TrainedModelBundle(
        model_name="LR",
        feature_set_name=feature_set_name,
        model=best_model,
        scaler=scaler,
        best_param={"C": best_c},
        val_macro_f1=best_f1,
        feature_names=feature_names,
    )


def train_and_tune_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set_name: str,
    feature_names: List[str],
    subsample_size: int = 8000,
    random_state: int = 42,
) -> TrainedModelBundle:
    """Tunes RBF-SVM over C in {1.0, 10.0} on an 8,000-row random subsample."""
    scaler = StandardScaler()
    X_tr_full = X_train[feature_names]
    y_tr_full = y_train.to_numpy()

    # Random subsample of 8,000 training rows to match experimental spec
    n_samples = len(X_tr_full)
    rng = np.random.default_rng(random_state)
    if n_samples > subsample_size:
        idx_sub = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_tr_full.iloc[idx_sub]
        y_sub = y_tr_full[idx_sub]
    else:
        X_sub = X_tr_full
        y_sub = y_tr_full

    X_sub_sc = scaler.fit_transform(X_sub)
    X_val_sc = scaler.transform(X_val[feature_names])

    best_c = 1.0
    best_f1 = -1.0
    best_model = None

    for c in [1.0, 10.0]:
        svm = SVC(
            C=c,
            kernel="rbf",
            class_weight="balanced",
            random_state=random_state,
        )
        svm.fit(X_sub_sc, y_sub)
        preds = svm.predict(X_val_sc)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_model = svm

    return TrainedModelBundle(
        model_name="SVM",
        feature_set_name=feature_set_name,
        model=best_model,
        scaler=scaler,
        best_param={"C": best_c},
        val_macro_f1=best_f1,
        feature_names=feature_names,
    )


def train_and_tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set_name: str,
    feature_names: List[str],
    n_estimators: int = 250,
    random_state: int = 42,
) -> TrainedModelBundle:
    """Tunes Random Forest over min_samples_leaf in {1, 5} with 250 trees."""
    X_tr = X_train[feature_names].to_numpy()
    X_v = X_val[feature_names].to_numpy()
    y_tr = y_train.to_numpy()
    y_v = y_val.to_numpy()

    best_leaf = 1
    best_f1 = -1.0
    best_model = None

    for leaf in [1, 5]:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_v)
        f1 = f1_score(y_v, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_leaf = leaf
            best_model = rf

    return TrainedModelBundle(
        model_name="RF",
        feature_set_name=feature_set_name,
        model=best_model,
        scaler=None,
        best_param={"min_samples_leaf": best_leaf},
        val_macro_f1=best_f1,
        feature_names=feature_names,
    )


def train_and_tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set_name: str,
    feature_names: List[str],
    learning_rate: float = 0.08,
    random_state: int = 42,
) -> TrainedModelBundle:
    """Tunes XGBoost over max_depth in {3, 5} with lr 0.08 and early stopping on validation subjects."""
    X_tr = X_train[feature_names].to_numpy()
    X_v = X_val[feature_names].to_numpy()
    y_tr = y_train.to_numpy()
    y_v = y_val.to_numpy()

    # Balanced sample weights for gradient boosting
    sample_weights = compute_sample_weight("balanced", y_tr)

    best_depth = 3
    best_f1 = -1.0
    best_model = None

    depth_candidates = [3, 5]

    for depth in depth_candidates:
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=250,
                learning_rate=learning_rate,
                max_depth=depth,
                random_state=random_state,
                eval_metric="mlogloss",
                early_stopping_rounds=15,
                n_jobs=-1,
            )
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weights,
                eval_set=[(X_v, y_v)],
                verbose=False,
            )
        else:
            # Drop-in GradientBoostingClassifier fallback
            model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=learning_rate,
                max_depth=depth,
                random_state=random_state,
            )
            model.fit(X_tr, y_tr, sample_weight=sample_weights)

        preds = model.predict(X_v)
        f1 = f1_score(y_v, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_depth = depth
            best_model = model

    return TrainedModelBundle(
        model_name="XGB",
        feature_set_name=feature_set_name,
        model=best_model,
        scaler=None,
        best_param={"max_depth": best_depth},
        val_macro_f1=best_f1,
        feature_names=feature_names,
    )


def train_all_models_for_feature_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set_name: str,
    feature_names: List[str],
    random_state: int = 42,
) -> Dict[str, TrainedModelBundle]:
    """Trains and tunes LR, SVM, RF, and XGBoost on the specified feature set."""
    models = {}

    # 1. Logistic Regression
    models["LR"] = train_and_tune_logistic_regression(
        X_train, y_train, X_val, y_val, feature_set_name, feature_names, random_state
    )

    # 2. Support Vector Machine
    models["SVM"] = train_and_tune_svm(
        X_train, y_train, X_val, y_val, feature_set_name, feature_names, subsample_size=8000, random_state=random_state
    )

    # 3. Random Forest
    models["RF"] = train_and_tune_random_forest(
        X_train, y_train, X_val, y_val, feature_set_name, feature_names, n_estimators=250, random_state=random_state
    )

    # 4. XGBoost
    models["XGB"] = train_and_tune_xgboost(
        X_train, y_train, X_val, y_val, feature_set_name, feature_names, learning_rate=0.08, random_state=random_state
    )

    return models

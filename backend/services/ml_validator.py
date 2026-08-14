"""
MediSynth.AI — Machine Learning Utility Validation
Demonstrates synthetic data utility via TSTR (Train on Synthetic, Test on Real).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.config import ML_TEST_SPLIT, ML_RANDOM_STATE, ML_N_ESTIMATORS
from backend.utils.logging_config import get_logger

logger = get_logger("ml_validator")


def _prepare_ml_data(df: pd.DataFrame, target_col: str):
    """
    Prepare data for ML: encode categoricals, handle missing values,
    split features/target.
    """
    data = df.copy()

    # Encode target
    le = LabelEncoder()
    if data[target_col].dtype == "object":
        data[target_col] = le.fit_transform(data[target_col].astype(str))
    else:
        le = None

    y = data[target_col].values

    # Features: drop target and non-numeric after encoding
    feature_cols = [c for c in data.columns if c != target_col]
    X = data[feature_cols].copy()

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "category"]).columns:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X.values, y, feature_cols, le


def _train_and_evaluate(X_train, y_train, X_test, y_test,
                         model_name: str = "RandomForest") -> Dict:
    """Train a classifier and return evaluation metrics."""
    if model_name == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=ML_N_ESTIMATORS,
            random_state=ML_RANDOM_STATE,
            max_depth=5,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS,
            random_state=ML_RANDOM_STATE,
            max_depth=10,
            n_jobs=-1,
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # ROC-AUC (handle multi-class)
    try:
        y_prob = model.predict_proba(X_test_scaled)
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, y_prob[:, 1]))
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_data = {
                "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 50)]],
                "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 50)]],
            }
        else:
            auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
            roc_data = None
    except Exception:
        auc = None
        roc_data = None

    # Feature importance
    importances = model.feature_importances_
    top_features = sorted(
        range(len(importances)),
        key=lambda i: importances[i],
        reverse=True,
    )[:10]

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "roc_curve": roc_data,
        "top_feature_indices": top_features,
        "feature_importances": [round(float(importances[i]), 4) for i in top_features],
    }


def _auto_detect_target(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect a suitable target column for classification."""
    # Look for common healthcare target columns
    priority_targets = [
        "heart_disease", "diabetes", "hypertension", "diagnosis",
        "outcome", "target", "label", "class", "disease",
        "readmission", "mortality", "stroke",
    ]
    for target in priority_targets:
        for col in df.columns:
            if col.lower() == target.lower():
                return col

    # Fallback: find binary/low-cardinality columns
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    for col in df.columns:
        if 2 < df[col].nunique() <= 10:
            return col

    return None


def validate_ml_utility(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                         target_col: Optional[str] = None) -> Dict:
    """
    Full ML utility validation via TSTR benchmark.

    1. Train on REAL → Test on REAL (baseline)
    2. Train on SYNTHETIC → Test on REAL (TSTR)
    3. Compare metrics and compute utility gap

    Returns comprehensive ML comparison report.
    """
    # Auto-detect target if not provided
    if target_col is None:
        target_col = _auto_detect_target(real_df)
        if target_col is None:
            return {
                "error": "Could not auto-detect target column. Please specify one.",
                "available_columns": list(real_df.columns),
            }

    if target_col not in real_df.columns or target_col not in synth_df.columns:
        return {"error": f"Target column '{target_col}' not found in data."}

    logger.info(f"ML validation with target column: {target_col}")

    # Align columns
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    real = real_df[common_cols].copy()
    synth = synth_df[common_cols].copy()

    # Prepare data
    try:
        X_real, y_real, feature_names, le = _prepare_ml_data(real, target_col)
        X_synth, y_synth, _, _ = _prepare_ml_data(synth, target_col)
    except Exception as e:
        return {"error": f"Data preparation failed: {str(e)}"}

    # Split real data for testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=ML_TEST_SPLIT, random_state=ML_RANDOM_STATE,
        stratify=y_real if len(np.unique(y_real)) > 1 else None,
    )

    results = {}

    # Benchmark 1: Train on Real, Test on Real (TRTR)
    logger.info("Training TRTR baseline (RandomForest)...")
    results["trtr_rf"] = _train_and_evaluate(
        X_real_train, y_real_train, X_real_test, y_real_test, "RandomForest"
    )

    # Benchmark 2: Train on Synthetic, Test on Real (TSTR)
    logger.info("Training TSTR (RandomForest)...")
    results["tstr_rf"] = _train_and_evaluate(
        X_synth, y_synth, X_real_test, y_real_test, "RandomForest"
    )

    # Benchmark 3: TRTR with GradientBoosting
    logger.info("Training TRTR baseline (GradientBoosting)...")
    results["trtr_gb"] = _train_and_evaluate(
        X_real_train, y_real_train, X_real_test, y_real_test, "GradientBoosting"
    )

    # Benchmark 4: TSTR with GradientBoosting
    logger.info("Training TSTR (GradientBoosting)...")
    results["tstr_gb"] = _train_and_evaluate(
        X_synth, y_synth, X_real_test, y_real_test, "GradientBoosting"
    )

    # Compute utility gaps
    def _gap(metric):
        trtr = results["trtr_rf"].get(metric)
        tstr = results["tstr_rf"].get(metric)
        if trtr is not None and tstr is not None:
            return round(abs(trtr - tstr), 4)
        return None

    utility_gaps = {
        "accuracy_gap": _gap("accuracy"),
        "f1_gap": _gap("f1_score"),
        "auc_gap": _gap("roc_auc"),
        "precision_gap": _gap("precision"),
        "recall_gap": _gap("recall"),
    }

    # Utility score (0-100): lower gap = higher utility
    gap_values = [v for v in utility_gaps.values() if v is not None]
    avg_gap = np.mean(gap_values) if gap_values else 0.5
    utility_score = round(max(0, 100 * (1 - avg_gap * 2)), 2)

    report = {
        "target_column": target_col,
        "target_classes": int(len(np.unique(y_real))),
        "real_train_size": len(X_real_train),
        "real_test_size": len(X_real_test),
        "synthetic_size": len(X_synth),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "results": results,
        "utility_gaps": utility_gaps,
        "utility_score": utility_score,
        "utility_grade": (
            "A" if utility_score >= 90 else
            "B" if utility_score >= 75 else
            "C" if utility_score >= 60 else
            "D" if utility_score >= 40 else "F"
        ),
        "comparison_chart": {
            "models": ["RandomForest", "GradientBoosting"],
            "metrics": ["accuracy", "f1_score", "roc_auc"],
            "real_scores": {
                "RandomForest": [
                    results["trtr_rf"]["accuracy"],
                    results["trtr_rf"]["f1_score"],
                    results["trtr_rf"]["roc_auc"],
                ],
                "GradientBoosting": [
                    results["trtr_gb"]["accuracy"],
                    results["trtr_gb"]["f1_score"],
                    results["trtr_gb"]["roc_auc"],
                ],
            },
            "synth_scores": {
                "RandomForest": [
                    results["tstr_rf"]["accuracy"],
                    results["tstr_rf"]["f1_score"],
                    results["tstr_rf"]["roc_auc"],
                ],
                "GradientBoosting": [
                    results["tstr_gb"]["accuracy"],
                    results["tstr_gb"]["f1_score"],
                    results["tstr_gb"]["roc_auc"],
                ],
            },
        },
    }

    logger.info(f"ML validation complete. Utility: {utility_score}/100 ({report['utility_grade']})")
    return report

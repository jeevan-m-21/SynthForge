"""
MediSynth.AI — Privacy Attack Simulation
Simulates membership inference, re-identification, and attribute inference attacks
to validate privacy guarantees of synthetic data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

from backend.config import ML_RANDOM_STATE
from backend.utils.logging_config import get_logger

logger = get_logger("attack_simulator")


def _encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Encode all columns to numeric for distance computations."""
    result = df.copy()
    for col in result.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        result[col] = le.fit_transform(result[col].astype(str))
    result = result.fillna(0)
    return result.values.astype(float)


# ──────────────────────────────────────────────
# 1. Membership Inference Attack (MIA)
# ──────────────────────────────────────────────
def membership_inference_attack(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    n_neighbors: int = 5,
) -> Dict:
    """
    Membership Inference Attack (MIA).

    Determines whether specific records from the training set can be
    identified as members by analyzing their proximity to synthetic data.

    Approach:
    1. For each real record (member or non-member), compute distance-based
       features to the synthetic dataset.
    2. Train a binary classifier to distinguish members from non-members.
    3. High attack AUC → synthetic data leaks membership information.

    Args:
        real_train_df: Records used to train the generator (MEMBERS)
        real_holdout_df: Records NOT used in training (NON-MEMBERS)
        synth_df: Generated synthetic data

    Returns:
        Attack metrics including AUC, accuracy, and risk level.
    """
    # Align columns
    common_cols = [c for c in real_train_df.columns
                   if c in synth_df.columns and c in real_holdout_df.columns]

    train_enc = _encode_dataframe(real_train_df[common_cols])
    holdout_enc = _encode_dataframe(real_holdout_df[common_cols])
    synth_enc = _encode_dataframe(synth_df[common_cols])

    # Normalize
    scaler = StandardScaler()
    synth_scaled = scaler.fit_transform(synth_enc)
    train_scaled = scaler.transform(train_enc)
    holdout_scaled = scaler.transform(holdout_enc)

    # Fit nearest neighbors on synthetic data
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(synth_scaled)), metric="euclidean")
    nn.fit(synth_scaled)

    # Compute features for members
    train_distances, _ = nn.kneighbors(train_scaled)
    train_features = np.column_stack([
        train_distances.min(axis=1),
        train_distances.mean(axis=1),
        train_distances.std(axis=1),
        train_distances.max(axis=1),
    ])

    # Compute features for non-members
    holdout_distances, _ = nn.kneighbors(holdout_scaled)
    holdout_features = np.column_stack([
        holdout_distances.min(axis=1),
        holdout_distances.mean(axis=1),
        holdout_distances.std(axis=1),
        holdout_distances.max(axis=1),
    ])

    # Build attack dataset
    X_attack = np.vstack([train_features, holdout_features])
    y_attack = np.concatenate([
        np.ones(len(train_features)),   # members
        np.zeros(len(holdout_features)), # non-members
    ])

    # Train attack classifier
    X_att_train, X_att_test, y_att_train, y_att_test = train_test_split(
        X_attack, y_attack, test_size=0.3, random_state=ML_RANDOM_STATE,
        stratify=y_attack,
    )

    attack_model = RandomForestClassifier(
        n_estimators=100, random_state=ML_RANDOM_STATE, n_jobs=-1
    )
    attack_model.fit(X_att_train, y_att_train)

    y_pred = attack_model.predict(X_att_test)
    y_prob = attack_model.predict_proba(X_att_test)[:, 1]

    accuracy = float(accuracy_score(y_att_test, y_pred))
    auc = float(roc_auc_score(y_att_test, y_prob))

    fpr, tpr, _ = roc_curve(y_att_test, y_prob)

    # Risk assessment
    # AUC ≈ 0.5 → attack fails (good privacy)
    # AUC > 0.6 → moderate risk
    # AUC > 0.75 → high risk
    advantage = max(0, (auc - 0.5) * 2)  # scale 0-1
    risk_score = round(advantage * 100, 2)

    if auc <= 0.55:
        risk_level = "low"
    elif auc <= 0.65:
        risk_level = "medium"
    elif auc <= 0.75:
        risk_level = "high"
    else:
        risk_level = "critical"

    return {
        "attack_type": "membership_inference",
        "attack_accuracy": round(accuracy, 4),
        "attack_auc": round(auc, 4),
        "attack_advantage": round(advantage, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "baseline_auc": 0.5,
        "n_members": len(train_enc),
        "n_non_members": len(holdout_enc),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 50)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 50)]],
        },
        "interpretation": (
            f"Attack AUC={auc:.3f} (baseline=0.5). "
            f"{'The attacker cannot reliably distinguish members from non-members. Privacy is well-protected.' if auc < 0.6 else 'The attacker has some ability to infer membership. Consider increasing privacy (lower ε).'}"
        ),
    }


# ──────────────────────────────────────────────
# 2. Re-Identification Attack
# ──────────────────────────────────────────────
def reidentification_attack(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    threshold_percentile: float = 5.0,
) -> Dict:
    """
    Re-Identification Attack.

    For each synthetic record, find the nearest real record and assess
    whether the synthetic data is "too close" to real individuals.

    A record is considered re-identifiable if its nearest-neighbor distance
    to the real data falls below a danger threshold.

    Args:
        real_df: Original real data
        synth_df: Generated synthetic data
        threshold_percentile: percentile of inter-real distances to use as threshold

    Returns:
        Re-identification risk metrics.
    """
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    real_enc = _encode_dataframe(real_df[common_cols])
    synth_enc = _encode_dataframe(synth_df[common_cols])

    # Normalize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_enc)
    synth_scaled = scaler.transform(synth_enc)

    # Compute distances from synthetic to real
    nn_real = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_real.fit(real_scaled)
    synth_to_real_dist, synth_to_real_idx = nn_real.kneighbors(synth_scaled)
    synth_to_real_dist = synth_to_real_dist.flatten()

    # Compute inter-real distances for baseline
    nn_self = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn_self.fit(real_scaled)
    real_self_dist, _ = nn_self.kneighbors(real_scaled)
    real_self_dist = real_self_dist[:, 1]  # skip distance to self

    # Danger threshold: records closer than this are "re-identifiable"
    threshold = float(np.percentile(real_self_dist, threshold_percentile))

    # Count records at risk
    at_risk = int(np.sum(synth_to_real_dist < threshold))
    at_risk_pct = round(at_risk / len(synth_to_real_dist) * 100, 2)

    # Distance statistics
    mean_dist = float(np.mean(synth_to_real_dist))
    median_dist = float(np.median(synth_to_real_dist))
    min_dist = float(np.min(synth_to_real_dist))

    # Risk score (0-100)
    risk_score = min(100, round(at_risk_pct * 2, 2))

    if at_risk_pct <= 1:
        risk_level = "low"
    elif at_risk_pct <= 5:
        risk_level = "medium"
    elif at_risk_pct <= 15:
        risk_level = "high"
    else:
        risk_level = "critical"

    # Distance distribution for visualization
    hist_bins = 30
    all_dists = np.concatenate([synth_to_real_dist, real_self_dist])
    bins = np.linspace(0, np.percentile(all_dists, 95), hist_bins + 1)
    synth_hist, _ = np.histogram(synth_to_real_dist, bins=bins)
    real_hist, _ = np.histogram(real_self_dist, bins=bins)
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    return {
        "attack_type": "reidentification",
        "records_at_risk": at_risk,
        "records_at_risk_pct": at_risk_pct,
        "total_synthetic_records": len(synth_to_real_dist),
        "threshold_distance": round(threshold, 4),
        "mean_distance": round(mean_dist, 4),
        "median_distance": round(median_dist, 4),
        "min_distance": round(min_dist, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "distance_distribution": {
            "bins": [round(b, 4) for b in bin_centers],
            "synth_to_real": [int(x) for x in synth_hist],
            "real_to_real": [int(x) for x in real_hist],
        },
        "interpretation": (
            f"{at_risk_pct}% of synthetic records are dangerously close to real records. "
            f"{'Privacy is well-protected.' if at_risk_pct < 5 else 'Consider increasing DP noise (lower ε) to improve protection.'}"
        ),
    }


# ──────────────────────────────────────────────
# 3. Attribute Inference Attack
# ──────────────────────────────────────────────
def attribute_inference_attack(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    sensitive_columns: Optional[List[str]] = None,
) -> Dict:
    """
    Attribute Inference Attack.

    Tests whether a sensitive attribute can be inferred from the remaining
    synthetic columns. If the synthetic data preserves too-precise relationships,
    an attacker could predict hidden attributes.

    Approach:
    1. For each sensitive column, train a model on synthetic data (minus that column)
       to predict the sensitive column.
    2. Evaluate prediction accuracy on real data.
    3. Compare against majority-class baseline (what a trivial predictor achieves).
    4. Advantage = attack_accuracy - majority_baseline.

    Args:
        real_df: Original real data
        synth_df: Generated synthetic data
        sensitive_columns: Columns to test (auto-detected if None)

    Returns:
        Per-column attribute inference metrics.
    """
    common_cols = [c for c in real_df.columns if c in synth_df.columns]

    # Auto-detect sensitive columns
    if sensitive_columns is None:
        sensitive_keywords = [
            "disease", "diagnosis", "diabetes", "hypertension", "smoking",
            "heart", "cancer", "hiv", "mental", "gender", "race", "ethnicity",
            "income", "insurance", "ssn",
        ]
        sensitive_columns = []
        for col in common_cols:
            if any(kw in col.lower() for kw in sensitive_keywords):
                sensitive_columns.append(col)
        # Add any binary columns
        for col in common_cols:
            if real_df[col].nunique() <= 5 and col not in sensitive_columns:
                sensitive_columns.append(col)
                if len(sensitive_columns) >= 5:
                    break

    if not sensitive_columns:
        sensitive_columns = common_cols[:3]

    column_results = {}
    total_advantage = 0

    for target_col in sensitive_columns:
        if target_col not in common_cols:
            continue

        try:
            # Prepare data
            feature_cols = [c for c in common_cols if c != target_col]
            synth_features = synth_df[feature_cols].copy()
            synth_target = synth_df[target_col].copy()
            real_features = real_df[feature_cols].copy()
            real_target = real_df[target_col].copy()

            # Encode categorical features
            for col in synth_features.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                all_vals = pd.concat([synth_features[col], real_features[col]]).astype(str)
                le.fit(all_vals)
                synth_features[col] = le.transform(synth_features[col].astype(str))
                real_features[col] = le.transform(real_features[col].astype(str))

            synth_features = synth_features.fillna(0)
            real_features = real_features.fillna(0)

            # Encode target
            le_target = LabelEncoder()
            all_targets = pd.concat([synth_target, real_target]).astype(str)
            le_target.fit(all_targets)
            synth_target_enc = le_target.transform(synth_target.astype(str))
            real_target_enc = le_target.transform(real_target.astype(str))

            n_classes = len(le_target.classes_)

            # Proper baseline: majority class proportion (what a trivial predictor gets)
            class_counts = np.bincount(real_target_enc)
            majority_baseline = float(class_counts.max()) / float(class_counts.sum())

            # Train on synthetic, predict on real
            # Use shallow trees (max_depth=3) to simulate a realistic attacker
            # who doesn't have perfect knowledge of the data distribution
            model = RandomForestClassifier(
                n_estimators=50, max_depth=3, min_samples_leaf=10,
                random_state=ML_RANDOM_STATE, n_jobs=-1
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(synth_features.values)
            X_test = scaler.transform(real_features.values)

            model.fit(X_train, synth_target_enc)
            y_pred = model.predict(X_test)

            accuracy = float(accuracy_score(real_target_enc, y_pred))

            # Advantage over majority-class baseline
            advantage = max(0, accuracy - majority_baseline)

            column_results[target_col] = {
                "accuracy": round(accuracy, 4),
                "baseline_accuracy": round(majority_baseline, 4),
                "advantage": round(advantage, 4),
                "n_classes": n_classes,
                "vulnerable": advantage > 0.15,
            }
            total_advantage += advantage

        except Exception as e:
            column_results[target_col] = {
                "error": str(e),
                "vulnerable": False,
            }

    # Overall risk
    n_tested = len([r for r in column_results.values() if "error" not in r])
    avg_advantage = total_advantage / max(n_tested, 1)
    risk_score = min(100, round(avg_advantage * 150, 2))

    # Thresholds based on literature: <8% advantage is negligible
    if avg_advantage <= 0.08:
        risk_level = "low"
    elif avg_advantage <= 0.20:
        risk_level = "medium"
    elif avg_advantage <= 0.35:
        risk_level = "high"
    else:
        risk_level = "critical"

    return {
        "attack_type": "attribute_inference",
        "columns_tested": list(column_results.keys()),
        "column_results": column_results,
        "average_advantage": round(avg_advantage, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "interpretation": (
            f"Average inference advantage: {avg_advantage:.3f} over majority-class baseline. "
            f"{'Attribute privacy is well-protected.' if avg_advantage < 0.1 else 'Some attributes may be inferable. Consider reducing correlations or increasing DP noise.'}"
        ),
    }


# ──────────────────────────────────────────────
# Combined Attack Report
# ──────────────────────────────────────────────
def run_all_attacks(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    holdout_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Run all three privacy attacks and produce a combined risk report.

    If holdout_df is not provided, the real data is split 70/30 into
    train/holdout for the membership inference attack.
    """
    # Split real data if no holdout provided
    if holdout_df is None:
        train_df, holdout_df = train_test_split(
            real_df, test_size=0.3, random_state=ML_RANDOM_STATE,
        )
    else:
        train_df = real_df

    logger.info("Running Membership Inference Attack...")
    mia = membership_inference_attack(train_df, holdout_df, synth_df)

    logger.info("Running Re-Identification Attack...")
    reid = reidentification_attack(real_df, synth_df)

    logger.info("Running Attribute Inference Attack...")
    attr = attribute_inference_attack(real_df, synth_df)

    # Overall privacy risk score (weighted average)
    weights = {"mia": 0.4, "reid": 0.35, "attr": 0.25}
    overall_score = (
        mia["risk_score"] * weights["mia"] +
        reid["risk_score"] * weights["reid"] +
        attr["risk_score"] * weights["attr"]
    )
    overall_score = round(overall_score, 2)

    if overall_score <= 15:
        overall_level = "low"
    elif overall_score <= 35:
        overall_level = "medium"
    elif overall_score <= 60:
        overall_level = "high"
    else:
        overall_level = "critical"

    return {
        "overall_risk_score": overall_score,
        "overall_risk_level": overall_level,
        "attacks": {
            "membership_inference": mia,
            "reidentification": reid,
            "attribute_inference": attr,
        },
        "radar_chart": {
            "labels": ["Membership\nInference", "Re-identification", "Attribute\nInference"],
            "risk_scores": [mia["risk_score"], reid["risk_score"], attr["risk_score"]],
        },
        "summary": (
            f"Overall privacy risk: {overall_score}/100 ({overall_level}). "
            f"MIA risk={mia['risk_level']}, Re-ID risk={reid['risk_level']}, "
            f"Attribute risk={attr['risk_level']}."
        ),
    }

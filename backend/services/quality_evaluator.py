"""
SynthForge — Unified Synthetic Data Quality Evaluation Engine (Phase 6)
Orchestrates 5-pillar evaluation:
1. Structural Fidelity
2. Statistical Fidelity
3. Relationship Fidelity
4. ML Utility
5. Privacy Risk & Protection

Maintains strict separation between Data Fidelity and Privacy Protection dimensions.
"""
import math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Any, Tuple

from backend.utils.logging_config import get_logger
from backend.services.schema_intelligence import (
    detect_target_column,
    drop_identifier_columns,
    get_identifier_columns,
)
from backend.services.dataset_profiler import profile_dataframe, DatasetProfile
from backend.services.statistical_validator import validate_statistical
from backend.services.ml_validator import validate_ml_utility
from backend.services.attack_simulator import run_all_attacks

logger = get_logger("quality_evaluator")


# ──────────────────────────────────────────────
# Helper Functions: Associations & Correlations
# ──────────────────────────────────────────────
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V statistic for categorical-categorical association."""
    x_clean = x.dropna().astype(str)
    y_clean = y.dropna().astype(str)
    if len(x_clean) < 2 or len(y_clean) < 2:
        return 0.0

    common_idx = x_clean.index.intersection(y_clean.index)
    if len(common_idx) < 2:
        return 0.0

    confusion_matrix = pd.crosstab(x_clean.loc[common_idx], y_clean.loc[common_idx])
    if confusion_matrix.empty or confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return 0.0

    chi2 = scipy_stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n <= 1:
        return 0.0

    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.sqrt(phi2corr / denom), 0.0, 1.0))


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Calculate Correlation Ratio (eta) for categorical-numerical association."""
    cats = categories.astype(str)
    meas = pd.to_numeric(measurements, errors="coerce")
    valid = ~(cats.isna() | meas.isna())
    cats = cats[valid]
    meas = meas[valid]
    if len(meas) < 2:
        return 0.0

    total_var = float(meas.var())
    if total_var <= 1e-12:
        return 0.0

    cat_means = meas.groupby(cats).mean()
    cat_counts = meas.groupby(cats).count()
    overall_mean = float(meas.mean())

    weighted_variance = float(((cat_means - overall_mean) ** 2 * cat_counts).sum() / len(meas))
    eta2 = weighted_variance / total_var
    return float(np.clip(np.sqrt(max(0.0, eta2)), 0.0, 1.0))


# ──────────────────────────────────────────────
# 1. Structural Fidelity Evaluation
# ──────────────────────────────────────────────
def evaluate_structural_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
) -> Dict[str, Any]:
    """
    Evaluates schema completeness, datatypes, missingness, cardinality,
    and duplicate patterns between real and synthetic data.
    """
    warnings: List[str] = []
    real_cols = list(real_df.columns)
    synth_cols = list(synth_df.columns)

    # 1. Column preservation
    common_cols = [c for c in real_cols if c in synth_cols]
    missing_cols = [c for c in real_cols if c not in synth_cols]
    extra_cols = [c for c in synth_cols if c not in real_cols]

    col_preservation_rate = len(common_cols) / max(len(real_cols), 1)
    if missing_cols:
        warnings.append(f"Synthetic data is missing columns: {missing_cols}")
    if extra_cols:
        warnings.append(f"Synthetic data contains extra unexpected columns: {extra_cols}")

    # 2. Datatype / Semantic compatibility
    dtype_matches = 0
    col_dtype_details = {}
    for col in common_cols:
        r_type = str(real_df[col].dtype)
        s_type = str(synth_df[col].dtype)
        r_num = pd.api.types.is_numeric_dtype(real_df[col])
        s_num = pd.api.types.is_numeric_dtype(synth_df[col])
        r_bool = pd.api.types.is_bool_dtype(real_df[col])
        s_bool = pd.api.types.is_bool_dtype(synth_df[col])

        compat = (r_type == s_type) or (r_num and s_num) or (r_bool and s_bool)
        if compat:
            dtype_matches += 1
        else:
            warnings.append(f"Column '{col}' datatype mismatch: real={r_type}, synth={s_type}")
        col_dtype_details[col] = {
            "real_dtype": r_type,
            "synth_dtype": s_type,
            "compatible": compat,
        }
    dtype_match_rate = dtype_matches / max(len(common_cols), 1)

    # 3. Missing-value rate differences
    missing_diffs = {}
    missing_diff_vals = []
    for col in common_cols:
        r_null_rate = float(real_df[col].isna().mean())
        s_null_rate = float(synth_df[col].isna().mean())
        diff = abs(r_null_rate - s_null_rate)
        missing_diffs[col] = {
            "real_null_rate": round(r_null_rate, 4),
            "synth_null_rate": round(s_null_rate, 4),
            "diff": round(diff, 4),
        }
        missing_diff_vals.append(diff)
    mean_missing_diff = float(np.mean(missing_diff_vals)) if missing_diff_vals else 0.0

    # 4. Constant column preservation
    constant_details = {}
    constant_matches = 0
    constant_count = 0
    for col in common_cols:
        r_clean = real_df[col].dropna()
        s_clean = synth_df[col].dropna()
        r_is_const = (r_clean.nunique() <= 1) and (len(r_clean) > 0)
        s_is_const = (s_clean.nunique() <= 1) and (len(s_clean) > 0)
        if r_is_const:
            constant_count += 1
            r_val = str(r_clean.iloc[0]) if not r_clean.empty else None
            s_val = str(s_clean.iloc[0]) if not s_clean.empty else None
            val_match = (r_val == s_val) and s_is_const
            if val_match:
                constant_matches += 1
            else:
                warnings.append(f"Constant column '{col}' value changed or varied in synthetic data.")
            constant_details[col] = {
                "real_constant_value": r_val,
                "synth_constant_value": s_val,
                "preserved": val_match,
            }
    const_preservation_rate = (
        (constant_matches / constant_count) if constant_count > 0 else 1.0
    )

    # 5. Categorical domain overlap & novel category overflow
    cat_overlap_scores = []
    overflow_details = {}
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(real_df[col]) or real_df[col].dtype == "object":
            r_cats = set(real_df[col].dropna().astype(str).unique())
            s_cats = set(synth_df[col].dropna().astype(str).unique())
            if r_cats:
                overlap = len(r_cats & s_cats) / len(r_cats | s_cats) if (r_cats | s_cats) else 1.0
                cat_overlap_scores.append(overlap)
                novel_cats = list(s_cats - r_cats)
                if novel_cats:
                    warnings.append(f"Column '{col}' introduced {len(novel_cats)} novel categories not in real data: {novel_cats[:5]}")
                overflow_details[col] = {
                    "real_categories_count": len(r_cats),
                    "synth_categories_count": len(s_cats),
                    "novel_categories": novel_cats,
                    "overlap_jaccard": round(overlap, 4),
                }
    domain_overlap_score = float(np.mean(cat_overlap_scores)) if cat_overlap_scores else 1.0

    # 6. Cardinality differences
    cardinality_diffs = {}
    card_diff_ratios = []
    for col in common_cols:
        r_card = int(real_df[col].nunique(dropna=True))
        s_card = int(synth_df[col].nunique(dropna=True))
        denom = max(r_card, 1)
        rel_diff = abs(r_card - s_card) / denom
        cardinality_diffs[col] = {
            "real_cardinality": r_card,
            "synth_cardinality": s_card,
            "relative_diff": round(rel_diff, 4),
        }
        card_diff_ratios.append(min(1.0, rel_diff))
    mean_card_diff = float(np.mean(card_diff_ratios)) if card_diff_ratios else 0.0

    # 7. Duplicate rate comparison
    r_dup_rate = float(real_df.duplicated().mean()) if len(real_df) > 0 else 0.0
    s_dup_rate = float(synth_df.duplicated().mean()) if len(synth_df) > 0 else 0.0
    dup_rate_diff = abs(r_dup_rate - s_dup_rate)

    # Compute normalized structural fidelity score (0–100)
    score_components = [
        col_preservation_rate * 100 * 0.25,
        dtype_match_rate * 100 * 0.20,
        max(0.0, 100 * (1.0 - mean_missing_diff)) * 0.20,
        const_preservation_rate * 100 * 0.10,
        domain_overlap_score * 100 * 0.15,
        max(0.0, 100 * (1.0 - dup_rate_diff)) * 0.10,
    ]
    structural_score = round(float(np.sum(score_components)), 2)

    status = (
        "passed" if structural_score >= 80 and not missing_cols else
        "warning" if structural_score >= 60 else "failed"
    )

    return {
        "score": structural_score,
        "status": status,
        "metrics": {
            "column_count_real": len(real_cols),
            "column_count_synth": len(synth_cols),
            "common_columns_count": len(common_cols),
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "column_preservation_rate": round(col_preservation_rate, 4),
            "dtype_match_rate": round(dtype_match_rate, 4),
            "mean_missing_difference": round(mean_missing_diff, 4),
            "constant_preservation_rate": round(const_preservation_rate, 4),
            "categorical_domain_overlap": round(domain_overlap_score, 4),
            "mean_cardinality_diff_ratio": round(mean_card_diff, 4),
            "real_duplicate_rate": round(r_dup_rate, 4),
            "synth_duplicate_rate": round(s_dup_rate, 4),
            "duplicate_rate_diff": round(dup_rate_diff, 4),
        },
        "details": {
            "dtypes": col_dtype_details,
            "missing_rates": missing_diffs,
            "constants": constant_details,
            "categorical_overflow": overflow_details,
            "cardinality": cardinality_diffs,
        },
        "warnings": warnings,
    }


# ──────────────────────────────────────────────
# 2. Statistical Fidelity Evaluation
# ──────────────────────────────────────────────
def evaluate_statistical_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    stat_report: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Standardizes statistical similarity metrics between real and synthetic datasets.
    Consumes existing statistical_validator report if supplied.
    """
    if stat_report is None:
        stat_report = validate_statistical(real_df, synth_df)

    warnings: List[str] = []
    overall_score = float(stat_report.get("overall_quality_score", 0.0))
    grade = stat_report.get("quality_grade", "F")

    # Inspect per-column reports for warnings
    col_reports = stat_report.get("column_reports", {})
    for col, rep in col_reports.items():
        if rep.get("type") == "numerical":
            ks = rep.get("ks_test", {})
            if ks.get("p_value") is not None and ks.get("p_value") < 0.01:
                warnings.append(f"Numeric column '{col}' has significant distribution shift (KS p={ks.get('p_value')}).")
        elif rep.get("type") == "categorical":
            chi = rep.get("chi_squared", {})
            if chi.get("p_value") is not None and chi.get("p_value") < 0.01:
                warnings.append(f"Categorical column '{col}' has significant frequency shift (Chi2 p={chi.get('p_value')}).")

    corr = stat_report.get("correlation", {})
    mae = corr.get("mean_absolute_error", 0.0)
    if mae > 0.3:
        warnings.append(f"High correlation divergence across numeric features (MAE = {mae}).")

    status = "passed" if overall_score >= 75 else ("warning" if overall_score >= 50 else "failed")

    return {
        "score": overall_score,
        "grade": grade,
        "status": status,
        "metrics": {
            "num_columns_analyzed": stat_report.get("num_columns_analyzed", 0),
            "num_numeric": stat_report.get("num_numeric", 0),
            "num_categorical": stat_report.get("num_categorical", 0),
            "correlation_mae": mae,
            "correlation_max_error": corr.get("max_absolute_error", 0.0),
            "correlation_quality_score": stat_report.get("correlation_quality_score", 0.0),
        },
        "column_reports": col_reports,
        "warnings": warnings,
    }


# ──────────────────────────────────────────────
# 3. Relationship Fidelity Evaluation
# ──────────────────────────────────────────────
def evaluate_relationship_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates multi-variable relationships:
    - Numeric Pearson & Spearman correlation comparisons
    - Categorical-categorical association matrices via Cramér's V
    - Target-to-feature association preservation when target exists
    """
    warnings: List[str] = []
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    real = real_df[common_cols]
    synth = synth_df[common_cols]

    numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = real.select_dtypes(include=["object", "category"]).columns.tolist()

    sub_scores: List[float] = []
    applicable_checks: List[str] = []

    # 1. Pearson Correlation
    pearson_mae = None
    if len(numeric_cols) >= 2:
        r_corr = real[numeric_cols].corr(method="pearson").fillna(0).values
        s_corr = synth[numeric_cols].corr(method="pearson").fillna(0).values
        p_delta = np.abs(r_corr - s_corr)
        pearson_mae = float(np.mean(p_delta))
        p_score = max(0.0, 100.0 - pearson_mae * 150.0)
        sub_scores.append(p_score)
        applicable_checks.append("pearson_correlation")
        if pearson_mae > 0.25:
            warnings.append(f"Numeric Pearson correlation MAE is elevated ({pearson_mae:.3f}).")

    # 2. Spearman Rank Correlation
    spearman_mae = None
    if len(numeric_cols) >= 2:
        try:
            r_scorr = real[numeric_cols].corr(method="spearman").fillna(0).values
            s_scorr = synth[numeric_cols].corr(method="spearman").fillna(0).values
            s_delta = np.abs(r_scorr - s_scorr)
            spearman_mae = float(np.mean(s_delta))
            s_score = max(0.0, 100.0 - spearman_mae * 150.0)
            sub_scores.append(s_score)
            applicable_checks.append("spearman_correlation")
        except Exception:
            pass

    # 3. Categorical Associations (Cramér's V)
    cramers_mae = None
    cramers_matrix_real = []
    cramers_matrix_synth = []
    valid_cat_cols = [c for c in cat_cols if real[c].dropna().nunique() > 1]

    if len(valid_cat_cols) >= 2:
        k = len(valid_cat_cols)
        c_real = np.ones((k, k))
        c_synth = np.ones((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                col_i, col_j = valid_cat_cols[i], valid_cat_cols[j]
                v_r = _cramers_v(real[col_i], real[col_j])
                v_s = _cramers_v(synth[col_i], synth[col_j])
                c_real[i, j] = c_real[j, i] = v_r
                c_synth[i, j] = c_synth[j, i] = v_s

        c_delta = np.abs(c_real - c_synth)
        cramers_mae = float(np.mean(c_delta[np.triu_indices(k, k=1)]))
        c_score = max(0.0, 100.0 - cramers_mae * 150.0)
        sub_scores.append(c_score)
        applicable_checks.append("cramers_v_association")
        cramers_matrix_real = c_real.tolist()
        cramers_matrix_synth = c_synth.tolist()
        if cramers_mae > 0.3:
            warnings.append(f"Categorical association (Cramér's V) MAE is elevated ({cramers_mae:.3f}).")

    # 4. Target-to-Feature Relationship
    target_assoc_mae = None
    target_assoc_details = {}
    if target_col and target_col in common_cols:
        target_is_num = pd.api.types.is_numeric_dtype(real[target_col])
        t_deltas = []
        for col in common_cols:
            if col == target_col:
                continue
            col_is_num = pd.api.types.is_numeric_dtype(real[col])
            try:
                if target_is_num and col_is_num:
                    r_val = float(real[[target_col, col]].corr().iloc[0, 1])
                    s_val = float(synth[[target_col, col]].corr().iloc[0, 1])
                elif not target_is_num and not col_is_num:
                    r_val = _cramers_v(real[target_col], real[col])
                    s_val = _cramers_v(synth[target_col], synth[col])
                else:
                    cat_col = target_col if not target_is_num else col
                    num_col = col if not target_is_num else target_col
                    r_val = _correlation_ratio(real[cat_col], real[num_col])
                    s_val = _correlation_ratio(synth[cat_col], synth[num_col])

                if not math.isnan(r_val) and not math.isnan(s_val):
                    delta = abs(r_val - s_val)
                    t_deltas.append(delta)
                    target_assoc_details[col] = {
                        "real_assoc": round(r_val, 4),
                        "synth_assoc": round(s_val, 4),
                        "delta": round(delta, 4),
                    }
            except Exception:
                continue

        if t_deltas:
            target_assoc_mae = float(np.mean(t_deltas))
            t_score = max(0.0, 100.0 - target_assoc_mae * 150.0)
            sub_scores.append(t_score)
            applicable_checks.append("target_feature_association")

    if sub_scores:
        rel_score = round(float(np.mean(sub_scores)), 2)
        applicable = True
    else:
        rel_score = 100.0
        applicable = False

    status = (
        "passed" if rel_score >= 75 else
        "warning" if rel_score >= 50 else "failed"
    )

    return {
        "applicable": applicable,
        "score": rel_score,
        "status": status,
        "applicable_checks": applicable_checks,
        "metrics": {
            "pearson_mae": round(pearson_mae, 4) if pearson_mae is not None else None,
            "spearman_mae": round(spearman_mae, 4) if spearman_mae is not None else None,
            "cramers_v_mae": round(cramers_mae, 4) if cramers_mae is not None else None,
            "target_association_mae": round(target_assoc_mae, 4) if target_assoc_mae is not None else None,
        },
        "details": {
            "categorical_columns_evaluated": valid_cat_cols,
            "cramers_matrix_real": cramers_matrix_real,
            "cramers_matrix_synth": cramers_matrix_synth,
            "target_column": target_col,
            "target_associations": target_assoc_details,
        },
        "warnings": warnings,
    }


# ──────────────────────────────────────────────
# 4. ML Utility Evaluation
# ──────────────────────────────────────────────
def evaluate_ml_utility(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: Optional[str] = None,
    ml_report: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Standardizes ML utility validation via TSTR benchmark.
    Consumes existing ml_validator report if supplied.
    """
    if target_col is None:
        target_col = detect_target_column(real_df)

    if ml_report is None:
        if target_col and target_col in real_df.columns and target_col in synth_df.columns:
            # Check target has at least 2 classes
            if real_df[target_col].dropna().nunique() >= 2:
                ml_report = validate_ml_utility(real_df, synth_df, target_col=target_col)
            else:
                ml_report = {"error": f"Target column '{target_col}' has fewer than 2 unique classes."}
        else:
            ml_report = {"error": "No valid target column found for ML evaluation."}

    warnings: List[str] = []
    if ml_report.get("error"):
        return {
            "applicable": False,
            "score": None,
            "grade": "N/A",
            "status": "not_applicable",
            "reason": ml_report["error"],
            "metrics": {},
            "warnings": [ml_report["error"]],
        }

    score = float(ml_report.get("utility_score", 0.0))
    grade = ml_report.get("utility_grade", "F")
    gaps = ml_report.get("utility_gaps", {})
    results = ml_report.get("results", {})

    acc_gap = gaps.get("accuracy_gap")
    f1_gap = gaps.get("f1_gap")
    auc_gap = gaps.get("auc_gap")

    if acc_gap is not None and acc_gap > 0.15:
        warnings.append(f"Large accuracy gap between TRTR and TSTR ({acc_gap:.4f}).")
    if f1_gap is not None and f1_gap > 0.15:
        warnings.append(f"Large F1 gap between TRTR and TSTR ({f1_gap:.4f}).")

    status = "passed" if score >= 75 else ("warning" if score >= 50 else "failed")

    return {
        "applicable": True,
        "score": score,
        "grade": grade,
        "status": status,
        "target_column": ml_report.get("target_column"),
        "target_classes": ml_report.get("target_classes"),
        "metrics": {
            "accuracy_gap": acc_gap,
            "f1_gap": f1_gap,
            "auc_gap": auc_gap,
            "precision_gap": gaps.get("precision_gap"),
            "recall_gap": gaps.get("recall_gap"),
            "trtr_rf_accuracy": results.get("trtr_rf", {}).get("accuracy"),
            "tstr_rf_accuracy": results.get("tstr_rf", {}).get("accuracy"),
            "trtr_rf_f1": results.get("trtr_rf", {}).get("f1_score"),
            "tstr_rf_f1": results.get("tstr_rf", {}).get("f1_score"),
            "trtr_rf_auc": results.get("trtr_rf", {}).get("roc_auc"),
            "tstr_rf_auc": results.get("tstr_rf", {}).get("roc_auc"),
        },
        "feature_importances": {
            "features": ml_report.get("feature_names", []),
            "trtr_rf_top_indices": results.get("trtr_rf", {}).get("top_feature_indices"),
            "tstr_rf_top_indices": results.get("tstr_rf", {}).get("top_feature_indices"),
        },
        "comparison_chart": ml_report.get("comparison_chart"),
        "warnings": warnings,
    }


# ──────────────────────────────────────────────
# 5. Privacy Risk & Protection Evaluation
# ──────────────────────────────────────────────
def check_exact_duplicate_collisions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> Tuple[int, float]:
    """
    Checks for exact 1-to-1 matching records between real and synthetic data.
    Returns (exact_match_count, exact_match_rate).
    """
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if not common_cols or len(real_df) == 0 or len(synth_df) == 0:
        return 0, 0.0

    r_clean = real_df[common_cols].dropna().astype(str)
    s_clean = synth_df[common_cols].dropna().astype(str)

    if r_clean.empty or s_clean.empty:
        return 0, 0.0

    # Inner merge to find exact matching rows
    matches = pd.merge(r_clean, s_clean, how="inner", on=common_cols)
    match_count = len(matches)
    match_rate = float(match_count / len(synth_df))
    return match_count, match_rate


def evaluate_privacy_risk(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    attack_report: Optional[Dict] = None,
    dp_metadata: Optional[Dict] = None,
    budget_state: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluates empirical privacy attacks, exact collisions, and differential privacy guarantees.
    Calculates both Raw Privacy Risk (0=none, 100=extreme) and Privacy Protection Score (100=max protection).
    """
    warnings: List[str] = []

    # 1. Exact Duplicate Collisions
    exact_count, exact_rate = check_exact_duplicate_collisions(real_df, synth_df)
    if exact_count > 0:
        warnings.append(
            f"EXACT MATCH COLLISION: {exact_count} synthetic record(s) ({exact_rate * 100:.2f}%) "
            f"identically match real training records."
        )

    # 2. Privacy Attacks
    if attack_report is None:
        try:
            attack_report = run_all_attacks(real_df, synth_df)
        except Exception as e:
            logger.warning(f"Attack simulation error: {e}")
            attack_report = {
                "overall_risk_score": 50.0,
                "overall_risk_level": "medium",
                "attacks": {},
            }

    attacks = attack_report.get("attacks", {})
    mia = attacks.get("membership_inference", {})
    reid = attacks.get("reidentification", {})
    attr = attacks.get("attribute_inference", {})

    mia_adv = mia.get("attack_advantage", 0.0)
    mia_auc = mia.get("attack_auc", 0.5)
    reid_pct = reid.get("records_at_risk_pct", 0.0)
    attr_adv = attr.get("average_advantage", 0.0)

    if mia_auc is not None and mia_auc > 0.65:
        warnings.append(f"High Membership Inference risk (AUC = {mia_auc:.3f}).")
    if reid_pct > 5.0:
        warnings.append(f"High Re-identification risk ({reid_pct:.1f}% records dangerously close to real records).")
    if attr_adv > 0.15:
        warnings.append(f"Elevated Attribute Inference advantage ({attr_adv:.3f} over majority baseline).")

    raw_risk_score = float(attack_report.get("overall_risk_score", 0.0))
    # Incorporate exact duplicate penalty to raw risk score
    if exact_count > 0:
        raw_risk_score = min(100.0, raw_risk_score + min(50.0, exact_rate * 500.0))

    raw_risk_score = round(raw_risk_score, 2)
    # Protection Score: 100 is maximum protection, 0 is complete breach
    privacy_protection_score = round(max(0.0, 100.0 - raw_risk_score), 2)

    risk_level = (
        "low" if raw_risk_score <= 20 else
        "medium" if raw_risk_score <= 40 else
        "high" if raw_risk_score <= 70 else "critical"
    )

    status = (
        "passed" if risk_level in ["low", "medium"] and exact_count == 0 else
        "warning" if risk_level == "high" or exact_count <= 2 else "failed"
    )

    return {
        "privacy_protection_score": privacy_protection_score,
        "raw_privacy_risk_score": raw_risk_score,
        "risk_level": risk_level,
        "status": status,
        "metrics": {
            "exact_duplicate_count": exact_count,
            "exact_duplicate_rate": round(exact_rate, 6),
            "mia_auc": mia_auc,
            "mia_advantage": mia_adv,
            "reid_records_at_risk_pct": reid_pct,
            "attribute_inference_avg_advantage": attr_adv,
            "dp_epsilon_actual": dp_metadata.get("epsilon_actual") if dp_metadata else None,
            "dp_mechanism": dp_metadata.get("mechanism") if dp_metadata else None,
            "budget_utilization_pct": budget_state.get("utilization_pct") if budget_state else None,
        },
        "attacks": attacks,
        "dp_metadata": dp_metadata,
        "budget_state": budget_state,
        "warnings": warnings,
    }


# ──────────────────────────────────────────────
# 6. Quality Evaluator Orchestrator
# ──────────────────────────────────────────────
class QualityEvaluator:
    """
    Unified Quality Evaluator orchestrating all 5 fidelity & privacy evaluation pillars.
    """

    @staticmethod
    def evaluate(
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        profile: Optional[DatasetProfile] = None,
        target_column: Optional[str] = None,
        stat_report: Optional[Dict] = None,
        ml_report: Optional[Dict] = None,
        attack_report: Optional[Dict] = None,
        dp_metadata: Optional[Dict] = None,
        budget_state: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Executes unified 5-pillar evaluation and builds the complete QualityReport.
        """
        logger.info(f"Starting unified quality evaluation on {len(real_df)} real vs {len(synth_df)} synthetic rows.")

        # Clean identifiers before metric computations
        real_eval = drop_identifier_columns(real_df)
        synth_eval = drop_identifier_columns(synth_df)

        if profile is None:
            try:
                profile = profile_dataframe(real_df)
            except Exception:
                profile = None

        if target_column is None and profile and profile.detected_metadata.target_column:
            target_column = profile.detected_metadata.target_column
        elif target_column is None:
            target_column = detect_target_column(real_df)

        # 1. Structural Fidelity
        structural = evaluate_structural_fidelity(real_df, synth_df, profile=profile)

        # 2. Statistical Fidelity
        statistical = evaluate_statistical_fidelity(real_eval, synth_eval, stat_report=stat_report)

        # 3. Relationship Fidelity
        relationship = evaluate_relationship_fidelity(real_eval, synth_eval, target_col=target_column)

        # 4. ML Utility
        ml_utility = evaluate_ml_utility(real_eval, synth_eval, target_col=target_column, ml_report=ml_report)

        # 5. Privacy Risk
        privacy = evaluate_privacy_risk(
            real_df, synth_df,
            attack_report=attack_report,
            dp_metadata=dp_metadata,
            budget_state=budget_state,
        )

        # ── Weighted Data Fidelity Score ──
        # Default weights
        weights = {
            "structural": 0.25,
            "statistical": 0.35,
            "relationship": 0.20,
            "ml_utility": 0.20,
        }

        active_weights = {}
        weighted_sum = 0.0

        # Structural is always active
        active_weights["structural"] = weights["structural"]
        weighted_sum += structural["score"] * weights["structural"]

        # Statistical is always active
        active_weights["statistical"] = weights["statistical"]
        weighted_sum += statistical["score"] * weights["statistical"]

        # Relationship
        if relationship["applicable"]:
            active_weights["relationship"] = weights["relationship"]
            weighted_sum += relationship["score"] * weights["relationship"]

        # ML Utility
        if ml_utility["applicable"] and ml_utility["score"] is not None:
            active_weights["ml_utility"] = weights["ml_utility"]
            weighted_sum += ml_utility["score"] * weights["ml_utility"]

        total_weight = sum(active_weights.values())
        data_fidelity_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

        def _get_grade(score: float) -> str:
            if score >= 90: return "A"
            if score >= 75: return "B"
            if score >= 60: return "C"
            if score >= 40: return "D"
            return "F"

        fidelity_grade = _get_grade(data_fidelity_score)
        privacy_grade = _get_grade(privacy["privacy_protection_score"])

        # Aggregate warnings
        all_warnings = (
            structural["warnings"] +
            statistical["warnings"] +
            relationship["warnings"] +
            ml_utility["warnings"] +
            privacy["warnings"]
        )

        # Readiness Verdict
        is_fidelity_ready = data_fidelity_score >= 65.0
        is_privacy_ready = privacy["privacy_protection_score"] >= 60.0 and privacy["metrics"]["exact_duplicate_count"] == 0

        if is_fidelity_ready and is_privacy_ready:
            trust_verdict = "Deployment Ready: Satisfies standard fidelity and privacy criteria."
        elif not is_privacy_ready and is_fidelity_ready:
            trust_verdict = "Privacy Review Required: Fidelity is adequate, but privacy guarantees or collision risks require adjustment."
        elif is_privacy_ready and not is_fidelity_ready:
            trust_verdict = "Fidelity Review Required: Privacy is protected, but statistical or ML utility is degraded."
        else:
            trust_verdict = "Revision Required: Both fidelity and privacy protection fall below standard deployment thresholds."

        executive_summary = {
            "data_fidelity_score": data_fidelity_score,
            "data_fidelity_grade": fidelity_grade,
            "privacy_protection_score": privacy["privacy_protection_score"],
            "privacy_protection_grade": privacy_grade,
            "raw_privacy_risk_score": privacy["raw_privacy_risk_score"],
            "privacy_risk_level": privacy["risk_level"],
            "trust_verdict": trust_verdict,
            "applicable_pillars": list(active_weights.keys()),
            "warnings_count": len(all_warnings),
            "disclaimer": (
                "IMPORTANT: Synthetic data trustworthiness cannot be determined by any single metric. "
                "Fidelity and Privacy represent separate trade-off dimensions and must be evaluated independently."
            ),
        }

        report = {
            "executive_summary": executive_summary,
            "structural_fidelity": structural,
            "statistical_fidelity": statistical,
            "relationship_fidelity": relationship,
            "ml_utility": ml_utility,
            "privacy_risk": privacy,
            "all_warnings": all_warnings,
        }

        logger.info(
            f"Quality evaluation complete. Fidelity: {data_fidelity_score}/100 ({fidelity_grade}), "
            f"Privacy Protection: {privacy['privacy_protection_score']}/100 ({privacy_grade})"
        )
        return report

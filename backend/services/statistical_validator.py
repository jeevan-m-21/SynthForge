"""
MediSynth.AI — Statistical Similarity Validation
Proves synthetic data quality using KS test, mean/variance comparison,
correlation matrix comparison, and chi-squared tests.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple

from backend.utils.logging_config import get_logger

logger = get_logger("statistical_validator")


def _ks_test(real_col: pd.Series, synth_col: pd.Series) -> Dict:
    """Kolmogorov-Smirnov test for distribution similarity."""
    real_clean = real_col.dropna().astype(float)
    synth_clean = synth_col.dropna().astype(float)
    if len(real_clean) < 2 or len(synth_clean) < 2:
        return {"statistic": None, "p_value": None, "similar": False}
    stat, p_val = scipy_stats.ks_2samp(real_clean, synth_clean)
    return {
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_val), 6),
        "similar": p_val > 0.05,  # fail to reject H0 → distributions similar
    }


def _mean_variance_comparison(real_col: pd.Series,
                               synth_col: pd.Series) -> Dict:
    """Compare mean and variance between real and synthetic columns."""
    real_clean = real_col.dropna().astype(float)
    synth_clean = synth_col.dropna().astype(float)
    if len(real_clean) < 2 or len(synth_clean) < 2:
        return {"real_mean": None, "synth_mean": None}

    real_mean = float(real_clean.mean())
    synth_mean = float(synth_clean.mean())
    real_var = float(real_clean.var())
    synth_var = float(synth_clean.var())

    mean_deviation = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10) * 100
    var_ratio = synth_var / (real_var + 1e-10)

    return {
        "real_mean": round(real_mean, 4),
        "synth_mean": round(synth_mean, 4),
        "mean_deviation_pct": round(mean_deviation, 2),
        "real_variance": round(real_var, 4),
        "synth_variance": round(synth_var, 4),
        "variance_ratio": round(var_ratio, 4),
    }


def _chi_squared_test(real_col: pd.Series, synth_col: pd.Series) -> Dict:
    """Chi-squared test for categorical column distribution similarity."""
    real_clean = real_col.dropna().astype(str)
    synth_clean = synth_col.dropna().astype(str)

    all_categories = sorted(set(real_clean.unique()) | set(synth_clean.unique()))
    if len(all_categories) < 2:
        return {"statistic": None, "p_value": None, "similar": True}

    real_counts = real_clean.value_counts()
    synth_counts = synth_clean.value_counts()

    real_freq = np.array([real_counts.get(c, 0) for c in all_categories], dtype=float)
    synth_freq = np.array([synth_counts.get(c, 0) for c in all_categories], dtype=float)

    # Normalize to same total
    total = real_freq.sum()
    if total == 0:
        return {"statistic": None, "p_value": None, "similar": True}
    synth_freq = synth_freq * (total / (synth_freq.sum() + 1e-10))

    # Add small constant to avoid zero expected frequencies
    real_freq += 0.001
    synth_freq += 0.001

    stat, p_val = scipy_stats.chisquare(synth_freq, f_exp=real_freq)
    return {
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_val), 6),
        "similar": p_val > 0.05,
    }


def _correlation_comparison(real_df: pd.DataFrame,
                             synth_df: pd.DataFrame) -> Dict:
    """Compare correlation matrices between real and synthetic data."""
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    synth_numeric = [c for c in numeric_cols if c in synth_df.columns]

    if len(synth_numeric) < 2:
        return {
            "real_correlation": [],
            "synth_correlation": [],
            "delta_correlation": [],
            "mean_absolute_error": 0,
            "columns": [],
        }

    real_corr = real_df[synth_numeric].corr().values
    synth_corr = synth_df[synth_numeric].corr().values
    delta = np.abs(real_corr - synth_corr)
    mae = float(np.mean(delta))

    return {
        "real_correlation": real_corr.tolist(),
        "synth_correlation": synth_corr.tolist(),
        "delta_correlation": delta.tolist(),
        "mean_absolute_error": round(mae, 6),
        "max_absolute_error": round(float(np.max(delta)), 6),
        "columns": synth_numeric,
    }


def _distribution_data(real_col: pd.Series, synth_col: pd.Series,
                        bins: int = 30) -> Dict:
    """Compute histogram data for distribution visualization."""
    real_clean = real_col.dropna().astype(float)
    synth_clean = synth_col.dropna().astype(float)

    if len(real_clean) < 2 or len(synth_clean) < 2:
        return {"bins": [], "real_counts": [], "synth_counts": []}

    all_values = np.concatenate([real_clean.values, synth_clean.values])
    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)

    real_hist, _ = np.histogram(real_clean, bins=bin_edges, density=True)
    synth_hist, _ = np.histogram(synth_clean, bins=bin_edges, density=True)

    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    return {
        "bins": [round(b, 4) for b in bin_centers],
        "real_counts": [round(float(c), 6) for c in real_hist],
        "synth_counts": [round(float(c), 6) for c in synth_hist],
    }


def _categorical_distribution_data(real_col: pd.Series,
                                     synth_col: pd.Series) -> Dict:
    """Compute category frequency data for visualization."""
    real_clean = real_col.dropna().astype(str)
    synth_clean = synth_col.dropna().astype(str)

    all_cats = sorted(set(real_clean.unique()) | set(synth_clean.unique()))

    real_vc = real_clean.value_counts(normalize=True)
    synth_vc = synth_clean.value_counts(normalize=True)

    return {
        "categories": all_cats,
        "real_proportions": [round(float(real_vc.get(c, 0)), 4) for c in all_cats],
        "synth_proportions": [round(float(synth_vc.get(c, 0)), 4) for c in all_cats],
    }


def validate_statistical(real_df: pd.DataFrame,
                          synth_df: pd.DataFrame) -> Dict:
    """
    Run full statistical validation between real and synthetic data.

    Returns comprehensive report with per-column metrics, correlation
    comparison, and overall quality score.
    """
    # Align columns
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    real = real_df[common_cols]
    synth = synth_df[common_cols]

    numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = real.select_dtypes(include=["object", "category"]).columns.tolist()

    # Per-column analysis
    column_reports = {}
    quality_scores = []

    for col in numeric_cols:
        ks = _ks_test(real[col], synth[col])
        mv = _mean_variance_comparison(real[col], synth[col])
        dist = _distribution_data(real[col], synth[col])

        # Column quality score (0-100)
        ks_score = max(0, 100 - ks["statistic"] * 200) if ks["statistic"] is not None else 50
        mean_score = max(0, 100 - mv.get("mean_deviation_pct", 100))
        col_quality = (ks_score * 0.6 + mean_score * 0.4)
        quality_scores.append(col_quality)

        column_reports[col] = {
            "type": "numerical",
            "ks_test": ks,
            "mean_variance": mv,
            "distribution": dist,
            "quality_score": round(col_quality, 2),
        }

    for col in cat_cols:
        chi2 = _chi_squared_test(real[col], synth[col])
        dist = _categorical_distribution_data(real[col], synth[col])

        col_quality = 100 if chi2.get("similar") else max(0, chi2.get("p_value", 0) * 200)
        quality_scores.append(col_quality)

        column_reports[col] = {
            "type": "categorical",
            "chi_squared": chi2,
            "distribution": dist,
            "quality_score": round(col_quality, 2),
        }

    # Correlation analysis
    correlation = _correlation_comparison(real, synth)
    corr_score = max(0, 100 - correlation["mean_absolute_error"] * 200)
    quality_scores.append(corr_score)

    # Overall quality score
    overall_quality = round(np.mean(quality_scores), 2) if quality_scores else 0

    report = {
        "overall_quality_score": overall_quality,
        "quality_grade": (
            "A" if overall_quality >= 90 else
            "B" if overall_quality >= 75 else
            "C" if overall_quality >= 60 else
            "D" if overall_quality >= 40 else "F"
        ),
        "num_columns_analyzed": len(common_cols),
        "num_numeric": len(numeric_cols),
        "num_categorical": len(cat_cols),
        "column_reports": column_reports,
        "correlation": correlation,
        "correlation_quality_score": round(corr_score, 2),
    }

    logger.info(f"Statistical validation complete. Quality: {overall_quality}/100 ({report['quality_grade']})")
    return report

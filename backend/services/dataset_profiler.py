"""
SynthForge — Dataset Profiler
Domain-independent tabular dataset profiling.
Builds on the Phase 1 schema_intelligence layer.
"""
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from backend.services.schema_intelligence import (
    detect_sensitive_columns,
    detect_target_column,
    get_identifier_columns,
)
from backend.utils.logging_config import get_logger

logger = get_logger("dataset_profiler")

_SAMPLE_SIZE = 5          # example values per column
_TEXT_AVG_LEN = 30        # avg char length threshold to call a col "text"
_TEXT_UNIQUE_RATIO = 0.7  # unique-ratio threshold for text heuristic
_DATETIME_SAMPLE = 20     # rows sampled for datetime inference


# ─────────────────────────────────────────────────────────────────────────────
# Semantic types
# ─────────────────────────────────────────────────────────────────────────────

class SemanticType:
    IDENTIFIER  = "identifier"
    NUMERIC     = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN     = "boolean"
    DATETIME    = "datetime"
    TEXT        = "text"
    CONSTANT    = "constant"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class NumericStats(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None


class CategoryEntry(BaseModel):
    value: str
    count: int
    percentage: float


class CategoricalStats(BaseModel):
    num_categories: int
    top_categories: List[CategoryEntry] = Field(default_factory=list)


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    nullable: bool
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any] = Field(default_factory=list)

    # Role flags
    identifier: bool = False
    target_candidate: bool = False
    sensitive_candidate: bool = False

    # Type-specific stats — only one populated per column
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None


class DetectedMetadata(BaseModel):
    identifier_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    potentially_sensitive_columns: List[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    dataset_name: str = ""
    row_count: int
    column_count: int
    missing_count: int
    missing_percentage: float
    duplicate_count: int
    duplicate_percentage: float
    memory_usage_bytes: int
    columns: List[ColumnProfile] = Field(default_factory=list)
    detected_metadata: DetectedMetadata = Field(default_factory=DetectedMetadata)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _r(val: Any, d: int = 4) -> Optional[float]:
    if val is None or pd.isna(val):
        return None
    try:
        return round(float(val), d)
    except (ValueError, TypeError):
        return None


def _safe_sample(series: pd.Series, n: int = _SAMPLE_SIZE) -> List[Any]:
    clean = series.dropna()
    if clean.empty:
        return []
    return [str(v) if not isinstance(v, (int, float, bool)) else v for v in clean.head(n).tolist()]


def _detect_datetime(series: pd.Series) -> bool:
    """Confident datetime detection — prefer false negatives over false positives."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    clean = series.dropna().astype(str).head(_DATETIME_SAMPLE)
    if len(clean) == 0:
        return False
    date_like = clean.str.match(
        r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"   # YYYY-MM-DD / DD/MM/YYYY
        r"|^\d{1,2}\s+\w{3}\s+\d{4}"          # 01 Jan 2020
    )
    if date_like.mean() >= 0.8:
        try:
            pd.to_datetime(clean, errors="raise")
            return True
        except Exception:
            return False
    return False


def _is_text(series: pd.Series, unique_count: int, total: int) -> bool:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    unique_ratio = unique_count / total if total else 0
    clean = series.dropna().astype(str)
    avg_len = clean.str.len().mean() if len(clean) else 0
    return avg_len >= _TEXT_AVG_LEN and unique_ratio >= _TEXT_UNIQUE_RATIO


def _semantic_type(col: str, series: pd.Series, is_id: bool, n_rows: int) -> str:
    if _detect_datetime(series):
        return SemanticType.DATETIME

    if is_id:
        return SemanticType.IDENTIFIER

    unique_count = series.nunique(dropna=True)

    if unique_count <= 1:
        return SemanticType.CONSTANT

    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BOOLEAN

    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique())
        if vals.issubset({0, 1, 0.0, 1.0}) and unique_count <= 2:
            return SemanticType.BOOLEAN

    if pd.api.types.is_numeric_dtype(series):
        return SemanticType.NUMERIC

    total = len(series.dropna())
    if _is_text(series, unique_count, total):
        return SemanticType.TEXT

    return SemanticType.CATEGORICAL


def _numeric_stats(series: pd.Series) -> NumericStats:
    clean = series.dropna()
    if clean.empty:
        return NumericStats()
    q = clean.quantile([0.25, 0.75])
    return NumericStats(
        min=_r(clean.min()),
        max=_r(clean.max()),
        mean=_r(clean.mean()),
        median=_r(clean.median()),
        std=_r(clean.std()),
        q25=_r(float(q.iloc[0])),
        q75=_r(float(q.iloc[1])),
    )


def _categorical_stats(series: pd.Series, top_n: int = 10) -> CategoricalStats:
    clean = series.dropna()
    n = len(clean)
    vc = clean.value_counts().head(top_n)
    top = [
        CategoryEntry(value=str(v), count=int(c), percentage=_r(c / n * 100 if n else 0, 2) or 0.0)
        for v, c in vc.items()
    ]
    return CategoricalStats(num_categories=int(series.nunique(dropna=True)), top_categories=top)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame, dataset_name: str = "") -> DatasetProfile:
    """
    Profile a pandas DataFrame and return a structured DatasetProfile.

    Works on any tabular CSV domain. Delegates schema detection to the
    Phase 1 schema_intelligence helpers.
    """
    if df is None or df.empty:
        col_count = 0 if df is None else len(df.columns)
        return DatasetProfile(
            dataset_name=dataset_name,
            row_count=0,
            column_count=col_count,
            missing_count=0,
            missing_percentage=0.0,
            duplicate_count=0,
            duplicate_percentage=0.0,
            memory_usage_bytes=0,
        )

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_total = int(df.isna().sum().sum())
    missing_pct = _r(missing_total / total_cells * 100 if total_cells else 0, 2) or 0.0
    dup_count = int(df.duplicated().sum())
    dup_pct = _r(dup_count / n_rows * 100 if n_rows else 0, 2) or 0.0
    mem_bytes = int(df.memory_usage(deep=True).sum())

    # Phase 1 helpers — run once
    id_cols = get_identifier_columns(df)
    target_col = detect_target_column(df)
    sensitive_cols = detect_sensitive_columns(df)

    column_profiles: List[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        is_id = col in id_cols
        unique_count = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())
        missing_pct_col = _r(missing_count / n_rows * 100 if n_rows else 0, 2) or 0.0
        unique_pct = _r(unique_count / n_rows * 100 if n_rows else 0, 2) or 0.0
        sem_type = _semantic_type(col, series, is_id, n_rows)

        num_stats: Optional[NumericStats] = None
        cat_stats: Optional[CategoricalStats] = None

        if sem_type == SemanticType.NUMERIC:
            num_stats = _numeric_stats(series)
        elif sem_type in (SemanticType.CATEGORICAL, SemanticType.BOOLEAN, SemanticType.CONSTANT):
            cat_stats = _categorical_stats(series)

        column_profiles.append(ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            semantic_type=sem_type,
            nullable=bool(series.isna().any()),
            missing_count=missing_count,
            missing_percentage=missing_pct_col,
            unique_count=unique_count,
            unique_percentage=unique_pct,
            sample_values=_safe_sample(series),
            identifier=is_id,
            target_candidate=(col == target_col),
            sensitive_candidate=(col in sensitive_cols),
            numeric_stats=num_stats,
            categorical_stats=cat_stats,
        ))

    return DatasetProfile(
        dataset_name=dataset_name,
        row_count=n_rows,
        column_count=n_cols,
        missing_count=missing_total,
        missing_percentage=missing_pct,
        duplicate_count=dup_count,
        duplicate_percentage=dup_pct,
        memory_usage_bytes=mem_bytes,
        columns=column_profiles,
        detected_metadata=DetectedMetadata(
            identifier_columns=id_cols,
            target_column=target_col,
            potentially_sensitive_columns=sensitive_cols,
        ),
    )

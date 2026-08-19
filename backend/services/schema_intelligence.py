"""Schema intelligence helpers for generic tabular datasets.

These utilities centralize lightweight heuristics that are reused by data
ingestion, generation, validation, and privacy analysis.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd  # type: ignore[reportMissingModuleSource]


TARGET_NAME_KEYWORDS = {
    "target",
    "label",
    "class",
    "outcome",
    "result",
    "status",
    "prediction",
    "category",
}

SENSITIVE_NAME_KEYWORDS = (
    "name",
    "email",
    "phone",
    "mobile",
    "address",
    "dob",
    "date_of_birth",
    "birth_date",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "income",
    "salary",
    "credit",
    "account",
    "password",
    "secret",
    "token",
    "ssn",
    "social_security",
)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _name_matches(name: str, patterns: Iterable[str]) -> bool:
    normalized = _normalize_name(name)
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_identifier_column(
    series: pd.Series,
    column_name: str,
    *,
    dataset_size: Optional[int] = None,
) -> bool:
    """Heuristically detect identifier-like columns.

    The heuristic balances explicit name patterns with uniqueness and type
    signals so that generic IDs such as customer_id, student_id, event_id, and
    row indices are detected without hard-coding domain-specific lists.
    """

    if series is None:
        return False

    name = _normalize_name(column_name)
    clean = series.dropna()
    if clean.empty:
        return False

    n_unique = clean.nunique(dropna=True)
    total = len(clean)
    unique_ratio = n_unique / total if total else 0.0
    size = dataset_size if dataset_size is not None else len(series)

    if name in {"id", "row_id", "record_id", "uuid", "uid", "index"}:
        return True

    if name.endswith("_id") or name.endswith("-id"):
        return True

    if _name_matches(name, (r"(^|_)id$", r"(^|_)uuid$", r"(^|_)uid$")):
        return True

    if size >= 20 and unique_ratio >= 0.95 and n_unique >= min(50, max(10, size // 2)):
        return True

    if pd.api.types.is_integer_dtype(series) and unique_ratio >= 0.98 and n_unique >= 20:
        return True

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = clean.astype(str).head(25)
        if len(sample) > 0 and sample.str.match(r"^[0-9a-fA-F-]{8,}$").mean() >= 0.6:
            return True

    return False


def get_identifier_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in df.columns
        if is_identifier_column(df[col], col, dataset_size=len(df))
    ]


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_columns = get_identifier_columns(df)
    if not id_columns:
        return df.copy()
    return df.drop(columns=id_columns, errors="ignore")


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detect a reasonable default target column for generic tabular ML tasks."""
    if df.empty:
        return None

    for col in df.columns:
        if _normalize_name(col) in TARGET_NAME_KEYWORDS:
            return col

    candidates = []
    for col in df.columns:
        if is_identifier_column(df[col], col, dataset_size=len(df)):
            continue

        unique_count = df[col].nunique(dropna=True)
        if unique_count < 2:
            continue

        if unique_count == 2:
            candidates.append((0, col))
        else:
            low_cardinality_cap = 10 if len(df) < 50 else max(3, len(df) // 20)
            if 2 < unique_count <= min(10, low_cardinality_cap):
                candidates.append((1, col))

    if candidates:
        candidates.sort(key=lambda item: (item[0], df[item[1]].nunique(dropna=True)))
        return candidates[0][1]

    return None


def detect_sensitive_columns(df: pd.DataFrame, max_columns: int = 5) -> List[str]:
    """Detect potentially sensitive columns for privacy attack simulation."""
    detected: List[str] = []

    for col in df.columns:
        if is_identifier_column(df[col], col, dataset_size=len(df)):
            continue

        normalized = _normalize_name(col)
        if any(keyword in normalized for keyword in SENSITIVE_NAME_KEYWORDS):
            detected.append(col)

    if len(detected) >= max_columns:
        return detected[:max_columns]

    for col in df.columns:
        if col in detected:
            continue
        if is_identifier_column(df[col], col, dataset_size=len(df)):
            continue
        unique_count = df[col].nunique(dropna=True)
        if 2 <= unique_count <= 5:
            detected.append(col)
            if len(detected) >= max_columns:
                break

    return detected[:max_columns]
"""Schema intelligence helpers for generic tabular datasets.

These utilities centralize lightweight heuristics that are reused by data
ingestion, generation, validation, and privacy analysis.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

import pandas as pd

# Keywords indicating explicit strong ID fields
STRONG_ID_EXACT = {
    "id", "uuid", "uid", "guid", "row_id", "record_id", "identifier", "ssn", "social_security",
}

STRONG_ID_SUFFIXES = (
    r"(^|_)id$",
    r"(^|_)uuid$",
    r"(^|_)uid$",
    r"(^|_)guid$",
    r"(^|_)ssn$",
)

# Explicit compound ID names (e.g. account_number, badge_no, policy_num)
KNOWN_ID_PATTERNS = (
    r"^(account|policy|badge|serial|license|customer|client|user|employee|student|patient|member|order|tracking|transaction|sensor|device|card|phone|mobile|invoice|receipt|ticket|passport|vehicle|driver)_(no|num|number|id|code|token|key)$",
    r"^(row|record|item|unit|case|sample)_(id|uuid|uid|guid|key|code)$",
)

# Measurement and quantity keywords/suffixes that should NOT be classified as identifiers
MEASUREMENT_PATTERNS = (
    r"(^|_)(distance|weight|amount|price|cost|fee|rate|percentage|pct|ratio|score|count|total|sum|avg|mean|median|min|max|days|hours|minutes|seconds|duration|temperature|voltage|current|power|energy|latitude|longitude|lat|lon|measurement|sensor|reading|value|qty|quantity)($|_)",
    r"(^|_)(km|kg|liters|litres|meters|metres|miles|grams|lbs|volts|hz|celsius|fahrenheit)($|_)",
)

# Strong outcome keywords (high confidence target)
STRONG_TARGET_KEYWORDS = {
    "target", "label", "outcome", "fraud", "is_fraud", "anomaly", "is_anomaly",
    "churn", "attrition", "delayed", "delay", "returned", "liked", "default",
    "default_status", "diagnosis", "placement_status", "attack_type", "survived",
    "converted", "readmitted", "failure", "risk_level", "risk",
}

# Moderate outcome keywords
MODERATE_TARGET_KEYWORDS = {
    "status", "result", "prediction", "response", "decision", "approved", "completed",
}

# Feature flag prefixes to penalize unless matching strong target
FEATURE_FLAG_PREFIXES = (
    "is_", "has_", "requires_", "can_", "should_", "use_", "allow_", "was_", "did_",
)

SENSITIVE_NAME_KEYWORDS = (
    "name", "email", "phone", "mobile", "address", "dob", "date_of_birth", "birth_date",
    "gender", "sex", "race", "ethnicity", "income", "salary", "credit", "account",
    "password", "secret", "token", "ssn", "social_security",
)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _name_matches(name: str, patterns: Iterable[str]) -> bool:
    normalized = _normalize_name(name)
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_datetime_like(series: pd.Series) -> bool:
    """Check if series is datetime or parseable date string."""
    if series is None or series.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    clean = series.dropna().astype(str).head(20)
    if len(clean) == 0:
        return False
    date_like = clean.str.match(
        r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
        r"|^\d{1,2}\s+\w{3}\s+\d{4}"
        r"|^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    )
    if date_like.mean() >= 0.8:
        try:
            pd.to_datetime(clean, errors="raise")
            return True
        except Exception:
            return False
    return False


def is_identifier_column(
    series: pd.Series,
    column_name: str,
    *,
    dataset_size: Optional[int] = None,
) -> bool:
    """Heuristically detect identifier-like columns using combined evidence.
    
    Protects continuous numeric measurements, datetimes, booleans, and constants
    while accurately capturing genuine ID columns across naming variations.
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

    # 1. Hard Protections
    # Constant or single value
    if n_unique <= 1:
        return False

    # Boolean or binary feature
    if pd.api.types.is_bool_dtype(series) or (n_unique == 2 and pd.api.types.is_numeric_dtype(series) and set(clean.unique()).issubset({0, 1, 0.0, 1.0})):
        return False

    # Datetime protection
    if is_datetime_like(series):
        return False

    # Continuous float protection (unless explicit strong ID name like "id", "uuid", "ssn")
    is_float = pd.api.types.is_float_dtype(series)
    is_explicit_id_name = name in STRONG_ID_EXACT or any(re.search(pat, name) for pat in (r"(^|_)id$", r"(^|_)uuid$", r"(^|_)uid$", r"(^|_)ssn$"))
    
    if is_float and not is_explicit_id_name:
        return False

    # Measurement / Metric name protection
    if any(re.search(pat, name) for pat in MEASUREMENT_PATTERNS) and not is_explicit_id_name:
        return False

    # 2. Strong Explicit Identifier Names
    if name in STRONG_ID_EXACT:
        return True

    if any(re.search(pat, name) for pat in KNOWN_ID_PATTERNS):
        # Allow repeated values (e.g. account_number in transactions, customer_id)
        if (pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and n_unique >= 2:
            return True

    if any(re.search(pat, name) for pat in STRONG_ID_SUFFIXES):
        # Suffix like _id, _uuid, _uid, _number, _num, _no, _code, _token, _key
        if any(re.search(pat, name) for pat in (r"(^|_)id$", r"(^|_)uuid$", r"(^|_)uid$", r"(^|_)ssn$")):
            if n_unique >= 2:
                return True
        if (pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            if unique_ratio >= 0.20 or n_unique >= 10:
                return True

    # 3. UUID / GUID / Hex Pattern Recognition in Strings
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = clean.astype(str).head(25)
        uuid_pat = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        hex_pat = r"^[0-9a-fA-F]{16,}$"
        if len(sample) > 0 and (sample.str.match(uuid_pat).mean() >= 0.6 or sample.str.match(hex_pat).mean() >= 0.6):
            return True

    # 4. Pure Integer Sequential / Index IDs with Very High Uniqueness
    if pd.api.types.is_integer_dtype(series) and size >= 20 and unique_ratio >= 0.98 and n_unique >= 20:
        if not any(re.search(pat, name) for pat in MEASUREMENT_PATTERNS):
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
    """Detect a reasonable default target column for generic tabular ML tasks using weighted scoring."""
    if df.empty:
        return None

    columns = list(df.columns)
    id_cols = get_identifier_columns(df)
    candidate_scores = []

    for idx, col in enumerate(columns):
        if col in id_cols:
            continue

        series = df[col]
        clean = series.dropna()
        if clean.empty:
            continue

        n_unique = clean.nunique()
        if n_unique < 2:
            continue

        # Reject datetime or non-target continuous float or high-cardinality as targets
        if is_datetime_like(series):
            continue
        if n_unique > 20:
            continue

        name = _normalize_name(col)
        score = 0.0

        # 1. Keyword scoring
        is_strong_target = name in STRONG_TARGET_KEYWORDS or any(kw == name or name.endswith(f"_{kw}") or name.startswith(f"{kw}_") for kw in STRONG_TARGET_KEYWORDS)
        is_moderate_target = name in MODERATE_TARGET_KEYWORDS or any(kw == name or name.endswith(f"_{kw}") or name.startswith(f"{kw}_") for kw in MODERATE_TARGET_KEYWORDS)

        # Disallow continuous float features from being treated as targets without explicit keyword match
        if pd.api.types.is_float_dtype(series) and not is_strong_target and not is_moderate_target:
            continue

        # Measurement names should not be targets unless explicitly matching outcome keywords
        if any(re.search(pat, name) for pat in MEASUREMENT_PATTERNS) and not is_strong_target:
            continue

        if is_strong_target:
            score += 80.0
        elif is_moderate_target:
            score += 40.0

        # 2. Feature flag penalty
        has_feature_flag_prefix = any(name.startswith(p) for p in FEATURE_FLAG_PREFIXES)
        if has_feature_flag_prefix and not is_strong_target:
            score -= 40.0

        # 3. Cardinality suitability
        if n_unique == 2:
            score += 20.0
        elif 3 <= n_unique <= 6:
            score += 10.0
        elif 7 <= n_unique <= 10:
            score += 5.0

        # 4. Weak position bonus (last column in dataframe)
        if idx == len(columns) - 1:
            score += 5.0

        candidate_scores.append((score, -idx, col))

    if not candidate_scores:
        return None

    candidate_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, _, best_col = candidate_scores[0]

    # Unsupervised threshold: must have positive confidence score >= 25.0
    if best_score < 25.0:
        return None

    return best_col


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
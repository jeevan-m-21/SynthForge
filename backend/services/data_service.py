"""
MediSynth.AI — Data Ingestion & Preprocessing Service
Handles CSV upload, validation, metadata detection, and preprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.config import UPLOAD_DIR
from backend.utils.logging_config import get_logger, audit_log
from backend.utils.security import generate_dataset_id, compute_data_fingerprint
from backend.models.database import register_dataset, datasets_store

logger = get_logger("data_service")


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column types for metadata: numeric, categorical, boolean, datetime."""
    types = {}
    for col in df.columns:
        if df[col].dtype in ["int64", "int32", "float64", "float32"]:
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if df[col].nunique() <= 2 and set(df[col].dropna().unique()).issubset({0, 1}):
                types[col] = "boolean"
            elif unique_ratio < 0.05 and df[col].nunique() <= 20:
                types[col] = "categorical"
            else:
                types[col] = "numerical"
        elif df[col].dtype == "bool":
            types[col] = "boolean"
        elif df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].dropna().head(10))
                types[col] = "datetime"
            except (ValueError, TypeError):
                types[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            types[col] = "categorical"
    return types


def preprocess_dataframe(df: pd.DataFrame,
                         column_types: Dict[str, str]) -> pd.DataFrame:
    """Clean and preprocess dataframe based on detected types."""
    result = df.copy()

    for col, ctype in column_types.items():
        if col not in result.columns:
            continue
        if ctype == "numerical":
            result[col] = pd.to_numeric(result[col], errors="coerce")
        elif ctype == "boolean":
            result[col] = result[col].astype(int)
        elif ctype == "categorical":
            result[col] = result[col].astype(str)
            result[col] = result[col].replace("nan", np.nan)

    return result


def ingest_csv(file_content: bytes, filename: str) -> Dict:
    """
    Ingest a CSV file: save, parse, detect types, register.

    Returns dataset metadata dict.
    """
    dataset_id = generate_dataset_id()
    fingerprint = compute_data_fingerprint(file_content)

    # Save raw file
    filepath = UPLOAD_DIR / f"{dataset_id}_{filename}"
    filepath.write_bytes(file_content)

    # Parse CSV
    df = pd.read_csv(filepath)

    # Drop ID-like columns (single unique values per row)
    id_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].dtype == "object":
            id_cols.append(col)
        elif col.lower() in ["id", "patient_id", "record_id", "index"]:
            id_cols.append(col)

    # Detect types
    column_types = detect_column_types(df)

    # Register in database
    register_dataset(
        dataset_id=dataset_id,
        filename=filename,
        filepath=str(filepath),
        num_rows=len(df),
        num_cols=len(df.columns),
        columns=list(df.columns),
        column_types=column_types,
        fingerprint=fingerprint,
    )

    audit_log(logger, "data_ingested", {
        "dataset_id": dataset_id,
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
    })

    # Build preview (first 5 rows)
    preview = df.head(5).to_dict(orient="records")

    # Compute basic stats
    stats = {}
    for col in df.columns:
        col_stats = {"type": column_types.get(col, "unknown")}
        if column_types.get(col) == "numerical":
            col_stats.update({
                "mean": round(float(df[col].mean()), 2) if not df[col].isna().all() else None,
                "std": round(float(df[col].std()), 2) if not df[col].isna().all() else None,
                "min": round(float(df[col].min()), 2) if not df[col].isna().all() else None,
                "max": round(float(df[col].max()), 2) if not df[col].isna().all() else None,
                "missing": int(df[col].isna().sum()),
            })
        elif column_types.get(col) == "categorical":
            col_stats.update({
                "unique": int(df[col].nunique()),
                "top": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                "missing": int(df[col].isna().sum()),
            })
        stats[col] = col_stats

    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "column_types": column_types,
        "id_columns": id_cols,
        "preview": preview,
        "stats": stats,
        "fingerprint": fingerprint[:16] + "...",
    }


def load_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
    """Load a registered dataset as DataFrame."""
    dataset = datasets_store.get(dataset_id)
    if not dataset:
        return None
    filepath = Path(dataset["filepath"])
    if not filepath.exists():
        return None
    return pd.read_csv(filepath)


def get_dataset_info(dataset_id: str) -> Optional[Dict]:
    """Get dataset metadata."""
    return datasets_store.get(dataset_id)


def list_datasets() -> List[Dict]:
    """List all registered datasets."""
    all_data = datasets_store.list_all()
    return list(all_data.values())

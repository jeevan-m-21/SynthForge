"""
MediSynth.AI — Synthetic Data Generator
Supports CTGAN, TVAE, and statistical fallback with DP noise integration.
"""
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from backend.config import (
    DEFAULT_MODEL_TYPE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_ROWS, MODELS_DIR, GENERATED_DIR
)
from backend.services.privacy_engine import (
    DPDataProcessor, PrivacyParams, PrivacyBudgetManager
)
from backend.services.data_service import load_dataset
from backend.utils.logging_config import get_logger, audit_log
from backend.utils.security import generate_job_id
from backend.models.database import create_job, update_job

logger = get_logger("generator")


class StatisticalGenerator:
    """
    Fallback statistical generator using Gaussian copula-like approach.
    No deep learning required — generates data by sampling from fitted
    marginal distributions and preserving correlations via copula.
    """

    def __init__(self):
        self.numeric_stats = {}
        self.categorical_probs = {}
        self.correlation_matrix = None
        self.columns = []
        self.column_types = {}

    def fit(self, df: pd.DataFrame):
        """Fit marginal distributions and correlation structure."""
        self.columns = list(df.columns)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Fit numeric marginals
        for col in numeric_cols:
            self.column_types[col] = "numeric"
            values = df[col].dropna()
            self.numeric_stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()) if values.std() > 0 else 0.01,
                "min": float(values.min()),
                "max": float(values.max()),
                "is_integer": df[col].dtype in ["int64", "int32"],
            }

        # Fit categorical marginals
        for col in cat_cols:
            self.column_types[col] = "categorical"
            vc = df[col].value_counts(normalize=True)
            self.categorical_probs[col] = {
                "categories": vc.index.tolist(),
                "probabilities": vc.values.tolist(),
            }

        # Compute correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            self.correlation_matrix = df[numeric_cols].corr().values
        else:
            self.correlation_matrix = None

    def sample(self, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data from fitted distributions."""
        data = {}
        numeric_cols = [c for c in self.columns if self.column_types.get(c) == "numeric"]
        cat_cols = [c for c in self.columns if self.column_types.get(c) == "categorical"]

        # Generate correlated numeric columns
        if self.correlation_matrix is not None and len(numeric_cols) > 1:
            # Use Cholesky decomposition for correlated sampling
            try:
                L = np.linalg.cholesky(
                    self.correlation_matrix +
                    np.eye(len(numeric_cols)) * 1e-6  # regularize
                )
                z = np.random.normal(0, 1, (num_rows, len(numeric_cols)))
                correlated = z @ L.T

                for i, col in enumerate(numeric_cols):
                    stats = self.numeric_stats[col]
                    values = correlated[:, i] * stats["std"] + stats["mean"]
                    values = np.clip(values, stats["min"], stats["max"])
                    if stats["is_integer"]:
                        values = np.round(values).astype(int)
                    data[col] = values
            except np.linalg.LinAlgError:
                # Fall back to independent sampling
                for col in numeric_cols:
                    stats = self.numeric_stats[col]
                    values = np.random.normal(stats["mean"], stats["std"], num_rows)
                    values = np.clip(values, stats["min"], stats["max"])
                    if stats["is_integer"]:
                        values = np.round(values).astype(int)
                    data[col] = values
        else:
            for col in numeric_cols:
                stats = self.numeric_stats[col]
                values = np.random.normal(stats["mean"], stats["std"], num_rows)
                values = np.clip(values, stats["min"], stats["max"])
                if stats["is_integer"]:
                    values = np.round(values).astype(int)
                data[col] = values

        # Generate categorical columns
        for col in cat_cols:
            probs = self.categorical_probs[col]
            data[col] = np.random.choice(
                probs["categories"],
                size=num_rows,
                p=probs["probabilities"],
            )

        # Preserve original column order
        return pd.DataFrame(data)[self.columns]


def _train_sdv_model(df: pd.DataFrame, model_type: str,
                     epochs: int, batch_size: int) -> object:
    """Train an SDV synthesizer (CTGAN or TVAE)."""
    try:
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        if model_type == "tvae":
            synthesizer = TVAESynthesizer(
                metadata,
                epochs=epochs,
                batch_size=batch_size,
            )
        else:  # ctgan
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=epochs,
                batch_size=batch_size,
            )

        synthesizer.fit(df)
        return synthesizer

    except ImportError:
        logger.warning("SDV not available, falling back to statistical generator")
        return None
    except Exception as e:
        logger.error(f"SDV training failed: {e}, falling back to statistical generator")
        return None


def generate_synthetic_data(
    dataset_id: str,
    num_rows: int = DEFAULT_NUM_ROWS,
    model_type: str = DEFAULT_MODEL_TYPE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    dp_mechanism: str = "gaussian",
    apply_dp: bool = True,
) -> Dict:
    """
    Full synthetic data generation pipeline:
    1. Load real data
    2. Train generative model (CTGAN/TVAE/Statistical)
    3. Generate synthetic samples
    4. Apply differential privacy
    5. Save and return results

    Returns generation result with metadata.
    """
    job_id = generate_job_id()
    create_job(job_id, dataset_id, "generation", {
        "num_rows": num_rows, "model_type": model_type,
        "epochs": epochs, "epsilon": epsilon, "delta": delta,
    })

    try:
        # Step 1: Load data
        update_job(job_id, status="running", progress=10)
        real_df = load_dataset(dataset_id)
        if real_df is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Remove ID columns
        id_cols = [c for c in real_df.columns
                   if c.lower() in ["id", "patient_id", "record_id", "index"]]
        working_df = real_df.drop(columns=id_cols, errors="ignore")

        # Step 2: Check privacy budget
        if apply_dp:
            can_spend, budget = PrivacyBudgetManager.spend(
                dataset_id, epsilon, delta, f"generate_{model_type}"
            )
            if not can_spend:
                raise ValueError(
                    f"Privacy budget exhausted. "
                    f"Remaining: ε={budget.remaining_epsilon:.4f}"
                )

        # Step 3: Train model
        update_job(job_id, progress=30)
        t_start = time.time()

        sdv_model = None
        stat_model = None

        if model_type in ["ctgan", "tvae"]:
            sdv_model = _train_sdv_model(working_df, model_type, epochs, batch_size)

        if sdv_model is None:
            # Fallback to statistical generator
            model_type = "statistical"
            stat_model = StatisticalGenerator()
            stat_model.fit(working_df)

        train_time = time.time() - t_start
        update_job(job_id, progress=70)

        # Step 4: Generate synthetic data
        if sdv_model:
            synthetic_df = sdv_model.sample(num_rows=num_rows)
        else:
            synthetic_df = stat_model.sample(num_rows)

        update_job(job_id, progress=85)

        # Step 5: Apply differential privacy
        dp_metadata = None
        if apply_dp:
            dp_processor = DPDataProcessor(PrivacyParams(
                epsilon=epsilon,
                delta=delta,
                mechanism=dp_mechanism,
                clip_bound=float(working_df.select_dtypes(include=[np.number]).values.max()
                                 - working_df.select_dtypes(include=[np.number]).values.min())
                if len(working_df.select_dtypes(include=[np.number]).columns) > 0 else 1.0,
            ))
            synthetic_df, dp_metadata = dp_processor.apply_dp(synthetic_df, working_df)

        # Step 6: Save synthetic data
        output_filename = f"{dataset_id}_{model_type}_{job_id}.csv"
        output_path = GENERATED_DIR / output_filename
        synthetic_df.to_csv(output_path, index=False)

        # Save model
        model_path = MODELS_DIR / f"{job_id}_model.pkl"
        try:
            if sdv_model:
                sdv_model.save(str(model_path))
            elif stat_model:
                with open(model_path, "wb") as f:
                    pickle.dump(stat_model, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")

        # Build result
        result = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "model_type": model_type,
            "num_rows_generated": len(synthetic_df),
            "num_columns": len(synthetic_df.columns),
            "columns": list(synthetic_df.columns),
            "output_file": str(output_path),
            "output_filename": output_filename,
            "training_time_seconds": round(train_time, 2),
            "dp_applied": apply_dp,
            "dp_metadata": dp_metadata,
            "preview": synthetic_df.head(5).to_dict(orient="records"),
            "privacy_budget": PrivacyBudgetManager.get_or_create(dataset_id).to_dict()
                if apply_dp else None,
        }

        update_job(job_id, status="completed", progress=100,
                   result=result, completed_at=time.time())

        audit_log(logger, "data_generated", {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "model": model_type,
            "rows": num_rows,
            "epsilon": epsilon,
        })

        return result

    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
        logger.error(f"Generation failed: {e}")
        raise

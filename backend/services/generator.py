"""
MediSynth.AI — Synthetic Data Generator
Supports CTGAN, TVAE, and statistical fallback with DP noise integration.
"""
import os
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
from backend.services.schema_intelligence import drop_identifier_columns
from backend.services.dataset_profiler import DatasetProfile, profile_dataframe, SemanticType
from backend.utils.logging_config import get_logger, audit_log
from backend.utils.security import generate_job_id
from backend.models.database import create_job, update_job

logger = get_logger("generator")


class StatisticalGenerator:
    """
    Profile-driven statistical generator using Gaussian copula-like approach.
    Samples from fitted marginal distributions, preserving numeric correlations via Cholesky,
    with explicit support for numeric, categorical, boolean, datetime, constant, text,
    and missing values.
    """

    def __init__(self):
        self.numeric_stats = {}
        self.categorical_probs = {}
        self.correlation_matrix = None
        self.columns = []
        self.column_types = {}
        self.boolean_stats = {}
        self.constant_stats = {}
        self.datetime_stats = {}
        self.text_stats = {}
        self.missing_rates = {}
        self._corr_cols = []

    def fit(self, df: pd.DataFrame, profile: Optional[DatasetProfile] = None):
        """Fit marginal distributions and correlation structure using DatasetProfile."""
        self.columns = list(df.columns)
        if profile is None:
            profile = profile_dataframe(df)

        col_profiles = {c.name: c for c in profile.columns}

        for col in self.columns:
            c_prof = col_profiles.get(col)
            sem_type = c_prof.semantic_type if c_prof else None
            clean = df[col].dropna()
            self.missing_rates[col] = float(df[col].isna().mean())

            if sem_type == SemanticType.CONSTANT or (clean.nunique() <= 1 and not clean.empty):
                self.column_types[col] = "constant"
                val = clean.iloc[0] if not clean.empty else None
                self.constant_stats[col] = {"value": val}
                self.categorical_probs[col] = {
                    "categories": [str(val) if val is not None else ""],
                    "probabilities": [1.0],
                }

            elif sem_type == SemanticType.BOOLEAN or pd.api.types.is_bool_dtype(df[col]):
                self.column_types[col] = "boolean"
                if pd.api.types.is_bool_dtype(df[col]):
                    rep_type = "bool"
                    p_true = float((clean == True).mean()) if not clean.empty else 0.5
                elif pd.api.types.is_numeric_dtype(df[col]):
                    rep_type = "int" if df[col].dtype in ["int64", "int32"] else "float"
                    p_true = float((clean == 1).mean()) if not clean.empty else 0.5
                else:
                    rep_type = "str"
                    true_vals = {"true", "yes", "1", "t", "y"}
                    p_true = float(clean.astype(str).str.lower().isin(true_vals).mean()) if not clean.empty else 0.5

                self.boolean_stats[col] = {"p_true": p_true, "rep_type": rep_type}
                self.categorical_probs[col] = {
                    "categories": [True, False] if rep_type == "bool" else ([1, 0] if rep_type == "int" else ["1", "0"]),
                    "probabilities": [p_true, 1.0 - p_true],
                }

            elif sem_type == SemanticType.DATETIME or pd.api.types.is_datetime64_any_dtype(df[col]):
                self.column_types[col] = "datetime"
                dt_clean = pd.to_datetime(clean, errors="coerce").dropna()
                if not dt_clean.empty:
                    timestamps = (dt_clean.astype("int64") // 10**9).astype(float)
                    self.datetime_stats[col] = {
                        "mean": float(timestamps.mean()),
                        "std": float(timestamps.std()) if len(timestamps) > 1 and timestamps.std() > 0 else 1.0,
                        "min": float(timestamps.min()),
                        "max": float(timestamps.max()),
                        "is_native_dt": pd.api.types.is_datetime64_any_dtype(df[col]),
                        "sample_format": str(clean.iloc[0]) if not clean.empty else None,
                    }
                else:
                    self.datetime_stats[col] = {
                        "mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0,
                        "is_native_dt": False, "sample_format": None,
                    }

            elif sem_type == SemanticType.TEXT:
                self.column_types[col] = "text"
                samples = clean.astype(str).tolist()
                pool = list(dict.fromkeys(samples))[:100]
                if not pool:
                    pool = [""]
                self.text_stats[col] = {"pool": pool}
                vc = clean.value_counts(normalize=True).head(50)
                self.categorical_probs[col] = {
                    "categories": vc.index.tolist() if not vc.empty else [""],
                    "probabilities": vc.values.tolist() if not vc.empty else [1.0],
                }

            elif sem_type == SemanticType.NUMERIC or pd.api.types.is_numeric_dtype(df[col]):
                self.column_types[col] = "numeric"
                self.numeric_stats[col] = {
                    "mean": float(clean.mean()) if not clean.empty else 0.0,
                    "std": float(clean.std()) if len(clean) > 1 and clean.std() > 0 else 0.01,
                    "min": float(clean.min()) if not clean.empty else 0.0,
                    "max": float(clean.max()) if not clean.empty else 1.0,
                    "is_integer": df[col].dtype in ["int64", "int32"],
                }

            else:
                self.column_types[col] = "categorical"
                vc = clean.value_counts(normalize=True)
                self.categorical_probs[col] = {
                    "categories": vc.index.tolist() if not vc.empty else [""],
                    "probabilities": vc.values.tolist() if not vc.empty else [1.0],
                }

        # Correlation structure across numeric and datetime columns
        self._corr_cols = [
            c for c in self.columns
            if self.column_types.get(c) in ["numeric", "datetime"]
        ]
        if len(self._corr_cols) > 1:
            corr_df = pd.DataFrame()
            for c in self._corr_cols:
                if self.column_types[c] == "numeric":
                    corr_df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    dt_s = pd.to_datetime(df[c], errors="coerce")
                    corr_df[c] = (dt_s.astype("int64") // 10**9).astype(float)
            corr_matrix = corr_df.corr().fillna(0.0).values
            if not np.isnan(corr_matrix).any():
                self.correlation_matrix = corr_matrix
            else:
                self.correlation_matrix = None
        else:
            self.correlation_matrix = None

    def sample(self, num_rows: int, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic data from fitted distributions.
        Uses an isolated local RNG (np.random.default_rng) for reproducible sampling without modifying global state.
        """
        rng = np.random.default_rng(seed)
        data = {}

        # Correlated sampling for numeric & datetime columns
        if self.correlation_matrix is not None and len(self._corr_cols) > 1:
            try:
                L = np.linalg.cholesky(
                    self.correlation_matrix +
                    np.eye(len(self._corr_cols)) * 1e-6
                )
                z = rng.normal(0, 1, (num_rows, len(self._corr_cols)))
                correlated = z @ L.T

                for i, col in enumerate(self._corr_cols):
                    ctype = self.column_types.get(col, "numeric")
                    if ctype == "numeric":
                        stats = self.numeric_stats[col]
                        vals = correlated[:, i] * stats["std"] + stats["mean"]
                        vals = np.clip(vals, stats["min"], stats["max"])
                        if stats["is_integer"]:
                            vals = np.round(vals).astype(int)
                        data[col] = vals
                    elif ctype == "datetime":
                        stats = self.datetime_stats[col]
                        ts_vals = correlated[:, i] * stats["std"] + stats["mean"]
                        ts_vals = np.clip(ts_vals, stats["min"], stats["max"])
                        dt_vals = pd.to_datetime(ts_vals, unit="s")
                        if stats.get("is_native_dt"):
                            data[col] = dt_vals
                        else:
                            fmt = "%Y-%m-%d %H:%M:%S" if " " in (stats.get("sample_format") or "") else "%Y-%m-%d"
                            data[col] = dt_vals.strftime(fmt)
            except np.linalg.LinAlgError:
                pass

        # Sample remaining columns or independent fallbacks
        for col in self.columns:
            if col in data:
                continue
            ctype = self.column_types.get(col, "numeric")

            if ctype == "numeric":
                stats = self.numeric_stats.get(col, {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0, "is_integer": False})
                vals = rng.normal(stats["mean"], stats["std"], num_rows)
                vals = np.clip(vals, stats["min"], stats["max"])
                if stats["is_integer"]:
                    vals = np.round(vals).astype(int)
                data[col] = vals

            elif ctype == "datetime":
                stats = self.datetime_stats.get(col, {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0, "is_native_dt": False, "sample_format": None})
                ts_vals = rng.normal(stats["mean"], stats["std"], num_rows)
                ts_vals = np.clip(ts_vals, stats["min"], stats["max"])
                dt_vals = pd.to_datetime(ts_vals, unit="s")
                if stats.get("is_native_dt"):
                    data[col] = dt_vals
                else:
                    fmt = "%Y-%m-%d %H:%M:%S" if " " in (stats.get("sample_format") or "") else "%Y-%m-%d"
                    data[col] = dt_vals.strftime(fmt)

            elif ctype == "boolean":
                stats = self.boolean_stats.get(col, {"p_true": 0.5, "rep_type": "bool"})
                p = max(0.0, min(1.0, stats["p_true"]))
                draws = rng.random(num_rows) < p
                rep = stats["rep_type"]
                if rep == "bool":
                    data[col] = draws
                elif rep == "int":
                    data[col] = draws.astype(int)
                elif rep == "float":
                    data[col] = draws.astype(float)
                else:
                    data[col] = np.where(draws, "1", "0")

            elif ctype == "constant":
                val = self.constant_stats.get(col, {}).get("value", None)
                data[col] = [val] * num_rows

            elif ctype == "text":
                pool = self.text_stats.get(col, {}).get("pool", [""])
                data[col] = rng.choice(pool, size=num_rows)

            elif ctype == "categorical":
                probs = self.categorical_probs.get(col, {"categories": [""], "probabilities": [1.0]})
                cats = probs["categories"]
                p_vals = probs["probabilities"]
                p_arr = np.array(p_vals, dtype=float)
                p_arr = p_arr / p_arr.sum() if p_arr.sum() > 0 else None
                data[col] = rng.choice(cats, size=num_rows, p=p_arr)

        res_df = pd.DataFrame(data)[self.columns]

        # Apply missingness rates
        for col in self.columns:
            rate = self.missing_rates.get(col, 0.0)
            if 0.0 < rate < 1.0:
                mask = rng.random(num_rows) < rate
                res_df.loc[mask, col] = np.nan

        return res_df


def _train_sdv_model(df: pd.DataFrame, model_type: str,
                     epochs: int, batch_size: int,
                     seed: Optional[int] = None) -> object:
    """Train an SDV synthesizer (CTGAN or TVAE) with controlled CPU seeding when seed is provided."""
    try:
        if seed is not None:
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass

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
    seed: Optional[int] = None,
) -> Dict:
    """
    Full profile-driven synthetic data generation pipeline:
    1. Load real data and profile schema
    2. Exclude detected identifier columns
    3. Train generative model (CTGAN/TVAE/Statistical) with optional seed
    4. Generate synthetic samples
    5. Apply differential privacy preserving target column
    6. Save and return results with full job metadata
    """
    job_id = generate_job_id()
    create_job(job_id, dataset_id, "generation", {
        "num_rows": num_rows, "model_type": model_type,
        "epochs": epochs, "epsilon": epsilon, "delta": delta,
        "seed": seed, "reproducible_run": seed is not None,
    })

    try:
        # Step 1: Load data and compute DatasetProfile
        update_job(job_id, status="running", progress=10)
        real_df = load_dataset(dataset_id)
        if real_df is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        profile = profile_dataframe(real_df, dataset_name=dataset_id)
        id_cols = profile.detected_metadata.identifier_columns
        target_col = profile.detected_metadata.target_column
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
            sdv_model = _train_sdv_model(working_df, model_type, epochs, batch_size, seed=seed)

        if sdv_model is None:
            model_type = "statistical"
            stat_model = StatisticalGenerator()
            stat_model.fit(working_df, profile=profile)

        train_time = time.time() - t_start
        update_job(job_id, progress=70)

        # Step 4: Generate synthetic data
        if sdv_model:
            if seed is not None:
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                except ImportError:
                    pass
            synthetic_df = sdv_model.sample(num_rows=num_rows)
        else:
            synthetic_df = stat_model.sample(num_rows, seed=seed)

        update_job(job_id, progress=85)

        # Step 5: Apply differential privacy
        dp_metadata = None
        if apply_dp:
            num_cols = working_df.select_dtypes(include=[np.number]).columns
            clip_bound = 1.0
            if len(num_cols) > 0:
                flat_vals = working_df[num_cols].values.flatten()
                valid_vals = flat_vals[~np.isnan(flat_vals)]
                if len(valid_vals) > 0:
                    clip_bound = float(valid_vals.max() - valid_vals.min()) or 1.0

            dp_processor = DPDataProcessor(PrivacyParams(
                epsilon=epsilon,
                delta=delta,
                mechanism=dp_mechanism,
                clip_bound=clip_bound,
            ))
            synthetic_df, dp_metadata = dp_processor.apply_dp(
                synthetic_df, working_df, target_column=target_col, seed=seed
            )

        # Step 6: Save synthetic data atomically
        output_filename = f"{dataset_id}_{model_type}_{job_id}.csv"
        output_path = GENERATED_DIR / output_filename
        temp_csv_path = GENERATED_DIR / f".tmp_{job_id}_{output_filename}"
        try:
            synthetic_df.to_csv(temp_csv_path, index=False)
            os.replace(temp_csv_path, output_path)
        except Exception:
            if temp_csv_path.exists():
                try:
                    temp_csv_path.unlink()
                except OSError:
                    pass
            raise

        # Save model atomically
        model_path = MODELS_DIR / f"{job_id}_model.pkl"
        temp_model_path = MODELS_DIR / f".tmp_{job_id}_model.pkl"
        try:
            if sdv_model:
                sdv_model.save(str(temp_model_path))
                os.replace(temp_model_path, model_path)
            elif stat_model:
                with open(temp_model_path, "wb") as f:
                    pickle.dump(stat_model, f)
                os.replace(temp_model_path, model_path)
        except Exception as e:
            if temp_model_path.exists():
                try:
                    temp_model_path.unlink()
                except OSError:
                    pass
            logger.warning(f"Could not save model: {e}")

        # Build result with full job metadata
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
            "seed": seed,
            "reproducible_run": seed is not None,
            "epochs": epochs if model_type in ["ctgan", "tvae"] else None,
            "batch_size": batch_size if model_type in ["ctgan", "tvae"] else None,
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
            "seed": seed,
            "reproducible_run": seed is not None,
        })

        return result

        return result

    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
        logger.error(f"Generation failed: {e}")
        raise

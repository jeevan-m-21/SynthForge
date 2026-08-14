"""
MediSynth.AI — Federated Learning Orchestrator
Implements FedAvg for multi-hospital synthetic data generation
without sharing raw data.

Key Properties:
- Order-independent aggregation (A + B == B + A)
- No raw data leaves hospital premises
- Optional DP noise on model updates
- Weighted averaging based on dataset size
"""
import numpy as np
import pandas as pd
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from backend.services.generator import StatisticalGenerator
from backend.services.privacy_engine import GaussianMechanism, PrivacyParams
from backend.utils.logging_config import get_logger, audit_log
from backend.utils.security import generate_federation_id
from backend.config import FL_DEFAULT_ROUNDS, FL_DEFAULT_LOCAL_EPOCHS

logger = get_logger("federated_learning")


@dataclass
class HospitalNode:
    """Represents a single hospital in the federation."""
    hospital_id: str
    name: str
    num_records: int = 0
    columns: List[str] = field(default_factory=list)
    local_model: Optional[Dict] = None
    _data: Optional[pd.DataFrame] = None

    def set_data(self, df: pd.DataFrame):
        """Store hospital data (stays local, never transmitted)."""
        self._data = df
        self.num_records = len(df)
        self.columns = list(df.columns)

    def get_data(self) -> Optional[pd.DataFrame]:
        return self._data

    def to_dict(self) -> Dict:
        return {
            "hospital_id": self.hospital_id,
            "name": self.name,
            "num_records": self.num_records,
            "columns": self.columns,
            "has_model": self.local_model is not None,
        }


@dataclass
class FederationState:
    """Tracks federation state across training rounds."""
    federation_id: str
    hospitals: Dict[str, HospitalNode] = field(default_factory=dict)
    global_model: Optional[Dict] = None
    rounds_completed: int = 0
    total_rounds: int = FL_DEFAULT_ROUNDS
    status: str = "initialized"
    history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "federation_id": self.federation_id,
            "num_hospitals": len(self.hospitals),
            "hospitals": {k: v.to_dict() for k, v in self.hospitals.items()},
            "rounds_completed": self.rounds_completed,
            "total_rounds": self.total_rounds,
            "status": self.status,
            "has_global_model": self.global_model is not None,
            "history": self.history[-10:],
        }


def _extract_model_params(generator: StatisticalGenerator) -> Dict:
    """Extract model parameters as a dictionary for federated exchange."""
    params = {
        "numeric_stats": {},
        "categorical_probs": {},
        "correlation_matrix": None,
        "columns": generator.columns,
        "column_types": generator.column_types,
    }

    for col, stats in generator.numeric_stats.items():
        params["numeric_stats"][col] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "is_integer": stats["is_integer"],
        }

    for col, probs in generator.categorical_probs.items():
        params["categorical_probs"][col] = {
            "categories": probs["categories"],
            "probabilities": probs["probabilities"],
        }

    if generator.correlation_matrix is not None:
        params["correlation_matrix"] = generator.correlation_matrix.tolist()

    return params


def _params_to_generator(params: Dict) -> StatisticalGenerator:
    """Reconstruct a StatisticalGenerator from federated parameters."""
    gen = StatisticalGenerator()
    gen.columns = params["columns"]
    gen.column_types = params["column_types"]

    for col, stats in params["numeric_stats"].items():
        gen.numeric_stats[col] = stats

    for col, probs in params["categorical_probs"].items():
        gen.categorical_probs[col] = probs

    if params["correlation_matrix"] is not None:
        gen.correlation_matrix = np.array(params["correlation_matrix"])

    return gen


def _weighted_average_params(
    param_list: List[Dict],
    weights: List[float],
    dp_noise_sigma: float = 0.0,
) -> Dict:
    """
    Federated Averaging (FedAvg) of model parameters.

    This is ORDER-INDEPENDENT: A + B == B + A because weighted averaging
    is commutative and associative.

    Args:
        param_list: list of model parameter dicts from each hospital
        weights: relative weights (proportional to dataset size)
        dp_noise_sigma: optional Gaussian noise to add for DP

    Returns:
        Averaged global model parameters.
    """
    if not param_list:
        raise ValueError("No parameters to average")

    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    # Start with first param structure
    global_params = {
        "columns": param_list[0]["columns"],
        "column_types": param_list[0]["column_types"],
        "numeric_stats": {},
        "categorical_probs": {},
        "correlation_matrix": None,
    }

    # Average numeric stats
    all_numeric_cols = set()
    for params in param_list:
        all_numeric_cols.update(params["numeric_stats"].keys())

    for col in all_numeric_cols:
        means = []
        stds = []
        mins = []
        maxs = []
        col_weights = []

        for params, w in zip(param_list, norm_weights):
            if col in params["numeric_stats"]:
                stats = params["numeric_stats"][col]
                means.append(stats["mean"] * w)
                stds.append(stats["std"] * w)
                mins.append(stats["min"])
                maxs.append(stats["max"])
                col_weights.append(w)

        if means:
            avg_mean = sum(means)
            avg_std = sum(stds)

            # Add DP noise if requested
            if dp_noise_sigma > 0:
                avg_mean += np.random.normal(0, dp_noise_sigma)
                avg_std = abs(avg_std + np.random.normal(0, dp_noise_sigma * 0.5))

            is_integer = param_list[0]["numeric_stats"].get(col, {}).get("is_integer", False)
            global_params["numeric_stats"][col] = {
                "mean": float(avg_mean),
                "std": max(0.01, float(avg_std)),
                "min": float(min(mins)),
                "max": float(max(maxs)),
                "is_integer": is_integer,
            }

    # Average categorical probabilities
    all_cat_cols = set()
    for params in param_list:
        all_cat_cols.update(params["categorical_probs"].keys())

    for col in all_cat_cols:
        all_categories = set()
        for params in param_list:
            if col in params["categorical_probs"]:
                all_categories.update(params["categorical_probs"][col]["categories"])
        all_categories = sorted(all_categories)

        avg_probs = np.zeros(len(all_categories))
        for params, w in zip(param_list, norm_weights):
            if col in params["categorical_probs"]:
                cats = params["categorical_probs"][col]["categories"]
                probs = params["categorical_probs"][col]["probabilities"]
                for cat, prob in zip(cats, probs):
                    idx = all_categories.index(cat)
                    avg_probs[idx] += prob * w

        # Normalize and add DP noise
        if dp_noise_sigma > 0:
            noise = np.abs(np.random.normal(0, dp_noise_sigma * 0.1, size=len(avg_probs)))
            avg_probs += noise
        avg_probs = avg_probs / avg_probs.sum()

        global_params["categorical_probs"][col] = {
            "categories": all_categories,
            "probabilities": avg_probs.tolist(),
        }

    # Average correlation matrices
    matrices = []
    for params, w in zip(param_list, norm_weights):
        if params["correlation_matrix"] is not None:
            matrices.append(np.array(params["correlation_matrix"]) * w)

    if matrices:
        avg_corr = sum(matrices)
        if dp_noise_sigma > 0:
            noise = np.random.normal(0, dp_noise_sigma * 0.1, size=avg_corr.shape)
            noise = (noise + noise.T) / 2  # symmetric
            avg_corr += noise
            # Re-clip to valid correlation range
            np.fill_diagonal(avg_corr, 1.0)
            avg_corr = np.clip(avg_corr, -1, 1)
        global_params["correlation_matrix"] = avg_corr.tolist()

    return global_params


# ──────────────────────────────────────────────
# Federation Manager
# ──────────────────────────────────────────────
class FederationManager:
    """Manages federated learning federations."""

    _federations: Dict[str, FederationState] = {}

    @classmethod
    def create_federation(cls, total_rounds: int = FL_DEFAULT_ROUNDS) -> FederationState:
        """Create a new federation."""
        fed_id = generate_federation_id()
        federation = FederationState(
            federation_id=fed_id,
            total_rounds=total_rounds,
        )
        cls._federations[fed_id] = federation

        audit_log(logger, "federation_created", {"federation_id": fed_id, "rounds": total_rounds})
        return federation

    @classmethod
    def get_federation(cls, federation_id: str) -> Optional[FederationState]:
        return cls._federations.get(federation_id)

    @classmethod
    def add_hospital(cls, federation_id: str, hospital_id: str,
                     name: str, data: pd.DataFrame) -> Dict:
        """Add a hospital's data to the federation."""
        fed = cls._federations.get(federation_id)
        if not fed:
            raise ValueError(f"Federation {federation_id} not found")

        node = HospitalNode(hospital_id=hospital_id, name=name)
        node.set_data(data)
        fed.hospitals[hospital_id] = node

        audit_log(logger, "hospital_added", {
            "federation_id": federation_id,
            "hospital_id": hospital_id,
            "records": len(data),
        })

        return node.to_dict()

    @classmethod
    def run_federated_training(
        cls,
        federation_id: str,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        apply_dp_to_updates: bool = True,
    ) -> Dict:
        """
        Execute federated training across all hospitals.

        1. Each hospital trains a local model on its own data
        2. Model parameters (NOT data) are extracted
        3. Parameters are averaged using FedAvg (weighted by dataset size)
        4. Optional: DP noise is added to parameter updates
        5. Global model is built from averaged parameters

        Returns federation state with training history.
        """
        fed = cls._federations.get(federation_id)
        if not fed:
            raise ValueError(f"Federation {federation_id} not found")

        if len(fed.hospitals) < 2:
            raise ValueError("Need at least 2 hospitals for federated learning")

        fed.status = "training"
        logger.info(f"Starting federated training with {len(fed.hospitals)} hospitals")

        # Compute DP noise level
        dp_sigma = 0.0
        if apply_dp_to_updates and dp_epsilon > 0:
            dp_sigma = GaussianMechanism.compute_sigma(1.0, dp_epsilon, dp_delta)

        for round_num in range(1, fed.total_rounds + 1):
            logger.info(f"Federation round {round_num}/{fed.total_rounds}")

            local_params = []
            weights = []

            # Step 1: Local training at each hospital
            for hosp_id, node in fed.hospitals.items():
                data = node.get_data()
                if data is None or len(data) == 0:
                    continue

                # Train local model
                local_gen = StatisticalGenerator()
                local_gen.fit(data)

                # Extract parameters (NEVER send raw data)
                params = _extract_model_params(local_gen)
                node.local_model = params
                local_params.append(params)
                weights.append(node.num_records)

                logger.info(f"  Hospital {node.name}: trained on {node.num_records} records")

            if not local_params:
                raise ValueError("No hospitals provided valid data")

            # Step 2: Federated averaging (order-independent)
            global_params = _weighted_average_params(
                local_params, weights, dp_noise_sigma=dp_sigma
            )

            fed.global_model = global_params
            fed.rounds_completed = round_num

            round_info = {
                "round": round_num,
                "hospitals_participated": len(local_params),
                "total_records": sum(weights),
                "dp_noise_sigma": round(dp_sigma, 6),
            }
            fed.history.append(round_info)

        fed.status = "completed"

        # Verify order-independence (A+B == B+A)
        if len(local_params) >= 2:
            forward = _weighted_average_params(local_params, weights, dp_noise_sigma=0)
            reverse = _weighted_average_params(
                list(reversed(local_params)),
                list(reversed(weights)),
                dp_noise_sigma=0,
            )
            # Check a few numeric stats match
            order_independent = True
            for col in forward["numeric_stats"]:
                if col in reverse["numeric_stats"]:
                    diff = abs(forward["numeric_stats"][col]["mean"] -
                              reverse["numeric_stats"][col]["mean"])
                    if diff > 1e-10:
                        order_independent = False
                        break
        else:
            order_independent = True

        audit_log(logger, "federation_trained", {
            "federation_id": federation_id,
            "rounds": fed.rounds_completed,
            "hospitals": len(fed.hospitals),
            "order_independent": order_independent,
        })

        result = fed.to_dict()
        result["order_independent_verified"] = order_independent
        result["dp_applied"] = apply_dp_to_updates
        result["dp_epsilon"] = dp_epsilon
        result["dp_delta"] = dp_delta
        return result

    @classmethod
    def generate_from_federation(
        cls,
        federation_id: str,
        num_rows: int = 1000,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Generate synthetic data from the global federated model."""
        fed = cls._federations.get(federation_id)
        if not fed:
            raise ValueError(f"Federation {federation_id} not found")
        if fed.global_model is None:
            raise ValueError("No global model available. Run training first.")

        # Reconstruct generator from global parameters
        generator = _params_to_generator(fed.global_model)
        synthetic_df = generator.sample(num_rows)

        metadata = {
            "federation_id": federation_id,
            "num_hospitals": len(fed.hospitals),
            "rounds_completed": fed.rounds_completed,
            "num_rows": num_rows,
            "hospitals": [
                {"name": h.name, "records": h.num_records}
                for h in fed.hospitals.values()
            ],
        }

        return synthetic_df, metadata

    @classmethod
    def list_federations(cls) -> List[Dict]:
        return [f.to_dict() for f in cls._federations.values()]

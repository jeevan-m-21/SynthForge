"""
MediSynth.AI — Pydantic Request/Response Schemas
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
class DatasetInfoResponse(BaseModel):
    dataset_id: str
    filename: str
    num_rows: int
    num_cols: int
    columns: List[str]
    column_types: Dict[str, str]
    preview: Optional[List[Dict]] = None
    stats: Optional[Dict] = None

class DatasetListResponse(BaseModel):
    datasets: List[Dict]


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────
class GenerateRequest(BaseModel):
    dataset_id: str
    num_rows: int = Field(default=1000, ge=10, le=1000000)
    model_type: str = Field(default="statistical", pattern="^(ctgan|tvae|statistical)$")
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=500, ge=32, le=10000)
    epsilon: float = Field(default=1.0, gt=0, le=100)
    delta: float = Field(default=1e-5, gt=0, lt=1)
    dp_mechanism: str = Field(default="gaussian", pattern="^(gaussian|laplace)$")
    apply_dp: bool = True


class GenerateResponse(BaseModel):
    job_id: str
    dataset_id: str
    model_type: str
    num_rows_generated: int
    dp_applied: bool
    dp_metadata: Optional[Dict] = None
    privacy_budget: Optional[Dict] = None
    preview: Optional[List[Dict]] = None
    training_time_seconds: Optional[float] = None


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
class ValidateStatisticalRequest(BaseModel):
    dataset_id: str
    synthetic_job_id: Optional[str] = None
    synthetic_file: Optional[str] = None


class ValidateMLRequest(BaseModel):
    dataset_id: str
    synthetic_job_id: Optional[str] = None
    synthetic_file: Optional[str] = None
    target_column: Optional[str] = None


# ──────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────
class AttackSimulationRequest(BaseModel):
    dataset_id: str
    synthetic_job_id: Optional[str] = None
    synthetic_file: Optional[str] = None


# ──────────────────────────────────────────────
# Federated Learning
# ──────────────────────────────────────────────
class CreateFederationRequest(BaseModel):
    total_rounds: int = Field(default=5, ge=1, le=50)


class FederatedTrainRequest(BaseModel):
    federation_id: str
    dp_epsilon: float = Field(default=1.0, gt=0, le=100)
    dp_delta: float = Field(default=1e-5, gt=0, lt=1)
    apply_dp: bool = True


class FederatedGenerateRequest(BaseModel):
    federation_id: str
    num_rows: int = Field(default=1000, ge=10, le=1000000)


# ──────────────────────────────────────────────
# Privacy Budget
# ──────────────────────────────────────────────
class PrivacyBudgetResponse(BaseModel):
    dataset_id: str
    total_epsilon_used: float
    remaining_epsilon: float
    utilization_pct: float
    warning_level: Optional[str]
    history: List[Dict]


# ──────────────────────────────────────────────
# Generic
# ──────────────────────────────────────────────
class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None

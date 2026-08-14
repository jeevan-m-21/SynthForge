"""
MediSynth.AI — API Routes
All REST endpoints for the system.
"""
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from typing import Any, Optional

from backend.api.schemas import (
    GenerateRequest, ValidateStatisticalRequest, ValidateMLRequest,
    AttackSimulationRequest, CreateFederationRequest, FederatedTrainRequest,
    FederatedGenerateRequest, StatusResponse,
)
from backend.services import data_service
from backend.services.generator import generate_synthetic_data
from backend.services.statistical_validator import validate_statistical
from backend.services.ml_validator import validate_ml_utility
from backend.services.attack_simulator import run_all_attacks
from backend.services.privacy_engine import PrivacyBudgetManager
from backend.services.federated_learning import FederationManager
from backend.models.database import (
    jobs_store, reports_store, save_report, datasets_store
)
from backend.utils.security import generate_job_id
from backend.config import GENERATED_DIR

router = APIRouter(prefix="/api")


def sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python natives."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ──────────────────────────────────────────────
# Data Endpoints
# ──────────────────────────────────────────────
@router.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload a CSV file for synthetic data generation."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    try:
        result = data_service.ingest_csv(content, file.filename)
        return sanitize({"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {str(e)}")


@router.get("/data/list")
async def list_datasets():
    """List all uploaded datasets."""
    datasets = data_service.list_datasets()
    return {"status": "success", "data": datasets}


@router.get("/data/info/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get detailed info about a dataset."""
    info = data_service.get_dataset_info(dataset_id)
    if not info:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    return {"status": "success", "data": info}


@router.get("/data/sample")
async def load_sample_data():
    """Load the built-in sample healthcare dataset."""
    from backend.config import DATA_DIR
    sample_path = DATA_DIR / "sample" / "healthcare_data.csv"
    if not sample_path.exists():
        # Generate sample data on-the-fly
        _generate_sample_data(sample_path)

    content = sample_path.read_bytes()
    result = data_service.ingest_csv(content, "healthcare_data.csv")
    return sanitize({"status": "success", "data": result})


def _generate_sample_data(path: Path):
    """Generate realistic sample healthcare data."""
    np.random = __import__("numpy").random
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "age": np.random.normal(55, 15, n).clip(18, 95).astype(int),
        "gender": np.random.choice(["Male", "Female"], n, p=[0.48, 0.52]),
        "blood_pressure_systolic": np.random.normal(130, 20, n).clip(80, 200).astype(int),
        "blood_pressure_diastolic": np.random.normal(82, 12, n).clip(50, 130).astype(int),
        "cholesterol": np.random.normal(200, 40, n).clip(100, 400).astype(int),
        "bmi": np.random.normal(27, 5, n).clip(15, 50).round(1),
        "glucose": np.random.normal(110, 30, n).clip(60, 300).astype(int),
        "heart_rate": np.random.normal(75, 12, n).clip(45, 130).astype(int),
        "smoking_status": np.random.choice(
            ["Never", "Former", "Current"], n, p=[0.45, 0.30, 0.25]
        ),
    })

    # Correlated targets
    risk = (
        (data["age"] > 55).astype(float) * 0.3 +
        (data["bmi"] > 30).astype(float) * 0.2 +
        (data["cholesterol"] > 240).astype(float) * 0.15 +
        (data["blood_pressure_systolic"] > 140).astype(float) * 0.2 +
        (data["glucose"] > 140).astype(float) * 0.15 +
        (data["smoking_status"] == "Current").astype(float) * 0.2
    )

    data["diabetes"] = (risk + np.random.normal(0, 0.15, n) > 0.45).astype(int)
    data["hypertension"] = (
        (data["blood_pressure_systolic"] > 140).astype(int) |
        (risk + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    )
    data["heart_disease"] = (risk + np.random.normal(0, 0.2, n) > 0.55).astype(int)

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


import numpy as np  # noqa: E402


# ──────────────────────────────────────────────
# Generation Endpoints
# ──────────────────────────────────────────────
@router.post("/generate")
async def generate(req: GenerateRequest):
    """Generate synthetic data from a dataset."""
    info = data_service.get_dataset_info(req.dataset_id)
    if not info:
        raise HTTPException(404, f"Dataset {req.dataset_id} not found")

    try:
        result = generate_synthetic_data(
            dataset_id=req.dataset_id,
            num_rows=req.num_rows,
            model_type=req.model_type,
            epochs=req.epochs,
            batch_size=req.batch_size,
            epsilon=req.epsilon,
            delta=req.delta,
            dp_mechanism=req.dp_mechanism,
            apply_dp=req.apply_dp,
        )
        return sanitize({"status": "success", "data": result})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


@router.get("/generate/download/{job_id}")
async def download_synthetic(job_id: str):
    """Download generated synthetic data as CSV."""
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.get("status") != "completed":
        raise HTTPException(400, "Job not completed yet")

    filepath = Path(job["result"]["output_file"])
    if not filepath.exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        filepath,
        media_type="text/csv",
        filename=job["result"]["output_filename"],
    )


@router.get("/generate/jobs")
async def list_jobs():
    """List all generation jobs."""
    return {"status": "success", "data": list(jobs_store.list_all().values())}


# ──────────────────────────────────────────────
# Privacy Budget Endpoints
# ──────────────────────────────────────────────
@router.get("/privacy/budget/{dataset_id}")
async def get_privacy_budget(dataset_id: str):
    """Get privacy budget status for a dataset."""
    budget = PrivacyBudgetManager.get_budget(dataset_id)
    if budget:
        return {"status": "success", "data": budget.to_dict()}

    # Check if dataset exists at all
    info = data_service.get_dataset_info(dataset_id)
    if not info:
        raise HTTPException(404, f"Dataset {dataset_id} not found")

    # Return fresh budget
    budget = PrivacyBudgetManager.get_or_create(dataset_id)
    return {"status": "success", "data": budget.to_dict()}


@router.get("/privacy/budgets")
async def list_privacy_budgets():
    """Get all privacy budgets."""
    return {"status": "success", "data": PrivacyBudgetManager.get_all_budgets()}


# ──────────────────────────────────────────────
# Statistical Validation Endpoints
# ──────────────────────────────────────────────
@router.post("/validate/statistical")
async def validate_stat(req: ValidateStatisticalRequest):
    """Run statistical similarity validation."""
    real_df = data_service.load_dataset(req.dataset_id)
    if real_df is None:
        raise HTTPException(404, f"Dataset {req.dataset_id} not found")

    synth_df = _load_synthetic(req.synthetic_job_id, req.synthetic_file, req.dataset_id)
    if synth_df is None:
        raise HTTPException(404, "Synthetic data not found. Generate data first.")

    # Remove ID-like columns
    for col in ["patient_id", "id", "record_id", "index"]:
        real_df = real_df.drop(columns=[col], errors="ignore")
        synth_df = synth_df.drop(columns=[col], errors="ignore")

    result = validate_statistical(real_df, synth_df)

    report_id = f"stat_{generate_job_id()}"
    save_report(report_id, "statistical", req.dataset_id, sanitize(result))

    return sanitize({"status": "success", "data": result})


# ──────────────────────────────────────────────
# ML Validation Endpoints
# ──────────────────────────────────────────────
@router.post("/validate/ml")
async def validate_ml(req: ValidateMLRequest):
    """Run ML utility validation (TSTR benchmark)."""
    real_df = data_service.load_dataset(req.dataset_id)
    if real_df is None:
        raise HTTPException(404, f"Dataset {req.dataset_id} not found")

    synth_df = _load_synthetic(req.synthetic_job_id, req.synthetic_file, req.dataset_id)
    if synth_df is None:
        raise HTTPException(404, "Synthetic data not found. Generate data first.")

    for col in ["patient_id", "id", "record_id", "index"]:
        real_df = real_df.drop(columns=[col], errors="ignore")
        synth_df = synth_df.drop(columns=[col], errors="ignore")

    result = validate_ml_utility(real_df, synth_df, req.target_column)

    if "error" in result:
        raise HTTPException(400, result["error"])

    report_id = f"ml_{generate_job_id()}"
    save_report(report_id, "ml_utility", req.dataset_id, sanitize(result))

    return sanitize({"status": "success", "data": result})


# ──────────────────────────────────────────────
# Attack Simulation Endpoints
# ──────────────────────────────────────────────
@router.post("/attacks/simulate")
async def simulate_attacks(req: AttackSimulationRequest):
    """Run all privacy attack simulations."""
    real_df = data_service.load_dataset(req.dataset_id)
    if real_df is None:
        raise HTTPException(404, f"Dataset {req.dataset_id} not found")

    synth_df = _load_synthetic(req.synthetic_job_id, req.synthetic_file, req.dataset_id)
    if synth_df is None:
        raise HTTPException(404, "Synthetic data not found. Generate data first.")

    for col in ["patient_id", "id", "record_id", "index"]:
        real_df = real_df.drop(columns=[col], errors="ignore")
        synth_df = synth_df.drop(columns=[col], errors="ignore")

    result = run_all_attacks(real_df, synth_df)

    report_id = f"attack_{generate_job_id()}"
    save_report(report_id, "attack_simulation", req.dataset_id, sanitize(result))

    return sanitize({"status": "success", "data": result})


# ──────────────────────────────────────────────
# Federated Learning Endpoints
# ──────────────────────────────────────────────
@router.post("/federated/create")
async def create_federation(req: CreateFederationRequest):
    """Create a new federation for multi-hospital collaboration."""
    fed = FederationManager.create_federation(req.total_rounds)
    return {"status": "success", "data": fed.to_dict()}


@router.post("/federated/add-hospital")
async def add_hospital(
    federation_id: str = Form(...),
    hospital_name: str = Form(...),
    file: UploadFile = File(...),
):
    """Add a hospital's data to a federation."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files supported")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Drop ID columns
    for col in ["patient_id", "id", "record_id", "index"]:
        df = df.drop(columns=[col], errors="ignore")

    hospital_id = f"hosp_{generate_job_id()}"
    try:
        result = FederationManager.add_hospital(
            federation_id, hospital_id, hospital_name, df
        )
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/federated/train")
async def federated_train(req: FederatedTrainRequest):
    """Run federated training across all hospitals."""
    try:
        result = FederationManager.run_federated_training(
            req.federation_id,
            dp_epsilon=req.dp_epsilon,
            dp_delta=req.dp_delta,
            apply_dp_to_updates=req.apply_dp,
        )
        return {"status": "success", "data": result}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/federated/generate")
async def federated_generate(req: FederatedGenerateRequest):
    """Generate synthetic data from federated global model."""
    try:
        synth_df, metadata = FederationManager.generate_from_federation(
            req.federation_id, req.num_rows
        )
        output_file = GENERATED_DIR / f"federated_{req.federation_id}.csv"
        synth_df.to_csv(output_file, index=False)

        return sanitize({
            "status": "success",
            "data": {
                **metadata,
                "preview": synth_df.head(5).to_dict(orient="records"),
                "output_file": str(output_file),
            },
        })
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/federated/list")
async def list_federations():
    """List all federations."""
    return {"status": "success", "data": FederationManager.list_federations()}


# ──────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────
@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get a saved report."""
    report = reports_store.get(report_id)
    if not report:
        raise HTTPException(404, f"Report {report_id} not found")
    return {"status": "success", "data": report}


@router.get("/reports")
async def list_reports():
    """List all reports."""
    return {"status": "success", "data": list(reports_store.list_all().values())}


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────
@router.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "system": "MediSynth.AI",
        "version": "1.0.0",
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _load_synthetic(job_id: Optional[str], file_path: Optional[str],
                    dataset_id: str) -> Optional[pd.DataFrame]:
    """Load synthetic data from job ID, file path, or most recent job."""
    if job_id:
        job = jobs_store.get(job_id)
        if job and job.get("result"):
            path = Path(job["result"]["output_file"])
            if path.exists():
                return pd.read_csv(path)

    if file_path:
        path = Path(file_path)
        if path.exists():
            return pd.read_csv(path)

    # Try to find most recent completed job for this dataset
    all_jobs = jobs_store.list_all()
    dataset_jobs = [
        j for j in all_jobs.values()
        if j.get("dataset_id") == dataset_id and j.get("status") == "completed"
    ]
    if dataset_jobs:
        latest = max(dataset_jobs, key=lambda j: j.get("created_at", ""))
        if latest.get("result"):
            path = Path(latest["result"]["output_file"])
            if path.exists():
                return pd.read_csv(path)

    return None

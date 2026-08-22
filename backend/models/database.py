import os
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import STORAGE_DIR


# ──────────────────────────────────────────────
# Storage Backend
# ──────────────────────────────────────────────
class JSONStore:
    """Thread-safe JSON file-backed key-value store with atomic write-and-replace."""

    def __init__(self, filepath: Path):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self._filepath.exists():
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self._filepath.with_suffix(f".tmp.{secrets.token_hex(4)}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, self._filepath)
        except Exception:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Dict):
        with self._lock:
            self._data[key] = value
            self._save()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._save()
                return True
            return False

    def list_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def update(self, key: str, updates: Dict):
        with self._lock:
            if key in self._data:
                self._data[key].update(updates)
                self._save()


# ──────────────────────────────────────────────
# Stores
# ──────────────────────────────────────────────
_db_dir = STORAGE_DIR / "db"
_db_dir.mkdir(parents=True, exist_ok=True)

datasets_store = JSONStore(_db_dir / "datasets.json")
jobs_store = JSONStore(_db_dir / "jobs.json")
privacy_budgets_store = JSONStore(_db_dir / "privacy_budgets.json")
reports_store = JSONStore(_db_dir / "reports.json")
federations_store = JSONStore(_db_dir / "federations.json")


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def register_dataset(dataset_id: str, filename: str, filepath: str,
                     num_rows: int, num_cols: int, columns: List[str],
                     column_types: Dict[str, str], fingerprint: str):
    """Register a new dataset in the store."""
    datasets_store.set(dataset_id, {
        "id": dataset_id,
        "filename": filename,
        "filepath": filepath,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
        "column_types": column_types,
        "fingerprint": fingerprint,
        "created_at": now_iso(),
    })
    # Initialize privacy budget
    privacy_budgets_store.set(dataset_id, {
        "dataset_id": dataset_id,
        "total_epsilon_used": 0.0,
        "total_delta_used": 0.0,
        "history": [],
        "created_at": now_iso(),
    })


def record_privacy_spend(dataset_id: str, epsilon: float, delta: float,
                         operation: str):
    """Record privacy budget expenditure."""
    budget = privacy_budgets_store.get(dataset_id)
    if budget:
        budget["total_epsilon_used"] += epsilon
        budget["total_delta_used"] += delta
        budget["history"].append({
            "epsilon": epsilon,
            "delta": delta,
            "operation": operation,
            "timestamp": now_iso(),
        })
        privacy_budgets_store.set(dataset_id, budget)


def create_job(job_id: str, dataset_id: str, job_type: str, params: Dict):
    """Create a new job record with created_at and updated_at timestamps."""
    now = now_iso()
    jobs_store.set(job_id, {
        "id": job_id,
        "dataset_id": dataset_id,
        "type": job_type,
        "params": params,
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    })


def update_job(job_id: str, **kwargs):
    """Update job status/result, automatically recording updated_at timestamp."""
    if "updated_at" not in kwargs:
        kwargs["updated_at"] = now_iso()
    jobs_store.update(job_id, kwargs)


def reconcile_stale_jobs(stale_timeout_seconds: Optional[int] = None) -> List[str]:
    """
    Reconcile stale/interrupted jobs on application startup or recovery.
    - Scans persisted jobs
    - Finds jobs left with status == 'running'
    - Transitions them to 'interrupted' status
    - Preserves all original job parameters, result, and metadata
    - Idempotent and defensive against corrupted records
    """
    reconciled_ids = []
    try:
        all_jobs = jobs_store.list_all()
    except Exception:
        return []

    now_str = now_iso()
    now_dt = datetime.now(timezone.utc)

    for job_id, job in list(all_jobs.items()):
        if not isinstance(job, dict):
            continue

        status = job.get("status")
        if status == "running":
            # If stale_timeout_seconds is given, only reconcile if (now - updated_at) >= timeout
            if stale_timeout_seconds is not None:
                updated_at_str = job.get("updated_at") or job.get("created_at")
                if updated_at_str:
                    try:
                        updated_dt = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                        if (now_dt - updated_dt).total_seconds() < stale_timeout_seconds:
                            continue
                    except Exception:
                        pass

            # Mark job as interrupted
            job["status"] = "interrupted"
            job["error"] = "Job interrupted by server shutdown or process termination"
            job["completed_at"] = now_str
            job["updated_at"] = now_str
            try:
                jobs_store.set(job_id, job)
                reconciled_ids.append(job_id)
            except Exception:
                pass

    return reconciled_ids


def save_report(report_id: str, report_type: str, dataset_id: str,
                data: Dict):
    """Save a validation/attack report."""
    reports_store.set(report_id, {
        "id": report_id,
        "type": report_type,
        "dataset_id": dataset_id,
        "data": data,
        "created_at": now_iso(),
    })

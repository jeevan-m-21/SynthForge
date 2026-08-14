"""
MediSynth.AI — Data Models & In-Memory Storage
JSON-file backed persistence for datasets, jobs, privacy budgets, and reports.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.config import STORAGE_DIR


# ──────────────────────────────────────────────
# Storage Backend
# ──────────────────────────────────────────────
class JSONStore:
    """Thread-safe JSON file-backed key-value store."""

    def __init__(self, filepath: Path):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self._filepath.exists():
            try:
                with open(self._filepath, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self._filepath, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

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
    """Create a new job record."""
    jobs_store.set(job_id, {
        "id": job_id,
        "dataset_id": dataset_id,
        "type": job_type,
        "params": params,
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": now_iso(),
        "completed_at": None,
    })


def update_job(job_id: str, **kwargs):
    """Update job status/result."""
    jobs_store.update(job_id, kwargs)


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

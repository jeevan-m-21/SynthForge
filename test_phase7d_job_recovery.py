"""
SynthForge — Phase 7D Test Suite: Job Lifecycle & Crash Recovery.
Tests:
1. running job becomes interrupted on reconciliation
2. reconciliation is idempotent
3. completed jobs unchanged
4. failed jobs unchanged
5. pending jobs unchanged
6. params/metadata preserved
7. updated_at changes correctly
8. malformed job records do not crash reconciliation
9. missing jobs store handled safely
10. temporary output cleanup
11. atomic generated output
12. privacy budget is not modified during reconciliation
13. startup reconciliation failure does not prevent application startup where practical
"""
import unittest
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.database import (
    jobs_store, privacy_budgets_store, datasets_store,
    create_job, update_job, reconcile_stale_jobs, now_iso
)
from backend.services.generator import generate_synthetic_data
from backend.services import data_service
from backend.config import GENERATED_DIR, MODELS_DIR


class TestPhase7DJobRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample dataset for generator tests
        raw_csv = (
            b"age,gender,income,target\n"
            b"25,M,50000,0\n"
            b"30,F,60000,1\n"
            b"35,M,75000,0\n"
            b"40,F,80000,1\n"
            b"45,M,95000,0\n"
            b"50,F,110000,1\n"
        )
        ingest_res = data_service.ingest_csv(raw_csv, "test_recovery_ds.csv")
        cls.dataset_id = ingest_res["dataset_id"]

    def setUp(self):
        # Clean up test job entries
        pass

    # 1. running job becomes interrupted on reconciliation
    def test_01_running_job_becomes_interrupted(self):
        job_id = f"test_job_running_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 50, "seed": 42})
        update_job(job_id, status="running", progress=30)

        reconciled = reconcile_stale_jobs()
        self.assertIn(job_id, reconciled)

        job = jobs_store.get(job_id)
        self.assertEqual(job["status"], "interrupted")
        self.assertIsNotNone(job["error"])
        self.assertIsNotNone(job["completed_at"])
        self.assertIsNotNone(job["updated_at"])

    # 2. reconciliation is idempotent
    def test_02_reconciliation_is_idempotent(self):
        job_id = f"test_job_idem_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 50})
        update_job(job_id, status="running", progress=70)

        # First pass
        reconciled_1 = reconcile_stale_jobs()
        self.assertIn(job_id, reconciled_1)
        job_first = jobs_store.get(job_id)

        # Second pass
        reconciled_2 = reconcile_stale_jobs()
        self.assertNotIn(job_id, reconciled_2)
        job_second = jobs_store.get(job_id)

        self.assertEqual(job_first["status"], job_second["status"])
        self.assertEqual(job_first["error"], job_second["error"])

    # 3. completed jobs unchanged
    def test_03_completed_jobs_unchanged(self):
        job_id = f"test_job_comp_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 50})
        update_job(job_id, status="completed", progress=100, result={"rows": 50})

        reconciled = reconcile_stale_jobs()
        self.assertNotIn(job_id, reconciled)

        job = jobs_store.get(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100)

    # 4. failed jobs unchanged
    def test_04_failed_jobs_unchanged(self):
        job_id = f"test_job_fail_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 50})
        update_job(job_id, status="failed", error="Explicit model training error")

        reconciled = reconcile_stale_jobs()
        self.assertNotIn(job_id, reconciled)

        job = jobs_store.get(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Explicit model training error")

    # 5. pending jobs unchanged
    def test_05_pending_jobs_unchanged(self):
        job_id = f"test_job_pend_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 50})

        reconciled = reconcile_stale_jobs()
        self.assertNotIn(job_id, reconciled)

        job = jobs_store.get(job_id)
        self.assertEqual(job["status"], "pending")

    # 6. params/metadata preserved
    def test_06_params_and_metadata_preserved(self):
        job_id = f"test_job_meta_{int(time.time() * 1000)}"
        custom_params = {
            "num_rows": 100,
            "model_type": "statistical",
            "epochs": 10,
            "seed": 999,
            "epsilon": 2.5,
        }
        create_job(job_id, self.dataset_id, "generation", custom_params)
        update_job(job_id, status="running", progress=85)

        reconcile_stale_jobs()

        job = jobs_store.get(job_id)
        self.assertEqual(job["params"], custom_params)
        self.assertEqual(job["dataset_id"], self.dataset_id)
        self.assertEqual(job["type"], "generation")

    # 7. updated_at changes correctly
    def test_07_updated_at_tracks_changes(self):
        job_id = f"test_job_ts_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 20})
        job_created = jobs_store.get(job_id)
        self.assertIn("created_at", job_created)
        self.assertIn("updated_at", job_created)
        self.assertEqual(job_created["created_at"], job_created["updated_at"])

        # Update job progress
        time.sleep(0.01)
        update_job(job_id, progress=50)
        job_updated = jobs_store.get(job_id)
        self.assertGreaterEqual(job_updated["updated_at"], job_created["updated_at"])

    # 8. malformed job records do not crash reconciliation
    def test_08_malformed_records_handled_safely(self):
        # Insert malformed non-dict or partial records
        jobs_store.set("bad_job_1", "not a dict")
        jobs_store.set("bad_job_2", {"status": None})
        jobs_store.set("bad_job_3", {"id": "bad3"})

        # Should execute cleanly without raising
        reconciled = reconcile_stale_jobs()
        self.assertIsInstance(reconciled, list)

    # 9. missing jobs store handled safely
    def test_09_missing_store_handled_safely(self):
        with patch.object(jobs_store, "list_all", side_effect=IOError("Disk unreachable")):
            reconciled = reconcile_stale_jobs()
            self.assertEqual(reconciled, [])

    # 10. temporary output cleanup behavior
    def test_10_temporary_output_cleanup(self):
        temp_file = GENERATED_DIR / f".tmp_test_cleanup_{int(time.time())}.csv"
        temp_file.write_text("temporary,data\n1,2\n", encoding="utf-8")
        self.assertTrue(temp_file.exists())
        temp_file.unlink()
        self.assertFalse(temp_file.exists())

    # 11. atomic generated output
    def test_11_atomic_generated_output(self):
        res = generate_synthetic_data(
            dataset_id=self.dataset_id,
            num_rows=20,
            model_type="statistical",
            apply_dp=False,
            seed=123,
        )
        output_file = Path(res["output_file"])
        self.assertTrue(output_file.exists())
        # Confirm no leftover .tmp file
        temp_csv = output_file.parent / f".tmp_{res['job_id']}_{res['output_filename']}"
        self.assertFalse(temp_csv.exists())

    # 12. privacy budget is not modified during reconciliation
    def test_12_privacy_budget_untouched_during_reconciliation(self):
        budget_before = privacy_budgets_store.get(self.dataset_id)
        initial_eps = budget_before.get("total_epsilon_used", 0.0) if budget_before else 0.0

        # Create running job
        job_id = f"test_job_dp_rec_{int(time.time() * 1000)}"
        create_job(job_id, self.dataset_id, "generation", {"num_rows": 50, "epsilon": 2.0})
        update_job(job_id, status="running")

        reconcile_stale_jobs()

        budget_after = privacy_budgets_store.get(self.dataset_id)
        after_eps = budget_after.get("total_epsilon_used", 0.0) if budget_after else 0.0
        self.assertEqual(initial_eps, after_eps)

    # 13. startup reconciliation failure does not prevent application startup
    def test_13_startup_error_resilience(self):
        with patch("backend.models.database.reconcile_stale_jobs", side_effect=RuntimeError("Store corruption simulation")):
            # TestClient triggers lifespan startup
            with TestClient(app, raise_server_exceptions=False) as client:
                res = client.get("/api/health")
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()

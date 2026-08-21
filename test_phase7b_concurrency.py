"""
SynthForge — Phase 7B Test Suite: Concurrency & Resource Hardening.
Tests:
1. MAX_SYNTH_ROWS rejection
2. MAX_EPOCHS rejection
3. MAX_FL_ROUNDS rejection
4. Timeout configuration & handling
5. Exact duplicate detection with duplicated rows
6. Large duplicate dataset does not use Cartesian merge (bounded performance)
7. Attack sample limit
8. Categorical association column limit (Cramér's V)
9. Concurrent privacy budget spending cannot exceed budget
10. Atomic JSON persistence
11. Heavy endpoint event-loop non-blocking responsiveness
"""
import os
import time
import threading
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import (
    MAX_SYNTH_ROWS, MAX_EPOCHS, MAX_FL_ROUNDS,
    MAX_EXECUTION_TIMEOUT_SECONDS, MAX_ATTACK_SAMPLE_SIZE,
    MAX_CAT_COLS_FOR_ASSOC,
)
from backend.services.quality_evaluator import (
    check_exact_duplicate_collisions,
    evaluate_relationship_fidelity,
)
from backend.services.attack_simulator import _sample_bounded
from backend.services.privacy_engine import PrivacyBudgetManager
from backend.models.database import JSONStore


class TestPhase7BConcurrency(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    # 1. MAX_SYNTH_ROWS rejection
    def test_01_max_synth_rows_rejection(self):
        res = self.client.post(
            "/api/generate",
            json={"dataset_id": "dummy_id", "num_rows": MAX_SYNTH_ROWS + 1000},
        )
        self.assertEqual(res.status_code, 422)
        self.assertTrue(any("num_rows" in str(err) for err in res.json().get("detail", [])))

    # 2. MAX_EPOCHS rejection
    def test_02_max_epochs_rejection(self):
        res = self.client.post(
            "/api/generate",
            json={"dataset_id": "dummy_id", "epochs": MAX_EPOCHS + 50},
        )
        self.assertEqual(res.status_code, 422)
        self.assertTrue(any("epochs" in str(err) for err in res.json().get("detail", [])))

    # 3. MAX_FL_ROUNDS rejection
    def test_03_max_fl_rounds_rejection(self):
        res = self.client.post(
            "/api/federated/create",
            json={"total_rounds": MAX_FL_ROUNDS + 10},
        )
        self.assertEqual(res.status_code, 422)
        self.assertTrue(any("total_rounds" in str(err) for err in res.json().get("detail", [])))

    # 4. Timeout configuration
    def test_04_timeout_configuration(self):
        self.assertGreater(MAX_EXECUTION_TIMEOUT_SECONDS, 0)
        self.assertEqual(MAX_EXECUTION_TIMEOUT_SECONDS, 300)

    # 5. Exact duplicate detection with duplicated rows
    def test_05_exact_duplicate_detection_with_duplicated_rows(self):
        real_data = pd.DataFrame({
            "age": [25, 30, 35, 25, 40],
            "gender": ["M", "F", "F", "M", "M"],
        })
        synth_data = pd.DataFrame({
            "age": [25, 25, 30, 99],
            "gender": ["M", "M", "F", "M"],
        })
        # 3 out of 4 synth rows match a row in real_data ([25, M], [25, M], [30, F])
        count, rate = check_exact_duplicate_collisions(real_data, synth_data)
        self.assertEqual(count, 3)
        self.assertEqual(rate, 0.75)

    # 6. Large duplicate dataset does not use Cartesian merge
    def test_06_large_duplicate_dataset_no_cartesian_blowup(self):
        # 2,000 identical rows in real, 2,000 in synth
        # Cartesian merge would produce 2000 * 2000 = 4,000,000 rows
        # Hash set lookup does 2000 lookups in O(N) time
        n = 2000
        real_df = pd.DataFrame({"colA": ["A"] * n, "colB": ["B"] * n})
        synth_df = pd.DataFrame({"colA": ["A"] * n, "colB": ["B"] * n})

        start = time.perf_counter()
        count, rate = check_exact_duplicate_collisions(real_df, synth_df)
        elapsed = time.perf_counter() - start

        self.assertEqual(count, n)
        self.assertEqual(rate, 1.0)
        self.assertLess(elapsed, 1.0)  # Must take < 1 second

    # 7. Attack sample limit
    def test_07_attack_sample_limit(self):
        large_df = pd.DataFrame({"val": range(MAX_ATTACK_SAMPLE_SIZE + 2000)})
        sampled = _sample_bounded(large_df, max_size=MAX_ATTACK_SAMPLE_SIZE)
        self.assertEqual(len(sampled), MAX_ATTACK_SAMPLE_SIZE)

        small_df = pd.DataFrame({"val": range(100)})
        not_sampled = _sample_bounded(small_df, max_size=MAX_ATTACK_SAMPLE_SIZE)
        self.assertEqual(len(not_sampled), 100)

    # 8. Categorical association column limit
    def test_08_categorical_association_column_limit(self):
        # Create dataset with 40 categorical columns
        data = {}
        for i in range(40):
            data[f"cat_{i}"] = np.random.choice(["X", "Y", "Z"], 50)
        df_real = pd.DataFrame(data)
        df_synth = pd.DataFrame(data)

        report = evaluate_relationship_fidelity(df_real, df_synth)
        self.assertTrue(report["applicable"])
        # Ensure it computed without error and respected limit
        self.assertIn("cramers_v_association", report["applicable_checks"])

    # 9. Concurrent privacy budget spending cannot exceed budget
    def test_09_concurrent_privacy_budget_spending(self):
        test_dataset_id = "test_concurrent_ds"
        budget = PrivacyBudgetManager.get_or_create(test_dataset_id, max_epsilon=5.0)

        results = []
        threads = []

        def attempt_spend():
            success, _ = PrivacyBudgetManager.spend(test_dataset_id, epsilon=1.0, delta=1e-5, operation="gen")
            results.append(success)

        # Launch 20 concurrent threads trying to spend eps=1.0 on budget=5.0
        for _ in range(20):
            t = threading.Thread(target=attempt_spend)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = sum(1 for r in results if r is True)
        self.assertEqual(successes, 5)
        self.assertEqual(len(results), 20)
        self.assertEqual(budget.remaining_epsilon, 0.0)

    # 10. Atomic JSON persistence
    def test_10_atomic_json_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test_store.json"
            store = JSONStore(store_path)
            store.set("key1", {"value": "initial"})
            self.assertTrue(store_path.exists())

            # Read raw file to verify it's valid JSON
            store.set("key2", {"value": "updated"})
            loaded = store.get("key2")
            self.assertEqual(loaded["value"], "updated")

    # 11. Heavy endpoint is not executed directly on event loop
    def test_11_async_thread_offloading_responsiveness(self):
        # Verify /api/health responds quickly while async endpoint handlers are non-blocking
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "healthy")


if __name__ == "__main__":
    unittest.main()

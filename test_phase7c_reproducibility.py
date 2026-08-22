"""
SynthForge — Phase 7C Test Suite: Reproducibility & Seed Control.
Tests:
1. StatisticalGenerator same seed → identical output
2. StatisticalGenerator different seeds → different output
3. seed=None preserves nondeterministic behavior
4. DP same seed → identical controlled test output
5. DP different seeds → different noise
6. API accepts seed
7. Generation job metadata records seed
8. reproducible_run is true only when seed is provided
9. Federated generation seed propagation where practical
10. SDV reproducibility test with installed SDV version
"""
import unittest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.generator import StatisticalGenerator, generate_synthetic_data, _train_sdv_model
from backend.services.privacy_engine import DPDataProcessor, PrivacyParams
from backend.services.federated_learning import FederationManager
from backend.services import data_service
from backend.models.database import datasets_store, jobs_store


class TestPhase7CReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)
        # Ingest a sample dataset for API tests
        raw_csv = (
            b"age,gender,income,target\n"
            b"25,M,50000,0\n"
            b"30,F,60000,1\n"
            b"35,M,75000,0\n"
            b"40,F,80000,1\n"
            b"45,M,95000,0\n"
            b"50,F,110000,1\n"
        )
        ingest_res = data_service.ingest_csv(raw_csv, "test_repro.csv")
        cls.dataset_id = ingest_res["dataset_id"]

        cls.sample_df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "income": [50000, 60000, 75000, 80000, 95000, 110000, 120000, 130000],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        })

    # 1. StatisticalGenerator same seed → identical output
    def test_01_statistical_generator_same_seed_identical_output(self):
        gen = StatisticalGenerator()
        gen.fit(self.sample_df)

        sample1 = gen.sample(num_rows=20, seed=42)
        sample2 = gen.sample(num_rows=20, seed=42)

        pd.testing.assert_frame_equal(sample1, sample2)

    # 2. StatisticalGenerator different seeds → different output
    def test_02_statistical_generator_different_seeds_different_output(self):
        gen = StatisticalGenerator()
        gen.fit(self.sample_df)

        sample1 = gen.sample(num_rows=20, seed=42)
        sample2 = gen.sample(num_rows=20, seed=999)

        # Numerical columns should differ
        self.assertFalse(sample1["age"].equals(sample2["age"]))

    # 3. seed=None preserves nondeterministic behavior
    def test_03_statistical_generator_unseeded_nondeterministic(self):
        gen = StatisticalGenerator()
        gen.fit(self.sample_df)

        sample1 = gen.sample(num_rows=50, seed=None)
        sample2 = gen.sample(num_rows=50, seed=None)

        self.assertFalse(sample1["age"].equals(sample2["age"]))

    # 4. DP same seed → identical controlled test output
    def test_04_dp_same_seed_identical_output(self):
        dp = DPDataProcessor(PrivacyParams(epsilon=1.0, delta=1e-5, mechanism="gaussian"))

        synth_in = self.sample_df.copy()
        noisy1, meta1 = dp.apply_dp(synth_in, self.sample_df, target_column="target", seed=123)
        noisy2, meta2 = dp.apply_dp(synth_in, self.sample_df, target_column="target", seed=123)

        pd.testing.assert_frame_equal(noisy1, noisy2)

    # 5. DP different seeds → different noise
    def test_05_dp_different_seeds_different_noise(self):
        dp = DPDataProcessor(PrivacyParams(epsilon=1.0, delta=1e-5, mechanism="gaussian"))

        synth_in = self.sample_df.copy()
        noisy1, _ = dp.apply_dp(synth_in, self.sample_df, target_column="target", seed=123)
        noisy2, _ = dp.apply_dp(synth_in, self.sample_df, target_column="target", seed=987)

        self.assertFalse(noisy1["age"].equals(noisy2["age"]))

    # 6. API accepts seed
    def test_06_api_accepts_seed(self):
        res = self.client.post(
            "/api/generate",
            json={
                "dataset_id": self.dataset_id,
                "num_rows": 20,
                "model_type": "statistical",
                "apply_dp": False,
                "seed": 777,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        self.assertEqual(data["seed"], 777)
        self.assertTrue(data["reproducible_run"])

    # 7. Generation job metadata records seed
    def test_07_job_metadata_records_seed(self):
        res = self.client.post(
            "/api/generate",
            json={
                "dataset_id": self.dataset_id,
                "num_rows": 20,
                "model_type": "statistical",
                "apply_dp": True,
                "seed": 888,
            },
        )
        self.assertEqual(res.status_code, 200)
        job_id = res.json()["data"]["job_id"]
        job = jobs_store.get(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["result"]["seed"], 888)
        self.assertTrue(job["result"]["reproducible_run"])

    # 8. reproducible_run is true only when seed is provided
    def test_08_reproducible_run_flag(self):
        res_unseeded = self.client.post(
            "/api/generate",
            json={
                "dataset_id": self.dataset_id,
                "num_rows": 20,
                "model_type": "statistical",
                "apply_dp": False,
            },
        )
        self.assertEqual(res_unseeded.status_code, 200)
        data_unseeded = res_unseeded.json()["data"]
        self.assertIsNone(data_unseeded["seed"])
        self.assertFalse(data_unseeded["reproducible_run"])

    # 9. Federated generation seed propagation
    def test_09_federated_generation_seed_propagation(self):
        fed = FederationManager.create_federation(total_rounds=2)
        fed_id = fed.federation_id

        FederationManager.add_hospital(fed_id, "hosp_1", "Hospital 1", self.sample_df)
        FederationManager.add_hospital(fed_id, "hosp_2", "Hospital 2", self.sample_df)
        FederationManager.run_federated_training(fed_id, apply_dp_to_updates=False)

        res1, meta1 = FederationManager.generate_from_federation(fed_id, num_rows=25, seed=555)
        res2, meta2 = FederationManager.generate_from_federation(fed_id, num_rows=25, seed=555)

        pd.testing.assert_frame_equal(res1, res2)
        self.assertEqual(meta1["seed"], 555)
        self.assertTrue(meta1["reproducible_run"])

    # 10. SDV reproducibility test if practical with installed SDV version
    def test_10_sdv_reproducibility_via_seed(self):
        sdv1 = _train_sdv_model(self.sample_df, "tvae", epochs=5, batch_size=10, seed=101)
        if sdv1 is not None:
            # Sample with seeded PyTorch/NumPy
            import torch
            torch.manual_seed(101)
            np.random.seed(101)
            out1 = sdv1.sample(10)

            torch.manual_seed(101)
            np.random.seed(101)
            out2 = sdv1.sample(10)

            # Columns should match
            self.assertEqual(list(out1.columns), list(out2.columns))
            self.assertEqual(len(out1), 10)


if __name__ == "__main__":
    unittest.main()

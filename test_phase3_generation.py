"""
SynthForge — Phase 3 Test Suite: Generic Profile-Driven Synthetic Data Generation.
"""
import io
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.services.generator import StatisticalGenerator, generate_synthetic_data
from backend.services.dataset_profiler import profile_dataframe, SemanticType
from backend.services.data_service import ingest_csv
from backend.services.privacy_engine import DPDataProcessor, PrivacyParams
from backend.services.federated_learning import FederationManager


class TestPhase3Generation(unittest.TestCase):
    def test_healthcare_generation(self):
        df = pd.DataFrame({
            "patient_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "age": [45, 52, 36, 61, 29, 70, 48, 55],
            "blood_pressure": [120, 135, 118, 142, 125, 150, 130, 128],
            "condition": ["hypertension", "diabetes", "hypertension", "asthma", "diabetes", "hypertension", "asthma", "diabetes"],
            "diagnosis": [1, 1, 0, 1, 0, 1, 0, 1],
        })
        profile = profile_dataframe(df, dataset_name="healthcare.csv")
        working_df = df.drop(columns=profile.detected_metadata.identifier_columns)

        gen = StatisticalGenerator()
        gen.fit(working_df, profile=profile)
        synth = gen.sample(50)

        self.assertNotIn("patient_id", synth.columns)
        self.assertEqual(list(synth.columns), list(working_df.columns))
        self.assertEqual(len(synth), 50)
        self.assertTrue(synth["diagnosis"].isin([0, 1]).all())

    def test_banking_generation(self):
        df = pd.DataFrame({
            "customer_id": [101, 102, 103, 104, 105, 106],
            "age": [25, 45, 35, 50, 23, 60],
            "income": [45000.0, 85000.0, 62000.0, 95000.0, 38000.0, 110000.0],
            "loan_amount": [5000, 20000, 12000, 25000, 4000, 30000],
            "default_status": ["no", "yes", "no", "yes", "no", "no"],
        })
        profile = profile_dataframe(df, dataset_name="banking.csv")
        working_df = df.drop(columns=profile.detected_metadata.identifier_columns)

        gen = StatisticalGenerator()
        gen.fit(working_df, profile=profile)
        synth = gen.sample(30)

        self.assertNotIn("customer_id", synth.columns)
        self.assertEqual(list(synth.columns), ["age", "income", "loan_amount", "default_status"])
        self.assertTrue(synth["default_status"].isin(["no", "yes"]).all())
        self.assertTrue((synth["income"] >= 38000.0).all())

    def test_education_generation(self):
        df = pd.DataFrame({
            "student_id": ["S1", "S2", "S3", "S4", "S5"],
            "department": ["CS", "EE", "CS", "ME", "CS"],
            "cgpa": [3.8, 3.2, 3.9, 2.9, 3.5],
            "attendance": [95.0, 82.0, 98.0, 75.0, 88.0],
            "placement_status": [1, 0, 1, 0, 1],
        })
        profile = profile_dataframe(df, dataset_name="education.csv")
        working_df = df.drop(columns=profile.detected_metadata.identifier_columns)

        gen = StatisticalGenerator()
        gen.fit(working_df, profile=profile)
        synth = gen.sample(25)

        self.assertNotIn("student_id", synth.columns)
        self.assertEqual(list(synth.columns), ["department", "cgpa", "attendance", "placement_status"])
        self.assertTrue(synth["placement_status"].isin([0, 1]).all())
        self.assertTrue(synth["department"].isin(["CS", "EE", "ME"]).all())

    def test_cybersecurity_generation(self):
        df = pd.DataFrame({
            "event_id": [201, 202, 203, 204, 205],
            "source_ip": ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14"],
            "destination_port": [80, 443, 22, 8080, 443],
            "packet_size": [512, 1024, 256, 2048, 128],
            "attack_type": ["benign", "dos", "benign", "portscan", "benign"],
        })
        profile = profile_dataframe(df, dataset_name="cyber.csv")
        working_df = df.drop(columns=profile.detected_metadata.identifier_columns)

        gen = StatisticalGenerator()
        gen.fit(working_df, profile=profile)
        synth = gen.sample(20)

        self.assertNotIn("event_id", synth.columns)
        self.assertEqual(list(synth.columns), ["source_ip", "destination_port", "packet_size", "attack_type"])
        self.assertTrue(synth["attack_type"].isin(["benign", "dos", "portscan"]).all())

    def test_boolean_columns_no_drift(self):
        df = pd.DataFrame({
            "native_bool": [True, False, True, True, False],
            "int_bool": [1, 0, 1, 1, 0],
            "target": [0, 1, 0, 1, 0],
        })
        profile = profile_dataframe(df)
        gen = StatisticalGenerator()
        gen.fit(df, profile=profile)
        synth = gen.sample(100)

        self.assertTrue(pd.api.types.is_bool_dtype(synth["native_bool"]))
        self.assertTrue(set(synth["int_bool"].unique()).issubset({0, 1}))
        # Ensure no float drift like 0.37
        for val in synth["int_bool"]:
            self.assertIn(val, (0, 1))

    def test_constant_columns(self):
        df = pd.DataFrame({
            "const_num": [42, 42, 42, 42],
            "const_str": ["FIXED", "FIXED", "FIXED", "FIXED"],
            "var_num": [10, 20, 30, 40],
        })
        profile = profile_dataframe(df)
        gen = StatisticalGenerator()
        gen.fit(df, profile=profile)
        synth = gen.sample(50)

        self.assertEqual(list(synth["const_num"].unique()), [42])
        self.assertEqual(list(synth["const_str"].unique()), ["FIXED"])

    def test_datetime_columns(self):
        df = pd.DataFrame({
            "date_str": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "metric": [100.0, 105.0, 110.0, 108.0],
        })
        profile = profile_dataframe(df)
        gen = StatisticalGenerator()
        gen.fit(df, profile=profile)
        synth = gen.sample(30)

        self.assertEqual(len(synth), 30)
        parsed_dates = pd.to_datetime(synth["date_str"], errors="coerce")
        self.assertFalse(parsed_dates.isna().any())

    def test_text_columns_safe_sampling(self):
        df = pd.DataFrame({
            "notes": [
                "Patient reported mild fever and cough.",
                "Follow-up after medication, no issues found.",
                "Routine checkup scheduled for next quarter.",
                "Lab results normal across all indicators.",
            ],
            "score": [1, 2, 3, 4],
        })
        profile = profile_dataframe(df)
        gen = StatisticalGenerator()
        gen.fit(df, profile=profile)
        synth = gen.sample(20)

        self.assertEqual(len(synth), 20)
        self.assertTrue(synth["notes"].isin(df["notes"]).all())

    def test_missing_values_preservation(self):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0, 4.0, None, 6.0, 7.0, None], # 3/8 null = 37.5%
            "b": ["x", "y", "z", "w", "x", "y", "z", "w"],    # 0% null
        })
        profile = profile_dataframe(df)
        gen = StatisticalGenerator()
        gen.fit(df, profile=profile)
        synth = gen.sample(200)

        self.assertGreater(synth["a"].isna().sum(), 0)
        self.assertEqual(synth["b"].isna().sum(), 0)

    def test_dp_preserves_target_only_and_not_binary_features(self):
        real_df = pd.DataFrame({
            "binary_feature": [1, 0, 1, 0, 1, 0, 1, 0],
            "numeric_feature": [100.0, 150.0, 120.0, 130.0, 110.0, 140.0, 115.0, 125.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        })
        synth_df = real_df.copy()

        dp = DPDataProcessor(PrivacyParams(epsilon=1.0, delta=1e-5))
        protected_df, meta = dp.apply_dp(synth_df, real_df, target_column="target")

        col_details = meta["column_details"]
        self.assertEqual(col_details["target"]["type"], "target_column")
        self.assertEqual(col_details["target"]["mechanism"], "preserved")
        self.assertIn(col_details["binary_feature"]["type"], ["numeric", "boolean"])

    def test_full_pipeline_generate_synthetic_data(self):
        csv_bytes = (
            "customer_id,age,income,is_active,status\n"
            "1,30,50000,1,active\n"
            "2,45,75000,0,inactive\n"
            "3,28,48000,1,active\n"
            "4,55,90000,1,active\n"
            "5,62,110000,0,inactive\n"
        ).encode("utf-8")
        ingested = ingest_csv(csv_bytes, "test_customer.csv")
        ds_id = ingested["dataset_id"]

        res = generate_synthetic_data(
            dataset_id=ds_id,
            num_rows=20,
            model_type="statistical",
            apply_dp=True,
            epsilon=1.0,
        )

        self.assertEqual(res["num_rows_generated"], 20)
        self.assertNotIn("customer_id", res["columns"])
        self.assertEqual(res["columns"], ["age", "income", "is_active", "status"])
        out_df = pd.read_csv(res["output_file"])
        self.assertEqual(len(out_df), 20)
        self.assertNotIn("customer_id", out_df.columns)

    def test_federated_learning_compatibility(self):
        fm = FederationManager()
        fed = fm.create_federation(total_rounds=2)
        fed_id = fed.federation_id
        hosp1_df = pd.DataFrame({"age": [25, 40, 55], "cholesterol": [180, 220, 240]})
        hosp2_df = pd.DataFrame({"age": [30, 45, 60], "cholesterol": [190, 210, 250]})
        fm.add_hospital(fed_id, "hosp_1", "HospA", hosp1_df)
        fm.add_hospital(fed_id, "hosp_2", "HospB", hosp2_df)

        train_res = fm.run_federated_training(fed_id)
        self.assertEqual(train_res["rounds_completed"], 2)
        gen_df, meta = fm.generate_from_federation(fed_id, num_rows=15)
        self.assertEqual(len(gen_df), 15)
        self.assertEqual(list(gen_df.columns), ["age", "cholesterol"])


if __name__ == "__main__":
    unittest.main()

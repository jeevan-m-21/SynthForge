"""
Unit and Integration Tests for Dataset Profiler (Phase 2).
"""
import io
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.services.dataset_profiler import (
    SemanticType,
    profile_dataframe,
    DatasetProfile,
)
from backend.main import app


class TestDatasetProfiler(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthcare_dataset(self):
        df = pd.DataFrame({
            "patient_id": [1001, 1002, 1003, 1004, 1005],
            "age": [45, 52, 36, 61, 29],
            "blood_pressure": [120.0, 135.5, 118.0, 142.0, 125.0],
            "condition": ["hypertension", "diabetes", "hypertension", "asthma", "diabetes"],
            "diagnosis": [1, 1, 0, 1, 0],
        })
        profile = profile_dataframe(df, dataset_name="healthcare_test.csv")

        self.assertEqual(profile.dataset_name, "healthcare_test.csv")
        self.assertEqual(profile.row_count, 5)
        self.assertEqual(profile.column_count, 5)
        self.assertIn("patient_id", profile.detected_metadata.identifier_columns)
        self.assertEqual(profile.detected_metadata.target_column, "diagnosis")

        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["patient_id"].semantic_type, SemanticType.IDENTIFIER)
        self.assertTrue(col_map["patient_id"].identifier)
        self.assertEqual(col_map["age"].semantic_type, SemanticType.NUMERIC)
        self.assertEqual(col_map["condition"].semantic_type, SemanticType.CATEGORICAL)
        self.assertTrue(col_map["diagnosis"].target_candidate)

    def test_banking_dataset(self):
        df = pd.DataFrame({
            "customer_id": [101, 102, 103, 104, 105],
            "income": [55000.0, 72000.0, 48000.0, 91000.0, 62000.0],
            "credit_score": [680, 720, 610, 790, 650],
            "loan_amount": [10000, 25000, 18000, 32000, 15000],
            "default_status": ["no", "yes", "no", "yes", "no"],
        })
        profile = profile_dataframe(df, dataset_name="banking_test.csv")

        self.assertEqual(profile.row_count, 5)
        self.assertIn("customer_id", profile.detected_metadata.identifier_columns)
        self.assertEqual(profile.detected_metadata.target_column, "default_status")

        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["income"].semantic_type, SemanticType.NUMERIC)
        self.assertIsNotNone(col_map["income"].numeric_stats)
        self.assertEqual(col_map["income"].numeric_stats.min, 48000.0)
        self.assertEqual(col_map["income"].numeric_stats.max, 91000.0)
        self.assertEqual(col_map["default_status"].semantic_type, SemanticType.CATEGORICAL)

    def test_education_dataset(self):
        df = pd.DataFrame({
            "student_id": ["STU001", "STU002", "STU003", "STU004"],
            "department": ["Computer Science", "Electrical", "Computer Science", "Mechanical"],
            "cgpa": [3.8, 3.2, 3.9, 2.9],
            "attendance": [95.0, 82.0, 98.0, 75.0],
            "placement_status": [1, 0, 1, 0],
        })
        profile = profile_dataframe(df, dataset_name="education_test.csv")

        self.assertEqual(profile.row_count, 4)
        self.assertIn("student_id", profile.detected_metadata.identifier_columns)
        self.assertEqual(profile.detected_metadata.target_column, "placement_status")

        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["student_id"].semantic_type, SemanticType.IDENTIFIER)
        self.assertEqual(col_map["department"].semantic_type, SemanticType.CATEGORICAL)
        self.assertEqual(col_map["cgpa"].semantic_type, SemanticType.NUMERIC)

    def test_cybersecurity_dataset(self):
        df = pd.DataFrame({
            "event_id": [201, 202, 203, 204],
            "source_ip": ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13"],
            "destination_port": [80, 443, 22, 8080],
            "packet_size": [512, 1024, 256, 2048],
            "attack_type": ["benign", "dos", "benign", "portscan"],
        })
        profile = profile_dataframe(df, dataset_name="cyber_test.csv")

        self.assertEqual(profile.row_count, 4)
        self.assertIn("event_id", profile.detected_metadata.identifier_columns)
        self.assertEqual(profile.detected_metadata.target_column, "attack_type")

        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["destination_port"].semantic_type, SemanticType.NUMERIC)
        self.assertEqual(col_map["attack_type"].semantic_type, SemanticType.CATEGORICAL)

    def test_numeric_columns_statistics(self):
        df = pd.DataFrame({
            "values": [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        profile = profile_dataframe(df)
        col = profile.columns[0]

        self.assertEqual(col.semantic_type, SemanticType.NUMERIC)
        self.assertIsNotNone(col.numeric_stats)
        self.assertEqual(col.numeric_stats.min, 10.0)
        self.assertEqual(col.numeric_stats.max, 50.0)
        self.assertEqual(col.numeric_stats.mean, 30.0)
        self.assertEqual(col.numeric_stats.median, 30.0)
        self.assertEqual(col.numeric_stats.q25, 20.0)
        self.assertEqual(col.numeric_stats.q75, 40.0)
        self.assertAlmostEqual(col.numeric_stats.std, float(np.std([10, 20, 30, 40, 50], ddof=1)), places=3)

    def test_categorical_columns_statistics(self):
        df = pd.DataFrame({
            "city": ["NY", "SF", "NY", "LA", "NY", "SF"]
        })
        profile = profile_dataframe(df)
        col = profile.columns[0]

        self.assertEqual(col.semantic_type, SemanticType.CATEGORICAL)
        self.assertIsNotNone(col.categorical_stats)
        self.assertEqual(col.categorical_stats.num_categories, 3)
        self.assertEqual(col.categorical_stats.top_categories[0].value, "NY")
        self.assertEqual(col.categorical_stats.top_categories[0].count, 3)
        self.assertEqual(col.categorical_stats.top_categories[0].percentage, 50.0)

    def test_boolean_columns(self):
        df = pd.DataFrame({
            "bool_native": [True, False, True, True],
            "bool_binary_num": [1, 0, 1, 0],
            "bool_binary_float": [1.0, 0.0, 1.0, 1.0],
        })
        profile = profile_dataframe(df)
        col_map = {c.name: c for c in profile.columns}

        self.assertEqual(col_map["bool_native"].semantic_type, SemanticType.BOOLEAN)
        self.assertEqual(col_map["bool_binary_num"].semantic_type, SemanticType.BOOLEAN)
        self.assertEqual(col_map["bool_binary_float"].semantic_type, SemanticType.BOOLEAN)

    def test_datetime_columns(self):
        df = pd.DataFrame({
            "date_iso": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "date_slash": ["01/01/2023", "02/01/2023", "03/01/2023", "04/01/2023"],
            "date_dt_type": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        })
        profile = profile_dataframe(df)
        col_map = {c.name: c for c in profile.columns}

        self.assertEqual(col_map["date_iso"].semantic_type, SemanticType.DATETIME)
        self.assertEqual(col_map["date_slash"].semantic_type, SemanticType.DATETIME)
        self.assertEqual(col_map["date_dt_type"].semantic_type, SemanticType.DATETIME)

    def test_text_columns(self):
        df = pd.DataFrame({
            "clinical_notes": [
                "Patient presented with severe headache and fever for 3 days.",
                "Routine follow up after surgery with no acute complaints or complications.",
                "Prescribed antibiotic treatment following lab examination results today.",
                "Discharged in stable condition with scheduled follow up next month.",
            ]
        })
        profile = profile_dataframe(df)
        self.assertEqual(profile.columns[0].semantic_type, SemanticType.TEXT)

    def test_missing_values(self):
        df = pd.DataFrame({
            "col_a": [1.0, None, 3.0, None],
            "col_b": ["x", "y", None, "w"],
        })
        profile = profile_dataframe(df)

        self.assertEqual(profile.row_count, 4)
        self.assertEqual(profile.column_count, 2)
        self.assertEqual(profile.missing_count, 3)
        self.assertEqual(profile.missing_percentage, 37.5)

        col_map = {c.name: c for c in profile.columns}
        self.assertTrue(col_map["col_a"].nullable)
        self.assertEqual(col_map["col_a"].missing_count, 2)
        self.assertEqual(col_map["col_a"].missing_percentage, 50.0)
        self.assertTrue(col_map["col_b"].nullable)
        self.assertEqual(col_map["col_b"].missing_count, 1)
        self.assertEqual(col_map["col_b"].missing_percentage, 25.0)

    def test_duplicate_rows(self):
        df = pd.DataFrame({
            "a": [1, 2, 1, 1],
            "b": ["x", "y", "x", "x"],
        })
        profile = profile_dataframe(df)

        self.assertEqual(profile.row_count, 4)
        self.assertEqual(profile.duplicate_count, 2)
        self.assertEqual(profile.duplicate_percentage, 50.0)

    def test_identifier_and_target_and_sensitive_detection(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4],
            "full_name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Dana White"],
            "email_address": ["a@test.com", "b@test.com", "c@test.com", "d@test.com"],
            "salary": [60000, 75000, 80000, 95000],
            "churn": [0, 1, 0, 1],
        })
        profile = profile_dataframe(df)

        self.assertIn("user_id", profile.detected_metadata.identifier_columns)
        self.assertEqual(profile.detected_metadata.target_column, "churn")
        self.assertIn("full_name", profile.detected_metadata.potentially_sensitive_columns)
        self.assertIn("email_address", profile.detected_metadata.potentially_sensitive_columns)

        col_map = {c.name: c for c in profile.columns}
        self.assertTrue(col_map["user_id"].identifier)
        self.assertTrue(col_map["churn"].target_candidate)
        self.assertTrue(col_map["full_name"].sensitive_candidate)
        self.assertTrue(col_map["email_address"].sensitive_candidate)

    def test_constant_columns(self):
        df = pd.DataFrame({
            "const_num": [42, 42, 42, 42],
            "const_str": ["FIXED", "FIXED", "FIXED", "FIXED"],
            "normal_num": [1, 2, 3, 4],
        })
        profile = profile_dataframe(df)
        col_map = {c.name: c for c in profile.columns}

        self.assertEqual(col_map["const_num"].semantic_type, SemanticType.CONSTANT)
        self.assertEqual(col_map["const_str"].semantic_type, SemanticType.CONSTANT)
        self.assertEqual(col_map["normal_num"].semantic_type, SemanticType.NUMERIC)

    def test_empty_dataset(self):
        empty_df = pd.DataFrame()
        profile = profile_dataframe(empty_df, dataset_name="empty.csv")

        self.assertEqual(profile.row_count, 0)
        self.assertEqual(profile.column_count, 0)
        self.assertEqual(profile.missing_count, 0)
        self.assertEqual(len(profile.columns), 0)

        empty_with_cols = pd.DataFrame(columns=["a", "b", "c"])
        profile2 = profile_dataframe(empty_with_cols)
        self.assertEqual(profile2.row_count, 0)
        self.assertEqual(profile2.column_count, 3)

    def test_api_endpoint_healthcare(self):
        csv_content = (
            "patient_id,age,cholesterol,diagnosis\n"
            "1,55,220,1\n"
            "2,48,190,0\n"
            "3,63,240,1\n"
            "4,39,175,0\n"
        )
        response = self.client.post(
            "/api/data/profile",
            files={"file": ("healthcare.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        profile = data["data"]
        self.assertEqual(profile["row_count"], 4)
        self.assertEqual(profile["column_count"], 4)
        self.assertIn("patient_id", profile["detected_metadata"]["identifier_columns"])
        self.assertEqual(profile["detected_metadata"]["target_column"], "diagnosis")
        self.assertNotIn("records", profile)
        self.assertNotIn("rows", profile)

    def test_api_endpoint_non_healthcare(self):
        csv_content = (
            "customer_id,income,credit_score,default_status\n"
            "101,50000,700,no\n"
            "102,60000,650,yes\n"
            "103,75000,720,no\n"
        )
        response = self.client.post(
            "/api/data/profile",
            files={"file": ("banking.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        profile = data["data"]
        self.assertEqual(profile["row_count"], 3)
        self.assertEqual(profile["column_count"], 4)
        self.assertEqual(profile["detected_metadata"]["target_column"], "default_status")

    def test_api_endpoint_empty_or_invalid_file(self):
        # Empty file
        res_empty = self.client.post(
            "/api/data/profile",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        self.assertEqual(res_empty.status_code, 400)

        # Non-csv extension
        res_txt = self.client.post(
            "/api/data/profile",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        self.assertEqual(res_txt.status_code, 400)


if __name__ == "__main__":
    unittest.main()

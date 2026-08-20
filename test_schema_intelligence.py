import unittest

import pandas as pd

from backend.services.schema_intelligence import (
    detect_sensitive_columns,
    detect_target_column,
    drop_identifier_columns,
    get_identifier_columns,
    is_identifier_column,
)
from backend.services.statistical_validator import validate_statistical


class SchemaIntelligenceTests(unittest.TestCase):
    def test_identifier_detection_across_common_domains(self):
        schemas = {
            "healthcare": pd.DataFrame({
                "patient_id": [1, 2, 3, 4],
                "age": [42, 51, 63, 37],
                "blood_pressure": [120, 140, 135, 128],
                "diagnosis": [0, 1, 0, 1],
            }),
            "banking": pd.DataFrame({
                "customer_id": [101, 102, 103, 104],
                "age": [31, 45, 29, 52],
                "income": [55000, 72000, 48000, 91000],
                "loan_amount": [10000, 25000, 18000, 32000],
                "default_status": ["no", "yes", "no", "yes"],
            }),
            "education": pd.DataFrame({
                "student_id": [11, 12, 13, 14],
                "department": ["CS", "EE", "CS", "ME"],
                "cgpa": [8.1, 7.4, 9.0, 6.8],
                "attendance": [91, 84, 95, 78],
                "placement_status": [1, 0, 1, 0],
            }),
            "cybersecurity": pd.DataFrame({
                "event_id": [1001, 1002, 1003, 1004],
                "source_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
                "destination_port": [80, 443, 22, 8080],
                "packet_size": [1200, 900, 1500, 700],
                "attack_type": ["scan", "malware", "scan", "phishing"],
            }),
        }

        expected_targets = {
            "healthcare": "diagnosis",
            "banking": "default_status",
            "education": "placement_status",
            "cybersecurity": "attack_type",
        }

        for name, frame in schemas.items():
            with self.subTest(schema=name):
                identifiers = get_identifier_columns(frame)
                self.assertIn(frame.columns[0], identifiers)
                self.assertTrue(is_identifier_column(frame[frame.columns[0]], frame.columns[0], dataset_size=len(frame)))
                self.assertEqual(detect_target_column(frame), expected_targets[name])

    def test_sensitive_column_detection_is_generic(self):
        frame = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "name": ["A", "B", "C", "D"],
            "email": ["a@example.com", "b@example.com", "c@example.com", "d@example.com"],
            "salary": [50000, 60000, 52000, 70000],
            "status": [0, 1, 0, 1],
        })

        sensitive = detect_sensitive_columns(frame)
        self.assertIn("name", sensitive)
        self.assertIn("email", sensitive)
        self.assertIn("salary", sensitive)
        self.assertNotIn("customer_id", sensitive)

    def test_statistical_validation_uses_schema_not_domain(self):
        real = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "age": [31, 45, 29, 52],
            "income": [55000, 72000, 48000, 91000],
            "loan_amount": [10000, 25000, 18000, 32000],
            "default_status": ["no", "yes", "no", "yes"],
        })
        synth = pd.DataFrame({
            "customer_id": [11, 12, 13, 14],
            "age": [32, 44, 30, 51],
            "income": [56000, 70500, 49500, 90000],
            "loan_amount": [9800, 25500, 17500, 31500],
            "default_status": ["no", "yes", "no", "yes"],
        })

        result = validate_statistical(drop_identifier_columns(real), drop_identifier_columns(synth))
        self.assertIn("overall_quality_score", result)
        self.assertEqual(result["num_numeric"], 3)
        self.assertEqual(result["num_categorical"], 1)


if __name__ == "__main__":
    unittest.main()
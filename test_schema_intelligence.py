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

    def test_phase5_identifier_protections_and_detection(self):
        # 1. Continuous floats with 100% uniqueness -> NOT identifier
        floats_series = pd.Series([10.512, 12.391, 15.823, 18.904, 21.055, 25.661, 30.128, 35.912, 40.111, 45.882,
                                   50.231, 55.441, 60.119, 65.884, 70.331, 75.221, 80.991, 85.123, 90.451, 95.992])
        self.assertFalse(is_identifier_column(floats_series, "distance_km", dataset_size=20))
        self.assertFalse(is_identifier_column(floats_series, "fuel_consumed_liters", dataset_size=20))
        self.assertFalse(is_identifier_column(floats_series, "weight_kg", dataset_size=20))
        self.assertFalse(is_identifier_column(floats_series, "transaction_amount", dataset_size=20))
        self.assertFalse(is_identifier_column(floats_series, "price", dataset_size=20))
        self.assertFalse(is_identifier_column(floats_series, "score", dataset_size=20))

        # 2. Sensor measurements -> NOT identifier
        sensor_series = pd.Series([3.301, 3.298, 3.305, 3.302, 3.299, 3.304, 3.301, 3.300, 3.297, 3.303,
                                  3.302, 3.299, 3.301, 3.306, 3.298, 3.300, 3.304, 3.302, 3.299, 3.303])
        self.assertFalse(is_identifier_column(sensor_series, "voltage_volts", dataset_size=20))
        self.assertFalse(is_identifier_column(sensor_series, "temperature_celsius", dataset_size=20))

        # 3. Datetime with 100% uniqueness -> NOT identifier
        dt_series = pd.Series([f"2024-01-01 {h:02d}:00:00" for h in range(24)])
        self.assertFalse(is_identifier_column(dt_series, "hire_date", dataset_size=24))
        self.assertFalse(is_identifier_column(dt_series, "transaction_time", dataset_size=24))
        self.assertFalse(is_identifier_column(dt_series, "stream_date", dataset_size=24))

        # 4. Repeated values in IDs -> IS identifier
        repeated_cust = pd.Series(["CUST-1", "CUST-2", "CUST-1", "CUST-3", "CUST-2", "CUST-4", "CUST-1", "CUST-5"])
        self.assertTrue(is_identifier_column(repeated_cust, "customer_id", dataset_size=8))

        repeated_acc = pd.Series(["ACC-101", "ACC-102", "ACC-101", "ACC-103", "ACC-102", "ACC-104", "ACC-101", "ACC-105"])
        self.assertTrue(is_identifier_column(repeated_acc, "account_number", dataset_size=8))
        self.assertTrue(is_identifier_column(repeated_acc, "account_no", dataset_size=8))
        self.assertTrue(is_identifier_column(repeated_acc, "policy_num", dataset_size=8))
        self.assertTrue(is_identifier_column(repeated_acc, "badge_no", dataset_size=8))

        # 5. UUID string -> IS identifier
        uuid_series = pd.Series(["123e4567-e89b-12d3-a456-426614174000", "123e4567-e89b-12d3-a456-426614174001",
                                "123e4567-e89b-12d3-a456-426614174002", "123e4567-e89b-12d3-a456-426614174003"])
        self.assertTrue(is_identifier_column(uuid_series, "device_uuid", dataset_size=4))

        # 6. Sequential integer IDs -> IS identifier
        seq_series = pd.Series(list(range(1, 25)))
        self.assertTrue(is_identifier_column(seq_series, "row_id", dataset_size=24))
        self.assertTrue(is_identifier_column(seq_series, "id", dataset_size=24))

    def test_phase5_target_candidate_scoring_and_unsupervised(self):
        # 1. category + returned -> returned
        ecom = pd.DataFrame({
            "category": ["Electronics", "Books", "Apparel", "Beauty"],
            "unit_price": [100.0, 20.0, 50.0, 30.0],
            "returned": [0, 1, 0, 1],
        })
        self.assertEqual(detect_target_column(ecom), "returned")

        # 2. is_manager + attrition -> attrition
        hr = pd.DataFrame({
            "salary": [60000, 80000, 95000, 110000],
            "is_manager": [0, 1, 0, 1],
            "attrition": [1, 0, 1, 0],
        })
        self.assertEqual(detect_target_column(hr), "attrition")

        # 3. requires_temperature_control + delayed -> delayed
        logistics = pd.DataFrame({
            "requires_temperature_control": [1, 0, 1, 0],
            "distance_km": [500.0, 1200.0, 300.0, 800.0],
            "delayed": [0, 1, 0, 1],
        })
        self.assertEqual(detect_target_column(logistics), "delayed")

        # 4. error_flag + is_anomaly -> is_anomaly
        iot = pd.DataFrame({
            "error_flag": [True, False, True, False],
            "temperature": [45.0, 48.0, 42.0, 50.0],
            "is_anomaly": [1, 0, 1, 0],
        })
        self.assertEqual(detect_target_column(iot), "is_anomaly")

        # 5. has_subtitles + liked -> liked
        movies = pd.DataFrame({
            "has_subtitles": [True, True, False, False],
            "watch_duration": [120, 90, 110, 85],
            "liked": [1, 0, 1, 1],
        })
        self.assertEqual(detect_target_column(movies), "liked")

        # 6. is_international + is_fraud -> is_fraud
        fin = pd.DataFrame({
            "is_international": [0, 1, 0, 0],
            "amount": [50.0, 1200.0, 30.0, 80.0],
            "is_fraud": [0, 1, 0, 0],
        })
        self.assertEqual(detect_target_column(fin), "is_fraud")

        # 7. Unsupervised numeric dataset -> target None
        unsupervised = pd.DataFrame({
            "sensor_1": [1.2, 2.3, 3.4, 4.5],
            "sensor_2": [10.1, 11.2, 12.3, 13.4],
            "sensor_3": [100.5, 101.6, 102.7, 103.8],
        })
        self.assertIsNone(detect_target_column(unsupervised))


if __name__ == "__main__":
    unittest.main()
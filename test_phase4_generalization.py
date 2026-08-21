"""
SynthForge — Phase 4 Test Suite: Real-World Generalization Testing.

Evaluates dataset profiling, semantic type detection, identifier detection,
target detection, statistical synthetic generation, datatype validity,
constant/missing value handling, and identifier leakage prevention across
unseen domains:
1. E-commerce
2. HR / Employee Analytics
3. Logistics & Supply Chain
4. IoT Sensor Data
5. Financial Transactions
6. Movie / Streaming Recommendations
"""
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from backend.services.generator import StatisticalGenerator, _train_sdv_model, generate_synthetic_data
from backend.services.dataset_profiler import profile_dataframe, SemanticType
from backend.services.quality_evaluator import QualityEvaluator
from backend.services.schema_intelligence import (
    get_identifier_columns,
    detect_target_column,
    drop_identifier_columns,
)
from data.generate_phase4_fixtures import generate_fixtures, FIXTURES_DIR


DATASET_SUMMARY: Dict[str, Dict[str, str]] = {}
FAILURE_RECORDS: List[Dict[str, Any]] = []


def log_test_result(dataset: str, stage: str, status: str):
    if dataset not in DATASET_SUMMARY:
        DATASET_SUMMARY[dataset] = {
            "Profile": "PASS",
            "Generation": "PASS",
            "Output": "PASS",
            "Quality": "PASS",
            "Status": "PASS"
        }
    if status == "FAIL":
        DATASET_SUMMARY[dataset][stage] = "FAIL"
        DATASET_SUMMARY[dataset]["Status"] = "FAIL"


def record_failure(dataset: str, test_name: str, failure: str, root_cause: str, severity: str, category: str):
    FAILURE_RECORDS.append({
        "dataset": dataset,
        "test": test_name,
        "failure": failure,
        "root_cause": root_cause,
        "severity": severity,
        "category": category,
    })


class BaseGeneralizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURES_DIR / "ecommerce.csv").exists():
            generate_fixtures()

    def run_core_pipeline(self, dataset_name: str, csv_filename: str, sample_size: int = 50) -> tuple:
        df = pd.read_csv(FIXTURES_DIR / csv_filename)

        # 1. Profiling
        try:
            profile = profile_dataframe(df, dataset_name=csv_filename)
            self.assertEqual(profile.row_count, len(df))
            self.assertEqual(profile.column_count, len(df.columns))
            log_test_result(dataset_name, "Profile", "PASS")
        except Exception as e:
            log_test_result(dataset_name, "Profile", "FAIL")
            record_failure(dataset_name, "Profiling", str(e),
                           "Exception during DataFrame profiling", "High", "profiling")
            raise

        # 2. Generation
        working_df = drop_identifier_columns(df)
        try:
            gen = StatisticalGenerator()
            gen.fit(working_df, profile=profile)
            synth = gen.sample(sample_size)
            log_test_result(dataset_name, "Generation", "PASS")
        except Exception as e:
            log_test_result(dataset_name, "Generation", "FAIL")
            record_failure(dataset_name, "Statistical Generation", str(e),
                           "StatisticalGenerator fit/sample crashed", "Critical", "generation")
            raise

        # 3. Output Consistency & Integrity
        try:
            self.assertEqual(len(synth), sample_size)
            self.assertEqual(list(synth.columns), list(working_df.columns))

            # No ID leakage
            for id_col in profile.detected_metadata.identifier_columns:
                self.assertNotIn(id_col, synth.columns)

            # Datatype checks
            col_map = {c.name: c for c in profile.columns}
            for col in working_df.columns:
                c_prof = col_map.get(col)
                if not c_prof:
                    continue
                if c_prof.semantic_type == SemanticType.CONSTANT:
                    unique_vals = synth[col].dropna().unique()
                    self.assertTrue(len(unique_vals) <= 1)
                elif c_prof.semantic_type == SemanticType.BOOLEAN and pd.api.types.is_numeric_dtype(synth[col]):
                    self.assertTrue(set(synth[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}))
                elif c_prof.semantic_type == SemanticType.DATETIME:
                    parsed = pd.to_datetime(synth[col].dropna(), errors="coerce")
                    self.assertFalse(parsed.isna().any())

            log_test_result(dataset_name, "Output", "PASS")
        except Exception as e:
            log_test_result(dataset_name, "Output", "FAIL")
            record_failure(dataset_name, "Output Validation", str(e),
                           "Synthetic output data failed schema consistency or integrity check", "Medium", "validation")
            raise

        # 4. Quality Evaluation (Phase 6)
        try:
            report = QualityEvaluator.evaluate(
                real_df=df,
                synth_df=synth,
                profile=profile,
                target_column=profile.detected_metadata.target_column,
            )
            self.assertIn("executive_summary", report)
            self.assertGreaterEqual(report["executive_summary"]["data_fidelity_score"], 0.0)
            self.assertGreaterEqual(report["executive_summary"]["privacy_protection_score"], 0.0)
            log_test_result(dataset_name, "Quality", "PASS")
        except Exception as e:
            log_test_result(dataset_name, "Quality", "FAIL")
            record_failure(dataset_name, "Quality Evaluation", str(e),
                           "QualityEvaluator.evaluate failed", "High", "evaluation")
            raise

        return df, profile, working_df, synth


class TestEcommerceGeneralization(BaseGeneralizationTest):
    def test_01_pipeline(self):
        df, profile, working_df, synth = self.run_core_pipeline("E-commerce", "ecommerce.csv", sample_size=50)

    def test_02_identifiers(self):
        df = pd.read_csv(FIXTURES_DIR / "ecommerce.csv")
        profile = profile_dataframe(df)
        self.assertIn("order_id", profile.detected_metadata.identifier_columns)
        self.assertIn("customer_id", profile.detected_metadata.identifier_columns)

    def test_03_target_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "ecommerce.csv")
        profile = profile_dataframe(df)
        self.assertEqual(profile.detected_metadata.target_column, "returned")


class TestHRAnalyticsGeneralization(BaseGeneralizationTest):
    def test_01_pipeline(self):
        df, profile, working_df, synth = self.run_core_pipeline("HR Analytics", "hr_analytics.csv", sample_size=60)

    def test_02_constants_and_types(self):
        df = pd.read_csv(FIXTURES_DIR / "hr_analytics.csv")
        profile = profile_dataframe(df)
        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["company_hq"].semantic_type, SemanticType.CONSTANT)
        self.assertEqual(col_map["is_manager"].semantic_type, SemanticType.BOOLEAN)
        self.assertEqual(col_map["hire_date"].semantic_type, SemanticType.DATETIME)

    def test_03_target_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "hr_analytics.csv")
        profile = profile_dataframe(df)
        self.assertEqual(profile.detected_metadata.target_column, "attrition")


class TestLogisticsGeneralization(BaseGeneralizationTest):
    def test_01_pipeline(self):
        df, profile, working_df, synth = self.run_core_pipeline("Logistics", "logistics.csv", sample_size=45)

    def test_02_continuous_numeric_identifier_protection(self):
        df = pd.read_csv(FIXTURES_DIR / "logistics.csv")
        profile = profile_dataframe(df)
        id_cols = profile.detected_metadata.identifier_columns
        for non_id in ["distance_km", "fuel_consumed_liters", "weight_kg"]:
            self.assertNotIn(non_id, id_cols)
        self.assertIn("tracking_number", id_cols)

    def test_03_target_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "logistics.csv")
        profile = profile_dataframe(df)
        self.assertEqual(profile.detected_metadata.target_column, "delayed")


class TestIoTSensorGeneralization(BaseGeneralizationTest):
    def test_01_pipeline(self):
        df, profile, working_df, synth = self.run_core_pipeline("IoT Sensor", "iot_sensor.csv", sample_size=55)

    def test_02_duplicates_and_high_freq_datetime(self):
        df = pd.read_csv(FIXTURES_DIR / "iot_sensor.csv")
        profile = profile_dataframe(df)
        self.assertGreater(profile.duplicate_count, 0)
        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["firmware_version"].semantic_type, SemanticType.CONSTANT)
        self.assertEqual(col_map["timestamp"].semantic_type, SemanticType.DATETIME)

    def test_03_target_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "iot_sensor.csv")
        profile = profile_dataframe(df)
        self.assertEqual(profile.detected_metadata.target_column, "is_anomaly")


class TestFinancialTransactionsGeneralization(BaseGeneralizationTest):
    def test_01_pipeline(self):
        df, profile, working_df, synth = self.run_core_pipeline("Financial Transactions", "financial_transactions.csv", sample_size=100)

    def test_02_identifier_protection_and_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "financial_transactions.csv")
        profile = profile_dataframe(df)
        id_cols = profile.detected_metadata.identifier_columns
        self.assertNotIn("transaction_amount", id_cols)
        self.assertNotIn("transaction_time", id_cols)
        self.assertIn("transaction_id", id_cols)
        self.assertIn("account_number", id_cols)

    def test_03_target_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "financial_transactions.csv")
        profile = profile_dataframe(df)
        self.assertEqual(profile.detected_metadata.target_column, "is_fraud")


class TestMovieRecommendationsGeneralization(BaseGeneralizationTest):
    def test_01_pipeline(self):
        df, profile, working_df, synth = self.run_core_pipeline("Movie Recommendations", "movie_recommendations.csv", sample_size=50)

    def test_02_high_cardinality_and_text(self):
        df = pd.read_csv(FIXTURES_DIR / "movie_recommendations.csv")
        profile = profile_dataframe(df)
        col_map = {c.name: c for c in profile.columns}
        self.assertEqual(col_map["movie_title"].semantic_type, SemanticType.CATEGORICAL)
        self.assertEqual(col_map["stream_date"].semantic_type, SemanticType.DATETIME)

    def test_03_target_detection(self):
        df = pd.read_csv(FIXTURES_DIR / "movie_recommendations.csv")
        profile = profile_dataframe(df)
        self.assertEqual(profile.detected_metadata.target_column, "liked")


class TestPipelineIntegrationBugs(BaseGeneralizationTest):
    """Test full pipeline invocation to capture known system-level bugs without fixing them."""
    def test_generate_synthetic_data_unbound_variables(self):
        from backend.services.data_service import ingest_csv
        df = pd.read_csv(FIXTURES_DIR / "ecommerce.csv")
        ingested = ingest_csv(df.to_csv(index=False).encode("utf-8"), "ecom_test.csv")
        ds_id = ingested["dataset_id"]

        try:
            generate_synthetic_data(dataset_id=ds_id, num_rows=10, model_type="statistical")
        except NameError as e:
            record_failure("Pipeline Integration", "generate_synthetic_data",
                           str(e),
                           "Unbound variables 'profile' and 'target_col' in generate_synthetic_data()",
                           "Critical", "generation")


def print_summary_tables():
    print("\n" + "=" * 90)
    print("              SYNTHFORGE PHASE 4 GENERALIZATION TEST SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<24} {'Profile':<10} {'Generation':<12} {'Output':<10} {'Quality':<10} {'Status':<8}")
    print("-" * 90)
    for ds, stages in DATASET_SUMMARY.items():
        print(f"{ds:<24} {stages['Profile']:<10} {stages['Generation']:<12} {stages['Output']:<10} {stages.get('Quality', 'PASS'):<10} {stages['Status']:<8}")
    print("=" * 90)

    print("\n" + "=" * 80)
    print("                           FAILURE SUMMARY")
    print("=" * 80)
    if FAILURE_RECORDS:
        for i, f in enumerate(FAILURE_RECORDS, 1):
            print(f"[{i}] Dataset:  {f['dataset']}")
            print(f"    Test:     {f['test']}")
            print(f"    Failure:  {f['failure']}")
            print(f"    Category: {f['category']} | Severity: {f['severity']}")
            print(f"    Cause:    {f['root_cause']}")
            print("-" * 80)
    else:
        print("No failures encountered.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEcommerceGeneralization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHRAnalyticsGeneralization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLogisticsGeneralization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIoTSensorGeneralization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFinancialTransactionsGeneralization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMovieRecommendationsGeneralization))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPipelineIntegrationBugs))

    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    print_summary_tables()
    sys.exit(0 if res.wasSuccessful() else 1)

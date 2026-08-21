"""
SynthForge — Phase 6 Test Suite: Unified Quality & Trustworthiness Evaluation.
"""
import unittest
import numpy as np
import pandas as pd

from backend.services.quality_evaluator import (
    QualityEvaluator,
    evaluate_structural_fidelity,
    evaluate_statistical_fidelity,
    evaluate_relationship_fidelity,
    evaluate_ml_utility,
    evaluate_privacy_risk,
    check_exact_duplicate_collisions,
)


class TestQualityEvaluator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.mixed_real = pd.DataFrame({
            "age": np.random.normal(50, 10, n).clip(18, 90).astype(int),
            "income": np.random.normal(60000, 15000, n).clip(20000, 120000),
            "department": np.random.choice(["Sales", "Engineering", "HR"], n),
            "promoted": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        })
        self.mixed_synth = pd.DataFrame({
            "age": np.random.normal(51, 9.8, n).clip(18, 90).astype(int),
            "income": np.random.normal(59500, 14800, n).clip(20000, 120000),
            "department": np.random.choice(["Sales", "Engineering", "HR"], n),
            "promoted": np.random.choice([0, 1], n, p=[0.68, 0.32]),
        })

    def test_normal_mixed_dataset(self):
        """Test unified quality evaluation on normal mixed numeric/categorical dataset."""
        report = QualityEvaluator.evaluate(self.mixed_real, self.mixed_synth, target_column="promoted")
        self.assertIn("executive_summary", report)
        self.assertIn("structural_fidelity", report)
        self.assertIn("statistical_fidelity", report)
        self.assertIn("relationship_fidelity", report)
        self.assertIn("ml_utility", report)
        self.assertIn("privacy_risk", report)

        summary = report["executive_summary"]
        self.assertGreaterEqual(summary["data_fidelity_score"], 0.0)
        self.assertLessEqual(summary["data_fidelity_score"], 100.0)
        self.assertGreaterEqual(summary["privacy_protection_score"], 0.0)
        self.assertLessEqual(summary["privacy_protection_score"], 100.0)
        self.assertIn(summary["data_fidelity_grade"], ["A", "B", "C", "D", "F"])

    def test_numeric_only_dataset(self):
        """Test dataset containing only numeric columns."""
        real = pd.DataFrame({
            "x1": np.random.normal(10, 2, 80),
            "x2": np.random.normal(20, 5, 80),
            "x3": np.random.normal(30, 8, 80),
        })
        synth = pd.DataFrame({
            "x1": np.random.normal(10.2, 2.1, 80),
            "x2": np.random.normal(19.8, 4.9, 80),
            "x3": np.random.normal(30.1, 7.8, 80),
        })
        report = QualityEvaluator.evaluate(real, synth)
        self.assertTrue(report["structural_fidelity"]["score"] > 80)
        self.assertTrue(report["relationship_fidelity"]["applicable"])
        self.assertIsNotNone(report["relationship_fidelity"]["metrics"]["pearson_mae"])
        self.assertIsNone(report["relationship_fidelity"]["metrics"]["cramers_v_mae"])

    def test_categorical_only_dataset(self):
        """Test dataset containing only categorical columns."""
        real = pd.DataFrame({
            "cat1": np.random.choice(["A", "B", "C"], 60),
            "cat2": np.random.choice(["X", "Y"], 60),
            "cat3": np.random.choice(["High", "Low"], 60),
        })
        synth = pd.DataFrame({
            "cat1": np.random.choice(["A", "B", "C"], 60),
            "cat2": np.random.choice(["X", "Y"], 60),
            "cat3": np.random.choice(["High", "Low"], 60),
        })
        report = QualityEvaluator.evaluate(real, synth)
        self.assertIsNone(report["relationship_fidelity"]["metrics"]["pearson_mae"])
        self.assertIsNotNone(report["relationship_fidelity"]["metrics"]["cramers_v_mae"])
        self.assertTrue(report["structural_fidelity"]["score"] > 80)

    def test_dataset_without_target(self):
        """Test unsupervised evaluation where no target column exists."""
        real = pd.DataFrame({
            "feat1": np.random.normal(0, 1, 50),
            "feat2": np.random.normal(5, 2, 50),
        })
        synth = pd.DataFrame({
            "feat1": np.random.normal(0.1, 1.0, 50),
            "feat2": np.random.normal(4.9, 2.1, 50),
        })
        report = QualityEvaluator.evaluate(real, synth, target_column=None)
        self.assertFalse(report["ml_utility"]["applicable"])
        self.assertNotIn("ml_utility", report["executive_summary"]["applicable_pillars"])
        self.assertGreaterEqual(report["executive_summary"]["data_fidelity_score"], 0.0)

    def test_constant_columns(self):
        """Test handling of constant columns without crashing."""
        real = pd.DataFrame({
            "const_col": ["fixed_val"] * 50,
            "val": np.random.normal(10, 2, 50),
        })
        synth = pd.DataFrame({
            "const_col": ["fixed_val"] * 50,
            "val": np.random.normal(10, 2, 50),
        })
        report = QualityEvaluator.evaluate(real, synth)
        self.assertEqual(report["structural_fidelity"]["metrics"]["constant_preservation_rate"], 1.0)

    def test_all_null_columns(self):
        """Test handling of columns that are completely null."""
        real = pd.DataFrame({
            "null_col": [np.nan] * 40,
            "num_col": np.random.normal(5, 1, 40),
        })
        synth = pd.DataFrame({
            "null_col": [np.nan] * 40,
            "num_col": np.random.normal(5, 1, 40),
        })
        report = QualityEvaluator.evaluate(real, synth)
        self.assertIsNotNone(report["structural_fidelity"]["score"])
        self.assertEqual(report["structural_fidelity"]["metrics"]["mean_missing_difference"], 0.0)

    def test_high_missingness(self):
        """Test dataset with very high missingness rate."""
        real = pd.DataFrame({
            "sparse_1": [np.nan if i % 10 != 0 else i for i in range(50)],
            "sparse_2": [np.nan if i % 5 != 0 else i for i in range(50)],
        })
        synth = pd.DataFrame({
            "sparse_1": [np.nan if i % 8 != 0 else i for i in range(50)],
            "sparse_2": [np.nan if i % 6 != 0 else i for i in range(50)],
        })
        report = QualityEvaluator.evaluate(real, synth)
        self.assertIsNotNone(report["structural_fidelity"]["score"])

    def test_small_dataset(self):
        """Test very small dataset (N = 8) without crashing."""
        real = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": ["x", "y", "x", "y", "x", "y", "x", "y"],
        })
        synth = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": ["x", "y", "x", "y", "x", "y", "x", "y"],
        })
        report = QualityEvaluator.evaluate(real, synth)
        self.assertIsNotNone(report["executive_summary"]["data_fidelity_score"])

    def test_unseen_synthetic_categories(self):
        """Test detection of synthetic category overflow."""
        real = pd.DataFrame({
            "color": ["red", "blue", "green", "red", "blue"],
            "val": [10, 20, 30, 40, 50],
        })
        synth = pd.DataFrame({
            "color": ["red", "blue", "purple", "yellow", "green"],  # purple, yellow unseen
            "val": [12, 18, 29, 41, 48],
        })
        res = evaluate_structural_fidelity(real, synth)
        overflow = res["details"]["categorical_overflow"]["color"]["novel_categories"]
        self.assertIn("purple", overflow)
        self.assertIn("yellow", overflow)
        self.assertTrue(any("novel categories" in w for w in res["warnings"]))

    def test_exact_duplicate_synthetic_records(self):
        """Test exact duplicate collision detection between real and synthetic data."""
        real = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
        })
        # Synthetic copy-pastes 2 rows from real
        synth = pd.DataFrame({
            "col1": [1, 2, 99, 100, 101],
            "col2": ["a", "b", "z", "w", "v"],
        })
        count, rate = check_exact_duplicate_collisions(real, synth)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(rate, 0.4)

        report = evaluate_privacy_risk(real, synth)
        self.assertEqual(report["metrics"]["exact_duplicate_count"], 2)
        self.assertTrue(any("EXACT MATCH COLLISION" in w for w in report["warnings"]))

    def test_different_column_ordering(self):
        """Test evaluation when columns in synth are ordered differently than real."""
        real = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        synth = pd.DataFrame({"c": [7, 8, 9], "a": [1, 2, 3], "b": [4, 5, 6]})
        report = QualityEvaluator.evaluate(real, synth)
        self.assertEqual(report["structural_fidelity"]["metrics"]["column_preservation_rate"], 1.0)
        self.assertEqual(len(report["structural_fidelity"]["metrics"]["missing_columns"]), 0)

    def test_missing_columns(self):
        """Test structural warning when synthetic data is missing a real column."""
        real = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        synth = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        res = evaluate_structural_fidelity(real, synth)
        self.assertIn("c", res["metrics"]["missing_columns"])
        self.assertLess(res["metrics"]["column_preservation_rate"], 1.0)
        self.assertTrue(any("missing columns" in w for w in res["warnings"]))

    def test_duplicate_rows(self):
        """Test within-dataset duplicate rate difference."""
        real = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        synth = pd.DataFrame({"a": [1, 1, 1, 1], "b": [10, 10, 10, 10]})  # 75% duplicate
        res = evaluate_structural_fidelity(real, synth)
        self.assertGreater(res["metrics"]["synth_duplicate_rate"], 0.5)
        self.assertGreater(res["metrics"]["duplicate_rate_diff"], 0.5)


if __name__ == "__main__":
    unittest.main()

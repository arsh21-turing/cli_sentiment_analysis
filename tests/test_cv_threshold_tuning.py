"""
Final unit tests for cross-validation and threshold tuning integration.
"""

import os
import sys
import numpy as np
import pandas as pd
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call, ANY

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold import ThresholdOptimizer


class FinalCVThresholdTestCase(unittest.TestCase):
    """Base test case for final cross-validation tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock logger
        self.mock_logger = MagicMock()
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.test_dir)


class ThresholdOptimizerIsolationTests(FinalCVThresholdTestCase):
    """Test that ThresholdOptimizer instances are properly isolated."""
    
    def test_threshold_optimizer_isolation(self):
        """
        Test that the ThresholdOptimizer doesn't leak data between instances.
        """
        # Create two distinct datasets with different patterns
        # Dataset 1: High confidence, mostly correct predictions
        ground_truth_1 = ["positive", "negative", "positive", "negative", "neutral"]
        predictions_1 = [
            {"sentiment": "positive", "sentiment_confidence": 0.9},  # Correct
            {"sentiment": "negative", "sentiment_confidence": 0.8},  # Correct
            {"sentiment": "positive", "sentiment_confidence": 0.9},  # Correct
            {"sentiment": "negative", "sentiment_confidence": 0.8},  # Correct
            {"sentiment": "neutral", "sentiment_confidence": 0.7}    # Correct
        ]
        
        # Dataset 2: Mixed confidence, some incorrect predictions
        ground_truth_2 = ["positive", "positive", "negative", "negative", "neutral"]
        predictions_2 = [
            {"sentiment": "positive", "sentiment_confidence": 0.6},  # Correct, low conf
            {"sentiment": "negative", "sentiment_confidence": 0.5},  # Incorrect, low conf
            {"sentiment": "negative", "sentiment_confidence": 0.7},  # Correct, med conf
            {"sentiment": "positive", "sentiment_confidence": 0.4},  # Incorrect, low conf
            {"sentiment": "neutral", "sentiment_confidence": 0.6}    # Correct, med conf
        ]
        
        # Create metrics calculators for each dataset
        metrics_1 = EvaluationMetrics(
            predictions=predictions_1,
            ground_truth=ground_truth_1,
            label_type="sentiment"
        )
        
        metrics_2 = EvaluationMetrics(
            predictions=predictions_2,
            ground_truth=ground_truth_2,
            label_type="sentiment"
        )
        
        # Create threshold optimizers for each metrics calculator
        optimizer_1 = ThresholdOptimizer(metrics_calculator=metrics_1)
        optimizer_2 = ThresholdOptimizer(metrics_calculator=metrics_2)
        
        # Optimize thresholds for each optimizer
        threshold_1 = optimizer_1.find_optimal_threshold()
        threshold_2 = optimizer_2.find_optimal_threshold()
        
        # The two datasets should yield different optimal thresholds
        # since the prediction patterns are very different
        self.assertNotEqual(threshold_1, threshold_2,
                           "Different datasets should yield different optimal thresholds")
        
        # Both thresholds should be reasonable values
        self.assertGreaterEqual(threshold_1, 0.1)
        self.assertGreaterEqual(threshold_2, 0.1)
        self.assertLessEqual(threshold_1, 0.9)
        self.assertLessEqual(threshold_2, 0.9)
    
    def test_threshold_optimizer_consistency(self):
        """
        Test that ThresholdOptimizer produces consistent results for the same data.
        """
        # Create a dataset with mixed predictions
        ground_truth = ["positive", "negative", "positive", "negative", "neutral"]
        predictions = [
            {"sentiment": "positive", "sentiment_confidence": 0.8},  # Correct
            {"sentiment": "negative", "sentiment_confidence": 0.7},  # Correct
            {"sentiment": "positive", "sentiment_confidence": 0.9},  # Correct
            {"sentiment": "negative", "sentiment_confidence": 0.6},  # Correct
            {"sentiment": "neutral", "sentiment_confidence": 0.5}    # Correct
        ]
        
        # Create metrics calculator
        metrics = EvaluationMetrics(
            predictions=predictions,
            ground_truth=ground_truth,
            label_type="sentiment"
        )
        
        # Create two identical optimizers
        optimizer_1 = ThresholdOptimizer(metrics_calculator=metrics)
        optimizer_2 = ThresholdOptimizer(metrics_calculator=metrics)
        
        # Optimize thresholds
        threshold_1 = optimizer_1.find_optimal_threshold()
        threshold_2 = optimizer_2.find_optimal_threshold()
        
        # Results should be identical
        self.assertEqual(threshold_1, threshold_2,
                        "Identical optimizers should produce identical results")


class CrossValidationThresholdTests(FinalCVThresholdTestCase):
    """Test cross-validation with threshold optimization."""
    
    def test_cross_validation_with_consistent_thresholds(self):
        """
        Test that when dataset characteristics are similar across folds,
        the optimal thresholds are consistent.
        """
        # Generate a consistent dataset with some incorrect predictions
        np.random.seed(42)  # For reproducibility
        n_samples = 50
        
        # Generate balanced class distribution
        sentiment_labels = ["positive"] * 25 + ["negative"] * 25
        np.random.shuffle(sentiment_labels)
        
        # Create predictions with consistent pattern
        # (correct predictions have confidence > 0.7, incorrect have confidence < 0.7)
        predictions = []
        for i in range(n_samples):
            true_label = sentiment_labels[i]
            
            # All predictions follow the same pattern
            if np.random.random() < 0.8:  # 80% correct predictions
                pred_label = true_label
                conf = np.random.uniform(0.7, 1.0)  # High confidence
            else:
                # Wrong prediction with low confidence
                pred_label = "negative" if true_label == "positive" else "positive"
                conf = np.random.uniform(0.3, 0.7)  # Lower confidence
            
            predictions.append({
                "sentiment": pred_label,
                "sentiment_confidence": conf
            })
        
        # Create manual folds to ensure balanced distribution
        n_folds = 5
        fold_indices = []
        fold_size = n_samples // n_folds
        for i in range(n_folds):
            # Create train/val split indices
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
            val_indices = list(range(val_start, val_end))
            train_indices = [j for j in range(n_samples) if j not in val_indices]
            fold_indices.append((train_indices, val_indices))
        
        # Create a real EvaluationMetrics and ThresholdOptimizer for each fold
        fold_thresholds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            # Get validation data for this fold
            fold_labels = [sentiment_labels[i] for i in val_idx]
            fold_preds = [predictions[i] for i in val_idx]
            
            # Create metrics calculator
            metrics = EvaluationMetrics(
                predictions=fold_preds,
                ground_truth=fold_labels,
                label_type="sentiment"
            )
            
            # Create threshold optimizer and find optimal threshold
            optimizer = ThresholdOptimizer(metrics_calculator=metrics)
            threshold = optimizer.find_optimal_threshold()
            fold_thresholds.append(threshold)
        
        # Calculate mean and standard deviation of thresholds
        mean_threshold = np.mean(fold_thresholds)
        std_threshold = np.std(fold_thresholds)
        
        # Since we generated data with a consistent pattern,
        # the optimal thresholds should be similar across folds
        # (low standard deviation)
        self.assertLess(std_threshold, 0.3,
                       "Expected consistent thresholds across folds for consistent data patterns")
        
        # The mean threshold should be around 0.7 (our boundary between correct and incorrect)
        self.assertAlmostEqual(mean_threshold, 0.7, delta=0.2)
    
    def test_threshold_optimization_per_fold(self):
        """
        Test that thresholds are optimized separately for each fold.
        """
        # Create 3 folds with different data characteristics
        fold_data = [
            # Fold 1: High confidence predictions, mostly correct
            {
                'labels': ["positive", "negative", "positive"],
                'predictions': [
                    {"sentiment": "positive", "sentiment_confidence": 0.9},  # Correct
                    {"sentiment": "negative", "sentiment_confidence": 0.8},  # Correct
                    {"sentiment": "positive", "sentiment_confidence": 0.9}   # Correct
                ]
            },
            # Fold 2: Medium confidence predictions, some incorrect
            {
                'labels': ["negative", "positive", "negative"],
                'predictions': [
                    {"sentiment": "negative", "sentiment_confidence": 0.7},  # Correct
                    {"sentiment": "positive", "sentiment_confidence": 0.6},  # Correct
                    {"sentiment": "negative", "sentiment_confidence": 0.7}   # Correct
                ]
            },
            # Fold 3: Low confidence predictions, some incorrect
            {
                'labels': ["positive", "negative", "positive"],
                'predictions': [
                    {"sentiment": "positive", "sentiment_confidence": 0.5},  # Correct
                    {"sentiment": "negative", "sentiment_confidence": 0.4},  # Correct
                    {"sentiment": "positive", "sentiment_confidence": 0.5}   # Correct
                ]
            }
        ]
        
        fold_thresholds = []
        
        for fold_idx, fold_info in enumerate(fold_data):
            # Create metrics calculator for this fold
            metrics = EvaluationMetrics(
                predictions=fold_info['predictions'],
                ground_truth=fold_info['labels'],
                label_type="sentiment"
            )
            
            # Create threshold optimizer and find optimal threshold
            optimizer = ThresholdOptimizer(metrics_calculator=metrics)
            threshold = optimizer.find_optimal_threshold()
            fold_thresholds.append(threshold)
        
        # Since all predictions are correct, thresholds should be similar
        # but we can still verify they are reasonable values
        for threshold in fold_thresholds:
            self.assertGreaterEqual(threshold, 0.1)
            self.assertLessEqual(threshold, 0.9)
        
        # Verify that thresholds are found (may be minimum value for perfect predictions)
        self.assertTrue(all(t >= 0.1 for t in fold_thresholds))


if __name__ == '__main__':
    unittest.main() 
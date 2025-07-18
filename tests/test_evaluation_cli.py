"""
Unit tests for the evaluation CLI functionality.
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO
import numpy as np
import pandas as pd

# Import the modules to test
from src.evaluation.cli import (
    setup_logging, 
    load_model, 
    evaluate_with_batches,
    compare_multiple_models,
    perform_cross_validation,
    tune_threshold_params,
    perform_statistical_tests,
    export_to_format,
    run_evaluation_cli
)
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.data_loader import TestDataLoader, BatchTestLoader
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold import ThresholdOptimizer


class EvaluationCLITestCase(unittest.TestCase):
    """Base test case for evaluation CLI tests with common setup."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a small test dataset
        self.create_test_data()
        
        # Mock logger for testing
        self.mock_logger = MagicMock()
        
        # Set up common arguments for CLI commands
        self.base_args = MagicMock()
        self.base_args.data = os.path.join(self.test_dir, "test_data.csv")
        self.base_args.format = "csv"
        self.base_args.text_column = "text"
        self.base_args.sentiment_column = "sentiment"
        self.base_args.emotion_column = "emotion"
        self.base_args.label_types = "sentiment,emotion"
        self.base_args.model = "transformer"
        self.base_args.settings = None
        self.base_args.output_dir = self.test_dir
        self.base_args.generate_plots = False
        self.base_args.interactive_plots = False
        self.base_args.batch_size = 0
        self.base_args.parallel = False
        self.base_args.log_level = "INFO"
        self.base_args.log_file = None
        self.base_args.quiet = True
        self.base_args.report_name = "test_report"
        self.base_args.export_format = "json"
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Create a small test dataset for testing."""
        # Create CSV test file
        test_data = pd.DataFrame({
            "text": [
                "This is great!",
                "I am so happy.",
                "This is terrible.",
                "I hate this.",
                "This is okay.",
                "I'm feeling neutral.",
                "Amazing product, love it!",
                "Worst experience ever."
            ],
            "sentiment": [
                "positive",
                "positive",
                "negative",
                "negative",
                "neutral",
                "neutral",
                "positive",
                "negative"
            ],
            "emotion": [
                "joy",
                "joy",
                "anger",
                "anger",
                "neutral",
                "neutral",
                "joy",
                "anger"
            ]
        })
        
        # Save to CSV
        test_data.to_csv(os.path.join(self.test_dir, "test_data.csv"), index=False)
        
        # Save to JSON
        test_data.to_json(os.path.join(self.test_dir, "test_data.json"), orient="records", lines=True)
        
        # Create an edge case with only one class
        single_class_data = pd.DataFrame({
            "text": ["Positive text 1", "Positive text 2", "Positive text 3"],
            "sentiment": ["positive", "positive", "positive"],
            "emotion": ["joy", "joy", "joy"]
        })
        single_class_data.to_csv(os.path.join(self.test_dir, "single_class_data.csv"), index=False)
        
        # Create an empty dataset
        empty_data = pd.DataFrame(columns=["text", "sentiment", "emotion"])
        empty_data.to_csv(os.path.join(self.test_dir, "empty_data.csv"), index=False)
        
        # Create a dataset with missing values
        missing_data = pd.DataFrame({
            "text": ["Text with missing sentiment", "Text with missing emotion", "Complete text"],
            "sentiment": [np.nan, "positive", "negative"],
            "emotion": ["joy", np.nan, "anger"]
        })
        missing_data.to_csv(os.path.join(self.test_dir, "missing_data.csv"), index=False)


class SetupLoggingTests(EvaluationCLITestCase):
    """Tests for setup_logging function."""
    
    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        logger = setup_logging()
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "evaluation_cli")
        
        # Check that we have at least one handler
        self.assertTrue(len(logger.handlers) > 0)
        
        # Check log level
        self.assertEqual(logger.level, 20)  # INFO level
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with log file."""
        log_file = os.path.join(self.test_dir, "test.log")
        logger = setup_logging(log_file=log_file)
        
        # Check if file handler is in handlers
        file_handlers = [h for h in logger.handlers if hasattr(h, 'baseFilename')]
        self.assertTrue(len(file_handlers) > 0)
        self.assertEqual(file_handlers[0].baseFilename, log_file)
    
    def test_setup_logging_different_levels(self):
        """Test setup_logging with different logging levels."""
        levels = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50
        }
        
        for level_name, level_value in levels.items():
            logger = setup_logging(log_level=level_name)
            self.assertEqual(logger.level, level_value)
    
    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level."""
        logger = setup_logging(log_level="INVALID")
        # Should default to INFO
        self.assertEqual(logger.level, 20)


class LoadModelTests(EvaluationCLITestCase):
    """Tests for load_model function."""
    
    @patch("src.evaluation.cli.SentimentEmotionTransformer")
    @patch("src.evaluation.cli.Settings")
    def test_load_transformer_model(self, mock_settings, mock_transformer_model):
        """Test loading a transformer model."""
        # Set up mock
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        mock_model_instance = MagicMock()
        mock_transformer_model.return_value = mock_model_instance
        
        # Test loading transformer model
        model = load_model("transformer", logger=self.mock_logger)
        
        # Assert correct model was loaded
        self.assertEqual(model, mock_model_instance)
        mock_transformer_model.assert_called_once()
        self.mock_logger.info.assert_called_with("Loading model: transformer")
    
    @patch("src.evaluation.cli.GroqModel")
    @patch("src.evaluation.cli.Settings")
    def test_load_groq_model(self, mock_settings, mock_groq_model):
        """Test loading a groq model."""
        # Set up mock
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        mock_model_instance = MagicMock()
        mock_groq_model.return_value = mock_model_instance
        
        # Test loading groq model
        model = load_model("groq", logger=self.mock_logger)
        
        # Assert correct model was loaded
        self.assertEqual(model, mock_model_instance)
        mock_groq_model.assert_called_once()
        self.mock_logger.info.assert_called_with("Loading model: groq")
    
    @patch("src.evaluation.cli.FallbackSystem")
    @patch("src.evaluation.cli.Settings")
    def test_load_fallback_model(self, mock_settings, mock_fallback_model):
        """Test loading a fallback model."""
        # Set up mock
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        mock_model_instance = MagicMock()
        mock_fallback_model.return_value = mock_model_instance
        
        # Test loading fallback model
        model = load_model("fallback", logger=self.mock_logger)
        
        # Assert correct model was loaded
        self.assertEqual(model, mock_model_instance)
        mock_fallback_model.assert_called_once()
        self.mock_logger.info.assert_called_with("Loading model: fallback")
    
    @patch("src.evaluation.cli.Settings")
    def test_load_unknown_model(self, mock_settings):
        """Test loading an unknown model type raises an error."""
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        
        # Test loading unknown model
        with self.assertRaises(ValueError):
            load_model("unknown_model", logger=self.mock_logger)
    
    @patch("src.evaluation.cli.SentimentEmotionTransformer")
    @patch("src.evaluation.cli.Settings")
    def test_load_model_with_settings(self, mock_settings, mock_transformer_model):
        """Test loading a model with custom settings file."""
        # Set up mock
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        mock_model_instance = MagicMock()
        mock_transformer_model.return_value = mock_model_instance
        
        # Test loading model with settings
        model = load_model("transformer", settings_path="custom_settings.json", logger=self.mock_logger)
        
        # Assert settings were loaded and used
        self.assertEqual(model, mock_model_instance)
        mock_settings.assert_called_with("custom_settings.json")
        mock_transformer_model.assert_called_with(settings=mock_settings_instance)


class EvaluateWithBatchesTests(EvaluationCLITestCase):
    """Tests for evaluate_with_batches function."""
    
    def setUp(self):
        super().setUp()
        # Create mocks
        self.mock_evaluator = MagicMock(spec=ModelEvaluator)
        self.mock_model = MagicMock()
        # Remove analyze method so it uses predict
        del self.mock_model.analyze
        self.mock_data_loader = MagicMock(spec=BatchTestLoader)
        
        # Set up data loader for batches
        self.mock_data_loader.text_column = "text"
        self.mock_data_loader.label_columns = {"sentiment": "sentiment", "emotion": "emotion"}
        self.mock_data_loader.num_batches.return_value = 2
        
        # Sample batch data
        self.batch1 = pd.DataFrame({
            "text": ["Text 1", "Text 2"],
            "sentiment": ["positive", "negative"],
            "emotion": ["joy", "anger"]
        })
        
        self.batch2 = pd.DataFrame({
            "text": ["Text 3", "Text 4"],
            "sentiment": ["neutral", "positive"],
            "emotion": ["neutral", "joy"]
        })
        
        # Make the loader yield these batches
        self.mock_data_loader.iterate_batches.return_value = [self.batch1, self.batch2]
        
        # Set up model predictions
        self.batch1_predictions = [
            {"sentiment": "positive", "sentiment_confidence": 0.8, "emotion": "joy", "emotion_confidence": 0.7},
            {"sentiment": "negative", "sentiment_confidence": 0.7, "emotion": "anger", "emotion_confidence": 0.6}
        ]
        
        self.batch2_predictions = [
            {"sentiment": "neutral", "sentiment_confidence": 0.6, "emotion": "neutral", "emotion_confidence": 0.5},
            {"sentiment": "positive", "sentiment_confidence": 0.9, "emotion": "joy", "emotion_confidence": 0.8}
        ]
        
        # Make the model return these predictions
        self.mock_model.predict.side_effect = [
            self.batch1_predictions,
            self.batch2_predictions
        ]
        
        # Set up ground truth
        self.mock_data_loader.get_labels.side_effect = lambda label_type: (
            ["positive", "negative", "neutral", "positive"] if label_type == "sentiment" else
            ["joy", "anger", "neutral", "joy"]
        )
        
        # Set up evaluator to return results
        self.mock_evaluator.evaluate_model.return_value = {
            "sentiment": {
                "threshold": 0.5,
                "precision": 0.9,
                "recall": 0.85,
                "f1": 0.87
            },
            "emotion": {
                "threshold": 0.5,
                "precision": 0.85,
                "recall": 0.8,
                "f1": 0.82
            }
        }
    
    def test_evaluate_with_batches(self):
        """Test the batch evaluation process."""
        # Run batch evaluation
        results = evaluate_with_batches(
            evaluator=self.mock_evaluator,
            model=self.mock_model,
            data_loader=self.mock_data_loader,
            batch_size=2,
            label_types=["sentiment", "emotion"],
            logger=self.mock_logger
        )
        
        # Check that logging happened correctly
        self.mock_logger.info.assert_any_call("Starting batch evaluation with batch size: 2")
        self.mock_logger.info.assert_any_call("Processing batch 1/2")
        self.mock_logger.info.assert_any_call("Processing batch 2/2")
        
        # Check that batches were processed
        self.assertEqual(self.mock_model.predict.call_count, 2)
        self.mock_model.predict.assert_any_call(["Text 1", "Text 2"])
        self.mock_model.predict.assert_any_call(["Text 3", "Text 4"])
        
        # Check that ground truth was retrieved
        self.mock_data_loader.get_labels.assert_any_call("sentiment")
        self.mock_data_loader.get_labels.assert_any_call("emotion")
        
        # Check that all predictions were passed to evaluator
        all_predictions = self.batch1_predictions + self.batch2_predictions
        self.mock_evaluator.evaluate_model.assert_called_once()
        call_args = self.mock_evaluator.evaluate_model.call_args[1]
        self.assertEqual(len(call_args["predictions"]), 4)
        self.assertEqual(call_args["label_types"], ["sentiment", "emotion"])
        
        # Check that results were returned correctly
        self.assertEqual(results["sentiment"]["f1"], 0.87)
        self.assertEqual(results["emotion"]["f1"], 0.82)
        self.mock_logger.info.assert_any_call("Batch evaluation complete")
    
    def test_evaluate_with_batches_single_label_type(self):
        """Test batch evaluation with only one label type."""
        # Run batch evaluation with only sentiment
        results = evaluate_with_batches(
            evaluator=self.mock_evaluator,
            model=self.mock_model,
            data_loader=self.mock_data_loader,
            batch_size=2,
            label_types=["sentiment"],
            logger=self.mock_logger
        )
        
        # Check that only sentiment ground truth was retrieved
        self.mock_data_loader.get_labels.assert_called_once_with("sentiment")
        
        # Check that all predictions were passed to evaluator
        self.mock_evaluator.evaluate_model.assert_called_once()
        call_args = self.mock_evaluator.evaluate_model.call_args[1]
        self.assertEqual(call_args["label_types"], ["sentiment"])
    
    def test_evaluate_with_batches_no_logging(self):
        """Test batch evaluation without a logger."""
        # Run batch evaluation without logger
        results = evaluate_with_batches(
            evaluator=self.mock_evaluator,
            model=self.mock_model,
            data_loader=self.mock_data_loader,
            batch_size=2,
            label_types=["sentiment", "emotion"]
        )
        
        # Check that processing still happened correctly
        self.assertEqual(self.mock_model.predict.call_count, 2)
        self.mock_evaluator.evaluate_model.assert_called_once()


class CompareMultipleModelsTests(EvaluationCLITestCase):
    """Tests for compare_multiple_models function."""
    
    def setUp(self):
        super().setUp()
        # Set up mocks
        self.mock_data_loader = MagicMock(spec=TestDataLoader)
        
        # Set up data loader
        self.mock_data_loader.text_column = "text"
        self.mock_data_loader.label_columns = {"sentiment": "sentiment", "emotion": "emotion"}
        self.mock_data_loader.data = None  # Will be loaded
        
        # Mock data
        self.mock_texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        self.mock_sentiment_labels = ["positive", "negative", "neutral", "positive"]
        self.mock_emotion_labels = ["joy", "anger", "neutral", "joy"]
        
        # Make the loader return data
        self.mock_data_loader.get_texts.return_value = self.mock_texts
        self.mock_data_loader.get_labels.side_effect = lambda label_type: (
            self.mock_sentiment_labels if label_type == "sentiment" else
            self.mock_emotion_labels if label_type == "emotion" else []
        )
        
        # Model configurations
        self.model_configs = [
            {
                "name": "transformer",
                "display_name": "Transformer Model"
            },
            {
                "name": "fallback",
                "display_name": "Fallback Model"
            }
        ]
    
    @patch("src.evaluation.cli.load_model")
    @patch("src.evaluation.cli.ModelEvaluator")
    @patch("src.evaluation.cli.perform_statistical_tests")
    def test_compare_multiple_models(self, mock_perform_statistical_tests, mock_model_evaluator, mock_load_model):
        """Test comparison of multiple models."""
        # Set up mocks
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_load_model.side_effect = [mock_model1, mock_model2]
        
        # Mock predictions
        mock_model1.predict.return_value = [
            {"sentiment": "positive", "sentiment_confidence": 0.8},
            {"sentiment": "negative", "sentiment_confidence": 0.7},
            {"sentiment": "neutral", "sentiment_confidence": 0.6},
            {"sentiment": "positive", "sentiment_confidence": 0.9}
        ]
        
        mock_model2.predict.return_value = [
            {"sentiment": "positive", "sentiment_confidence": 0.7},
            {"sentiment": "negative", "sentiment_confidence": 0.8},
            {"sentiment": "positive", "sentiment_confidence": 0.6},  # Different
            {"sentiment": "positive", "sentiment_confidence": 0.9}
        ]
        
        # Mock evaluator
        mock_evaluator_instance = MagicMock()
        mock_model_evaluator.return_value = mock_evaluator_instance
        
        # Mock evaluation results
        mock_evaluator_instance.evaluate_model.side_effect = [
            {"sentiment": {"threshold": 0.5, "f1": 0.85, "precision": 0.8, "recall": 0.9}},
            {"sentiment": {"threshold": 0.5, "f1": 0.75, "precision": 0.7, "recall": 0.8}}
        ]
        
        # Mock significance test
        mock_perform_statistical_tests.return_value = {
            "transformer vs fallback": {
                "sentiment": {
                    "test_method": "bootstrap",
                    "better_model": 1,  # First model is better
                    "metrics": {"f1": {"significant": True, "p_value": 0.02}}
                }
            }
        }
        
        # Run comparison
        comparison_results = compare_multiple_models(
            model_configs=self.model_configs,
            data_loader=self.mock_data_loader,
            label_types=["sentiment"],
            metrics=["f1", "precision", "recall"],
            significance_test="bootstrap",
            output_dir=self.test_dir,
            logger=self.mock_logger
        )
        
        # Check logging
        self.mock_logger.info.assert_any_call("Starting comparison of 2 models")
        
        # Check that both models were loaded and evaluated
        self.assertEqual(mock_load_model.call_count, 2)
        self.assertEqual(mock_evaluator_instance.evaluate_model.call_count, 2)
        
        # Check that predictions were made for both models
        mock_model1.predict.assert_called_once_with(self.mock_texts)
        mock_model2.predict.assert_called_once_with(self.mock_texts)
        
        # Check that significance test was performed
        mock_perform_statistical_tests.assert_called_once()
        
        # Check result structure
        self.assertIn("models", comparison_results)
        self.assertIn("individual_results", comparison_results)
        self.assertIn("best_models", comparison_results)
        self.assertIn("statistical_tests", comparison_results)
        self.assertIn("best_model", comparison_results)
        self.assertIn("transformer", comparison_results["individual_results"])
        self.assertIn("fallback", comparison_results["individual_results"])
        
        # First model should be identified as best
        self.assertEqual(comparison_results["best_models"]["sentiment"], "transformer")
    
    @patch("src.evaluation.cli.load_model")
    @patch("src.evaluation.cli.ModelEvaluator")
    def test_compare_models_with_one_model_error(self, mock_model_evaluator, mock_load_model):
        """Test comparison when one model has an error."""
        # Set up mocks
        mock_model1 = MagicMock()
        mock_load_model.side_effect = [mock_model1, ValueError("Model error")]
        
        # Mock predictions
        mock_model1.predict.return_value = [
            {"sentiment": "positive", "sentiment_confidence": 0.8},
            {"sentiment": "negative", "sentiment_confidence": 0.7},
            {"sentiment": "neutral", "sentiment_confidence": 0.6},
            {"sentiment": "positive", "sentiment_confidence": 0.9}
        ]
        
        # Mock evaluator
        mock_evaluator_instance = MagicMock()
        mock_model_evaluator.return_value = mock_evaluator_instance
        
        # Mock evaluation results
        mock_evaluator_instance.evaluate_model.return_value = {
            "sentiment": {"threshold": 0.5, "f1": 0.85, "precision": 0.8, "recall": 0.9}
        }
        
        # Run comparison
        comparison_results = compare_multiple_models(
            model_configs=self.model_configs,
            data_loader=self.mock_data_loader,
            label_types=["sentiment"],
            metrics=["f1"],
            logger=self.mock_logger
        )
        
        # Check that the first model was evaluated
        self.assertEqual(mock_evaluator_instance.evaluate_model.call_count, 1)
        
        # Check that the error was recorded for the second model
        self.assertIn("transformer", comparison_results["individual_results"])
        self.assertIn("fallback", comparison_results["individual_results"])
        self.assertIn("error", comparison_results["individual_results"]["fallback"])
        
        # First model should still be identified as best
        self.assertEqual(comparison_results["best_models"]["sentiment"], "transformer")


class PerformCrossValidationTests(EvaluationCLITestCase):
    """Tests for perform_cross_validation function."""
    
    def setUp(self):
        super().setUp()
        # Set up mocks
        self.mock_evaluator = MagicMock(spec=ModelEvaluator)
        self.mock_model = MagicMock()
        self.mock_data_loader = MagicMock(spec=TestDataLoader)
        
        # Mock data
        self.mock_texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5", "Text 6"]
        self.mock_sentiment_labels = ["positive", "negative", "neutral", "positive", "negative", "positive"]
        self.mock_emotion_labels = ["joy", "anger", "neutral", "joy", "anger", "joy"]
        
        # Set up data loader
        self.mock_data_loader.text_column = "text"
        self.mock_data_loader.label_columns = {"sentiment": "sentiment", "emotion": "emotion"}
        # Create mock data DataFrame
        self.mock_data = pd.DataFrame({
            "text": self.mock_texts,
            "sentiment": self.mock_sentiment_labels,
            "emotion": self.mock_emotion_labels
        })
        self.mock_data_loader.data = self.mock_data
        
        # Make the loader return data
        self.mock_data_loader.get_texts.return_value = self.mock_texts
        self.mock_data_loader.get_labels.side_effect = lambda label_type: (
            self.mock_sentiment_labels if label_type == "sentiment" else
            self.mock_emotion_labels if label_type == "emotion" else []
        )
        
        # Mock predictions for each fold
        self.fold1_predictions = [
            {"sentiment": "positive", "sentiment_confidence": 0.8, "emotion": "joy", "emotion_confidence": 0.7},
            {"sentiment": "negative", "sentiment_confidence": 0.7, "emotion": "anger", "emotion_confidence": 0.6},
            {"sentiment": "neutral", "sentiment_confidence": 0.6, "emotion": "neutral", "emotion_confidence": 0.5},
        ]
        
        self.fold2_predictions = [
            {"sentiment": "positive", "sentiment_confidence": 0.9, "emotion": "joy", "emotion_confidence": 0.8},
            {"sentiment": "negative", "sentiment_confidence": 0.8, "emotion": "anger", "emotion_confidence": 0.7},
            {"sentiment": "positive", "sentiment_confidence": 0.7, "emotion": "joy", "emotion_confidence": 0.6},
        ]
        
        # Make the model return these predictions for different folds
        # Need 4 predictions total: 2 label types × 2 folds each
        self.mock_model.predict.side_effect = [
            self.fold1_predictions,  # sentiment fold 1
            self.fold2_predictions,  # sentiment fold 2
            self.fold1_predictions,  # emotion fold 1
            self.fold2_predictions   # emotion fold 2
        ]
    
    @patch("src.evaluation.cli.EvaluationMetrics")
    @patch("src.evaluation.cli.ThresholdOptimizer")
    def test_perform_cross_validation(self, mock_threshold_optimizer, mock_evaluation_metrics):
        """Test cross-validation with default settings."""
        # Set up mocks
        mock_metrics_instance = MagicMock()
        mock_evaluation_metrics.return_value = mock_metrics_instance
        mock_metrics_instance.calculate_f1.return_value = 0.85
        mock_metrics_instance.calculate_precision_recall.return_value = (0.9, 0.8)
        
        mock_optimizer_instance = MagicMock()
        mock_threshold_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.find_optimal_threshold.return_value = 0.5
        mock_optimizer_instance.sweep_thresholds.return_value = {0.1: {}, 0.2: {}}
        
        # Run cross-validation
        cv_results = perform_cross_validation(
            evaluator=self.mock_evaluator,
            model=self.mock_model,
            data_loader=self.mock_data_loader,
            label_types=["sentiment", "emotion"],
            cv_folds=2,
            metrics=["f1", "precision", "recall"],
            logger=self.mock_logger
        )
        
        # Check that logging happened
        self.mock_logger.info.assert_any_call("Performing 2-fold cross-validation")
        
        # Check that model predictions were requested for each fold
        self.assertEqual(self.mock_model.predict.call_count, 4)  # 2 label types × 2 folds
        
        # Check result structure
        self.assertIn("sentiment", cv_results)
        self.assertIn("emotion", cv_results)
        self.assertIn("fold_results", cv_results["sentiment"])
        self.assertIn("mean_threshold", cv_results["sentiment"])
        self.assertIn("std_threshold", cv_results["sentiment"])
        self.assertIn("mean_f1", cv_results["sentiment"])
        self.assertIn("std_f1", cv_results["sentiment"])
    
    @patch("src.evaluation.cli.EvaluationMetrics")
    @patch("src.evaluation.cli.ThresholdOptimizer")
    def test_cross_validation_without_threshold_optimization(self, mock_threshold_optimizer, mock_evaluation_metrics):
        """Test cross-validation without threshold optimization."""
        # Set up mocks
        mock_metrics_instance = MagicMock()
        mock_evaluation_metrics.return_value = mock_metrics_instance
        mock_metrics_instance.calculate_f1.return_value = 0.85
        
        # Run cross-validation without optimizing thresholds
        cv_results = perform_cross_validation(
            evaluator=self.mock_evaluator,
            model=self.mock_model,
            data_loader=self.mock_data_loader,
            label_types=["sentiment"],
            cv_folds=2,
            metrics=["f1"],
            optimize_threshold=False,
            logger=self.mock_logger
        )
        
        # Check that threshold optimizer was not used
        mock_threshold_optimizer.assert_not_called()
        
        # Check that default threshold was used
        self.assertIn("mean_threshold", cv_results["sentiment"])
        self.assertEqual(cv_results["sentiment"]["mean_threshold"], 0.5)


class TuneThresholdParamsTests(EvaluationCLITestCase):
    """Tests for tune_threshold_params function."""
    
    def setUp(self):
        super().setUp()
        # Set up mock evaluator, predictions and ground truth
        self.mock_evaluator = MagicMock(spec=ModelEvaluator)
        
        # Sample predictions
        self.predictions = [
            {"sentiment": "positive", "sentiment_confidence": 0.8},
            {"sentiment": "negative", "sentiment_confidence": 0.7},
            {"sentiment": "neutral", "sentiment_confidence": 0.6},
            {"sentiment": "positive", "sentiment_confidence": 0.9},
            {"sentiment": "negative", "sentiment_confidence": 0.5},
            {"sentiment": "positive", "sentiment_confidence": 0.85}
        ]
        
        # Ground truth
        self.ground_truth = [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative", 
            "positive"
        ]
        
        # For tests with dictionary ground truth
        self.dict_ground_truth = {"sentiment": self.ground_truth}
        
        # Mock the EvaluationMetrics and ThresholdOptimizer
        self.mock_metrics_calculator = MagicMock(spec=EvaluationMetrics)
        self.mock_threshold_optimizer = MagicMock(spec=ThresholdOptimizer)
        
        # Mock the calculate_f1 method
        self.mock_metrics_calculator.calculate_f1.return_value = 0.85
        
        # Simple parameter grid
        self.param_grid = {
            "threshold_start": [0.2, 0.3],
            "threshold_end": [0.7, 0.8],
            "threshold_step": [0.1]
        }
    
    @patch("src.evaluation.cli.EvaluationMetrics")
    @patch("src.evaluation.cli.ThresholdOptimizer")
    def test_tune_threshold_params(self, mock_threshold_optimizer, mock_evaluation_metrics):
        """Test threshold parameter tuning with grid search and cross-validation."""
        # Set up mocks
        mock_metrics_instance = MagicMock()
        mock_evaluation_metrics.return_value = mock_metrics_instance
        mock_metrics_instance.calculate_f1.return_value = 0.85
        
        mock_optimizer_instance = MagicMock()
        mock_threshold_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.find_optimal_threshold.return_value = 0.5
        
        # Run parameter tuning
        result = tune_threshold_params(
            evaluator=self.mock_evaluator,
            predictions=self.predictions,
            ground_truth=self.dict_ground_truth,
            label_type="sentiment",
            param_grid=self.param_grid,
            cv_folds=2,
            logger=self.mock_logger
        )
        
        # Check that logging happened
        self.mock_logger.info.assert_any_call("Tuning threshold parameters for sentiment")
        
        # Check result structure
        self.assertIn("best_params", result)
        self.assertIn("best_score", result)
        self.assertIn("cv_results", result)
    
    @patch("src.evaluation.cli.EvaluationMetrics")
    @patch("src.evaluation.cli.ThresholdOptimizer")
    def test_tune_threshold_params_with_dict_ground_truth(self, mock_threshold_optimizer, mock_evaluation_metrics):
        """Test threshold tuning with dictionary ground truth."""
        # Set up mocks
        mock_metrics_instance = MagicMock()
        mock_evaluation_metrics.return_value = mock_metrics_instance
        
        mock_optimizer_instance = MagicMock()
        mock_threshold_optimizer.return_value = mock_optimizer_instance
        
        # Run parameter tuning with dict ground truth
        result = tune_threshold_params(
            evaluator=self.mock_evaluator,
            predictions=self.predictions,
            ground_truth=self.dict_ground_truth,
            label_type="sentiment",
            param_grid=self.param_grid,
            cv_folds=2,
            logger=self.mock_logger
        )
        
        # Check that it ran successfully
        self.assertIn("best_params", result)


class ExportToFormatTests(EvaluationCLITestCase):
    """Tests for export_to_format function."""
    
    def test_export_to_csv(self):
        """Test exporting results to CSV format."""
        # Sample results
        results = {
            "models": ["model1", "model2"],
            "best_model": "model1",
            "metrics": {"f1": 0.85, "precision": 0.8}
        }
        
        output_path = os.path.join(self.test_dir, "test_results.csv")
        
        # Export to CSV
        export_to_format(results, output_path, "csv")
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check that file contains data
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("key", content)
            self.assertIn("value", content)
            self.assertIn("model1", content)
    
    def test_export_to_pickle(self):
        """Test exporting results to pickle format."""
        # Sample results
        results = {
            "models": ["model1", "model2"],
            "best_model": "model1",
            "metrics": {"f1": 0.85, "precision": 0.8}
        }
        
        output_path = os.path.join(self.test_dir, "test_results.pkl")
        
        # Export to pickle
        export_to_format(results, output_path, "pickle")
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path))
    
    def test_export_unsupported_format(self):
        """Test exporting with unsupported format raises error."""
        results = {"test": "data"}
        output_path = os.path.join(self.test_dir, "test_results.xyz")
        
        # Should raise ValueError for unsupported format
        with self.assertRaises(ValueError):
            export_to_format(results, output_path, "xyz")


class PerformStatisticalTestsTests(EvaluationCLITestCase):
    """Tests for perform_statistical_tests function."""
    
    def setUp(self):
        super().setUp()
        # Set up sample results
        self.results1 = {"threshold": 0.5, "precision": 0.8, "recall": 0.7, "f1": 0.75}
        self.results2 = {"threshold": 0.6, "precision": 0.75, "recall": 0.8, "f1": 0.77}
        
        # Set up sample predictions
        self.predictions1 = [
            {"sentiment": "positive", "sentiment_confidence": 0.8},
            {"sentiment": "negative", "sentiment_confidence": 0.7},
            {"sentiment": "neutral", "sentiment_confidence": 0.6},
            {"sentiment": "positive", "sentiment_confidence": 0.9}
        ]
        
        self.predictions2 = [
            {"sentiment": "positive", "sentiment_confidence": 0.9},
            {"sentiment": "negative", "sentiment_confidence": 0.8},
            {"sentiment": "positive", "sentiment_confidence": 0.6},  # Different from 1
            {"sentiment": "positive", "sentiment_confidence": 0.7}
        ]
        
        # Set up ground truth
        self.ground_truth = ["positive", "negative", "neutral", "positive"]
    
    def test_perform_statistical_tests_mcnemar(self):
        """Test McNemar's test for statistical significance."""
        # Run test
        result = perform_statistical_tests(
            individual_results={
                "model1": {"evaluation": {"sentiment": self.results1}},
                "model2": {"evaluation": {"sentiment": self.results2}}
            },
            label_types=["sentiment"],
            test_type="mcnemar",
            logger=self.mock_logger
        )
        
        # Check result structure
        self.assertIn("model1 vs model2", result)
        self.assertIn("sentiment", result["model1 vs model2"])
        self.assertEqual(result["model1 vs model2"]["sentiment"]["test_method"], "mcnemar")
    
    def test_perform_statistical_tests_bootstrap(self):
        """Test bootstrap test for statistical significance."""
        # Run test
        result = perform_statistical_tests(
            individual_results={
                "model1": {"evaluation": {"sentiment": self.results1}},
                "model2": {"evaluation": {"sentiment": self.results2}}
            },
            label_types=["sentiment"],
            test_type="bootstrap",
            logger=self.mock_logger
        )
        
        # Check result structure
        self.assertIn("model1 vs model2", result)
        self.assertIn("sentiment", result["model1 vs model2"])
        self.assertEqual(result["model1 vs model2"]["sentiment"]["test_method"], "bootstrap")
    
    def test_perform_statistical_tests_t_test(self):
        """Test t-test for statistical significance."""
        # Run test
        result = perform_statistical_tests(
            individual_results={
                "model1": {"evaluation": {"sentiment": self.results1}},
                "model2": {"evaluation": {"sentiment": self.results2}}
            },
            label_types=["sentiment"],
            test_type="t-test",
            logger=self.mock_logger
        )
        
        # Check result structure
        self.assertIn("model1 vs model2", result)
        self.assertIn("sentiment", result["model1 vs model2"])
        self.assertEqual(result["model1 vs model2"]["sentiment"]["test_method"], "t-test")
    
    def test_perform_statistical_tests_insufficient_models(self):
        """Test that insufficient models returns empty result."""
        # Run test with only one model
        result = perform_statistical_tests(
            individual_results={
                "model1": {"evaluation": {"sentiment": self.results1}}
            },
            label_types=["sentiment"],
            test_type="bootstrap",
            logger=self.mock_logger
        )
        
        # Should return empty dict
        self.assertEqual(result, {})
        self.mock_logger.warning.assert_called_with("Need at least 2 models for statistical tests")


if __name__ == "__main__":
    unittest.main() 
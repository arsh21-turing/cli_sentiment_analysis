"""
Unit tests for evaluation CLI command handlers and validation functions.
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd

from src.evaluation.cli import (
    # Main functions
    _run_evaluate,
    _run_compare,
    _run_tune, 
    _run_cross_validate,
    run_evaluation_cli,
    
    # Validation functions
    _validate_data_args,
    _validate_model_args,
    _validate_output_args,
    _validate_threshold_args,
    _validate_cv_args,
    
    # Subparser configuration functions
    _configure_eval_parser,
    _configure_compare_parser,
    _configure_tune_parser,
    _configure_cv_parser
)


class SimpleMock:
    """A simple mock class that doesn't automatically create attributes."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        # Return None for any attribute that wasn't explicitly set
        return None


class CLIValidationTestCase(unittest.TestCase):
    """Base test case with common setup for validation function tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.test_csv = os.path.join(self.test_dir, "test.csv")
        with open(self.test_csv, "w") as f:
            f.write("text,sentiment\nHello,positive\nBye,negative\n")
            
        self.test_json = os.path.join(self.test_dir, "test.json")
        with open(self.test_json, "w") as f:
            f.write('[{"text":"Hello","sentiment":"positive"},{"text":"Bye","sentiment":"negative"}]')
            
        self.test_settings = os.path.join(self.test_dir, "settings.json")
        with open(self.test_settings, "w") as f:
            f.write('{"model_settings": {"key": "value"}}')
            
        # Set up a mock logger
        self.mock_logger = MagicMock()
            
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)


class ValidateDataArgsTests(CLIValidationTestCase):
    """Tests for _validate_data_args function."""

    def test_valid_data_args(self):
        """Test with valid data arguments."""
        args = SimpleMock(
            data=self.test_csv,
            format="csv",
            text_column="text"
        )
        
        self.assertTrue(_validate_data_args(args, self.mock_logger))
        # No error or warning should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_missing_data_file(self):
        """Test with non-existent data file."""
        args = SimpleMock(
            data=os.path.join(self.test_dir, "nonexistent.csv"),
            format="csv",
            text_column="text"
        )
        
        self.assertFalse(_validate_data_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_mismatched_format(self):
        """Test with format that doesn't match file extension."""
        args = SimpleMock(
            data=self.test_csv,
            format="json",  # Mismatch with .csv
            text_column="text"
        )
        
        # Should pass but with a warning
        self.assertTrue(_validate_data_args(args, self.mock_logger))
        # Warning should be logged
        self.mock_logger.warning.assert_called_once()
        
    def test_missing_text_column(self):
        """Test with missing text column."""
        args = SimpleMock(
            data=self.test_csv,
            format="csv",
            text_column=""  # Empty text column
        )
        
        self.assertFalse(_validate_data_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()


class ValidateModelArgsTests(CLIValidationTestCase):
    """Tests for _validate_model_args function."""
        
    def test_valid_model_args(self):
        """Test with valid model arguments."""
        args = SimpleMock(
            model="transformer",
            settings=None  # Optional
        )
        
        self.assertTrue(_validate_model_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_invalid_model_name(self):
        """Test with invalid model name."""
        args = SimpleMock(
            model="invalid_model",  # Not one of the valid models
            settings=None
        )
        
        self.assertFalse(_validate_model_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_nonexistent_settings_file(self):
        """Test with non-existent settings file."""
        args = SimpleMock(
            model="transformer",
            settings=os.path.join(self.test_dir, "nonexistent_settings.json")
        )
        
        self.assertFalse(_validate_model_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_valid_models_list(self):
        """Test with valid models list for compare command."""
        args = SimpleMock(
            models="transformer,groq,fallback"
        )
        
        self.assertTrue(_validate_model_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_invalid_models_list(self):
        """Test with invalid model in list for compare command."""
        args = SimpleMock(
            models="transformer,invalid_model,fallback"
        )
        
        self.assertFalse(_validate_model_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_mismatched_model_names_count(self):
        """Test with mismatched model names count."""
        args = SimpleMock(
            models="transformer,groq,fallback",
            model_names="Model 1,Model 2"  # Only 2 names for 3 models
        )
        
        self.assertFalse(_validate_model_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_mismatched_settings_files_count(self):
        """Test with mismatched settings files count."""
        args = SimpleMock(
            models="transformer,groq,fallback",
            model_names="Model 1,Model 2,Model 3",
            settings_files="file1.json,file2.json"  # Only 2 files for 3 models
        )
        
        self.assertFalse(_validate_model_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()


class ValidateOutputArgsTests(CLIValidationTestCase):
    """Tests for _validate_output_args function."""
        
    def test_valid_output_args(self):
        """Test with valid output arguments."""
        args = SimpleMock(
            output_dir=self.test_dir,
            log_file=None  # Optional
        )
        
        self.assertTrue(_validate_output_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    @patch('os.makedirs')
    def test_cannot_create_output_dir(self, mock_makedirs):
        """Test when output directory cannot be created."""
        mock_makedirs.side_effect = OSError("Permission denied")
        
        args = SimpleMock(
            output_dir="/invalid/path",
            log_file=None
        )
        
        self.assertFalse(_validate_output_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_valid_log_file(self):
        """Test with valid log file path."""
        args = SimpleMock(
            output_dir=self.test_dir,
            log_file=os.path.join(self.test_dir, "test.log")
        )
        
        self.assertTrue(_validate_output_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    @patch('builtins.open')
    def test_cannot_write_log_file(self, mock_open_func):
        """Test when log file cannot be written."""
        mock_open_func.side_effect = OSError("Permission denied")
        
        args = SimpleMock(
            output_dir=self.test_dir,
            log_file="/invalid/path/test.log"
        )
        
        self.assertFalse(_validate_output_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()


class ValidateThresholdArgsTests(CLIValidationTestCase):
    """Tests for _validate_threshold_args function."""
        
    def test_valid_threshold_args(self):
        """Test with valid threshold arguments."""
        args = SimpleMock(
            threshold_start=0.1,
            threshold_end=0.9,
            threshold_step=0.1
        )
        
        self.assertTrue(_validate_threshold_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_invalid_threshold_range(self):
        """Test with invalid threshold range (start >= end)."""
        args = SimpleMock(
            threshold_start=0.9,
            threshold_end=0.1  # End < start
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_invalid_threshold_step_zero(self):
        """Test with invalid threshold step (zero)."""
        args = SimpleMock(
            threshold_start=0.1,
            threshold_end=0.9,
            threshold_step=0  # Zero step
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_invalid_threshold_step_too_large(self):
        """Test with invalid threshold step (too large)."""
        args = SimpleMock(
            threshold_start=0.1,
            threshold_end=0.9,
            threshold_step=1.0  # Step >= range
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_valid_threshold_lists_for_tune(self):
        """Test with valid threshold lists for tune command."""
        args = SimpleMock(
            threshold_starts="0.1,0.2,0.3",
            threshold_ends="0.7,0.8,0.9",
            threshold_steps="0.01,0.05,0.1"
        )
        
        self.assertTrue(_validate_threshold_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_invalid_threshold_start_value(self):
        """Test with invalid threshold start value (negative)."""
        args = SimpleMock(
            threshold_starts="-0.1,0.2,0.3",  # Negative start
            threshold_ends="0.7,0.8,0.9",
            threshold_steps="0.01,0.05,0.1"
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_invalid_threshold_end_value(self):
        """Test with invalid threshold end value (> 1)."""
        args = SimpleMock(
            threshold_starts="0.1,0.2,0.3",
            threshold_ends="0.7,1.1,0.9",  # End > 1
            threshold_steps="0.01,0.05,0.1"
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_invalid_threshold_step_value(self):
        """Test with invalid threshold step value (negative)."""
        args = SimpleMock(
            threshold_starts="0.1,0.2,0.3",
            threshold_ends="0.7,0.8,0.9",
            threshold_steps="0.01,-0.05,0.1"  # Negative step
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_non_numeric_threshold_values(self):
        """Test with non-numeric threshold values."""
        args = SimpleMock(
            threshold_starts="0.1,0.2,abc",  # Non-numeric value
            threshold_ends="0.7,0.8,0.9",
            threshold_steps="0.01,0.05,0.1"
        )
        
        self.assertFalse(_validate_threshold_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()


class ValidateCVArgsTests(CLIValidationTestCase):
    """Tests for _validate_cv_args function."""
        
    def test_valid_cv_args(self):
        """Test with valid cross-validation arguments."""
        args = SimpleMock(cv=5)
        
        self.assertTrue(_validate_cv_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_negative_cv_folds(self):
        """Test with negative CV folds."""
        args = SimpleMock(cv=-1)
        
        self.assertFalse(_validate_cv_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()
        
    def test_one_cv_fold_warning(self):
        """Test with one CV fold (should warn)."""
        args = SimpleMock(cv=1)
        
        self.assertTrue(_validate_cv_args(args, self.mock_logger))
        # Warning should be logged
        self.mock_logger.warning.assert_called_once()
        
    def test_valid_folds_for_cross_validate(self):
        """Test with valid folds for cross-validate command."""
        args = SimpleMock(folds=5)
        
        self.assertTrue(_validate_cv_args(args, self.mock_logger))
        # No error should be logged
        self.mock_logger.error.assert_not_called()
        
    def test_invalid_folds_for_cross_validate(self):
        """Test with invalid folds for cross-validate command."""
        args = SimpleMock(folds=1)  # Too few for cross-validate command
        
        self.assertFalse(_validate_cv_args(args, self.mock_logger))
        # Error should be logged
        self.mock_logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main() 
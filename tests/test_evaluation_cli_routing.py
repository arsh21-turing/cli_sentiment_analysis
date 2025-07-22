"""
Unit tests for command routing logic in the evaluation CLI.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call

from src.evaluation.cli import (
    run_evaluation_cli,
    _validate_data_args,
    _validate_model_args,
    _validate_output_args,
    _validate_threshold_args,
    _validate_cv_args,
    _run_evaluate,
    _run_compare,
    _run_tune,
    _run_cross_validate
)


class SimpleMock:
    """A simple mock class that doesn't automatically create attributes."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        # Return None for any attribute that wasn't explicitly set
        return None


class CommandRoutingTestCase(unittest.TestCase):
    """Base test case for command routing tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.test_dir)


class CommandRoutingTests(CommandRoutingTestCase):
    """Tests for the command routing logic in the CLI."""
    
    @patch('src.evaluation.cli.argparse.ArgumentParser')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    @patch('src.evaluation.cli._run_evaluate')
    @patch('src.evaluation.cli._run_compare')
    @patch('src.evaluation.cli._run_tune')
    @patch('src.evaluation.cli._run_cross_validate')
    def test_evaluate_command_routing(self, mock_cross_validate, mock_tune, 
                                     mock_compare, mock_evaluate, mock_makedirs, 
                                     mock_setup_logging, mock_arg_parser):
        """Test that evaluate command routes to _run_evaluate."""
        # Mock the parser and arguments
        mock_parser = MagicMock()
        mock_arg_parser.return_value = mock_parser
        
        # Set up mock subparsers
        mock_subparsers = MagicMock()
        mock_parser.add_subparsers.return_value = mock_subparsers
        
        # Create mock parsers for each command
        mock_subparsers.add_parser.side_effect = [MagicMock() for _ in range(4)]
        
        # Set up mock parsed args with evaluate command
        mock_args = SimpleMock(
            command="evaluate",
            log_level="INFO",
            log_file=None,
            quiet=False,
            output_dir=self.test_dir,
            batch_size=0  # Set a proper value instead of MagicMock
        )
        mock_parser.parse_args.return_value = mock_args
        
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run the CLI
        with patch('sys.argv', ['cli.py', 'evaluate']):
            run_evaluation_cli()
            
        # Check that only _run_evaluate was called
        mock_evaluate.assert_called_once()
        mock_compare.assert_not_called()
        mock_tune.assert_not_called()
        mock_cross_validate.assert_not_called()
    
    @patch('src.evaluation.cli.argparse.ArgumentParser')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    @patch('src.evaluation.cli._run_evaluate')
    @patch('src.evaluation.cli._run_compare')
    @patch('src.evaluation.cli._run_tune')
    @patch('src.evaluation.cli._run_cross_validate')
    def test_compare_command_routing(self, mock_cross_validate, mock_tune, 
                                    mock_compare, mock_evaluate, mock_makedirs, 
                                    mock_setup_logging, mock_arg_parser):
        """Test that compare command routes to _run_compare."""
        # Mock the parser and arguments
        mock_parser = MagicMock()
        mock_arg_parser.return_value = mock_parser
        
        # Set up mock subparsers
        mock_subparsers = MagicMock()
        mock_parser.add_subparsers.return_value = mock_subparsers
        
        # Create mock parsers for each command
        mock_subparsers.add_parser.side_effect = [MagicMock() for _ in range(4)]
        
        # Set up mock parsed args with compare command
        mock_args = SimpleMock(
            command="compare",
            log_level="INFO",
            log_file=None,
            quiet=False,
            output_dir=self.test_dir,
            batch_size=0  # Set a proper value instead of MagicMock
        )
        mock_parser.parse_args.return_value = mock_args
        
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run the CLI
        with patch('sys.argv', ['cli.py', 'compare']):
            run_evaluation_cli()
            
        # Check that only _run_compare was called
        mock_evaluate.assert_not_called()
        mock_compare.assert_called_once()
        mock_tune.assert_not_called()
        mock_cross_validate.assert_not_called()
    
    @patch('src.evaluation.cli.argparse.ArgumentParser')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    @patch('src.evaluation.cli._run_evaluate')
    @patch('src.evaluation.cli._run_compare')
    @patch('src.evaluation.cli._run_tune')
    @patch('src.evaluation.cli._run_cross_validate')
    def test_tune_command_routing(self, mock_cross_validate, mock_tune, 
                                 mock_compare, mock_evaluate, mock_makedirs, 
                                 mock_setup_logging, mock_arg_parser):
        """Test that tune command routes to _run_tune."""
        # Mock the parser and arguments
        mock_parser = MagicMock()
        mock_arg_parser.return_value = mock_parser
        
        # Set up mock subparsers
        mock_subparsers = MagicMock()
        mock_parser.add_subparsers.return_value = mock_subparsers
        
        # Create mock parsers for each command
        mock_subparsers.add_parser.side_effect = [MagicMock() for _ in range(4)]
        
        # Set up mock parsed args with tune command
        mock_args = SimpleMock(
            command="tune",
            log_level="INFO",
            log_file=None,
            quiet=False,
            output_dir=self.test_dir,
            batch_size=0  # Set a proper value instead of MagicMock
        )
        mock_parser.parse_args.return_value = mock_args
        
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run the CLI
        with patch('sys.argv', ['cli.py', 'tune']):
            run_evaluation_cli()
            
        # Check that only _run_tune was called
        mock_evaluate.assert_not_called()
        mock_compare.assert_not_called()
        mock_tune.assert_called_once()
        mock_cross_validate.assert_not_called()
    
    @patch('src.evaluation.cli.argparse.ArgumentParser')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    @patch('src.evaluation.cli._run_evaluate')
    @patch('src.evaluation.cli._run_compare')
    @patch('src.evaluation.cli._run_tune')
    @patch('src.evaluation.cli._run_cross_validate')
    def test_cross_validate_command_routing(self, mock_cross_validate, mock_tune, 
                                          mock_compare, mock_evaluate, mock_makedirs, 
                                          mock_setup_logging, mock_arg_parser):
        """Test that cross-validate command routes to _run_cross_validate."""
        # Mock the parser and arguments
        mock_parser = MagicMock()
        mock_arg_parser.return_value = mock_parser
        
        # Set up mock subparsers
        mock_subparsers = MagicMock()
        mock_parser.add_subparsers.return_value = mock_subparsers
        
        # Create mock parsers for each command
        mock_subparsers.add_parser.side_effect = [MagicMock() for _ in range(4)]
        
        # Set up mock parsed args with cross-validate command
        mock_args = SimpleMock(
            command="cross-validate",
            log_level="INFO",
            log_file=None,
            quiet=False,
            output_dir=self.test_dir,
            batch_size=0  # Set a proper value instead of MagicMock
        )
        mock_parser.parse_args.return_value = mock_args
        
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run the CLI
        with patch('sys.argv', ['cli.py', 'cross-validate']):
            run_evaluation_cli()
            
        # Check that only _run_cross_validate was called
        mock_evaluate.assert_not_called()
        mock_compare.assert_not_called()
        mock_tune.assert_not_called()
        mock_cross_validate.assert_called_once()
    
    @patch('src.evaluation.cli.argparse.ArgumentParser')
    @patch('src.evaluation.cli.setup_logging')
    def test_unknown_command_routing(self, mock_setup_logging, mock_arg_parser):
        """Test that unknown commands are handled properly."""
        # Mock the parser and arguments
        mock_parser = MagicMock()
        mock_arg_parser.return_value = mock_parser
        
        # Set up mock subparsers
        mock_subparsers = MagicMock()
        mock_parser.add_subparsers.return_value = mock_subparsers
        
        # Create mock parsers for each command
        mock_subparsers.add_parser.side_effect = [MagicMock() for _ in range(4)]
        
        # Set up mock parsed args with unknown command
        mock_args = SimpleMock(
            command="unknown",
            log_level="INFO",
            log_file=None,
            quiet=False,
            output_dir=self.test_dir
        )
        mock_parser.parse_args.return_value = mock_args
        
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run the CLI and expect it to exit
        with patch('sys.argv', ['cli.py', 'unknown']):
            with patch('sys.exit') as mock_exit:
                run_evaluation_cli()
                mock_exit.assert_called_once_with(1)


class ValidationOrderTests(CommandRoutingTestCase):
    """Tests to verify that validation functions are called in the correct order."""
    
    @patch('src.evaluation.cli._validate_data_args')
    @patch('src.evaluation.cli._validate_model_args')
    @patch('src.evaluation.cli._validate_output_args')
    @patch('src.evaluation.cli._validate_threshold_args')
    @patch('src.evaluation.cli._validate_cv_args')
    def test_validate_order_in_run_evaluate(self, mock_validate_cv, mock_validate_threshold,
                                          mock_validate_output, mock_validate_model, 
                                          mock_validate_data):
        """Test that validation functions are called in the correct order in _run_evaluate."""
        # Set up mocks to return True
        mock_validate_data.return_value = True
        mock_validate_model.return_value = True
        mock_validate_output.return_value = True
        mock_validate_threshold.return_value = True
        mock_validate_cv.return_value = True
        
        # Create mock args with all required attributes
        args = SimpleMock(
            data="test.csv",
            model="transformer",
            output_dir=self.test_dir,
            threshold_start=0.1,
            threshold_end=0.9,
            threshold_step=0.05,
            cv=5,
            label_types="sentiment,emotion",
            sentiment_column="sentiment",
            emotion_column="emotion",
            text_column="text",
            batch_size=0,
            quiet=False,
            log_level="INFO",
            log_file=None
        )
        
        # Create mock logger
        mock_logger = MagicMock()
        
        # Mock data loading to prevent file access
        with patch('src.evaluation.cli.TestDataLoader') as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            
            # Call _run_evaluate
            _run_evaluate(args, self.test_dir, mock_logger)
            
            # Check that validation functions were called in the correct order
            mock_validate_data.assert_called_once()
            mock_validate_model.assert_called_once()
            mock_validate_output.assert_called_once()
            mock_validate_threshold.assert_called_once()
            mock_validate_cv.assert_called_once()
            
            # Check call order
            calls = [
                call(args, mock_logger),
                call(args, mock_logger),
                call(args, mock_logger),
                call(args, mock_logger),
                call(args, mock_logger)
            ]
            mock_validate_data.assert_has_calls([calls[0]])
            mock_validate_model.assert_has_calls([calls[1]])
            mock_validate_output.assert_has_calls([calls[2]])
            mock_validate_threshold.assert_has_calls([calls[3]])
            mock_validate_cv.assert_has_calls([calls[4]])
    
    @patch('src.evaluation.cli._validate_data_args')
    @patch('src.evaluation.cli._validate_model_args')
    @patch('src.evaluation.cli._validate_output_args')
    @patch('src.evaluation.cli._validate_cv_args')
    def test_validate_order_in_run_compare(self, mock_validate_cv,
                                          mock_validate_output, mock_validate_model, 
                                          mock_validate_data):
        """Test that validation functions are called in the correct order in _run_compare."""
        # Set up mocks to return True
        mock_validate_data.return_value = True
        mock_validate_model.return_value = True
        mock_validate_output.return_value = True
        mock_validate_cv.return_value = True
        
        # Create mock args with all required attributes
        args = SimpleMock(
            data="test.csv",
            models="transformer,groq",
            output_dir=self.test_dir,
            cv=5,
            label_types="sentiment,emotion",
            sentiment_column="sentiment",
            emotion_column="emotion",
            text_column="text",
            batch_size=0,
            quiet=False,
            log_level="INFO",
            log_file=None,
            metrics="f1,precision,recall"  # Add missing metrics attribute
        )
        
        # Create mock logger
        mock_logger = MagicMock()
        
        # Mock data loading to prevent file access
        with patch('src.evaluation.cli.TestDataLoader') as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader_class.return_value = mock_loader
            
            # Call _run_compare
            _run_compare(args, self.test_dir, mock_logger)
            
            # Check that validation functions were called in the correct order
            mock_validate_data.assert_called_once()
            mock_validate_model.assert_called_once()
            mock_validate_output.assert_called_once()
            mock_validate_cv.assert_called_once()
            
            # Check call order
            calls = [
                call(args, mock_logger),
                call(args, mock_logger),
                call(args, mock_logger),
                call(args, mock_logger)
            ]
            mock_validate_data.assert_has_calls([calls[0]])
            mock_validate_model.assert_has_calls([calls[1]])
            mock_validate_output.assert_has_calls([calls[2]])
            mock_validate_cv.assert_has_calls([calls[3]])


class ValidationEarlyReturnTests(CommandRoutingTestCase):
    """Tests to verify that validation failures cause early returns."""
    
    def test_data_validation_failure_in_evaluate(self):
        """Test that data validation failure causes early return in _run_evaluate."""
        with patch('src.evaluation.cli._validate_data_args') as mock_validate_data:
            mock_validate_data.return_value = False
            
            args = SimpleMock(
                data="nonexistent.csv",
                output_dir=self.test_dir,
                label_types="sentiment,emotion",
                sentiment_column="sentiment",
                emotion_column="emotion",
                text_column="text",
                batch_size=0,
                quiet=False,
                log_level="INFO",
                log_file=None
            )
            mock_logger = MagicMock()
            
            # Should return early without calling other validations
            with self.assertRaises(ValueError):
                _run_evaluate(args, self.test_dir, mock_logger)
            
            # Only data validation should be called
            mock_validate_data.assert_called_once()
    
    def test_model_validation_failure_in_evaluate(self):
        """Test that model validation failure causes early return in _run_evaluate."""
        with patch('src.evaluation.cli._validate_data_args') as mock_validate_data, \
             patch('src.evaluation.cli._validate_model_args') as mock_validate_model:
            
            mock_validate_data.return_value = True
            mock_validate_model.return_value = False
            
            args = SimpleMock(
                data="test.csv", 
                model="invalid_model",
                output_dir=self.test_dir,
                label_types="sentiment,emotion",
                sentiment_column="sentiment",
                emotion_column="emotion",
                text_column="text",
                batch_size=0,
                quiet=False,
                log_level="INFO",
                log_file=None
            )
            mock_logger = MagicMock()
            
            # Should return early after model validation fails
            with self.assertRaises(ValueError):
                _run_evaluate(args, self.test_dir, mock_logger)
            
            # Both data and model validation should be called
            mock_validate_data.assert_called_once()
            mock_validate_model.assert_called_once()
    
    def test_output_validation_failure_in_evaluate(self):
        """Test that output validation failure causes early return in _run_evaluate."""
        with patch('src.evaluation.cli._validate_data_args') as mock_validate_data, \
             patch('src.evaluation.cli._validate_model_args') as mock_validate_model, \
             patch('src.evaluation.cli._validate_output_args') as mock_validate_output:
            
            mock_validate_data.return_value = True
            mock_validate_model.return_value = True
            mock_validate_output.return_value = False
            
            args = SimpleMock(
                data="test.csv", 
                model="transformer", 
                output_dir="/invalid/path",
                label_types="sentiment,emotion",
                sentiment_column="sentiment",
                emotion_column="emotion",
                text_column="text",
                batch_size=0,
                quiet=False,
                log_level="INFO",
                log_file=None
            )
            mock_logger = MagicMock()
            
            # Should return early after output validation fails
            with self.assertRaises(ValueError):
                _run_evaluate(args, self.test_dir, mock_logger)
            
            # All three validations should be called
            mock_validate_data.assert_called_once()
            mock_validate_model.assert_called_once()
            mock_validate_output.assert_called_once()
    
    def test_threshold_validation_failure_in_evaluate(self):
        """Test that threshold validation failure causes early return in _run_evaluate."""
        with patch('src.evaluation.cli._validate_data_args') as mock_validate_data, \
             patch('src.evaluation.cli._validate_model_args') as mock_validate_model, \
             patch('src.evaluation.cli._validate_output_args') as mock_validate_output, \
             patch('src.evaluation.cli._validate_threshold_args') as mock_validate_threshold:
            
            mock_validate_data.return_value = True
            mock_validate_model.return_value = True
            mock_validate_output.return_value = True
            mock_validate_threshold.return_value = False
            
            args = SimpleMock(
                data="test.csv", 
                model="transformer", 
                output_dir=self.test_dir,
                threshold_start=0.9,
                threshold_end=0.1,  # Invalid: start > end
                label_types="sentiment,emotion",
                sentiment_column="sentiment",
                emotion_column="emotion",
                text_column="text",
                batch_size=0,
                quiet=False,
                log_level="INFO",
                log_file=None
            )
            mock_logger = MagicMock()
            
            # Should return early after threshold validation fails
            with self.assertRaises(ValueError):
                _run_evaluate(args, self.test_dir, mock_logger)
            
            # All four validations should be called
            mock_validate_data.assert_called_once()
            mock_validate_model.assert_called_once()
            mock_validate_output.assert_called_once()
            mock_validate_threshold.assert_called_once()
    
    def test_cv_validation_failure_in_evaluate(self):
        """Test that CV validation failure causes early return in _run_evaluate."""
        with patch('src.evaluation.cli._validate_data_args') as mock_validate_data, \
             patch('src.evaluation.cli._validate_model_args') as mock_validate_model, \
             patch('src.evaluation.cli._validate_output_args') as mock_validate_output, \
             patch('src.evaluation.cli._validate_threshold_args') as mock_validate_threshold, \
             patch('src.evaluation.cli._validate_cv_args') as mock_validate_cv:
            
            mock_validate_data.return_value = True
            mock_validate_model.return_value = True
            mock_validate_output.return_value = True
            mock_validate_threshold.return_value = True
            mock_validate_cv.return_value = False
            
            args = SimpleMock(
                data="test.csv", 
                model="transformer", 
                output_dir=self.test_dir,
                threshold_start=0.1,
                threshold_end=0.9,
                cv=-1,  # Invalid: negative CV
                label_types="sentiment,emotion",
                sentiment_column="sentiment",
                emotion_column="emotion",
                text_column="text",
                batch_size=0,
                quiet=False,
                log_level="INFO",
                log_file=None
            )
            mock_logger = MagicMock()
            
            # Should return early after CV validation fails
            with self.assertRaises(ValueError):
                _run_evaluate(args, self.test_dir, mock_logger)
            
            # All five validations should be called
            mock_validate_data.assert_called_once()
            mock_validate_model.assert_called_once()
            mock_validate_output.assert_called_once()
            mock_validate_threshold.assert_called_once()
            mock_validate_cv.assert_called_once()


class SubparserConfigurationTests(CommandRoutingTestCase):
    """Tests to verify that the subparser configurations are correct."""
    
    def test_evaluate_parser_configuration(self):
        """Test that the evaluate parser is configured with the correct arguments."""
        # Create a mock parser
        mock_parser = MagicMock()
        
        # Configure it with the evaluate options
        from src.evaluation.cli import _configure_eval_parser
        _configure_eval_parser(mock_parser)
        
        # Check that required argument groups were created
        self.assertEqual(mock_parser.add_argument_group.call_count, 7)
        
        # Check that specific key arguments were added
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--data", type=str, required=True, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--model", type=str, default="transformer", 
            choices=["transformer", "groq", "fallback"], help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--threshold-start", type=float, default=0.1, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--output-dir", type=str, default="evaluation_results", help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--batch-size", type=int, default=0, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--cv", type=int, default=0, help=unittest.mock.ANY
        )
    
    def test_compare_parser_configuration(self):
        """Test that the compare parser is configured with the correct arguments."""
        # Create a mock parser
        mock_parser = MagicMock()
        
        # Configure it with the compare options
        from src.evaluation.cli import _configure_compare_parser
        _configure_compare_parser(mock_parser)
        
        # Check that required argument groups were created
        self.assertEqual(mock_parser.add_argument_group.call_count, 7)
        
        # Check that specific key arguments were added
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--data", type=str, required=True, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--models", type=str, required=True, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--significance-test", type=str, default="bootstrap",
            choices=["mcnemar", "bootstrap", "t-test"], help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--output-dir", type=str, default="evaluation_results", help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--batch-size", type=int, default=0, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--cv", type=int, default=0, help=unittest.mock.ANY
        )
    
    def test_tune_parser_configuration(self):
        """Test that the tune parser is configured with the correct arguments."""
        # Create a mock parser
        mock_parser = MagicMock()
        
        # Configure it with the tune options
        from src.evaluation.cli import _configure_tune_parser
        _configure_tune_parser(mock_parser)
        
        # Check that required argument groups were created
        self.assertEqual(mock_parser.add_argument_group.call_count, 7)
        
        # Check that specific key arguments were added
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--data", type=str, required=True, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--model", type=str, default="transformer",
            choices=["transformer", "groq", "fallback"], help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--threshold-starts", type=str, default="0.1,0.2,0.3", help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--threshold-ends", type=str, default="0.7,0.8,0.9", help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--threshold-steps", type=str, default="0.01,0.02,0.05,0.1", help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--cv", type=int, default=5, help=unittest.mock.ANY
        )
    
    def test_cross_validate_parser_configuration(self):
        """Test that the cross-validate parser is configured with the correct arguments."""
        # Create a mock parser
        mock_parser = MagicMock()
        
        # Configure it with the cross-validate options
        from src.evaluation.cli import _configure_cv_parser
        _configure_cv_parser(mock_parser)
        
        # Check that required argument groups were created
        self.assertEqual(mock_parser.add_argument_group.call_count, 6)
        
        # Check that specific key arguments were added
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--data", type=str, required=True, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--model", type=str, default="transformer",
            choices=["transformer", "groq", "fallback"], help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--folds", type=int, default=5, help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--metrics", type=str, default="f1,precision,recall", help=unittest.mock.ANY
        )
        mock_parser.add_argument_group().add_argument.assert_any_call(
            "--optimize-thresholds", action="store_true", help=unittest.mock.ANY
        )


class EndToEndCommandRoutingTests(CommandRoutingTestCase):
    """Tests for end-to-end command routing with actual argument parsing."""
    
    @patch('sys.argv', ['cli.py', 'evaluate', '--data', 'test.csv', '--model', 'transformer'])
    @patch('src.evaluation.cli._run_evaluate')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    def test_end_to_end_evaluate_routing(self, mock_makedirs, mock_setup_logging,
                                        mock_run_evaluate):
        """Test end-to-end routing for evaluate command with actual argument parsing."""
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run CLI
        try:
            run_evaluation_cli()
        except SystemExit:
            # Ignore any exit due to missing files, etc.
            pass
        
        # Check that evaluate command was executed
        self.assertEqual(mock_run_evaluate.call_count, 1)
        
        # Check args passed to _run_evaluate
        args = mock_run_evaluate.call_args[0][0]
        self.assertEqual(args.data, 'test.csv')
        self.assertEqual(args.model, 'transformer')
    
    @patch('sys.argv', ['cli.py', 'compare', '--data', 'test.csv', '--models', 'transformer,fallback'])
    @patch('src.evaluation.cli._run_compare')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    def test_end_to_end_compare_routing(self, mock_makedirs, mock_setup_logging,
                                       mock_run_compare):
        """Test end-to-end routing for compare command with actual argument parsing."""
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run CLI
        try:
            run_evaluation_cli()
        except SystemExit:
            # Ignore any exit due to missing files, etc.
            pass
        
        # Check that compare command was executed
        self.assertEqual(mock_run_compare.call_count, 1)
        
        # Check args passed to _run_compare
        args = mock_run_compare.call_args[0][0]
        self.assertEqual(args.data, 'test.csv')
        self.assertEqual(args.models, 'transformer,fallback')
    
    @patch('sys.argv', ['cli.py', 'tune', '--data', 'test.csv', '--model', 'transformer', 
                        '--threshold-starts', '0.1,0.2', '--threshold-ends', '0.8,0.9'])
    @patch('src.evaluation.cli._run_tune')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    def test_end_to_end_tune_routing(self, mock_makedirs, mock_setup_logging,
                                    mock_run_tune):
        """Test end-to-end routing for tune command with actual argument parsing."""
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run CLI
        try:
            run_evaluation_cli()
        except SystemExit:
            # Ignore any exit due to missing files, etc.
            pass
        
        # Check that tune command was executed
        self.assertEqual(mock_run_tune.call_count, 1)
        
        # Check args passed to _run_tune
        args = mock_run_tune.call_args[0][0]
        self.assertEqual(args.data, 'test.csv')
        self.assertEqual(args.model, 'transformer')
        self.assertEqual(args.threshold_starts, '0.1,0.2')
        self.assertEqual(args.threshold_ends, '0.8,0.9')
    
    @patch('sys.argv', ['cli.py', 'cross-validate', '--data', 'test.csv', '--model', 'transformer', 
                        '--folds', '3', '--metrics', 'f1,precision'])
    @patch('src.evaluation.cli._run_cross_validate')
    @patch('src.evaluation.cli.setup_logging')
    @patch('src.evaluation.cli.os.makedirs')
    def test_end_to_end_cross_validate_routing(self, mock_makedirs, mock_setup_logging,
                                              mock_run_cross_validate):
        """Test end-to-end routing for cross-validate command with actual argument parsing."""
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Run CLI
        try:
            run_evaluation_cli()
        except SystemExit:
            # Ignore any exit due to missing files, etc.
            pass
        
        # Check that cross-validate command was executed
        self.assertEqual(mock_run_cross_validate.call_count, 1)
        
        # Check args passed to _run_cross_validate
        args = mock_run_cross_validate.call_args[0][0]
        self.assertEqual(args.data, 'test.csv')
        self.assertEqual(args.model, 'transformer')
        self.assertEqual(args.folds, 3)
        self.assertEqual(args.metrics, 'f1,precision')
    
    @patch('sys.argv', ['cli.py'])  # No command
    @patch('src.evaluation.cli.argparse.ArgumentParser')
    def test_end_to_end_no_command(self, mock_arg_parser):
        """Test end-to-end behavior when no command is provided."""
        # Create mock parser
        mock_parser = MagicMock()
        mock_arg_parser.return_value = mock_parser
        
        # Mock parsed args with no command
        mock_args = SimpleMock(
            command=None,
            log_level="INFO",
            log_file=None,
            quiet=False,
            output_dir=self.test_dir
        )
        mock_parser.parse_args.return_value = mock_args
        
        # Run CLI and expect it to exit
        with patch('sys.exit') as mock_exit:
            run_evaluation_cli()
            mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main() 
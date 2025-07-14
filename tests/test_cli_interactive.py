# tests/test_cli_interactive.py

import unittest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
import argparse
import sys
import os
import io
from contextlib import redirect_stdout

# Import CLI functionality to test
from src.utils.cli import (
    parse_args,
    load_transformer_model,
    analyze_text,
    run_interactive_mode,
    set_thresholds_interactive,
    compare_models,
    run_comparison_mode,
    load_comparison_models,
    main
)

class TestCLI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for CLI testing."""
        # Create mock model and response
        self.mock_model = MagicMock()
        self.mock_model.name = "Test Model"
        self.mock_model.analyze.return_value = {
            "text": "Test text",
            "sentiment": {"label": "positive", "score": 0.8, "confident": True, "threshold": 0.7},
            "emotion": {"label": "joy", "score": 0.75, "confident": True, "threshold": 0.6}
        }
        self.mock_model.sentiment_threshold = 0.7
        self.mock_model.emotion_threshold = 0.6
        
        # Sample command line arguments
        self.text_args = ["--text", "Test text"]
        self.file_args = ["--file", "test.txt"]
        self.interactive_args = ["--interactive"]
        self.compare_args = ["--compare-interactive", "--compare-models", "model1,model2"]
        self.threshold_args = ["--sentiment-threshold", "0.8", "--emotion-threshold", "0.7"]
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_defaults(self, mock_parse_args):
        """Test argument parsing with default values."""
        # Setup mock return value for parse_args
        mock_args = argparse.Namespace(
            text=None,
            file=None,
            interactive=False,
            compare_interactive=False,
            sentiment_model="nlptown/bert-base-multilingual-uncased-sentiment",
            emotion_model="bhadresh-savani/distilbert-base-uncased-emotion",
            sentiment_threshold=0.7,
            emotion_threshold=0.6,
            local_model_path=None,
            compare_models=None,
            model_name=None,
            show_probabilities=False,
            output=None,
            format="text"
        )
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = parse_args()
        
        # Check the result - should set interactive to True when no input method specified
        self.assertTrue(args.interactive)
        self.assertEqual(args.sentiment_threshold, 0.7)
        self.assertEqual(args.emotion_threshold, 0.6)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_text_input(self, mock_parse_args):
        """Test argument parsing with text input."""
        # Setup mock with text input
        mock_args = argparse.Namespace(
            text="Test text",
            file=None,
            interactive=False,
            compare_interactive=False,
            sentiment_model="default-model",
            emotion_model="default-emotion",
            sentiment_threshold=0.7,
            emotion_threshold=0.6,
            local_model_path=None,
            compare_models=None,
            model_name=None,
            show_probabilities=False,
            output=None,
            format="text"
        )
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = parse_args()
        
        # Check the result
        self.assertEqual(args.text, "Test text")
        self.assertFalse(args.interactive)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_custom_thresholds(self, mock_parse_args):
        """Test argument parsing with custom threshold values."""
        # Setup mock with custom thresholds
        mock_args = argparse.Namespace(
            text="Test text",
            file=None,
            interactive=False,
            compare_interactive=False,
            sentiment_model="default-model",
            emotion_model="default-emotion",
            sentiment_threshold=0.85,
            emotion_threshold=0.55,
            local_model_path=None,
            compare_models=None,
            model_name=None,
            show_probabilities=False,
            output=None,
            format="text"
        )
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = parse_args()
        
        # Check threshold values
        self.assertEqual(args.sentiment_threshold, 0.85)
        self.assertEqual(args.emotion_threshold, 0.55)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_comparison_mode(self, mock_parse_args):
        """Test argument parsing for comparison mode."""
        # Setup mock with comparison mode
        mock_args = argparse.Namespace(
            text=None,
            file=None,
            interactive=False,
            compare_interactive=True,
            sentiment_model="default-model",
            emotion_model="default-emotion",
            sentiment_threshold=0.7,
            emotion_threshold=0.6,
            local_model_path=None,
            compare_models="model1,model2:model3:CustomName",
            model_name=None,
            show_probabilities=False,
            output=None,
            format="text"
        )
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = parse_args()
        
        # Check comparison mode settings
        self.assertTrue(args.compare_interactive)
        self.assertEqual(args.compare_models, "model1,model2:model3:CustomName")
    
    @patch('src.utils.cli.SentimentEmotionTransformer')
    @patch('src.utils.cli.print')
    def test_load_transformer_model(self, mock_print, MockTransformer):
        """Test model loading with different parameters."""
        # Mock the transformer model
        mock_instance = MagicMock()
        MockTransformer.return_value = mock_instance
        
        # Test basic model loading
        result = load_transformer_model(
            "sentiment-model",
            "emotion-model",
            0.7,
            0.6
        )
        
        # Check model was created with correct parameters
        MockTransformer.assert_called_with(
            sentiment_model="sentiment-model",
            emotion_model="emotion-model",
            sentiment_threshold=0.7,
            emotion_threshold=0.6,
            local_model_path=None,
            name=None
        )
        self.assertEqual(result, mock_instance)
        
        # Test with custom name
        MockTransformer.reset_mock()
        result = load_transformer_model(
            "sentiment-model",
            "emotion-model",
            0.7,
            0.6,
            model_name="Custom Model"
        )
        
        MockTransformer.assert_called_with(
            sentiment_model="sentiment-model",
            emotion_model="emotion-model",
            sentiment_threshold=0.7,
            emotion_threshold=0.6,
            local_model_path=None,
            name="Custom Model"
        )
        
        # Test with local model path
        MockTransformer.reset_mock()
        result = load_transformer_model(
            "sentiment-model",
            "emotion-model",
            0.7,
            0.6,
            local_model_path="/models",
            model_name="Local Model"
        )
        
        MockTransformer.assert_called_with(
            sentiment_model="sentiment-model",
            emotion_model="emotion-model",
            sentiment_threshold=0.7,
            emotion_threshold=0.6,
            local_model_path="/models",
            name="Local Model"
        )
        
        # Test error handling
        MockTransformer.side_effect = Exception("Model loading error")
        with self.assertRaises(Exception):
            load_transformer_model(
                "sentiment-model",
                "emotion-model",
                0.7,
                0.6
            )
    
    @patch('src.utils.cli.load_transformer_model')
    def test_load_comparison_models(self, mock_load_model):
        """Test loading multiple models for comparison."""
        # Setup mock models
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model3 = MagicMock()
        mock_load_model.side_effect = [mock_model1, mock_model2, mock_model3]
        
        # Test with basic comparison models string
        args = MagicMock()
        args.sentiment_model = "default-sentiment"
        args.emotion_model = "default-emotion"
        args.sentiment_threshold = 0.7
        args.emotion_threshold = 0.6
        args.local_model_path = None
        args.model_name = "Default"
        args.compare_models = "model1,model2:model3"
        
        result = load_comparison_models(args)
        
        # Should load 3 models (default + 2 specified)
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_load_model.call_count, 3)
        
        # Check parameters for model1 and model2:model3
        calls = mock_load_model.call_args_list
        self.assertEqual(calls[1][0][0], "model1")  # First comparison model
        self.assertEqual(calls[1][0][1], "default-emotion")  # Default emotion model
        
        self.assertEqual(calls[2][0][0], "model2")  # Second comparison model sentiment
        self.assertEqual(calls[2][0][1], "model3")  # Second comparison model emotion
        
        # Test with custom thresholds
        mock_load_model.reset_mock()
        mock_load_model.side_effect = [mock_model1, mock_model2]
        
        args.compare_models = "model1:model2:Custom:0.9:0.8"
        
        result = load_comparison_models(args)
        
        # Check custom thresholds were passed
        calls = mock_load_model.call_args_list
        self.assertEqual(calls[1][0][0], "model1")  # Sentiment model
        self.assertEqual(calls[1][0][1], "model2")  # Emotion model
        self.assertEqual(calls[1][0][2], 0.9)  # Sentiment threshold
        self.assertEqual(calls[1][0][3], 0.8)  # Emotion threshold
        self.assertEqual(calls[1][1]["name"], "Custom")  # Custom name
        
        # Test handling of errors
        mock_load_model.reset_mock()
        mock_load_model.side_effect = [mock_model1, Exception("Error"), mock_model3]
        
        args.compare_models = "model1,model2,model3"
        
        # Should continue despite error with one model
        result = load_comparison_models(args)
        self.assertEqual(len(result), 2)  # Only 2 models succeed
    
    def test_analyze_text(self):
        """Test text analysis with mock model."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.analyze.return_value = {
            "text": "Test text",
            "sentiment": {"label": "positive", "score": 0.8},
            "emotion": {"label": "joy", "score": 0.75}
        }
        
        # Test basic analysis
        result, formatted = analyze_text("Test text", mock_model)
        
        # Check model was called
        mock_model.analyze.assert_called_with("Test text")
        
        # Check result contains model output
        self.assertEqual(result["text"], "Test text")
        self.assertEqual(result["sentiment"]["label"], "positive")
        
        # Check formatted output is a string
        self.assertIsInstance(formatted, str)
        
        # Test with probabilities
        result, formatted = analyze_text("Test text", mock_model, show_probabilities=True)
        
        # Check formatted output is still a string with probabilities
        self.assertIsInstance(formatted, str)
    
    @patch('src.utils.cli.ModelComparison')
    def test_compare_models(self, MockComparison):
        """Test comparing multiple models on the same text."""
        # Setup mock models
        mock_model1 = MagicMock(name="Model 1")
        mock_model1.analyze.return_value = {
            "text": "Test text",
            "model": "Model 1",
            "sentiment": {"label": "positive", "score": 0.8}
        }
        
        mock_model2 = MagicMock(name="Model 2")
        mock_model2.analyze.return_value = {
            "text": "Test text",
            "model": "Model 2",
            "sentiment": {"label": "neutral", "score": 0.6}
        }
        
        # Setup mock comparison
        mock_comparison = MagicMock()
        MockComparison.return_value = mock_comparison
        
        mock_comparison.compare.return_value = {
            "text": "Test text",
            "sentiment_agreement": 0.5,
            "emotion_agreement": 0.5,
            "results": [
                mock_model1.analyze.return_value,
                mock_model2.analyze.return_value
            ]
        }
        mock_comparison.format_comparison.return_value = "Formatted comparison"
        
        # Test comparison
        models = [mock_model1, mock_model2]
        result, formatted = compare_models("Test text", models)
        
        # Check ModelComparison was created with models
        MockComparison.assert_called_with(models)
        
        # Check comparison was called
        mock_comparison.compare.assert_called_with("Test text")
        
        # Check format_comparison was called
        mock_comparison.format_comparison.assert_called_with(
            mock_comparison.compare.return_value, False
        )
        
        # Check result contains expected data
        self.assertIn("sentiment_agreement", result)
        self.assertEqual(result["sentiment_agreement"], 0.5)
        
        # Check formatted output
        self.assertEqual(formatted, "Formatted comparison")
        
        # Test error with empty model list
        with self.assertRaises(ValueError):
            compare_models("Test text", [])
    
    @patch('builtins.input')
    @patch('src.utils.cli.os')
    @patch('src.utils.cli.export_results')
    def test_run_interactive_mode(self, mock_export, mock_os, mock_input):
        """Test interactive mode handling of commands."""
        # Setup mock inputs sequence - just test basic functionality
        mock_input.side_effect = [
            "This is a test text",  # Regular text analysis
            ":quit"                  # Quit immediately
        ]
        
        # Patch readline to avoid file operations
        with patch('src.utils.cli.readline'):
            # Create output buffer to capture prints
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                # Run interactive mode with mock model
                run_interactive_mode(self.mock_model, show_probabilities=False)
            
            output = buffer.getvalue()
            
            # Check welcome message was displayed
            self.assertIn("Interactive Sentiment & Emotion Analysis", output)
            
            # Check model was called for text analysis
            self.mock_model.analyze.assert_called_with("This is a test text")
    
    @patch('builtins.input')
    def test_set_thresholds_interactive(self, mock_input):
        """Test interactive threshold setting."""
        # Setup mock inputs for threshold setting
        mock_input.side_effect = [
            "0.9",  # New sentiment threshold
            "0.65"  # New emotion threshold
        ]
        
        # Create output buffer to capture prints
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            set_thresholds_interactive(self.mock_model)
        
        output = buffer.getvalue()
        
        # Check current thresholds were shown
        self.assertIn("Current thresholds", output)
        self.assertIn("0.70", output)  # Original sentiment threshold
        self.assertIn("0.60", output)  # Original emotion threshold
        
        # Check model thresholds were updated
        self.mock_model.set_thresholds.assert_called_with(0.9, 0.65)
        
        # Check updated thresholds were shown
        self.assertIn("Thresholds Updated", output)
        
        # Test keeping current threshold (empty input)
        mock_input.reset_mock()
        self.mock_model.reset_mock()
        
        mock_input.side_effect = [
            "",     # Keep current sentiment threshold
            ""      # Keep current emotion threshold
        ]
        
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            set_thresholds_interactive(self.mock_model)
        
        # Should keep current thresholds
        self.mock_model.set_thresholds.assert_called_with(
            self.mock_model.sentiment_threshold, 
            self.mock_model.emotion_threshold
        )
        
        # Test invalid input handling
        mock_input.reset_mock()
        self.mock_model.reset_mock()
        
        mock_input.side_effect = [
            "invalid",  # Invalid sentiment input
            "0.8",      # Valid retry
            "2.0",      # Out of range
            "0.7"       # Valid retry
        ]
        
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            set_thresholds_interactive(self.mock_model)
        
        output = buffer.getvalue()
        
        # Check error messages
        self.assertIn("Please enter a valid number", output)
        self.assertIn("Threshold must be between 0.0 and 1.0", output)
        
        # Should set thresholds with valid retries
        self.mock_model.set_thresholds.assert_called_with(0.8, 0.7)
    
    @patch('builtins.input')
    def test_run_comparison_mode(self, mock_input):
        """Test interactive comparison mode features."""
        # Setup mock models for comparison
        mock_model1 = MagicMock(name="Model 1")
        mock_model1.name = "Model 1"
        mock_model1.analyze.return_value = {
            "text": "Test text",
            "model": "Model 1",
            "sentiment": {"label": "positive", "score": 0.8}
        }
        mock_model1.get_model_info.return_value = {
            "name": "Model 1", 
            "sentiment_model": "model1", 
            "emotion_model": "emotion1",
            "sentiment_threshold": 0.7,
            "emotion_threshold": 0.6,
            "device": "cpu"
        }
        
        mock_model2 = MagicMock(name="Model 2")
        mock_model2.name = "Model 2"
        mock_model2.analyze.return_value = {
            "text": "Test text",
            "model": "Model 2",
            "sentiment": {"label": "neutral", "score": 0.6}
        }
        mock_model2.get_model_info.return_value = {
            "name": "Model 2", 
            "sentiment_model": "model2", 
            "emotion_model": "emotion2",
            "sentiment_threshold": 0.8,
            "emotion_threshold": 0.7,
            "device": "cpu"
        }
        
        # Setup inputs sequence - simplified test
        mock_input.side_effect = [
            "This is a test comparison",  # Regular text comparison
            ":quit"                       # Quit immediately
        ]
        
        # Patch objects we need
        with patch('src.utils.cli.ModelComparison') as MockComparison, \
             patch('src.utils.cli.readline'):
                
            # Setup mock comparison
            mock_comparison = MagicMock()
            MockComparison.return_value = mock_comparison
            mock_comparison.compare.return_value = {
                "text": "This is a test comparison",
                "sentiment_agreement": 0.5,
                "emotion_agreement": 0.5,
                "results": [
                    mock_model1.analyze.return_value,
                    mock_model2.analyze.return_value
                ]
            }
            mock_comparison.format_comparison.return_value = "Formatted comparison"
            
            models = [mock_model1, mock_model2]
            
            # Create output buffer to capture prints
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_comparison_mode(models, show_probabilities=False)
            
            output = buffer.getvalue()
            
            # Check welcome message
            self.assertIn("Interactive Model Comparison", output)
            self.assertIn("Comparing 2 models", output)
            
            # Check comparison was performed
            mock_comparison.compare.assert_called_with("This is a test comparison")
            
            # Test with too few models
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_comparison_mode([mock_model1], show_probabilities=False)
            
            output = buffer.getvalue()
            self.assertIn("requires at least 2 models", output)
    
    @patch('src.utils.cli.parse_args')
    @patch('src.utils.cli.load_transformer_model')
    @patch('src.utils.cli.analyze_text')
    @patch('sys.exit')
    def test_main_function_single_text(self, mock_exit, mock_analyze, mock_load_model, mock_parse_args):
        """Test main function with single text analysis."""
        # Setup mock args with text analysis
        args = MagicMock()
        args.text = "Test text"
        args.file = None
        args.interactive = False
        args.compare_interactive = False
        args.sentiment_model = "model1"
        args.emotion_model = "emotion1"
        args.sentiment_threshold = 0.7
        args.emotion_threshold = 0.6
        args.local_model_path = None
        args.compare_models = None
        args.model_name = None
        args.show_probabilities = False
        args.output = None
        args.format = "text"
        
        mock_parse_args.return_value = args
        
        # Mock model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Mock analysis result
        mock_analyze.return_value = (
            {"text": "Test text"},  # Result dict
            "Formatted output"       # Formatted string
        )
        
        # Create output buffer to capture prints
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main()
        
        output = buffer.getvalue()
        
        # Check model was loaded
        mock_load_model.assert_called_with(
            "model1", "emotion1", 0.7, 0.6, None, None
        )
        
        # Check text was analyzed
        mock_analyze.assert_called_with("Test text", mock_model, False)
        
        # Check formatted output was printed
        self.assertIn("Formatted output", output)
        
        # Test error handling
        mock_load_model.side_effect = Exception("Error loading model")
        
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main()
        
        output = buffer.getvalue()
        self.assertIn("Error: Error loading model", output)
        mock_exit.assert_called_with(1)
    
    @patch('src.utils.cli.parse_args')
    @patch('src.utils.cli.load_comparison_models')
    @patch('src.utils.cli.run_comparison_mode')
    def test_main_function_comparison(self, mock_run_comparison, mock_load_models, mock_parse_args):
        """Test main function with model comparison."""
        # Setup mock args with comparison mode
        args = MagicMock()
        args.text = None
        args.file = None
        args.interactive = False
        args.compare_interactive = True
        args.sentiment_model = "model1"
        args.emotion_model = "emotion1"
        args.sentiment_threshold = 0.7
        args.emotion_threshold = 0.6
        args.local_model_path = None
        args.compare_models = "model1,model2"
        args.model_name = None
        args.show_probabilities = False
        args.output = None
        args.format = "text"
        
        mock_parse_args.return_value = args
        
        # Mock models loading
        mock_models = [MagicMock(), MagicMock()]
        mock_load_models.return_value = mock_models
        
        # Create output buffer to capture prints
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main()
        
        # Check models were loaded
        mock_load_models.assert_called_with(args)
        
        # Check comparison mode was run
        mock_run_comparison.assert_called_with(mock_models, False)

if __name__ == "__main__":
    unittest.main() 
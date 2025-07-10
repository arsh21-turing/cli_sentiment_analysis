"""
Integration tests for the CLI's single-sentence analysis functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import subprocess
import tempfile
import json
import re
from io import StringIO

# Path manipulation to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import SentimentEmotionTransformer
import src.main as main_module

class TestCLISingleSentenceAnalysis(unittest.TestCase):
    """Test cases for the CLI's single-sentence analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock transformer for testing
        self.mock_transformer = MagicMock(spec=SentimentEmotionTransformer)
        
        # Mock analyze method to return predictable results
        def mock_analyze(text):
            if "positive" in text.lower():
                return {
                    "sentiment": "positive",
                    "sentiment_score": 0.85,
                    "emotion": None,
                    "emotion_score": None,
                    "text": text
                }
            elif "negative" in text.lower():
                return {
                    "sentiment": "negative",
                    "sentiment_score": 0.78,
                    "emotion": "sadness",
                    "emotion_score": 0.67,
                    "text": text
                }
            else:
                return {
                    "sentiment": "neutral",
                    "sentiment_score": 0.62,
                    "emotion": None,
                    "emotion_score": None,
                    "text": text
                }
        
        self.mock_transformer.analyze.side_effect = mock_analyze
        
        # Create a patcher for the SentimentEmotionTransformer class
        self.transformer_patcher = patch(
            'src.main.SentimentEmotionTransformer',
            return_value=self.mock_transformer
        )
        self.mock_transformer_class = self.transformer_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.transformer_patcher.stop()

    def test_cli_text_argument(self):
        """Test that the CLI correctly handles the --text argument."""
        # Capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Call main with arguments
            sys.argv = ['main.py', '--text', 'This is a test sentence']
            main_module.main()
            
            # Get captured output
            output = fake_stdout.getvalue()
            
            # Verify that the transformer was initialized
            self.mock_transformer_class.assert_called_once()
            
            # Verify that the analyze method was called with the correct text
            self.mock_transformer.analyze.assert_called_with('This is a test sentence')
            
            # Check that the output contains expected text
            self.assertIn('Analyzing: "This is a test sentence"', output)
            self.assertIn('Sentiment: NEUTRAL', output)
            
    def test_positive_sentiment_output(self):
        """Test output formatting for positive sentiment."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            sys.argv = ['main.py', '--text', 'This is a positive sentence']
            main_module.main()
            
            output = fake_stdout.getvalue()
            
            # Verify sentiment output
            self.assertIn('Sentiment: POSITIVE', output)
            self.assertIn('Score: 0.85', output)
            
            # Verify emotion is not present (as it's None for positive sentiment)
            self.assertNotIn('Emotion:', output)
            
    def test_negative_sentiment_with_emotion_output(self):
        """Test output formatting for negative sentiment with emotion."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            sys.argv = ['main.py', '--text', 'This is a negative sentence']
            main_module.main()
            
            output = fake_stdout.getvalue()
            
            # Verify sentiment output
            self.assertIn('Sentiment: NEGATIVE', output)
            self.assertIn('Score: 0.78', output)
            
            # Verify emotion output
            self.assertIn('Emotion: SADNESS', output)
            self.assertIn('Score: 0.67', output)

    def test_threshold_arguments(self):
        """Test that threshold arguments are correctly passed to the transformer."""
        with patch('sys.stdout', new=StringIO()):
            sys.argv = ['main.py', 
                       '--text', 'This is a test sentence',
                       '--sentiment-threshold', '0.75',
                       '--emotion-threshold', '0.6']
            main_module.main()
            
            # Verify thresholds were correctly passed to the constructor
            self.mock_transformer_class.assert_called_with(
                sentiment_model_name=None,
                emotion_model_name=None,
                sentiment_threshold=0.75,
                emotion_threshold=0.6
            )
            
    def test_model_arguments(self):
        """Test that model name arguments are correctly passed to the transformer."""
        with patch('sys.stdout', new=StringIO()):
            sys.argv = ['main.py', 
                       '--text', 'This is a test sentence',
                       '--sentiment-model', 'test-sentiment-model',
                       '--emotion-model', 'test-emotion-model']
            main_module.main()
            
            # Verify model names were correctly passed to the constructor
            self.mock_transformer_class.assert_called_with(
                sentiment_model_name='test-sentiment-model',
                emotion_model_name='test-emotion-model',
                sentiment_threshold=0.6,  # Default value
                emotion_threshold=0.5     # Default value
            )
            
    @patch('colorama.init')
    def test_colorama_initialization(self, mock_colorama_init):
        """Test that colorama is initialized."""
        with patch('sys.stdout', new=StringIO()):
            sys.argv = ['main.py', '--text', 'This is a test sentence']
            main_module.main()
            
            # Verify colorama.init() was called
            mock_colorama_init.assert_called_once()
            
    def test_progress_indicators(self):
        """Test that progress indicators are shown during model loading."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            sys.argv = ['main.py', '--text', 'This is a test sentence']
            
            # Mock the sys.stdout.write to capture progress indicators
            original_stdout_write = sys.stdout.write
            write_calls = []
            
            def mock_stdout_write(text):
                write_calls.append(text)
                return original_stdout_write(text)
                
            sys.stdout.write = mock_stdout_write
            
            try:
                main_module.main()
                
                # Verify progress indicators were shown
                progress_texts = [call for call in write_calls if "Loading models" in call]
                self.assertTrue(len(progress_texts) > 0, "No progress indicator shown")
                
            finally:
                # Restore stdout.write
                sys.stdout.write = original_stdout_write
                
    def test_help_message(self):
        """Test that help message is shown when no text or file is provided."""
        with patch('argparse.ArgumentParser.print_help') as mock_print_help:
            # No arguments provided should show help
            sys.argv = ['main.py']
            
            # This will raise SystemExit because argparse will exit
            # when required arguments are missing
            with self.assertRaises(SystemExit):
                main_module.main()

    @unittest.skipIf('CI' in os.environ, "Skip subprocess tests in CI environment")
    def test_cli_process_execution(self):
        """Test that the CLI can be executed as a subprocess."""
        # Create a temporary script to run
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
            temp_file.write("""
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

# Mock the actual transformer
from src.models.transformer import SentimentEmotionTransformer
SentimentEmotionTransformer = MagicMock()
SentimentEmotionTransformer.return_value.analyze.return_value = {
    "sentiment": "positive",
    "sentiment_score": 0.85,
    "emotion": None,
    "emotion_score": None,
    "text": "test"
}

# Run the CLI
import src.main
sys.argv = ['main.py', '--text', 'test']
src.main.main()
print("CLI execution successful")
            """)
            script_path = temp_file.name
            
        try:
            # Run the script as a subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True
            )
            
            # Check that the process executed successfully
            self.assertEqual(result.returncode, 0, 
                            f"Process failed with: {result.stderr}")
            self.assertIn("CLI execution successful", result.stdout)
            
        finally:
            # Clean up the temporary script
            os.unlink(script_path)


# Command-line execution
if __name__ == "__main__":
    unittest.main() 
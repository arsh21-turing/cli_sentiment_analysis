"""
Integration tests for the CLI's file processing functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import json
import csv
import re
from io import StringIO
from pathlib import Path

# Path manipulation to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import SentimentEmotionTransformer
import src.main as main_module
from tqdm import tqdm

class TestCLIFileProcessing(unittest.TestCase):
    """Test cases for the CLI's file processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock transformer for testing
        self.mock_transformer = MagicMock(spec=SentimentEmotionTransformer)
        
        # Configure mock analyze method
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
            elif "angry" in text.lower() or "anger" in text.lower():
                return {
                    "sentiment": "negative",
                    "sentiment_score": 0.82,
                    "emotion": "anger",
                    "emotion_score": 0.74,
                    "text": text
                }
            elif "fear" in text.lower() or "afraid" in text.lower():
                return {
                    "sentiment": "negative",
                    "sentiment_score": 0.75,
                    "emotion": "fear",
                    "emotion_score": 0.70,
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
        
        # Create a patcher for tqdm to avoid progress bar in tests
        self.tqdm_patcher = patch('src.main.tqdm', wraps=tqdm)
        self.mock_tqdm = self.tqdm_patcher.start()
        
        # Create a test file with mixed sentiments
        self.test_file = self._create_test_file()

    def tearDown(self):
        """Tear down test fixtures."""
        self.transformer_patcher.stop()
        self.tqdm_patcher.stop()
        
        # Remove test file
        if hasattr(self, 'test_file') and os.path.exists(self.test_file):
            os.unlink(self.test_file)
            
        # Remove any generated output files
        base_name = os.path.splitext(os.path.basename(self.test_file))[0]
        for ext in ['_analysis.csv', '_analysis.json']:
            output_file = f"{base_name}{ext}"
            if os.path.exists(output_file):
                os.unlink(output_file)

    def _create_test_file(self):
        """Create a test file with various sentences."""
        test_lines = [
            "This is a positive statement about the product.",
            "I am very satisfied with the service quality.",
            "This is a completely neutral observation.",
            "A factual statement without sentiment.",
            "I am disappointed with the outcome, it's negative.",
            "The product broke after one day, very negative experience.",
            "I'm angry about the poor customer service!",
            "I'm afraid this solution won't work for our needs.",
            "",  # Empty line to test handling
            "Mixed feelings but mostly positive experience.",
            "Another neutral statement to analyze."
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write('\n'.join(test_lines))
            return temp_file.name

    def test_file_argument_processing(self):
        """Test that the CLI correctly processes the --file argument."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Call main with file argument
            sys.argv = ['main.py', '--file', self.test_file]
            main_module.main()
            
            output = fake_stdout.getvalue()
            
            # Verify that the transformer was initialized
            self.mock_transformer_class.assert_called_once()
            
            # Verify file processing started
            self.assertIn(f"Processing", output)
            self.assertIn(os.path.basename(self.test_file), output)
            
            # Verify progress bar was used
            self.mock_tqdm.assert_called_once()
            
            # Verify the summary was generated
            self.assertIn("ANALYSIS SUMMARY", output)
            self.assertIn("Sentiment Distribution:", output)
            
            # Check that the analyze method was called for each line in the file
            # (except empty line)
            self.assertEqual(self.mock_transformer.analyze.call_count, 10)

    def test_batch_size_argument(self):
        """Test that batch size argument is respected."""
        with patch('sys.stdout', new=StringIO()):
            # Call main with custom batch size
            sys.argv = ['main.py', '--file', self.test_file, '--batch-size', '2']
            main_module.main()
            
            # Verify tqdm was configured with the right total (total lines, not batches)
            total_lines = sum(1 for line in open(self.test_file) if line.strip())
            self.mock_tqdm.assert_called_once()
            self.assertEqual(self.mock_tqdm.call_args[1]['total'], total_lines)
            
            # Check for correct number of analyze calls
            self.assertEqual(self.mock_transformer.analyze.call_count, 10)
            
    def test_csv_output_format(self):
        """Test that CSV output format works correctly."""
        with patch('sys.stdout', new=StringIO()):
            # Get the base name for checking output file
            base_name = os.path.splitext(os.path.basename(self.test_file))[0]
            csv_output = f"{base_name}_analysis.csv"
            
            # Call main with CSV output format
            sys.argv = ['main.py', '--file', self.test_file, '--output-format', 'csv']
            main_module.main()
            
            # Verify CSV file was created
            self.assertTrue(os.path.exists(csv_output), f"CSV output file {csv_output} not created")
            
            # Check CSV content
            with open(csv_output, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
                # Verify we have the right number of rows
                self.assertEqual(len(rows), 10, "CSV should have 10 rows (excluding empty line)")
                
                # Check column headers
                self.assertTrue(all(field in reader.fieldnames 
                              for field in ['text', 'sentiment', 'sentiment_score', 'emotion', 'emotion_score']))
                
                # Check some values
                positive_rows = [row for row in rows if row['sentiment'] == 'positive']
                negative_rows = [row for row in rows if row['sentiment'] == 'negative']
                neutral_rows = [row for row in rows if row['sentiment'] == 'neutral']
                
                self.assertTrue(len(positive_rows) > 0, "No positive sentiments found in CSV")
                self.assertTrue(len(negative_rows) > 0, "No negative sentiments found in CSV")
                self.assertTrue(len(neutral_rows) > 0, "No neutral sentiments found in CSV")
                
                # Check that emotions are present for negative sentiments
                for row in negative_rows:
                    self.assertTrue(row['emotion'], 
                                  f"Negative sentiment row should have emotion: {row}")

    def test_json_output_format(self):
        """Test that JSON output format works correctly."""
        with patch('sys.stdout', new=StringIO()):
            # Get the base name for checking output file
            base_name = os.path.splitext(os.path.basename(self.test_file))[0]
            json_output = f"{base_name}_analysis.json"
            
            # Call main with JSON output format
            sys.argv = ['main.py', '--file', self.test_file, '--output-format', 'json']
            main_module.main()
            
            # Verify JSON file was created
            self.assertTrue(os.path.exists(json_output), f"JSON output file {json_output} not created")
            
            # Check JSON content
            with open(json_output, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                
                # Verify we have the right number of entries
                self.assertEqual(len(data), 10, "JSON should have 10 entries (excluding empty line)")
                
                # Check that each entry has the required fields
                for entry in data:
                    self.assertTrue('sentiment' in entry, "Entry missing sentiment field")
                    self.assertTrue('sentiment_score' in entry, "Entry missing sentiment_score field")
                    self.assertTrue('emotion' in entry, "Entry missing emotion field")
                    self.assertTrue('emotion_score' in entry, "Entry missing emotion_score field")
                    self.assertTrue('display_text' in entry, "Entry missing display_text field")
                
                # Check sentiment distribution
                sentiments = [entry['sentiment'] for entry in data]
                self.assertTrue('positive' in sentiments, "No positive sentiments found in JSON")
                self.assertTrue('negative' in sentiments, "No negative sentiments found in JSON")
                self.assertTrue('neutral' in sentiments, "No neutral sentiments found in JSON")

    def test_summary_statistics(self):
        """Test that summary statistics are generated correctly."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Call main with file argument
            sys.argv = ['main.py', '--file', self.test_file]
            main_module.main()
            
            output = fake_stdout.getvalue()
            
            # Check for summary section
            self.assertIn("ANALYSIS SUMMARY", output)
            
            # Check for sentiment distribution
            self.assertIn("Sentiment Distribution:", output)
            self.assertIn("POSITIVE", output)
            self.assertIn("NEGATIVE", output)
            self.assertIn("NEUTRAL", output)
            
            # Check for emotion distribution (should be present since we have negative sentiments)
            self.assertIn("Emotion Distribution:", output)
            
            # Verify average scores are reported
            self.assertIn("Average sentiment score", output)
            
            # Check for percentage reporting (values are represented as percentages)
            percentage_pattern = r"\d+\.\d+%"
            self.assertTrue(re.search(percentage_pattern, output), 
                          "Percentages should be shown in the summary")
            
            # Check for bar chart visualization (using █ character)
            self.assertIn("█", output, "Bar chart visualization missing")

    def test_error_handling_for_nonexistent_file(self):
        """Test error handling when the input file doesn't exist."""
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            # Call main with non-existent file
            sys.argv = ['main.py', '--file', 'nonexistent_file.txt']
            main_module.main()
            
            output = fake_stdout.getvalue()
            
            # Check for error message
            self.assertIn("Error: File not found", output)
            
            # Verify that the transformer was still initialized but analyze wasn't called
            self.mock_transformer_class.assert_called_once()
            self.mock_transformer.analyze.assert_not_called()

    def test_error_handling_for_empty_file(self):
        """Test error handling when the input file is empty."""
        # Create an empty file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as empty_file:
            empty_file_path = empty_file.name
        
        try:
            with patch('sys.stdout', new=StringIO()) as fake_stdout:
                # Call main with empty file
                sys.argv = ['main.py', '--file', empty_file_path]
                main_module.main()
                
                output = fake_stdout.getvalue()
                
                # Check for error message
                self.assertIn("Error: File is empty", output)
                
                # Verify that the transformer was initialized but analyze wasn't called
                self.mock_transformer_class.assert_called_once()
                self.mock_transformer.analyze.assert_not_called()
        finally:
            # Clean up the empty file
            if os.path.exists(empty_file_path):
                os.unlink(empty_file_path)

    def test_model_and_threshold_with_file_processing(self):
        """Test that model and threshold arguments work with file processing."""
        with patch('sys.stdout', new=StringIO()):
            # Call main with file and custom parameters
            sys.argv = [
                'main.py',
                '--file', self.test_file,
                '--sentiment-model', 'custom-sentiment-model',
                '--emotion-model', 'custom-emotion-model',
                '--sentiment-threshold', '0.7',
                '--emotion-threshold', '0.6'
            ]
            main_module.main()
            
            # Verify transformer was initialized with custom parameters
            self.mock_transformer_class.assert_called_with(
                sentiment_model_name='custom-sentiment-model',
                emotion_model_name='custom-emotion-model',
                sentiment_threshold=0.7,
                emotion_threshold=0.6
            )

    def test_tqdm_configuration(self):
        """Test that tqdm is configured correctly for progress display."""
        with patch('sys.stdout', new=StringIO()):
            # Call main with file argument
            sys.argv = ['main.py', '--file', self.test_file]
            main_module.main()
            
            # Verify tqdm was configured correctly
            self.mock_tqdm.assert_called_once()
            
            # Check tqdm parameters
            tqdm_kwargs = self.mock_tqdm.call_args[1]
            self.assertEqual(tqdm_kwargs['desc'], "Analyzing")
            self.assertEqual(tqdm_kwargs['unit'], "lines")
            
            # Check total is set to the number of non-empty lines
            total_lines = sum(1 for line in open(self.test_file) if line.strip())
            self.assertEqual(tqdm_kwargs['total'], total_lines)


# Command-line execution
if __name__ == "__main__":
    unittest.main() 
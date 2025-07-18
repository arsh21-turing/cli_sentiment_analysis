# tests/test_output_formatting.py

import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import sys
import os
import json
import csv
from io import StringIO
import re

# Import colorama for testing color stripping
from colorama import Fore, Style, init

# Import output formatting functions
from src.utils.output import (
    format_probabilities,
    format_sentiment_result,
    format_emotion_result,
    format_analysis_result,
    create_progress_bar,
    flatten_result_for_csv,
    export_to_json,
    export_to_csv
)

class TestOutputFormatting(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for use in test cases."""
        # Sample probability distributions
        self.sample_probabilities = {
            "positive": 0.82,
            "neutral": 0.15,
            "negative": 0.03
        }
        
        # Sample sentiment results with confidence indicators and thresholds
        self.sample_sentiment_result = {
            "label": "positive",
            "score": 0.82,
            "confident": True,
            "threshold": 0.7,
            "raw_probabilities": self.sample_probabilities
        }
        
        # Sample emotion results
        self.sample_emotion_result = {
            "label": "joy",
            "score": 0.65,
            "confident": True,
            "threshold": 0.6,
            "raw_probabilities": {
                "joy": 0.65,
                "sadness": 0.1,
                "anger": 0.05,
                "fear": 0.05,
                "surprise": 0.1,
                "love": 0.05
            }
        }
        
        # Complete analysis result combining sentiment and emotion
        self.sample_analysis_result = {
            "text": "I am really happy with this product!",
            "model": "test-model",
            "sentiment": self.sample_sentiment_result,
            "emotion": self.sample_emotion_result,
            "confidence": 0.82
        }
        
        # Negative sentiment sample for testing color variations
        self.negative_sentiment_result = {
            "label": "negative",
            "score": 0.75,
            "confident": True,
            "threshold": 0.7,
            "raw_probabilities": {
                "positive": 0.1,
                "neutral": 0.15,
                "negative": 0.75
            }
        }
        
        # Low confidence result for testing threshold indicators
        self.low_confidence_result = {
            "label": "neutral",
            "score": 0.5,
            "confident": False,
            "threshold": 0.7,
            "raw_probabilities": {
                "positive": 0.3,
                "neutral": 0.5,
                "negative": 0.2
            }
        }
        
        # Sample for testing negative emotions
        self.negative_emotion_result = {
            "label": "anger",
            "score": 0.78,
            "confident": True,
            "threshold": 0.6,
            "raw_probabilities": {
                "joy": 0.05,
                "sadness": 0.12,
                "anger": 0.78,
                "fear": 0.02,
                "surprise": 0.01,
                "love": 0.02
            }
        }
        
        # Initialize colorama for tests
        init()
    
    def strip_colors(self, text):
        """Remove ANSI color codes from text for easier testing."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def test_format_probabilities(self):
        """Test probability bar formatting with various inputs."""
        # Test standard probability formatting
        result = format_probabilities(self.sample_probabilities)
        result_clean = self.strip_colors(result)
        
        # Check that all labels are present
        self.assertIn("positive", result_clean)
        self.assertIn("neutral", result_clean)
        self.assertIn("negative", result_clean)
        
        # Check that percentages are displayed
        self.assertIn("82.0%", result_clean)
        self.assertIn("15.0%", result_clean)
        self.assertIn("3.0%", result_clean)
        
        # Check that bars are present and proportional
        self.assertIn("█", result_clean)
        self.assertIn("░", result_clean)
        positive_bars = result_clean.count("█", 
                         result_clean.find("positive"), 
                         result_clean.find("82.0%"))
        self.assertGreater(positive_bars, 15)  # Should have more than 15 blocks for 82%
        
        # Test sorting (highest probability first)
        pos_idx = result_clean.find("positive")
        neu_idx = result_clean.find("neutral")
        neg_idx = result_clean.find("negative")
        self.assertTrue(pos_idx < neu_idx < neg_idx)
        
        # Test empty input
        empty_result = format_probabilities({})
        self.assertIn("No probability data available", empty_result)
        
        # Test extremely small values
        tiny_probs = {"label1": 0.001, "label2": 0.999}
        tiny_result = self.strip_colors(format_probabilities(tiny_probs))
        self.assertIn("0.1%", tiny_result)  # Should round to 0.1%
        self.assertIn("99.9%", tiny_result)
    
    def test_format_sentiment_result(self):
        """Test sentiment result formatting with various sentiment types and confidence levels."""
        # Test positive sentiment formatting with high confidence
        positive_result = format_sentiment_result(self.sample_sentiment_result)
        positive_clean = self.strip_colors(positive_result)
        
        self.assertIn("Sentiment: Positive", positive_clean)
        self.assertIn("82.0%", positive_clean)
        self.assertIn("✓", positive_clean)  # Confident indicator
        self.assertIn("threshold: 70.0%", positive_clean)
        
        # Test negative sentiment formatting
        negative_result = format_sentiment_result(self.negative_sentiment_result)
        negative_clean = self.strip_colors(negative_result)
        
        self.assertIn("Sentiment: Negative", negative_clean)
        self.assertIn("75.0%", negative_clean)
        self.assertIn("✓", negative_clean)
        
        # Test neutral sentiment with low confidence
        neutral_result = format_sentiment_result(self.low_confidence_result)
        neutral_clean = self.strip_colors(neutral_result)
        
        self.assertIn("Sentiment: Neutral", neutral_clean)
        self.assertIn("50.0%", neutral_clean)
        self.assertIn("✗", neutral_clean)  # Not confident indicator
        self.assertIn("threshold: 70.0%", neutral_clean)
        
        # Test with probabilities display
        with_probs = format_sentiment_result(self.sample_sentiment_result, show_probabilities=True)
        with_probs_clean = self.strip_colors(with_probs)
        
        self.assertIn("Sentiment Probabilities:", with_probs_clean)
        self.assertIn("positive", with_probs_clean)
        self.assertIn("82.0%", with_probs_clean)
        self.assertIn("█", with_probs_clean)  # Check for bar chart
    
    def test_format_emotion_result(self):
        """Test emotion result formatting with different emotions and confidence levels."""
        # Test positive emotion formatting
        emotion_result = format_emotion_result(self.sample_emotion_result)
        emotion_clean = self.strip_colors(emotion_result)
        
        self.assertIn("Emotion: Joy", emotion_clean)
        self.assertIn("65.0%", emotion_clean)
        self.assertIn("✓", emotion_clean)  # Confident indicator
        self.assertIn("threshold: 60.0%", emotion_clean)
        
        # Test negative emotion formatting
        anger_result = format_emotion_result(self.negative_emotion_result)
        anger_clean = self.strip_colors(anger_result)
        
        self.assertIn("Emotion: Anger", anger_clean)
        self.assertIn("78.0%", anger_clean)
        self.assertIn("✓", anger_clean)
        
        # Test with no emotion detected
        none_emotion = {
            "label": None,
            "score": 0.0,
            "confident": False,
            "threshold": 0.6,
            "raw_probabilities": {}
        }
        none_result = format_emotion_result(none_emotion)
        none_clean = self.strip_colors(none_result)
        
        self.assertIn("Emotion: None detected", none_clean)
        self.assertIn("threshold: 60.0%", none_clean)
        
        # Test with probabilities display
        with_probs = format_emotion_result(self.sample_emotion_result, show_probabilities=True)
        with_probs_clean = self.strip_colors(with_probs)
        
        self.assertIn("Emotion Probabilities:", with_probs_clean)
        self.assertIn("joy", with_probs_clean)
        self.assertIn("65.0%", with_probs_clean)
        self.assertIn("█", with_probs_clean)  # Check for bar chart
    
    def test_format_analysis_result(self):
        """Test complete analysis result formatting combining sentiment and emotion."""
        # Test complete analysis formatting
        analysis_result = format_analysis_result(self.sample_analysis_result)
        analysis_clean = self.strip_colors(analysis_result)
        
        # Check text display
        self.assertIn("I am really happy with this product!", analysis_clean)
        
        # Check sentiment section
        self.assertIn("Sentiment: Positive", analysis_clean)
        self.assertIn("82.0%", analysis_clean)
        
        # Check emotion section
        self.assertIn("Emotion: Joy", analysis_clean)
        self.assertIn("65.0%", analysis_clean)
        
        # Test with probabilities
        with_probs = format_analysis_result(self.sample_analysis_result, show_probabilities=True)
        with_probs_clean = self.strip_colors(with_probs)
        
        self.assertIn("Sentiment Probabilities:", with_probs_clean)
        self.assertIn("Emotion Probabilities:", with_probs_clean)
        
        # Test with long text (should be truncated)
        long_text_result = self.sample_analysis_result.copy()
        long_text_result["text"] = "x" * 150
        long_result = format_analysis_result(long_text_result)
        long_clean = self.strip_colors(long_result)
        
        self.assertIn("...", long_clean)
        self.assertTrue(len(long_clean.split("\n")[0]) < 150)
        
        # Test with negative sentiment and emotion
        negative_result = self.sample_analysis_result.copy()
        negative_result["sentiment"] = self.negative_sentiment_result
        negative_result["emotion"] = self.negative_emotion_result
        negative_analysis = format_analysis_result(negative_result)
        negative_clean = self.strip_colors(negative_analysis)
        
        self.assertIn("Sentiment: Negative", negative_clean)
        self.assertIn("Emotion: Anger", negative_clean)
    
    def test_create_progress_bar(self):
        """Test progress bar generation at different completion levels."""
        # Test progress bar at 0%
        zero_bar = create_progress_bar(0, 100)
        self.assertIn("0.0%", zero_bar)
        self.assertIn("░", zero_bar)
        self.assertNotIn("█", zero_bar)
        
        # Test progress bar at 50%
        half_bar = create_progress_bar(50, 100)
        self.assertIn("50.0%", half_bar)
        self.assertIn("█", half_bar)
        self.assertIn("░", half_bar)
        
        # Test progress bar at 100%
        full_bar = create_progress_bar(100, 100)
        self.assertIn("100.0%", full_bar)
        self.assertIn("█", full_bar)
        self.assertNotIn("░", full_bar) # Should be all filled blocks
        
        # Test custom width
        custom_bar = create_progress_bar(5, 10, width=10)
        self.assertEqual(custom_bar.count("█"), 5)
        self.assertEqual(custom_bar.count("░"), 5)
        
        # Test that counts are displayed
        count_display = create_progress_bar(42, 123)
        self.assertIn("(42/123)", count_display)
    
    def test_flatten_result_for_csv(self):
        """Test flattening nested result structures for CSV export."""
        # Test with a properly structured result for flattening
        proper_result = {
            "text": "Test text",
            "sentiment": "positive",
            "sentiment_score": 0.85,
            "emotion": "joy",
            "emotion_score": 0.75,
            "confidence": 0.8,
            "sentiment_probabilities": {"positive": 0.85, "neutral": 0.1, "negative": 0.05},
            "emotion_probabilities": {"joy": 0.75, "sadness": 0.1, "anger": 0.05}
        }
        
        flat_proper = flatten_result_for_csv(proper_result)
        self.assertEqual(flat_proper["text"], "Test text")
        self.assertEqual(flat_proper["sentiment"], "positive")
        self.assertEqual(flat_proper["sentiment_score"], 0.85)
        self.assertEqual(flat_proper["emotion"], "joy")
        self.assertEqual(flat_proper["emotion_score"], 0.75)
        self.assertEqual(flat_proper["confidence"], 0.8)
        
        # Check probability fields are flattened with correct prefixes
        self.assertIn("sentiment_prob_positive", flat_proper)
        self.assertEqual(flat_proper["sentiment_prob_positive"], 0.85)
        self.assertIn("sentiment_prob_neutral", flat_proper)
        self.assertEqual(flat_proper["sentiment_prob_neutral"], 0.1)
        
        self.assertIn("emotion_prob_joy", flat_proper)
        self.assertEqual(flat_proper["emotion_prob_joy"], 0.75)
        self.assertIn("emotion_prob_sadness", flat_proper)
        self.assertEqual(flat_proper["emotion_prob_sadness"], 0.1)
        
        # Test with missing fields
        minimal_result = {"text": "test only"}
        flat_minimal = flatten_result_for_csv(minimal_result)
        self.assertEqual(flat_minimal["text"], "test only")
        self.assertEqual(flat_minimal["sentiment"], "unknown")
        self.assertEqual(flat_minimal["sentiment_score"], 0)
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_export_to_json(self, mock_makedirs, mock_file):
        """Test JSON export with mocked file operations."""
        # Test JSON export
        results = [self.sample_analysis_result]
        export_to_json(results, "test/path.json")
        
        # Check that directory was created
        mock_makedirs.assert_called_once()
        
        # Check that file was opened for writing
        mock_file.assert_called_once_with("test/path.json", "w", encoding="utf-8")
        
        # Check that correct JSON was written - json.dump writes multiple times
        handle = mock_file()
        # Get all write calls and join them to reconstruct the JSON
        write_calls = [call.args[0] for call in handle.write.call_args_list]
        written_json_str = ''.join(write_calls)
        written_json = json.loads(written_json_str)
        
        self.assertEqual(len(written_json), 1)
        self.assertEqual(written_json[0]["text"], "I am really happy with this product!")
        self.assertEqual(written_json[0]["sentiment"]["label"], "positive")
        self.assertEqual(written_json[0]["emotion"]["label"], "joy")
        
        # Test with multiple results
        mock_file.reset_mock()
        mock_makedirs.reset_mock()
        
        multi_results = [
            self.sample_analysis_result,
            {**self.sample_analysis_result, "text": "Different text"}
        ]
        
        export_to_json(multi_results, "test/multi.json")
        
        handle = mock_file()
        write_calls = [call.args[0] for call in handle.write.call_args_list]
        written_json_str = ''.join(write_calls)
        written_json = json.loads(written_json_str)
        self.assertEqual(len(written_json), 2)
    
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_export_to_csv(self, mock_makedirs, mock_file):
        """Test CSV export with mocked file operations."""
        # Test CSV export with properly structured data
        proper_results = [{
            "text": "Test text",
            "sentiment": "positive",
            "sentiment_score": 0.85,
            "emotion": "joy",
            "emotion_score": 0.75,
            "confidence": 0.8,
            "sentiment_probabilities": {"positive": 0.85, "neutral": 0.1, "negative": 0.05},
            "emotion_probabilities": {"joy": 0.75, "sadness": 0.1, "anger": 0.05}
        }]
        
        export_to_csv(proper_results, "test/path.csv")
        
        # Check directory creation
        mock_makedirs.assert_called_once()
        
        # Check file opening
        mock_file.assert_called_once_with("test/path.csv", "w", encoding="utf-8", newline="")
        
        # Check that the file was written to
        handle = mock_file()
        self.assertTrue(handle.write.called)
        
        # Test with minimal data
        mock_makedirs.reset_mock()
        mock_file.reset_mock()
        
        minimal_results = [{"text": "Just text"}]
        export_to_csv(minimal_results, "test/minimal.csv")
        
        # Verify it handled the minimal case
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()

if __name__ == "__main__":
    unittest.main() 
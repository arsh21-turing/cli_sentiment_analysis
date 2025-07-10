"""
Unit tests for the SentimentEmotionTransformer class.
"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import os
import tempfile
import json

# Path manipulation to ensure imports work correctly
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import SentimentEmotionTransformer

class TestSentimentEmotionTransformer(unittest.TestCase):
    """Test cases for the SentimentEmotionTransformer class."""

    @patch('src.models.transformer.pipeline')
    @patch('src.models.transformer.AutoTokenizer')
    def setUp(self, mock_tokenizer, mock_pipeline):
        """Set up test fixtures."""
        # Configure mocks for faster testing
        self.mock_tokenizer = mock_tokenizer
        self.mock_pipeline = mock_pipeline
        
        # Mock pipeline return values for sentiment analysis
        self.mock_pipeline.side_effect = self._mock_pipeline_factory
        
        # Initialize the transformer with mocked components
        self.transformer = SentimentEmotionTransformer(
            sentiment_threshold=0.6,
            emotion_threshold=0.5,
        )
    
    def _mock_pipeline_factory(self, task, model, tokenizer, device):
        """Create appropriate mock pipeline based on the task."""
        if task == "sentiment-analysis":
            return self._create_sentiment_mock_pipeline()
        elif task == "text-classification":
            return self._create_emotion_mock_pipeline()
        return MagicMock()
    
    def _create_sentiment_mock_pipeline(self):
        """Create a mock sentiment analysis pipeline."""
        mock = MagicMock()
        
        # Define behavior for different input texts
        def side_effect(text):
            if "positive" in text.lower():
                return [{"label": "5 stars", "score": 0.92}]
            elif "negative" in text.lower():
                return [{"label": "1 star", "score": 0.88}]
            elif "neutral" in text.lower():
                return [{"label": "3 stars", "score": 0.75}]
            elif text.strip() == "":
                # Handle empty text
                return [{"label": "3 stars", "score": 0.5}]
            else:
                # Default response for other inputs
                return [{"label": "4 stars", "score": 0.65}]
                
        mock.side_effect = side_effect
        return mock
    
    def _create_emotion_mock_pipeline(self):
        """Create a mock emotion detection pipeline."""
        mock = MagicMock()
        
        # Define behavior for different input texts
        def side_effect(text):
            if "happy" in text.lower() or "joy" in text.lower():
                return [{"label": "joy", "score": 0.85}]
            elif "sad" in text.lower() or "unhappy" in text.lower():
                return [{"label": "sadness", "score": 0.82}]
            elif "angry" in text.lower() or "mad" in text.lower():
                return [{"label": "anger", "score": 0.91}]
            elif "afraid" in text.lower() or "scared" in text.lower():
                return [{"label": "fear", "score": 0.78}]
            elif "surprised" in text.lower() or "shock" in text.lower():
                return [{"label": "surprise", "score": 0.73}]
            elif "love" in text.lower() or "adore" in text.lower():
                return [{"label": "love", "score": 0.89}]
            elif text.strip() == "":
                # Handle empty text
                return [{"label": "neutral", "score": 0.3}]
            else:
                # Default response for other inputs
                return [{"label": "neutral", "score": 0.45}]
                
        mock.side_effect = side_effect
        return mock

    # Test model initialization
    def test_initialization_with_defaults(self):
        """Test that the transformer initializes with default values."""
        self.assertEqual(
            self.transformer.sentiment_model_name,
            SentimentEmotionTransformer.DEFAULT_SENTIMENT_MODEL
        )
        self.assertEqual(
            self.transformer.emotion_model_name,
            SentimentEmotionTransformer.DEFAULT_EMOTION_MODEL
        )
        self.assertEqual(self.transformer.sentiment_threshold, 0.6)
        self.assertEqual(self.transformer.emotion_threshold, 0.5)
    
    @patch('src.models.transformer.pipeline')
    @patch('src.models.transformer.AutoTokenizer')
    def test_initialization_with_custom_values(self, mock_tokenizer, mock_pipeline):
        """Test that the transformer accepts custom models and thresholds."""
        # Mock the pipeline to avoid actual model loading
        mock_pipeline.side_effect = self._mock_pipeline_factory
        
        custom_transformer = SentimentEmotionTransformer(
            sentiment_model_name="nlptown/bert-base-multilingual-uncased-sentiment",  # Use real model name
            emotion_model_name="bhadresh-savani/distilbert-base-uncased-emotion",  # Use real model name
            sentiment_threshold=0.75,
            emotion_threshold=0.65,
        )
        
        self.assertEqual(custom_transformer.sentiment_model_name, "nlptown/bert-base-multilingual-uncased-sentiment")
        self.assertEqual(custom_transformer.emotion_model_name, "bhadresh-savani/distilbert-base-uncased-emotion")
        self.assertEqual(custom_transformer.sentiment_threshold, 0.75)
        self.assertEqual(custom_transformer.emotion_threshold, 0.65)
    
    # Test sentiment analysis
    def test_positive_sentiment_detection(self):
        """Test correct detection of positive sentiment."""
        result = self.transformer.analyze_sentiment("This is a positive experience")
        
        self.assertEqual(result["sentiment"], "positive")
        self.assertTrue(result["score"] > 0.6)  # Score should be high for clear sentiments
    
    def test_negative_sentiment_detection(self):
        """Test correct detection of negative sentiment."""
        result = self.transformer.analyze_sentiment("This is a negative experience")
        
        self.assertEqual(result["sentiment"], "negative")
        self.assertTrue(result["score"] > 0.6)  # Score should be high for clear sentiments
    
    def test_neutral_sentiment_detection(self):
        """Test correct detection of neutral sentiment."""
        result = self.transformer.analyze_sentiment("This is a neutral statement")
        
        self.assertEqual(result["sentiment"], "neutral")
        self.assertTrue(result["score"] > 0)  # Score should be positive
    
    # Test emotion detection
    def test_joy_emotion_detection(self):
        """Test correct detection of joy emotion."""
        result = self.transformer.analyze_emotion("I am so happy about this")
        
        self.assertEqual(result["emotion"], "joy")
        self.assertTrue(result["score"] > 0.5)
    
    def test_sadness_emotion_detection(self):
        """Test correct detection of sadness emotion."""
        result = self.transformer.analyze_emotion("I feel so sad and unhappy")
        
        self.assertEqual(result["emotion"], "sadness")
        self.assertTrue(result["score"] > 0.5)
    
    def test_anger_emotion_detection(self):
        """Test correct detection of anger emotion."""
        result = self.transformer.analyze_emotion("I am very angry about this situation")
        
        self.assertEqual(result["emotion"], "anger")
        self.assertTrue(result["score"] > 0.5)
    
    def test_below_threshold_emotion_returns_none(self):
        """Test that emotions below threshold return None."""
        # Create a transformer with high threshold
        high_threshold_transformer = SentimentEmotionTransformer(
            emotion_threshold=0.95  # Set very high threshold
        )
        
        # The mock returns at most 0.91 for anger, which is below our threshold
        result = high_threshold_transformer.analyze_emotion("I am angry")
        
        self.assertIsNone(result["emotion"])  # Should be None due to high threshold
    
    # Test combined analysis
    def test_combined_sentiment_and_emotion_analysis(self):
        """Test the combined analyze method."""
        result = self.transformer.analyze("I am very angry about this negative experience")
        
        self.assertEqual(result["sentiment"], "negative")
        self.assertIsNotNone(result["sentiment_score"])
        self.assertEqual(result["emotion"], "anger")
        self.assertIsNotNone(result["emotion_score"])
        self.assertEqual(result["text"], "I am very angry about this negative experience")
    
    def test_positive_sentiment_skips_emotion_detection(self):
        """Test that positive sentiment skips emotion detection by default."""
        with patch.object(self.transformer, 'analyze_emotion') as mock_analyze_emotion:
            result = self.transformer.analyze("This is a positive experience")
            
            # Check that analyze_emotion wasn't called
            mock_analyze_emotion.assert_not_called()
            
            # Check that emotion fields are None
            self.assertIsNone(result["emotion"])
            self.assertIsNone(result["emotion_score"])
    
    # Test edge cases
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.transformer.analyze("")
        
        # Verify we get some kind of result (even if not very confident)
        self.assertIsNotNone(result["sentiment"])
        self.assertIsNotNone(result["sentiment_score"])
    
    @patch('src.models.transformer.pipeline')
    def test_model_errors(self, mock_pipeline):
        """Test handling of model errors."""
        # Mock pipeline to raise an exception
        mock_pipeline.side_effect = Exception("Model error")
        
        with self.assertRaises(Exception):
            SentimentEmotionTransformer()
    
    # Integration-style tests
    @pytest.mark.slow
    @unittest.skipIf('CI' in os.environ, "Skip slow tests in CI")
    def test_actual_model_loading(self):
        """Test actual model loading (slow, requires internet)."""
        # Remove mock setup for this test
        self.mock_pipeline.reset_mock()
        self.mock_tokenizer.reset_mock()
        
        # Create actual transformer
        actual_transformer = SentimentEmotionTransformer()
        
        # Test with real text
        result = actual_transformer.analyze_sentiment("I love this product")
        self.assertIsNotNone(result["sentiment"])
        self.assertIsNotNone(result["score"])

if __name__ == "__main__":
    unittest.main()

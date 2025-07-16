"""
Tests for the intelligent fallback system.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import argparse
import json
import tempfile
import os

from src.models.fallback import FallbackSystem
from src.utils.settings import Settings
from src.utils.labels import SentimentLabels, EmotionLabels, LabelMapper
from src.utils.cli import (
    setup_fallback_arguments,
    initialize_fallback_system,
    configure_fallback_from_args,
    show_fallback_settings
)

class TestFallbackSystem(unittest.TestCase):
    """Test the FallbackSystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock models
        self.primary_model = MagicMock()
        self.primary_model.name = "MockPrimaryModel"
        
        self.groq_model = MagicMock()
        self.groq_model.name = "MockGroqModel"
        
        # Create settings
        self.settings = Settings()
        self.settings.set_fallback_threshold(0.35)
        self.settings.set_fallback_strategy("weighted")
        
        # Create fallback system
        self.fallback = FallbackSystem(
            primary_model=self.primary_model,
            groq_model=self.groq_model,
            settings=self.settings
        )
    
    def test_initialization(self):
        """Test proper initialization of the fallback system"""
        # Test with default settings
        fallback = FallbackSystem(self.primary_model)
        self.assertEqual(fallback.primary_model, self.primary_model)
        self.assertEqual(fallback.sentiment_threshold, 0.5)
        self.assertEqual(fallback.emotion_threshold, 0.4)
        self.assertEqual(fallback.fallback_threshold, 0.35)
        self.assertEqual(fallback.fallback_strategy, "weighted")
        self.assertEqual(fallback.always_fallback, False)
        
        # Test with custom settings
        custom_settings = Settings()
        custom_settings.set_sentiment_threshold(0.7)
        custom_settings.set_emotion_threshold(0.6)
        custom_settings.set_fallback_threshold(0.5)
        custom_settings.set_fallback_strategy("highest_confidence")
        custom_settings.set_always_fallback(True)
        
        fallback = FallbackSystem(self.primary_model, settings=custom_settings)
        self.assertEqual(fallback.sentiment_threshold, 0.7)
        self.assertEqual(fallback.emotion_threshold, 0.6)
        self.assertEqual(fallback.fallback_threshold, 0.5)
        self.assertEqual(fallback.fallback_strategy, "highest_confidence")
        self.assertEqual(fallback.always_fallback, True)
    
    def test_low_confidence_detection(self):
        """Test detection of low confidence results"""
        # Low confidence sentiment
        result_low_sentiment = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.3},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.8}
        }
        self.assertTrue(self.fallback.is_low_confidence(result_low_sentiment))
        
        # Low confidence emotion
        result_low_emotion = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.3}
        }
        self.assertTrue(self.fallback.is_low_confidence(result_low_emotion))
        
        # Both low confidence
        result_both_low = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.3},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.3}
        }
        self.assertTrue(self.fallback.is_low_confidence(result_both_low))
        
        # Both above threshold
        result_high_confidence = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        self.assertFalse(self.fallback.is_low_confidence(result_high_confidence))
    
    def test_conflict_detection_sentiment_emotion_mismatch(self):
        """Test detection of conflicts between sentiment and emotion labels"""
        # Conflict: positive sentiment with negative emotion
        result_conflict1 = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_conflict1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "sentiment_emotion_mismatch")
        
        # Conflict: positive sentiment with fear emotion
        result_conflict2 = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.FEAR, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_conflict2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "sentiment_emotion_mismatch")
        
        # Conflict: positive sentiment with anger emotion
        result_conflict3 = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.ANGER, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_conflict3)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "sentiment_emotion_mismatch")
        
        # Conflict: negative sentiment with joy emotion
        result_conflict4 = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_conflict4)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "sentiment_emotion_mismatch")
        
        # Conflict: negative sentiment with love emotion
        result_conflict5 = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.LOVE, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_conflict5)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "sentiment_emotion_mismatch")
        
        # No conflict: matching sentiment and emotion
        result_no_conflict1 = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_no_conflict1)
        self.assertEqual(len(conflicts), 0)
        
        # No conflict: negative sentiment with sadness
        result_no_conflict2 = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_no_conflict2)
        self.assertEqual(len(conflicts), 0)
        
        # No conflict: neutral sentiment with any emotion
        result_no_conflict3 = {
            "sentiment": {"label": SentimentLabels.NEUTRAL, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.7}
        }
        conflicts = self.fallback.detect_conflicts(result_no_conflict3)
        self.assertEqual(len(conflicts), 0)
    
    def test_conflict_detection_conflicting_emotions(self):
        """Test detection of conflicts between multiple emotions"""
        # Conflicting emotions: very close scores
        result_conflict = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {
                "label": EmotionLabels.JOY, 
                "score": 0.45,
                "raw_probabilities": {
                    EmotionLabels.JOY: 0.45,
                    EmotionLabels.LOVE: 0.42,
                    EmotionLabels.SURPRISE: 0.05,
                    EmotionLabels.ANGER: 0.03,
                    EmotionLabels.SADNESS: 0.03,
                    EmotionLabels.FEAR: 0.02
                }
            }
        }
        conflicts = self.fallback.detect_conflicts(result_conflict)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "conflicting_emotions")
        
        # No conflict: clear winner emotion
        result_no_conflict = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {
                "label": EmotionLabels.JOY, 
                "score": 0.7,
                "raw_probabilities": {
                    EmotionLabels.JOY: 0.7,
                    EmotionLabels.LOVE: 0.15,
                    EmotionLabels.SURPRISE: 0.05,
                    EmotionLabels.ANGER: 0.03,
                    EmotionLabels.SADNESS: 0.05,
                    EmotionLabels.FEAR: 0.02
                }
            }
        }
        conflicts = self.fallback.detect_conflicts(result_no_conflict)
        self.assertEqual(len(conflicts), 0)
        
        # No conflict: emotions close but low confidence
        result_no_conflict2 = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {
                "label": EmotionLabels.JOY, 
                "score": 0.25,
                "raw_probabilities": {
                    EmotionLabels.JOY: 0.25,
                    EmotionLabels.LOVE: 0.22,
                    EmotionLabels.SURPRISE: 0.15,
                    EmotionLabels.ANGER: 0.13,
                    EmotionLabels.SADNESS: 0.15,
                    EmotionLabels.FEAR: 0.10
                }
            }
        }
        conflicts = self.fallback.detect_conflicts(result_no_conflict2)
        self.assertEqual(len(conflicts), 0)  # Low overall confidence is handled by is_low_confidence, not detect_conflicts
    
    def test_should_use_fallback(self):
        """Test fallback triggering logic"""
        # Should use fallback: low confidence
        result_low_confidence = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.3},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.8}
        }
        self.assertTrue(self.fallback.should_use_fallback(result_low_confidence))
        
        # Should use fallback: conflicts
        result_conflict = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.7}
        }
        self.assertTrue(self.fallback.should_use_fallback(result_conflict))
        
        # Should not use fallback: high confidence, no conflicts
        result_good = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        self.assertFalse(self.fallback.should_use_fallback(result_good))
        
        # Should use fallback: conflicting emotions
        result_conflict_emotions = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {
                "label": EmotionLabels.JOY, 
                "score": 0.45,
                "raw_probabilities": {
                    EmotionLabels.JOY: 0.45,
                    EmotionLabels.LOVE: 0.42,
                    EmotionLabels.SURPRISE: 0.05,
                    EmotionLabels.ANGER: 0.03,
                    EmotionLabels.SADNESS: 0.03,
                    EmotionLabels.FEAR: 0.02
                }
            }
        }
        self.assertTrue(self.fallback.should_use_fallback(result_conflict_emotions))
    
    def test_analyze_no_fallback_needed(self):
        """Test analysis when fallback is not needed"""
        # Set up primary model to return high-confidence results
        primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        self.primary_model.analyze.return_value = primary_result
        
        # Analyze text
        result = self.fallback.analyze("This is a test")
        
        # Verify primary model was called
        self.primary_model.analyze.assert_called_once_with("This is a test")
        
        # Verify Groq model was not called (not needed)
        self.groq_model.analyze.assert_not_called()
        
        # Verify result matches primary result
        self.assertEqual(result, primary_result)
    
    def test_analyze_with_fallback(self):
        """Test analysis when fallback is needed"""
        # Set up primary model to return low-confidence results
        primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.3},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        self.primary_model.analyze.return_value = primary_result
        
        # Set up fallback model response
        fallback_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.9},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.8}
        }
        self.groq_model.analyze.return_value = fallback_result
        
        # Analyze text
        result = self.fallback.analyze("This is a test")
        
        # Verify both models were called
        self.primary_model.analyze.assert_called_once_with("This is a test")
        self.groq_model.analyze.assert_called_once_with("This is a test")
        
        # Verify result has fallback info
        self.assertIn("fallback_info", result)
        self.assertEqual(result["fallback_info"]["reason"], "low_confidence")
        
        # With weighted strategy, should have combined or used fallback for low confidence sentiment
        if self.fallback.fallback_strategy == "weighted":
            self.assertTrue(
                result["sentiment"]["score"] > primary_result["sentiment"]["score"] or 
                result["sentiment"] == fallback_result["sentiment"]
            )
    
    def test_always_fallback(self):
        """Test always using fallback regardless of confidence"""
        # Enable always fallback
        self.fallback.always_fallback = True
        
        # Set up primary model to return high-confidence results
        primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        self.primary_model.analyze.return_value = primary_result
        
        # Set up fallback model response
        fallback_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.9},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.8}
        }
        self.groq_model.analyze.return_value = fallback_result
        
        # Analyze text
        result = self.fallback.analyze("This is a test")
        
        # Verify both models were called despite high confidence
        self.primary_model.analyze.assert_called_once_with("This is a test")
        self.groq_model.analyze.assert_called_once_with("This is a test")
        
        # Verify result has fallback info
        self.assertIn("fallback_info", result)
        
        # Disable always fallback for other tests
        self.fallback.always_fallback = False


class TestFallbackStrategies(unittest.TestCase):
    """Test different fallback conflict resolution strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock models
        self.primary_model = MagicMock()
        self.groq_model = MagicMock()
        
        # Create settings
        self.settings = Settings()
        
        # Create test data
        self.primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.6},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.6}
        }
        
        self.fallback_result = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.8}
        }
        
        # Create fallback system with default strategy
        self.fallback = FallbackSystem(
            primary_model=self.primary_model,
            groq_model=self.groq_model,
            settings=self.settings
        )
    
    def test_fallback_first_strategy(self):
        """Test fallback_first strategy"""
        # Set strategy
        self.fallback.fallback_strategy = "fallback_first"
        
        # Resolve conflicts
        result = self.fallback.resolve_conflicts(self.primary_result, self.fallback_result)
        
        # Verify fallback result was used for both sentiment and emotion
        self.assertEqual(result["sentiment"], self.fallback_result["sentiment"])
        self.assertEqual(result["emotion"], self.fallback_result["emotion"])
    
    def test_primary_first_strategy(self):
        """Test primary_first strategy"""
        # Set strategy
        self.fallback.fallback_strategy = "primary_first"
        
        # Resolve conflicts
        result = self.fallback.resolve_conflicts(self.primary_result, self.fallback_result)
        
        # Verify primary result was used for both sentiment and emotion
        self.assertEqual(result["sentiment"], self.primary_result["sentiment"])
        self.assertEqual(result["emotion"], self.primary_result["emotion"])
        
        # Test with missing primary sentiment
        primary_missing_sentiment = {
            "emotion": {"label": EmotionLabels.JOY, "score": 0.6}
        }
        
        result = self.fallback.resolve_conflicts(primary_missing_sentiment, self.fallback_result)
        
        # Should use fallback for sentiment but primary for emotion
        self.assertEqual(result["sentiment"], self.fallback_result["sentiment"])
        self.assertEqual(result["emotion"], primary_missing_sentiment["emotion"])
    
    def test_highest_confidence_strategy(self):
        """Test highest_confidence strategy"""
        # Set strategy
        self.fallback.fallback_strategy = "highest_confidence"
        
        # Resolve conflicts
        result = self.fallback.resolve_conflicts(self.primary_result, self.fallback_result)
        
        # Verify highest confidence result was used for both sentiment and emotion
        self.assertEqual(result["sentiment"], self.fallback_result["sentiment"])  # 0.8 > 0.6
        self.assertEqual(result["emotion"], self.fallback_result["emotion"])      # 0.8 > 0.6
        
        # Test with mixed confidence levels
        primary_mixed = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.9},  # Higher than fallback
            "emotion": {"label": EmotionLabels.JOY, "score": 0.5}            # Lower than fallback
        }
        
        result = self.fallback.resolve_conflicts(primary_mixed, self.fallback_result)
        
        # Should use primary for sentiment but fallback for emotion
        self.assertEqual(result["sentiment"], primary_mixed["sentiment"])
        self.assertEqual(result["emotion"], self.fallback_result["emotion"])
    
    def test_weighted_strategy_different_labels(self):
        """Test weighted strategy with different labels"""
        # Set strategy
        self.fallback.fallback_strategy = "weighted"
        
        # Resolve conflicts with significantly higher fallback confidence
        result = self.fallback.resolve_conflicts(self.primary_result, self.fallback_result)
        
        # Should use fallback for both sentiment and emotion since scores are significantly higher
        self.assertEqual(result["sentiment"], self.fallback_result["sentiment"])
        self.assertEqual(result["emotion"], self.fallback_result["emotion"])
        
        # Test with fallback confidence only slightly higher (less than 30% higher)
        fallback_slightly_higher = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.7},  # Only 17% higher than primary
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.7}        # Only 17% higher than primary
        }
        
        result = self.fallback.resolve_conflicts(self.primary_result, fallback_slightly_higher)
        
        # Should use primary since fallback isn't significantly higher
        self.assertEqual(result["sentiment"]["label"], self.primary_result["sentiment"]["label"])
        self.assertEqual(result["emotion"]["label"], self.primary_result["emotion"]["label"])
    
    def test_weighted_strategy_same_labels(self):
        """Test weighted strategy with same labels but different confidence"""
        # Set strategy
        self.fallback.fallback_strategy = "weighted"
        
        # Create test data with same labels but different confidence
        primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.6},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.6}
        }
        
        fallback_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.9},  # Same label, higher confidence
            "emotion": {"label": EmotionLabels.JOY, "score": 0.9}            # Same label, higher confidence
        }
        
        # Create a copy to compare against
        primary_copy = {
            "sentiment": primary_result["sentiment"].copy(),
            "emotion": primary_result["emotion"].copy()
        }
        
        # Resolve conflicts
        result = self.fallback.resolve_conflicts(primary_result, fallback_result)
        
        # Should combine scores for both sentiment and emotion (weighted average)
        # weighted = primary*0.7 + fallback*0.3 when same label
        self.assertEqual(result["sentiment"]["label"], primary_result["sentiment"]["label"])
        self.assertEqual(result["emotion"]["label"], primary_result["emotion"]["label"])
        
        # Verify scores were adjusted
        self.assertGreater(result["sentiment"]["score"], primary_copy["sentiment"]["score"])
        self.assertGreater(result["emotion"]["score"], primary_copy["emotion"]["score"])
        
        # Since fallback is significantly more confident (0.9 vs 0.6), weighted strategy
        # should use fallback when fallback_score > primary_score * 1.3 (0.9 > 0.6 * 1.3 = 0.78)
        self.assertEqual(result["sentiment"]["score"], fallback_result["sentiment"]["score"])
        self.assertEqual(result["emotion"]["score"], fallback_result["emotion"]["score"])
        
        # Since fallback was used entirely, no weighted combination was done
        # If we want to test weighted combination, use closer scores


class TestFallbackInfoFormatting(unittest.TestCase):
    """Test formatting of fallback information"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock models
        self.primary_model = MagicMock()
        self.primary_model.name = "PrimaryModel"
        
        self.groq_model = MagicMock()
        self.groq_model.name = "GroqModel"
        
        # Create test data
        self.primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.6},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.6}
        }
        
        self.fallback_result = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.8}
        }
        
        # Create fallback system with default strategy
        self.fallback = FallbackSystem(
            primary_model=self.primary_model,
            groq_model=self.groq_model
        )
    
    def test_format_fallback_info_low_confidence(self):
        """Test formatting fallback info for low confidence"""
        # Set up test data
        primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.3},  # Low confidence
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        
        fallback_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.9},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.8}
        }
        
        # Format fallback info
        info = self.fallback.format_fallback_info(primary_result, fallback_result)
        
        # Check basic fields
        self.assertEqual(info["reason"], "low_confidence")
        self.assertEqual(info["primary_model"], "PrimaryModel")
        self.assertEqual(info["fallback_model"], "GroqModel")
        self.assertEqual(info["strategy_used"], "weighted")
        
        # Check confidence metrics
        self.assertIn("primary_confidence", info)
        self.assertIn("fallback_confidence", info)
        self.assertEqual(info["primary_confidence"]["sentiment_confidence"], 0.3)
        self.assertEqual(info["primary_confidence"]["emotion_confidence"], 0.7)
        self.assertEqual(info["fallback_confidence"]["sentiment_confidence"], 0.9)
        self.assertEqual(info["fallback_confidence"]["emotion_confidence"], 0.8)
        
        # Check source information
        self.assertIn("sentiment_source", info)
        self.assertIn("emotion_source", info)
        
        # Since labels match and strategy is weighted, should be "combined"
        self.assertEqual(info["sentiment_source"], "combined")
        # Since labels match, should use "combined"
        self.assertEqual(info["emotion_source"], "combined")
        
        # No conflicts for this case
        self.assertIn("conflicts", info)
        self.assertEqual(len(info["conflicts"]), 0)
    
    def test_format_fallback_info_conflicts(self):
        """Test formatting fallback info for conflicts"""
        # Set up test data with conflict
        primary_result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.7}  # Conflicting with positive sentiment
        }
        
        fallback_result = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.7},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.9}
        }
        
        # Format fallback info
        info = self.fallback.format_fallback_info(primary_result, fallback_result)
        
        # Check basic fields
        self.assertEqual(info["reason"], "conflicts")
        self.assertEqual(info["primary_model"], "PrimaryModel")
        self.assertEqual(info["fallback_model"], "GroqModel")
        
        # Check conflicts
        self.assertIn("conflicts", info)
        self.assertEqual(len(info["conflicts"]), 1)
        self.assertEqual(info["conflicts"][0]["type"], "sentiment_emotion_mismatch")
        
        # Check source information
        self.assertIn("sentiment_source", info)
        self.assertIn("emotion_source", info)
    
    def test_determine_source(self):
        """Test determination of which model's prediction was used"""
        # Test with primary only
        primary = {"label": SentimentLabels.POSITIVE, "score": 0.8}
        fallback = {}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "primary")
        
        # Test with fallback only
        primary = {}
        fallback = {"label": SentimentLabels.POSITIVE, "score": 0.8}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "fallback")
        
        # Test with neither
        primary = {}
        fallback = {}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "none")
        
        # Test with different labels, strategy "fallback_first"
        self.fallback.fallback_strategy = "fallback_first"
        primary = {"label": SentimentLabels.POSITIVE, "score": 0.8}
        fallback = {"label": SentimentLabels.NEGATIVE, "score": 0.7}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "fallback")
        
        # Test with different labels, strategy "primary_first"
        self.fallback.fallback_strategy = "primary_first"
        primary = {"label": SentimentLabels.POSITIVE, "score": 0.8}
        fallback = {"label": SentimentLabels.NEGATIVE, "score": 0.7}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "primary")
        
        # Test with different labels, strategy "highest_confidence"
        self.fallback.fallback_strategy = "highest_confidence"
        primary = {"label": SentimentLabels.POSITIVE, "score": 0.6}
        fallback = {"label": SentimentLabels.NEGATIVE, "score": 0.8}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "fallback")
        
        # Test with same labels, strategy "weighted"
        self.fallback.fallback_strategy = "weighted"
        primary = {"label": SentimentLabels.POSITIVE, "score": 0.7}
        fallback = {"label": SentimentLabels.POSITIVE, "score": 0.8}
        
        source = self.fallback._determine_source(primary, fallback)
        self.assertEqual(source, "combined")


class TestFallbackThresholdsAndSettings(unittest.TestCase):
    """Test threshold and settings configuration for the fallback system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock models
        self.primary_model = MagicMock()
        self.groq_model = MagicMock()
        
        # Create fallback system with default settings
        self.fallback = FallbackSystem(
            primary_model=self.primary_model,
            groq_model=self.groq_model
        )
    
    def test_set_fallback_thresholds(self):
        """Test setting fallback thresholds"""
        # Set custom thresholds
        self.fallback.set_fallback_thresholds(
            sentiment_threshold=0.75,
            emotion_threshold=0.65,
            fallback_threshold=0.55
        )
        
        # Verify thresholds were updated
        self.assertEqual(self.fallback.sentiment_threshold, 0.75)
        self.assertEqual(self.fallback.emotion_threshold, 0.65)
        self.assertEqual(self.fallback.fallback_threshold, 0.55)
        
        # Test with invalid values (should clamp to valid range)
        self.fallback.set_fallback_thresholds(
            sentiment_threshold=1.5,     # Too high, should clamp to 1.0
            emotion_threshold=-0.2,      # Too low, should clamp to 0.0
            fallback_threshold=100       # Too high, should clamp to 1.0
        )
        
        # Verify thresholds were clamped to valid range
        self.assertEqual(self.fallback.sentiment_threshold, 1.0)
        self.assertEqual(self.fallback.emotion_threshold, 0.0)
        self.assertEqual(self.fallback.fallback_threshold, 1.0)
        
        # Test with partial update
        self.fallback.set_fallback_thresholds(sentiment_threshold=0.5)
        
        # Verify only sentiment threshold was updated
        self.assertEqual(self.fallback.sentiment_threshold, 0.5)
        self.assertEqual(self.fallback.emotion_threshold, 0.0)  # Unchanged
        self.assertEqual(self.fallback.fallback_threshold, 1.0)  # Unchanged
    
    def test_get_confidence_metrics(self):
        """Test extracting confidence metrics from results"""
        # Test with complete result
        result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        
        metrics = self.fallback.get_confidence_metrics(result)
        self.assertEqual(metrics["sentiment_confidence"], 0.8)
        self.assertEqual(metrics["emotion_confidence"], 0.7)
        self.assertEqual(metrics["overall_confidence"], 0.75)  # Average of both
        
        # Test with sentiment only
        result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8}
        }
        
        metrics = self.fallback.get_confidence_metrics(result)
        self.assertEqual(metrics["sentiment_confidence"], 0.8)
        self.assertNotIn("emotion_confidence", metrics)
        self.assertNotIn("overall_confidence", metrics)
        
        # Test with emotion only
        result = {
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7}
        }
        
        metrics = self.fallback.get_confidence_metrics(result)
        self.assertEqual(metrics["emotion_confidence"], 0.7)
        self.assertNotIn("sentiment_confidence", metrics)
        self.assertNotIn("overall_confidence", metrics)


class TestFallbackCLIIntegration(unittest.TestCase):
    """Test CLI integration with the fallback system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create settings object
        self.settings = Settings()
        
        # Create mock transformer model
        self.transformer_model = MagicMock()
        self.transformer_model.set_fallback_system = MagicMock()
        
        # Create parser and add fallback arguments
        self.parser = argparse.ArgumentParser()
        setup_fallback_arguments(self.parser)
    
    def test_setup_fallback_arguments(self):
        """Test adding fallback arguments to parser"""
        # Parse help output to verify arguments were added
        with patch('sys.stdout') as mock_stdout:
            try:
                self.parser.parse_args(['--help'])
            except SystemExit:
                pass
        
        # Get the help text
        help_text = mock_stdout.write.call_args[0][0]
        
        # Check for fallback arguments in help text
        self.assertIn('--use-fallback', help_text)
        self.assertIn('--no-fallback', help_text)
        self.assertIn('--fallback-threshold', help_text)
        self.assertIn('--always-fallback', help_text)
        self.assertIn('--show-fallback-details', help_text)
        self.assertIn('--fallback-strategy', help_text)
        self.assertIn('--groq-api-key', help_text)
        self.assertIn('--groq-model', help_text)
        self.assertIn('--set-fallback', help_text)
    
    def test_configure_fallback_from_args_enable(self):
        """Test configuring fallback settings from args - enable case"""
        # Create args with fallback enabled
        args = self.parser.parse_args([
            '--use-fallback',
            '--fallback-threshold', '0.4',
            '--fallback-strategy', 'highest_confidence',
            '--always-fallback',
            '--show-fallback-details'
        ])
        
        # Configure settings from args
        configure_fallback_from_args(args, self.settings)
        
        # Verify settings were updated correctly
        self.assertTrue(self.settings.use_fallback)
        self.assertEqual(self.settings.get_fallback_threshold(), 0.4)
        self.assertEqual(self.settings.get_fallback_strategy(), 'highest_confidence')
        self.assertTrue(self.settings.always_fallback)
        self.assertTrue(self.settings.show_fallback_details)
    
    def test_configure_fallback_from_args_disable(self):
        """Test configuring fallback settings from args - disable case"""
        # Enable fallback first
        self.settings.set_fallback_enabled(True)
        
        # Create args with fallback disabled
        args = self.parser.parse_args(['--no-fallback'])
        
        # Configure settings from args
        configure_fallback_from_args(args, self.settings)
        
        # Verify fallback was disabled
        self.assertFalse(self.settings.use_fallback)
    
    def test_initialize_fallback_system(self):
        """Test initializing the fallback system"""
        # Enable fallback
        self.settings.set_fallback_enabled(True)
        
        # Create args with groq API key
        args = argparse.Namespace()
        args.groq_api_key = 'test_api_key'
        args.groq_model = 'llama2-70b-4096'  # Use valid model
        
        # Test that the function actually works now that GroqModel exists
        result = initialize_fallback_system(args, self.settings, self.transformer_model)
        
        # Verify a fallback system was created
        self.assertIsNotNone(result)
        
        # Verify the transformer model had its fallback system set
        self.transformer_model.set_fallback_system.assert_called_once()
    
    def test_initialize_fallback_system_disabled(self):
        """Test initializing fallback system when disabled"""
        # Disable fallback
        self.settings.set_fallback_enabled(False)
        
        # Create args
        args = argparse.Namespace()
        
        # Initialize fallback system
        result = initialize_fallback_system(args, self.settings, self.transformer_model)
        
        # Verify None was returned when disabled
        self.assertIsNone(result)
        
        # Verify transformer model's fallback system was not set
        self.transformer_model.set_fallback_system.assert_not_called()
    
    def test_show_fallback_settings(self):
        """Test displaying fallback settings"""
        # Configure settings
        self.settings.set_fallback_enabled(True)
        self.settings.set_always_fallback(True)
        self.settings.set_show_fallback_details(True)
        self.settings.set_fallback_threshold(0.4)
        self.settings.set_fallback_strategy('highest_confidence')
        
        # Capture printed output
        with patch('builtins.print') as mock_print:
            show_fallback_settings(self.settings)
        
        # Verify correct information was printed
        # Check that print was called with expected strings
        mock_print.assert_any_call("\nFallback System Settings:")
        mock_print.assert_any_call("  Enabled: True")
        mock_print.assert_any_call("  Always use fallback: True")
        mock_print.assert_any_call("  Show details: True")
        mock_print.assert_any_call("  Fallback threshold: 0.4")
        mock_print.assert_any_call("  Conflict resolution strategy: highest_confidence")


class TestOutputFormatterFallbackIntegration(unittest.TestCase):
    """Test output formatter integration with fallback system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import here to avoid circular imports in actual code
        from src.utils.output import OutputFormatter
        from src.utils.labels import LabelMapper
        
        # Create settings
        self.settings = Settings()
        self.settings.set_fallback_enabled(True)
        self.settings.set_show_fallback_details(True)
        
        # Create label mapper and output formatter
        self.label_mapper = LabelMapper(self.settings)
        self.formatter = OutputFormatter(self.label_mapper, self.settings)
        
        # Create test results
        self.result_with_fallback = {
            "sentiment": {"label": "positive", "score": 0.85},
            "emotion": {"label": "joy", "score": 0.75},
            "fallback_info": {
                "reason": "low_confidence",
                "primary_model": "TransformerModel",
                "fallback_model": "GroqModel",
                "primary_confidence": {
                    "sentiment_confidence": 0.3,
                    "emotion_confidence": 0.75,
                    "overall_confidence": 0.525
                },
                "fallback_confidence": {
                    "sentiment_confidence": 0.85,
                    "emotion_confidence": 0.75,
                    "overall_confidence": 0.8
                },
                "conflicts": [],
                "strategy_used": "weighted",
                "sentiment_source": "fallback",
                "emotion_source": "combined"
            }
        }
        
        self.result_with_conflicts = {
            "sentiment": {"label": "negative", "score": 0.7},
            "emotion": {"label": "sadness", "score": 0.8},
            "fallback_info": {
                "reason": "conflicts",
                "primary_model": "TransformerModel",
                "fallback_model": "GroqModel",
                "primary_confidence": {
                    "sentiment_confidence": 0.7,
                    "emotion_confidence": 0.6,
                    "overall_confidence": 0.65
                },
                "fallback_confidence": {
                    "sentiment_confidence": 0.7,
                    "emotion_confidence": 0.8,
                    "overall_confidence": 0.75
                },
                "conflicts": [
                    {
                        "type": "sentiment_emotion_mismatch",
                        "sentiment": "positive",
                        "emotion": "sadness",
                        "description": "Positive sentiment conflicts with sadness emotion"
                    }
                ],
                "strategy_used": "highest_confidence",
                "sentiment_source": "fallback",
                "emotion_source": "fallback"
            }
        }
    
    @patch('src.utils.output.OutputFormatter.format_header')
    @patch('src.utils.output.OutputFormatter.format_sentiment_result')
    @patch('src.utils.output.OutputFormatter.format_emotion_result')
    @patch('src.utils.output.OutputFormatter.format_fallback_section')
    @patch('src.utils.output.OutputFormatter.log_fallback_decision')
    def test_format_analysis_result_with_fallback(
        self, mock_log, mock_fallback_section, 
        mock_emotion_result, mock_sentiment_result, mock_header
    ):
        """Test formatting analysis result with fallback information"""
        # Configure mocks
        mock_header.return_value = "HEADER"
        mock_sentiment_result.return_value = "SENTIMENT"
        mock_emotion_result.return_value = "EMOTION"
        mock_fallback_section.return_value = "FALLBACK"
        
        # Format result
        result = self.formatter.format_analysis_result(self.result_with_fallback)
        
        # Verify methods were called with correct arguments
        mock_sentiment_result.assert_called_once_with(self.result_with_fallback["sentiment"], False)
        mock_emotion_result.assert_called_once_with(self.result_with_fallback["emotion"], False)
        mock_fallback_section.assert_called_once_with(self.result_with_fallback)
        mock_log.assert_called_once_with(self.result_with_fallback["fallback_info"])
        
        # Verify output contains key sections
        self.assertIn("SENTIMENT", result)
        self.assertIn("EMOTION", result)
        self.assertIn("FALLBACK", result)
    
    def test_format_fallback_section(self):
        """Test formatting fallback details section"""
        # Format fallback section
        with patch('src.utils.output.OutputFormatter.format_header') as mock_header:
            mock_header.return_value = "HEADER"
            result = self.formatter.format_fallback_section(self.result_with_fallback)
        
        # Verify output contains key information
        self.assertIn("low_confidence", result)
        self.assertIn("TransformerModel", result)
        self.assertIn("GroqModel", result)
        self.assertIn("weighted", result)
        self.assertIn("Sentiment Confidence", result)
        self.assertIn("Emotion Confidence", result)
        self.assertIn("Result Sources", result)
    
    def test_format_fallback_section_with_conflicts(self):
        """Test formatting fallback section with conflicts"""
        # Format fallback section with conflicts
        with patch('src.utils.output.OutputFormatter.format_header') as mock_header:
            mock_header.return_value = "HEADER"
            result = self.formatter.format_fallback_section(self.result_with_conflicts)
        
        # Verify output contains conflict information
        self.assertIn("conflicts", result.lower())
        self.assertIn("Detected Conflicts", result)
        self.assertIn("Positive sentiment conflicts with sadness emotion", result)
    
    @patch('logging.Logger.info')
    def test_log_fallback_decision(self, mock_info):
        """Test logging fallback decision information"""
        # Log fallback decision
        self.formatter.log_fallback_decision(self.result_with_fallback["fallback_info"])
        
        # Verify logger was called with correct information
        mock_info.assert_any_call("Fallback triggered: low_confidence")
        mock_info.assert_any_call("Primary confidence: sentiment=0.30, emotion=0.75")
        mock_info.assert_any_call("Fallback confidence: sentiment=0.85, emotion=0.75")
        mock_info.assert_any_call("Resolution strategy: weighted")
        mock_info.assert_any_call("Sentiment source: fallback")
        mock_info.assert_any_call("Emotion source: combined")
    
    def test_json_output_with_fallback_info(self):
        """Test JSON output formatting with fallback information"""
        # Configure settings for JSON output
        self.settings.json_stream = True
        
        # Patch json.dumps to capture the data structure
        with patch('json.dumps') as mock_dumps:
            mock_dumps.return_value = "{}"
            self.formatter.format_analysis_result(self.result_with_fallback)
            
            # Get the data that would be JSON serialized
            json_data = mock_dumps.call_args[0][0]
            
            # Verify fallback info is included in JSON output
            self.assertIn("fallback_info", json_data)
            self.assertEqual(json_data["fallback_info"]["reason"], "low_confidence")
            self.assertEqual(json_data["fallback_info"]["strategy_used"], "weighted")
            self.assertIn("primary_confidence", json_data["fallback_info"])
            self.assertIn("fallback_confidence", json_data["fallback_info"])


class TestFallbackIntegration(unittest.TestCase):
    """End-to-end integration tests for fallback system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create settings
        self.settings = Settings()
        self.settings.set_fallback_enabled(True)
        self.settings.set_fallback_threshold(0.4)
        self.settings.set_show_fallback_details(True)
        
        # Create components
        self.label_mapper = LabelMapper(self.settings)
        
        # Mock models to avoid actual API calls during testing
        self.transformer = MagicMock()
        self.transformer.name = "MockTransformer"
        
        self.groq_model = MagicMock()
        self.groq_model.name = "MockGroqModel"
        
        # Create fallback system
        self.fallback = FallbackSystem(
            primary_model=self.transformer,
            groq_model=self.groq_model,
            settings=self.settings
        )
        
        # Create formatter
        from src.utils.output import OutputFormatter
        self.formatter = OutputFormatter(self.label_mapper, self.settings)
    
    def test_end_to_end_low_confidence_case(self):
        """Test end-to-end flow for low confidence case"""
        # Configure mock transformer to return low confidence result
        self.transformer.analyze.return_value = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.35},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.6}
        }
        
        # Configure mock groq model to return high confidence result
        self.groq_model.analyze.return_value = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.9},
            "emotion": {"label": EmotionLabels.SADNESS, "score": 0.85}
        }
        
        # Set fallback on transformer
        mock_transformer = MagicMock()
        mock_transformer.name = "TransformerWithFallback"
        mock_transformer.analyze.side_effect = self.fallback.analyze
        
        # Analyze text
        result = mock_transformer.analyze("This is a test text")
        
        # Verify both models were called
        self.transformer.analyze.assert_called_once_with("This is a test text")
        self.groq_model.analyze.assert_called_once_with("This is a test text")
        
        # Verify result has fallback information
        self.assertIn("fallback_info", result)
        self.assertEqual(result["fallback_info"]["reason"], "low_confidence")
        
        # With weighted strategy and significantly higher Groq confidence,
        # should use Groq's predictions
        self.assertEqual(result["sentiment"]["label"], SentimentLabels.NEGATIVE)
        self.assertEqual(result["emotion"]["label"], EmotionLabels.SADNESS)
        
        # Verify source information is correct
        self.assertEqual(result["fallback_info"]["sentiment_source"], "fallback")
        self.assertEqual(result["fallback_info"]["emotion_source"], "fallback")
        
        # Format result
        formatted = self.formatter.format_analysis_result(result)
        
        # Verify formatted output contains key information
        self.assertIn("Fallback System Details", formatted)
        self.assertIn("Low confidence", formatted)
        self.assertIn("Confidence Comparison", formatted)
        self.assertIn("Result Sources", formatted)
    
    def test_end_to_end_conflict_case(self):
        """Test end-to-end flow for conflict case"""
        # Configure mock transformer to return conflicting result
        self.transformer.analyze.return_value = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {
                "label": EmotionLabels.SADNESS, 
                "score": 0.7,
                "raw_probabilities": {
                    EmotionLabels.SADNESS: 0.7,
                    EmotionLabels.ANGER: 0.1,
                    EmotionLabels.FEAR: 0.1,
                    EmotionLabels.JOY: 0.05,
                    EmotionLabels.LOVE: 0.03,
                    EmotionLabels.SURPRISE: 0.02
                }
            }
        }
        
        # Configure mock groq model result
        self.groq_model.analyze.return_value = {
            "sentiment": {"label": SentimentLabels.NEGATIVE, "score": 0.75},
            "emotion": {
                "label": EmotionLabels.SADNESS, 
                "score": 0.85,
                "raw_probabilities": {
                    EmotionLabels.SADNESS: 0.85,
                    EmotionLabels.ANGER: 0.05,
                    EmotionLabels.FEAR: 0.05,
                    EmotionLabels.JOY: 0.02,
                    EmotionLabels.LOVE: 0.02,
                    EmotionLabels.SURPRISE: 0.01
                }
            }
        }
        
        # Set fallback on transformer
        mock_transformer = MagicMock()
        mock_transformer.name = "TransformerWithFallback"
        mock_transformer.analyze.side_effect = self.fallback.analyze
        
        # Analyze text
        result = mock_transformer.analyze("This is a test text with conflict")
        
        # Verify both models were called
        self.transformer.analyze.assert_called_once()
        self.groq_model.analyze.assert_called_once()
        
        # Verify result has fallback information
        self.assertIn("fallback_info", result)
        self.assertEqual(result["fallback_info"]["reason"], "conflicts")
        
        # Verify conflicts were detected
        self.assertGreaterEqual(len(result["fallback_info"]["conflicts"]), 1)
        self.assertEqual(result["fallback_info"]["conflicts"][0]["type"], "sentiment_emotion_mismatch")
        
        # Format result
        formatted = self.formatter.format_analysis_result(result)
        
        # Verify formatted output contains key information
        self.assertIn("Fallback System Details", formatted)
        self.assertIn("Detected Conflicts", formatted)
        self.assertIn("Positive sentiment conflicts with sadness emotion", formatted)
    
    def test_cli_integration(self):
        """Test integration with CLI"""
        # Create args
        args = type('Args', (), {
            'use_fallback': True,
            'fallback_threshold': 0.45,
            'always_fallback': True,
            'show_fallback_details': True,
            'fallback_strategy': 'highest_confidence',
            'groq_api_key': 'test_api_key',
            'groq_model': 'test_model'
        })()
        
        # Configure settings from args
        self.settings.set_fallback_enabled(False)  # Start disabled
        configure_fallback_from_args(args, self.settings)
        
        # Verify settings were updated
        self.assertTrue(self.settings.use_fallback)
        self.assertEqual(self.settings.get_fallback_threshold(), 0.45)
        self.assertTrue(self.settings.always_fallback)
        self.assertTrue(self.settings.show_fallback_details)
        self.assertEqual(self.settings.get_fallback_strategy(), 'highest_confidence')
        
        # Initialize fallback system (mock the complex parts)
        with patch('src.utils.cli.initialize_fallback_system') as mock_init:
            mock_fallback_instance = MagicMock()
            mock_init.return_value = mock_fallback_instance
            
            # Call initialize
            transformer_model = MagicMock()
            result = initialize_fallback_system(args, self.settings, transformer_model)
            
            # Verify proper initialization
            mock_init.assert_called_once_with(args, self.settings, transformer_model)
            self.assertEqual(result, mock_fallback_instance)
    
    def test_json_stream_format(self):
        """Test JSON stream output format with fallback info"""
        # Enable JSON stream output
        self.settings.json_stream = True
        
        # Create result with fallback info
        result = {
            "sentiment": {"label": SentimentLabels.POSITIVE, "score": 0.8},
            "emotion": {"label": EmotionLabels.JOY, "score": 0.7},
            "fallback_info": {
                "reason": "conflicts",
                "strategy_used": "weighted",
                "primary_model": "Transformer",
                "fallback_model": "Groq",
                "sentiment_source": "primary",
                "emotion_source": "fallback",
                "primary_confidence": {"sentiment_confidence": 0.8, "emotion_confidence": 0.4},
                "fallback_confidence": {"sentiment_confidence": 0.7, "emotion_confidence": 0.7},
                "conflicts": [{"type": "conflicting_emotions", "description": "Test conflict"}]
            }
        }
        
        # Format as JSON
        with patch('json.dumps') as mock_dumps:
            # Force JSON dump to return the actual input to verify structure
            mock_dumps.side_effect = lambda x, **kwargs: str(x)
            
            formatted = self.formatter.format_analysis_result(result)
            
            # Get the data that was passed to json.dumps
            json_data = mock_dumps.call_args[0][0]
            
            # Verify json data structure
            self.assertIn("sentiment", json_data)
            self.assertIn("emotion", json_data)
            self.assertIn("fallback_info", json_data)
            self.assertEqual(json_data["fallback_info"]["reason"], "conflicts")
            self.assertEqual(json_data["fallback_info"]["strategy_used"], "weighted")
            self.assertEqual(json_data["fallback_info"]["sentiment_source"], "primary")
            self.assertEqual(json_data["fallback_info"]["emotion_source"], "fallback")
    
    def test_settings_persistence(self):
        """Test persistence of fallback settings"""
        # Configure settings
        self.settings.set_fallback_enabled(True)
        self.settings.set_fallback_threshold(0.42)
        self.settings.set_always_fallback(True)
        self.settings.set_show_fallback_details(True)
        self.settings.set_fallback_strategy("highest_confidence")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name
            
            # Save and reload settings
            with patch.object(self.settings, 'get_config_path', return_value=temp_path):
                self.settings.save_settings()
                
                # Create new settings object
                new_settings = Settings()
                
                # Should start with defaults
                self.assertFalse(new_settings.use_fallback)  # Default is False
                self.assertNotEqual(new_settings.get_fallback_threshold(), 0.42)
                
                # Load saved settings
                with patch.object(new_settings, 'get_config_path', return_value=temp_path):
                    new_settings.load_settings()
                
                # Verify settings were loaded correctly
                self.assertTrue(new_settings.use_fallback)
                self.assertEqual(new_settings.get_fallback_threshold(), 0.42)
                self.assertTrue(new_settings.always_fallback)
                self.assertTrue(new_settings.show_fallback_details)
                self.assertEqual(new_settings.get_fallback_strategy(), "highest_confidence")
        
        # Clean up
        os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main() 
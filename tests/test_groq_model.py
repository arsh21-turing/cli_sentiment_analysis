"""
Tests for the Groq API model integration.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import json
import os

from src.models.groq import GroqModel
from src.utils.settings import Settings
from src.utils.labels import LabelMapper, SentimentLabels, EmotionLabels

class TestGroqModel(unittest.TestCase):
    """Test the GroqModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variable
        self.env_patcher = patch.dict('os.environ', {'GROQ_API_KEY': 'test_api_key'})
        self.env_patcher.start()
        
        # Create settings
        self.settings = Settings()
        
        # Create label mapper
        self.label_mapper = LabelMapper(self.settings)
        
        # Create model instance
        self.model = GroqModel(
            settings=self.settings,
            label_mapper=self.label_mapper
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
    
    def test_initialization(self):
        """Test proper initialization of the model"""
        # Test default initialization
        self.assertEqual(self.model.api_key, 'test_api_key')
        self.assertEqual(self.model.model_id, 'llama2-70b-4096')
        self.assertEqual(self.model.name, 'GroqModel')
        
        # Test with custom parameters
        custom_model = GroqModel(
            api_key='custom_api_key',
            model='mixtral-8x7b-32768',
            name='CustomGroqModel'
        )
        
        self.assertEqual(custom_model.api_key, 'custom_api_key')
        self.assertEqual(custom_model.model_id, 'mixtral-8x7b-32768')
        self.assertEqual(custom_model.name, 'CustomGroqModel')
    
    def test_validate_model_id(self):
        """Test model ID validation"""
        # Valid built-in models
        self.assertEqual(self.model._validate_model_id('llama2-70b-4096'), 'llama2-70b-4096')
        self.assertEqual(self.model._validate_model_id('mixtral-8x7b-32768'), 'mixtral-8x7b-32768')
        self.assertEqual(self.model._validate_model_id('gemma-7b-it'), 'gemma-7b-it')
        
        # Custom valid model ID format
        self.assertEqual(self.model._validate_model_id('custom-model-123'), 'custom-model-123')
        
        # Invalid model ID should return default
        with patch('logging.Logger.warning') as mock_warning:
            self.assertEqual(self.model._validate_model_id('invalid model id!'), 'llama2-70b-4096')
            mock_warning.assert_called_once()
    
    @patch('requests.post')
    def test_api_call(self, mock_post):
        """Test API call with retry logic"""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Call API
        result = self.model._call_groq_api("Test prompt")
        
        # Verify API was called with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        # Check URL
        self.assertEqual(args[0], 'https://api.groq.com/openai/v1/chat/completions')
        
        # Check headers
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test_api_key')
        self.assertEqual(kwargs['headers']['Content-Type'], 'application/json')
        
        # Check data
        self.assertEqual(kwargs['json']['model'], 'llama2-70b-4096')
        self.assertEqual(len(kwargs['json']['messages']), 2)
        self.assertEqual(kwargs['json']['messages'][0]['role'], 'system')
        self.assertEqual(kwargs['json']['messages'][1]['role'], 'user')
        self.assertEqual(kwargs['json']['messages'][1]['content'], 'Test prompt')
        
        # Check response
        self.assertEqual(result, '{"label": "positive", "score": 0.9}')
    
    @patch('requests.post')
    def test_api_call_retry(self, mock_post):
        """Test API call with retry logic on failure"""
        # Configure mock to fail on first attempt, succeed on second
        mock_error_response = MagicMock()
        mock_error_response.raise_for_status.side_effect = Exception("API Error")
        
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        mock_success_response.raise_for_status = MagicMock()
        
        # First call fails, second succeeds
        mock_post.side_effect = [mock_error_response, mock_success_response]
        
        # Reduce retry delay for faster test
        self.model.retry_delay = 0.01
        
        # Call API
        result = self.model._call_groq_api("Test prompt")
        
        # Verify API was called twice
        self.assertEqual(mock_post.call_count, 2)
        
        # Check response
        self.assertEqual(result, '{"label": "positive", "score": 0.9}')
    
    @patch('requests.post')
    def test_api_call_all_retries_fail(self, mock_post):
        """Test API call when all retries fail"""
        # Configure mock to always fail
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        # Reduce retry delay and retries for faster test
        self.model.retry_delay = 0.01
        self.model.max_retries = 2
        
        # Call API - should raise exception
        with self.assertRaises(RuntimeError):
            self.model._call_groq_api("Test prompt")
        
        # Verify API was called max_retries times
        self.assertEqual(mock_post.call_count, 2)
    
    def test_format_prompt(self):
        """Test prompt formatting for different analysis types"""
        # Sentiment prompt
        sentiment_prompt = self.model.format_prompt("This is a test.", "sentiment")
        self.assertIn("Analyze the sentiment", sentiment_prompt)
        self.assertIn('"positive|negative|neutral"', sentiment_prompt)
        self.assertIn("Text to analyze: \"This is a test.\"", sentiment_prompt)
        
        # Emotion prompt
        emotion_prompt = self.model.format_prompt("This is a test.", "emotion")
        self.assertIn("Analyze the primary emotion", emotion_prompt)
        self.assertIn('"joy|sadness|anger|fear|surprise|love"', emotion_prompt)
        self.assertIn("Text to analyze: \"This is a test.\"", emotion_prompt)
        
        # Combined prompt
        combined_prompt = self.model.format_prompt("This is a test.", "combined")
        self.assertIn("Analyze both the sentiment and primary emotion", combined_prompt)
        self.assertIn('"sentiment":', combined_prompt)
        self.assertIn('"emotion":', combined_prompt)
        self.assertIn("Text to analyze: \"This is a test.\"", combined_prompt)
        
        # Invalid analysis type
        with self.assertRaises(ValueError):
            self.model.format_prompt("This is a test.", "invalid")
    
    def test_parse_response_sentiment(self):
        """Test parsing sentiment response"""
        # Valid sentiment response
        response = '{"label": "positive", "score": 0.9, "raw_probabilities": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}}'
        result = self.model.parse_response(response, "sentiment")
        
        self.assertEqual(result["label"], "positive")
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["raw_probabilities"]["positive"], 0.9)
        
        # Invalid sentiment response
        response = 'invalid json'
        result = self.model.parse_response(response, "sentiment")
        
        # Should return a default result with low confidence
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.3)
    
    def test_parse_response_emotion(self):
        """Test parsing emotion response"""
        # Valid emotion response
        response = '{"label": "joy", "score": 0.8, "raw_probabilities": {"joy": 0.8, "sadness": 0.05, "anger": 0.05, "fear": 0.05, "surprise": 0.03, "love": 0.02}}'
        result = self.model.parse_response(response, "emotion")
        
        self.assertEqual(result["label"], "joy")
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["raw_probabilities"]["joy"], 0.8)
        
        # Invalid emotion response
        response = 'invalid json'
        result = self.model.parse_response(response, "emotion")
        
        # Should return a default result with low confidence
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.3)
    
    def test_parse_response_combined(self):
        """Test parsing combined response"""
        # Valid combined response
        response = '''
        {
            "sentiment": {
                "label": "positive",
                "score": 0.9,
                "raw_probabilities": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
            },
            "emotion": {
                "label": "joy",
                "score": 0.8,
                "raw_probabilities": {"joy": 0.8, "sadness": 0.05, "anger": 0.05, "fear": 0.05, "surprise": 0.03, "love": 0.02}
            }
        }
        '''
        result = self.model.parse_response(response, "combined")
        
        self.assertIn("sentiment", result)
        self.assertIn("emotion", result)
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["sentiment"]["score"], 0.9)
        self.assertEqual(result["emotion"]["label"], "joy")
        self.assertEqual(result["emotion"]["score"], 0.8)
        
        # Invalid combined response
        response = 'invalid json'
        result = self.model.parse_response(response, "combined")
        
        # Should return a default result with low confidence
        self.assertIn("sentiment", result)
        self.assertIn("emotion", result)
        self.assertEqual(result["sentiment"]["label"], "neutral")
        self.assertEqual(result["sentiment"]["score"], 0.3)
        self.assertEqual(result["emotion"]["label"], "neutral")
        self.assertEqual(result["emotion"]["score"], 0.3)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_sentiment(self, mock_call_api):
        """Test sentiment analysis"""
        # Configure mock
        mock_call_api.return_value = '{"label": "positive", "score": 0.9, "raw_probabilities": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}}'
        
        # Analyze sentiment
        result = self.model.analyze_sentiment("This is a positive test.")
        
        # Verify API was called
        mock_call_api.assert_called_once()
        
        # Verify call was made with sentiment prompt
        self.assertIn("Analyze the sentiment", mock_call_api.call_args[0][0])
        
        # Verify result (handle potential tuple from label mapper)
        label = result["label"]
        if isinstance(label, tuple):
            label = label[0]
        self.assertEqual(label, "positive")
        self.assertEqual(result["score"], 0.9)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_emotion(self, mock_call_api):
        """Test emotion analysis"""
        # Configure mock
        mock_call_api.return_value = '{"label": "joy", "score": 0.8, "raw_probabilities": {"joy": 0.8, "sadness": 0.05, "anger": 0.05, "fear": 0.05, "surprise": 0.03, "love": 0.02}}'
        
        # Analyze emotion
        result = self.model.analyze_emotion("This is a joyful test.")
        
        # Verify API was called
        mock_call_api.assert_called_once()
        
        # Verify call was made with emotion prompt
        self.assertIn("Analyze the primary emotion", mock_call_api.call_args[0][0])
        
        # Verify result (handle potential tuple from label mapper)
        label = result["label"]
        if isinstance(label, tuple):
            label = label[0]
        self.assertEqual(label, "joy")
        self.assertEqual(result["score"], 0.8)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_combined(self, mock_call_api):
        """Test combined analysis"""
        # Configure mock
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "positive",
                "score": 0.9,
                "raw_probabilities": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
            },
            "emotion": {
                "label": "joy",
                "score": 0.8,
                "raw_probabilities": {"joy": 0.8, "sadness": 0.05, "anger": 0.05, "fear": 0.05, "surprise": 0.03, "love": 0.02}
            }
        }
        '''
        
        # Analyze combined
        result = self.model.analyze("This is a positive and joyful test.")
        
        # Verify API was called
        mock_call_api.assert_called_once()
        
        # Verify call was made with combined prompt
        self.assertIn("Analyze both the sentiment and primary emotion", mock_call_api.call_args[0][0])
        
        # Verify result (handle potential tuples from label mapper)
        self.assertIn("sentiment", result)
        self.assertIn("emotion", result)
        
        sentiment_label = result["sentiment"]["label"]
        if isinstance(sentiment_label, tuple):
            sentiment_label = sentiment_label[0]
        self.assertEqual(sentiment_label, "positive")
        self.assertEqual(result["sentiment"]["score"], 0.9)
        
        emotion_label = result["emotion"]["label"]
        if isinstance(emotion_label, tuple):
            emotion_label = emotion_label[0]
        self.assertEqual(emotion_label, "joy")
        self.assertEqual(result["emotion"]["score"], 0.8)
    
    def test_set_model(self):
        """Test changing the model"""
        # Change model
        self.model.set_model("mixtral-8x7b-32768")
        
        # Verify model was changed
        self.assertEqual(self.model.model_id, "mixtral-8x7b-32768")
        
        # Change to unknown model
        with patch('logging.Logger.warning') as mock_warning:
            self.model.set_model("custom-model-123")
            self.assertEqual(self.model.model_id, "custom-model-123")
            mock_warning.assert_called_once()
    
    def test_get_model_info(self):
        """Test getting model information"""
        # Get info for built-in model
        info = self.model.get_model_info()
        
        self.assertEqual(info["name"], "GroqModel")
        self.assertEqual(info["model_id"], "llama2-70b-4096")
        self.assertEqual(info["sentiment_threshold"], 0.5)
        self.assertEqual(info["emotion_threshold"], 0.4)
        self.assertEqual(info["model_description"], "Llama 2 (70B parameters)")
        self.assertEqual(info["context_length"], 4096)
        
        # Get info for custom model
        self.model.set_model("custom-model-123")
        info = self.model.get_model_info()
        
        self.assertEqual(info["model_id"], "custom-model-123")
        self.assertEqual(info["model_description"], "Custom model")
    
    def test_set_thresholds(self):
        """Test setting thresholds"""
        # Set thresholds
        self.model.set_thresholds(sentiment_threshold=0.7, emotion_threshold=0.6)
        
        # Verify thresholds were set
        self.assertEqual(self.model.sentiment_threshold, 0.7)
        self.assertEqual(self.model.emotion_threshold, 0.6)
        
        # Test with invalid values (should clamp)
        self.model.set_thresholds(sentiment_threshold=1.5, emotion_threshold=-0.5)
        
        # Verify thresholds were clamped
        self.assertEqual(self.model.sentiment_threshold, 1.0)
        self.assertEqual(self.model.emotion_threshold, 0.0)
    
    def test_cache_functionality(self):
        """Test response caching"""
        # Clear cache
        self.model._response_cache.clear()
        
        # Verify cache is empty
        self.assertEqual(len(self.model._response_cache), 0)
        
        # Add an item to cache
        cache_key = f"{self.model.model_id}:test_prompt"
        self.model._response_cache[cache_key] = "test_response"
        
        # Verify cache contains the item
        self.assertEqual(len(self.model._response_cache), 1)
        
        # Get cache stats
        stats = self.model.get_cache_stats()
        self.assertEqual(stats["cache_size"], 1)
        self.assertEqual(stats["cache_bytes"], len("test_response"))
        
        # Clear cache
        self.model.clear_cache()
        
        # Verify cache is empty again
        self.assertEqual(len(self.model._response_cache), 0)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_with_label_mapping(self, mock_call_api):
        """Test analysis with label mapping"""
        # Configure mock
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "positive",
                "score": 0.9,
                "raw_probabilities": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
            },
            "emotion": {
                "label": "joy",
                "score": 0.8,
                "raw_probabilities": {"joy": 0.8, "sadness": 0.05, "anger": 0.05, "fear": 0.05, "surprise": 0.03, "love": 0.02}
            }
        }
        '''
        
        # Mock the label mapper
        self.model.label_mapper = MagicMock()
        self.model.label_mapper.map_sentiment_label.return_value = "POSITIVE"
        self.model.label_mapper.map_emotion_label.return_value = "JOY"
        
        # Analyze text
        result = self.model.analyze("Test text")
        
        # Verify label mapper was called
        self.model.label_mapper.map_sentiment_label.assert_called_once_with("positive", 0.9, threshold=0.5)
        self.model.label_mapper.map_emotion_label.assert_called_once_with("joy", 0.8, threshold=0.4)
        
        # Verify mapped labels were used
        self.assertEqual(result["sentiment"]["label"], "POSITIVE")
        self.assertEqual(result["emotion"]["label"], "JOY")
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_integration_with_fallback_system(self, mock_call_api):
        """Test integration with the fallback system"""
        # Create fallback system
        from src.models.fallback import FallbackSystem
        
        # Create primary model
        primary_model = MagicMock()
        primary_model.name = "PrimaryModel"
        
        # Create response for primary model
        primary_model.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.3},  # Low confidence
            "emotion": {"label": "joy", "score": 0.7}
        }
        
        # Create response for Groq model
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "negative",
                "score": 0.9,
                "raw_probabilities": {"positive": 0.05, "negative": 0.9, "neutral": 0.05}
            },
            "emotion": {
                "label": "sadness",
                "score": 0.85,
                "raw_probabilities": {"joy": 0.05, "sadness": 0.85, "anger": 0.03, "fear": 0.03, "surprise": 0.02, "love": 0.02}
            }
        }
        '''
        
        # Create fallback system
        fallback = FallbackSystem(
            primary_model=primary_model,
            groq_model=self.model,
            settings=self.settings
        )
        
        # Analyze text
        result = fallback.analyze("Test text")
        
        # Verify both models were called
        primary_model.analyze.assert_called_once()
        mock_call_api.assert_called_once()
        
        # Due to low confidence in primary model's sentiment and higher confidence in Groq,
        # should prefer Groq's sentiment analysis
        # Note: The label mapper may return a tuple (label, bool), so extract first element
        sentiment_label = result["sentiment"]["label"]
        if isinstance(sentiment_label, tuple):
            sentiment_label = sentiment_label[0]
        self.assertEqual(sentiment_label, "negative")
        
        # Verify fallback info was added
        self.assertIn("fallback_info", result)
        self.assertEqual(result["fallback_info"]["reason"], "low_confidence")
        self.assertEqual(result["fallback_info"]["sentiment_source"], "fallback")
        self.assertEqual(result["fallback_info"]["emotion_source"], "fallback")


if __name__ == '__main__':
    unittest.main() 
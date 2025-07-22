"""
Tests for Groq API error handling and resilience.
"""
import unittest
from unittest.mock import MagicMock, patch, call, PropertyMock
import json
import os
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
import time

from src.models.groq import GroqModel
from src.utils.settings import Settings

class TestGroqAPIErrorHandling(unittest.TestCase):
    """Test error handling in the GroqModel class for API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variable for API key
        self.env_patcher = patch.dict('os.environ', {'GROQ_API_KEY': 'test_api_key'})
        self.env_patcher.start()
        
        # Clear response cache
        GroqModel._response_cache = {}
        
        # Create model instance
        self.model = GroqModel(name="TestModel")
        
        # Configure model for faster tests
        self.model.retry_delay = 0.01
        self.model.timeout = 1
        self.model.max_retries = 3
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    def test_missing_api_key(self):
        """Test behavior when API key is missing."""
        # Remove API key
        with patch.dict('os.environ', {}, clear=True):
            # Create model without API key
            model = GroqModel()
            
            # Verify warning was logged
            with patch('logging.Logger.warning') as mock_warning:
                # Access the api_key property to trigger warning
                self.assertIsNone(model.api_key)
                mock_warning.assert_called_once()
            
            # Attempt to call API should raise ValueError
            with self.assertRaises(ValueError):
                model._call_groq_api("Test prompt")
    
    @patch('requests.post')
    def test_network_timeout(self, mock_post):
        """Test handling of network timeouts."""
        # Configure post to time out
        mock_post.side_effect = Timeout("Request timed out")
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("timed out", str(context.exception).lower())
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, self.model.max_retries)
    
    @patch('requests.post')
    def test_connection_error(self, mock_post):
        """Test handling of connection errors."""
        # Configure post to raise connection error
        mock_post.side_effect = ConnectionError("Connection failed")
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("connection failed", str(context.exception).lower())
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, self.model.max_retries)
    
    @patch('requests.post')
    def test_http_error_401_unauthorized(self, mock_post):
        """Test handling of HTTP 401 Unauthorized errors."""
        # Create mock response with 401 status
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("401 Client Error: Unauthorized")
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("unauthorized", str(context.exception).lower())
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, self.model.max_retries)
    
    @patch('requests.post')
    def test_http_error_429_rate_limit(self, mock_post):
        """Test handling of HTTP 429 Too Many Requests errors."""
        # Create mock response with 429 status
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("429 Client Error: Too Many Requests")
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("many requests", str(context.exception).lower())
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, self.model.max_retries)
    
    @patch('requests.post')
    def test_http_error_500_server_error(self, mock_post):
        """Test handling of HTTP 500 Internal Server Error."""
        # Create mock response with 500 status
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("500 Server Error: Internal Server Error")
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("server error", str(context.exception).lower())
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, self.model.max_retries)
    
    @patch('requests.post')
    def test_malformed_json_response(self, mock_post):
        """Test handling of malformed JSON in response."""
        # Create mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "{invalid", 0)
        mock_response.text = "{invalid json"
        mock_post.return_value = mock_response
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("json", str(context.exception).lower())
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, self.model.max_retries)
    
    @patch('requests.post')
    def test_unexpected_response_structure(self, mock_post):
        """Test handling of unexpected structure in JSON response."""
        # Create mock response with unexpected structure
        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "structure"}  # Missing 'choices'
        mock_post.return_value = mock_response
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("unexpected api response format", str(context.exception).lower())
        
        # Verify retry mechanism (should only try once because response structure is valid JSON)
        self.assertEqual(mock_post.call_count, 1)
    
    @patch('requests.post')
    def test_missing_content_in_response(self, mock_post):
        """Test handling of missing content in API response."""
        # Create mock response with missing content
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {}}  # Missing 'content'
            ]
        }
        mock_post.return_value = mock_response
        
        # Attempt API call
        with self.assertRaises(RuntimeError) as context:
            self.model._call_groq_api("Test prompt")
        
        # Verify error message
        self.assertIn("unexpected api response format", str(context.exception).lower())
        
        # Verify no retry (valid JSON but incorrect structure)
        self.assertEqual(mock_post.call_count, 1)
    
    @patch('requests.post')
    def test_retry_success_after_failure(self, mock_post):
        """Test successful retry after initial failure."""
        # First call fails with timeout, second succeeds
        mock_error_response = MagicMock()
        mock_error_response.raise_for_status.side_effect = Timeout("Request timed out")
        
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        
        # Configure post to fail, then succeed
        mock_post.side_effect = [Timeout("Request timed out"), mock_success_response]
        
        # Call API
        result = self.model._call_groq_api("Test prompt")
        
        # Verify retry mechanism
        self.assertEqual(mock_post.call_count, 2)
        
        # Verify successful result
        self.assertEqual(result, '{"label": "positive", "score": 0.9}')
    
    @patch('requests.post')
    def test_exponential_backoff_timing(self, mock_post):
        """Test that exponential backoff is working."""
        # Configure post to always fail
        mock_post.side_effect = ConnectionError("Connection failed")
        
        # Set longer delay to measure
        self.model.retry_delay = 0.1
        
        # Record start time
        start_time = time.time()
        
        # Attempt API call (will fail after retries)
        with self.assertRaises(RuntimeError):
            self.model._call_groq_api("Test prompt")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Expected delay calculation:
        # 1st retry: 0.1 seconds
        # 2nd retry: 0.1 * 2 = 0.2 seconds
        # 3rd retry: 0.1 * 2^2 = 0.4 seconds
        # Total expected: ~0.7 seconds plus processing time
        
        # Verify minimum time elapsed (allowing for processing overhead)
        # Lower bound slightly relaxed for test reliability
        expected_min_time = 0.6  # slightly less than theoretical 0.7
        self.assertGreater(elapsed_time, expected_min_time)
    
    @patch('requests.post')
    def test_response_caching(self, mock_post):
        """Test that response caching works properly."""
        # Configure mock to return success
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        mock_post.return_value = mock_response
        
        # Call API twice with same prompt
        prompt = "Test prompt for caching"
        result1 = self.model._call_groq_api(prompt)
        result2 = self.model._call_groq_api(prompt)
        
        # Verify API was called only once
        mock_post.assert_called_once()
        
        # Verify both results are the same
        self.assertEqual(result1, result2)
        
        # Verify cache contains the response
        cache_key = f"{self.model.model_id}:{prompt}"
        self.assertIn(cache_key, self.model._response_cache)
    
    @patch('requests.post')
    def test_cache_invalidation_on_model_change(self, mock_post):
        """Test that cache is invalidated when model is changed."""
        # Configure mock to return success
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        mock_post.return_value = mock_response
        
        # Call API with initial model
        prompt = "Test prompt for model change"
        result1 = self.model._call_groq_api(prompt)
        
        # Verify cache contains the response
        self.assertEqual(len(self.model._response_cache), 1)
        
        # Change model
        self.model.set_model("mixtral-8x7b-32768")
        
        # Verify cache was cleared
        self.assertEqual(len(self.model._response_cache), 0)
        
        # Call API again with same prompt
        result2 = self.model._call_groq_api(prompt)
        
        # Verify API was called twice
        self.assertEqual(mock_post.call_count, 2)
        
        # Verify new cache entry was created
        self.assertEqual(len(self.model._response_cache), 1)
        
        # Verify new cache key includes new model ID
        new_cache_key = f"mixtral-8x7b-32768:{prompt}"
        self.assertIn(new_cache_key, self.model._response_cache)
    
    @patch('requests.post')
    def test_no_cache_on_error(self, mock_post):
        """Test that errors are not cached."""
        # First call fails, second succeeds
        mock_post.side_effect = [
            Timeout("Request timed out"), 
            MagicMock(
                json=lambda: {
                    "choices": [
                        {"message": {"content": '{"label": "positive", "score": 0.9}'}}
                    ]
                }
            )
        ]
        
        # Set retries to 1 to make test faster
        self.model.max_retries = 1
        
        # First call should fail
        prompt = "Test prompt for error caching"
        with self.assertRaises(RuntimeError):
            self.model._call_groq_api(prompt)
        
        # Verify cache is empty (error should not be cached)
        self.assertEqual(len(self.model._response_cache), 0)
        
        # Reset retries to default
        self.model.max_retries = 3
        
        # Second call should succeed
        result = self.model._call_groq_api(prompt)
        
        # Verify result
        self.assertEqual(result, '{"label": "positive", "score": 0.9}')
        
        # Verify cache now contains successful response
        self.assertEqual(len(self.model._response_cache), 1)
    
    @patch('requests.post')
    def test_clear_cache_method(self, mock_post):
        """Test the clear_cache method."""
        # Configure mock to return success
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        mock_post.return_value = mock_response
        
        # Call API to populate cache
        prompt = "Test prompt for clear cache"
        self.model._call_groq_api(prompt)
        
        # Verify cache contains the response
        self.assertEqual(len(self.model._response_cache), 1)
        
        # Clear cache
        self.model.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.model._response_cache), 0)
    
    @patch('requests.post')
    def test_cache_usage_with_different_prompts(self, mock_post):
        """Test cache behavior with different prompts."""
        # Configure mock to return success
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"label": "positive", "score": 0.9}'}}
            ]
        }
        mock_post.return_value = mock_response
        
        # Call API with different prompts
        prompt1 = "Test prompt 1"
        prompt2 = "Test prompt 2"
        
        result1 = self.model._call_groq_api(prompt1)
        result2 = self.model._call_groq_api(prompt2)
        
        # Verify API was called twice
        self.assertEqual(mock_post.call_count, 2)
        
        # Verify cache contains both responses
        self.assertEqual(len(self.model._response_cache), 2)
        
        # Call again with first prompt
        result3 = self.model._call_groq_api(prompt1)
        
        # Verify API was not called again
        self.assertEqual(mock_post.call_count, 2)
        
        # Verify result matches
        self.assertEqual(result1, result3)


class TestErrorHandlingInHighLevelMethods(unittest.TestCase):
    """Test error handling in the high-level analysis methods of GroqModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variable for API key
        self.env_patcher = patch.dict('os.environ', {'GROQ_API_KEY': 'test_api_key'})
        self.env_patcher.start()
        
        # Create model instance
        self.model = GroqModel(name="TestModel")
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    @patch('logging.Logger.error')
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_sentiment_with_json_error(self, mock_call_api, mock_logger_error):
        """Test analyze_sentiment with malformed JSON response."""
        # Configure mock to return invalid JSON
        mock_call_api.return_value = "{invalid json"
        
        # Call analyze_sentiment
        result = self.model.analyze_sentiment("Test text")
        
        # Verify error was logged
        mock_logger_error.assert_called_once()
        
        # Verify fallback result was returned with low confidence
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.3)
    
    @patch('logging.Logger.error')
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_emotion_with_missing_fields(self, mock_call_api, mock_logger_error):
        """Test analyze_emotion with missing fields in response."""
        # Configure mock to return JSON with missing fields
        mock_call_api.return_value = '{"incomplete": "response"}'
        
        # Call analyze_emotion
        result = self.model.analyze_emotion("Test text")
        
        # Verify error was logged
        mock_logger_error.assert_called_once()
        
        # Verify fallback result was returned with low confidence
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.3)
    
    @patch('logging.Logger.error')
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_combined_with_invalid_scores(self, mock_call_api, mock_logger_error):
        """Test analyze with invalid score values in response."""
        # Configure mock to return JSON with invalid score
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "positive", 
                "score": "not a number",
                "raw_probabilities": {"positive": 0.9}
            },
            "emotion": {
                "label": "joy",
                "score": 0.8,
                "raw_probabilities": {"joy": 0.8}
            }
        }
        '''
        
        # Call analyze
        result = self.model.analyze("Test text")
        
        # Verify error was logged
        mock_logger_error.assert_called_once()
        
        # Verify fallback result was returned for sentiment
        self.assertEqual(result["sentiment"]["label"], "neutral")
        self.assertEqual(result["sentiment"]["score"], 0.3)
        
        # Verify emotion data was still parsed correctly
        self.assertEqual(result["emotion"]["label"], "joy")
        self.assertEqual(result["emotion"]["score"], 0.8)
    
    @patch('logging.Logger.error')
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_with_invalid_labels(self, mock_call_api, mock_logger_error):
        """Test analyze with invalid label values in response."""
        # Configure mock to return JSON with invalid label
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "invalid_label", 
                "score": 0.9,
                "raw_probabilities": {"invalid_label": 0.9}
            },
            "emotion": {
                "label": "joy",
                "score": 0.8,
                "raw_probabilities": {"joy": 0.8}
            }
        }
        '''
        
        # Mock the label mapper
        self.model.label_mapper = MagicMock()
        self.model.label_mapper.map_sentiment_label.return_value = "unknown"
        self.model.label_mapper.map_emotion_label.return_value = "JOY"
        
        # Call analyze
        result = self.model.analyze("Test text")
        
        # Verify label mapper was called with invalid label
        self.model.label_mapper.map_sentiment_label.assert_called_once_with("invalid_label", 0.9, threshold=0.5)
        
        # Verify the mapper's result was used
        self.assertEqual(result["sentiment"]["label"], "unknown")
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_with_api_exception(self, mock_call_api):
        """Test analyze when API call raises an exception."""
        # Configure mock to raise exception
        mock_call_api.side_effect = RuntimeError("API call failed")
        
        # Call analyze_sentiment
        with self.assertRaises(RuntimeError):
            self.model.analyze_sentiment("Test text")
        
        # Call analyze_emotion
        with self.assertRaises(RuntimeError):
            self.model.analyze_emotion("Test text")
        
        # Call analyze
        with self.assertRaises(RuntimeError):
            self.model.analyze("Test text")
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_analyze_with_floating_point_issues(self, mock_call_api):
        """Test analyze with floating point precision issues in JSON."""
        # Configure mock with floating point number that might cause precision issues
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "positive", 
                "score": 0.12345678901234567890,
                "raw_probabilities": {"positive": 0.12345678901234567890}
            }
        }
        '''
        
        # Call analyze_sentiment
        result = self.model.analyze_sentiment("Test text")
        
        # Verify result has proper float value
        self.assertIsInstance(result["score"], float)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_format_prompt_with_special_characters(self, mock_call_api):
        """Test formatting prompts with special characters."""
        # Configure mock
        mock_call_api.return_value = '{"label": "neutral", "score": 0.5}'
        
        # Text with special characters
        text = 'Test with "quotes" and \n newlines and \\backslashes'
        
        # Call analyze_sentiment
        self.model.analyze_sentiment(text)
        
        # Get the prompt that was used
        prompt = mock_call_api.call_args[0][0]
        
        # Verify text was properly escaped in the prompt
        self.assertIn('Test with "quotes" and \\n newlines and \\\\backslashes', prompt)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_handling_of_empty_response_text(self, mock_call_api):
        """Test handling of empty response text."""
        # Configure mock to return empty string
        mock_call_api.return_value = ''
        
        # Call analyze_sentiment
        result = self.model.analyze_sentiment("Test text")
        
        # Verify fallback result was returned
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.3)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_handling_of_null_values_in_response(self, mock_call_api):
        """Test handling of null values in JSON response."""
        # Configure mock with null values
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": null, 
                "score": null,
                "raw_probabilities": null
            }
        }
        '''
        
        # Call analyze
        result = self.model.analyze("Test text")
        
        # Verify sentiment was handled with defaults
        self.assertEqual(result["sentiment"]["label"], "neutral")
        self.assertEqual(result["sentiment"]["score"], 0.3)
        self.assertEqual(result["sentiment"]["raw_probabilities"], {})
    
    @patch('logging.Logger.error')
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_handle_unexpected_json_structure(self, mock_call_api, mock_logger_error):
        """Test handling of unexpected JSON structure that doesn't match prompt."""
        # Configure mock with unexpected but valid JSON
        mock_call_api.return_value = '''
        {
            "unexpected": {
                "field": "value"
            },
            "anotherUnexpected": 123
        }
        '''
        
        # Call analyze
        result = self.model.analyze("Test text")
        
        # Verify error was logged
        mock_logger_error.assert_called_once()
        
        # Verify fallback result was returned
        self.assertIn("sentiment", result)
        self.assertIn("emotion", result)
        self.assertEqual(result["sentiment"]["label"], "neutral")
        self.assertEqual(result["emotion"]["label"], "neutral")
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_handling_of_boolean_score(self, mock_call_api):
        """Test handling of boolean value for score instead of float."""
        # Configure mock with boolean score
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "positive", 
                "score": true,
                "raw_probabilities": {"positive": 0.9}
            }
        }
        '''
        
        # Call analyze_sentiment
        result = self.model.analyze_sentiment("Test text")
        
        # Verify score was converted to float
        self.assertIsInstance(result["score"], float)
        self.assertEqual(result["score"], 0.3)  # Fallback value


class TestIntegrationWithFallbackSystem(unittest.TestCase):
    """Test GroqModel integration with the fallback system under error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variable for API key
        self.env_patcher = patch.dict('os.environ', {'GROQ_API_KEY': 'test_api_key'})
        self.env_patcher.start()
        
        # Create model instances
        self.transformer = MagicMock(name="MockTransformer")
        self.groq_model = GroqModel(name="GroqModel")
        
        # Create settings
        self.settings = Settings()
        self.settings.set_fallback_enabled(True)
        
        # Import FallbackSystem
        from src.models.fallback import FallbackSystem
        
        # Create the fallback system
        self.fallback = FallbackSystem(
            primary_model=self.transformer,
            groq_model=self.groq_model,
            settings=self.settings
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_fallback_with_groq_api_failure(self, mock_call_api):
        """Test fallback system behavior when Groq API call fails."""
        # Configure transformer to return low confidence result
        self.transformer.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.3},  # Low confidence
            "emotion": {"label": "joy", "score": 0.7}
        }
        
        # Configure Groq API to fail
        mock_call_api.side_effect = RuntimeError("API call failed")
        
        # Call analyze
        with patch('logging.Logger.error') as mock_logger_error:
            # Should not propagate Groq API error
            result = self.fallback.analyze("Test text")
            
            # Verify error was logged
            mock_logger_error.assert_called()
        
        # Verify transformer result was used as fallback
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["sentiment"]["score"], 0.3)
        self.assertEqual(result["emotion"]["label"], "joy")
        self.assertEqual(result["emotion"]["score"], 0.7)
        
        # Verify fallback info was not added (since fallback failed)
        self.assertNotIn("fallback_info", result)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_fallback_with_groq_api_malformed_response(self, mock_call_api):
        """Test fallback system behavior with malformed Groq API response."""
        # Configure transformer to return low confidence result
        self.transformer.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.3},  # Low confidence
            "emotion": {"label": "joy", "score": 0.7}
        }
        
        # Configure Groq API to return malformed JSON
        mock_call_api.return_value = "{invalid json"
        
        # Call analyze
        result = self.fallback.analyze("Test text")
        
        # Verify fallback info exists
        self.assertIn("fallback_info", result)
        
        # Verify transformer result takes precedence due to Groq error
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["emotion"]["label"], "joy")
        
        # Verify fallback info shows error
        self.assertEqual(result["fallback_info"]["sentiment_source"], "primary")
        self.assertEqual(result["fallback_info"]["emotion_source"], "primary")
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_fallback_with_partial_groq_failure(self, mock_call_api):
        """Test fallback system with partial Groq API failure (missing fields)."""
        # Configure transformer to return result with conflicting sentiment/emotion
        self.transformer.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.8},
            "emotion": {"label": "sadness", "score": 0.7}  # Conflicts with positive
        }
        
        # Configure Groq API to return incomplete response (missing emotion)
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "negative", 
                "score": 0.9,
                "raw_probabilities": {"negative": 0.9}
            }
        }
        '''
        
        # Call analyze
        result = self.fallback.analyze("Test text")
        
        # Verify fallback info exists
        self.assertIn("fallback_info", result)
        
        # Verify sentiment used Groq data (higher confidence)
        self.assertEqual(result["sentiment"]["label"], "negative")
        self.assertEqual(result["sentiment"]["score"], 0.9)
        
        # Verify emotion used transformer data (only available source)
        self.assertEqual(result["emotion"]["label"], "sadness")
        self.assertEqual(result["emotion"]["score"], 0.7)
        
        # Verify fallback info reflects sources
        self.assertEqual(result["fallback_info"]["sentiment_source"], "fallback")
        self.assertEqual(result["fallback_info"]["emotion_source"], "primary")
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_always_fallback_with_groq_error(self, mock_call_api):
        """Test always-fallback mode when Groq API call fails."""
        # Enable always-fallback mode
        self.fallback.always_fallback = True
        
        # Configure transformer to return high confidence result
        self.transformer.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.9},
            "emotion": {"label": "joy", "score": 0.8}
        }
        
        # Configure Groq API to fail
        mock_call_api.side_effect = RuntimeError("API call failed")
        
        # Call analyze
        result = self.fallback.analyze("Test text")
        
        # Verify transformer result was used
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["sentiment"]["score"], 0.9)
        self.assertEqual(result["emotion"]["label"], "joy")
        self.assertEqual(result["emotion"]["score"], 0.8)
        
        # Verify fallback info was not added
        self.assertNotIn("fallback_info", result)
    
    @patch('src.models.groq.GroqModel._call_groq_api')
    def test_resolve_conflicts_with_low_confidence_groq_response(self, mock_call_api):
        """Test conflict resolution when Groq API returns low confidence."""
        # Configure transformer to return result with conflicting sentiment/emotion
        self.transformer.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.8},
            "emotion": {"label": "sadness", "score": 0.7}  # Conflicts with positive
        }
        
        # Configure Groq API to return low confidence
        mock_call_api.return_value = '''
        {
            "sentiment": {
                "label": "negative", 
                "score": 0.3,
                "raw_probabilities": {"negative": 0.3}
            },
            "emotion": {
                "label": "sadness",
                "score": 0.4,
                "raw_probabilities": {"sadness": 0.4}
            }
        }
        '''
        
        # Set fallback strategy to highest_confidence
        self.fallback.fallback_strategy = "highest_confidence"
        
        # Call analyze
        result = self.fallback.analyze("Test text")
        
        # Verify fallback info exists
        self.assertIn("fallback_info", result)
        
        # Verify transformer results used (higher confidence)
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["sentiment"]["score"], 0.8)
        self.assertEqual(result["emotion"]["label"], "sadness")
        self.assertEqual(result["emotion"]["score"], 0.7)
        
        # Verify fallback info reflects sources
        self.assertEqual(result["fallback_info"]["sentiment_source"], "primary")
        self.assertEqual(result["fallback_info"]["emotion_source"], "primary")
        
        # Verify conflicts still detected
        self.assertGreaterEqual(len(result["fallback_info"]["conflicts"]), 1)


if __name__ == '__main__':
    unittest.main() 
"""
Groq API client utilities for the Streamlit application.
"""

import requests
import json
import time
import streamlit as st
import concurrent.futures
from typing import List, Dict, Callable, Optional, Any, Union
import numpy as np


class GroqClient:
    """
    Client for interacting with Groq API.
    """
    
    def __init__(self, api_key=None, model="llama3-70b-8192"):
        """
        Initialize the Groq client.
        
        Args:
            api_key (str, optional): Groq API key
            model (str): Groq model to use for analysis
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model_speeds = {
            "llama3-70b-8192": 0.8,      # Faster
            "llama3-8b-8192": 0.7,       # Fast
            "mixtral-8x7b-32768": 0.9,   # Medium
            "gemma-7b-it": 0.6           # Fast
        }
    
    def validate_api_key(self):
        """
        Validates if API key is properly set.
        
        Returns:
            bool: True if API key is set, False otherwise
        """
        return bool(self.api_key and self.api_key.strip())
    
    def analyze_sentiment(self, text):
        """
        Analyzes sentiment using Groq API.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        prompt = f"""Analyze the sentiment of the following text and respond with a JSON object containing:
1. "prediction": The overall sentiment (positive, negative, or neutral)
2. "confidence": A confidence score between 0 and 1
3. "positive": The positive sentiment score between 0 and 1
4. "negative": The negative sentiment score between 0 and 1
5. "neutral": The neutral sentiment score between 0 and 1

The scores should add up to 1.0.

Text to analyze: "{text}"

Respond with ONLY a valid JSON object containing the fields above.
"""
        
        response = self._send_request(prompt)
        return self.parse_sentiment_response(response)
    
    def analyze_emotion(self, text):
        """
        Analyzes emotions using Groq API.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Emotion analysis results
        """
        prompt = f"""Analyze the emotions expressed in the following text and respond with a JSON object containing:
1. "scores": An object with keys for the following emotions and values between 0 and 1:
   - joy
   - sadness
   - anger
   - fear
   - surprise
   - disgust
   - trust
   - anticipation
2. "primary_emotion": The emotion with the highest score

Text to analyze: "{text}"

Respond with ONLY a valid JSON object containing the fields above.
"""
        
        response = self._send_request(prompt)
        return self.parse_emotion_response(response)
    
    def analyze_text(self, text, analysis_type="both"):
        """
        Combined analysis of text.
        
        Args:
            text (str): Text to analyze
            analysis_type (str): Type of analysis ("sentiment", "emotion", or "both")
            
        Returns:
            dict: Combined analysis results
        """
        results = {}
        
        if analysis_type in ["sentiment", "both"]:
            results["sentiment"] = self.analyze_sentiment(text)
            
        if analysis_type in ["emotion", "both"]:
            results["emotion"] = self.analyze_emotion(text)
            
        results["text"] = text
        results["model"] = self.model
        
        return results
    
    def process_batch(self, texts: List[str], analysis_type="both", batch_size=5, 
                      progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """
        Process a batch of texts using the Groq API.
        
        Args:
            texts (List[str]): List of texts to analyze
            analysis_type (str): Type of analysis ("sentiment", "emotion", or "both")
            batch_size (int): Number of texts to process in parallel
            progress_callback (callable, optional): Function to call with progress updates
            
        Returns:
            dict: Batch analysis results
        """
        if not texts:
            return {
                "error": "No texts provided",
                "sentiment_counts": {},
                "emotion_distribution": {},
                "average_confidence": 0.0,
                "detailed_results": []
            }
        
        # Initialize counters and results
        total_texts = len(texts)
        processed_count = 0
        detailed_results = []
        
        # Track start time for performance metrics
        start_time = time.time()
        
        # Initialize sentiment and emotion counters
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0}
        emotion_distribution = {}
        total_confidence = 0.0
        
        # Process texts in batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Define the task for each text
            def process_text_task(text):
                if not text or text.isspace():
                    return {
                        "text": text,
                        "error": "Empty text",
                        "sentiment": {"prediction": "unknown", "confidence": 0},
                        "emotion": {"primary_emotion": "unknown", "scores": {}}
                    }
                
                try:
                    return self.analyze_text(text, analysis_type)
                except Exception as e:
                    return {
                        "text": text,
                        "error": str(e),
                        "sentiment": {"prediction": "unknown", "confidence": 0},
                        "emotion": {"primary_emotion": "unknown", "scores": {}}
                    }
            
            # Submit tasks for all texts
            future_to_text = {executor.submit(process_text_task, text): text for text in texts}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_text):
                result = future.result()
                processed_count += 1
                
                # Extract data from result
                if "error" not in result:
                    # Process sentiment results
                    if "sentiment" in result:
                        sentiment = result["sentiment"]["prediction"]
                        confidence = result["sentiment"]["confidence"]
                        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                        total_confidence += confidence
                    else:
                        sentiment_counts["unknown"] += 1
                    
                    # Process emotion results
                    if "emotion" in result:
                        primary_emotion = result["emotion"]["primary_emotion"]
                        emotion_distribution[primary_emotion] = emotion_distribution.get(primary_emotion, 0) + 1
                        # Add emotion scores to the result
                        if "scores" in result["emotion"]:
                            for emotion, score in result["emotion"]["scores"].items():
                                pass  # We already have the scores in the result
                else:
                    sentiment_counts["unknown"] += 1
                
                # Build detailed result entry
                detailed_result = {
                    "text": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                }
                
                if "sentiment" in result:
                    detailed_result["sentiment"] = result["sentiment"]["prediction"]
                    detailed_result["confidence"] = result["sentiment"]["confidence"]
                
                if "emotion" in result:
                    detailed_result["primary_emotion"] = result["emotion"]["primary_emotion"]
                    detailed_result["emotion_scores"] = result["emotion"]["scores"]
                
                if "error" in result:
                    detailed_result["error"] = result["error"]
                
                detailed_results.append(detailed_result)
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(processed_count / total_texts)
        
        # Calculate average confidence
        average_confidence = total_confidence / total_texts if total_texts > 0 else 0
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return combined results
        return {
            "total_texts": total_texts,
            "processed_texts": processed_count,
            "sentiment_counts": sentiment_counts,
            "emotion_distribution": emotion_distribution,
            "average_confidence": average_confidence,
            "processing_time": processing_time,
            "processing_speed": processed_count / processing_time if processing_time > 0 else 0,
            "detailed_results": detailed_results,
            "model": self.model,
            "analysis_type": analysis_type
        }
    
    def parse_sentiment_response(self, response):
        """
        Parses sentiment from response.
        
        Args:
            response (str): Raw API response text
            
        Returns:
            dict: Formatted sentiment results
        """
        try:
            # Extract content from response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Clean the content - remove any markdown formatting
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse JSON directly
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If it fails, try to extract JSON from text (in case there's surrounding text)
                import re
                # Look for JSON object patterns
                json_patterns = [
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
                    r'\{[^}]*\}',  # Simple JSON
                ]
                
                for pattern in json_patterns:
                    json_match = re.search(pattern, content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    # If no JSON found, try to extract key-value pairs
                    result = self._extract_key_value_pairs(content)
            
            # Validate and normalize results
            required_fields = ["prediction", "confidence", "positive", "negative", "neutral"]
            if not all(field in result for field in required_fields):
                missing = [field for field in required_fields if field not in result]
                # Try to infer missing fields
                result = self._infer_missing_sentiment_fields(result)
            
            # Normalize prediction to lowercase
            result["prediction"] = result["prediction"].lower()
            
            # Make sure values are in the right range
            for field in ["confidence", "positive", "negative", "neutral"]:
                result[field] = max(0.0, min(1.0, float(result[field])))
            
            return result
            
        except Exception as e:
            st.error(f"Error parsing sentiment response: {str(e)}")
            st.info(f"Raw response content: {response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0
            }
    
    def parse_emotion_response(self, response):
        """
        Parses emotions from response.
        
        Args:
            response (str): Raw API response text
            
        Returns:
            dict: Formatted emotion results
        """
        try:
            # Extract content from response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Clean the content - remove any markdown formatting
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse JSON directly
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If it fails, try to extract JSON from text (in case there's surrounding text)
                import re
                # Look for JSON object patterns
                json_patterns = [
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
                    r'\{[^}]*\}',  # Simple JSON
                ]
                
                for pattern in json_patterns:
                    json_match = re.search(pattern, content, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    # If no JSON found, try to extract key-value pairs
                    result = self._extract_emotion_key_value_pairs(content)
            
            # Validate and normalize results
            if "scores" not in result:
                # Try to infer scores from available data
                result = self._infer_emotion_scores(result)
            
            # Expected emotions
            expected_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
            
            # Ensure all emotions are present with valid values
            for emotion in expected_emotions:
                if emotion not in result["scores"]:
                    result["scores"][emotion] = 0.0
                else:
                    result["scores"][emotion] = max(0.0, min(1.0, float(result["scores"][emotion])))
            
            # If primary_emotion is not provided, determine it from scores
            if "primary_emotion" not in result:
                if result["scores"]:
                    result["primary_emotion"] = max(result["scores"].items(), key=lambda x: x[1])[0]
                else:
                    result["primary_emotion"] = "unknown"
            
            return result
            
        except Exception as e:
            st.error(f"Error parsing emotion response: {str(e)}")
            st.info(f"Raw response content: {response.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
            return {
                "scores": {emotion: 0.0 for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]},
                "primary_emotion": "unknown"
            }
    
    def get_available_models(self):
        """
        Returns list of available models.
        
        Returns:
            list: List of available model names
        """
        return [
            "llama3-70b-8192",
            "llama3-8b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
    
    def test_connection(self):
        """
        Tests connection to Groq API.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not self.validate_api_key():
            return False
        
        try:
            # Simple test prompt
            test_prompt = "Respond with 'ok' if you can read this message."
            
            response = self._send_request(test_prompt)
            
            # Check if we got a valid response
            if response and "choices" in response:
                return True
            return False
        except Exception:
            return False
    
    def estimate_processing_time(self, num_texts: int, model: str = None) -> float:
        """
        Estimates processing time for batch in seconds.
        
        Args:
            num_texts (int): Number of texts to process
            model (str, optional): Model to use, defaults to current model
            
        Returns:
            float: Estimated time in seconds
        """
        if model is None:
            model = self.model
        speed_factor = self.model_speeds.get(model, 1.0)
        base_time_per_text = 1.5  # seconds
        time_per_text = base_time_per_text * speed_factor
        total_time = time_per_text * num_texts
        overhead = 2.0 + (num_texts * 0.05)
        return total_time + overhead
    
    def _send_request(self, prompt):
        """
        Sends a request to the Groq API.
        
        Args:
            prompt (str): Prompt to send
            
        Returns:
            dict: API response
        """
        if not self.validate_api_key():
            raise ValueError("API key is not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for more deterministic results
            "max_tokens": 1000
        }
        
        # Maximum retries
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=30  # 30 second timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit error - wait and retry
                    retry_count += 1
                    time.sleep(2 * retry_count)  # Exponential backoff
                else:
                    # Other error
                    error_message = f"API Error: {response.status_code} - {response.text}"
                    raise Exception(error_message)
            
            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception("Request timed out after multiple attempts")
                time.sleep(2 * retry_count)  # Exponential backoff
            
            except Exception as e:
                raise Exception(f"Request failed: {str(e)}")
        
        raise Exception("Maximum retries reached")
    
    def _extract_key_value_pairs(self, content):
        """
        Extracts key-value pairs from text content when JSON parsing fails.
        
        Args:
            content (str): Text content to parse
            
        Returns:
            dict: Extracted key-value pairs
        """
        import re
        
        result = {}
        
        # Look for patterns like "key": value or key: value
        patterns = [
            r'"([^"]+)":\s*([^,\s]+)',  # "key": value
            r'([a-zA-Z_]+):\s*([^,\s]+)',  # key: value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for key, value in matches:
                # Clean up the key and value
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        result[key] = value.lower() == 'true'
                    elif '.' in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    result[key] = value
        
        return result
    
    def _infer_missing_sentiment_fields(self, result):
        """
        Infers missing sentiment fields based on available data.
        
        Args:
            result (dict): Partial sentiment result
            
        Returns:
            dict: Complete sentiment result with inferred fields
        """
        # Ensure we have all required fields
        required_fields = ["prediction", "confidence", "positive", "negative", "neutral"]
        
        # If we have a prediction but no confidence, set default confidence
        if "prediction" in result and "confidence" not in result:
            result["confidence"] = 0.8
        
        # If we have individual scores but no prediction, infer prediction
        if "prediction" not in result:
            if "positive" in result and "negative" in result and "neutral" in result:
                scores = {"positive": result["positive"], "negative": result["negative"], "neutral": result["neutral"]}
                result["prediction"] = max(scores, key=scores.get)
            else:
                result["prediction"] = "neutral"
        
        # If we have prediction but no individual scores, set defaults
        if "positive" not in result:
            result["positive"] = 0.33 if result["prediction"] == "positive" else 0.0
        if "negative" not in result:
            result["negative"] = 0.33 if result["prediction"] == "negative" else 0.0
        if "neutral" not in result:
            result["neutral"] = 0.33 if result["prediction"] == "neutral" else 0.0
        
        # If we have individual scores but no confidence, calculate it
        if "confidence" not in result:
            if "positive" in result and "negative" in result and "neutral" in result:
                result["confidence"] = max(result["positive"], result["negative"], result["neutral"])
            else:
                result["confidence"] = 0.5
        
        return result
    
    def _extract_emotion_key_value_pairs(self, content):
        """
        Extracts emotion key-value pairs from text content when JSON parsing fails.
        
        Args:
            content (str): Text content to parse
            
        Returns:
            dict: Extracted emotion data
        """
        import re
        
        result = {"scores": {}}
        
        # Look for emotion patterns
        emotion_patterns = [
            r'"([^"]+)":\s*([^,\s]+)',  # "emotion": value
            r'([a-zA-Z_]+):\s*([^,\s]+)',  # emotion: value
        ]
        
        expected_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
        
        for pattern in emotion_patterns:
            matches = re.findall(pattern, content)
            for key, value in matches:
                # Clean up the key and value
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")
                
                # Check if this is an emotion
                if key in expected_emotions:
                    try:
                        result["scores"][key] = max(0.0, min(1.0, float(value)))
                    except ValueError:
                        result["scores"][key] = 0.0
                elif key == "primary_emotion":
                    result["primary_emotion"] = value
        
        return result
    
    def _infer_emotion_scores(self, result):
        """
        Infers missing emotion scores based on available data.
        
        Args:
            result (dict): Partial emotion result
            
        Returns:
            dict: Complete emotion result with inferred scores
        """
        expected_emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
        
        # If we don't have scores, create default structure
        if "scores" not in result:
            result["scores"] = {}
        
        # Ensure all emotions are present
        for emotion in expected_emotions:
            if emotion not in result["scores"]:
                result["scores"][emotion] = 0.0
        
        # If we have a primary emotion but no scores, set that emotion to high value
        if "primary_emotion" in result and result["primary_emotion"] in expected_emotions:
            result["scores"][result["primary_emotion"]] = 0.8
        
        return result
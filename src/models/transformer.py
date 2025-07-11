"""
Enhanced transformer model for sentiment and emotion analysis with preprocessing and probability distributions.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from unittest.mock import MagicMock

# Import our preprocessing utility
from ..utils.preprocessing import TextPreprocessor

class SentimentEmotionTransformer:
    """
    A transformer-based model for sentiment and emotion analysis.
    
    Features:
    - Sentiment classification (positive, neutral, negative)
    - Emotion detection (sadness, joy, love, anger, fear, surprise)
    - Raw probability distributions
    - Local model loading
    - Comprehensive text preprocessing
    - Model identification for comparison features
    - Separate thresholds for sentiment and emotion
    """
    
    def __init__(
        self, 
        sentiment_model: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
        emotion_model: str = 'bhadresh-savani/distilbert-base-uncased-emotion',
        sentiment_threshold: float = 0.7,
        emotion_threshold: float = 0.6,
        local_model_path: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the SentimentEmotionTransformer with models for sentiment and emotion analysis.
        
        Args:
            sentiment_model: Model identifier for sentiment analysis
            emotion_model: Model identifier for emotion analysis
            sentiment_threshold: Confidence threshold for sentiment predictions
            emotion_threshold: Confidence threshold for emotion predictions
            local_model_path: Path to locally saved models (if None, will download from HuggingFace)
            name: The name to identify this model in comparisons (if None, uses the model identifiers)
        """
        self.device = self._get_device()
        self.sentiment_threshold = sentiment_threshold
        self.emotion_threshold = emotion_threshold
        self.sentiment_model_id = sentiment_model
        self.emotion_model_id = emotion_model
        self.local_model_path = local_model_path
        
        # Set model name for comparison
        if name:
            self.name = name
        else:
            sentiment_name = sentiment_model.split('/')[-1]
            emotion_name = emotion_model.split('/')[-1]
            self.name = f"{sentiment_name}+{emotion_name}"
        
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor(
            remove_urls=True,
            remove_html=True,
            fix_encoding=True,
            handle_emojis='keep',
            lowercase=True
        )
        
        # Load models (either from local path or HuggingFace)
        if local_model_path:
            sentiment_path = os.path.join(local_model_path, 'sentiment')
            emotion_path = os.path.join(local_model_path, 'emotion')
            
            self.sentiment_model, self.sentiment_tokenizer = self.load_local_model(
                'sentiment', sentiment_path
            )
            self.emotion_model, self.emotion_tokenizer = self.load_local_model(
                'emotion', emotion_path
            )
        else:
            # Load from HuggingFace
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                sentiment_model
            ).to(self.device)
            
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
                emotion_model
            ).to(self.device)
        
        # Define label mappings
        self.sentiment_labels = {
            1: "negative",
            2: "neutral",
            3: "neutral",  # Map both 2 and 3 to neutral
            4: "positive",
            5: "positive"
        }
        
        self.emotion_labels = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
    
    def set_thresholds(self, sentiment_threshold: Optional[float] = None, emotion_threshold: Optional[float] = None) -> None:
        """
        Update the confidence thresholds for sentiment and emotion analysis.
        
        Args:
            sentiment_threshold: New threshold for sentiment analysis (0.0 to 1.0)
            emotion_threshold: New threshold for emotion analysis (0.0 to 1.0)
        """
        if sentiment_threshold is not None:
            self.sentiment_threshold = max(0.0, min(1.0, sentiment_threshold))
        
        if emotion_threshold is not None:
            self.emotion_threshold = max(0.0, min(1.0, emotion_threshold))
    
    def _get_device(self) -> str:
        """
        Determine the available device for model inference.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_local_model(
        self, 
        model_type: str, 
        model_path: str
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load model and tokenizer from local path.
        
        Args:
            model_type: Type of model ('sentiment' or 'emotion')
            model_path: Path to locally saved model
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            FileNotFoundError: If model path doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model path not found: {model_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        return model, tokenizer
    
    def get_raw_probabilities(self, model_output: torch.Tensor) -> Dict[str, float]:
        """
        Convert model output tensor to a dictionary of label-probability pairs.
        
        Args:
            model_output: The raw model output tensor
            
        Returns:
            Dictionary mapping labels to probability scores
        """
        # Convert to numpy and apply softmax to get probabilities
        probs = torch.nn.functional.softmax(model_output, dim=1).detach().cpu().numpy()[0]
        
        # Create dictionary based on model type
        if len(probs) == len(self.sentiment_labels):
            return {self.sentiment_labels[i+1]: float(probs[i]) for i in range(len(probs))}
        else:
            return {self.emotion_labels[i]: float(probs[i]) for i in range(len(probs))}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        
        Uses sentiment_threshold to determine confidence level.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary with sentiment classification result, confidence score,
            and raw probability distribution
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Tokenize for model input
        inputs = self.sentiment_tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        # Get raw probabilities
        raw_probs = self.get_raw_probabilities(outputs.logits)
        
        # Get sentiment label and score
        predicted_class = torch.argmax(outputs.logits, dim=1).item() + 1
        sentiment = self.sentiment_labels[predicted_class]
        score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_class-1].item()
        
        # Check confidence against sentiment threshold
        is_confident = score >= self.sentiment_threshold
        
        # If not confident, use "neutral" as fallback
        if not is_confident and sentiment != "neutral":
            # Find the neutral score
            neutral_score = max(raw_probs.get("neutral", 0.0), 0.0)
            
            # If neutral score is reasonable, use it instead
            if neutral_score > 0.2:  # At least some neutral signal
                sentiment = "neutral"
                score = neutral_score
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confident": is_confident,
            "threshold": self.sentiment_threshold,
            "raw_probabilities": raw_probs
        }
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotion of the given text.
        
        Uses emotion_threshold to determine confidence level.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary with emotion classification result, confidence score,
            and raw probability distribution
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Tokenize for model input
        inputs = self.emotion_tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
        
        # Get raw probabilities
        raw_probs = self.get_raw_probabilities(outputs.logits)
        
        # Get emotion label and score
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        emotion = self.emotion_labels[predicted_class]
        score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][predicted_class].item()
        
        # Check confidence against emotion threshold
        is_confident = score >= self.emotion_threshold
        
        return {
            "emotion": emotion if is_confident else None,
            "score": score,
            "confident": is_confident,
            "threshold": self.emotion_threshold,
            "raw_probabilities": raw_probs
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform complete sentiment and emotion analysis on the given text.
        
        Uses both sentiment_threshold and emotion_threshold to judge confidence.
        Optimization: Only perform emotion analysis for negative sentiment
        to improve performance.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary with combined sentiment and emotion analysis results
        """
        sentiment_result = self.analyze_sentiment(text)
        
        # Only analyze emotion for negative sentiment or if not confident in positive/neutral
        if (sentiment_result["sentiment"] == "negative" or 
            not sentiment_result["confident"]):
            emotion_result = self.analyze_emotion(text)
        else:
            # For confident positive/neutral sentiment, skip emotion analysis
            emotion_result = {
                "emotion": None,
                "score": 0.0,
                "confident": False,
                "threshold": self.emotion_threshold,
                "raw_probabilities": {}
            }
        
        return {
            "text": text,
            "model": self.name,
            "sentiment": {
                "label": sentiment_result["sentiment"],
                "score": sentiment_result["score"],
                "confident": sentiment_result["confident"],
                "threshold": sentiment_result["threshold"],
                "raw_probabilities": sentiment_result["raw_probabilities"]
            },
            "emotion": {
                "label": emotion_result["emotion"],
                "score": emotion_result["score"],
                "confident": emotion_result["confident"],
                "threshold": emotion_result["threshold"],
                "raw_probabilities": emotion_result["raw_probabilities"]
            },
            "confidence": max(sentiment_result["score"], emotion_result.get("score", 0.0))
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about this model for display in comparison results.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.name,
            "sentiment_model": self.sentiment_model_id,
            "emotion_model": self.emotion_model_id,
            "device": self.device,
            "sentiment_threshold": self.sentiment_threshold,
            "emotion_threshold": self.emotion_threshold,
            "local_model": bool(self.local_model_path)
        }

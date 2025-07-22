"""
Enhanced transformer model for sentiment and emotion analysis with preprocessing and probability distributions.
"""

import os
import re
import warnings

# Suppress expected model loading warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

# Set environment variable to suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ---------------------------------------------------------------------------
# Optional heavy dependency (torch)
# ---------------------------------------------------------------------------
import sys
import types

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – lightweight stub for test envs
    torch = types.ModuleType("torch")  # type: ignore

    class _Cuda:
        @staticmethod
        def is_available() -> bool:  # noqa: D401
            return False

    torch.cuda = _Cuda()  # type: ignore

    class _MPSBackend:
        @staticmethod
        def is_available() -> bool:  # noqa: D401
            return False

    torch.backends = types.SimpleNamespace(mps=_MPSBackend())  # type: ignore

    torch.Tensor = object  # type: ignore

    sys.modules.setdefault("torch", torch)

# Optional lightweight numpy stub for test environments
try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import types as _types, sys as _sys

    np = _types.ModuleType("numpy")  # type: ignore

    # Provide minimal attributes used by the code (empty placeholder)
    np.ndarray = object  # type: ignore

    def _array(*_, **__):  # type: ignore
        return []

    np.array = _array  # type: ignore
    _sys.modules.setdefault("numpy", np)

from typing import Dict, List, Optional, Tuple, Union, Any
# Provide minimal fallback for `transformers` when missing in lightweight test envs
try:
    from transformers import (
        AutoModelForSequenceClassification,  # type: ignore
        AutoTokenizer,  # type: ignore
        pipeline,  # type: ignore
    )
except ModuleNotFoundError:  # pragma: no cover – lightweight stub
    import types as _types, sys as _sys

    _tfm = _types.ModuleType("transformers")

    def _dummy_pipeline(*_, **__):  # type: ignore
        return lambda text: {"sentiment": {"label": "neutral", "score": 0.5}}

    class _Dummy:
        @classmethod
        def from_pretrained(cls, *_a, **_kw):  # noqa: D401
            return cls()

    _tfm.pipeline = _dummy_pipeline  # type: ignore
    _tfm.AutoTokenizer = _Dummy  # type: ignore
    _tfm.AutoModelForSequenceClassification = _Dummy  # type: ignore

    _sys.modules.setdefault("transformers", _tfm)

    from transformers import (  # type: ignore  # now picks up stub
        AutoModelForSequenceClassification,  # type: ignore
        AutoTokenizer,  # type: ignore
        pipeline,  # type: ignore
    )
from unittest.mock import MagicMock

# Import our preprocessing utility
from ..utils.preprocessing import TextPreprocessor
from ..utils.logging_system import log_model_prediction

class SentimentEmotionTransformer:
    """
    Improved transformer-based model for sentiment and emotion analysis
    using RoBERTa model specifically trained for sentiment that properly
    identifies negative sentiments.
    """

    def __init__(
        self,
        sentiment_model: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        emotion_model: str = 'bhadresh-savani/distilbert-base-uncased-emotion',
        settings=None,
        label_mapper=None,
        local_model_path: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        use_cache: bool = True,
        verify_negation: bool = True
    ):
        """
        Initialize the SentimentEmotionTransformer with specified models.
        
        Args:
            sentiment_model: Hugging Face model ID for sentiment analysis
            emotion_model: Hugging Face model ID for emotion analysis
            settings: Settings object for thresholds and configuration
            label_mapper: LabelMapper object for consistent label mapping
            local_model_path: Dict with paths to local models (keys: 'sentiment', 'emotion')
            name: Optional name for the model (useful for comparisons)
            use_cache: Whether to use result caching for performance
            verify_negation: Whether to verify negative sentiment with rule-based patterns
        """
        self.settings = settings
        self.label_mapper = label_mapper
        self.name = name or "TransformerModel"
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.verify_negation = verify_negation
        
        # Sentiment labels for RoBERTa model
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Emotion labels for the emotion model
        self.emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        # Negation patterns for rule-based verification
        self.negation_patterns = [
            r'\b(?:not|no|never|none|neither|nor|without)\b',
            r"\b(?:can't|cannot|couldn't|won't|wouldn't|shouldn't|didn't|doesn't|don't)\b",
            r'\b(?:barely|hardly|rarely|seldom)\b',
            r'\b(?:isn\'t|aren\'t|wasn\'t|weren\'t)\b',
            r'\b(?:dislike|hate|despise|loathe|abhor|detest)\b',
            r'\b(?:awful|terrible|horrible|bad|worst|poor|disappointing|unsatisfactory)\b',
            r'\b(?:unpleasant|unfortunate|undesirable|unsatisfactory|inadequate)\b',
            r'^\s*(?:no|nope)\s*$'  # Just "no" or "nope"
        ]
        
        # Set device (CUDA, MPS, or CPU)
        self.device = self._set_device()
        
        # Suppress warnings globally for this initialization
        warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
        
        # Load models with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if local_model_path and 'sentiment' in local_model_path:
                self.sentiment_tokenizer, self.sentiment_model = self.load_local_model(
                    'sentiment', local_model_path['sentiment'])
            else:
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
                
            if local_model_path and 'emotion' in local_model_path:
                self.emotion_tokenizer, self.emotion_model = self.load_local_model(
                    'emotion', local_model_path['emotion'])
            else:
                self.emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model)
                self.emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model)
        
        # Move models to appropriate device
        self.sentiment_model.to(self.device)
        self.emotion_model.to(self.device)
        
        # Set thresholds from settings if available
        self.sentiment_threshold = 0.5
        self.emotion_threshold = 0.4
        if settings:
            self.sentiment_threshold = settings.get_sentiment_threshold()
            self.emotion_threshold = settings.get_emotion_threshold()
            
        # Initialize fallback system reference
        self.fallback_system = None
    
    def _set_device(self) -> torch.device:
        """
        Determine and set the appropriate device (CUDA, MPS, or CPU).
        
        Returns:
            torch.device: The device to use for model inference
        """
        # Force CPU for now to avoid memory issues
        return torch.device('cpu')
        
        # Uncomment below for GPU support when memory issues are resolved
        # if torch.cuda.is_available():
        #     try:
        #         # Check if we have enough GPU memory
        #         torch.cuda.empty_cache()
        #         if torch.cuda.memory_allocated() < torch.cuda.get_device_properties(0).total_memory * 0.8:
        #             return torch.device('cuda')
        #         else:
        #             print("⚠️ GPU memory insufficient, falling back to CPU")
        #             return torch.device('cpu')
        #     except Exception:
        #         print("⚠️ CUDA error, falling back to CPU")
        #         return torch.device('cpu')
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     return torch.device('mps')
        # else:
        #     return torch.device('cpu')
            
    def load_local_model(self, model_type: str, model_path: str) -> Tuple:
        """
        Load a model and tokenizer from a local path.
        
        Args:
            model_type: Type of model ('sentiment' or 'emotion')
            model_path: Path to the local model
            
        Returns:
            Tuple of (tokenizer, model)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
        
    @log_model_prediction("transformer")
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing 'label', 'score', and 'raw_probabilities'
        """
        # Check cache if enabled
        if self.use_cache and text in self.cache and 'sentiment' in self.cache[text]:
            return self.cache[text]['sentiment']
            
        # Prepare inputs
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            
        # Process outputs
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        raw_probabilities = {self.sentiment_labels[i]: float(scores[i]) for i in range(len(self.sentiment_labels))}
        
        # Get predicted label and score
        predicted_idx = np.argmax(scores)
        label = self.sentiment_labels[predicted_idx]
        score = float(scores[predicted_idx])
        
        # Use rule-based verification for negative sentiment if enabled
        if self.verify_negation:
            result = self.verify_negative_sentiment(text, raw_probabilities)
            if result:
                label = 'negative'
                score = max(raw_probabilities['negative'], 0.6)  # Ensure confidence is at least medium-high
                raw_probabilities = {
                    'negative': score,
                    'neutral': min(raw_probabilities['neutral'], 0.3),
                    'positive': min(raw_probabilities['positive'], 0.1)
                }
        
        # Map label through label mapper if available
        if self.label_mapper:
            mapped_label = self.map_labels(label, score, is_emotion=False)
            if mapped_label != label:
                label = mapped_label
        
        result = {
            'label': label,
            'score': score,
            'raw_probabilities': raw_probabilities
        }
        
        # Store in cache if enabled
        if self.use_cache:
            if text not in self.cache:
                self.cache[text] = {}
            self.cache[text]['sentiment'] = result
            
        return result
        
    @log_model_prediction("transformer")
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotions in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing 'label', 'score', and 'raw_probabilities'
        """
        # Check cache if enabled
        if self.use_cache and text in self.cache and 'emotion' in self.cache[text]:
            return self.cache[text]['emotion']
            
        # Prepare inputs
        inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            
        # Process outputs
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        raw_probabilities = {self.emotion_labels[i]: float(scores[i]) for i in range(len(self.emotion_labels))}
        
        # Get predicted label and score
        predicted_idx = np.argmax(scores)
        label = self.emotion_labels[predicted_idx]
        score = float(scores[predicted_idx])
        
        # Map label through label mapper if available
        if self.label_mapper:
            mapped_label = self.map_labels(label, score, is_emotion=True)
            if mapped_label != label:
                label = mapped_label
        
        result = {
            'label': label,
            'score': score,
            'raw_probabilities': raw_probabilities
        }
        
        # Store in cache if enabled
        if self.use_cache:
            if text not in self.cache:
                self.cache[text] = {}
            self.cache[text]['emotion'] = result
            
        return result
        
    @log_model_prediction("transformer")
    def analyze(self, text: str, use_fallback: Optional[bool] = None) -> Dict[str, Any]:
        """
        Perform combined sentiment and emotion analysis.
        
        Args:
            text: The text to analyze
            use_fallback: Whether to use the fallback system (overrides settings)
            
        Returns:
            Dict with combined sentiment and emotion analysis
        """
        # Check cache if enabled
        if self.use_cache and text in self.cache and 'combined' in self.cache[text]:
            return self.cache[text]['combined'].copy()
            
        # Start with sentiment analysis
        sentiment_result = self.analyze_sentiment(text)
        
        # Determine whether to run fallback
        should_use_fallback = False
        if use_fallback is not None:
            should_use_fallback = use_fallback
        elif self.fallback_system and self.settings:
            should_use_fallback = self.settings.use_fallback()
        
        # Use fallback if needed and available
        fallback_info = None
        if should_use_fallback and sentiment_result['score'] < self.sentiment_threshold and self.fallback_system:
            fallback_result = self.fallback_system.analyze_sentiment(text)
            
            # If fallback confidence is higher, use it
            if fallback_result['score'] > sentiment_result['score']:
                old_label = sentiment_result['label']
                old_score = sentiment_result['score']
                
                sentiment_result = fallback_result
                
                # Record fallback info for explanation
                fallback_info = {
                    'used': True,
                    'reason': f"Low confidence ({old_score:.2f} < {self.sentiment_threshold:.2f})",
                    'original_label': old_label,
                    'original_score': old_score,
                    'fallback_label': fallback_result['label'],
                    'fallback_score': fallback_result['score']
                }
            else:
                # Record that fallback was attempted but not used
                fallback_info = {
                    'used': False,
                    'reason': f"Fallback confidence ({fallback_result['score']:.2f}) not higher than original ({sentiment_result['score']:.2f})",
                    'original_label': sentiment_result['label'],
                    'original_score': sentiment_result['score'],
                    'fallback_label': fallback_result['label'],
                    'fallback_score': fallback_result['score']
                }
        
        # Only run emotion analysis if sentiment is negative or neutral
        # This optimization avoids unnecessary processing for clearly positive text
        emotion_result = None
        if sentiment_result['label'].lower() in ['negative', 'neutral'] or sentiment_result['score'] < 0.8:
            emotion_result = self.analyze_emotion(text)
        
        # Combine results
        result = {
            'text': text,
            'sentiment': sentiment_result,
        }
        
        if emotion_result:
            result['emotion'] = emotion_result
            
        if fallback_info:
            result['fallback_info'] = fallback_info
        
        # Store in cache if enabled
        if self.use_cache:
            if text not in self.cache:
                self.cache[text] = {}
            self.cache[text]['combined'] = result.copy()

        return result
    
    def get_raw_probabilities(self, model_output) -> Dict[str, float]:
        """
        Extract raw probability scores from model output.
        
        Args:
            model_output: Output from the model inference
            
        Returns:
            Dict mapping labels to probability scores
        """
        scores = torch.nn.functional.softmax(model_output.logits, dim=1).cpu().numpy()[0]
        if len(scores) == len(self.sentiment_labels):
            return {self.sentiment_labels[i]: float(scores[i]) for i in range(len(self.sentiment_labels))}
        else:
            return {self.emotion_labels[i]: float(scores[i]) for i in range(len(self.emotion_labels))}
    
    def set_thresholds(self, sentiment_threshold: Optional[float] = None, emotion_threshold: Optional[float] = None) -> None:
        """
        Update the confidence thresholds.
        
        Args:
            sentiment_threshold: New sentiment confidence threshold
            emotion_threshold: New emotion confidence threshold
        """
        if sentiment_threshold is not None:
            self.sentiment_threshold = sentiment_threshold
        if emotion_threshold is not None:
            self.emotion_threshold = emotion_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model for comparison.
        
        Returns:
            Dict with model metadata
        """
        return {
            'name': self.name,
            'type': 'transformer',
            'sentiment_model': self.sentiment_model.config._name_or_path,
            'emotion_model': self.emotion_model.config._name_or_path,
            'device': str(self.device),
            'sentiment_threshold': self.sentiment_threshold,
            'emotion_threshold': self.emotion_threshold
        }
    
    def map_labels(self, label: str, score: float, is_emotion: bool = False) -> str:
        """
        Apply label mapping with threshold check.
        
        Args:
            label: Original label from the model
            score: Confidence score
            is_emotion: Whether this is an emotion label
            
        Returns:
            Mapped label
        """
        if not self.label_mapper:
            return label
            
        threshold = self.emotion_threshold if is_emotion else self.sentiment_threshold
        if score < threshold:
            # Low confidence, consider returning a 'uncertain' or 'low confidence' label
            if is_emotion:
                return self.label_mapper.get_emotion_label('uncertain')
            else:
                return self.label_mapper.get_sentiment_label('neutral')
                
        # Map the label through the label mapper
        if is_emotion:
            return self.label_mapper.get_emotion_label(label)
        else:
            return self.label_mapper.get_sentiment_label(label)
    
    def set_fallback_system(self, fallback_system) -> None:
        """
        Set the fallback system for low confidence predictions.
        
        Args:
            fallback_system: Fallback system to use
        """
        self.fallback_system = fallback_system
        
    def remove_fallback_system(self) -> None:
        """Remove the fallback system to disable fallback."""
        self.fallback_system = None
        
    def get_confidence_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract confidence metrics from result for evaluation.
        
        Args:
            result: Analysis result
            
        Returns:
            Dict with confidence metrics
        """
        metrics = {}
        
        if 'sentiment' in result:
            metrics['sentiment_confidence'] = result['sentiment']['score']
            metrics['sentiment_label'] = result['sentiment']['label']
            
            # Add positive/neutral/negative probabilities if available
            if 'raw_probabilities' in result['sentiment']:
                for label, prob in result['sentiment']['raw_probabilities'].items():
                    metrics[f'sentiment_prob_{label}'] = prob
                    
        if 'emotion' in result:
            metrics['emotion_confidence'] = result['emotion']['score']
            metrics['emotion_label'] = result['emotion']['label']
            
            # Add emotion probabilities if available
            if 'raw_probabilities' in result['emotion']:
                for label, prob in result['emotion']['raw_probabilities'].items():
                    metrics[f'emotion_prob_{label}'] = prob
            
        return metrics

    def enable_cache(self) -> None:
        """Enable result caching for improved performance."""
        self.use_cache = True
        if self.cache is None:
            self.cache = {}
            
    def disable_cache(self) -> None:
        """Disable result caching."""
        self.use_cache = False
        self.cache = None
        
    def set_cache_options(self, ttl_days: Optional[int] = None, max_size_mb: Optional[int] = None) -> None:
        """
        Configure cache parameters.
        
        Args:
            ttl_days: Time-to-live in days for cache entries
            max_size_mb: Maximum cache size in MB
        """
        # This is a placeholder - a full implementation would need to track entry times and sizes
        pass
    
    def detect_negation_patterns(self, text: str) -> bool:
        """
        Detect common negation patterns in text that indicate negative sentiment.
        
        Args:
            text: The text to analyze
            
        Returns:
            True if negation patterns are found, False otherwise
        """
        for pattern in self.negation_patterns:
            if re.search(pattern, text.lower()):
                return True
        return False
        
    def verify_negative_sentiment(self, text: str, model_output: Dict[str, float]) -> bool:
        """
        Verify if text contains negative sentiment based on negation patterns.
        Useful for catching cases where the model misses negative sentiment.
        
        Args:
            text: The text to analyze
            model_output: Raw probability outputs from the model
        
        Returns:
            True if text likely has negative sentiment that the model missed
        """
        # If the model already predicts negative sentiment with good confidence, no need to verify
        if model_output['negative'] > 0.6:
            return False
            
        # If the text is short and contains negation patterns, it's likely negative
        # This catches cases like "No, I don't like it" which models sometimes misclassify
        if len(text.split()) < 15 and self.detect_negation_patterns(text):
            # Make sure it's not a double negative ("not bad" = positive)
            double_neg_patterns = [
                r'\bnot\s+bad\b', 
                r'\bnot\s+terrible\b', 
                r'\bnot\s+awful\b', 
                r'\bnot\s+horrible\b',
                r'\bnot\s+disappointed\b',
                r'\bnot\s+dissatisfied\b'
            ]
            
            for pattern in double_neg_patterns:
                if re.search(pattern, text.lower()):
                    return False  # Double negative found, likely positive
                    
            return True  # Negation found, likely negative
        
        # Check for strong negative phrases that models sometimes miss
        strong_negative_phrases = [
            r'\bhate\b',
            r'\bterrible\b',
            r'\bawful\b', 
            r'\bhorrible\b',
            r'\bwaste of\b',
            r'\bdisgust(ing|ed)\b',
            r'\bpoor\s+quality\b',
            r'\bdisappointed\b',
            r'\bunacceptable\b',
            r'\bridiculous\b',
            r'\bworthless\b',
        ]
        
        for pattern in strong_negative_phrases:
            if re.search(pattern, text.lower()):
                # If there's a strong negative word but model confidence for negative is very low
                # it's likely the model is wrong
                if model_output['negative'] < 0.3:
                    return True
                    
        return False

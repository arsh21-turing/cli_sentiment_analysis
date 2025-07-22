"""
Advanced emotion detection functionality for the Smart CLI Sentiment & Emotion Analyzer.
"""

from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import os
import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class EmotionIntensity(Enum):
    """Emotion intensity levels."""
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4

class EmotionCategory(Enum):
    """Base emotion categories derived from psychological research."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    LOVE = "love"
    NEUTRAL = "neutral"
    OTHER = "other"

@dataclass
class EmotionResult:
    """Structured result from emotion detection."""
    primary_emotion: EmotionCategory
    primary_score: float
    primary_intensity: EmotionIntensity
    secondary_emotion: Optional[EmotionCategory] = None
    secondary_score: Optional[float] = None
    secondary_intensity: Optional[EmotionIntensity] = None
    all_emotions: Optional[Dict[EmotionCategory, float]] = None
    model_name: Optional[str] = None
    text: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize the result data."""
        # Ensure primary_score is in [0, 1] range
        if not (0 <= self.primary_score <= 1):
            raise ValueError(f"Emotion score must be between 0 and 1, got {self.primary_score}")
        
        # Default all_emotions to empty dict if not provided
        if self.all_emotions is None:
            self.all_emotions = {}
            if self.primary_emotion != EmotionCategory.NEUTRAL:
                self.all_emotions[self.primary_emotion] = self.primary_score
            if self.secondary_emotion and self.secondary_score:
                self.all_emotions[self.secondary_emotion] = self.secondary_score

    def as_dict(self) -> Dict[str, Any]:
        """Convert the emotion result to a dictionary."""
        result = {
            "primary_emotion": self.primary_emotion.value,
            "primary_score": self.primary_score,
            "primary_intensity": self.primary_intensity.name,
            "model": self.model_name
        }
        
        if self.secondary_emotion:
            result["secondary_emotion"] = self.secondary_emotion.value
            result["secondary_score"] = self.secondary_score
            result["secondary_intensity"] = self.secondary_intensity.name if self.secondary_intensity else None
            
        # Include all emotions if available
        if self.all_emotions:
            result["all_emotions"] = {e.value: s for e, s in self.all_emotions.items()}
            
        return result

    def visualize(self, title: str = "Emotion Analysis", min_score: float = 0.05) -> Optional[plt.Figure]:
        """
        Visualize the emotion distribution as a radar chart or bar chart.
        
        Args:
            title: Title for the visualization
            min_score: Minimum score to include in visualization
            
        Returns:
            Matplotlib figure if matplotlib is available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available for visualization")
            return None
            
        if not self.all_emotions:
            return None
            
        # Filter emotions with scores above min_score
        emotions_to_plot = {e.value: s for e, s in self.all_emotions.items() if s >= min_score}
        
        if not emotions_to_plot:
            return None
            
        # Create a figure
        fig = plt.figure(figsize=(10, 6))
        
        # Determine if we have enough emotions for a radar chart
        if len(emotions_to_plot) >= 3:
            # Radar chart for 3+ emotions
            ax = fig.add_subplot(111, polar=True)
            
            # Set up the radar chart
            categories = list(emotions_to_plot.keys())
            N = len(categories)
            
            # Set angles for radar chart
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            values = list(emotions_to_plot.values())
            values += values[:1]  # Close the loop
            
            # Draw the chart
            ax.plot(angles, values, linewidth=1, linestyle='solid')
            ax.fill(angles, values, alpha=0.1)
            
            # Fix axis to start at top
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Set category labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis limits
            ax.set_ylim(0, max(values) * 1.1)
            
            # Add a title
            plt.title(title)
            
        else:
            # Bar chart for 1-2 emotions
            ax = fig.add_subplot(111)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(emotions_to_plot))
            ax.barh(y_pos, list(emotions_to_plot.values()))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(emotions_to_plot.keys()))
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Score')
            ax.set_title(title)
            
        return fig


class EmotionDetector:
    """
    Advanced emotion detection with support for multiple models, 
    confidence scoring, and emotion mapping.
    """
    
    # Common emotion mappings between different models
    # This helps normalize outputs from different emotion detection models
    EMOTION_MAPPING = {
        # HuggingFace bhadresh-savani/distilbert-base-uncased-emotion
        "distilbert-emotion": {
            "sadness": EmotionCategory.SADNESS,
            "joy": EmotionCategory.JOY,
            "love": EmotionCategory.LOVE,
            "anger": EmotionCategory.ANGER,
            "fear": EmotionCategory.FEAR,
            "surprise": EmotionCategory.SURPRISE,
        },
        # HuggingFace arpanghoshal/EmoRoBERTa
        "emoroberta": {
            "sadness": EmotionCategory.SADNESS,
            "joy": EmotionCategory.JOY,
            "love": EmotionCategory.LOVE,
            "anger": EmotionCategory.ANGER,
            "fear": EmotionCategory.FEAR,
            "surprise": EmotionCategory.SURPRISE,
            "disgust": EmotionCategory.DISGUST,
            "anticipation": EmotionCategory.ANTICIPATION,
            "trust": EmotionCategory.TRUST,
        },
        # General GoEmotion mapping (Google's emotion taxonomy)
        "goemotion": {
            "admiration": EmotionCategory.TRUST,
            "amusement": EmotionCategory.JOY,
            "anger": EmotionCategory.ANGER,
            "annoyance": EmotionCategory.ANGER,
            "approval": EmotionCategory.TRUST,
            "caring": EmotionCategory.LOVE,
            "confusion": EmotionCategory.SURPRISE,
            "curiosity": EmotionCategory.ANTICIPATION,
            "desire": EmotionCategory.ANTICIPATION,
            "disappointment": EmotionCategory.SADNESS,
            "disapproval": EmotionCategory.DISGUST,
            "disgust": EmotionCategory.DISGUST,
            "embarrassment": EmotionCategory.FEAR,
            "excitement": EmotionCategory.JOY,
            "fear": EmotionCategory.FEAR,
            "gratitude": EmotionCategory.JOY,
            "grief": EmotionCategory.SADNESS,
            "joy": EmotionCategory.JOY,
            "love": EmotionCategory.LOVE,
            "nervousness": EmotionCategory.FEAR,
            "optimism": EmotionCategory.ANTICIPATION,
            "pride": EmotionCategory.JOY,
            "realization": EmotionCategory.SURPRISE,
            "relief": EmotionCategory.JOY,
            "remorse": EmotionCategory.SADNESS,
            "sadness": EmotionCategory.SADNESS,
            "surprise": EmotionCategory.SURPRISE,
            "neutral": EmotionCategory.NEUTRAL,
        }
    }
    
    # Intensity thresholds for different emotion levels
    INTENSITY_THRESHOLDS = {
        EmotionIntensity.NONE: 0.0,
        EmotionIntensity.WEAK: 0.25,
        EmotionIntensity.MODERATE: 0.5, 
        EmotionIntensity.STRONG: 0.75,
        EmotionIntensity.EXTREME: 0.9
    }
    
    def __init__(
        self,
        model_name: str = "distilbert-emotion",
        threshold: float = 0.5,
        secondary_threshold: float = 0.3,
        intensity_thresholds: Optional[Dict[EmotionIntensity, float]] = None,
        emotion_mapping: Optional[Dict[str, EmotionCategory]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the emotion detector with the specified model and configuration.
        
        Args:
            model_name: Name of the emotion model to use
                ("distilbert-emotion" or "emoroberta" for pre-configured mappings)
            threshold: Minimum confidence threshold for primary emotion
            secondary_threshold: Minimum confidence for secondary emotion
            intensity_thresholds: Custom thresholds for intensity levels
            emotion_mapping: Custom mapping from model outputs to EmotionCategory
            device: Device to run inference on (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold
        self.device = device or self._detect_device()
        
        # Use provided intensity thresholds or defaults
        self.intensity_thresholds = intensity_thresholds or self.INTENSITY_THRESHOLDS.copy()
        
        # Configure emotion mapping based on model_name or custom mapping
        self.emotion_mapping = emotion_mapping
        if not self.emotion_mapping:
            # Try to use pre-configured mapping
            if model_name in self.EMOTION_MAPPING:
                self.emotion_mapping = self.EMOTION_MAPPING[model_name]
            else:
                # Default to distilbert mapping if model not recognized
                logging.warning(f"No pre-configured mapping for model '{model_name}', using default mapping")
                self.emotion_mapping = self.EMOTION_MAPPING["distilbert-emotion"]
        
        # Initialize the model
        self._initialize_model()
        
    def _detect_device(self) -> str:
        """Detect the best available device for inference."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
        
    def _initialize_model(self):
        """Initialize the emotion detection model."""
        try:
            from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
            
            logging.info(f"Loading emotion model: {self.model_name}")
            
            # Handle different model initialization based on model_name
            if self.model_name == "distilbert-emotion":
                # Pre-configured model path
                model_path = "bhadresh-savani/distilbert-base-uncased-emotion"
                self.model = pipeline(
                    task="text-classification",
                    model=model_path,
                    device=self.device
                )
            elif self.model_name == "emoroberta":
                # Pre-configured model path
                model_path = "arpanghoshal/EmoRoBERTa"
                self.model = pipeline(
                    task="text-classification", 
                    model=model_path,
                    device=self.device
                )
            else:
                # Custom model path
                self.model = pipeline(
                    task="text-classification",
                    model=self.model_name,
                    device=self.device
                )
                
            logging.info(f"Emotion model loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Error loading emotion model: {str(e)}")
            raise

    def _get_emotion_intensity(self, score: float) -> EmotionIntensity:
        """
        Determine emotion intensity based on confidence score.
        
        Args:
            score: Confidence score from the model
            
        Returns:
            EmotionIntensity enum value
        """
        if score < self.intensity_thresholds[EmotionIntensity.WEAK]:
            return EmotionIntensity.NONE
        elif score < self.intensity_thresholds[EmotionIntensity.MODERATE]:
            return EmotionIntensity.WEAK
        elif score < self.intensity_thresholds[EmotionIntensity.STRONG]:
            return EmotionIntensity.MODERATE
        elif score < self.intensity_thresholds[EmotionIntensity.EXTREME]:
            return EmotionIntensity.STRONG
        else:
            return EmotionIntensity.EXTREME

    def _map_emotion(self, emotion_label: str) -> EmotionCategory:
        """
        Map model-specific emotion label to standardized EmotionCategory.
        
        Args:
            emotion_label: Raw emotion label from the model
            
        Returns:
            Mapped EmotionCategory
        """
        # Handle case sensitivity by converting to lowercase
        emotion_label_lower = emotion_label.lower()
        
        # Try to find a direct mapping
        if emotion_label_lower in self.emotion_mapping:
            return self.emotion_mapping[emotion_label_lower]
            
        # Try to find partial matches
        for key, category in self.emotion_mapping.items():
            if key in emotion_label_lower or emotion_label_lower in key:
                return category
                
        # Fallback to OTHER if no mapping found
        logging.warning(f"Unknown emotion '{emotion_label}' couldn't be mapped")
        return EmotionCategory.OTHER

    def detect_emotion_pipeline(self, text: str) -> EmotionResult:
        """
        Detect emotions using the HuggingFace pipeline.
        
        Args:
            text: The text to analyze
            
        Returns:
            EmotionResult with primary and secondary emotions
        """
        if not text.strip():
            return EmotionResult(
                primary_emotion=EmotionCategory.NEUTRAL,
                primary_score=1.0,
                primary_intensity=EmotionIntensity.NONE,
                model_name=self.model_name,
                text=text
            )
        
        try:
            # Get raw prediction from the model
            predictions = self.model(text)
            
            # Process all emotion predictions
            all_emotions = {}
            for pred in predictions:
                emotion_label = pred["label"]
                score = pred["score"]
                emotion_category = self._map_emotion(emotion_label)
                
                # Aggregate scores for the same emotion category
                if emotion_category in all_emotions:
                    all_emotions[emotion_category] = max(all_emotions[emotion_category], score)
                else:
                    all_emotions[emotion_category] = score
            
            # Sort emotions by score
            sorted_emotions = sorted(
                all_emotions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # If no emotions above threshold, return neutral
            if not sorted_emotions or sorted_emotions[0][1] < self.threshold:
                return EmotionResult(
                    primary_emotion=EmotionCategory.NEUTRAL,
                    primary_score=1.0,
                    primary_intensity=EmotionIntensity.NONE,
                    all_emotions=all_emotions,
                    model_name=self.model_name,
                    text=text
                )
            
            # Get primary emotion
            primary_emotion, primary_score = sorted_emotions[0]
            primary_intensity = self._get_emotion_intensity(primary_score)
            
            # Get secondary emotion if available and above threshold
            secondary_emotion = None
            secondary_score = None
            secondary_intensity = None
            
            if len(sorted_emotions) > 1 and sorted_emotions[1][1] >= self.secondary_threshold:
                secondary_emotion, secondary_score = sorted_emotions[1]
                secondary_intensity = self._get_emotion_intensity(secondary_score)
            
            return EmotionResult(
                primary_emotion=primary_emotion,
                primary_score=primary_score,
                primary_intensity=primary_intensity,
                secondary_emotion=secondary_emotion,
                secondary_score=secondary_score,
                secondary_intensity=secondary_intensity,
                all_emotions=all_emotions,
                model_name=self.model_name,
                text=text
            )
            
        except Exception as e:
            logging.error(f"Error in emotion detection: {str(e)}")
            
            # Fallback to neutral on error
            return EmotionResult(
                primary_emotion=EmotionCategory.NEUTRAL,
                primary_score=0.0,
                primary_intensity=EmotionIntensity.NONE,
                model_name=self.model_name,
                text=text
            )

    def detect_emotion(self, text: str) -> EmotionResult:
        """
        Main method to detect emotions in text.
        Wrapper around the pipeline implementation for future extensibility.
        
        Args:
            text: The text to analyze
            
        Returns:
            EmotionResult with detected emotions
        """
        return self.detect_emotion_pipeline(text)
    
    def detect_emotion_batch(self, texts: List[str]) -> List[EmotionResult]:
        """
        Detect emotions in a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of EmotionResult, one for each text
        """
        return [self.detect_emotion(text) for text in texts]


# Utility functions
def get_emotional_valence(emotion_result: EmotionResult) -> float:
    """
    Calculate the emotional valence (positivity/negativity) from emotion result.
    
    Args:
        emotion_result: The emotion detection result
        
    Returns:
        A value between -1.0 (very negative) and 1.0 (very positive)
    """
    # Emotion valence map (psychological research based)
    valence_map = {
        EmotionCategory.JOY: 0.8,
        EmotionCategory.LOVE: 0.9,
        EmotionCategory.TRUST: 0.6,
        EmotionCategory.ANTICIPATION: 0.4,
        EmotionCategory.SURPRISE: 0.1,  # Surprise can be positive or negative
        EmotionCategory.NEUTRAL: 0.0,
        EmotionCategory.FEAR: -0.7,
        EmotionCategory.SADNESS: -0.8,
        EmotionCategory.ANGER: -0.9,
        EmotionCategory.DISGUST: -0.6,
        EmotionCategory.OTHER: 0.0
    }
    
    # Calculate weighted valence from all emotions
    total_score = 0.0
    weighted_valence = 0.0
    
    for emotion, score in emotion_result.all_emotions.items():
        valence = valence_map.get(emotion, 0.0)
        weighted_valence += valence * score
        total_score += score
        
    # If no emotions detected, return neutral valence
    if total_score == 0:
        return 0.0
        
    return weighted_valence / total_score


def get_emotion_description(emotion_result: EmotionResult) -> str:
    """
    Generate a human-readable description of the emotion result.
    
    Args:
        emotion_result: The emotion detection result
        
    Returns:
        A natural language description of the emotional content
    """
    if emotion_result.primary_emotion == EmotionCategory.NEUTRAL:
        return "The text appears to be emotionally neutral."
    
    # Intensity descriptions
    intensity_desc = {
        EmotionIntensity.WEAK: "slight",
        EmotionIntensity.MODERATE: "moderate",
        EmotionIntensity.STRONG: "strong",
        EmotionIntensity.EXTREME: "extreme"
    }
    
    primary_desc = intensity_desc.get(emotion_result.primary_intensity, "")
    
    # Basic description with primary emotion
    description = f"The text expresses {primary_desc} {emotion_result.primary_emotion.value}"
    
    # Add secondary emotion if present
    if emotion_result.secondary_emotion and emotion_result.secondary_intensity:
        secondary_desc = intensity_desc.get(emotion_result.secondary_intensity, "")
        description += f" with {secondary_desc} {emotion_result.secondary_emotion.value}"
        
    # Add valence information
    valence = get_emotional_valence(emotion_result)
    if valence > 0.5:
        description += ", indicating an overall very positive emotional tone."
    elif valence > 0.1:
        description += ", indicating a somewhat positive emotional tone."
    elif valence < -0.5:
        description += ", indicating an overall very negative emotional tone."
    elif valence < -0.1:
        description += ", indicating a somewhat negative emotional tone."
    else:
        description += ", with a relatively neutral overall tone."
        
    return description


def analyze_emotion(text: str, threshold: float = 0.3, use_fallback: bool = False) -> Dict[str, Any]:
    """
    Analyze emotion of a text using the emotion detector.
    
    Args:
        text (str): Text to analyze
        threshold (float): Confidence threshold
        use_fallback (bool): Whether to use API fallback
        
    Returns:
        Dict: Emotion analysis results
    """
    try:
        # Create emotion detector instance
        detector = EmotionDetector(threshold=threshold)
        
        # Analyze the text
        result = detector.detect_emotion(text)
        
        # Convert to dictionary format
        emotion_data = result.as_dict()
        
        # Add prediction and confidence for compatibility
        emotion_data["prediction"] = result.primary_emotion.value
        emotion_data["confidence"] = result.primary_score
        
        # Add scores for compatibility
        if result.all_emotions:
            emotion_data["scores"] = {e.value: s for e, s in result.all_emotions.items()}
        else:
            # Create basic scores if all_emotions is not available
            emotion_data["scores"] = {
                result.primary_emotion.value: result.primary_score
            }
            if result.secondary_emotion and result.secondary_score:
                emotion_data["scores"][result.secondary_emotion.value] = result.secondary_score
        
        # Add top emotions for compatibility
        if result.all_emotions:
            sorted_emotions = sorted(result.all_emotions.items(), key=lambda x: x[1], reverse=True)
            emotion_data["top_emotions"] = [(e.value, s) for e, s in sorted_emotions[:3]]
        else:
            emotion_data["top_emotions"] = [(result.primary_emotion.value, result.primary_score)]
            if result.secondary_emotion and result.secondary_score:
                emotion_data["top_emotions"].append((result.secondary_emotion.value, result.secondary_score))
        
        return emotion_data
        
    except Exception as e:
        # Fallback to simple analysis if detector fails
        if use_fallback:
            # Simple rule-based fallback
            emotion_keywords = {
                "joy": ["happy", "joy", "excited", "pleased", "delighted", "cheerful"],
                "sadness": ["sad", "depressed", "melancholy", "gloomy", "sorrowful"],
                "anger": ["angry", "mad", "furious", "irritated", "annoyed"],
                "fear": ["afraid", "scared", "terrified", "anxious", "worried"],
                "surprise": ["surprised", "shocked", "amazed", "astonished"],
                "disgust": ["disgusted", "revolted", "appalled", "sickened"],
                "trust": ["trust", "confident", "reliable", "faithful"],
                "anticipation": ["excited", "eager", "hopeful", "expectant"]
            }
            
            text_lower = text.lower()
            emotion_scores = {}
            
            for emotion, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    emotion_scores[emotion] = min(0.8, score * 0.2)
            
            if emotion_scores:
                primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                prediction = primary_emotion[0]
                confidence = primary_emotion[1]
            else:
                prediction = "neutral"
                confidence = 0.5
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "scores": emotion_scores,
                "top_emotions": [(emotion, score) for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)]
            }
        else:
            raise e

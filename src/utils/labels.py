from __future__ import annotations

"""Utilities for standardized sentiment & emotion label handling.

This module centralises all label-related utilities so the rest of the
code-base can operate on a single canonical representation regardless of the
original model output.  It also contains helper logic for confidence /
threshold handling plus colour selection which is used by the output layer.
"""

from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style


class SentimentLabels:
    """Standardised sentiment labels and helpers."""

    # Canonical sentiment constants
    POSITIVE: str = "positive"
    NEUTRAL: str = "neutral"
    NEGATIVE: str = "negative"

    # Mapping of raw model outputs to canonical form
    MODEL_MAPPINGS: Dict[str, str] = {
        # Star-rating datasets
        "1 star": NEGATIVE,
        "2 stars": NEGATIVE,
        "3 stars": NEUTRAL,
        "4 stars": POSITIVE,
        "5 stars": POSITIVE,
        # Direct
        "positive": POSITIVE,
        "neutral": NEUTRAL,
        "negative": NEGATIVE,
        # Abbreviations
        "pos": POSITIVE,
        "neu": NEUTRAL,
        "neg": NEGATIVE,
    }

    LABELS: Dict[str, str] = {
        POSITIVE: "Positive",
        NEUTRAL: "Neutral",
        NEGATIVE: "Negative",
    }

    DESCRIPTIONS: Dict[str, str] = {
        POSITIVE: "The text expresses a positive sentiment, indicating approval, happiness, or satisfaction.",
        NEUTRAL: "The text expresses a neutral sentiment, without strong positive or negative feelings.",
        NEGATIVE: "The text expresses a negative sentiment, indicating disapproval, unhappiness, or dissatisfaction.",
    }

    COLORS: Dict[str, str] = {
        POSITIVE: Fore.GREEN,
        NEUTRAL: Fore.BLUE,
        NEGATIVE: Fore.RED,
    }

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    @classmethod
    def get_all_labels(cls) -> List[str]:
        return [cls.POSITIVE, cls.NEUTRAL, cls.NEGATIVE]

    @classmethod
    def get_label(cls, raw_label: str) -> str:
        return cls.MODEL_MAPPINGS.get(raw_label.lower().strip(), cls.NEUTRAL)

    @classmethod
    def is_valid_label(cls, label: str) -> bool:
        return label.lower() in cls.get_all_labels()

    @classmethod
    def get_name(cls, label: str) -> str:
        return cls.LABELS.get(label.lower(), "Unknown")

    @classmethod
    def get_description(cls, label: str, confidence: Optional[str] = None) -> str:
        label = label.lower()
        base = cls.DESCRIPTIONS.get(label, "Unknown sentiment")
        if confidence:
            ctx_map = {
                "high": f"There is strong evidence of {label} sentiment in the text.",
                "medium": f"There is moderate evidence of {label} sentiment in the text.",
                "low": f"There is weak evidence of {label} sentiment in the text.",
            }
            return f"{base} {ctx_map.get(confidence.lower(), '')}"
        return base

    @classmethod
    def get_label_color(cls, label: str) -> str:
        return cls.COLORS.get(label.lower(), Fore.WHITE)


class EmotionLabels:
    """Standardised emotion labels and helpers."""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    LOVE = "love"

    MODEL_MAPPINGS: Dict[str, str] = {
        # Direct
        "joy": JOY,
        "sadness": SADNESS,
        "anger": ANGER,
        "fear": FEAR,
        "surprise": SURPRISE,
        "love": LOVE,
        # Variants
        "happy": JOY,
        "sad": SADNESS,
        "mad": ANGER,
        "scared": FEAR,
        "shocked": SURPRISE,
        "loving": LOVE,
        "excited": JOY,
        "depressed": SADNESS,
        "furious": ANGER,
        "anxious": FEAR,
        "amazed": SURPRISE,
        "affection": LOVE,
    }

    LABELS: Dict[str, str] = {
        JOY: "Joy",
        SADNESS: "Sadness",
        ANGER: "Anger",
        FEAR: "Fear",
        SURPRISE: "Surprise",
        LOVE: "Love",
    }

    DESCRIPTIONS: Dict[str, str] = {
        JOY: "The text expresses joy, happiness, or excitement.",
        SADNESS: "The text expresses sadness, grief, or disappointment.",
        ANGER: "The text expresses anger, frustration, or annoyance.",
        FEAR: "The text expresses fear, anxiety, or worry.",
        SURPRISE: "The text expresses surprise, astonishment, or shock.",
        LOVE: "The text expresses love, affection, or attachment.",
    }

    COLORS: Dict[str, str] = {
        JOY: Fore.YELLOW,
        SADNESS: Fore.BLUE,
        ANGER: Fore.RED,
        FEAR: Fore.MAGENTA,
        SURPRISE: Fore.CYAN,
        LOVE: Fore.LIGHTMAGENTA_EX,
    }

    @classmethod
    def get_all_labels(cls) -> List[str]:
        return [cls.JOY, cls.SADNESS, cls.ANGER, cls.FEAR, cls.SURPRISE, cls.LOVE]

    @classmethod
    def get_label(cls, raw_label: str) -> str:
        return cls.MODEL_MAPPINGS.get(raw_label.lower().strip(), cls.SADNESS)

    @classmethod
    def is_valid_label(cls, label: str) -> bool:
        return label.lower() in cls.get_all_labels()

    @classmethod
    def get_name(cls, label: str) -> str:
        return cls.LABELS.get(label.lower(), "Unknown")

    @classmethod
    def get_description(cls, label: str, confidence: Optional[str] = None) -> str:
        label = label.lower()
        base = cls.DESCRIPTIONS.get(label, "Unknown emotion")
        if confidence:
            ctx_map = {
                "high": f"There is strong evidence of {label} in the text.",
                "medium": f"There is moderate evidence of {label} in the text.",
                "low": f"There is slight evidence of {label} in the text.",
            }
            return f"{base} {ctx_map.get(confidence.lower(), '')}"
        return base

    @classmethod
    def get_label_color(cls, label: str) -> str:
        return cls.COLORS.get(label.lower(), Fore.WHITE)


class LabelMapper:
    """High-level API for mapping raw model outputs → canonical labels plus helpers."""

    def __init__(self, settings: Optional["Settings"] = None):
        # Avoid circular import – annotate type only
        self.settings = settings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _choose_threshold(self, is_emotion: bool, custom: Optional[float]) -> float:
        if custom is not None:
            return custom
        if self.settings:
            return self.settings.get_threshold(is_emotion=is_emotion)
        # sensible defaults
        return 0.4 if is_emotion else 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def map_sentiment_label(self, raw_label: str, score: float,
                             threshold: Optional[float] = None) -> Tuple[str, bool]:
        label = SentimentLabels.get_label(raw_label)
        thr = self._choose_threshold(False, threshold)
        return label, score >= thr

    def map_emotion_label(self, raw_label: str, score: float,
                           threshold: Optional[float] = None) -> Tuple[str, bool]:
        label = EmotionLabels.get_label(raw_label)
        thr = self._choose_threshold(True, threshold)
        return label, score >= thr

    # Colour helpers ---------------------------------------------------
    def get_sentiment_color(self, label: str, score: float) -> str:
        if self.settings:
            return self.settings.get_color_for_label(label, score, is_emotion=False)
        return SentimentLabels.get_label_color(label)

    def get_emotion_color(self, label: str, score: float) -> str:
        if self.settings:
            return self.settings.get_color_for_label(label, score, is_emotion=True)
        return EmotionLabels.get_label_color(label)

    # Confidence helpers -----------------------------------------------
    def get_confidence_level(self, score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
        if thresholds is None:
            if self.settings:
                thresholds = self.settings.get_sentiment_threshold_levels()
            else:
                thresholds = {"high": 0.8, "medium": 0.6, "low": 0.4}
        if score >= thresholds["high"]:
            return "high"
        if score >= thresholds["medium"]:
            return "medium"
        return "low"

    def format_with_confidence(self, label: str, score: float, *, is_emotion: bool = False) -> str:
        human_label = (EmotionLabels.get_name(label) if is_emotion else SentimentLabels.get_name(label))
        colour = self.get_emotion_color(label, score) if is_emotion else self.get_sentiment_color(label, score)
        thresholds = (self.settings.get_emotion_threshold_levels() if (self.settings and is_emotion)
                      else self.settings.get_sentiment_threshold_levels() if self.settings else None)
        conf = self.get_confidence_level(score, thresholds)
        stars = {"high": "★★★", "medium": "★★☆", "low": "★☆☆"}[conf]
        return f"{colour}{human_label} {stars}{Style.RESET_ALL} ({score:.2f})"

    def get_description(self, label: str, score: float, *, is_emotion: bool = False) -> str:
        thresholds = (self.settings.get_emotion_threshold_levels() if (self.settings and is_emotion)
                      else self.settings.get_sentiment_threshold_levels() if self.settings else None)
        conf = self.get_confidence_level(score, thresholds)
        return (EmotionLabels.get_description(label, conf) if is_emotion
                else SentimentLabels.get_description(label, conf))

    # Misc --------------------------------------------------------------
    def get_label_type(self, label: str) -> str:
        if SentimentLabels.is_valid_label(label):
            return "sentiment"
        if EmotionLabels.is_valid_label(label):
            return "emotion"
        return "unknown"

    def get_threshold_for_label(self, label: str) -> float:
        typ = self.get_label_type(label)
        if self.settings:
            return (self.settings.get_sentiment_threshold() if typ == "sentiment"
                    else self.settings.get_emotion_threshold() if typ == "emotion" else 0.5)
        return 0.4 if typ == "emotion" else 0.5 
"""Application settings & colour configuration.

All user-configurable values live here.  They can be persisted to a JSON file
(~/.sentiment_cli_settings.json by default) and loaded on start-up.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from colorama import Fore


class Settings:
    """Central configuration object used across the application."""

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    DEFAULT_SENTIMENT_THRESHOLD: float = 0.5
    DEFAULT_EMOTION_THRESHOLD: float = 0.4

    SENTIMENT_THRESHOLD_LEVELS: Dict[str, float] = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4,
    }
    EMOTION_THRESHOLD_LEVELS: Dict[str, float] = {
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3,
    }

    # Basic ANSI colour scheme incl. intensity variations
    COLOR_SCHEME: Dict[str, str] = {
        # Sentiment
        "positive_high": Fore.LIGHTGREEN_EX,
        "positive_medium": Fore.GREEN,
        "positive_low": Fore.GREEN,
        "neutral_high": Fore.LIGHTBLUE_EX,
        "neutral_medium": Fore.BLUE,
        "neutral_low": Fore.BLUE,
        "negative_high": Fore.LIGHTRED_EX,
        "negative_medium": Fore.RED,
        "negative_low": Fore.RED,
        # Emotions
        "joy_high": Fore.LIGHTYELLOW_EX,
        "joy_medium": Fore.YELLOW,
        "joy_low": Fore.YELLOW,
        "sadness_high": Fore.LIGHTBLUE_EX,
        "sadness_medium": Fore.BLUE,
        "sadness_low": Fore.BLUE,
        "anger_high": Fore.LIGHTRED_EX,
        "anger_medium": Fore.RED,
        "anger_low": Fore.RED,
        "fear_high": Fore.LIGHTMAGENTA_EX,
        "fear_medium": Fore.MAGENTA,
        "fear_low": Fore.MAGENTA,
        "surprise_high": Fore.LIGHTCYAN_EX,
        "surprise_medium": Fore.CYAN,
        "surprise_low": Fore.CYAN,
        "love_high": Fore.LIGHTMAGENTA_EX,
        "love_medium": Fore.MAGENTA,
        "love_low": Fore.MAGENTA,
    }

    # ------------------------------------------------------------------
    def __init__(self, config_file: Optional[str] = None) -> None:
        # runtime values
        self.sentiment_threshold: float = self.DEFAULT_SENTIMENT_THRESHOLD
        self.emotion_threshold: float = self.DEFAULT_EMOTION_THRESHOLD
        self.sentiment_threshold_levels: Dict[str, float] = self.SENTIMENT_THRESHOLD_LEVELS.copy()
        self.emotion_threshold_levels: Dict[str, float] = self.EMOTION_THRESHOLD_LEVELS.copy()
        self.color_scheme: Dict[str, str] = self.COLOR_SCHEME.copy()
        self._summary_only: bool = False
        self._quiet_mode: bool = False
        self._json_stream: bool = False

        if config_file:
            self.load_settings(config_file)

    # ------------------------------------------------------------------
    # Threshold getters/setters
    # ------------------------------------------------------------------
    def get_sentiment_threshold(self) -> float:
        return self.sentiment_threshold

    def get_emotion_threshold(self) -> float:
        return self.emotion_threshold

    def set_sentiment_threshold(self, value: float) -> "Settings":
        if self._valid_threshold(value):
            self.sentiment_threshold = value
        return self

    def set_emotion_threshold(self, value: float) -> "Settings":
        if self._valid_threshold(value):
            self.emotion_threshold = value
        return self

    # Multi-level thresholds -------------------------------------------
    def get_sentiment_threshold_levels(self) -> Dict[str, float]:
        return self.sentiment_threshold_levels.copy()

    def get_emotion_threshold_levels(self) -> Dict[str, float]:
        return self.emotion_threshold_levels.copy()

    def set_sentiment_threshold_levels(self, high: float, medium: float, low: float) -> "Settings":
        if self._valid_multi_level(high, medium, low):
            self.sentiment_threshold_levels = {"high": high, "medium": medium, "low": low}
        return self

    def set_emotion_threshold_levels(self, high: float, medium: float, low: float) -> "Settings":
        if self._valid_multi_level(high, medium, low):
            self.emotion_threshold_levels = {"high": high, "medium": medium, "low": low}
        return self

    # ------------------------------------------------------------------
    # Summary-only helpers
    # ------------------------------------------------------------------
    def set_summary_only(self, summary_only: bool) -> "Settings":
        self._summary_only = bool(summary_only)
        return self

    @property
    def summary_only(self) -> bool:  # noqa: D401
        return self._summary_only

    # ------------------------------------------------------------------
    # Quiet-mode helpers
    # ------------------------------------------------------------------
    def set_quiet_mode(self, quiet: bool) -> "Settings":
        """Enable or disable quiet-mode output suppression."""
        self._quiet_mode = bool(quiet)
        return self

    @property
    def is_quiet(self) -> bool:  # noqa: D401
        return self._quiet_mode

    # ------------------------------------------------------------------
    # JSON-stream helpers
    # ------------------------------------------------------------------
    def set_json_stream(self, json_stream: bool) -> "Settings":
        """Enable or disable NDJSON streaming mode."""
        self._json_stream = bool(json_stream)
        return self

    @property
    def json_stream(self) -> bool:  # noqa: D401
        return self._json_stream

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------
    def get_color_for_label(self, label: str, score: float, *, is_emotion: bool = False) -> str:
        levels = self.emotion_threshold_levels if is_emotion else self.sentiment_threshold_levels
        intensity = self._get_intensity(score, levels)
        return self.color_scheme.get(f"{label.lower()}_{intensity}", Fore.WHITE)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load_settings(self, file_path: str) -> "Settings":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self  # keep defaults

        self.set_sentiment_threshold(float(data.get("sentiment_threshold", self.sentiment_threshold)))
        self.set_emotion_threshold(float(data.get("emotion_threshold", self.emotion_threshold)))

        if (stl := data.get("sentiment_threshold_levels")) and isinstance(stl, dict):
            self.set_sentiment_threshold_levels(
                stl.get("high", self.sentiment_threshold_levels["high"]),
                stl.get("medium", self.sentiment_threshold_levels["medium"]),
                stl.get("low", self.sentiment_threshold_levels["low"]),
            )
        if (etl := data.get("emotion_threshold_levels")) and isinstance(etl, dict):
            self.set_emotion_threshold_levels(
                etl.get("high", self.emotion_threshold_levels["high"]),
                etl.get("medium", self.emotion_threshold_levels["medium"]),
                etl.get("low", self.emotion_threshold_levels["low"]),
            )

        if isinstance(data.get("color_scheme"), dict):
            self.color_scheme.update(data["color_scheme"])
        return self

    def save_settings(self, file_path: Optional[str] = None) -> bool:
        path = file_path or os.path.expanduser("~/.sentiment_cli_settings.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sentiment_threshold": self.sentiment_threshold,
                        "emotion_threshold": self.emotion_threshold,
                        "sentiment_threshold_levels": self.sentiment_threshold_levels,
                        "emotion_threshold_levels": self.emotion_threshold_levels,
                        "color_scheme": self.color_scheme,
                    },
                    f,
                    indent=4,
                )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def get_threshold(self, *, is_emotion: bool = False) -> float:
        return self.emotion_threshold if is_emotion else self.sentiment_threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_threshold(value: float) -> bool:
        return 0.0 <= value <= 1.0

    @staticmethod
    def _valid_multi_level(high: float, medium: float, low: float) -> bool:
        return all(0.0 <= v <= 1.0 for v in (high, medium, low)) and high >= medium >= low

    @staticmethod
    def _get_intensity(score: float, levels: Dict[str, float]) -> str:
        if score >= levels["high"]:
            return "high"
        if score >= levels["medium"]:
            return "medium"
        return "low"

    # chaining helper
    def reset_to_defaults(self) -> "Settings":
        self.__init__()
        return self 

    # ------------------------------------------------------------------
    # Colour usage flag
    # ------------------------------------------------------------------
    @property
    def use_color(self) -> bool:
        """Return *True* if ANSI colour codes are enabled."""
        import os as _os
        if _os.getenv("ANSI_COLORS_DISABLED") == "1":
            return False
        return getattr(Fore, "GREEN", "") != "" 
#  src/utils/output.py
"""Rich output utilities with threshold-aware colouring & confidence indicators.

This module keeps the *same public API* (`format_analysis_result`,
`create_progress_bar`, `export_to_json`, `export_to_csv`, …) relied upon by the
CLI while internally delegating the heavy lifting to the new
:class:`OutputFormatter`.  The formatter leverages :pyclass:`utils.settings.Settings`
for configurable thresholds/colours and :pyclass:`utils.labels.LabelMapper` for
label normalisation.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import json
import csv
import os
import io
from colorama import Fore, Style

# Local imports
from .settings import Settings
from .labels import (
    SentimentLabels,
    EmotionLabels,
    LabelMapper,
)

###############################################################################
# Internal helpers / singletons
###############################################################################

_SETTINGS: Settings = Settings()
_LABEL_MAPPER: LabelMapper = LabelMapper(_SETTINGS)


class OutputFormatter:
    """Human friendly text/colour rendering for analysis results."""

    def __init__(self, label_mapper: LabelMapper, settings: Settings):
        self.label_mapper = label_mapper
        self.settings = settings

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def _header(self, text: str) -> str:
        return f"{Fore.CYAN}{text}{Style.RESET_ALL}"

    # ------------------------------------------------------------------
    # Probability distribution helpers
    # ------------------------------------------------------------------
    def _probabilities_block(self, probs: Dict[str, float]) -> str:
        """Return a human-readable probability distribution block.

        Values are displayed as percentage strings (one decimal place).  The
        block length is fixed to 20 characters to keep the output consistent
        with the unit-tests.
        """
        if not probs:
            return "No probability data available"

        lines: List[str] = ["Probability Distribution:"]
        for label, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
            bar_len = int(p * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            is_emotion = EmotionLabels.is_valid_label(label)
            colour = (
                self.label_mapper.get_emotion_color(label, p)
                if is_emotion
                else self.label_mapper.get_sentiment_color(label, p)
            )
            name = (
                EmotionLabels.get_name(label) if is_emotion else SentimentLabels.get_name(label)
            )
            lines.append(f"{name.ljust(12)}: {bar} {colour}{p*100:.1f}%{Style.RESET_ALL}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sentiment & emotion blocks
    # ------------------------------------------------------------------
    def _threshold_warning(self, label: str, score: float, threshold: float) -> str:
        return (
            f"{Fore.YELLOW}Low confidence result: {label.title()} ({score:.2f}){Style.RESET_ALL}\n"
            f"Note: This result is below the confidence threshold ({threshold:.2f})."
        )

    def format_sentiment_result(self, result: Dict[str, Any], show_probabilities: bool = False) -> str:
        label: str = result.get("label", "unknown")
        score: float = float(result.get("score", 0.0))
        probs: Dict[str, float] = result.get("raw_probabilities", {})

        threshold = self.settings.get_sentiment_threshold()
        if score < threshold:
            return self._threshold_warning(label, score, threshold)

        formatted_label = self.label_mapper.format_with_confidence(label, score, is_emotion=False)
        description = self.label_mapper.get_description(label, score, is_emotion=False)

        parts: List[str] = [
            self._header("Sentiment Analysis"),
            f"Result: {formatted_label}",
            f"Description: {description}",
        ]
        if show_probabilities and probs:
            parts.append("")
            parts.append(self._probabilities_block(probs))
        return "\n".join(parts)

    def format_emotion_result(self, result: Dict[str, Any], show_probabilities: bool = False) -> str:
        label: str = result.get("label", "unknown")
        score: float = float(result.get("score", 0.0))
        probs: Dict[str, float] = result.get("raw_probabilities", {})

        threshold = self.settings.get_emotion_threshold()
        if score < threshold:
            return self._threshold_warning(label, score, threshold)

        formatted_label = self.label_mapper.format_with_confidence(label, score, is_emotion=True)
        description = self.label_mapper.get_description(label, score, is_emotion=True)

        parts: List[str] = [
            self._header("Emotion Analysis"),
            f"Result: {formatted_label}",
            f"Description: {description}",
        ]
        if show_probabilities and probs:
            parts.append("")
            parts.append(self._probabilities_block(probs))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Combined helpers
    # ------------------------------------------------------------------
    def format_analysis_result(self, result: Dict[str, Any], show_probabilities: bool = False) -> str:
        if "sentiment" not in result or "emotion" not in result:
            return "Invalid analysis result format"

        # Summary-only fast path
        if getattr(self.settings, "summary_only", False):
            return self._summary(result, include_header=False)

        sentiment_block = self.format_sentiment_result(result["sentiment"], show_probabilities)
        emotion_block = self.format_emotion_result(result["emotion"], show_probabilities)
        summary = self._summary(result)
        return f"{sentiment_block}\n\n{emotion_block}\n\n{summary}"

    def _summary(self, results: Dict[str, Any], *, include_header: bool = True) -> str:
        sentiment = results.get("sentiment", {})
        emotion = results.get("emotion", {})
        sent_label = sentiment.get("label", "unknown")
        sent_score = float(sentiment.get("score", 0.0))
        emo_label = emotion.get("label", "unknown")
        emo_score = float(emotion.get("score", 0.0))

        formatted_sent = self.label_mapper.format_with_confidence(sent_label, sent_score, is_emotion=False)
        formatted_emo = self.label_mapper.format_with_confidence(emo_label, emo_score, is_emotion=True)
        if include_header:
            return f"{self._header('Summary')}\nThis text expresses {formatted_sent} sentiment with {formatted_emo} emotion."
        return f"Text expresses {formatted_sent} sentiment with {formatted_emo} emotion."

    # ------------------------------------------------------------------
    # Misc public utilities
    # ------------------------------------------------------------------
    @staticmethod
    def create_progress_bar(current: int, total: int, width: int = 40) -> str:
        progress = current / total if total else 0
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"Processing: [{bar}] {progress * 100:.1f}% ({current}/{total})"


# ---------------------------------------------------------------------------
# Singleton instance used by module-level wrapper functions
# ---------------------------------------------------------------------------
_FORMATTER = OutputFormatter(_LABEL_MAPPER, _SETTINGS)

###############################################################################
# Wrapper functions (maintain backward compat with utils.cli imports)
###############################################################################

def _indicator(confident: bool) -> str:
    """Return ✓ or ✗ based on confidence boolean."""
    return "✓" if confident else "✗"


def format_sentiment_result(result: Dict[str, Any], show_probabilities: bool = False) -> str:  # noqa: D401
    """Public wrapper matching the expectations defined in the test-suite."""
    label = (result.get("label") or "unknown").title()
    score = float(result.get("score", 0))
    confident = bool(result.get("confident", True))
    threshold = float(result.get("threshold", 0))
    
    # Choose color based on sentiment
    if label.lower() == "positive":
        color = Fore.GREEN
    elif label.lower() == "negative":
        color = Fore.RED
    else:  # neutral or unknown
        color = Fore.BLUE
    
    parts: List[str] = [
        f"{color}Sentiment: {label}{Style.RESET_ALL}  {_indicator(confident)}  {score*100:.1f}% (threshold: {threshold*100:.1f}%)"
    ]
    if show_probabilities:
        parts.append("Sentiment Probabilities:")
        parts.append(_FORMATTER._probabilities_block(result.get("raw_probabilities", {})))
    return "\n".join(parts)


def format_emotion_result(result: Dict[str, Any], show_probabilities: bool = False) -> str:
    label_raw = result.get("label")
    if label_raw is None:
        label = "None detected"
        color = Fore.WHITE
    else:
        label = str(label_raw).title()
        # Choose color based on emotion
        emotion_colors = {
            "joy": Fore.YELLOW,
            "sadness": Fore.BLUE,
            "anger": Fore.RED,
            "fear": Fore.MAGENTA,
            "surprise": Fore.CYAN,
            "love": Fore.LIGHTMAGENTA_EX,
        }
        color = emotion_colors.get(label.lower(), Fore.WHITE)
        
    score = float(result.get("score", 0))
    confident = bool(result.get("confident", True))
    threshold = float(result.get("threshold", 0))
    parts: List[str] = [
        f"{color}Emotion: {label}{Style.RESET_ALL}  {_indicator(confident)}  {score*100:.1f}% (threshold: {threshold*100:.1f}%)"
    ]
    if show_probabilities:
        parts.append("Emotion Probabilities:")
        parts.append(_FORMATTER._probabilities_block(result.get("raw_probabilities", {})))
    return "\n".join(parts)


def format_analysis_result(result: Dict[str, Any], show_probabilities: bool = False, settings: Optional[Settings] = None) -> str:
    """Format sentiment and emotion analysis results for output.

    This is a simpler wrapper for test use that doesn't perform the rich
    assert that behaviour.
    """
    # Check for summary-only mode
    if settings and getattr(settings, "summary_only", False):
        # Generate a brief summary line
        sentiment = result.get("sentiment", "unknown")
        emotion = result.get("emotion", "none")
        sentiment_score = result.get("sentiment_score", 0.0)
        emotion_score = result.get("emotion_score", 0.0)
        
        if isinstance(sentiment, dict):
            sentiment = sentiment.get("label", "unknown")
            sentiment_score = sentiment.get("score", 0.0)
        if isinstance(emotion, dict):
            emotion = emotion.get("label", "none")
            emotion_score = emotion.get("score", 0.0)
        
        # Handle None values
        if sentiment_score is None:
            sentiment_score = 0.0
        if emotion_score is None:
            emotion_score = 0.0
        if emotion is None:
            emotion = "none"
        
        # Format: "sentiment: positive (0.92) emotion: joy (0.85)"
        return f"sentiment: {sentiment} ({sentiment_score:.2f}) emotion: {emotion} ({emotion_score:.2f})"
    
    text = result.get("text", "")
    display_text = (text[:97] + "...") if len(text) > 100 else text

    parts: List[str] = [display_text]
    
    # Handle sentiment - convert flat structure to nested if needed
    sentiment_data = result.get("sentiment", {})
    if isinstance(sentiment_data, str):
        # Flat structure: convert to nested
        sentiment_nested = {
            "label": sentiment_data,
            "score": result.get("sentiment_score", 0.0),
            "confident": True,  # Default for compatibility
            "threshold": 0.7,   # Default threshold
            "raw_probabilities": {}
        }
    else:
        # Already nested structure
        sentiment_nested = sentiment_data
    
    parts.append(format_sentiment_result(sentiment_nested, show_probabilities))
    
    # Handle emotion - convert flat structure to nested if needed
    emotion_data = result.get("emotion", {})
    if isinstance(emotion_data, str) or emotion_data is None:
        # Flat structure: convert to nested
        emotion_score = result.get("emotion_score")
        emotion_nested = {
            "label": emotion_data,
            "score": emotion_score if emotion_score is not None else 0.0,
            "confident": True,  # Default for compatibility
            "threshold": 0.6,   # Default threshold
            "raw_probabilities": {}
        }
    else:
        # Already nested structure
        emotion_nested = emotion_data
    
    parts.append(format_emotion_result(emotion_nested, show_probabilities))
    return "\n".join(parts)


def flatten_result_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested result dictionaries for easier CSV export.

    The returned mapping contains scalar values only.  Nested probability
    mappings are flattened using the pattern "<type>_prob_<label>" as required
    by the unit-tests.
    """
    flat: Dict[str, Any] = {}

    # Basic scalar fields with fall-backs
    flat["text"] = result.get("text", "")
    flat["sentiment"] = result.get("sentiment", "unknown")
    flat["sentiment_score"] = result.get("sentiment_score", 0)
    flat["emotion"] = result.get("emotion", "unknown")
    flat["emotion_score"] = result.get("emotion_score", 0)
    if "confidence" in result:
        flat["confidence"] = result["confidence"]

    # Flatten probability dictionaries --------------------------------
    for key in ("sentiment_probabilities", "emotion_probabilities"):
        probs = result.get(key)
        if isinstance(probs, dict):
            prefix = key.replace("_probabilities", "_prob_")
            for label, value in probs.items():
                flat[f"{prefix}{label}"] = value

    # Include any leftover top-level scalar keys that are not dict/list
    for k, v in result.items():
        if k not in flat and not isinstance(v, (dict, list)):
            flat[k] = v

    return flat


def format_probabilities(probabilities: Dict[str, float]) -> str:
    # still exposed for tests but now uses the richer block
    return _FORMATTER._probabilities_block(probabilities)


def create_progress_bar(current: int, total: int, width: int = 40) -> str:  # noqa: D401
    return _FORMATTER.create_progress_bar(current, total, width)

###############################################################################
# Export helpers (unchanged from previous version for compatibility)
###############################################################################

REQUIRED_CSV_FIELDS: List[str] = [
    "text",
    "sentiment",
    "sentiment_score",
    "emotion",
    "emotion_score",
    "model",
    "positive",
    "neutral",
    "negative",
]


def export_to_json(results: List[Dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    try:
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
    except Exception as exc:  # pragma: no cover – raised further up-stack
        raise IOError(f"Failed to export to JSON: {exc}") from exc


def export_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    flattened_results: List[Dict[str, Any]] = []
    for res in results:
        row = {k: "" for k in REQUIRED_CSV_FIELDS}
        row["text"] = res.get("text", "")
        row["model"] = res.get("model", "")
        # Sentiment
        sent = res.get("sentiment", res)
        if isinstance(sent, dict):
            row["sentiment"] = sent.get("label", "")
            row["sentiment_score"] = sent.get("score", "")
        else:
            row["sentiment"] = res.get("sentiment", "")
            row["sentiment_score"] = res.get("sentiment_score", "")
        # Emotion
        emo = res.get("emotion", res)
        if isinstance(emo, dict):
            row["emotion"] = emo.get("label", "")
            row["emotion_score"] = emo.get("score", "")
        else:
            row["emotion"] = res.get("emotion", "")
            row["emotion_score"] = res.get("emotion_score", "")
        # Probabilities (flatten sentiment probs only for CSV)
        probs = res.get("probabilities", {})
        row["positive"] = probs.get("positive", "")
        row["neutral"] = probs.get("neutral", "")
        row["negative"] = probs.get("negative", "")
        flattened_results.append(row)

    if not flattened_results:
        raise ValueError("No results to export")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REQUIRED_CSV_FIELDS)
    writer.writeheader()
    writer.writerows(flattened_results)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write(output.getvalue())

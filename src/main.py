#!/usr/bin/env python3
"""
Smart CLI Sentiment & Emotion Analyzer
A command-line tool for sentiment and emotion analysis of text.
"""

import argparse
import sys
import os

# Ensure project root is on PYTHONPATH for subprocess tests
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Always propagate project root to subprocess environment via PYTHONPATH
existing_py_path = os.environ.get('PYTHONPATH', '')
if _PROJECT_ROOT not in existing_py_path.split(os.pathsep):
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [existing_py_path, _PROJECT_ROOT]))

from typing import Dict, List, Optional, Any, Literal
from colorama import init, Fore, Style
import json
from tqdm import tqdm
import time
import readline  # For better input editing capabilities

# Try relative imports first, fallback to absolute for direct execution
try:
    from .models.transformer import SentimentEmotionTransformer
    from .models.comparison import ModelComparison
    from .utils.output import (
        format_analysis_result, 
        create_progress_bar, 
        export_to_json, 
        export_to_csv
    )
    from .utils.settings import Settings
    from .utils.labels import LabelMapper
except ImportError:
    # Fallback for when script is run directly
    from src.models.transformer import SentimentEmotionTransformer
    from src.models.comparison import ModelComparison
    from src.utils.output import (
        format_analysis_result, 
        create_progress_bar, 
        export_to_json, 
        export_to_csv
    )
    from src.utils.settings import Settings
    from src.utils.labels import LabelMapper

# Global quiet mode flag
QUIET_MODE = False # noqa: PLW0603

# ---------------------------------------------------------------------------
# Helper print wrappers usable across module
# ---------------------------------------------------------------------------

from typing import Any as _Any


def info_print(msg: str, *, end: str = "\n", file: _Any = sys.stdout):  # noqa: D401
    if not QUIET_MODE:
        print(msg, end=end, file=file)


def print_error(msg: str, *, file: _Any = sys.stderr):  # noqa: D401
    print(f"{Fore.RED}Error: {msg}{Style.RESET_ALL}", file=file)


# ---------------------------------------------------------------------------
# JSON stream helper
# ---------------------------------------------------------------------------


def format_result_as_json(result: Dict[str, Any], *, include_probabilities: bool = False, text: Optional[str] = None) -> str:  # noqa: D401
    """Convert a single analysis *result* into a compact JSON string suitable for NDJSON output."""

    payload: Dict[str, Any] = {}

    if text is not None:
        payload["text"] = text
    elif "text" in result:
        payload["text"] = result["text"]

    # Handle sentiment - support both nested dict and flat structure
    if "sentiment" in result:
        if isinstance(result["sentiment"], dict):
            # Nested structure: {"sentiment": {"label": "positive", "score": 0.9}}
            sent_obj = {
                "label": result["sentiment"].get("label", ""),
                "score": result["sentiment"].get("score", 0.0),
            }
            if include_probabilities and "raw_probabilities" in result["sentiment"]:
                sent_obj["probabilities"] = result["sentiment"].get("raw_probabilities", {})
        else:
            # Flat structure: {"sentiment": "positive", "sentiment_score": 0.9}
            sent_obj = {
                "label": result["sentiment"],
                "score": result.get("sentiment_score", 0.0),
            }
        payload["sentiment"] = sent_obj

    # Handle emotion - support both nested dict and flat structure
    if "emotion" in result:
        if isinstance(result["emotion"], dict):
            # Nested structure: {"emotion": {"label": "joy", "score": 0.8}}
            emo_obj = {
                "label": result["emotion"].get("label", ""),
                "score": result["emotion"].get("score", 0.0),
            }
            if include_probabilities and "raw_probabilities" in result["emotion"]:
                emo_obj["probabilities"] = result["emotion"].get("raw_probabilities", {})
        else:
            # Flat structure: {"emotion": "joy", "emotion_score": 0.8}
            emo_obj = {
                "label": result["emotion"],
                "score": result.get("emotion_score", 0.0),
            }
        payload["emotion"] = emo_obj

    return json.dumps(payload, ensure_ascii=False)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Smart CLI Sentiment & Emotion Analyzer"
    )
    
    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-t", "--text", 
        type=str, 
        help="Single text to analyze"
    )
    input_group.add_argument(
        "-f", "--file", 
        type=str, 
        help="Path to file with multiple texts (one per line)"
    )
    input_group.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive mode for real-time analysis"
    )
    input_group.add_argument(
        "-c", "--compare-interactive",
        action="store_true",
        help="Start interactive mode with model comparison"
    )
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--sentiment-model", 
        type=str, 
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        help="Sentiment model to use"
    )
    model_group.add_argument(
        "--emotion-model", 
        type=str, 
        default="bhadresh-savani/distilbert-base-uncased-emotion",
        help="Emotion model to use"
    )
    model_group.add_argument(
        "--local-model-path", 
        type=str, 
        help="Path to locally saved models"
    )
    model_group.add_argument(
        "--sentiment-threshold", 
        type=float, 
        default=0.7,
        help=f"Confidence threshold for sentiment predictions (0.0-1.0, default: {Settings().get_sentiment_threshold()})"
    )
    model_group.add_argument(
        "--emotion-threshold", 
        type=float, 
        default=0.6,
        help=f"Confidence threshold for emotion predictions (0.0-1.0, default: {Settings().get_emotion_threshold()})"
    )
    model_group.add_argument(
        "--compare-models",
        type=str,
        help="Comma-separated list of model names to compare (for comparison mode)"
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        help="Custom name for the model (useful in comparison mode)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--show-probabilities", 
        action="store_true", 
        help="Show raw probability distributions for all classes"
    )
    output_group.add_argument(
        "--output", 
        type=str, 
        help="Output file path for saving results (without extension)"
    )
    output_group.add_argument(
        "--format", 
        type=str, 
        choices=["text", "json", "csv", "all"],
        default="text",
        help="Output format for saving results (text, json, csv, or all)"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bars and informational messages; only final results are printed",
    )
    output_group.add_argument(
        "--no-colour", "--no-color", "-nc",
        action="store_true",
        help="Disable ANSI colour codes in output (useful for redirecting output)",
    )
    output_group.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Print only the single-line sentiment + emotion summary",
    )

    output_group.add_argument(
        "--json-stream", "-j",
        action="store_true",
        help="Output each analysis as a JSON line (NDJSON) to stdout",
    )
    
    # Global settings options
    parser.add_argument(
        "--reset-settings",
        action="store_true",
        help="Reset all saved settings back to their defaults and exit",
    )
    
    args = parser.parse_args()

    # Default to interactive mode if no text or file specified
    if not args.text and not args.file and not args.interactive and not args.compare_interactive:
        args.interactive = True

    return args


def main():
    """
    Main entry point for CLI.
    """
    # Parse command line arguments
    args = parse_args()

    # If NDJSON streaming requested, force quiet mode for clean output
    if getattr(args, "json_stream", False):
        args.quiet = True

    # Global quiet mode flag
    global QUIET_MODE  # noqa: PLW0603
    QUIET_MODE = getattr(args, "quiet", False)

    # info_print and print_error already defined globally and respect QUIET_MODE

    # Handle no-colour flag OR json-stream flag BEFORE any coloured output
    if getattr(args, "no_colour", False) or getattr(args, "json_stream", False):
        # Re-initialise colorama to strip all ANSI codes
        init(strip=True, autoreset=True)

        class _NoColor(str):
            def __getattr__(self, name):
                return ""

        import colorama as _colorama_mod
        _colorama_mod.Fore = _NoColor()
        _colorama_mod.Back = _NoColor()
        _colorama_mod.Style = _NoColor()

        os.environ["ANSI_COLORS_DISABLED"] = "1"

    else:
        # Force colors even in non-TTY environments (e.g., subprocess)
        init(autoreset=True, convert=False, strip=False)
    
    # Apply summary-only preference to settings when created later
    settings = Settings()

    settings.set_quiet_mode(QUIET_MODE)
    settings.set_summary_only(getattr(args, "summary_only", False))
    settings.set_json_stream(getattr(args, "json_stream", False))

    # Handle settings reset early and exit
    if getattr(args, "reset_settings", False):
        if Settings().reset_to_defaults().save_settings():
            print(f"{Fore.GREEN}Settings have been reset to defaults.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Settings reset to defaults, but could not be persisted to disk.{Style.RESET_ALL}")
        return

    info_print(f"Analyzing: \"{args.text}\"")
    
    try:
        # Load the model
        model = SentimentEmotionTransformer(
            sentiment_model=args.sentiment_model,
            emotion_model=args.emotion_model,
            sentiment_threshold=args.sentiment_threshold,
            emotion_threshold=args.emotion_threshold,
            local_model_path=args.local_model_path,
            name=args.model_name,
        )
        
        # Attach settings to the model
        model.settings = settings
        
        # Process based on input method
        if args.text:
            result = model.analyze(args.text)
            
            if getattr(args, "json_stream", False):
                formatted = format_result_as_json(result, include_probabilities=args.show_probabilities)
                print(formatted)
            else:
                formatted = format_analysis_result(result, args.show_probabilities, settings)
                print(formatted)
        
        elif args.file:
            # Process file
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print_error(f"Error reading file: {e}")
                sys.exit(1)
            
            info_print(f"Processing {len(lines)} texts from {args.file}...")
            
            for i, line in enumerate(lines):
                # Show progress (skip when quiet)
                if not QUIET_MODE:
                    progress = create_progress_bar(i, len(lines))
                    print(f"\r{progress}", end="")

                # Analyze text
                result = model.analyze(line)
                result.setdefault("text", line)

                # Handle JSON-stream output
                if getattr(args, "json_stream", False):
                    json_line = format_result_as_json(result, include_probabilities=args.show_probabilities)
                    print(json_line)
                else:
                    formatted = format_analysis_result(result, args.show_probabilities, settings)
                    print(formatted)
            
            # Complete the progress bar
            if not QUIET_MODE:
                print(create_progress_bar(len(lines), len(lines)))
            info_print(f"Completed analyzing {len(lines)} lines.")
    
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
    
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# src/utils/cli.py
import argparse
import sys
import os
from typing import Dict, List, Optional, Any, Literal
from colorama import init, Fore, Style
import json
from tqdm import tqdm
import time
import readline  # For better input editing capabilities

from ..models.transformer import SentimentEmotionTransformer
from ..models.comparison import ModelComparison
from .output import (
    format_analysis_result, 
    create_progress_bar, 
    export_to_json, 
    export_to_csv
)
from .settings import Settings  # new import for settings management
from .labels import LabelMapper

# Global quiet mode flag
QUIET_MODE = False # noqa: PLW0603

# ---------------------------------------------------------------------------
# Helper print wrappers usable across module
# ---------------------------------------------------------------------------

from typing import Any as _Any


def info_print(msg: str, *, end: str = "\n", file: _Any = sys.stdout):  # noqa: D401
    if not QUIET_MODE:
        print(msg, end=end, file=file)


def print_error(msg: str):  # noqa: D401
    print(f"{Fore.RED}{msg}{Style.RESET_ALL}", file=sys.stderr)

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
    
    # If no input method is specified, default to interactive mode
    if not (args.text or args.file or args.interactive or args.compare_interactive):
        args.interactive = True
        
    return args

def load_transformer_model(
    sentiment_model: str,
    emotion_model: str,
    sentiment_threshold: float,
    emotion_threshold: float,
    local_model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> SentimentEmotionTransformer:
    """
    Load a transformer model with the specified parameters.
    
    Args:
        sentiment_model: Sentiment model identifier
        emotion_model: Emotion model identifier
        sentiment_threshold: Sentiment threshold
        emotion_threshold: Emotion threshold
        local_model_path: Optional path to locally saved models
        model_name: Optional custom name for the model
        settings: Optional settings object
        
    Returns:
        Initialized SentimentEmotionTransformer model
    """
    info_print(f"Loading model {model_name or sentiment_model}...")
    
    try:
        model = SentimentEmotionTransformer(
            sentiment_model=sentiment_model,
            emotion_model=emotion_model,
            sentiment_threshold=sentiment_threshold,
            emotion_threshold=emotion_threshold,
            local_model_path=local_model_path,
            name=model_name,
        )
        
        # Attach settings to the model if provided
        if settings is not None:
            model.settings = settings
        
        info_print("Model loaded successfully!")
        return model
    
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        sys.exit(1)

def load_comparison_models(args) -> List[SentimentEmotionTransformer]:
    """
    Load models for comparison.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of transformer models to compare
    """
    models = []
    
    # Add default model
    default_model = load_transformer_model(
        args.sentiment_model,
        args.emotion_model,
        args.sentiment_threshold,
        args.emotion_threshold,
        args.local_model_path,
        name=args.model_name or "Default",
    )
    models.append(default_model)
    
    # Add comparison models if specified
    if args.compare_models:
        model_specs = args.compare_models.split(",")
        
        for i, spec in enumerate(model_specs):
            parts = spec.split(":")
            if len(parts) == 1:
                # Just the sentiment model
                sentiment_model = parts[0].strip()
                emotion_model = args.emotion_model
                sentiment_threshold = args.sentiment_threshold
                emotion_threshold = args.emotion_threshold
                name = f"Model {i+1}"
            elif len(parts) == 2:
                # Sentiment and emotion model
                sentiment_model = parts[0].strip()
                emotion_model = parts[1].strip()
                sentiment_threshold = args.sentiment_threshold
                emotion_threshold = args.emotion_threshold
                name = f"Model {i+1}"
            elif len(parts) == 3:
                # Sentiment, emotion, and name
                sentiment_model = parts[0].strip()
                emotion_model = parts[1].strip()
                name = parts[2].strip()
                sentiment_threshold = args.sentiment_threshold
                emotion_threshold = args.emotion_threshold
            elif len(parts) == 5:
                # Sentiment, emotion, name, sentiment threshold, emotion threshold
                sentiment_model = parts[0].strip()
                emotion_model = parts[1].strip()
                name = parts[2].strip()
                try:
                    sentiment_threshold = float(parts[3].strip())
                    emotion_threshold = float(parts[4].strip())
                except ValueError:
                    print(f"{Fore.YELLOW}Invalid threshold in model spec: {spec}. Using default thresholds.{Style.RESET_ALL}")
                    sentiment_threshold = args.sentiment_threshold
                    emotion_threshold = args.emotion_threshold
            else:
                print(f"{Fore.YELLOW}Invalid model specification: {spec}. Using default models.{Style.RESET_ALL}")
                continue
            
            try:
                model = load_transformer_model(
                    sentiment_model,
                    emotion_model,
                    sentiment_threshold,
                    emotion_threshold,
                    args.local_model_path,
                    name=name,
                )
                models.append(model)
            except Exception as e:
                print_error(f"Error loading comparison model {name}: {e}")
    
    return models

def set_thresholds_interactive(model: SentimentEmotionTransformer) -> None:
    """
    Interactive function to set thresholds for sentiment and emotion.
    
    Args:
        model: The model to adjust thresholds for
    """
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Threshold Adjustment{Fore.CYAN}                                      ║
╠══════════════════════════════════════════════════════════╣
║ {Fore.WHITE}Current thresholds:{Fore.CYAN}                                       ║
║ {Fore.WHITE}Sentiment: {Fore.YELLOW}{model.sentiment_threshold:.2f}{Fore.CYAN}{' ' * 49}║
║ {Fore.WHITE}Emotion:   {Fore.YELLOW}{model.emotion_threshold:.2f}{Fore.CYAN}{' ' * 49}║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    # Get new sentiment threshold
    while True:
        try:
            sentiment_input = input(f"{Fore.GREEN}Enter new sentiment threshold (0.0-1.0) or press Enter to keep current: {Style.RESET_ALL}")
            
            if not sentiment_input.strip():
                # Keep current threshold
                sentiment_threshold = model.sentiment_threshold
                break
                
            sentiment_threshold = float(sentiment_input)
            if 0 <= sentiment_threshold <= 1:
                break
            else:
                print(f"{Fore.RED}Threshold must be between 0.0 and 1.0{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
    
    # Get new emotion threshold
    while True:
        try:
            emotion_input = input(f"{Fore.GREEN}Enter new emotion threshold (0.0-1.0) or press Enter to keep current: {Style.RESET_ALL}")
            
            if not emotion_input.strip():
                # Keep current threshold
                emotion_threshold = model.emotion_threshold
                break
                
            emotion_threshold = float(emotion_input)
            if 0 <= emotion_threshold <= 1:
                break
            else:
                print(f"{Fore.RED}Threshold must be between 0.0 and 1.0{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
    
    # Update model thresholds
    model.set_thresholds(sentiment_threshold, emotion_threshold)
    
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Thresholds Updated:{Fore.CYAN}                                       ║
║ {Fore.WHITE}Sentiment: {Fore.GREEN}{model.sentiment_threshold:.2f}{Fore.CYAN}{' ' * 49}║
║ {Fore.WHITE}Emotion:   {Fore.GREEN}{model.emotion_threshold:.2f}{Fore.CYAN}{' ' * 49}║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

def analyze_text(
    text: str, 
    model: SentimentEmotionTransformer,
    show_probabilities: bool = False
) -> tuple[Dict[str, Any], str]:
    """
    Analyze a single text.
    
    Args:
        text: Input text to analyze
        model: SentimentEmotionTransformer model
        show_probabilities: Whether to show raw probabilities
        
    Returns:
        Tuple of (analysis result dictionary, formatted result string)
    """
    result = model.analyze(text)

    # Attach original text for downstream usage
    result.setdefault("text", text)

    if getattr(model.settings, "json_stream", False):
        formatted = format_result_as_json(result, include_probabilities=show_probabilities)
    else:
        formatted = format_analysis_result(result, show_probabilities, model.settings)

    return result, formatted

def compare_models(
    text: str,
    models: List[SentimentEmotionTransformer],
    show_probabilities: bool = False
) -> tuple[Dict[str, Any], str]:
    """
    Compare multiple models on a single text.
    
    Args:
        text: Input text to analyze
        models: List of models to compare
        show_probabilities: Whether to show raw probabilities
        
    Returns:
        Tuple of (comparison result dictionary, formatted result string)
    """
    if not models:
        raise ValueError("No models provided for comparison")
    
    comparison = ModelComparison(models)
    result = comparison.compare(text)

    json_stream_enabled = hasattr(models[0], "settings") and getattr(models[0].settings, "json_stream", False) is True

    if json_stream_enabled:
        # Build compact JSON structure
        formatted = json.dumps(result, ensure_ascii=False)
    else:
        formatted = comparison.format_comparison(result, show_probabilities)

    return result, formatted

def export_results(
    results: list,
    output_format: str,
    output_file: str
) -> None:
    """
    Export results in the specified format.
    """
    if output_format == "text" or output_format == "all":
        text_output = f"{output_file}.txt"
        try:
            formatted_results = [format_analysis_result(r, True, None) for r in results]
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(formatted_results))
            info_print(f"Results saved to {text_output}")
        except Exception as e:
            info_print(f"Error saving text results: {e}")

    if output_format == "json" or output_format == "all":
        json_output = f"{output_file}.json"
        try:
            from .output import export_to_json
            export_to_json(results, json_output)
            info_print(f"Results saved to {json_output}")
        except Exception as e:
            info_print(f"Error saving JSON results: {e}")

    if output_format == "csv" or output_format == "all":
        csv_output = f"{output_file}.csv"
        try:
            from .output import export_to_csv
            export_to_csv(results, csv_output)
            info_print(f"Results saved to {csv_output}")
        except Exception as e:
            info_print(f"Error saving CSV results: {e}")

    if output_format not in ("text", "json", "csv", "all"):
        info_print(f"Error saving: Unknown format '{output_format}'")

def export_comparison_results(
    comparison_results: list,
    output_format: str,
    output_file: str
) -> None:
    """
    Export comparison results in the specified format.
    """
    if not comparison_results:
        info_print("No comparison results to export")
        return

    from src.models.comparison import ModelComparison
    comparison = ModelComparison()  # Just for exporting

    if output_format == "text" or output_format == "all":
        text_output = f"{output_file}.txt"
        try:
            with open(text_output, 'w', encoding='utf-8') as f:
                for i, result in enumerate(comparison_results):
                    formatted = comparison.format_comparison(result, True)
                    # Strip color codes
                    formatted = comparison._strip_color_codes(formatted)
                    f.write(" " + formatted)  # Add leading space
                    if i < len(comparison_results) - 1:
                        f.write("\n\n")
            info_print(f"Comparison results saved to {text_output}")
        except Exception as e:
            print_error(f"Error saving text comparison results: {e}")

    if output_format == "json" or output_format == "all":
        json_output = f"{output_file}.json"
        try:
            serializable = [comparison._prepare_for_serialization(r) for r in comparison_results]
            with open(json_output, 'w', encoding='utf-8') as f:
                import json
                json.dump(serializable, f, indent=2)
            info_print(f"Comparison results saved to {json_output}")
        except Exception as e:
            print_error(f"Error saving JSON comparison results: {e}")

    if output_format == "csv" or output_format == "all":
        csv_output = f"{output_file}.csv"
        try:
            with open(csv_output, 'w', encoding='utf-8', newline='') as f:
                import csv
                writer = csv.writer(f)
                header = ["Text", "Model", "Sentiment", "Sentiment Score", "Emotion", "Emotion Score", "Execution Time", "Sentiment Agreement", "Emotion Agreement"]
                writer.writerow(header)
                for comp_result in comparison_results:
                    text = comp_result.get("text", "")
                    sent_agreement = comp_result.get("sentiment_agreement", 0.0)
                    emo_agreement = comp_result.get("emotion_agreement", 0.0)
                    for result in comp_result.get("results", []):
                        row = [
                            text,
                            result.get("model", "Unknown"),
                            result.get("sentiment", {}).get("label", ""),
                            result.get("sentiment", {}).get("score", 0.0),
                            result.get("emotion", {}).get("label", ""),
                            result.get("emotion", {}).get("score", 0.0),
                            result.get("execution_time", 0.0),
                            sent_agreement,
                            emo_agreement
                        ]
                        writer.writerow(row)
            info_print(f"Comparison results saved to {csv_output}")
        except Exception as e:
            print_error(f"Error saving CSV comparison results: {e}")

    if output_format not in ("text", "json", "csv", "all"):
        info_print(f"Error saving: Unknown format '{output_format}'")

def process_batch_file(
    file_path: str,
    model: SentimentEmotionTransformer,
    show_probabilities: bool = False,
    output_format: Optional[Literal["text", "json", "csv", "all"]] = None,
    output_file: Optional[str] = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Process multiple texts from a file.
    
    Args:
        file_path: Path to input file (one text per line)
        model: SentimentEmotionTransformer model
        show_probabilities: Whether to show raw probabilities
        output_format: Format to export results
        output_file: Output filepath without extension
        
    Returns:
        Tuple of (raw results, formatted results)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print_error(f"Error reading file: {e}")
        sys.exit(1)
    
    results = []
    formatted_results = []
    
    # Process with progress bar
    info_print(f"Processing {len(lines)} texts from {file_path}...")
    
    for i, line in enumerate(lines):
        # Show progress (skip when quiet)
        if not QUIET_MODE:
            progress = create_progress_bar(i, len(lines))
            print(f"\r{progress}", end="")

        # Analyze text
        result = model.analyze(line)
        result.setdefault("text", line)

        results.append(result)

        # Handle JSON-stream output
        if getattr(model.settings, "json_stream", False):
            json_line = format_result_as_json(result, include_probabilities=show_probabilities)
            print(json_line)
            formatted_results.append(json_line)
        else:
            formatted = format_analysis_result(result, show_probabilities, model.settings)
            formatted_results.append(formatted)
    
    # Complete the progress bar
    if not QUIET_MODE:
        print(create_progress_bar(len(lines), len(lines)))
    info_print(f"Completed analyzing {len(lines)} lines.")
    
    # Export results if output file specified
    if output_file and output_format:
        export_results(results, output_format, output_file)
    
    return results, formatted_results

def process_batch_comparison(
    file_path: str,
    models: List[SentimentEmotionTransformer],
    show_probabilities: bool = False,
    output_format: Optional[Literal["text", "json", "csv", "all"]] = None,
    output_file: Optional[str] = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Process multiple texts from a file with model comparison.
    
    Args:
        file_path: Path to input file (one text per line)
        models: List of models to compare
        show_probabilities: Whether to show raw probabilities
        output_format: Format to export results
        output_file: Output filepath without extension
        
    Returns:
        Tuple of (raw comparison results, formatted comparison results)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print_error(f"Error reading file: {e}")
        sys.exit(1)
    
    comparison = ModelComparison(models)
    results = []
    formatted_results = []
    
    # Process with progress bar
    info_print(f"Processing {len(lines)} texts with {len(models)} models from {file_path}...")
    
    for i, line in enumerate(lines):
        # Show progress
        if not QUIET_MODE:
            progress = create_progress_bar(i, len(lines))
            print(f"\r{progress}", end="")

        # Compare models on text
        result = comparison.compare(line)
        results.append(result)

        json_stream_enabled = hasattr(models[0], "settings") and getattr(models[0].settings, "json_stream", False) is True
        if json_stream_enabled:
            # Build compact JSON representation similar to compare_models above
            comp_payload: Dict[str, Any] = {
                "text": line,
                "comparison": {},
            }
            for m_name, res in result.items():
                comp_payload["comparison"][m_name] = {
                    "sentiment": {
                        "label": res["sentiment"].get("label", ""),
                        "score": res["sentiment"].get("score", 0.0),
                    },
                    "emotion": {
                        "label": res["emotion"].get("label", ""),
                        "score": res["emotion"].get("score", 0.0),
                    },
                }
                if show_probabilities:
                    if "raw_probabilities" in res["sentiment"]:
                        comp_payload["comparison"][m_name]["sentiment"]["probabilities"] = res["sentiment"].get("raw_probabilities", {})
                    if "raw_probabilities" in res["emotion"]:
                        comp_payload["comparison"][m_name]["emotion"]["probabilities"] = res["emotion"].get("raw_probabilities", {})

            formatted_line = json.dumps(comp_payload, ensure_ascii=False)
            print(formatted_line)
            formatted_results.append(formatted_line)
        else:
            formatted = comparison.format_comparison(result, show_probabilities)
            formatted_results.append(formatted)
    
    # Complete the progress bar
    if not QUIET_MODE:
        print(create_progress_bar(len(lines), len(lines)))
    info_print("Comparison complete")
    
    # Export results if output file specified
    if output_file and output_format:
        export_comparison_results(results, output_format, output_file)
    
    return results, formatted_results

def run_interactive_mode(
    model: SentimentEmotionTransformer,
    show_probabilities: bool = False
) -> None:
    """
    Run interactive mode for real-time analysis.
    
    Args:
        model: SentimentEmotionTransformer model
        show_probabilities: Whether to show raw probabilities
    """
    # Configure history file for readline
    history_file = os.path.expanduser("~/.sentiment_analyzer_history")
    try:
        readline.read_history_file(history_file)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    
    # Store analyzed texts and results for possible export
    results = []
    
    # Display welcome message
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Interactive Sentiment & Emotion Analysis{Fore.CYAN}                  ║
╠══════════════════════════════════════════════════════════╣
║ {Fore.GREEN}Type any text to analyze its sentiment and emotion.{Fore.CYAN}        ║
║ {Fore.GREEN}Commands:{Fore.CYAN}                                                 ║
║   {Fore.YELLOW}:help{Fore.CYAN}      - Show this help message                       ║
║   {Fore.YELLOW}:probabilities{Fore.CYAN} - Toggle showing probability distributions  ║
║   {Fore.YELLOW}:export <format> <filename>{Fore.CYAN} - Export results               ║
║               {Fore.WHITE}(formats: json, csv, text, all){Fore.CYAN}             ║
║   {Fore.YELLOW}:history{Fore.CYAN}    - Show analysis history                       ║
║   {Fore.YELLOW}:clear{Fore.CYAN}      - Clear the screen                            ║
║   {Fore.YELLOW}:quit{Fore.CYAN}       - Exit interactive mode                       ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    # Main interactive loop
    while True:
        try:
            # Get input from user
            user_input = input(f"{Fore.GREEN}> {Style.RESET_ALL}").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Process commands
            if user_input.startswith(":"):
                command = user_input.lower()
                
                # Help command
                if command == ":help":
                    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Available Commands:{Fore.CYAN}                                        ║
╠══════════════════════════════════════════════════════════╣
║   {Fore.YELLOW}:help{Fore.CYAN}      - Show this help message                       ║
║   {Fore.YELLOW}:probabilities{Fore.CYAN} - Toggle showing probability distributions  ║
║   {Fore.YELLOW}:export <format> <filename>{Fore.CYAN} - Export results               ║
║               {Fore.WHITE}(formats: json, csv, text, all){Fore.CYAN}             ║
║   {Fore.YELLOW}:history{Fore.CYAN}    - Show analysis history                       ║
║   {Fore.YELLOW}:clear{Fore.CYAN}      - Clear the screen                            ║
║   {Fore.YELLOW}:quit{Fore.CYAN}       - Exit interactive mode                       ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
                
                # Toggle probabilities
                elif command == ":probabilities":
                    show_probabilities = not show_probabilities
                    print(f"{Fore.YELLOW}Probability display: {Fore.GREEN if show_probabilities else Fore.RED}{show_probabilities}{Style.RESET_ALL}")
                
                # Export results
                elif command.startswith(":export "):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 3:
                        print(f"{Fore.RED}Usage: :export <format> <filename>{Style.RESET_ALL}")
                        continue
                    
                    _, export_format, export_file = parts
                    if export_format.lower() not in ["json", "csv", "text", "all"]:
                        print(f"{Fore.RED}Invalid format. Use json, csv, text, or all.{Style.RESET_ALL}")
                        continue
                    
                    # Check if we have results to export
                    if not results:
                        print(f"{Fore.YELLOW}No results to export yet.{Style.RESET_ALL}")
                        continue
                    
                    # Export results
                    try:
                        export_results(results, export_format, export_file)
                    except Exception as e:
                        print_error(f"Export failed: {e}")
                
                # Show history
                elif command == ":history":
                    if not results:
                        print(f"{Fore.YELLOW}No analysis history yet.{Style.RESET_ALL}")
                        continue
                    
                    print(f"{Fore.CYAN}Analysis History ({len(results)} entries):{Style.RESET_ALL}")
                    for i, result in enumerate(results[-5:]):  # Show last 5 entries
                        text = result.get("text", "")
                        if len(text) > 50:
                            text = text[:47] + "..."
                        
                        sentiment = result.get("sentiment", {}).get("label", "unknown")
                        emotion = result.get("emotion", {}).get("label", "none")
                        
                        print(f"{i+1}. {Fore.WHITE}\"{text}\"{Style.RESET_ALL} - Sentiment: {sentiment}, Emotion: {emotion}")
                    
                    if len(results) > 5:
                        print(f"{Fore.YELLOW}... and {len(results) - 5} more entries{Style.RESET_ALL}")
                
                # Clear screen
                elif command == ":clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                
                # Exit
                elif command in [":quit", ":exit", ":q"]:
                    print(f"{Fore.YELLOW}Exiting interactive mode. Goodbye!{Style.RESET_ALL}")
                    
                    # Save readline history
                    try:
                        readline.write_history_file(history_file)
                    except Exception:
                        pass
                    
                    break
                
                # Unknown command
                else:
                    print(f"{Fore.RED}Unknown command: {command}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Type :help for available commands{Style.RESET_ALL}")
            
            # Process text analysis
            else:
                # Analyze the input
                result, formatted = analyze_text(user_input, model, show_probabilities)
                print(f"\n{formatted}\n")
                
                # Add to results history
                results.append(result)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Use :quit to exit{Style.RESET_ALL}")
        
        except Exception as e:
            print_error(f"Error: {e}")
            sys.exit(1)

def run_comparison_mode(
    models: List[SentimentEmotionTransformer],
    show_probabilities: bool = False
) -> None:
    """
    Run interactive comparison mode for real-time model comparison.
    
    Args:
        models: List of models to compare
        show_probabilities: Whether to show raw probabilities
    """
    if not models or len(models) < 2:
        print(f"{Fore.RED}Comparison mode requires at least 2 models. Using interactive mode instead.{Style.RESET_ALL}")
        if models:
            run_interactive_mode(models[0], show_probabilities)
        return
    
    # Configure history file for readline
    history_file = os.path.expanduser("~/.sentiment_comparison_history")
    try:
        readline.read_history_file(history_file)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    
    # Initialize comparison and results storage
    comparison = ModelComparison(models)
    results = []
    
    # Display welcome message
    model_names = [model.name for model in models]
    model_list = ", ".join(model_names)
    
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Interactive Model Comparison{Fore.CYAN}                              ║
╠══════════════════════════════════════════════════════════╣
║ {Fore.WHITE}Comparing {len(models)} models: {model_list[:40]}{'...' if len(model_list) > 40 else ''}{Fore.CYAN} ║
║ {Fore.GREEN}Type any text to analyze with all models.{Fore.CYAN}                 ║
║ {Fore.GREEN}Commands:{Fore.CYAN}                                                 ║
║   {Fore.YELLOW}:help{Fore.CYAN}      - Show this help message                       ║
║   {Fore.YELLOW}:models{Fore.CYAN}    - Show all loaded models                       ║
║   {Fore.YELLOW}:probabilities{Fore.CYAN} - Toggle showing probability distributions  ║
║   {Fore.YELLOW}:export <format> <filename>{Fore.CYAN} - Export comparison results    ║
║               {Fore.WHITE}(formats: json, csv, text, all){Fore.CYAN}             ║
║   {Fore.YELLOW}:stats{Fore.CYAN}     - Show agreement statistics                    ║
║   {Fore.YELLOW}:history{Fore.CYAN}   - Show comparison history                      ║
║   {Fore.YELLOW}:clear{Fore.CYAN}     - Clear the screen                             ║
║   {Fore.YELLOW}:quit{Fore.CYAN}      - Exit comparison mode                         ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    # Main interactive loop
    while True:
        try:
            # Get input from user
            user_input = input(f"{Fore.GREEN}compare> {Style.RESET_ALL}").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Process commands
            if user_input.startswith(":"):
                command = user_input.lower()
                
                # Help command
                if command == ":help":
                    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Available Commands:{Fore.CYAN}                                        ║
╠══════════════════════════════════════════════════════════╣
║   {Fore.YELLOW}:help{Fore.CYAN}      - Show this help message                       ║
║   {Fore.YELLOW}:models{Fore.CYAN}    - Show all loaded models                       ║
║   {Fore.YELLOW}:probabilities{Fore.CYAN} - Toggle showing probability distributions  ║
║   {Fore.YELLOW}:export <format> <filename>{Fore.CYAN} - Export comparison results    ║
║               {Fore.WHITE}(formats: json, csv, text, all){Fore.CYAN}             ║
║   {Fore.YELLOW}:stats{Fore.CYAN}     - Show agreement statistics                    ║
║   {Fore.YELLOW}:history{Fore.CYAN}   - Show comparison history                      ║
║   {Fore.YELLOW}:clear{Fore.CYAN}     - Clear the screen                             ║
║   {Fore.YELLOW}:quit{Fore.CYAN}      - Exit comparison mode                         ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
                
                # Show loaded models
                elif command == ":models":
                    print(f"{Fore.CYAN}Loaded Models:{Style.RESET_ALL}")
                    for i, model in enumerate(models):
                        info = model.get_model_info()
                        print(f"{i+1}. {Fore.WHITE}{info['name']}{Style.RESET_ALL}")
                        print(f"   Sentiment: {info['sentiment_model']}")
                        print(f"   Emotion: {info['emotion_model']}")
                        print(f"   Device: {info['device']}")
                
                # Toggle probabilities
                elif command == ":probabilities":
                    show_probabilities = not show_probabilities
                    print(f"{Fore.YELLOW}Probability display: {Fore.GREEN if show_probabilities else Fore.RED}{show_probabilities}{Style.RESET_ALL}")
                
                # Export results
                elif command.startswith(":export "):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 3:
                        print(f"{Fore.RED}Usage: :export <format> <filename>{Style.RESET_ALL}")
                        continue
                    
                    _, export_format, export_file = parts
                    if export_format.lower() not in ["json", "csv", "text", "all"]:
                        print(f"{Fore.RED}Invalid format. Use json, csv, text, or all.{Style.RESET_ALL}")
                        continue
                    
                    # Check if we have results to export
                    if not results:
                        print(f"{Fore.YELLOW}No comparison results to export yet.{Style.RESET_ALL}")
                        continue
                    
                    # Export results
                    try:
                        export_comparison_results(results, export_format, export_file)
                    except Exception as e:
                        print_error(f"Export failed: {e}")
                
                # Show agreement statistics
                elif command == ":stats":
                    if not results:
                        print(f"{Fore.YELLOW}No comparison results yet.{Style.RESET_ALL}")
                        continue
                    
                    # Calculate overall agreement stats
                    sent_agreements = [r.get("sentiment_agreement", 0.0) for r in results]
                    emo_agreements = [r.get("emotion_agreement", 0.0) for r in results]
                    
                    avg_sent = sum(sent_agreements) / len(sent_agreements) if sent_agreements else 0
                    avg_emo = sum(emo_agreements) / len(emo_agreements) if emo_agreements else 0
                    avg_overall = (avg_sent + avg_emo) / 2
                    
                    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║ {Fore.WHITE}Agreement Statistics (across {len(results)} comparisons){Fore.CYAN}            ║
╠══════════════════════════════════════════════════════════╣
║ {Fore.WHITE}Sentiment Agreement: {Fore.GREEN if avg_sent >= 0.8 else Fore.YELLOW if avg_sent >= 0.5 else Fore.RED}{avg_sent*100:.1f}%{Fore.CYAN}{' ' * 40}║
║ {Fore.WHITE}Emotion Agreement:   {Fore.GREEN if avg_emo >= 0.8 else Fore.YELLOW if avg_emo >= 0.5 else Fore.RED}{avg_emo*100:.1f}%{Fore.CYAN}{' ' * 40}║
║ {Fore.WHITE}Overall Agreement:   {Fore.GREEN if avg_overall >= 0.8 else Fore.YELLOW if avg_overall >= 0.5 else Fore.RED}{avg_overall*100:.1f}%{Fore.CYAN}{' ' * 40}║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")
                
                # Show history
                elif command == ":history":
                    if not results:
                        print(f"{Fore.YELLOW}No comparison history yet.{Style.RESET_ALL}")
                        continue
                    
                    print(f"{Fore.CYAN}Comparison History ({len(results)} entries):{Style.RESET_ALL}")
                    for i, result in enumerate(results[-5:]):  # Show last 5 entries
                        text = result.get("text", "")
                        if len(text) > 50:
                            text = text[:47] + "..."
                        
                        sent_agreement = result.get("sentiment_agreement", 0.0) * 100
                        emo_agreement = result.get("emotion_agreement", 0.0) * 100
                        
                        print(f"{i+1}. {Fore.WHITE}\"{text}\"{Style.RESET_ALL}")
                        print(f"   Sentiment agreement: {sent_agreement:.1f}%, Emotion agreement: {emo_agreement:.1f}%")
                    
                    if len(results) > 5:
                        print(f"{Fore.YELLOW}... and {len(results) - 5} more entries{Style.RESET_ALL}")
                
                # Clear screen
                elif command == ":clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                
                # Exit
                elif command in [":quit", ":exit", ":q"]:
                    print(f"{Fore.YELLOW}Exiting comparison mode. Goodbye!{Style.RESET_ALL}")
                    
                    # Save readline history
                    try:
                        readline.write_history_file(history_file)
                    except Exception:
                        pass
                    
                    break
                
                # Unknown command
                else:
                    print(f"{Fore.RED}Unknown command: {command}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Type :help for available commands{Style.RESET_ALL}")
            
            # Process text comparison
            else:
                # Compare models on the input
                result = comparison.compare(user_input)
                formatted = comparison.format_comparison(result, show_probabilities)
                print(f"\n{formatted}\n")
                
                # Add to results history
                results.append(result)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Use :quit to exit{Style.RESET_ALL}")
        
        except Exception as e:
            print_error(f"Error: {e}")
            sys.exit(1)

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
    
    try:
        # Handle comparison mode
        if args.compare_interactive or args.compare_models:
            models = load_comparison_models(args)
            
            if args.compare_interactive:
                run_comparison_mode(models, args.show_probabilities)
            elif args.text:
                result, formatted = compare_models(args.text, models, args.show_probabilities)
                print(formatted)
                
                if args.output:
                    export_comparison_results([result], args.format, args.output)
            elif args.file:
                results, _ = process_batch_comparison(
                    args.file, models, args.show_probabilities, 
                    args.format if args.output else None, args.output
                )
            return
            
        # Standard mode (single model)
        model = load_transformer_model(
            args.sentiment_model,
            args.emotion_model,
            args.sentiment_threshold,
            args.emotion_threshold,
            args.local_model_path,
            args.model_name,
            settings # Pass settings to load_transformer_model
        )
        
        # Process based on input method
        if args.interactive:
            run_interactive_mode(model, args.show_probabilities)
        
        elif args.text:
            result, formatted = analyze_text(args.text, model, args.show_probabilities)
            print(formatted)
            
            # Export single result if requested
            if args.output:
                export_results([result], args.format, args.output)
        
        elif args.file:
            results, formatted_results = process_batch_file(
                args.file, model, args.show_probabilities,
                args.format if args.output else None, args.output
            )
            
            # Print first few results if not saving to file and not in JSON stream mode
            if not args.output and not getattr(args, "json_stream", False):
                # Check if summary-only mode is enabled
                is_summary_only = getattr(args, "summary_only", False)
                
                if is_summary_only:
                    # In summary-only mode, print results directly without headers
                    for formatted in formatted_results:
                        print(formatted)
                else:
                    # Normal mode with "Result X:" headers
                    max_display = 5
                    for i, formatted in enumerate(formatted_results[:max_display]):
                        print(f"\n{Fore.CYAN}Result {i+1}:{Style.RESET_ALL}")
                        print(formatted)
                    
                    # If more results, show message
                    if len(results) > max_display:
                        print(f"\n{Fore.YELLOW}... and {len(results) - max_display} more results{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Use --output option to save all results to a file{Style.RESET_ALL}")
        
        else:
            # This should never happen due to default behavior in parse_args
            print_error(f"No input method specified. Use --text, --file, or --interactive")
    
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
    
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

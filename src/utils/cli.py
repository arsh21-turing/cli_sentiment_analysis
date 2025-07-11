# src/utils/cli.py
import argparse
import sys
import os
from typing import Dict, List, Optional, Any, Literal
from colorama import init, Fore, Style
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
        help="Confidence threshold for sentiment predictions (0.0-1.0)"
    )
    model_group.add_argument(
        "--emotion-threshold", 
        type=float, 
        default=0.6,
        help="Confidence threshold for emotion predictions (0.0-1.0)"
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
    model_name: Optional[str] = None
) -> SentimentEmotionTransformer:
    """
    Load a transformer model with the specified parameters.
    
    Args:
        sentiment_model: Sentiment model identifier
        emotion_model: Emotion model identifier
        confidence_threshold: Confidence threshold for predictions
        local_model_path: Optional path to locally saved models
        model_name: Optional custom name for the model
        
    Returns:
        Initialized SentimentEmotionTransformer model
    """
    print(f"{Fore.BLUE}Loading model {model_name or sentiment_model}...{Style.RESET_ALL}")
    
    try:
        model = SentimentEmotionTransformer(
            sentiment_model=sentiment_model,
            emotion_model=emotion_model,
            sentiment_threshold=sentiment_threshold,
            emotion_threshold=emotion_threshold,
            local_model_path=local_model_path,
            name=model_name
        )
        print(f"{Fore.GREEN}Model loaded successfully!{Style.RESET_ALL}")
        return model
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
        raise

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
        name=args.model_name or "Default"
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
                    name=name
                )
                models.append(model)
            except Exception as e:
                print(f"{Fore.RED}Error loading comparison model {name}: {e}{Style.RESET_ALL}")
    
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
    formatted = format_analysis_result(result, show_probabilities)
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
            formatted_results = [format_analysis_result(r, True) for r in results]
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(formatted_results))
            print(f"{Fore.GREEN}Results saved to {text_output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving text results: {e}{Style.RESET_ALL}")

    if output_format == "json" or output_format == "all":
        json_output = f"{output_file}.json"
        try:
            from .output import export_to_json
            export_to_json(results, json_output)
            print(f"{Fore.GREEN}Results saved to {json_output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving JSON results: {e}{Style.RESET_ALL}")

    if output_format == "csv" or output_format == "all":
        csv_output = f"{output_file}.csv"
        try:
            from .output import export_to_csv
            export_to_csv(results, csv_output)
            print(f"{Fore.GREEN}Results saved to {csv_output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving CSV results: {e}{Style.RESET_ALL}")

    if output_format not in ("text", "json", "csv", "all"):
        print(f"{Fore.RED}Error saving: Unknown format '{output_format}'{Style.RESET_ALL}")

def export_comparison_results(
    comparison_results: list,
    output_format: str,
    output_file: str
) -> None:
    """
    Export comparison results in the specified format.
    """
    if not comparison_results:
        print(f"{Fore.YELLOW}No comparison results to export{Style.RESET_ALL}")
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
            print(f"{Fore.GREEN}Comparison results saved to {text_output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving text comparison results: {e}{Style.RESET_ALL}")

    if output_format == "json" or output_format == "all":
        json_output = f"{output_file}.json"
        try:
            serializable = [comparison._prepare_for_serialization(r) for r in comparison_results]
            with open(json_output, 'w', encoding='utf-8') as f:
                import json
                json.dump(serializable, f, indent=2)
            print(f"{Fore.GREEN}Comparison results saved to {json_output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving JSON comparison results: {e}{Style.RESET_ALL}")

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
            print(f"{Fore.GREEN}Comparison results saved to {csv_output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving CSV comparison results: {e}{Style.RESET_ALL}")

    if output_format not in ("text", "json", "csv", "all"):
        print(f"{Fore.RED}Error saving: Unknown format '{output_format}'{Style.RESET_ALL}")

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
        print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    results = []
    formatted_results = []
    
    # Process with progress bar
    print(f"{Fore.BLUE}Processing {len(lines)} texts...{Style.RESET_ALL}")
    
    for i, line in enumerate(lines):
        # Show progress
        progress = create_progress_bar(i, len(lines))
        print(f"\r{progress}", end="")
        
        # Analyze text
        result = model.analyze(line)
        formatted = format_analysis_result(result, show_probabilities)
        
        results.append(result)
        formatted_results.append(formatted)
    
    # Complete the progress bar
    print(f"\r{create_progress_bar(len(lines), len(lines))}")
    print(f"{Fore.GREEN}Analysis complete!{Style.RESET_ALL}")
    
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
        print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    comparison = ModelComparison(models)
    results = []
    formatted_results = []
    
    # Process with progress bar
    print(f"{Fore.BLUE}Processing {len(lines)} texts with {len(models)} models...{Style.RESET_ALL}")
    
    for i, line in enumerate(lines):
        # Show progress
        progress = create_progress_bar(i, len(lines))
        print(f"\r{progress}", end="")
        
        # Compare models on text
        result = comparison.compare(line)
        formatted = comparison.format_comparison(result, show_probabilities)
        
        results.append(result)
        formatted_results.append(formatted)
    
    # Complete the progress bar
    print(f"\r{create_progress_bar(len(lines), len(lines))}")
    print(f"{Fore.GREEN}Comparison complete!{Style.RESET_ALL}")
    
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
                        print(f"{Fore.RED}Export failed: {e}{Style.RESET_ALL}")
                
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
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

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
                        print(f"{Fore.RED}Export failed: {e}{Style.RESET_ALL}")
                
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
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

def main():
    """
    Main entry point for CLI.
    """
    # Initialize colorama
    init()
    
    # Parse command line arguments
    args = parse_args()
    
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
            args.model_name
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
            
            # Print first few results if not saving to file
            if not args.output:
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
            print(f"{Fore.RED}No input method specified. Use --text, --file, or --interactive{Style.RESET_ALL}")
    
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)

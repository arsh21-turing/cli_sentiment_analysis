# src/utils/output.py
from typing import Dict, Any, Optional, List
from colorama import Fore, Style
import json
import csv
import os
import io

def format_probability_bars(probability: float, width: int = 20, char: str = "█", 
                           label: str = "", max_label: int = 15) -> str:
    """
    Format a probability value as a visual bar chart.
    
    Args:
        probability: Probability value between 0 and 1
        width: Width of the bar in characters
        char: Character to use for filled portion
        label: Optional label to prepend
        max_label: Maximum length for label (truncates with "…")
        
    Returns:
        Formatted bar string
    """
    # Clamp probability to valid range
    probability = max(0.0, min(1.0, probability))
    
    # Calculate filled portion
    filled = int(probability * width)
    empty = width - filled
    
    # Create bar
    bar = char * filled + "░" * empty
    
    # Handle label
    if label:
        if len(label) > max_label:
            label = label[:max_label-1] + "…"
        return f"{label} | {bar}"
    else:
        return bar

def colorize_sentiment(sentiment: str, color_map: Dict[str, str]) -> str:
    """
    Colorize sentiment text using ANSI color codes.
    
    Args:
        sentiment: Sentiment label
        color_map: Dictionary mapping sentiment to color names
        
    Returns:
        ANSI colorized string
    """
    color_name = color_map.get(sentiment.lower(), "white")
    
    # Map color names to ANSI codes
    color_codes = {
        "red": "\x1b[31m",
        "green": "\x1b[32m", 
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "white": "\x1b[37m"
    }
    
    color_code = color_codes.get(color_name, "\x1b[37m")  # Default to white
    reset_code = "\x1b[0m"
    
    return f"{color_code}{sentiment}{reset_code}"

def format_confidence(confidence: float) -> str:
    """
    Format confidence score with percentage and indicator.
    
    Args:
        confidence: Confidence value between 0 and 1
        
    Returns:
        Formatted confidence string
    """
    percentage = confidence * 100
    indicator = "★" if percentage >= 80 else "☆"
    return f"{percentage:.1f}% {indicator}"

def format_threshold(threshold: float) -> str:
    """
    Format threshold value with percentage.
    
    Args:
        threshold: Threshold value between 0 and 1
        
    Returns:
        Formatted threshold string
    """
    percentage = threshold * 100
    return f"Threshold: {percentage:.0f}%"

def progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a progress bar string.
    
    Args:
        current: Current progress value
        total: Total items to process
        width: Width of the progress bar in characters
        
    Returns:
        Formatted progress bar string
    """
    if total == 0:
        return f"Progress: [{'░' * width}] 0% (0/0)"
    
    progress = current / total
    completed = int(width * progress)
    remaining = width - completed
    
    bar = "█" * completed + "░" * remaining
    percentage = progress * 100
    
    return f"Progress: [{bar}] {percentage:.0f}% ({current}/{total})"

def format_probabilities(probabilities: Dict[str, float]) -> str:
    """
    Format raw probability distributions for display.
    
    Args:
        probabilities: Dictionary mapping labels to probability scores
        
    Returns:
        Formatted string representation of probabilities
    """
    if not probabilities:
        return "No probability data available"
        
    # Sort probabilities by value in descending order
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Format each probability with a bar chart representation
    formatted = []
    for label, prob in sorted_probs:
        # Create a visual bar using block characters
        bar_length = int(prob * 20)  # Scale to 20 chars max
        bar = '█' * bar_length + '░' * (20 - bar_length)
        
        # Format with percentage
        percentage = prob * 100
        formatted.append(f"  {label.ljust(10)} | {bar} {percentage:.1f}%")
    
    return "\n".join(formatted)

def format_sentiment_result(result: Dict[str, Any], show_probabilities: bool = False) -> str:
    """
    Format sentiment analysis result for display.
    
    Args:
        result: Sentiment analysis result dictionary
        show_probabilities: Whether to include raw probabilities
        
    Returns:
        Formatted string representation of sentiment result
    """
    # Handle both string and dict formats
    if isinstance(result, str):
        label = result
        score = 0.0
        probabilities = {}
        is_confident = True
        threshold = 0.7
    else:
        label = result.get("label", result.get("sentiment", "unknown"))
        score = result.get("score", result.get("sentiment_score", 0)) * 100  # Convert to percentage
        probabilities = result.get("raw_probabilities", result.get("sentiment_probabilities", {}))
        is_confident = result.get("confident", True)
        threshold = result.get("threshold", 0.7) * 100  # Convert to percentage
    
    # Color coding based on sentiment
    if label == "positive":
        color = Fore.GREEN
    elif label == "negative":
        color = Fore.RED
    else:
        color = Fore.YELLOW
    
    # Add confidence indicator
    confidence_indicator = f"{Fore.GREEN}✓" if is_confident else f"{Fore.RED}✗"
    
    formatted = f"{color}Sentiment: {label.capitalize()} ({score:.1f}%) {confidence_indicator} [threshold: {threshold:.1f}%]{Style.RESET_ALL}"
    
    # Add probability distribution if requested
    if show_probabilities and probabilities:
        prob_str = format_probabilities(probabilities)
        formatted += f"\n{Fore.CYAN}Sentiment Probabilities:{Style.RESET_ALL}\n{prob_str}"
    
    return formatted

def format_emotion_result(result: Dict[str, Any], show_probabilities: bool = False) -> str:
    """
    Format emotion analysis result for display.
    
    Args:
        result: Emotion analysis result dictionary
        show_probabilities: Whether to include raw probabilities
        
    Returns:
        Formatted string representation of emotion result
    """
    # Handle both string and dict formats
    if isinstance(result, str):
        label = result
        score = 0.0
        probabilities = {}
        is_confident = False
        threshold = 0.6
    else:
        label = result.get("label", result.get("emotion", None))
        score = result.get("score", result.get("emotion_score", 0)) * 100  # Convert to percentage
        probabilities = result.get("raw_probabilities", result.get("emotion_probabilities", {}))
        is_confident = result.get("confident", False)
        threshold = result.get("threshold", 0.6) * 100  # Convert to percentage
    
    # If no emotion detected or score is 0
    if not label or score == 0:
        return f"{Fore.BLUE}Emotion: None detected [threshold: {threshold:.1f}%]{Style.RESET_ALL}"
    
    # Color coding based on emotion
    if label == "joy" or label == "love":
        color = Fore.GREEN
    elif label == "anger" or label == "fear":
        color = Fore.RED
    elif label == "sadness":
        color = Fore.BLUE
    else:
        color = Fore.YELLOW
    
    # Add confidence indicator
    confidence_indicator = f"{Fore.GREEN}✓" if is_confident else f"{Fore.RED}✗"
    
    formatted = f"{color}Emotion: {label.capitalize()} ({score:.1f}%) {confidence_indicator} [threshold: {threshold:.1f}%]{Style.RESET_ALL}"
    
    # Add probability distribution if requested
    if show_probabilities and probabilities:
        prob_str = format_probabilities(probabilities)
        formatted += f"\n{Fore.CYAN}Emotion Probabilities:{Style.RESET_ALL}\n{prob_str}"
    
    return formatted

def format_analysis_result(result: Dict[str, Any], show_probabilities: bool = False) -> str:
    """
    Format complete analysis result for display.
    
    Args:
        result: Complete analysis result dictionary
        show_probabilities: Whether to include raw probabilities
        
    Returns:
        Formatted string representation of complete analysis
    """
    text = result.get("text", "")
    
    # Handle new nested structure
    if "sentiment" in result and isinstance(result["sentiment"], dict):
        sentiment_result = result["sentiment"]
    else:
        # Create sentiment result dict from the analysis result (backward compatibility)
        sentiment_result = {
            "label": result.get("sentiment", "unknown"),
            "score": result.get("sentiment_score", 0),
            "raw_probabilities": result.get("sentiment_probabilities", {}),
            "confident": result.get("sentiment_confident", True),
            "threshold": result.get("sentiment_threshold", 0.7)
        }
    
    if "emotion" in result and isinstance(result["emotion"], dict):
        emotion_result = result["emotion"]
    else:
        # Create emotion result dict from the analysis result (backward compatibility)
        emotion_result = {
            "label": result.get("emotion", None),
            "score": result.get("emotion_score", 0),
            "raw_probabilities": result.get("emotion_probabilities", {}),
            "confident": result.get("emotion_confident", False),
            "threshold": result.get("emotion_threshold", 0.6)
        }
    
    # Format the text input with limit to prevent huge outputs
    max_length = 100
    if len(text) > max_length:
        text_display = text[:max_length] + "..."
    else:
        text_display = text
    
    formatted = [f"{Fore.WHITE}Text: \"{text_display}\"{Style.RESET_ALL}"]
    formatted.append(format_sentiment_result(sentiment_result, show_probabilities))
    formatted.append(format_emotion_result(emotion_result, show_probabilities))
    
    return "\n".join(formatted)

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a progress bar for batch processing.
    
    Args:
        current: Current progress value
        total: Total items to process
        width: Width of the progress bar in characters
        
    Returns:
        Formatted progress bar string
    """
    progress = current / total
    completed = int(width * progress)
    remaining = width - completed
    
    bar = "█" * completed + "░" * remaining
    percentage = progress * 100
    
    return f"Progress: [{bar}] {percentage:.1f}% ({current}/{total})"

def flatten_result_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a nested result dictionary to a flat dictionary for CSV export.
    
    Args:
        result: Complete analysis result dictionary
        
    Returns:
        Flattened dictionary with keys suitable for CSV
    """
    flattened = {
        "text": result.get("text", ""),
        "sentiment": result.get("sentiment", "unknown"),
        "sentiment_score": result.get("sentiment_score", 0),
        "emotion": result.get("emotion", ""),
        "emotion_score": result.get("emotion_score", 0),
        "confidence": result.get("confidence", 0)
    }
    
    # Add sentiment probabilities with prefixed keys
    sentiment_probs = result.get("sentiment_probabilities", {})
    for label, prob in sentiment_probs.items():
        flattened[f"sentiment_prob_{label}"] = prob
    
    # Add emotion probabilities with prefixed keys
    emotion_probs = result.get("emotion_probabilities", {})
    for label, prob in emotion_probs.items():
        flattened[f"emotion_prob_{label}"] = prob
    
    return flattened

def export_to_json(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Export analysis results to JSON file.
    
    Args:
        results: List of analysis result dictionaries
        filepath: Path to save the JSON file
        
    Raises:
        IOError: If unable to write to file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    try:
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
    except Exception as e:
        raise IOError(f"Failed to export to JSON: {e}")

def export_to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Export analysis results to CSV file.
    
    Args:
        results: List of analysis result dictionaries
        filepath: Path to save the CSV file
        
    Raises:
        IOError: If unable to write to file
    """
    # Always include these fields
    required_fields = [
        "text", "sentiment", "sentiment_score", "emotion", "emotion_score", "model",
        "positive", "neutral", "negative"
    ]
    flattened_results = []
    for result in results:
        row = {k: "" for k in required_fields}
        row["text"] = result.get("text", "")
        row["model"] = result.get("model", "")
        # Sentiment
        if "sentiment" in result and isinstance(result["sentiment"], dict):
            row["sentiment"] = result["sentiment"].get("label", "")
            row["sentiment_score"] = result["sentiment"].get("score", "")
        else:
            row["sentiment"] = result.get("sentiment", "")
            row["sentiment_score"] = result.get("sentiment_score", "")
        # Emotion
        if "emotion" in result and isinstance(result["emotion"], dict):
            row["emotion"] = result["emotion"].get("label", "")
            row["emotion_score"] = result["emotion"].get("score", "")
        else:
            row["emotion"] = result.get("emotion", "")
            row["emotion_score"] = result.get("emotion_score", "")
        # Probabilities
        probs = result.get("probabilities", {})
        row["positive"] = probs.get("positive", "")
        row["neutral"] = probs.get("neutral", "")
        row["negative"] = probs.get("negative", "")
        flattened_results.append(row)
    if not flattened_results:
        raise ValueError("No results to export")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    try:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=required_fields)
        writer.writeheader()
        writer.writerows(flattened_results)
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(output.getvalue())
    except Exception as e:
        raise IOError(f"Failed to export to CSV: {e}")

#!/usr/bin/env python3
"""
Smart CLI Sentiment & Emotion Analyzer
A command-line tool for sentiment and emotion analysis of text.
"""

import argparse
import sys
import os
import time
from typing import Dict, Any, Optional, List
from collections import Counter
import csv
import json
from tqdm import tqdm

# Use colorama module for color utilities, import init dynamically for easier mocking
import colorama
from colorama import Fore, Style

# Import our transformer model
from .models.transformer import SentimentEmotionTransformer

# Ensure project root is on PYTHONPATH for subprocess tests
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Always propagate project root to subprocess environment via PYTHONPATH
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

existing_py_path = os.environ.get('PYTHONPATH', '')
if _PROJECT_ROOT not in existing_py_path.split(os.pathsep):
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [existing_py_path, _PROJECT_ROOT]))

def print_with_color(text: str, color: str, bold: bool = False) -> None:
    """Print text with specified color and optional bold formatting."""
    style_prefix = Style.BRIGHT if bold else ""
    print(f"{style_prefix}{color}{text}{Style.RESET_ALL}")

def format_sentiment(sentiment: str, score: float) -> None:
    """Format and print sentiment with appropriate color."""
    if sentiment == "positive":
        color = Fore.GREEN
        emoji = "😊"
    elif sentiment == "negative":
        color = Fore.RED
        emoji = "😞"
    else:  # neutral
        color = Fore.BLUE
        emoji = "😐"
        
    print_with_color(
        f"Sentiment: {sentiment.upper()} {emoji} (Score: {score:.2f})",
        color,
        bold=True
    )

def format_emotion(emotion: str, score: float) -> None:
    """Format and print emotion with appropriate color."""
    if not emotion:
        return
        
    # Map emotions to colors and emojis
    emotion_map = {
        "joy": (Fore.YELLOW, "😄"),
        "sadness": (Fore.BLUE, "😢"),
        "anger": (Fore.RED, "😠"),
        "fear": (Fore.MAGENTA, "😨"),
        "surprise": (Fore.CYAN, "😲"),
        "love": (Fore.LIGHTMAGENTA_EX, "❤️"),
        # Default for any other emotions
        "default": (Fore.WHITE, "🤔")
    }
    
    color, emoji = emotion_map.get(emotion.lower(), emotion_map["default"])
    print_with_color(
        f"Emotion: {emotion.upper()} {emoji} (Score: {score:.2f})",
        color,
        bold=True
    )

def show_progress(message: str) -> None:
    """Show a simple progress message."""
    sys.stdout.write(f"\r{message}...")
    sys.stdout.flush()

def process_file(file_path: str, analyzer: SentimentEmotionTransformer, 
                batch_size: int = 10, output_format: str = None) -> Dict[str, Any]:
    """
    Process a text file line by line and analyze sentiment/emotion.
    
    Args:
        file_path: Path to the text file to analyze
        analyzer: The SentimentEmotionTransformer instance
        batch_size: Number of lines to process in each batch
        output_format: Optional format for saving results (csv/json)
        
    Returns:
        Dictionary with summary statistics
    """
    if not os.path.exists(file_path):
        print_with_color(f"Error: File not found - {file_path}", Fore.RED, bold=True)
        return {}
    
    # Prepare for batched processing
    results = []
    sentiment_counts = Counter()
    emotion_counts = Counter()
    total_sentiment_score = 0.0
    total_emotion_score = 0.0
    emotion_count = 0  # Count of entries with emotions (for averaging)
    
    # Read all lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    total_lines = len(lines)
    if total_lines == 0:
        print_with_color("Error: File is empty", Fore.RED, bold=True)
        return {}
    
    print_with_color(f"Processing {total_lines} lines from {file_path}", Fore.CYAN, bold=True)
    print(f"Processing in batches of {batch_size} lines...")
    print()
    
    # Process in batches with progress bar
    with tqdm(total=total_lines, desc="Analyzing", unit="lines") as progress_bar:
        for i in range(0, total_lines, batch_size):
            batch = lines[i:i+batch_size]
            batch_results = []
            
            for line in batch:
                if not line:
                    continue
                    
                # Analyze the line
                result = analyzer.analyze(line)
                
                # Truncate text if too long for display
                display_text = line[:100] + "..." if len(line) > 100 else line
                result["display_text"] = display_text
                
                # Update statistics
                sentiment_counts[result["sentiment"]] += 1
                total_sentiment_score += result["sentiment_score"]
                
                if result["emotion"]:
                    emotion_counts[result["emotion"]] += 1
                    total_emotion_score += result["emotion_score"]
                    emotion_count += 1
                
                batch_results.append(result)
            
            # Update the results list
            results.extend(batch_results)
            
            # Update progress bar
            progress_bar.update(len(batch))
    
    # Calculate summary statistics
    summary = {
        "total_lines": total_lines,
        "sentiment_distribution": {
            sentiment: count / total_lines for sentiment, count in sentiment_counts.items()
        },
        "emotion_distribution": {
            emotion: count / emotion_count if emotion_count else 0 
            for emotion, count in emotion_counts.items()
        },
        "avg_sentiment_score": total_sentiment_score / total_lines if total_lines else 0,
        "avg_emotion_score": total_emotion_score / emotion_count if emotion_count else 0,
        "sentiment_counts": dict(sentiment_counts),
        "emotion_counts": dict(emotion_counts),
        "results": results
    }
    
    # Save results if output format is specified
    if output_format:
        save_results(results, file_path, output_format)
    
    return summary

def save_results(results: List[Dict[str, Any]], input_file_path: str, format: str) -> None:
    """
    Save analysis results to a file.
    
    Args:
        results: List of analysis results
        input_file_path: Original input file path
        format: Output format (csv or json)
    """
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    if format.lower() == 'csv':
        output_path = f"{base_name}_analysis.csv"
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['text', 'sentiment', 'sentiment_score', 'emotion', 'emotion_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'text': result.get('display_text', ''),
                    'sentiment': result.get('sentiment', ''),
                    'sentiment_score': result.get('sentiment_score', 0),
                    'emotion': result.get('emotion', ''),
                    'emotion_score': result.get('emotion_score', 0)
                })
        
        print_with_color(f"Results saved to {output_path}", Fore.GREEN)
    
    elif format.lower() == 'json':
        output_path = f"{base_name}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=2)
            
        print_with_color(f"Results saved to {output_path}", Fore.GREEN)
    
    else:
        print_with_color(f"Unsupported output format: {format}", Fore.RED)

def display_summary(summary: Dict[str, Any]) -> None:
    """Display summary statistics of file analysis."""
    if not summary:
        return
    
    print("\n" + "="*60)
    print_with_color("📊 ANALYSIS SUMMARY", Fore.CYAN, bold=True)
    print("="*60)
    
    print(f"Total lines analyzed: {summary['total_lines']}")
    print()
    
    # Sentiment distribution
    print_with_color("Sentiment Distribution:", Fore.CYAN, bold=True)
    for sentiment, percentage in sorted(
            summary['sentiment_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
        # Choose color based on sentiment
        if sentiment == "positive":
            color = Fore.GREEN
        elif sentiment == "negative":
            color = Fore.RED
        else:
            color = Fore.BLUE
            
        # Print bar chart
        bar_length = int(percentage * 40)
        bar = '█' * bar_length
        print_with_color(
            f"{sentiment.upper():10}: {bar} {percentage*100:.1f}% ({summary['sentiment_counts'][sentiment]})",
            color
        )
    
    print()
    
    # Emotion distribution (if any emotions found)
    if summary.get('emotion_counts'):
        print_with_color("Emotion Distribution:", Fore.CYAN, bold=True)
        
        emotion_map = {
            "joy": Fore.YELLOW,
            "sadness": Fore.BLUE,
            "anger": Fore.RED,
            "fear": Fore.MAGENTA,
            "surprise": Fore.CYAN,
            "love": Fore.LIGHTMAGENTA_EX
        }
        
        for emotion, percentage in sorted(
                summary['emotion_distribution'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
            # Choose color based on emotion
            color = emotion_map.get(emotion.lower(), Fore.WHITE)
            
            # Print bar chart
            bar_length = int(percentage * 40) if percentage else 0
            bar = '█' * bar_length
            print_with_color(
                f"{emotion:10}: {bar} {percentage*100:.1f}% ({summary['emotion_counts'][emotion]})",
                color
            )
    
    print()
    print_with_color(
        f"Average sentiment score: {summary['avg_sentiment_score']:.2f}", 
        Fore.WHITE, 
        bold=True
    )
    
    if summary['avg_emotion_score'] > 0:
        print_with_color(
            f"Average emotion score: {summary['avg_emotion_score']:.2f}", 
            Fore.WHITE, 
            bold=True
        )
    
    print("="*60)

def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment and emotion in text"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", 
        type=str,
        help="Text to analyze for sentiment and emotion"
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to a text file for batch analysis (one text per line)"
    )
    
    parser.add_argument(
        "--sentiment-model",
        type=str,
        help="Hugging Face model for sentiment analysis (default: nlptown/bert-base-multilingual-uncased-sentiment)"
    )
    
    parser.add_argument(
        "--emotion-model",
        type=str,
        help="Hugging Face model for emotion detection (default: bhadresh-savani/distilbert-base-uncased-emotion)"
    )
    
    parser.add_argument(
        "--sentiment-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for sentiment classification (default: 0.6)"
    )
    
    parser.add_argument(
        "--emotion-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for emotion detection (default: 0.5)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for file processing (default: 10)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "json"],
        help="Save results to file in the specified format"
    )
    
    args = parser.parse_args()
    
    # Show a welcome message
    print_with_color("🔍 Smart CLI Sentiment & Emotion Analyzer", Fore.CYAN, bold=True)
    print()
    
    # Initialize colorama for cross-platform colored output
    colorama.init()
    
    # Initialize the transformer with progress indicator
    show_progress("Loading models")
    
    analyzer = SentimentEmotionTransformer(
        sentiment_model_name=args.sentiment_model,
        emotion_model_name=args.emotion_model,
        sentiment_threshold=args.sentiment_threshold,
        emotion_threshold=args.emotion_threshold
    )
    
    # Clear the progress indicator
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()

    if args.text:
        # Process single text input
        print_with_color(f"Analyzing: \"{args.text}\"", Fore.WHITE)
        print()
        
        # Add a small animation to show processing
        show_progress("Analyzing sentiment and emotion")
        result = analyzer.analyze(args.text)
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        
        # Format and display results
        format_sentiment(result["sentiment"], result["sentiment_score"])
        
        if result["emotion"]:
            format_emotion(result["emotion"], result["emotion_score"])
        elif result["sentiment"] == "negative":
            print_with_color("No specific emotion detected above threshold", Fore.YELLOW)
        
        print()
        print_with_color("✨ Analysis complete", Fore.CYAN)
    
    elif args.file:
        # Process file
        summary = process_file(
            args.file, 
            analyzer, 
            batch_size=args.batch_size, 
            output_format=args.output_format
        )
        
        # Display summary
        if summary:
            display_summary(summary)
            print_with_color("✨ File analysis complete", Fore.CYAN)

if __name__ == "__main__":
    main()

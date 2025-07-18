"""
Export utilities for analysis results with comprehensive error handling.
"""

import csv
import json
import os
from typing import List, Dict, Any, Optional
from .validation import ValidationError, validate_export_path


def flatten_result(result: Dict[str, Any], include_probabilities: bool = False) -> Dict[str, Any]:
    """
    Flatten a result dictionary for CSV export.
    
    Args:
        result: Analysis result to flatten
        include_probabilities: Whether to include raw probabilities
        
    Returns:
        Flattened result dictionary
    """
    flattened = {
        'text': result.get('text', ''),
        'timestamp': result.get('timestamp', ''),
    }
    
    # Add sentiment data
    if 'sentiment' in result:
        sentiment = result['sentiment']
        flattened.update({
            'sentiment_label': sentiment.get('label', ''),
            'sentiment_confidence': sentiment.get('confidence', 0.0),
        })
        
        if include_probabilities and 'raw_probabilities' in sentiment:
            for label, prob in sentiment['raw_probabilities'].items():
                flattened[f'sentiment_prob_{label}'] = prob
    
    # Add emotion data
    if 'emotion' in result:
        emotion = result['emotion']
        flattened.update({
            'emotion_label': emotion.get('label', ''),
            'emotion_confidence': emotion.get('confidence', 0.0),
        })
        
        if include_probabilities and 'raw_probabilities' in emotion:
            for label, prob in emotion['raw_probabilities'].items():
                flattened[f'emotion_prob_{label}'] = prob
    
    # Add metadata
    if 'metadata' in result:
        metadata = result['metadata']
        flattened.update({
            'model_used': metadata.get('model_used', ''),
            'processing_time': metadata.get('processing_time', 0.0),
            'fallback_used': metadata.get('fallback_used', False),
        })
    
    return flattened


def export_to_csv(results: List[Dict[str, Any]], file_path: str, include_probabilities: bool = False) -> None:
    """
    Export results to CSV file with error handling.
    
    Args:
        results: Analysis results to export
        file_path: Path to export to
        include_probabilities: Whether to include raw probabilities
        
    Raises:
        ValidationError: If export fails
    """
    try:
        # Validate export path
        validate_export_path(file_path)
        
        # Flatten results for CSV
        flattened = [flatten_result(r, include_probabilities) for r in results]
        
        if not flattened:
            raise ValidationError(
                "No results to export.",
                "Analyze some text before exporting."
            )
        
        # Write to CSV
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
            writer.writeheader()
            writer.writerows(flattened)
            
    except csv.Error as e:
        raise ValidationError(
            f"CSV export error: {str(e)}",
            "Check file permissions and disk space."
        )
    except IOError as e:
        raise ValidationError(
            f"File error during CSV export: {str(e)}",
            "Check file permissions and disk space."
        )


def export_to_json(results: List[Dict[str, Any]], file_path: str, include_probabilities: bool = False) -> None:
    """
    Export results to JSON file with error handling.
    
    Args:
        results: Analysis results to export
        file_path: Path to export to
        include_probabilities: Whether to include raw probabilities
        
    Raises:
        ValidationError: If export fails
    """
    try:
        # Validate export path
        validate_export_path(file_path)
        
        if not results:
            raise ValidationError(
                "No results to export.",
                "Analyze some text before exporting."
            )
        
        # Process results for JSON
        if not include_probabilities:
            # Remove raw probabilities to make output cleaner
            processed_results = []
            for result in results:
                processed = result.copy()
                
                if "sentiment" in processed and "raw_probabilities" in processed["sentiment"]:
                    processed["sentiment"] = {
                        k: v for k, v in processed["sentiment"].items() 
                        if k != "raw_probabilities"
                    }
                
                if "emotion" in processed and "raw_probabilities" in processed["emotion"]:
                    processed["emotion"] = {
                        k: v for k, v in processed["emotion"].items() 
                        if k != "raw_probabilities"
                    }
                
                processed_results.append(processed)
        else:
            # Keep raw probabilities
            processed_results = results
        
        # Write to JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)
            
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"JSON export error: {str(e)}",
            "There may be some invalid characters in the results."
        )
    except IOError as e:
        raise ValidationError(
            f"File error during JSON export: {str(e)}",
            "Check file permissions and disk space."
        ) 
# ---------------------------------------------------------------------------
# Optional progress-bar dependency
# ---------------------------------------------------------------------------
import os
try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback for environments w/o tqdm

    import sys
    import types

    def tqdm(iterable, *_, **__):  # type: ignore
        """Fallback tqdm returning input iterable unchanged with no progress bar."""
        return iterable

    # Expose a dummy *tqdm* module so tests can patch 'tqdm.tqdm'
    _dummy_mod = types.ModuleType("tqdm")
    _dummy_mod.tqdm = tqdm  # type: ignore[attr-defined]
    sys.modules.setdefault("tqdm", _dummy_mod)
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Iterator
import statistics
from datetime import datetime
import json
import csv

from ..utils.file_loader import load_data, get_file_iterator, TextFileIterator
from importlib import import_module


def process_batch(file_path: str, 
                 analyzer, 
                 batch_size: Optional[int] = 100, 
                 parallel: bool = False, 
                 max_workers: int = 4,
                 **file_loader_kwargs) -> Dict[str, Any]:
    """
    Process a batch of texts from a file path and analyze them.
    
    Parameters:
    - file_path (str): Path to either a directory or a file
    - analyzer: The analyzer model to use for text analysis
    - batch_size (int, optional): Number of texts to process in each batch
    - parallel (bool, optional): Whether to use parallel processing
    - max_workers (int): Maximum number of worker threads for parallel processing
    - **file_loader_kwargs: Additional arguments for the file loader 
      (extensions, csv_column, recursive, etc.)
    
    Returns:
    - dict: Contains individual results and aggregate statistics
    """
    # Create a file iterator
    try:
        file_iterator = get_file_iterator(
            file_path=file_path,
            batch_size=batch_size,
            **file_loader_kwargs
        )
        
        # Process with the iterator
        return process_with_iterator(
            file_iterator=file_iterator,
            analyzer=analyzer,
            parallel=parallel,
            max_workers=max_workers
        )
    except ValueError as e:
        raise ValueError(f"Error creating file iterator: {str(e)}")


def process_with_iterator(file_iterator: TextFileIterator,
                         analyzer,
                         parallel: bool = False,
                         max_workers: int = 4) -> Dict[str, Any]:
    """
    Process texts using an existing file iterator.
    
    Parameters:
    - file_iterator: A TextFileIterator instance
    - analyzer: The analyzer model to use
    - parallel (bool): Whether to use parallel processing
    - max_workers (int): Maximum number of worker threads
    
    Returns:
    - dict: Contains individual results and aggregate statistics
    """
    # Start timing
    start_time = datetime.now()
    
    # Display file path information
    path_display = file_iterator.path
    if os.path.isfile(path_display):
        path_display = os.path.basename(path_display)
        
    print(f"\nProcessing texts from {path_display}")
    if file_iterator.recursive and os.path.isdir(file_iterator.path):
        print(f"  (Searching recursively for files with extensions: {', '.join(file_iterator.extensions)})")
    
    # Initialize results
    all_results = []
    batch_count = 0
    total_texts = 0
    
    # Process batches
    try:
        for batch_texts in file_iterator:
            batch_count += 1
            total_texts += len(batch_texts)
            
            print(f"\nProcessing batch {batch_count} ({len(batch_texts)} texts)")
            
            # Process this batch
            if parallel and len(batch_texts) > 1:
                batch_results = _process_parallel(batch_texts, analyzer, max_workers)
            else:
                batch_results = _process_sequential(batch_texts, analyzer)
            
            all_results.extend(batch_results)
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
    finally:
        # Close the file iterator
        file_iterator.close()
    
    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    
    # Compile statistics
    stats = _compile_statistics(all_results)
    
    # Add timing information
    stats["processing_time"] = {
        "total_seconds": elapsed_seconds,
        "texts_per_second": total_texts / elapsed_seconds if elapsed_seconds > 0 else 0,
        "formatted": str(elapsed_time)
    }
    
    # Build and return result dictionary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "source": file_iterator.path,
        "total_texts": total_texts,
        "batch_count": batch_count,
        "timestamp": timestamp,
        "individual_results": all_results,
        "statistics": stats
    }


def _get_tqdm():
    """Fetch the current tqdm.tqdm callable (after any monkey-patching)."""
    try:
        return import_module("tqdm").tqdm  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover – fallback
        return tqdm  # type: ignore[name-defined]


def _process_sequential(texts: List[str], analyzer) -> List[Dict[str, Any]]:
    """Process texts sequentially with a progress bar."""
    bar = _get_tqdm()
    results = []
    for text in bar(texts, desc="Analyzing", unit="text"):
        try:
            result = analyzer.analyze(text)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate long texts
                "analysis": result
            })
        except Exception as e:
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "error": str(e)
            })
    return results


def _process_parallel(texts: List[str], analyzer, max_workers: int) -> List[Dict[str, Any]]:
    """Process texts in parallel with a progress bar."""
    results = [None] * len(texts)  # Pre-allocate results list
    
    # Define the worker function
    def process_text(idx_text):
        idx, text = idx_text
        try:
            result = analyzer.analyze(text)
            return idx, {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "analysis": result
            }
        except Exception as e:
            return idx, {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "error": str(e)
            }
    
    bar = _get_tqdm()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_text, (i, text)): i for i, text in enumerate(texts)}
        
        for future in bar(concurrent.futures.as_completed(futures),
                         total=len(texts),
                         desc="Analyzing",
                         unit="text"):
            idx, result = future.result()
            results[idx] = result
    
    return results


def _compile_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile statistics from individual results."""
    # Extract sentiment and emotion data
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "error": 0}
    emotion_counts = {}
    confidence_scores = []
    error_count = 0
    
    for result in results:
        if "error" in result:
            error_count += 1
            sentiment_counts["error"] += 1
            continue
            
        analysis = result["analysis"]
        
        # Count sentiments
        if "sentiment" in analysis:
            sentiment = analysis["sentiment"]["label"]
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            # Track confidence scores
            if "score" in analysis["sentiment"]:
                confidence_scores.append(analysis["sentiment"]["score"])
        
        # Count emotions
        if "emotion" in analysis:
            emotion = analysis["emotion"]["label"]
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
    
    # Calculate confidence statistics if we have scores
    confidence_stats = {}
    if confidence_scores:
        confidence_stats = {
            "avg_confidence": statistics.mean(confidence_scores),
            "min_confidence": min(confidence_scores),
            "max_confidence": max(confidence_scores)
        }
        if len(confidence_scores) > 1:
            confidence_stats["std_dev"] = statistics.stdev(confidence_scores)
    
    return {
        "sentiment_distribution": sentiment_counts,
        "emotion_distribution": emotion_counts,
        "confidence_stats": confidence_stats,
        "error_rate": error_count / len(results) if results else 0,
        "processed_count": len(results) - error_count
    }


def export_batch_results(results: Dict[str, Any], format: str) -> str:
    """
    Export batch processing results to a file.
    
    Parameters:
    - results: Batch processing results
    - format: Export format ("csv" or "json")
    
    Returns:
    - str: Path to the exported file
    """
    # Generate filename based on source and timestamp
    source_name = os.path.basename(results["source"]) or "batch_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "csv":
        # Export to CSV
        filename = f"{source_name}_{timestamp}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["Text", "Sentiment", "Sentiment Score", "Emotion", "Emotion Score", "Error"])
            
            # Write data
            for item in results["individual_results"]:
                if "error" in item:
                    writer.writerow([item["text"], "", "", "", "", item["error"]])
                else:
                    analysis = item["analysis"]
                    sentiment = analysis.get("sentiment", {})
                    emotion = analysis.get("emotion", {})
                    
                    writer.writerow([
                        item["text"],
                        sentiment.get("label", ""),
                        sentiment.get("score", ""),
                        emotion.get("label", ""),
                        emotion.get("score", ""),
                        ""
                    ])
    else:  # json
        # Export to JSON
        filename = f"{source_name}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    return filename


def display_batch_summary(results: Dict[str, Any]) -> None:
    """Display a summary of batch processing results."""
    stats = results["statistics"]
    
    print("\n===== Batch Processing Summary =====")
    print(f"Source: {results['source']}")
    print(f"Total texts processed: {results['total_texts']}")
    print(f"Batch count: {results.get('batch_count', 1)}")
    print(f"Timestamp: {results['timestamp']}")
    
    # Display processing time if available
    if "processing_time" in stats:
        print(f"\n--- Processing Time ---")
        print(f"Total: {stats['processing_time']['formatted']}")
        print(f"Texts per second: {stats['processing_time']['texts_per_second']:.2f}")
    
    print("\n--- Sentiment Distribution ---")
    for sentiment, count in stats["sentiment_distribution"].items():
        pct = count/results['total_texts']*100 if results['total_texts'] > 0 else 0
        print(f"{sentiment}: {count} texts ({pct:.1f}%)")
    
    if stats["emotion_distribution"]:
        print("\n--- Emotion Distribution ---")
        for emotion, count in sorted(stats["emotion_distribution"].items(), 
                                   key=lambda x: x[1], 
                                   reverse=True):
            print(f"{emotion}: {count} texts")
    
    if stats["confidence_stats"]:
        print("\n--- Confidence Stats ---")
        print(f"Average confidence: {stats['confidence_stats']['avg_confidence']:.2f}")
        print(f"Range: {stats['confidence_stats']['min_confidence']:.2f} - {stats['confidence_stats']['max_confidence']:.2f}")
        if "std_dev" in stats["confidence_stats"]:
            print(f"Standard deviation: {stats['confidence_stats']['std_dev']:.2f}")
    
    print("\n--- Error Rate ---")
    print(f"Errors: {stats['error_rate']*100:.1f}% ({int(stats['error_rate']*results['total_texts'])} texts)")
    print("====================================")

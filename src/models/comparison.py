from typing import Dict, List, Any, Optional, Tuple, Union
from ..models.transformer import SentimentEmotionTransformer
import time
import json
import csv
import os
from colorama import Fore, Style

class ModelComparison:
    """
    Utility for comparing multiple sentiment and emotion analysis models.
    
    Features:
    - Side-by-side comparison of multiple models
    - Agreement statistics between models
    - Export of comparison results in various formats
    """
    
    def __init__(self, models: Optional[List[SentimentEmotionTransformer]] = None):
        """
        Initialize the model comparison utility.
        
        Args:
            models: List of SentimentEmotionTransformer models to compare
        """
        self.models = models or []
    
    def add_model(self, model: SentimentEmotionTransformer) -> None:
        """
        Add a model to the comparison set.
        
        Args:
            model: SentimentEmotionTransformer model to add
        """
        self.models.append(model)
    
    def compare(self, text: str) -> Dict[str, Any]:
        """
        Run all models on the text and compare their predictions.
        
        Args:
            text: Text to analyze with all models
            
        Returns:
            Comparison results dictionary
        """
        if not self.models:
            raise ValueError("No models available for comparison")
        
        # Run all models on the text
        results = []
        times = []
        
        for model in self.models:
            start_time = time.time()
            result = model.analyze(text)
            end_time = time.time()
            
            # Add execution time to result
            result["execution_time"] = end_time - start_time
            results.append(result)
            times.append(end_time - start_time)
        
        # Calculate agreement statistics
        sentiment_agreement = self._calculate_agreement(results, "sentiment", "label")
        emotion_agreement = self._calculate_agreement(results, "emotion", "label")
        
        # Identify differences
        sentiment_differences = self._identify_differences(results, "sentiment", "label")
        emotion_differences = self._identify_differences(results, "emotion", "label")
        
        # Find best model (highest confidence)
        best_model_idx = self._find_best_model(results)
        
        return {
            "text": text,
            "model_count": len(self.models),
            "models": [model.get_model_info() for model in self.models],
            "results": results,
            "execution_times": times,
            "average_time": sum(times) / len(times) if times else 0,
            "sentiment_agreement": sentiment_agreement,
            "emotion_agreement": emotion_agreement,
            "sentiment_differences": sentiment_differences,
            "emotion_differences": emotion_differences,
            "best_model_index": best_model_idx,
            "best_model_name": self.models[best_model_idx].name if best_model_idx is not None else None,
        }
    
    def _calculate_agreement(self, results: List[Dict[str, Any]], feature: str, label_key: str) -> float:
        """
        Calculate agreement percentage for a feature across all models.
        
        Args:
            results: List of analysis results
            feature: Feature to check agreement for (e.g., "sentiment", "emotion")
            label_key: Key to access the label in the result dict
            
        Returns:
            Agreement percentage (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        # Extract labels from results
        labels = []
        for result in results:
            if feature in result and label_key in result[feature]:
                labels.append(result[feature][label_key])
            else:
                labels.append(None)
        
        # Count agreements (matches with most common label)
        if not labels:
            return 0.0
        
        # Find most common label (excluding None)
        non_none_labels = [l for l in labels if l is not None]
        if not non_none_labels:
            return 0.0
        
        # Count occurrences of each label
        label_counts = {}
        for label in non_none_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find most common label
        most_common = max(label_counts.items(), key=lambda x: x[1])
        most_common_label, most_common_count = most_common
        
        # Calculate agreement percentage
        agreement = most_common_count / len(results)
        
        return agreement
    
    def _identify_differences(self, results: List[Dict[str, Any]], feature: str, label_key: str) -> List[Tuple[str, str, float]]:
        """
        Identify models with differing predictions.
        
        Args:
            results: List of analysis results
            feature: Feature to check (e.g., "sentiment", "emotion")
            label_key: Key to access the label in the result dict
            
        Returns:
            List of tuples (model_name, different_label, confidence_score)
        """
        if not results or len(results) < 2:
            return []
        
        # Extract labels and compute most common
        labels = []
        model_names = []
        confidence_scores = []
        
        for result in results:
            model_names.append(result.get("model", "Unknown"))
            
            if feature in result and label_key in result[feature]:
                labels.append(result[feature][label_key])
                confidence_scores.append(result[feature].get("score", 0.0))
            else:
                labels.append(None)
                confidence_scores.append(0.0)
        
        # Find most common label (excluding None)
        non_none_labels = [l for l in labels if l is not None]
        if not non_none_labels:
            return []
        
        # Count occurrences of each label
        label_counts = {}
        for label in non_none_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find most common label
        most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]
        
        # Identify differences
        differences = []
        for i, label in enumerate(labels):
            if label is not None and label != most_common_label:
                differences.append((model_names[i], label, confidence_scores[i]))
        
        return differences
    
    def _find_best_model(self, results: List[Dict[str, Any]]) -> Optional[int]:
        """
        Find the model with the highest confidence score.
        
        Args:
            results: List of analysis results
            
        Returns:
            Index of the best model or None if no valid results
        """
        if not results:
            return None
        
        best_score = -1
        best_idx = None
        
        for i, result in enumerate(results):
            confidence = result.get("confidence", 0.0)
            if confidence > best_score:
                best_score = confidence
                best_idx = i
        
        return best_idx
    
    def get_agreement_stats(self, comparison_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate detailed agreement statistics for comparison results.
        
        Args:
            comparison_result: Result from compare() method
            
        Returns:
            Dictionary with agreement statistics
        """
        return {
            "sentiment_agreement": comparison_result.get("sentiment_agreement", 0.0),
            "emotion_agreement": comparison_result.get("emotion_agreement", 0.0),
            "overall_agreement": (
                comparison_result.get("sentiment_agreement", 0.0) + 
                comparison_result.get("emotion_agreement", 0.0)
            ) / 2,
        }
    
    def format_comparison(self, comparison_result: Dict[str, Any], show_probabilities: bool = False) -> str:
        """
        Format comparison results for display.
        
        Args:
            comparison_result: Result from compare() method
            show_probabilities: Whether to show probability distributions
            
        Returns:
            Formatted string representation of comparison
        """
        text = comparison_result.get("text", "")
        results = comparison_result.get("results", [])
        sentiment_agreement = comparison_result.get("sentiment_agreement", 0.0) * 100
        emotion_agreement = comparison_result.get("emotion_agreement", 0.0) * 100
        best_index = comparison_result.get("best_model_index")
        
        # Format header
        formatted = [
            f"{Fore.CYAN}╔{'═' * 70}╗",
            f"║ {Fore.WHITE}Model Comparison Results{Fore.CYAN}{' ' * 48}║",
            f"╠{'═' * 70}╣",
            f"║ {Fore.WHITE}Text: \"{text[:50]}{'...' if len(text) > 50 else ''}\"{Fore.CYAN}{' ' * (60 - min(len(text), 50))}║",
            f"║ {Fore.WHITE}Models: {len(results)}{Fore.CYAN}{' ' * 60}║",
            f"╠{'═' * 70}╣"
        ]
        
        # Table header
        model_col = "Model"
        sentiment_col = "Sentiment"
        score_col = "Score"
        emotion_col = "Emotion"
        time_col = "Time (s)"
        
        header_row = f"║ {Fore.WHITE}{model_col.ljust(20)} | {sentiment_col.ljust(10)} | {score_col.ljust(6)} | {emotion_col.ljust(10)} | {time_col.ljust(8)}{Fore.CYAN} ║"
        formatted.append(header_row)
        formatted.append(f"╠{'═' * 70}╣")
        
        # Model results
        for i, result in enumerate(results):
            model_name = result.get("model", "Unknown")
            sentiment = result.get("sentiment", {}).get("label", "N/A")
            sentiment_score = result.get("sentiment", {}).get("score", 0.0) * 100
            emotion = result.get("emotion", {}).get("label", "N/A")
            exec_time = result.get("execution_time", 0.0)
            
            # Highlight the best model
            prefix = f"{Fore.GREEN}* " if i == best_index else "  "
            
            # Color sentiment
            if sentiment == "positive":
                sentiment_color = Fore.GREEN
            elif sentiment == "negative":
                sentiment_color = Fore.RED
            else:
                sentiment_color = Fore.YELLOW
            
            # Color emotion
            if emotion in ["joy", "love"]:
                emotion_color = Fore.GREEN
            elif emotion in ["anger", "fear", "sadness"]:
                emotion_color = Fore.RED
            elif emotion == "surprise":
                emotion_color = Fore.YELLOW
            else:
                emotion_color = Fore.WHITE
            
            row = (
                f"║ {prefix}{Fore.WHITE}{model_name[:18].ljust(18)} | "
                f"{sentiment_color}{sentiment.ljust(10)}{Fore.CYAN} | "
                f"{Fore.WHITE}{sentiment_score:.1f}%".ljust(8) + f" | "
                f"{emotion_color}{str(emotion).ljust(10)}{Fore.CYAN} | "
                f"{Fore.WHITE}{exec_time:.3f}s".ljust(10) + f"{Fore.CYAN} ║"
            )
            formatted.append(row)
        
        # Agreement stats
        formatted.append(f"╠{'═' * 70}╣")
        formatted.append(f"║ {Fore.WHITE}Agreement Stats:{Fore.CYAN}{' ' * 53}║")
        formatted.append(f"║ {Fore.WHITE}Sentiment: {Fore.GREEN if sentiment_agreement >= 80 else Fore.YELLOW if sentiment_agreement >= 50 else Fore.RED}{sentiment_agreement:.1f}%{Fore.CYAN}{' ' * 52}║")
        formatted.append(f"║ {Fore.WHITE}Emotion: {Fore.GREEN if emotion_agreement >= 80 else Fore.YELLOW if emotion_agreement >= 50 else Fore.RED}{emotion_agreement:.1f}%{Fore.CYAN}{' ' * 54}║")
        
        # Differences section if there are any
        sentiment_diffs = comparison_result.get("sentiment_differences", [])
        emotion_diffs = comparison_result.get("emotion_differences", [])
        
        if sentiment_diffs or emotion_diffs:
            formatted.append(f"╠{'═' * 70}╣")
            formatted.append(f"║ {Fore.WHITE}Differences:{Fore.CYAN}{' ' * 57}║")
            
            if sentiment_diffs:
                formatted.append(f"║ {Fore.WHITE}Sentiment:{Fore.CYAN}{' ' * 58}║")
                for model, label, score in sentiment_diffs:
                    color = Fore.GREEN if label == "positive" else Fore.RED if label == "negative" else Fore.YELLOW
                    formatted.append(f"║   {Fore.WHITE}{model[:15]}: {color}{label} ({score*100:.1f}%){Fore.CYAN}{' ' * (70 - 25 - len(model[:15]) - len(label) - 7)}║")
            
            if emotion_diffs:
                formatted.append(f"║ {Fore.WHITE}Emotion:{Fore.CYAN}{' ' * 60}║")
                for model, label, score in emotion_diffs:
                    if label in ["joy", "love"]:
                        color = Fore.GREEN
                    elif label in ["anger", "fear", "sadness"]:
                        color = Fore.RED
                    else:
                        color = Fore.YELLOW
                    formatted.append(f"║   {Fore.WHITE}{model[:15]}: {color}{label} ({score*100:.1f}%){Fore.CYAN}{' ' * (70 - 25 - len(model[:15]) - len(str(label)) - 7)}║")
        
        # Add probabilities if requested
        if show_probabilities:
            for i, result in enumerate(results):
                model_name = result.get("model", "Unknown")
                sentiment_probs = result.get("sentiment", {}).get("raw_probabilities", {})
                emotion_probs = result.get("emotion", {}).get("raw_probabilities", {})
                
                if sentiment_probs or emotion_probs:
                    formatted.append(f"╠{'═' * 70}╣")
                    formatted.append(f"║ {Fore.WHITE}Probabilities for {model_name}:{Fore.CYAN}{' ' * (70 - 20 - len(model_name))}║")
                    
                    if sentiment_probs:
                        formatted.append(f"║ {Fore.WHITE}Sentiment:{Fore.CYAN}{' ' * 58}║")
                        for label, prob in sorted(sentiment_probs.items(), key=lambda x: x[1], reverse=True):
                            bar_len = int(prob * 20)
                            bar = '█' * bar_len + '░' * (20 - bar_len)
                            color = Fore.GREEN if label == "positive" else Fore.RED if label == "negative" else Fore.YELLOW
                            formatted.append(f"║   {color}{label.ljust(10)} | {bar} {prob*100:.1f}%{Fore.CYAN}{' ' * (70 - 45)}║")
                    
                    if emotion_probs:
                        formatted.append(f"║ {Fore.WHITE}Emotion:{Fore.CYAN}{' ' * 60}║")
                        for label, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
                            bar_len = int(prob * 20)
                            bar = '█' * bar_len + '░' * (20 - bar_len)
                            if label in ["joy", "love"]:
                                color = Fore.GREEN
                            elif label in ["anger", "fear", "sadness"]:
                                color = Fore.RED
                            else:
                                color = Fore.YELLOW
                            formatted.append(f"║   {color}{label.ljust(10)} | {bar} {prob*100:.1f}%{Fore.CYAN}{' ' * (70 - 45)}║")
        
        # Footer
        formatted.append(f"╚{'═' * 70}╝{Style.RESET_ALL}")
        
        return "\n".join(formatted)
    
    def export_comparison(self, comparison_result: Dict[str, Any], filepath: str, format: str = "json") -> None:
        """
        Export comparison results to a file.
        
        Args:
            comparison_result: Result from compare() method
            filepath: Path to save the file
            format: Export format ('json', 'csv', or 'txt')
            
        Raises:
            ValueError: If format is invalid
        """
        if not comparison_result:
            raise ValueError("No comparison results to export")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Export based on format
        if format.lower() == "json":
            # Create a copy that's safe to serialize
            serializable = self._prepare_for_serialization(comparison_result)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)
        
        elif format.lower() == "csv":
            results = comparison_result.get("results", [])
            if not results:
                raise ValueError("No results to export")
            
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ["Text", "Model", "Sentiment", "Sentiment Score", 
                         "Emotion", "Emotion Score", "Execution Time"]
                writer.writerow(header)
                
                # Write data rows
                for result in results:
                    row = [
                        comparison_result.get("text", ""),
                        result.get("model", "Unknown"),
                        result.get("sentiment", {}).get("label", ""),
                        result.get("sentiment", {}).get("score", 0.0),
                        result.get("emotion", {}).get("label", ""),
                        result.get("emotion", {}).get("score", 0.0),
                        result.get("execution_time", 0.0)
                    ]
                    writer.writerow(row)
        
        elif format.lower() == "txt":
            # Export as plain text (without color codes)
            formatted = self.format_comparison(comparison_result, show_probabilities=True)
            # Remove color codes
            formatted = self._strip_color_codes(formatted)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted)
        
        else:
            raise ValueError(f"Invalid export format: {format}. Use 'json', 'csv', or 'txt'.")
    
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """
        Prepare an object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items() if k != 'model_instance'}
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._prepare_for_serialization(obj.__dict__)
        else:
            return obj
    
    def _strip_color_codes(self, text: str) -> str:
        """
        Remove ANSI color codes from a string.
        
        Args:
            text: Text with color codes
            
        Returns:
            Text without color codes
        """
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text) 
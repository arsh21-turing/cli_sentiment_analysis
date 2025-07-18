"""
Module for confidence threshold optimization.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from src.evaluation.metrics import EvaluationMetrics


class ThresholdOptimizer:
    """Class for finding optimal confidence thresholds."""
    
    def __init__(
        self,
        metrics_calculator: EvaluationMetrics,
        metric_name: str = "f1",
        step_size: float = 0.05
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            metrics_calculator: Instance of metrics calculator
            metric_name: Metric to optimize ("f1", "precision", "recall")
            step_size: Step size for threshold sweep
        """
        self.metrics_calculator = metrics_calculator
        self.metric_name = metric_name
        self.step_size = step_size
        self.threshold_results = {}
        self.optimized_thresholds = {}
        
    def sweep_thresholds(self, start: float = 0.1, end: float = 0.9) -> Dict[float, Dict[str, float]]:
        """
        Perform threshold sweep and return metrics at each threshold.
        
        Args:
            start: Starting threshold value
            end: Ending threshold value
            
        Returns:
            Dictionary mapping thresholds to metric dictionaries
        """
        thresholds = np.arange(start, end + self.step_size, self.step_size)
        self.threshold_results = self.metrics_calculator.calculate_metrics_at_thresholds(thresholds)
        return self.threshold_results
    
    def find_optimal_threshold(self, label_type: str = "sentiment") -> float:
        """
        Find optimal threshold for specified label type.
        
        Args:
            label_type: Label type to optimize for
            
        Returns:
            Optimal threshold value
        """
        # Perform sweep if not already done
        if not self.threshold_results:
            self.sweep_thresholds()
            
        # Find threshold that maximizes the target metric
        best_threshold = 0.5  # Default
        best_score = 0.0
        
        for threshold, metrics in self.threshold_results.items():
            score = metrics.get(self.metric_name, 0.0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        # Store optimal threshold
        self.optimized_thresholds[label_type] = best_threshold
        
        return best_threshold
    
    def get_threshold_metrics(self) -> Dict[float, Dict[str, float]]:
        """
        Get metrics at each evaluated threshold.
        
        Returns:
            Dictionary of threshold metrics
        """
        # Ensure we have results
        if not self.threshold_results:
            self.sweep_thresholds()
            
        return self.threshold_results
    
    def get_optimal_thresholds(self) -> Dict[str, float]:
        """
        Get optimal thresholds for all label types.
        
        Returns:
            Dictionary mapping label types to optimal thresholds
        """
        # Ensure we have optimal thresholds
        if not self.optimized_thresholds:
            self.find_optimal_threshold()
            
        return self.optimized_thresholds
    
    def get_threshold_performance_data(self) -> pd.DataFrame:
        """
        Get performance data across all thresholds for plotting.
        
        Returns:
            DataFrame with performance data at each threshold
        """
        # Ensure we have results
        if not self.threshold_results:
            self.sweep_thresholds()
            
        # Convert to DataFrame for easier plotting
        data = []
        for threshold, metrics in self.threshold_results.items():
            row = {"threshold": threshold}
            row.update(metrics)
            data.append(row)
            
        return pd.DataFrame(data)
    
    def save_threshold_data(self, file_path: str) -> None:
        """
        Save threshold optimization results to a file.
        
        Args:
            file_path: Path to save threshold data
        """
        # Ensure we have results
        if not self.threshold_results:
            self.sweep_thresholds()
            
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for threshold, metrics in self.threshold_results.items():
            serializable_results[float(threshold)] = {
                k: float(v) for k, v in metrics.items()
            }
            
        # Prepare data to save
        data = {
            "threshold_results": serializable_results,
            "optimized_thresholds": self.optimized_thresholds,
            "metric_name": self.metric_name,
            "label_type": self.metrics_calculator.label_type
        }
        
        # Save data to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_threshold_data(self, file_path: str) -> Dict:
        """
        Load threshold optimization results from a file.
        
        Args:
            file_path: Path to load threshold data from
            
        Returns:
            Loaded threshold data
        """
        # Load data from file
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert keys back to floats for threshold results
        self.threshold_results = {
            float(k): v for k, v in data.get("threshold_results", {}).items()
        }
        
        self.optimized_thresholds = data.get("optimized_thresholds", {})
        self.metric_name = data.get("metric_name", "f1")
        
        return data
    
    def apply_optimal_thresholds(self, predictions: List[Dict], label_types: Optional[List[str]] = None) -> Dict:
        """
        Apply optimal thresholds to predictions.
        
        Args:
            predictions: List of prediction dictionaries
            label_types: List of label types to apply thresholds for
            
        Returns:
            Dictionary mapping label types to thresholded predictions
        """
        # Ensure we have optimal thresholds
        if not self.optimized_thresholds:
            self.find_optimal_threshold()
            
        if label_types is None:
            label_types = list(self.optimized_thresholds.keys())
            
        # Apply thresholds to predictions
        thresholded_predictions = {}
        
        for label_type in label_types:
            if label_type not in self.optimized_thresholds:
                # Find optimal threshold for this label type
                self.find_optimal_threshold(label_type)
                
            threshold = self.optimized_thresholds[label_type]
            
            # Apply threshold to predictions for this label type
            thresholded_preds = []
            for pred in predictions:
                confidence = pred.get(f"{label_type}_confidence", 0.5)
                prediction = pred.get(label_type)
                
                if confidence >= threshold:
                    thresholded_preds.append(prediction)
                else:
                    # Below threshold: use alternative prediction logic
                    # This is a simplified approach; real implementation would depend on model
                    if label_type == "sentiment":
                        # For sentiment, invert prediction
                        thresholded_preds.append(
                            "negative" if prediction == "positive" else "positive"
                        )
                    else:
                        # For other types, just mark as low confidence
                        thresholded_preds.append("low_confidence")
                        
            thresholded_predictions[label_type] = thresholded_preds
            
        return thresholded_predictions 
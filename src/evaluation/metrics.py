"""
Module for metrics calculation for model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc, classification_report
)


class EvaluationMetrics:
    """Class for calculating evaluation metrics for sentiment/emotion models."""
    
    def __init__(
        self,
        predictions: Union[List, Dict],
        ground_truth: Union[List, Dict],
        label_type: str = "sentiment",
        prediction_key: str = "confidence",
        class_labels: Optional[List] = None
    ):
        """
        Initialize metrics calculator.
        
        Args:
            predictions: Model predictions with confidence scores
            ground_truth: Ground truth labels
            label_type: Type of prediction to evaluate ("sentiment" or "emotion")
            prediction_key: Key in predictions dict containing confidence values
            class_labels: List of class labels for categorization
        """
        self.label_type = label_type
        self.prediction_key = prediction_key
        self.class_labels = class_labels
        
        # Process predictions and ground truth to standard format
        self._process_inputs(predictions, ground_truth)
        
    def _process_inputs(self, predictions, ground_truth):
        """Process model predictions and ground truth to standard format."""
        # Handle list vs. dictionary inputs
        if isinstance(predictions, list) and isinstance(predictions[0], dict):
            # Extract values based on label type
            if f"{self.label_type}_confidence" in predictions[0]:
                self.confidence_scores = [
                    p.get(f"{self.label_type}_confidence", 0.0)
                    for p in predictions
                ]
                self.predicted_labels = [
                    p.get(self.label_type, "")
                    for p in predictions
                ]
            elif self.label_type in predictions[0]:
                # Use prediction dictionaries with label_type as key
                self.predicted_labels = [p.get(self.label_type, "") for p in predictions]
                self.confidence_scores = [p.get(self.prediction_key, 0.5) for p in predictions]
            else:
                # Assume predictions is already the list of labels
                self.predicted_labels = predictions
                self.confidence_scores = [0.5] * len(predictions)  # Default confidence
        else:
            # Assume predictions is already the list of labels
            self.predicted_labels = predictions
            self.confidence_scores = [0.5] * len(predictions)  # Default confidence
            
        # Process ground truth
        if isinstance(ground_truth, list) and isinstance(ground_truth[0], dict):
            self.true_labels = [g.get(self.label_type, "") for g in ground_truth]
        else:
            self.true_labels = ground_truth
            
        # Ensure all lists have the same length
        if not (len(self.predicted_labels) == len(self.confidence_scores) == len(self.true_labels)):
            raise ValueError(
                "Mismatch in lengths of predictions, confidence scores, and ground truth"
            )
            
        # If class_labels not provided, infer from unique labels
        if self.class_labels is None:
            # Filter out None, empty strings, and convert all to strings for sorting
            true_labels_clean = [str(label) for label in self.true_labels if label not in [None, '', 'nan']]
            predicted_labels_clean = [str(label) for label in self.predicted_labels if label not in [None, '', 'nan']]
            self.class_labels = sorted(list(set(true_labels_clean) | set(predicted_labels_clean)))
    
    def calculate_precision_recall(self, threshold: float = 0.5, average: str = "macro") -> Tuple[float, float]:
        """
        Calculate precision and recall scores.
        
        Args:
            threshold: Confidence threshold for prediction
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Tuple of (precision, recall)
        """
        # Apply threshold to get final predictions
        thresholded_preds = self.get_predictions_at_threshold(threshold)
        
        # Calculate precision and recall
        precision = precision_score(
            self.true_labels, 
            thresholded_preds, 
            average=average,
            zero_division=0,
            labels=self.class_labels
        )
        
        recall = recall_score(
            self.true_labels, 
            thresholded_preds, 
            average=average,
            zero_division=0,
            labels=self.class_labels
        )
        
        return precision, recall
    
    def calculate_f1(self, threshold: float = 0.5, average: str = "macro") -> float:
        """
        Calculate F1 score.
        
        Args:
            threshold: Confidence threshold for prediction
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            F1 score
        """
        # Apply threshold to get final predictions
        thresholded_preds = self.get_predictions_at_threshold(threshold)
        
        # Calculate F1 score
        score = f1_score(
            self.true_labels, 
            thresholded_preds, 
            average=average,
            zero_division=0,
            labels=self.class_labels
        )
        
        return score
    
    def generate_confusion_matrix(self, threshold: float = 0.5) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            threshold: Confidence threshold for prediction
            
        Returns:
            Confusion matrix as numpy array
        """
        # Apply threshold to get final predictions
        thresholded_preds = self.get_predictions_at_threshold(threshold)
        
        # Generate confusion matrix
        cm = confusion_matrix(
            self.true_labels, 
            thresholded_preds,
            labels=self.class_labels
        )
        
        return cm
    
    def calculate_metrics_at_thresholds(self, thresholds=None) -> Dict[float, Dict[str, float]]:
        """
        Calculate metrics across threshold range.
        
        Args:
            thresholds: List of threshold values to evaluate
            
        Returns:
            Dictionary mapping thresholds to metric dictionaries
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
            
        results = {}
        for threshold in thresholds:
            precision, recall = self.calculate_precision_recall(threshold)
            f1 = self.calculate_f1(threshold)
            
            results[threshold] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
        return results
    
    def get_optimal_threshold(self, metric: str = "f1", average: str = "macro") -> float:
        """
        Find threshold that maximizes the specified metric.
        
        Args:
            metric: Metric to optimize ("f1", "precision", "recall")
            average: Averaging method for metrics calculation
            
        Returns:
            Optimal threshold value
        """
        # Use a fine-grained threshold search
        thresholds = np.arange(0.1, 1.0, 0.01)
        
        best_threshold = 0.5  # Default
        best_score = 0.0
        
        for threshold in thresholds:
            score = 0.0
            
            if metric == "f1":
                score = self.calculate_f1(threshold, average)
            elif metric == "precision":
                score, _ = self.calculate_precision_recall(threshold, average)
            elif metric == "recall":
                _, score = self.calculate_precision_recall(threshold, average)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        return best_threshold
    
    def get_classification_report(self, threshold: float = 0.5) -> Dict:
        """
        Get detailed classification metrics.
        
        Args:
            threshold: Confidence threshold for prediction
            
        Returns:
            Dictionary containing classification report
        """
        # Apply threshold to get final predictions
        thresholded_preds = self.get_predictions_at_threshold(threshold)
        
        # Generate classification report
        report = classification_report(
            self.true_labels, 
            thresholded_preds,
            output_dict=True,
            zero_division=0,
            labels=self.class_labels
        )
        
        return report
    
    def calculate_auc(self) -> float:
        """
        Calculate Area Under the ROC Curve.
        
        Returns:
            AUC score (average across classes for multiclass)
        """
        # For binary classification
        if len(self.class_labels) == 2:
            # Get binary class predictions (positive class confidence scores)
            binary_scores = np.array(self.confidence_scores)
            binary_true = np.array([1 if l == self.class_labels[1] else 0 for l in self.true_labels])
            
            fpr, tpr, _ = roc_curve(binary_true, binary_scores)
            return auc(fpr, tpr)
        
        # For multi-class: one-vs-rest approach
        n_classes = len(self.class_labels)
        aucs = []
        
        for i, class_label in enumerate(self.class_labels):
            # Get binary class predictions for this class
            binary_true = np.array([1 if l == class_label else 0 for l in self.true_labels])
            
            # Get confidence scores for this class (simplified assumption)
            binary_scores = np.array([
                score if pred == class_label else 1 - score
                for pred, score in zip(self.predicted_labels, self.confidence_scores)
            ])
            
            try:
                fpr, tpr, _ = roc_curve(binary_true, binary_scores)
                aucs.append(auc(fpr, tpr))
            except ValueError:
                # Handle error when only one class is present
                aucs.append(0.5)  # Default AUC
        
        # Return average AUC
        return np.mean(aucs)
    
    def get_metrics_summary(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Get summary of all metrics.
        
        Args:
            threshold: Threshold to use (optimal threshold if None)
            
        Returns:
            Dictionary of metrics
        """
        if threshold is None:
            threshold = self.get_optimal_threshold()
            
        precision, recall = self.calculate_precision_recall(threshold)
        f1 = self.calculate_f1(threshold)
        auc_score = self.calculate_auc()
        
        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score,
            "label_type": self.label_type,
            "num_samples": len(self.true_labels)
        }
    
    def get_predictions_at_threshold(self, threshold: float = 0.5) -> List:
        """
        Get thresholded predictions.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            List of predicted labels based on threshold
        """
        # Apply threshold
        thresholded_preds = []
        
        for pred, conf in zip(self.predicted_labels, self.confidence_scores):
            # Handle NaN values in confidence scores
            if pd.isna(conf) or conf is None:
                thresholded_preds.append("neutral")
            elif conf >= threshold:
                thresholded_preds.append(pred)
            else:
                # If below threshold, choose alternative class (binary) or most common (multi)
                if len(self.class_labels) == 2:
                    # For binary, just invert the prediction
                    thresholded_preds.append(
                        self.class_labels[0] if pred == self.class_labels[1] else self.class_labels[1]
                    )
                else:
                    # For multiclass, default to most common class
                    most_common = max(set(self.true_labels), key=self.true_labels.count)
                    thresholded_preds.append(most_common)
        
        return thresholded_preds
    
    def get_thresholded_results(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Get results DataFrame at the specified threshold.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            DataFrame with true labels, predictions and confidence scores
        """
        thresholded_preds = self.get_predictions_at_threshold(threshold)
        
        return pd.DataFrame({
            "true_label": self.true_labels,
            "predicted_label": thresholded_preds,
            "confidence": self.confidence_scores,
            "correct": [t == p for t, p in zip(self.true_labels, thresholded_preds)]
        })
    
    def calculate_macro_f1(self, threshold: float = 0.5) -> float:
        """
        Calculate macro-averaged F1 score.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Macro-averaged F1 score
        """
        return self.calculate_f1(threshold, average="macro")
    
    def calculate_weighted_f1(self, threshold: float = 0.5) -> float:
        """
        Calculate weighted-averaged F1 score.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Weighted-averaged F1 score
        """
        return self.calculate_f1(threshold, average="weighted")
    
    def get_precision_recall_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for precision-recall curve plotting.
        
        Returns:
            Tuple of (precision, recall, thresholds) arrays
        """
        # For binary classification
        if len(self.class_labels) == 2:
            # Get binary class predictions for positive class
            binary_true = np.array([1 if l == self.class_labels[1] else 0 for l in self.true_labels])
            binary_scores = np.array(self.confidence_scores)
            
            precision, recall, thresholds = precision_recall_curve(binary_true, binary_scores)
            return precision, recall, thresholds
            
        # For multi-class, use macro-averaged PR curve
        all_precisions = []
        all_recalls = []
        
        # Calculate PR curves for each class
        for class_label in self.class_labels:
            binary_true = np.array([1 if l == class_label else 0 for l in self.true_labels])
            binary_scores = np.array([
                conf if pred == class_label else 0
                for pred, conf in zip(self.predicted_labels, self.confidence_scores)
            ])
            
            try:
                precision, recall, _ = precision_recall_curve(binary_true, binary_scores)
                all_precisions.append(precision)
                all_recalls.append(recall)
            except ValueError:
                # Handle error when only one class is present
                continue
        
        if not all_precisions:
            # Fallback: create simple PR curve
            thresholds = np.linspace(0, 1, 100)
            precision = np.array([0.5] * len(thresholds))
            recall = np.array([0.5] * len(thresholds))
            return precision, recall, thresholds
        
        # Ensure all arrays have the same length by interpolating to the shortest length
        min_length = min(len(p) for p in all_precisions)
        
        # Interpolate all arrays to the same length
        interpolated_precisions = []
        interpolated_recalls = []
        
        for precision, recall in zip(all_precisions, all_recalls):
            if len(precision) > min_length:
                # Interpolate to shorter length
                indices = np.linspace(0, len(precision) - 1, min_length)
                precision_interp = np.interp(indices, np.arange(len(precision)), precision)
                recall_interp = np.interp(indices, np.arange(len(recall)), recall)
            else:
                precision_interp = precision[:min_length]
                recall_interp = recall[:min_length]
                
            interpolated_precisions.append(precision_interp)
            interpolated_recalls.append(recall_interp)
        
        # Calculate mean precision and recall
        mean_precision = np.mean(interpolated_precisions, axis=0)
        mean_recall = np.mean(interpolated_recalls, axis=0)
        
        # Create thresholds
        thresholds = np.linspace(0, 1, len(mean_precision))
        
        return mean_precision, mean_recall, thresholds
    
    def get_classification_metrics_by_class(self, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Get metrics broken down by class.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Dictionary mapping classes to their metrics
        """
        report = self.get_classification_report(threshold)
        
        class_metrics = {}
        for class_label in self.class_labels:
            if class_label in report:
                class_metrics[class_label] = report[class_label]
                
        return class_metrics 
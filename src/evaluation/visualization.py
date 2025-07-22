"""
Module for evaluation result visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold import ThresholdOptimizer


class EvaluationVisualizer:
    """Class for creating visualizations for evaluation results."""
    
    def __init__(
        self,
        metrics_calculator: EvaluationMetrics,
        threshold_optimizer: Optional[ThresholdOptimizer] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            metrics_calculator: Instance of metrics calculator
            threshold_optimizer: Optional threshold optimizer instance
        """
        self.metrics_calculator = metrics_calculator
        self.threshold_optimizer = threshold_optimizer
        
        # Initialize figures dictionary
        self.figures = {}
        
    def plot_precision_recall_curve(self, label_type: str = "sentiment", save_path: Optional[str] = None):
        """
        Create precision-recall curve.
        
        Args:
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Get PR curve data
        precision, recall, thresholds = self.metrics_calculator.get_precision_recall_curve_data()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot PR curve
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve ({label_type.capitalize()})')
        ax.grid(True)
        
        # Add optimal threshold marker if available
        if self.threshold_optimizer is not None:
            optimal_threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
            
            # Find closest threshold in our data
            if len(thresholds) > 0:
                idx = np.argmin(np.abs(thresholds - optimal_threshold))
                opt_precision = precision[idx]
                opt_recall = recall[idx]
                
                ax.plot(opt_recall, opt_precision, 'ro', markersize=10, label=f'Optimal threshold: {optimal_threshold:.2f}')
                ax.legend()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['pr_curve'] = fig
        
        return fig
    
    def plot_roc_curve(self, label_type: str = "sentiment", save_path: Optional[str] = None):
        """
        Create ROC curve.
        
        Args:
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Get ROC curve data
        # For simplicity in this example - would normally call ROC curve calculation method
        
        # Create binary classification data for ROC curve
        true_binary = np.array([
            1 if l == self.metrics_calculator.class_labels[-1] else 0
            for l in self.metrics_calculator.true_labels
        ])
        score_binary = np.array(self.metrics_calculator.confidence_scores)
        
        # Calculate ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(true_binary, score_binary)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')  # Diagonal
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve ({label_type.capitalize()})')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['roc_curve'] = fig
        
        return fig
    
    def plot_confusion_matrix(
        self,
        threshold: Optional[float] = None,
        label_type: str = "sentiment",
        save_path: Optional[str] = None
    ):
        """
        Create confusion matrix visualization.
        
        Args:
            threshold: Confidence threshold (uses optimal if None)
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Get threshold to use
        if threshold is None and self.threshold_optimizer is not None:
            threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        elif threshold is None:
            threshold = 0.5
            
        # Get confusion matrix
        cm = self.metrics_calculator.generate_confusion_matrix(threshold)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.metrics_calculator.class_labels,
            yticklabels=self.metrics_calculator.class_labels,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(f'Confusion Matrix ({label_type.capitalize()}) @ threshold={threshold:.2f}')
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['confusion_matrix'] = fig
        
        return fig
    
    def plot_f1_vs_threshold(self, label_type: str = "sentiment", save_path: Optional[str] = None):
        """
        Create plot of F1 score vs threshold.
        
        Args:
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.threshold_optimizer is None:
            print("No threshold optimizer available. Cannot generate F1 vs threshold plot.")
            return None
            
        # Get threshold performance data
        threshold_data = self.threshold_optimizer.get_threshold_performance_data()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot F1 vs threshold
        ax.plot(threshold_data['threshold'], threshold_data['f1'], 'b-', linewidth=2)
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 Score vs. Threshold ({label_type.capitalize()})')
        ax.grid(True)
        
        # Add optimal threshold marker
        optimal_threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        optimal_f1 = threshold_data.loc[threshold_data['threshold'] == optimal_threshold, 'f1'].values[0]
        
        ax.plot(optimal_threshold, optimal_f1, 'ro', markersize=10, 
                label=f'Optimal threshold: {optimal_threshold:.2f} (F1: {optimal_f1:.2f})')
        ax.legend()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['f1_vs_threshold'] = fig
        
        return fig
    
    def plot_threshold_comparison(
        self,
        metrics: List[str] = ["precision", "recall", "f1"],
        label_type: str = "sentiment",
        save_path: Optional[str] = None
    ):
        """
        Create comparative plot of metrics across thresholds.
        
        Args:
            metrics: List of metrics to include
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.threshold_optimizer is None:
            print("No threshold optimizer available. Cannot generate threshold comparison plot.")
            return None
            
        # Get threshold performance data
        threshold_data = self.threshold_optimizer.get_threshold_performance_data()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot each metric
        for metric in metrics:
            if metric in threshold_data.columns:
                ax.plot(threshold_data['threshold'], threshold_data[metric], linewidth=2, label=metric.capitalize())
        
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Metrics vs. Threshold ({label_type.capitalize()})')
        ax.grid(True)
        ax.legend()
        
        # Add optimal threshold marker
        optimal_threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        ax.axvline(x=optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal threshold: {optimal_threshold:.2f}')
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['threshold_comparison'] = fig
        
        return fig
    
    def plot_class_performance(
        self,
        threshold: Optional[float] = None,
        label_type: str = "sentiment",
        metric: str = "f1",
        save_path: Optional[str] = None
    ):
        """
        Create bar chart showing performance by class.
        
        Args:
            threshold: Confidence threshold (uses optimal if None)
            label_type: Label type to visualize
            metric: Metric to show ("f1", "precision", "recall")
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Get threshold to use
        if threshold is None and self.threshold_optimizer is not None:
            threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        elif threshold is None:
            threshold = 0.5
            
        # Get class performance metrics
        class_metrics = self.metrics_calculator.get_classification_metrics_by_class(threshold)
        
        # Extract specified metric for each class
        classes = []
        values = []
        
        for class_label, metrics in class_metrics.items():
            classes.append(class_label)
            values.append(metrics.get(metric, 0.0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = ax.bar(classes, values, color='skyblue')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_xlabel('Class')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Class ({label_type.capitalize()}) @ threshold={threshold:.2f}')
        ax.set_ylim(0, 1.0)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['class_performance'] = fig
        
        return fig
    
    def create_interactive_threshold_tuner(self, label_type: str = "sentiment"):
        """
        Create interactive plot for threshold tuning.
        
        Args:
            label_type: Label type to visualize
            
        Returns:
            Plotly figure object or None if Plotly is not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available. Cannot create interactive plot.")
            return None
            
        if self.threshold_optimizer is None:
            print("No threshold optimizer available. Cannot create interactive threshold tuner.")
            return None
            
        # Get threshold performance data
        threshold_data = self.threshold_optimizer.get_threshold_performance_data()
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=["Metrics vs. Threshold", "Prediction Counts"]
        )
        
        # Add metrics traces
        fig.add_trace(
            go.Scatter(x=threshold_data['threshold'], y=threshold_data['precision'], 
                      name='Precision', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=threshold_data['threshold'], y=threshold_data['recall'], 
                      name='Recall', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=threshold_data['threshold'], y=threshold_data['f1'], 
                      name='F1 Score', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Add optimal threshold marker
        optimal_threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        
        # Create prediction counts for different thresholds
        thresholds = sorted(list(set(threshold_data['threshold'])))
        
        positive_counts = []
        negative_counts = []
        
        for threshold in thresholds:
            # Get predictions at this threshold
            predictions = self.metrics_calculator.get_predictions_at_threshold(threshold)
            
            # Count predictions
            if len(self.metrics_calculator.class_labels) == 2:
                # For binary classification
                pos_class = self.metrics_calculator.class_labels[1]  # Assume second class is positive
                pos_count = sum(1 for p in predictions if p == pos_class)
                neg_count = len(predictions) - pos_count
                
                positive_counts.append(pos_count)
                negative_counts.append(neg_count)
            else:
                # For multi-class, just count positive predictions as those with top confidence
                pos_count = sum(1 for p, c in zip(
                    self.metrics_calculator.predicted_labels, 
                    self.metrics_calculator.confidence_scores
                ) if c >= threshold)
                neg_count = len(predictions) - pos_count
                
                positive_counts.append(pos_count)
                negative_counts.append(neg_count)
                
        # Add prediction count traces
        fig.add_trace(
            go.Bar(x=thresholds, y=positive_counts, name='Positive Predictions', marker_color='green'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=thresholds, y=negative_counts, name='Negative Predictions', marker_color='red'),
            row=2, col=1
        )
        
        # Add vertical line for optimal threshold
        fig.add_vline(
            x=optimal_threshold, 
            line_width=2, 
            line_dash="dash", 
            line_color="purple",
            annotation_text=f"Optimal Threshold: {optimal_threshold:.2f}", 
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Threshold Tuner ({label_type.capitalize()})',
            xaxis_title='Confidence Threshold',
            legend=dict(x=0, y=1.1, orientation='h'),
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text='Score', range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text='Count', row=2, col=1)
        
        # Keep interactive figure
        self.figures['interactive_threshold'] = fig
        
        return fig
    
    def plot_metrics_summary(
        self,
        threshold: Optional[float] = None,
        label_type: str = "sentiment",
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive metrics summary plot.
        
        Args:
            threshold: Confidence threshold (uses optimal if None)
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Get threshold to use
        if threshold is None and self.threshold_optimizer is not None:
            threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        elif threshold is None:
            threshold = 0.5
            
        # Get metrics summary
        metrics_summary = self.metrics_calculator.get_metrics_summary(threshold)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Metrics Summary ({label_type.capitalize()}) @ threshold={threshold:.2f}', fontsize=16)
        
        # Plot confusion matrix
        cm = self.metrics_calculator.generate_confusion_matrix(threshold)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.metrics_calculator.class_labels,
            yticklabels=self.metrics_calculator.class_labels,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Labels')
        axes[0, 0].set_ylabel('True Labels')
        
        # Plot metrics bar chart
        metrics_to_plot = ['precision', 'recall', 'f1']
        values = [metrics_summary.get(m, 0) for m in metrics_to_plot]
        
        bars = axes[0, 1].bar(metrics_to_plot, values, color=['blue', 'green', 'red'])
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].annotate(f'{height:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom')
        
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].set_title('Performance Metrics')
        
        # Plot class-wise metrics
        class_metrics = self.metrics_calculator.get_classification_metrics_by_class(threshold)
        
        # Prepare data for class metrics
        classes = list(class_metrics.keys())
        class_precision = [class_metrics[c].get('precision', 0) for c in classes]
        class_recall = [class_metrics[c].get('recall', 0) for c in classes]
        class_f1 = [class_metrics[c].get('f1-score', 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        # Plot class metrics
        axes[1, 0].bar(x - width, class_precision, width, label='Precision', color='blue')
        axes[1, 0].bar(x, class_recall, width, label='Recall', color='green')
        axes[1, 0].bar(x + width, class_f1, width, label='F1', color='red')
        
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes)
        axes[1, 0].set_ylim(0, 1.0)
        axes[1, 0].set_title('Performance by Class')
        axes[1, 0].legend()
        
        # Plot prediction distribution
        predictions = self.metrics_calculator.get_thresholded_results(threshold)
        correct_counts = predictions['correct'].value_counts()
        
        axes[1, 1].pie(
            [correct_counts.get(True, 0), correct_counts.get(False, 0)],
            labels=['Correct', 'Incorrect'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90
        )
        axes[1, 1].set_title('Prediction Accuracy')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['metrics_summary'] = fig
        
        return fig
    
    def plot_f1_heatmap(self, label_type: str = "sentiment", save_path: Optional[str] = None):
        """
        Create heatmap of F1 scores by class.
        
        Args:
            label_type: Label type to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Get threshold to use
        if self.threshold_optimizer is not None:
            threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        else:
            threshold = 0.5
            
        # Get classification report for all classes
        report = self.metrics_calculator.get_classification_report(threshold)
        
        # Extract class data
        classes = []
        f1_scores = []
        support = []
        
        for class_label, metrics in report.items():
            if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
                
            classes.append(class_label)
            f1_scores.append(metrics.get('f1-score', 0))
            support.append(metrics.get('support', 0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a matrix for the heatmap - each class as a row
        heatmap_data = np.array(f1_scores).reshape(-1, 1)
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='YlGnBu')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('F1 Score')
        
        # Configure axes
        ax.set_xticks([])  # Hide x-ticks
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes)
        
        # Annotate cells with F1 score and support
        for i in range(len(classes)):
            text = f"F1: {f1_scores[i]:.2f}\nSupport: {support[i]}"
            ax.text(0, i, text, va='center', ha='center', color='black')
            
        ax.set_title(f'F1 Score Heatmap by Class ({label_type.capitalize()})')
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['f1_heatmap'] = fig
        
        return fig
    
    def create_dashboard(self, output_file: Optional[str] = None) -> None:
        """
        Create comprehensive evaluation dashboard with all metrics.
        
        Args:
            output_file: Optional path to save dashboard HTML
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available. Cannot create interactive dashboard.")
            return None
        
        # Create a new interactive dashboard that combines all metrics
        
        # Create a subplot figure with multiple plots
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter", "colspan": 2}, None],
            ],
            subplot_titles=[
                "Precision-Recall Curve", "ROC Curve",
                "Confusion Matrix", "Performance by Class",
                "Metrics vs. Threshold"
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )
                
        # Get data for the dashboard
        label_type = self.metrics_calculator.label_type
        
        # Get threshold
        if self.threshold_optimizer is not None:
            threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
        else:
            threshold = 0.5
        
        # 1. Precision-Recall Curve
        precision, recall, _ = self.metrics_calculator.get_precision_recall_curve_data()
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name='Precision-Recall',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 2. ROC Curve
        # Get binary data for ROC curve
        true_binary = np.array([
            1 if l == self.metrics_calculator.class_labels[-1] else 0
            for l in self.metrics_calculator.true_labels
        ])
        score_binary = np.array(self.metrics_calculator.confidence_scores)
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(true_binary, score_binary)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC={roc_auc:.2f})',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Baseline',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Confusion Matrix
        cm = self.metrics_calculator.generate_confusion_matrix(threshold)
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=self.metrics_calculator.class_labels,
                y=self.metrics_calculator.class_labels,
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
            ),
            row=2, col=1
        )
        
        # 4. Performance by Class
        class_metrics = self.metrics_calculator.get_classification_metrics_by_class(threshold)
        
        classes = list(class_metrics.keys())
        f1_scores = [class_metrics[c].get('f1-score', 0) for c in classes]
        
        fig.add_trace(
            go.Bar(
                x=classes,
                y=f1_scores,
                name='F1 Score',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        # 5. Metrics vs. Threshold
        if self.threshold_optimizer is not None:
            threshold_data = self.threshold_optimizer.get_threshold_performance_data()
            
            fig.add_trace(
                go.Scatter(
                    x=threshold_data['threshold'],
                    y=threshold_data['precision'],
                    mode='lines',
                    name='Precision',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=threshold_data['threshold'],
                    y=threshold_data['recall'],
                    mode='lines',
                    name='Recall',
                    line=dict(color='green')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=threshold_data['threshold'],
                    y=threshold_data['f1'],
                    mode='lines',
                    name='F1 Score',
                    line=dict(color='red', width=3)
                ),
                row=3, col=1
            )
            
            # Add optimal threshold marker
            optimal_threshold = self.threshold_optimizer.get_optimal_thresholds().get(label_type, 0.5)
            
            fig.add_vline(
                x=optimal_threshold,
                line_width=2,
                line_dash="dash",
                line_color="purple",
                row=3, col=1
            )
            
            # Add annotation for optimal threshold
            optimal_f1 = threshold_data.loc[threshold_data['threshold'] == optimal_threshold, 'f1'].values[0]
            
            fig.add_annotation(
                x=optimal_threshold,
                y=optimal_f1 + 0.05,
                text=f"Optimal Threshold: {optimal_threshold:.2f}<br>F1: {optimal_f1:.2f}",
                showarrow=True,
                arrowhead=2,
                row=3, col=1
            )
            
        # Update layout
        fig.update_layout(
            title=f'Evaluation Dashboard - {label_type.capitalize()}',
            showlegend=True,
            legend=dict(x=0, y=-0.1, orientation='h'),
            height=900,
            width=1000,
            margin=dict(l=50, r=50, t=100, b=100),
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Recall', row=1, col=1)
        fig.update_yaxes(title_text='Precision', row=1, col=1)
        
        fig.update_xaxes(title_text='False Positive Rate', row=1, col=2)
        fig.update_yaxes(title_text='True Positive Rate', row=1, col=2)
        
        fig.update_xaxes(title_text='Predicted Label', row=2, col=1)
        fig.update_yaxes(title_text='True Label', row=2, col=1)
        
        fig.update_xaxes(title_text='Class', row=2, col=2)
        fig.update_yaxes(title_text='F1 Score', row=2, col=2)
        
        fig.update_xaxes(title_text='Confidence Threshold', row=3, col=1)
        fig.update_yaxes(title_text='Score', row=3, col=1)
        
        # Save dashboard if path provided
        if output_file:
            fig.write_html(output_file)
        
        # Store dashboard
        self.figures['dashboard'] = fig
        
        return fig
    
    def export_visualizations(self, output_dir: str, formats: List[str] = ["png", "pdf", "html"]) -> Dict:
        """
        Export all visualizations to specified formats.
        
        Args:
            output_dir: Directory to save visualizations in
            formats: List of formats to export ("png", "pdf", "html")
            
        Returns:
            Dictionary mapping figure names to saved file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Export each figure
        for name, fig in self.figures.items():
            saved_files[name] = {}
            
            # Skip if figure is None
            if fig is None:
                continue
                
            # Determine if matplotlib or plotly figure
            if isinstance(fig, plt.Figure):
                # Matplotlib figure
                for fmt in formats:
                    if fmt in ['png', 'pdf', 'jpg', 'svg']:
                        file_path = os.path.join(output_dir, f"{name}.{fmt}")
                        fig.savefig(file_path, bbox_inches='tight', dpi=300)
                        saved_files[name][fmt] = file_path
            elif PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
                # Plotly figure
                for fmt in formats:
                    file_path = os.path.join(output_dir, f"{name}.{fmt}")
                    
                    if fmt == 'html':
                        fig.write_html(file_path)
                        saved_files[name][fmt] = file_path
                    elif fmt in ['png', 'jpg', 'pdf', 'svg']:
                        fig.write_image(file_path)
                        saved_files[name][fmt] = file_path
            
        return saved_files
    
    def create_comparative_plot(
        self,
        other_metrics: Dict[str, EvaluationMetrics],
        label_type: str = "sentiment",
        metric: str = "f1",
        save_path: Optional[str] = None
    ):
        """
        Create plot comparing different models.
        
        Args:
            other_metrics: Dictionary mapping model names to metrics calculators
            label_type: Label type to visualize
            metric: Metric to compare ("f1", "precision", "recall")
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Add current model
        model_names = ['Current Model']
        model_metrics = [self.metrics_calculator.get_metrics_summary().get(metric, 0)]
        
        # Add other models
        for name, metrics_calc in other_metrics.items():
            model_names.append(name)
            model_metrics.append(metrics_calc.get_metrics_summary().get(metric, 0))
            
        # Plot bar chart
        bars = ax.bar(model_names, model_metrics, color='skyblue')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
            
        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric.capitalize()} Score')
        ax.set_title(f'Model Comparison - {metric.capitalize()} Score ({label_type.capitalize()})')
        ax.set_ylim(0, 1.0)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['model_comparison'] = fig
        
        return fig
    
    def plot_confidence_distribution(
        self,
        label_type: str = "sentiment",
        by_class: bool = False,
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of prediction confidence.
        
        Args:
            label_type: Label type to visualize
            by_class: Whether to separate distributions by class
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get confidence data
        confidence_scores = self.metrics_calculator.confidence_scores
        true_labels = self.metrics_calculator.true_labels
        
        if not by_class:
            # Plot overall confidence distribution
            sns.histplot(confidence_scores, kde=True, ax=ax, color='blue')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Prediction Confidence Distribution ({label_type.capitalize()})')
        else:
            # Plot confidence distribution by class
            confidence_df = pd.DataFrame({
                'confidence': confidence_scores,
                'class': true_labels
            })
            
            sns.histplot(
                data=confidence_df, 
                x='confidence', 
                hue='class', 
                kde=True, 
                ax=ax,
                palette='colorblind'
            )
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Prediction Confidence Distribution by Class ({label_type.capitalize()})')
            
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        # Store figure
        self.figures['confidence_distribution'] = fig
        
        return fig
    
    def create_streamlit_components(self) -> Dict:
        """
        Create components for integration with Streamlit UI.
        
        Returns:
            Dictionary of components for Streamlit
        """
        import matplotlib
        matplotlib.use('Agg')
        
        components = {}
        
        # Create metrics summary
        metrics_summary = self.metrics_calculator.get_metrics_summary()
        components['metrics'] = metrics_summary
        
        # Create confusion matrix plot
        cm_fig = self.plot_confusion_matrix()
        components['confusion_matrix'] = cm_fig
        
        # Create precision-recall curve
        pr_fig = self.plot_precision_recall_curve()
        components['pr_curve'] = pr_fig
        
        # Create class performance plot
        class_fig = self.plot_class_performance()
        components['class_performance'] = class_fig
        
        # Create interactive threshold tuner if Plotly is available
        if PLOTLY_AVAILABLE:
            interactive_fig = self.create_interactive_threshold_tuner()
            components['threshold_tuner'] = interactive_fig
            
        return components 
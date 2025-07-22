"""
Module for coordinating the evaluation process.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Union
import pickle

from src.evaluation.data_loader import TestDataLoader
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold import ThresholdOptimizer
from src.evaluation.visualization import EvaluationVisualizer


class ModelEvaluator:
    """Main class for model evaluation."""
    
    def __init__(
        self, 
        data_path: Optional[str] = None,
        predictions: Optional[List[Dict]] = None,
        ground_truth: Optional[List] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            data_path: Path to labeled test data
            predictions: Model predictions to evaluate
            ground_truth: Ground truth labels
        """
        self.data_path = data_path
        self.predictions = predictions
        self.ground_truth = ground_truth
        
        self.data_loader = None
        self.metrics_calculators = {}
        self.threshold_optimizers = {}
        self.visualizers = {}
        self.evaluation_results = {}
        
        # If data path is provided, load it
        if self.data_path:
            self.load_test_data(self.data_path)
            
        # If both predictions and ground truth are provided, perform evaluation
        if self.predictions and self.ground_truth:
            self.evaluate_model(predictions=self.predictions)
    
    def load_test_data(self, data_path: str, **kwargs) -> None:
        """
        Load labeled test data.
        
        Args:
            data_path: Path to labeled test data
            **kwargs: Additional arguments for TestDataLoader
        """
        self.data_path = data_path
        self.data_loader = TestDataLoader(data_path, **kwargs)
        
        # Load data and set ground truth
        self.data_loader.load_data()
        self.ground_truth = self.data_loader.get_labels()
    
    def evaluate_model(
        self,
        model=None,
        predictions: Optional[List[Dict]] = None,
        label_types: List[str] = ["sentiment", "emotion"]
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Model to evaluate (if predictions not provided)
            predictions: Model predictions to evaluate
            label_types: Label types to evaluate
            
        Returns:
            Dictionary of evaluation results by label type
        """
        # Ensure we have data to evaluate
        if not self.data_loader and not self.ground_truth:
            raise ValueError("No test data available. Load data first with load_test_data().")
            
        # If predictions not provided, but model is, use model to generate predictions
        if predictions is None and model is not None:
            if self.data_loader:
                texts = self.data_loader.get_texts()
                predictions = model.predict(texts)
            else:
                raise ValueError("No data available for model prediction. Load data first.")
                
        elif predictions is None:
            raise ValueError("Either model or predictions must be provided.")
            
        self.predictions = predictions
        
        # Perform evaluation for each label type
        results = {}
        
        for label_type in label_types:
            # Skip if label type not in ground truth
            if isinstance(self.ground_truth, dict) and label_type not in self.ground_truth:
                continue
                
            # Get ground truth for this label type
            if isinstance(self.ground_truth, dict):
                gt = self.ground_truth[label_type]
            else:
                gt = self.ground_truth
                
            # Create metrics calculator
            metrics_calculator = EvaluationMetrics(
                predictions=self.predictions,
                ground_truth=gt,
                label_type=label_type
            )
            
            # Create threshold optimizer
            threshold_optimizer = ThresholdOptimizer(
                metrics_calculator=metrics_calculator,
                metric_name="f1"
            )
            
            # Find optimal threshold
            optimal_threshold = threshold_optimizer.find_optimal_threshold(label_type)
            
            # Create visualizer
            visualizer = EvaluationVisualizer(
                metrics_calculator=metrics_calculator,
                threshold_optimizer=threshold_optimizer
            )
            
            # Store components
            self.metrics_calculators[label_type] = metrics_calculator
            self.threshold_optimizers[label_type] = threshold_optimizer
            self.visualizers[label_type] = visualizer
            
            # Get evaluation results
            results[label_type] = metrics_calculator.get_metrics_summary(optimal_threshold)
            
        # Store overall results
        self.evaluation_results = results
        
        return results
            
    def find_optimal_thresholds(
        self,
        label_types: List[str] = ["sentiment", "emotion"],
        metric: str = "f1"
    ) -> Dict[str, float]:
        """
        Find optimal thresholds for specified label types.
        
        Args:
            label_types: Label types to find thresholds for
            metric: Metric to optimize
            
        Returns:
            Dictionary mapping label types to optimal thresholds
        """
        thresholds = {}
        
        for label_type in label_types:
            if label_type not in self.threshold_optimizers:
                raise ValueError(f"No threshold optimizer available for {label_type}. Run evaluate_model() first.")
                
            # Set metric if different from current
            if self.threshold_optimizers[label_type].metric_name != metric:
                self.threshold_optimizers[label_type].metric_name = metric
                
            # Find optimal threshold
            threshold = self.threshold_optimizers[label_type].find_optimal_threshold(label_type)
            thresholds[label_type] = threshold
            
        return thresholds
    
    def visualize_results(self, label_type: str = "sentiment", output_dir: Optional[str] = None) -> None:
        """
        Generate visualizations for evaluation results.
        
        Args:
            label_type: Label type to visualize
            output_dir: Optional directory to save visualizations
        """
        if label_type not in self.visualizers:
            raise ValueError(f"No visualizer available for {label_type}. Run evaluate_model() first.")
            
        visualizer = self.visualizers[label_type]
        
        # Create standard visualizations
        visualizer.plot_confusion_matrix(label_type=label_type)
        visualizer.plot_precision_recall_curve(label_type=label_type)
        visualizer.plot_roc_curve(label_type=label_type)
        visualizer.plot_f1_vs_threshold(label_type=label_type)
        visualizer.plot_class_performance(label_type=label_type)
        visualizer.plot_metrics_summary(label_type=label_type)
        
        # Create interactive visualizations if possible
        visualizer.create_interactive_threshold_tuner(label_type=label_type)
        visualizer.create_dashboard()
        
        # Export visualizations if output directory provided
        if output_dir:
            visualizer.export_visualizations(output_dir)
    
    def export_results(self, output_path: str, format: str = "json") -> None:
        """
        Export evaluation results.
        
        Args:
            output_path: Path to save results
            format: Format to save in ("json", "csv", "pickle")
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
            
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2)
        elif format == "csv":
            # Convert nested dict to flat DataFrame
            rows = []
            for label_type, results in self.evaluation_results.items():
                row = {"label_type": label_type}
                row.update(results)
                rows.append(row)
                
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(self.evaluation_results, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_report(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report text if output_path not provided, otherwise None
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
            
        # Generate report text
        report = []
        report.append("# Evaluation Report")
        report.append("")
        
        if self.data_path:
            report.append(f"Test data: {self.data_path}")
            report.append(f"Number of samples: {len(self.data_loader.get_texts())}")
            report.append("")
        
        # Add summary of results
        report.append("## Summary")
        report.append("")
        
        for label_type, results in self.evaluation_results.items():
            report.append(f"### {label_type.capitalize()}")
            report.append(f"- Optimal threshold: {results['threshold']:.2f}")
            report.append(f"- F1 score: {results['f1']:.4f}")
            report.append(f"- Precision: {results['precision']:.4f}")
            report.append(f"- Recall: {results['recall']:.4f}")
            report.append(f"- AUC: {results.get('auc', 0):.4f}")
            report.append("")
            
        # Add detailed metrics by class
        report.append("## Detailed Metrics by Class")
        report.append("")
        
        for label_type, metrics_calc in self.metrics_calculators.items():
            threshold = self.evaluation_results[label_type]['threshold']
            class_metrics = metrics_calc.get_classification_metrics_by_class(threshold)
            
            report.append(f"### {label_type.capitalize()} - Class Metrics")
            report.append("")
            report.append("| Class | Precision | Recall | F1 Score | Support |")
            report.append("|-------|-----------|--------|----------|---------|")
            
            for class_name, metrics in class_metrics.items():
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                support = metrics.get('support', 0)
                
                report.append(f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |")
                
            report.append("")
        
        # Add threshold optimization information
        report.append("## Threshold Optimization")
        report.append("")
        
        for label_type, threshold_opt in self.threshold_optimizers.items():
            optimal_threshold = threshold_opt.get_optimal_thresholds().get(label_type, 0.5)
            
            report.append(f"### {label_type.capitalize()}")
            report.append(f"- Optimal threshold: {optimal_threshold:.2f}")
            report.append(f"- Optimization metric: {threshold_opt.metric_name}")
            report.append(f"- Threshold range: 0.1 - 0.9, step size: {threshold_opt.step_size}")
            report.append("")
            
            report.append("#### Performance at Different Thresholds")
            report.append("")
            report.append("| Threshold | Precision | Recall | F1 Score |")
            report.append("|-----------|-----------|--------|----------|")
            
            threshold_data = threshold_opt.get_threshold_metrics()
            for threshold, metrics in sorted(threshold_data.items()):
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1', 0)
                
                report.append(f"| {threshold:.2f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |")
                
            report.append("")
        
        # Join report lines
        report_text = "\n".join(report)
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            return None
            
        return report_text
    
    def compare_models(
        self,
        models_dict: Dict[str, Dict],
        label_type: str = "sentiment",
        metric: str = "f1"
    ) -> Dict:
        """
        Compare performance of multiple models.
        
        Args:
            models_dict: Dictionary mapping model names to prediction dictionaries
            label_type: Label type to compare on
            metric: Metric to use for comparison
            
        Returns:
            Dictionary of comparative results
        """
        # Ensure we have ground truth
        if not self.ground_truth:
            raise ValueError("No ground truth available. Load test data first.")
            
        comparison_results = {}
        
        # Current model results (if available)
        if self.evaluation_results:
            current_score = self.evaluation_results.get(label_type, {}).get(metric, 0)
            comparison_results["current_model"] = current_score
            
        # Evaluate each model
        for model_name, model_predictions in models_dict.items():
            # Create metrics calculator
            if isinstance(self.ground_truth, dict) and label_type in self.ground_truth:
                gt = self.ground_truth[label_type]
            else:
                gt = self.ground_truth
                
            metrics_calculator = EvaluationMetrics(
                predictions=model_predictions,
                ground_truth=gt,
                label_type=label_type
            )
            
            # Create threshold optimizer
            threshold_optimizer = ThresholdOptimizer(
                metrics_calculator=metrics_calculator,
                metric_name=metric
            )
            
            # Find optimal threshold
            optimal_threshold = threshold_optimizer.find_optimal_threshold(label_type)
            
            # Get metric at optimal threshold
            metrics = metrics_calculator.get_metrics_summary(optimal_threshold)
            score = metrics.get(metric, 0)
            
            # Store result
            comparison_results[model_name] = score
            
        # Order results by score
        sorted_results = dict(sorted(
            comparison_results.items(), 
            key=lambda item: item[1], 
            reverse=True
        ))
        
        return sorted_results
    
    def run_full_evaluation(
        self, 
        model=None, 
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            model: Model to evaluate
            data_path: Path to labeled test data
            output_dir: Optional directory for outputs
            
        Returns:
            Dictionary of evaluation results
        """
        # Load data if path provided
        if data_path:
            self.load_test_data(data_path)
            
        # Ensure we have data
        if not self.data_loader and not self.ground_truth:
            raise ValueError("No test data available. Provide data_path or load_test_data() first.")
            
        # Get texts for prediction if using model
        if model is not None:
            texts = self.data_loader.get_texts()
            predictions = model.predict(texts)
        else:
            # Check if we have predictions
            if not self.predictions:
                raise ValueError("Either model or predictions must be provided.")
            predictions = self.predictions
            
        # Run evaluation
        results = self.evaluate_model(predictions=predictions)
        
        # Generate visualizations
        for label_type in results.keys():
            self.visualize_results(label_type)
            
        # Export results if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export results as JSON
            self.export_results(os.path.join(output_dir, "evaluation_results.json"))
            
            # Generate and save report
            self.generate_report(os.path.join(output_dir, "evaluation_report.md"))
            
            # Export visualizations
            for label_type, visualizer in self.visualizers.items():
                vis_dir = os.path.join(output_dir, f"{label_type}_visualizations")
                visualizer.export_visualizations(vis_dir)
                
            # Export dashboard if available
            for label_type, visualizer in self.visualizers.items():
                dashboard_path = os.path.join(output_dir, f"{label_type}_dashboard.html")
                visualizer.create_dashboard(dashboard_path)
                
        return results
    
    def get_evaluation_summary(self, label_type: str = "sentiment") -> Dict:
        """
        Get summary of evaluation results.
        
        Args:
            label_type: Label type to get summary for
            
        Returns:
            Dictionary with evaluation summary
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
            
        if label_type not in self.evaluation_results:
            raise ValueError(f"No results for label type: {label_type}")
            
        return self.evaluation_results[label_type]
    
    def save_state(self, file_path: str) -> None:
        """
        Save evaluator state for later use.
        
        Args:
            file_path: Path to save state to
        """
        state = {
            "data_path": self.data_path,
            "predictions": self.predictions,
            "ground_truth": self.ground_truth,
            "evaluation_results": self.evaluation_results
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state(self, file_path: str) -> None:
        """
        Load evaluator state from file.
        
        Args:
            file_path: Path to load state from
        """
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            
        self.data_path = state.get("data_path")
        self.predictions = state.get("predictions")
        self.ground_truth = state.get("ground_truth")
        self.evaluation_results = state.get("evaluation_results", {})
        
        # Reload data if path available
        if self.data_path:
            self.load_test_data(self.data_path)
            
        # Recreate components if predictions and ground truth available
        if self.predictions and self.ground_truth:
            for label_type in self.evaluation_results.keys():
                # Get ground truth for this label type
                if isinstance(self.ground_truth, dict):
                    gt = self.ground_truth[label_type]
                else:
                    gt = self.ground_truth
                    
                # Recreate metrics calculator
                metrics_calculator = EvaluationMetrics(
                    predictions=self.predictions,
                    ground_truth=gt,
                    label_type=label_type
                )
                
                # Recreate threshold optimizer
                threshold_optimizer = ThresholdOptimizer(
                    metrics_calculator=metrics_calculator
                )
                
                # Recreate visualizer
                visualizer = EvaluationVisualizer(
                    metrics_calculator=metrics_calculator,
                    threshold_optimizer=threshold_optimizer
                )
                
                # Store components
                self.metrics_calculators[label_type] = metrics_calculator
                self.threshold_optimizers[label_type] = threshold_optimizer
                self.visualizers[label_type] = visualizer 
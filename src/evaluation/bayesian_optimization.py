"""
Bayesian optimization for threshold parameter tuning.
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Union, Any
import logging

try:
    import skopt
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    # Define a fallback decorator if skopt is not available
    def use_named_args(dimensions):
        def decorator(func):
            return func
        return decorator

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold import ThresholdOptimizer


class BayesianThresholdOptimizer:
    """
    Bayesian optimization for finding optimal confidence thresholds.
    Uses Gaussian process or random forests to efficiently find optimal thresholds
    with fewer evaluations than grid search.
    """
    
    def __init__(
        self,
        metrics_calculator: EvaluationMetrics,
        metric_name: str = "f1",
        n_calls: int = 20,
        method: str = "gp",
        random_state: int = 42,
        exploration_factor: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Bayesian threshold optimizer.
        
        Args:
            metrics_calculator: Instance of metrics calculator
            metric_name: Metric to optimize ("f1", "precision", "recall")
            n_calls: Number of evaluations for optimization
            method: Optimization method ("gp" for Gaussian Process, "rf" for Random Forest)
            random_state: Random seed for reproducibility
            exploration_factor: Exploration factor (higher values=more exploration)
            logger: Optional logger instance
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization. Install with: pip install scikit-optimize")
            
        self.metrics_calculator = metrics_calculator
        self.metric_name = metric_name
        self.n_calls = n_calls
        self.method = method
        self.random_state = random_state
        self.exploration_factor = exploration_factor
        self.logger = logger
        
        # Results storage
        self.threshold_range = (0.1, 0.9)  # Default range
        self.best_threshold = None
        self.optimization_results = None
        self.label_type = self.metrics_calculator.label_type
        self.thresholds_evaluated = []
        self.scores_evaluated = []
        
    def _objective_function(self, threshold: float) -> float:
        """
        Objective function to minimize (negative metric score).
        
        Args:
            threshold: Threshold value to evaluate
            
        Returns:
            Negative metric score (for minimization)
        """
        if self.logger:
            self.logger.debug(f"Evaluating threshold: {threshold:.4f}")
            
        # Calculate metric based on name
        if self.metric_name == "f1":
            score = self.metrics_calculator.calculate_f1(threshold=threshold)
        elif self.metric_name == "precision":
            score, _ = self.metrics_calculator.calculate_precision_recall(threshold=threshold)
        elif self.metric_name == "recall":
            _, score = self.metrics_calculator.calculate_precision_recall(threshold=threshold)
        else:
            # Default to F1
            score = self.metrics_calculator.calculate_f1(threshold=threshold)
        
        # Store evaluation result
        self.thresholds_evaluated.append(threshold)
        self.scores_evaluated.append(score)
            
        # Return negative score for minimization
        return -score
    
    def optimize(self, threshold_range: Tuple[float, float] = (0.1, 0.9)) -> float:
        """
        Run Bayesian optimization to find optimal threshold.
        
        Args:
            threshold_range: Tuple of (min_threshold, max_threshold)
            
        Returns:
            Optimal threshold value
        """
        self.threshold_range = threshold_range
        
        if self.logger:
            self.logger.info(
                f"Starting Bayesian optimization for {self.label_type} with "
                f"{self.n_calls} evaluations using {self.method} method"
            )
        
        # Define the search space
        space = [Real(threshold_range[0], threshold_range[1], name='threshold')]
        
        # Define the objective function as a closure
        def objective(params):
            # skopt passes a list of parameters
            threshold = params[0] if isinstance(params, (list, tuple)) else params
            return self._objective_function(threshold)
        
        # Determine optimization method
        if self.method == "gp":
            # Gaussian Process optimization
            optimizer = gp_minimize
            extra_args = {
                "acq_func": "EI",  # Expected Improvement
                "noise": self.exploration_factor ** 2,  # Noise level
                "n_random_starts": min(5, self.n_calls // 4),  # Start with some random points
            }
        elif self.method == "rf":
            # Random Forest optimization
            optimizer = forest_minimize
            extra_args = {
                "base_estimator": "RF",  # Random Forest
                "n_random_starts": min(5, self.n_calls // 4),  # Start with some random points
            }
        else:
            if self.logger:
                self.logger.warning(f"Unknown method: {self.method}, defaulting to GP")
            # Default to GP
            optimizer = gp_minimize
            extra_args = {
                "acq_func": "EI",
                "noise": self.exploration_factor ** 2,
                "n_random_starts": min(5, self.n_calls // 4),
            }
        
        # Run optimization
        result = optimizer(
            objective,
            space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            verbose=self.logger is not None and self.logger.level <= logging.DEBUG,
            **extra_args
        )
        
        # Store results
        self.optimization_results = result
        self.best_threshold = result.x[0]
        
        if self.logger:
            self.logger.info(
                f"Optimal threshold found: {self.best_threshold:.4f} "
                f"with {self.metric_name}={-result.fun:.4f}"
            )
        
        return self.best_threshold
    
    def get_best_threshold(self) -> float:
        """
        Get the optimal threshold.
        
        Returns:
            Optimal threshold value
        """
        if self.best_threshold is None:
            # Run optimization with default range if not already done
            self.optimize()
            
        return self.best_threshold
    
    def get_threshold_performance_data(self) -> Dict:
        """
        Get threshold performance data.
        
        Returns:
            Dictionary of threshold performance data
        """
        if self.optimization_results is None:
            # Run optimization with default range if not already done
            self.optimize()
            
        # Create data dictionary
        data = {
            "thresholds_evaluated": self.thresholds_evaluated,
            "scores_evaluated": [-s for s in self.optimization_results.func_vals],  # Convert back to positive scores
            "best_threshold": self.best_threshold,
            "best_score": -self.optimization_results.fun,  # Convert back to positive score
            "n_evaluations": len(self.optimization_results.x_iters),
            "exploration_factor": self.exploration_factor,
            "method": self.method
        }
        
        return data
    
    def plot_optimization_results(self, output_path: Optional[str] = None):
        """
        Plot the optimization results.
        
        Args:
            output_path: Optional path to save the plot
        
        Returns:
            matplotlib.Figure if available
        """
        try:
            import matplotlib.pyplot as plt
            from skopt.plots import plot_convergence, plot_objective
            
            if self.optimization_results is None:
                if self.logger:
                    self.logger.warning("No optimization results to plot")
                return None
            
            # Create a figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot convergence
            plot_convergence(self.optimization_results, ax=ax1)
            ax1.set_title(f"Convergence Plot ({self.label_type})")
            
            # Plot evaluated thresholds and scores
            thresholds = self.thresholds_evaluated
            scores = [-score for score in self.scores_evaluated]  # Convert back to positive scores
            
            # Sort by threshold for better visualization
            sorted_indices = np.argsort(thresholds)
            sorted_thresholds = [thresholds[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]
            
            ax2.plot(sorted_thresholds, sorted_scores, 'o-', markersize=5)
            ax2.set_title(f"{self.metric_name.capitalize()} vs Threshold ({self.label_type})")
            ax2.set_xlabel("Threshold")
            ax2.set_ylabel(f"{self.metric_name.capitalize()} Score")
            ax2.grid(True)
            
            # Mark the best threshold
            best_score = -self.optimization_results.fun
            ax2.plot(self.best_threshold, best_score, 'ro', markersize=10,
                    label=f"Best: {self.best_threshold:.3f} ({best_score:.3f})")
            ax2.legend()
            
            plt.tight_layout()
            
            # Save to file if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            return fig
        except ImportError:
            if self.logger:
                self.logger.warning("Matplotlib or scikit-optimize plotting not available")
            return None


def bayesian_optimize_thresholds(
    metrics_calculator: EvaluationMetrics,
    label_type: str = "sentiment",
    metric: str = "f1",
    n_calls: int = 20,
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    method: str = "gp",
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Perform Bayesian optimization to find optimal threshold.
    
    Args:
        metrics_calculator: Metrics calculator instance
        label_type: Type of label to optimize for (sentiment, emotion)
        metric: Metric to optimize (f1, precision, recall)
        n_calls: Number of evaluations for optimization
        threshold_range: Range of thresholds to search
        method: Optimization method (gp, rf)
        logger: Optional logger
        
    Returns:
        Dictionary of optimization results
    """
    optimizer = BayesianThresholdOptimizer(
        metrics_calculator=metrics_calculator,
        metric_name=metric,
        n_calls=n_calls,
        method=method,
        logger=logger
    )
    
    # Run optimization
    optimal_threshold = optimizer.optimize(threshold_range)
    
    # Get performance data
    perf_data = optimizer.get_threshold_performance_data()
    
    return {
        "optimal_threshold": optimal_threshold,
        "performance_data": perf_data
    } 
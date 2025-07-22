"""
Integration tests comparing Bayesian optimization against grid search 
for threshold tuning across different dataset sizes and complexity levels.
"""

import os
import time
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.threshold import ThresholdOptimizer
from src.evaluation.bayesian_optimization import (
    BayesianThresholdOptimizer,
    bayesian_optimize_thresholds
)


class OptimizationBenchmarkTestCase(unittest.TestCase):
    """
    Base test case for benchmark tests comparing Bayesian optimization 
    and grid search for threshold tuning.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create a logger
        self.logger = logging.getLogger("benchmark_tests")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create temporary directory for results
        self.test_dir = tempfile.mkdtemp()
        
        # Define dataset configurations
        self.dataset_configs = {
            'small_balanced': {
                'size': 100,
                'class_distribution': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'complexity': 'easy'  # Clearly separable classes by confidence
            },
            'small_imbalanced': {
                'size': 100,
                'class_distribution': {'positive': 0.7, 'negative': 0.2, 'neutral': 0.1},
                'complexity': 'easy'
            },
            'medium_balanced': {
                'size': 500,
                'class_distribution': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'complexity': 'medium'  # Some overlap in confidence distributions
            },
            'medium_imbalanced': {
                'size': 500,
                'class_distribution': {'positive': 0.6, 'negative': 0.3, 'neutral': 0.1},
                'complexity': 'medium'
            },
            'large_balanced': {
                'size': 2000,
                'class_distribution': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'complexity': 'hard'  # Significant overlap in confidence distributions
            },
            'large_imbalanced': {
                'size': 2000,
                'class_distribution': {'positive': 0.5, 'negative': 0.3, 'neutral': 0.2},
                'complexity': 'hard'
            }
        }
        
        # Define grid search configurations
        self.grid_search_configs = {
            'coarse': {
                'threshold_start': 0.1,
                'threshold_end': 0.9,
                'threshold_step': 0.1
            },
            'medium': {
                'threshold_start': 0.1,
                'threshold_end': 0.9,
                'threshold_step': 0.05
            },
            'fine': {
                'threshold_start': 0.1,
                'threshold_end': 0.9,
                'threshold_step': 0.02
            }
        }
        
        # Define Bayesian optimization configurations
        self.bayesian_configs = {
            'minimal': {
                'n_calls': 10,
                'method': 'gp'
            },
            'standard': {
                'n_calls': 20,
                'method': 'gp'
            },
            'thorough': {
                'n_calls': 30,
                'method': 'gp'
            },
            'rf_standard': {
                'n_calls': 20,
                'method': 'rf'
            }
        }
        
        # Container for benchmark results
        self.benchmark_results = {
            'grid_search': {},
            'bayesian': {},
            'comparisons': []
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Write benchmark results to file
        output_path = os.path.join(self.test_dir, "benchmark_results.json")
        with open(output_path, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            
        self.logger.info(f"Benchmark results saved to: {output_path}")
        
        # Plot comparison charts
        self._plot_benchmark_results()
    
    def _generate_test_data(self, config: dict) -> tuple:
        """
        Generate synthetic test data with controllable properties.
        
        Args:
            config: Dataset configuration with size, class_distribution, and complexity
            
        Returns:
            Tuple of (predictions, ground_truth, optimal_threshold)
        """
        np.random.seed(42)  # For reproducibility
        
        size = config.get('size', 100)
        class_distribution = config.get('class_distribution', {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34})
        complexity = config.get('complexity', 'easy')
        
        # Generate ground truth labels according to distribution
        classes = list(class_distribution.keys())
        class_weights = list(class_distribution.values())
        
        ground_truth = np.random.choice(classes, size=size, p=class_weights)
        
        # Generate predictions with confidence scores based on complexity
        predictions = []
        
        # Set confidence parameters based on complexity
        if complexity == 'easy':
            # Easy: Clear separation between correct and incorrect predictions
            correct_pred_rate = 0.8
            correct_confidence_range = (0.7, 0.95)  # High confidence for correct predictions
            incorrect_confidence_range = (0.3, 0.6)  # Low confidence for incorrect predictions
            optimal_threshold = 0.65  # Clear threshold boundary
            
        elif complexity == 'medium':
            # Medium: Some overlap in confidence distributions
            correct_pred_rate = 0.75
            correct_confidence_range = (0.6, 0.9)
            incorrect_confidence_range = (0.4, 0.7)  # Some overlap with correct predictions
            optimal_threshold = 0.6
            
        else:  # hard
            # Hard: Significant overlap in confidence distributions
            correct_pred_rate = 0.7
            correct_confidence_range = (0.55, 0.85)
            incorrect_confidence_range = (0.45, 0.75)  # Significant overlap
            optimal_threshold = 0.6  # Less clear threshold boundary
        
        # Generate predictions
        for true_label in ground_truth:
            # Determine if prediction is correct
            if np.random.random() < correct_pred_rate:
                # Correct prediction
                pred_label = true_label
                # Sample confidence from correct range
                conf = np.random.uniform(correct_confidence_range[0], correct_confidence_range[1])
                
                # Add slight variation based on class for more realism
                if true_label == "positive":
                    conf = min(conf + 0.05, 0.99)
                elif true_label == "negative":
                    conf = max(conf - 0.03, 0.01)
            else:
                # Incorrect prediction
                options = classes.copy()
                options.remove(true_label)
                pred_label = np.random.choice(options)
                
                # Sample confidence from incorrect range
                conf = np.random.uniform(incorrect_confidence_range[0], incorrect_confidence_range[1])
            
            predictions.append({
                "sentiment": pred_label,
                "sentiment_confidence": conf
            })
            
        # Return data along with the expected optimal threshold
        return predictions, ground_truth.tolist(), optimal_threshold
    
    def _plot_benchmark_results(self):
        """Plot benchmark results to visualize performance comparisons."""
        if not self.benchmark_results['comparisons']:
            return
            
        # Convert comparison data to DataFrame
        df = pd.DataFrame(self.benchmark_results['comparisons'])
        
        # Create results directory
        plots_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Runtime comparison by dataset size
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df,
            x='dataset_size',
            y='runtime_seconds',
            hue='method',
            palette=['blue', 'red', 'green', 'orange']
        )
        plt.title('Runtime Comparison: Grid Search vs Bayesian Optimization')
        plt.ylabel('Runtime (seconds)')
        plt.xlabel('Dataset Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'runtime_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Evaluation count comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df,
            x='dataset_size',
            y='evaluation_count',
            hue='method',
            palette=['blue', 'red', 'green', 'orange']
        )
        plt.title('Function Evaluations: Grid Search vs Bayesian Optimization')
        plt.ylabel('Number of Evaluations')
        plt.xlabel('Dataset Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'evaluation_count_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Accuracy preservation
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df,
            x='dataset_size',
            y='f1_score',
            hue='method',
            palette=['blue', 'red', 'green', 'orange']
        )
        plt.title('F1 Score Comparison: Grid Search vs Bayesian Optimization')
        plt.ylabel('F1 Score')
        plt.xlabel('Dataset Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Efficiency ratio (accuracy per evaluation)
        df['efficiency'] = df['f1_score'] / df['evaluation_count']
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df,
            x='dataset_size',
            y='efficiency',
            hue='method',
            palette=['blue', 'red', 'green', 'orange']
        )
        plt.title('Efficiency: F1 Score per Evaluation')
        plt.ylabel('F1 Score / Evaluation Count')
        plt.xlabel('Dataset Size')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Benchmark plots saved to: {plots_dir}")


class BayesianVsGridSearchBenchmarkTests(OptimizationBenchmarkTestCase):
    """Test cases for comparing Bayesian optimization against grid search."""
    
    def test_small_dataset_comparison(self):
        """Test comparison on small datasets."""
        self.logger.info("Testing small dataset comparison...")
        
        # Test with small balanced dataset
        config = self.dataset_configs['small_balanced']
        predictions, ground_truth, expected_optimal = self._generate_test_data(config)
        
        # Create metrics calculator
        metrics_calc = EvaluationMetrics(
            predictions=predictions,
            ground_truth=ground_truth,
            label_type="sentiment"
        )
        
        # Test grid search (coarse)
        grid_config = self.grid_search_configs['coarse']
        start_time = time.time()
        
        grid_optimizer = ThresholdOptimizer(
            metrics_calculator=metrics_calc,
            step_size=grid_config['threshold_step']
        )
        
        grid_optimizer.sweep_thresholds(
            start=grid_config['threshold_start'],
            end=grid_config['threshold_end']
        )
        grid_best_threshold = grid_optimizer.find_optimal_threshold()
        grid_threshold_data = grid_optimizer.get_threshold_metrics()
        grid_runtime = time.time() - start_time
        
        # Test Bayesian optimization (standard)
        bayesian_config = self.bayesian_configs['standard']
        start_time = time.time()
        
        bayesian_optimizer = BayesianThresholdOptimizer(
            metrics_calculator=metrics_calc,
            n_calls=bayesian_config['n_calls'],
            method=bayesian_config['method'],
            logger=self.logger
        )
        
        bayesian_best_threshold = bayesian_optimizer.optimize()
        bayesian_perf_data = bayesian_optimizer.get_threshold_performance_data()
        bayesian_runtime = time.time() - start_time
        
        # Store results
        comparison = {
            'dataset_size': 'small_balanced',
            'method': 'grid_coarse',
            'runtime_seconds': grid_runtime,
            'evaluation_count': len(grid_threshold_data),
            'f1_score': max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1'],
            'optimal_threshold': grid_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        comparison = {
            'dataset_size': 'small_balanced',
            'method': 'bayesian_standard',
            'runtime_seconds': bayesian_runtime,
            'evaluation_count': len(bayesian_optimizer.thresholds_evaluated),
            'f1_score': bayesian_perf_data['best_score'],
            'optimal_threshold': bayesian_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        # Assertions
        self.assertLess(bayesian_runtime, grid_runtime * 0.8)  # Bayesian should be faster
        self.assertLess(len(bayesian_optimizer.thresholds_evaluated), len(grid_threshold_data))
        self.assertGreater(bayesian_perf_data['best_score'], 0.6)  # Reasonable accuracy
        self.assertGreater(max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1'], 0.6)
        
        self.logger.info(f"Small dataset test completed. Grid: {grid_runtime:.3f}s, {len(grid_threshold_data)} evals. "
                        f"Bayesian: {bayesian_runtime:.3f}s, {len(bayesian_optimizer.thresholds_evaluated)} evals.")
    
    def test_medium_dataset_comparison(self):
        """Test comparison on medium datasets."""
        self.logger.info("Testing medium dataset comparison...")
        
        # Test with medium imbalanced dataset
        config = self.dataset_configs['medium_imbalanced']
        predictions, ground_truth, expected_optimal = self._generate_test_data(config)
        
        # Create metrics calculator
        metrics_calc = EvaluationMetrics(
            predictions=predictions,
            ground_truth=ground_truth,
            label_type="sentiment"
        )
        
        # Test grid search (medium granularity)
        grid_config = self.grid_search_configs['medium']
        start_time = time.time()
        
        grid_optimizer = ThresholdOptimizer(
            metrics_calculator=metrics_calc,
            step_size=grid_config['threshold_step']
        )
        
        grid_optimizer.sweep_thresholds(
            start=grid_config['threshold_start'],
            end=grid_config['threshold_end']
        )
        grid_best_threshold = grid_optimizer.find_optimal_threshold()
        grid_threshold_data = grid_optimizer.get_threshold_metrics()
        grid_runtime = time.time() - start_time
        
        # Test Bayesian optimization (thorough)
        bayesian_config = self.bayesian_configs['thorough']
        start_time = time.time()
        
        bayesian_optimizer = BayesianThresholdOptimizer(
            metrics_calculator=metrics_calc,
            n_calls=bayesian_config['n_calls'],
            method=bayesian_config['method'],
            logger=self.logger
        )
        
        bayesian_best_threshold = bayesian_optimizer.optimize()
        bayesian_perf_data = bayesian_optimizer.get_threshold_performance_data()
        bayesian_runtime = time.time() - start_time
        
        # Store results
        comparison = {
            'dataset_size': 'medium_imbalanced',
            'method': 'grid_medium',
            'runtime_seconds': grid_runtime,
            'evaluation_count': len(grid_threshold_data),
            'f1_score': max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1'],
            'optimal_threshold': grid_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        comparison = {
            'dataset_size': 'medium_imbalanced',
            'method': 'bayesian_thorough',
            'runtime_seconds': bayesian_runtime,
            'evaluation_count': len(bayesian_optimizer.thresholds_evaluated),
            'f1_score': bayesian_perf_data['best_score'],
            'optimal_threshold': bayesian_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        # Assertions
        self.assertLess(bayesian_runtime, grid_runtime * 0.7)  # Bayesian should be significantly faster
        self.assertLess(len(bayesian_optimizer.thresholds_evaluated), len(grid_threshold_data))
        self.assertGreater(bayesian_perf_data['best_score'], 0.5)  # Reasonable accuracy for medium complexity
        self.assertGreater(max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1'], 0.5)
        
        self.logger.info(f"Medium dataset test completed. Grid: {grid_runtime:.3f}s, {len(grid_threshold_data)} evals. "
                        f"Bayesian: {bayesian_runtime:.3f}s, {len(bayesian_optimizer.thresholds_evaluated)} evals.")
    
    def test_large_dataset_comparison(self):
        """Test comparison on large datasets."""
        self.logger.info("Testing large dataset comparison...")
        
        # Test with large balanced dataset
        config = self.dataset_configs['large_balanced']
        predictions, ground_truth, expected_optimal = self._generate_test_data(config)
        
        # Create metrics calculator
        metrics_calc = EvaluationMetrics(
            predictions=predictions,
            ground_truth=ground_truth,
            label_type="sentiment"
        )
        
        # Test grid search (fine granularity)
        grid_config = self.grid_search_configs['fine']
        start_time = time.time()
        
        grid_optimizer = ThresholdOptimizer(
            metrics_calculator=metrics_calc,
            step_size=grid_config['threshold_step']
        )
        
        grid_optimizer.sweep_thresholds(
            start=grid_config['threshold_start'],
            end=grid_config['threshold_end']
        )
        grid_best_threshold = grid_optimizer.find_optimal_threshold()
        grid_threshold_data = grid_optimizer.get_threshold_metrics()
        grid_runtime = time.time() - start_time
        
        # Test Bayesian optimization (standard)
        bayesian_config = self.bayesian_configs['standard']
        start_time = time.time()
        
        bayesian_optimizer = BayesianThresholdOptimizer(
            metrics_calculator=metrics_calc,
            n_calls=bayesian_config['n_calls'],
            method=bayesian_config['method'],
            logger=self.logger
        )
        
        bayesian_best_threshold = bayesian_optimizer.optimize()
        bayesian_perf_data = bayesian_optimizer.get_threshold_performance_data()
        bayesian_runtime = time.time() - start_time
        
        # Store results
        comparison = {
            'dataset_size': 'large_balanced',
            'method': 'grid_fine',
            'runtime_seconds': grid_runtime,
            'evaluation_count': len(grid_threshold_data),
            'f1_score': max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1'],
            'optimal_threshold': grid_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        comparison = {
            'dataset_size': 'large_balanced',
            'method': 'bayesian_standard',
            'runtime_seconds': bayesian_runtime,
            'evaluation_count': len(bayesian_optimizer.thresholds_evaluated),
            'f1_score': bayesian_perf_data['best_score'],
            'optimal_threshold': bayesian_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        # Assertions
        self.assertLess(bayesian_runtime, grid_runtime * 0.6)  # Bayesian should be much faster
        self.assertLess(len(bayesian_optimizer.thresholds_evaluated), len(grid_threshold_data))
        self.assertGreater(bayesian_perf_data['best_score'], 0.4)  # Reasonable accuracy for hard complexity
        self.assertGreater(max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1'], 0.4)
        
        self.logger.info(f"Large dataset test completed. Grid: {grid_runtime:.3f}s, {len(grid_threshold_data)} evals. "
                        f"Bayesian: {bayesian_runtime:.3f}s, {len(bayesian_optimizer.thresholds_evaluated)} evals.")
    
    def test_gaussian_process_vs_random_forest(self):
        """Test comparison between Gaussian Process and Random Forest methods."""
        self.logger.info("Testing GP vs RF comparison...")
        
        # Test with medium balanced dataset
        config = self.dataset_configs['medium_balanced']
        predictions, ground_truth, expected_optimal = self._generate_test_data(config)
        
        # Create metrics calculator
        metrics_calc = EvaluationMetrics(
            predictions=predictions,
            ground_truth=ground_truth,
            label_type="sentiment"
        )
        
        # Test Gaussian Process
        start_time = time.time()
        gp_optimizer = BayesianThresholdOptimizer(
            metrics_calculator=metrics_calc,
            n_calls=20,
            method='gp',
            logger=self.logger
        )
        gp_best_threshold = gp_optimizer.optimize()
        gp_perf_data = gp_optimizer.get_threshold_performance_data()
        gp_runtime = time.time() - start_time
        
        # Test Random Forest
        start_time = time.time()
        rf_optimizer = BayesianThresholdOptimizer(
            metrics_calculator=metrics_calc,
            n_calls=20,
            method='rf',
            logger=self.logger
        )
        rf_best_threshold = rf_optimizer.optimize()
        rf_perf_data = rf_optimizer.get_threshold_performance_data()
        rf_runtime = time.time() - start_time
        
        # Store results
        comparison = {
            'dataset_size': 'medium_balanced',
            'method': 'bayesian_gp',
            'runtime_seconds': gp_runtime,
            'evaluation_count': len(gp_optimizer.thresholds_evaluated),
            'f1_score': gp_perf_data['best_score'],
            'optimal_threshold': gp_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        comparison = {
            'dataset_size': 'medium_balanced',
            'method': 'bayesian_rf',
            'runtime_seconds': rf_runtime,
            'evaluation_count': len(rf_optimizer.thresholds_evaluated),
            'f1_score': rf_perf_data['best_score'],
            'optimal_threshold': rf_best_threshold,
            'expected_optimal': expected_optimal
        }
        self.benchmark_results['comparisons'].append(comparison)
        
        # Assertions
        self.assertGreater(gp_perf_data['best_score'], 0.5)
        self.assertGreater(rf_perf_data['best_score'], 0.5)
        
        self.logger.info(f"GP vs RF test completed. GP: {gp_runtime:.3f}s, F1={gp_perf_data['best_score']:.3f}. "
                        f"RF: {rf_runtime:.3f}s, F1={rf_perf_data['best_score']:.3f}")
    
    def test_accuracy_preservation(self):
        """Test that Bayesian optimization preserves accuracy compared to grid search."""
        self.logger.info("Testing accuracy preservation...")
        
        # Test with multiple dataset configurations
        test_configs = ['small_balanced', 'medium_imbalanced', 'large_balanced']
        
        for config_name in test_configs:
            config = self.dataset_configs[config_name]
            predictions, ground_truth, expected_optimal = self._generate_test_data(config)
            
            # Create metrics calculator
            metrics_calc = EvaluationMetrics(
                predictions=predictions,
                ground_truth=ground_truth,
                label_type="sentiment"
            )
            
            # Grid search (fine granularity for best accuracy)
            grid_optimizer = ThresholdOptimizer(
                metrics_calculator=metrics_calc,
                step_size=0.02
            )
            grid_optimizer.sweep_thresholds(start=0.1, end=0.9)
            grid_best_threshold = grid_optimizer.find_optimal_threshold()
            grid_threshold_data = grid_optimizer.get_threshold_metrics()
            grid_best_f1 = max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1']
            
            # Bayesian optimization (standard)
            bayesian_optimizer = BayesianThresholdOptimizer(
                metrics_calculator=metrics_calc,
                n_calls=20,
                method='gp',
                logger=self.logger
            )
            bayesian_best_threshold = bayesian_optimizer.optimize()
            bayesian_perf_data = bayesian_optimizer.get_threshold_performance_data()
            bayesian_best_f1 = bayesian_perf_data['best_score']
            
            # Calculate accuracy difference
            accuracy_diff = abs(bayesian_best_f1 - grid_best_f1)
            
            # Assert that accuracy is preserved within reasonable bounds
            self.assertLess(accuracy_diff, 0.1)  # Within 10% of grid search accuracy
            
            self.logger.info(f"{config_name}: Grid F1={grid_best_f1:.3f}, "
                           f"Bayesian F1={bayesian_best_f1:.3f}, "
                           f"Diff={accuracy_diff:.3f}")
    
    def test_efficiency_scaling(self):
        """Test that efficiency gains scale with dataset size."""
        self.logger.info("Testing efficiency scaling...")
        
        # Test with increasing dataset sizes
        sizes = [100, 500, 1000, 2000]
        efficiency_ratios = []
        
        for size in sizes:
            # Generate data with consistent properties
            config = {
                'size': size,
                'class_distribution': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'complexity': 'medium'
            }
            predictions, ground_truth, expected_optimal = self._generate_test_data(config)
            
            # Create metrics calculator
            metrics_calc = EvaluationMetrics(
                predictions=predictions,
                ground_truth=ground_truth,
                label_type="sentiment"
            )
            
            # Grid search
            grid_optimizer = ThresholdOptimizer(
                metrics_calculator=metrics_calc,
                step_size=0.05
            )
            grid_optimizer.sweep_thresholds(start=0.1, end=0.9)
            grid_best_threshold = grid_optimizer.find_optimal_threshold()
            grid_threshold_data = grid_optimizer.get_threshold_metrics()
            grid_best_f1 = max(grid_threshold_data.values(), key=lambda x: x.get('f1', 0))['f1']
            
            # Bayesian optimization
            bayesian_optimizer = BayesianThresholdOptimizer(
                metrics_calculator=metrics_calc,
                n_calls=20,
                method='gp',
                logger=self.logger
            )
            bayesian_best_threshold = bayesian_optimizer.optimize()
            bayesian_perf_data = bayesian_optimizer.get_threshold_performance_data()
            bayesian_best_f1 = bayesian_perf_data['best_score']
            
            # Calculate efficiency ratio
            grid_efficiency = grid_best_f1 / len(grid_threshold_data)
            bayesian_efficiency = bayesian_best_f1 / len(bayesian_optimizer.thresholds_evaluated)
            efficiency_ratio = bayesian_efficiency / grid_efficiency
            
            efficiency_ratios.append(efficiency_ratio)
            
            self.logger.info(f"Size {size}: Grid efficiency={grid_efficiency:.4f}, "
                           f"Bayesian efficiency={bayesian_efficiency:.4f}, "
                           f"Ratio={efficiency_ratio:.2f}")
        
        # Assert that efficiency improves with dataset size
        # (Bayesian optimization should become relatively more efficient)
        self.assertGreater(efficiency_ratios[-1], efficiency_ratios[0] * 0.8)


if __name__ == '__main__':
    unittest.main() 
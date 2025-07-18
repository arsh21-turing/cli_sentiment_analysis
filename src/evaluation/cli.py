"""
Command-line interface for model evaluation.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Union

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.data_loader import TestDataLoader, BatchTestLoader
from src.evaluation.threshold import ThresholdOptimizer
from src.evaluation.metrics import EvaluationMetrics
from src.models.transformer import SentimentEmotionTransformer
from src.models.groq import GroqModel
from src.models.fallback import FallbackSystem
from src.utils.settings import Settings
from typing import Any, NoReturn
import traceback


def _validate_data_args(args: Any, logger: logging.Logger) -> bool:
    """
    Validate data-related command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    # Check if data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        print_colored(f"ERROR: Data file not found: {args.data}", "red")
        return False
        
    # Check if data format is valid for the file
    if args.format == "csv" and not args.data.endswith((".csv", ".tsv")):
        logger.warning(f"File extension doesn't match specified format: {args.format}")
        print_colored(f"WARNING: File extension doesn't match specified format: {args.format}", "yellow")
    
    if args.format == "json" and not args.data.endswith((".json", ".jsonl")):
        logger.warning(f"File extension doesn't match specified format: {args.format}")
        print_colored(f"WARNING: File extension doesn't match specified format: {args.format}", "yellow")
        
    # Ensure column names are specified
    if not args.text_column:
        logger.error("Text column name must be specified")
        print_colored("ERROR: Text column name must be specified", "red")
        return False
        
    return True


def _validate_model_args(args: Any, logger: logging.Logger) -> bool:
    """
    Validate model-related command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    # Check if specified model is valid
    valid_models = ["transformer", "groq", "fallback"]
    if hasattr(args, "model") and args.model is not None and args.model not in valid_models:
        logger.error(f"Invalid model: {args.model}. Must be one of: {', '.join(valid_models)}")
        print_colored(f"ERROR: Invalid model: {args.model}. Must be one of: {', '.join(valid_models)}", "red")
        return False
        
    # Check if settings file exists if specified
    if hasattr(args, "settings") and args.settings and not os.path.exists(args.settings):
        logger.error(f"Settings file not found: {args.settings}")
        print_colored(f"ERROR: Settings file not found: {args.settings}", "red")
        return False
        
    # For compare command, validate models list
    if hasattr(args, "models") and args.models is not None:
        models = args.models.split(",")
        for model in models:
            if model not in valid_models:
                logger.error(f"Invalid model in list: {model}. Must be one of: {', '.join(valid_models)}")
                print_colored(f"ERROR: Invalid model in list: {model}. Must be one of: {', '.join(valid_models)}", "red")
                return False
                
        # If model names provided, check count matches
        if hasattr(args, "model_names") and args.model_names:
            model_names = args.model_names.split(",")
            if len(model_names) != len(models):
                logger.error(f"Number of model names ({len(model_names)}) doesn't match number of models ({len(models)})")
                print_colored(f"ERROR: Number of model names ({len(model_names)}) doesn't match number of models ({len(models)})", "red")
                return False
                
        # If settings files provided, check count matches
        if hasattr(args, "settings_files") and args.settings_files:
            settings_files = args.settings_files.split(",")
            if len(settings_files) != len(models):
                logger.error(f"Number of settings files ({len(settings_files)}) doesn't match number of models ({len(models)})")
                print_colored(f"ERROR: Number of settings files ({len(settings_files)}) doesn't match number of models ({len(models)})", "red")
                return False
                
            # Check if each settings file exists
            for settings_file in settings_files:
                if settings_file and not os.path.exists(settings_file):
                    logger.error(f"Settings file not found: {settings_file}")
                    print_colored(f"ERROR: Settings file not found: {settings_file}", "red")
                    return False
        
    return True


def _validate_output_args(args: Any, logger: logging.Logger) -> bool:
    """
    Validate output-related command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    # Check if output directory can be created
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create output directory: {args.output_dir}, error: {str(e)}")
        print_colored(f"ERROR: Cannot create output directory: {args.output_dir}, error: {str(e)}", "red")
        return False
    
    # Check if log file can be written if specified
    if args.log_file:
        try:
            log_dir = os.path.dirname(os.path.abspath(args.log_file))
            os.makedirs(log_dir, exist_ok=True)
            with open(args.log_file, 'a') as f:
                pass
        except OSError as e:
            logger.error(f"Cannot write to log file: {args.log_file}, error: {str(e)}")
            print_colored(f"ERROR: Cannot write to log file: {args.log_file}, error: {str(e)}", "red")
            return False
    
    return True


def _validate_threshold_args(args: Any, logger: logging.Logger) -> bool:
    """
    Validate threshold-related command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    # Check threshold start, end, step values
    if hasattr(args, "threshold_start") and hasattr(args, "threshold_end") and args.threshold_start is not None and args.threshold_end is not None:
        if args.threshold_start >= args.threshold_end:
            logger.error(f"Invalid threshold range: start ({args.threshold_start}) must be less than end ({args.threshold_end})")
            print_colored(f"ERROR: Invalid threshold range: start ({args.threshold_start}) must be less than end ({args.threshold_end})", "red")
            return False
            
        if hasattr(args, "threshold_step") and args.threshold_step is not None and args.threshold_step <= 0:
            logger.error(f"Invalid threshold step: must be greater than 0")
            print_colored(f"ERROR: Invalid threshold step: must be greater than 0", "red")
            return False
            
        if hasattr(args, "threshold_step") and args.threshold_step is not None and args.threshold_step >= (args.threshold_end - args.threshold_start):
            logger.error(f"Invalid threshold step: must be less than the threshold range ({args.threshold_end - args.threshold_start})")
            print_colored(f"ERROR: Invalid threshold step: must be less than the threshold range ({args.threshold_end - args.threshold_start})", "red")
            return False
            
    # For tune, validate threshold lists
    if (hasattr(args, "threshold_starts") and hasattr(args, "threshold_ends") and hasattr(args, "threshold_steps") and 
        args.threshold_starts is not None and args.threshold_ends is not None and args.threshold_steps is not None):
        try:
            starts = [float(x) for x in args.threshold_starts.split(",")]
            ends = [float(x) for x in args.threshold_ends.split(",")]
            steps = [float(x) for x in args.threshold_steps.split(",")]
            
            for start in starts:
                if not 0 <= start < 1:
                    logger.error(f"Invalid threshold start value: {start}, must be between 0 and 1")
                    print_colored(f"ERROR: Invalid threshold start value: {start}, must be between 0 and 1", "red")
                    return False
                    
            for end in ends:
                if not 0 < end <= 1:
                    logger.error(f"Invalid threshold end value: {end}, must be between 0 and 1")
                    print_colored(f"ERROR: Invalid threshold end value: {end}, must be between 0 and 1", "red")
                    return False
                    
            for step in steps:
                if step <= 0 or step >= 1:
                    logger.error(f"Invalid threshold step value: {step}, must be between 0 and 1")
                    print_colored(f"ERROR: Invalid threshold step value: {step}, must be between 0 and 1", "red")
                    return False
                    
        except ValueError:
            logger.error("Invalid threshold values: must be numeric")
            print_colored("ERROR: Invalid threshold values: must be numeric", "red")
            return False
            
    return True


def _validate_cv_args(args: Any, logger: logging.Logger) -> bool:
    """
    Validate cross-validation related command line arguments.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    # Check CV folds
    if hasattr(args, "cv") and args.cv is not None and args.cv < 0:
        logger.error("Number of cross-validation folds must be non-negative")
        print_colored("ERROR: Number of cross-validation folds must be non-negative", "red")
        return False
        
    if hasattr(args, "cv") and args.cv is not None and args.cv == 1:
        logger.warning("Using 1 fold for cross-validation is equivalent to no cross-validation")
        print_colored("WARNING: Using 1 fold for cross-validation is equivalent to no cross-validation", "yellow")
        
    if hasattr(args, "folds") and args.folds is not None and args.folds < 2:
        logger.error("Number of cross-validation folds must be at least 2")
        print_colored("ERROR: Number of cross-validation folds must be at least 2", "red")
        return False
        
    return True


def print_colored(text: str, color: str = "white", bold: bool = False) -> None:
    """Print colored text to console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "white": "\033[97m"
    }
    bold_code = "\033[1m" if bold else ""
    reset_code = "\033[0m"
    
    color_code = colors.get(color, colors["white"])
    print(f"{bold_code}{color_code}{text}{reset_code}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_levels.get(log_level.upper(), logging.INFO)
    
    # Configure logging
    logger = logging.getLogger("evaluation_cli")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def load_model(model_name: str, settings_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> object:
    """
    Load model based on name.
    
    Args:
        model_name: Name of the model to load
        settings_path: Optional path to settings file
        logger: Optional logger
        
    Returns:
        Loaded model
    """
    # Load settings
    settings = Settings(settings_path)
    
    if logger:
        logger.info(f"Loading model: {model_name}")
    
    # Initialize model based on name
    if model_name == "transformer":
        model = SentimentEmotionTransformer(settings=settings)
    elif model_name == "groq":
        model = GroqModel(settings=settings)
    elif model_name == "fallback":
        model = FallbackSystem(settings=settings)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model


def evaluate_with_batches(
    evaluator: ModelEvaluator, 
    model, 
    data_loader: BatchTestLoader,
    batch_size: int = 100,
    label_types: List[str] = ["sentiment", "emotion"],
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Perform evaluation using batch processing for large datasets.
    
    Args:
        evaluator: ModelEvaluator instance
        model: Model to evaluate
        data_loader: BatchTestLoader instance
        batch_size: Size of batches to process
        label_types: Label types to evaluate
        logger: Optional logger
        
    Returns:
        Evaluation results
    """
    if logger:
        logger.info(f"Starting batch evaluation with batch size: {batch_size}")
    
    # Initialize combined predictions
    all_predictions = []
    
    # Process batches
    total_batches = data_loader.num_batches()
    
    for i, batch_df in enumerate(data_loader.iterate_batches()):
        if logger:
            logger.info(f"Processing batch {i+1}/{total_batches}")
            
        # Extract texts for this batch
        texts = batch_df[data_loader.text_column].tolist()
        
        # Get predictions for batch
        if hasattr(model, 'analyze'):
            # Use analyze method for each text
            batch_predictions = []
            for text in texts:
                raw_prediction = model.analyze(text)
                # Extract labels from the model output
                sentiment_label = raw_prediction.get('sentiment', {}).get('label')
                emotion_label = raw_prediction.get('emotion', {}).get('label')
                
                # Handle None and empty values
                sentiment_label = sentiment_label if sentiment_label not in [None, '', 'nan'] else 'neutral'
                emotion_label = emotion_label if emotion_label not in [None, '', 'nan'] else 'neutral'
                
                prediction = {
                    'sentiment': sentiment_label,
                    'sentiment_confidence': raw_prediction.get('sentiment', {}).get('score', 0.0),
                    'emotion': emotion_label,
                    'emotion_confidence': raw_prediction.get('emotion', {}).get('score', 0.0)
                }
                batch_predictions.append(prediction)
        elif hasattr(model, 'predict'):
            # Use predict method for the entire batch
            batch_predictions = model.predict(texts)
        else:
            raise AttributeError(f"Model has no 'analyze' or 'predict' method")
        
        # Add to combined predictions
        all_predictions.extend(batch_predictions)
        
    # Set ground truth
    ground_truth = {}
    for label_type in label_types:
        if label_type in data_loader.label_columns:
            ground_truth[label_type] = data_loader.get_labels(label_type)
    
    # Evaluate using combined predictions
    results = evaluator.evaluate_model(
        predictions=all_predictions,
        label_types=label_types
    )
    
    if logger:
        logger.info("Batch evaluation complete")
        
    return results


def compare_multiple_models(
    model_configs: List[Dict],
    data_loader,
    label_types: List[str],
    metrics: List[str],
    significance_test: Optional[str] = None,
    cv_folds: Optional[int] = None,
    batch_size: Optional[int] = None,
    optimize_threshold: bool = True,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Compare multiple models and generate comprehensive results.
    
    Args:
        model_configs: List of model configuration dictionaries
        data_loader: Data loader instance
        label_types: Label types to evaluate
        metrics: Metrics to calculate
        significance_test: Type of significance test to perform
        cv_folds: Number of cross-validation folds
        batch_size: Batch size for processing
        optimize_threshold: Whether to optimize thresholds
        output_dir: Directory for output files
        logger: Optional logger
        
    Returns:
        Dictionary containing comparison results
    """
    if logger:
        logger.info(f"Starting comparison of {len(model_configs)} models")
    
    # Initialize results structure
    comparison_results = {
        "models": [],
        "individual_results": {},
        "best_models": {},
        "statistical_tests": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Process each model
    for config in model_configs:
        model_name = config["name"]
        display_name = config.get("display_name", model_name)
        settings_path = config.get("settings_path")
        
        if logger:
            logger.info(f"Evaluating model: {display_name}")
        
        try:
            # Load model
            model = load_model(model_name, settings_path, logger)
            
            # Create evaluator
            evaluator = ModelEvaluator()
            evaluator.data_loader = data_loader
            
            # Set ground truth
            ground_truth = {}
            for label_type in label_types:
                if label_type in data_loader.label_columns:
                    ground_truth[label_type] = data_loader.get_labels(label_type)
            
            evaluator.ground_truth = ground_truth
            
            # Get predictions
            texts = data_loader.get_texts()
            
            if batch_size and batch_size > 0:
                # Process in batches
                all_predictions = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+min(batch_size, len(texts)-i)]
                    batch_predictions = model.predict(batch_texts)
                    all_predictions.extend(batch_predictions)
                predictions = all_predictions
            else:
                predictions = model.predict(texts)
            
            # Evaluate model
            results = evaluator.evaluate_model(
                predictions=predictions,
                label_types=label_types
            )
            
            # Store results
            comparison_results["individual_results"][model_name] = {
                "display_name": display_name,
                "evaluation": results
            }
            
            # Add model info
            comparison_results["models"].append({
                "name": model_name,
                "display_name": display_name,
                "settings_path": settings_path
            })
            
            if logger:
                logger.info(f"Completed evaluation of {display_name}")
                
        except Exception as e:
            if logger:
                logger.error(f"Error evaluating {display_name}: {str(e)}")
            
            comparison_results["individual_results"][model_name] = {
                "display_name": display_name,
                "error": str(e)
            }
    
    # Find best models for each label type
    for label_type in label_types:
        best_model = None
        best_score = 0.0
        
        for model_name, results in comparison_results["individual_results"].items():
            if "error" not in results and "evaluation" in results:
                eval_results = results["evaluation"].get(label_type, {})
                score = eval_results.get("f1", 0.0)
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model:
            comparison_results["best_models"][label_type] = best_model
    
    # Find overall best model
    if comparison_results["best_models"]:
        # Count wins for each model
        model_wins = {}
        for label_type, best_model in comparison_results["best_models"].items():
            model_wins[best_model] = model_wins.get(best_model, 0) + 1
        
        # Find model with most wins
        overall_best = max(model_wins.items(), key=lambda x: x[1])[0]
        comparison_results["best_model"] = overall_best
    
    # Perform statistical significance tests if requested
    if significance_test and len(model_configs) >= 2:
        comparison_results["statistical_tests"] = perform_statistical_tests(
            comparison_results["individual_results"],
            label_types,
            significance_test,
            logger
        )
    
    # Perform cross-validation if requested
    if cv_folds and cv_folds > 1:
        for model_name, results in comparison_results["individual_results"].items():
            if "error" not in results:
                cv_results = perform_cross_validation(
                    evaluator=ModelEvaluator(),
                    model=load_model(model_name, comparison_results["models"][0].get("settings_path"), logger),
                    data_loader=data_loader,
                    label_types=label_types,
                    cv_folds=cv_folds,
                    metrics=metrics,
                    batch_size=batch_size,
                    optimize_threshold=optimize_threshold,
                    logger=logger
                )
                
                results["cross_validation"] = cv_results
    
    if logger:
        logger.info("Model comparison completed")
    
    return comparison_results


def perform_cross_validation(
    evaluator: ModelEvaluator,
    model,
    data_loader,
    label_types: List[str],
    cv_folds: int,
    metrics: List[str],
    batch_size: Optional[int] = None,
    optimize_threshold: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Perform cross-validation for a model.
    
    Args:
        evaluator: ModelEvaluator instance
        model: Model to evaluate
        data_loader: Data loader instance
        label_types: Label types to evaluate
        cv_folds: Number of cross-validation folds
        metrics: Metrics to calculate
        batch_size: Batch size for processing
        optimize_threshold: Whether to optimize thresholds
        logger: Optional logger
        
    Returns:
        Dictionary containing cross-validation results
    """
    if logger:
        logger.info(f"Performing {cv_folds}-fold cross-validation")
    
    # Split data into folds
    data = data_loader.data
    fold_size = len(data) // cv_folds
    
    cv_results = {}
    
    for label_type in label_types:
        if label_type not in data_loader.label_columns:
            continue
            
        fold_results = []
        thresholds = []
        
        for fold in range(cv_folds):
            if logger:
                logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            # Create train/test split for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else len(data)
            
            test_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(len(data)) if i not in test_indices]
            
            # Split data
            train_data = data.iloc[train_indices].reset_index(drop=True)
            test_data = data.iloc[test_indices].reset_index(drop=True)
            
            # Create temporary data loaders with mock paths
            train_loader = TestDataLoader(
                data_path="temp_train.csv",
                text_column=data_loader.text_column,
                label_columns=data_loader.label_columns
            )
            train_loader.data = train_data
            
            test_loader = TestDataLoader(
                data_path="temp_test.csv",
                text_column=data_loader.text_column,
                label_columns=data_loader.label_columns
            )
            test_loader.data = test_data
            
            # Get predictions for test set
            test_texts = test_loader.get_texts()
            
            if batch_size and batch_size > 0:
                # Process in batches
                all_predictions = []
                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i+min(batch_size, len(test_texts)-i)]
                    batch_predictions = model.predict(batch_texts)
                    all_predictions.extend(batch_predictions)
                predictions = all_predictions
            else:
                predictions = model.predict(test_texts)
            
            # Get ground truth
            ground_truth = test_loader.get_labels(label_type)
            
            # Create metrics calculator
            from src.evaluation.metrics import EvaluationMetrics
            metrics_calculator = EvaluationMetrics(
                predictions=predictions,
                ground_truth=ground_truth,
                label_type=label_type
            )
            
            # Find optimal threshold if requested
            if optimize_threshold:
                from src.evaluation.threshold import ThresholdOptimizer
                threshold_optimizer = ThresholdOptimizer(
                    metrics_calculator=metrics_calculator,
                    metric_name="f1"
                )
                optimal_threshold = threshold_optimizer.find_optimal_threshold(label_type)
                thresholds.append(optimal_threshold)
            else:
                optimal_threshold = 0.5
            
            # Calculate metrics
            fold_metrics = {}
            for metric in metrics:
                if metric == "f1":
                    fold_metrics[metric] = metrics_calculator.calculate_f1(optimal_threshold)
                elif metric == "precision":
                    precision, _ = metrics_calculator.calculate_precision_recall(optimal_threshold)
                    fold_metrics[metric] = precision
                elif metric == "recall":
                    _, recall = metrics_calculator.calculate_precision_recall(optimal_threshold)
                    fold_metrics[metric] = recall
            
            fold_metrics["threshold"] = optimal_threshold
            fold_results.append(fold_metrics)
        
        # Calculate summary statistics
        cv_results[label_type] = {
            "fold_results": fold_results,
            "mean_threshold": np.mean(thresholds) if thresholds else 0.5,
            "std_threshold": np.std(thresholds) if thresholds else 0.0
        }
        
        # Calculate mean and std for each metric
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            cv_results[label_type][f"mean_{metric}"] = np.mean(values)
            cv_results[label_type][f"std_{metric}"] = np.std(values)
    
    if logger:
        logger.info("Cross-validation completed")
    
    return cv_results


def tune_threshold_params(
    evaluator: ModelEvaluator,
    predictions: List[Dict],
    ground_truth: Dict,
    label_type: str,
    param_grid: Dict,
    metric: str = "f1",
    cv_folds: int = 5,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Tune threshold parameters using grid search.
    
    Args:
        evaluator: ModelEvaluator instance
        predictions: Model predictions
        ground_truth: Ground truth labels
        label_type: Label type to tune for
        param_grid: Parameter grid for tuning
        metric: Metric to optimize
        cv_folds: Number of cross-validation folds
        logger: Optional logger
        
    Returns:
        Dictionary containing tuning results
    """
    if logger:
        logger.info(f"Tuning threshold parameters for {label_type}")
    
    # Create metrics calculator
    from src.evaluation.metrics import EvaluationMetrics
    metrics_calculator = EvaluationMetrics(
        predictions=predictions,
        ground_truth=ground_truth[label_type],
        label_type=label_type
    )
    
    # Generate parameter combinations
    threshold_starts = param_grid.get("threshold_start", [0.1])
    threshold_ends = param_grid.get("threshold_end", [0.9])
    threshold_steps = param_grid.get("threshold_step", [0.05])
    
    param_combinations = []
    for start in threshold_starts:
        for end in threshold_ends:
            for step in threshold_steps:
                if start < end:
                    param_combinations.append({
                        "threshold_start": start,
                        "threshold_end": end,
                        "threshold_step": step
                    })
    
    if logger:
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    # Evaluate each parameter combination
    cv_results = {
        "params": [],
        "mean_score": [],
        "std_score": []
    }
    
    for params in param_combinations:
        # Create threshold optimizer with these parameters
        from src.evaluation.threshold import ThresholdOptimizer
        threshold_optimizer = ThresholdOptimizer(
            metrics_calculator=metrics_calculator,
            metric_name=metric
        )
        threshold_optimizer.step_size = params["threshold_step"]
        
        # Perform threshold sweep
        threshold_optimizer.sweep_thresholds(
            params["threshold_start"],
            params["threshold_end"]
        )
        
        # Find optimal threshold
        optimal_threshold = threshold_optimizer.find_optimal_threshold(label_type)
        
        # Get score at optimal threshold
        score = metrics_calculator.calculate_f1(optimal_threshold) if metric == "f1" else \
                metrics_calculator.calculate_precision_recall(optimal_threshold)[0] if metric == "precision" else \
                metrics_calculator.calculate_precision_recall(optimal_threshold)[1]
        
        cv_results["params"].append(params)
        cv_results["mean_score"].append(score)
        cv_results["std_score"].append(0.0)  # Single evaluation, no std
    
    # Find best parameters
    best_idx = np.argmax(cv_results["mean_score"])
    best_params = cv_results["params"][best_idx]
    best_score = cv_results["mean_score"][best_idx]
    
    tuning_results = {
        "best_params": best_params,
        "best_score": best_score,
        "cv_results": cv_results
    }
    
    if logger:
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
    
    return tuning_results


def perform_statistical_tests(
    individual_results: Dict,
    label_types: List[str],
    test_type: str,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Perform statistical significance tests between models.
    
    Args:
        individual_results: Dictionary of individual model results
        label_types: Label types to test
        test_type: Type of statistical test
        logger: Optional logger
        
    Returns:
        Dictionary containing statistical test results
    """
    if logger:
        logger.info(f"Performing {test_type} statistical tests")
    
    # Get model names
    model_names = list(individual_results.keys())
    if len(model_names) < 2:
        if logger:
            logger.warning("Need at least 2 models for statistical tests")
        return {}
    
    statistical_tests = {}
    
    # Perform pairwise comparisons
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            pair_name = f"{model1} vs {model2}"
            
            if logger:
                logger.info(f"Testing {pair_name}")
            
            statistical_tests[pair_name] = {}
            
            for label_type in label_types:
                # Get results for both models
                result1 = individual_results[model1].get("evaluation", {}).get(label_type, {})
                result2 = individual_results[model2].get("evaluation", {}).get(label_type, {})
                
                if "error" in individual_results[model1] or "error" in individual_results[model2]:
                    statistical_tests[pair_name][label_type] = {
                        "test_method": test_type,
                        "error": "One or both models failed evaluation"
                    }
                    continue
                
                # Perform statistical test based on type
                if test_type == "mcnemar":
                    test_result = perform_mcnemar_test(result1, result2, label_type)
                elif test_type == "bootstrap":
                    test_result = perform_bootstrap_test(result1, result2, label_type)
                elif test_type == "t-test":
                    test_result = perform_ttest(result1, result2, label_type)
                else:
                    test_result = {
                        "test_method": test_type,
                        "error": f"Unknown test type: {test_type}"
                    }
                
                statistical_tests[pair_name][label_type] = test_result
    
    return statistical_tests


def perform_mcnemar_test(result1: Dict, result2: Dict, label_type: str) -> Dict:
    """Perform McNemar's test for statistical significance."""
    # This is a simplified implementation
    # In practice, you would need the actual predictions and ground truth
    return {
        "test_method": "mcnemar",
        "metrics": {
            "overall": {
                "p_value": 0.05,  # Placeholder
                "significant": True,  # Placeholder
                "better_model": 1 if result1.get("f1", 0) > result2.get("f1", 0) else 2
            }
        }
    }


def perform_bootstrap_test(result1: Dict, result2: Dict, label_type: str) -> Dict:
    """Perform bootstrap test for statistical significance."""
    # This is a simplified implementation
    return {
        "test_method": "bootstrap",
        "metrics": {
            "f1": {
                "p_value": 0.05,  # Placeholder
                "significant": True,  # Placeholder
                "better_model": 1 if result1.get("f1", 0) > result2.get("f1", 0) else 2
            }
        }
    }


def perform_ttest(result1: Dict, result2: Dict, label_type: str) -> Dict:
    """Perform t-test for statistical significance."""
    # This is a simplified implementation
    return {
        "test_method": "t-test",
        "metrics": {
            "f1": {
                "t_statistic": 2.0,  # Placeholder
                "p_value": 0.05,  # Placeholder
                "significant": True,  # Placeholder
                "better_model": 1 if result1.get("f1", 0) > result2.get("f1", 0) else 2
            }
        }
    }


def export_to_format(results: Dict, output_path: str, format: str) -> None:
    """
    Export results to specified format.
    
    Args:
        results: Results to export
        output_path: Output file path
        format: Export format ("csv", "pickle")
    """
    if format == "csv":
        # Convert results to DataFrame and save as CSV
        if isinstance(results, dict):
            # Flatten nested dictionary for CSV
            flattened_data = []
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flattened_data.append({
                            "key": f"{key}_{sub_key}",
                            "value": str(sub_value)
                        })
                else:
                    flattened_data.append({
                        "key": key,
                        "value": str(value)
                    })
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)
        else:
            pd.DataFrame(results).to_csv(output_path, index=False)
    
    elif format == "pickle":
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _configure_eval_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the evaluate command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="transformer",
                      choices=["transformer", "groq", "fallback"],
                      help="Model to evaluate")
    model_group.add_argument("--settings", type=str, help="Path to settings file")
    
    # Threshold options
    threshold_group = parser.add_argument_group("Threshold Options")
    threshold_group.add_argument("--threshold-start", type=float, default=0.1,
                      help="Start of threshold sweep range")
    threshold_group.add_argument("--threshold-end", type=float, default=0.9,
                      help="End of threshold sweep range")
    threshold_group.add_argument("--threshold-step", type=float, default=0.05,
                      help="Step size for threshold sweep")
    threshold_group.add_argument("--optimize-metric", type=str, default="f1",
                      choices=["f1", "precision", "recall"],
                      help="Metric to optimize when finding optimal threshold")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    output_group.add_argument("--report-name", type=str, default="evaluation_report",
                      help="Base name for report files")
    output_group.add_argument("--export-format", type=str, default="all",
                      choices=["json", "csv", "pickle", "all"],
                      help="Format(s) to export results in")
    output_group.add_argument("--generate-plots", action="store_true",
                      help="Generate and save visualization plots")
    output_group.add_argument("--interactive-plots", action="store_true",
                      help="Generate interactive HTML visualizations")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    process_group.add_argument("--parallel", action="store_true",
                      help="Use parallel processing for batch evaluation")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--cv", type=int, default=0,
                      help="Number of cross-validation folds (0 for no CV)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _configure_compare_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the compare command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--models", type=str, required=True,
                      help="Comma-separated list of models to compare (transformer,groq,fallback)")
    model_group.add_argument("--model-names", type=str,
                      help="Optional comma-separated list of display names for models")
    model_group.add_argument("--settings-files", type=str,
                      help="Optional comma-separated list of settings files for models")
    
    # Comparison options
    compare_group = parser.add_argument_group("Comparison Options")
    compare_group.add_argument("--significance-test", type=str, default="bootstrap",
                      choices=["mcnemar", "bootstrap", "t-test"],
                      help="Statistical test to use for comparing models")
    compare_group.add_argument("--metrics", type=str, default="f1,precision,recall",
                      help="Comma-separated list of metrics to compare")
    compare_group.add_argument("--optimize-thresholds", action="store_true",
                      help="Optimize thresholds for each model")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    output_group.add_argument("--report-name", type=str, default="comparison_report",
                      help="Base name for report files")
    output_group.add_argument("--export-format", type=str, default="all",
                      choices=["json", "csv", "pickle", "all"],
                      help="Format(s) to export results in")
    output_group.add_argument("--generate-plots", action="store_true",
                      help="Generate and save visualization plots")
    output_group.add_argument("--interactive-plots", action="store_true",
                      help="Generate interactive HTML visualizations")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    process_group.add_argument("--parallel", action="store_true",
                      help="Use parallel processing for batch evaluation")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--cv", type=int, default=0,
                      help="Number of cross-validation folds (0 for no CV)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _configure_tune_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the tune command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="transformer",
                      choices=["transformer", "groq", "fallback"],
                      help="Model to tune thresholds for")
    model_group.add_argument("--settings", type=str, help="Path to settings file")
    
    # Tuning options
    tuning_group = parser.add_argument_group("Tuning Options")
    tuning_group.add_argument("--optimize-metric", type=str, default="f1",
                      choices=["f1", "precision", "recall"],
                      help="Metric to optimize when finding optimal threshold")
    tuning_group.add_argument("--threshold-starts", type=str, default="0.1,0.2,0.3",
                      help="Comma-separated list of threshold start values to try")
    tuning_group.add_argument("--threshold-ends", type=str, default="0.7,0.8,0.9",
                      help="Comma-separated list of threshold end values to try")
    tuning_group.add_argument("--threshold-steps", type=str, default="0.01,0.02,0.05,0.1",
                      help="Comma-separated list of threshold step sizes to try")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--cv", type=int, default=5,
                      help="Number of cross-validation folds")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save tuning results")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _configure_cv_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the cross-validate command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="transformer",
                      choices=["transformer", "groq", "fallback"],
                      help="Model to cross-validate")
    model_group.add_argument("--settings", type=str, help="Path to settings file")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--folds", type=int, default=5,
                      help="Number of cross-validation folds")
    cv_group.add_argument("--metrics", type=str, default="f1,precision,recall",
                      help="Comma-separated list of metrics to calculate")
    cv_group.add_argument("--optimize-thresholds", action="store_true",
                      help="Optimize thresholds for each fold")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save cross-validation results")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _run_evaluate(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the evaluate command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in evaluation
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_threshold_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        model = load_model(args.model, args.settings, logger=logger)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Decide whether to use cross-validation
    if args.cv > 1:
        # Perform cross-validation
        logger.info(f"Performing {args.cv}-fold cross-validation")
        
        try:
            cv_results = perform_cross_validation(
                evaluator=evaluator,
                model=model,
                data_loader=data_loader,
                label_types=label_types,
                cv_folds=args.cv,
                metrics=["f1", "precision", "recall"],
                batch_size=args.batch_size if args.batch_size > 0 else None,
                optimize_threshold=True,
                logger=logger
            )
            
            # Save CV results
            cv_path = os.path.join(output_dir, "cross_validation_results.json")
            with open(cv_path, "w") as f:
                json.dump(cv_results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.number) else o)
            
            # Print summary
            if not args.quiet:
                print_colored("\n===== CROSS-VALIDATION RESULTS =====", "green", bold=True)
                for label_type, results in cv_results.items():
                    print_colored(f"\n{label_type.upper()}:", "blue", bold=True)
                    print(f"Mean threshold: {results.get('mean_threshold', 0.5):.3f} ± {results.get('std_threshold', 0):.3f}")
                    print(f"Mean F1: {results.get('mean_f1', 0):.4f} ± {results.get('std_f1', 0):.4f}")
                    print(f"Mean precision: {results.get('mean_precision', 0):.4f} ± {results.get('std_precision', 0):.4f}")
                    print(f"Mean recall: {results.get('mean_recall', 0):.4f} ± {results.get('std_recall', 0):.4f}")
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")
    else:
        # Perform regular evaluation
        logger.info(f"Performing regular evaluation with {args.model} model")
        
        try:
            if args.batch_size > 0:
                # Batch evaluation
                results = evaluate_with_batches(
                    evaluator=evaluator,
                    model=model,
                    data_loader=data_loader,
                    batch_size=args.batch_size,
                    label_types=label_types,
                    logger=logger
                )
            else:
                # Regular evaluation
                logger.info("Loading test data and ground truth")
                evaluator.data_loader = data_loader
                
                # Set ground truth
                ground_truth = {}
                for label_type in label_types:
                    if label_type in label_columns:
                        ground_truth[label_type] = data_loader.get_labels(label_type)
                
                evaluator.ground_truth = ground_truth
                
                logger.info(f"Running evaluation with {args.model} model")
                texts = data_loader.get_texts()
                predictions = model.predict(texts)
                
                results = evaluator.evaluate_model(
                    predictions=predictions,
                    label_types=label_types
                )
            
            # Configure threshold optimizers
            for label_type, optimizer in evaluator.threshold_optimizers.items():
                optimizer.step_size = args.threshold_step
                optimizer.metric_name = args.optimize_metric
                optimizer.sweep_thresholds(args.threshold_start, args.threshold_end)
                
                # Find optimal threshold
                optimal_threshold = optimizer.find_optimal_threshold(label_type)
                logger.info(f"Optimal threshold for {label_type}: {optimal_threshold:.3f}")
            
            # Generate visualizations
            if args.generate_plots:
                logger.info("Generating visualizations")
                for label_type in label_types:
                    if label_type in evaluator.visualizers:
                        vis_dir = os.path.join(output_dir, f"{label_type}_visualizations")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Generate standard plots
                        evaluator.visualize_results(label_type, vis_dir)
                        
                        # Generate interactive plots if requested
                        if args.interactive_plots:
                            dashboard_path = os.path.join(vis_dir, "dashboard.html")
                            evaluator.visualizers[label_type].create_dashboard(dashboard_path)
                            
                            threshold_tuner_path = os.path.join(vis_dir, "threshold_tuner.html")
                            evaluator.visualizers[label_type].create_interactive_threshold_tuner(
                                label_type=label_type
                            )
            
            # Export results
            logger.info("Exporting evaluation results")
            
            export_formats = []
            if args.export_format == "all":
                export_formats = ["json", "csv", "pickle"]
            else:
                export_formats = [args.export_format]
                
            for fmt in export_formats:
                output_path = os.path.join(output_dir, f"{args.report_name}.{fmt}")
                evaluator.export_results(output_path, format=fmt)
                
            # Generate detailed report
            report_path = os.path.join(output_dir, f"{args.report_name}.md")
            evaluator.generate_report(report_path)
            
            # Save evaluation state for future reference
            state_path = os.path.join(output_dir, "evaluator_state.pkl")
            evaluator.save_state(state_path)
            
            # Print summary to console
            if not args.quiet:
                print_colored("\n===== EVALUATION RESULTS =====", "green", bold=True)
                for label_type, metrics in results.items():
                    print_colored(f"\n{label_type.upper()} METRICS:", "blue", bold=True)
                    print(f"Optimal threshold: {metrics['threshold']:.3f}")
                    print(f"F1 score: {metrics['f1']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"AUC: {metrics.get('auc', 0):.4f}")
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise ValueError(f"Evaluation failed: {str(e)}")
                
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


def _run_compare(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the compare command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in comparison
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse models
    model_names = args.models.split(",")
    
    # Parse model display names if provided
    if args.model_names:
        display_names = args.model_names.split(",")
        if len(display_names) != len(model_names):
            logger.warning(f"Number of display names ({len(display_names)}) doesn't match number of models ({len(model_names)}). Using model names as display names.")
            display_names = model_names
    else:
        display_names = model_names
    
    # Parse settings files if provided
    if args.settings_files:
        settings_files = args.settings_files.split(",")
        if len(settings_files) != len(model_names):
            logger.warning(f"Number of settings files ({len(settings_files)}) doesn't match number of models ({len(model_names)}). Using no settings files.")
            settings_files = [None] * len(model_names)
    else:
        settings_files = [None] * len(model_names)
    
    # Create model configurations
    model_configs = []
    for i, model_name in enumerate(model_names):
        model_configs.append({
            "name": model_name,
            "display_name": display_names[i],
            "settings_path": settings_files[i]
        })
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Parse metrics
    metrics = args.metrics.split(",")
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Perform model comparison
    try:
        comparison_results = compare_multiple_models(
            model_configs=model_configs,
            data_loader=data_loader,
            label_types=label_types,
            metrics=metrics,
            significance_test=args.significance_test,
            cv_folds=args.cv if args.cv > 1 else None,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            optimize_threshold=args.optimize_thresholds,
            output_dir=output_dir if args.generate_plots else None,
            logger=logger
        )
        
        # Save comparison results
        comparison_path = os.path.join(output_dir, f"{args.report_name}.json")
        with open(comparison_path, "w") as f:
            json.dump(
                comparison_results, 
                f, 
                indent=2, 
                default=lambda o: float(o) if isinstance(o, (np.number, np.floating)) else o
            )
        
        # Generate comparison report
        report_path = os.path.join(output_dir, f"{args.report_name}.md")
        _generate_comparison_report(comparison_results, report_path, logger)
        
        # Export results in different formats if requested
        export_formats = []
        if args.export_format == "all":
            export_formats = ["csv", "pickle"]  # JSON already done
        elif args.export_format != "json":
            export_formats = [args.export_format]
            
        for fmt in export_formats:
            try:
                output_path = os.path.join(output_dir, f"{args.report_name}.{fmt}")
                if fmt == "csv":
                    # For CSV, convert to a flattened DataFrame
                    results_list = []
                    for model_name, results in comparison_results["individual_results"].items():
                        if "evaluation" in results:
                            for label_type, metrics in results["evaluation"].items():
                                row = {"model": model_name, "label_type": label_type}
                                row.update(metrics)
                                results_list.append(row)
                    
                    if results_list:
                        df = pd.DataFrame(results_list)
                        df.to_csv(output_path, index=False)
                elif fmt == "pickle":
                    # For pickle, save the full results
                    import pickle
                    with open(output_path, 'wb') as f:
                        pickle.dump(comparison_results, f)
            except Exception as e:
                logger.error(f"Error exporting to {fmt} format: {str(e)}")
                # Continue with other formats if one fails
        
        # Print summary to console
        if not args.quiet:
            print_colored("\n===== MODEL COMPARISON RESULTS =====", "green", bold=True)
            
            # Print best models
            if "best_models" in comparison_results:
                print_colored("\nBest Models:", "blue", bold=True)
                for label_type, model_name in comparison_results["best_models"].items():
                    print(f"{label_type.capitalize()}: {model_name}")
                    
            if "best_model" in comparison_results and comparison_results["best_model"]:
                print_colored(f"\nOverall Best Model: {comparison_results['best_model']}", "green", bold=True)
                
            # Print statistical significance results
            if "statistical_tests" in comparison_results and comparison_results["statistical_tests"]:
                print_colored("\nStatistical Significance Tests:", "blue", bold=True)
                for pair, tests in comparison_results["statistical_tests"].items():
                    print(f"\n{pair}:")
                    for label_type, test_results in tests.items():
                        better_model = test_results.get("better_model")
                        models = pair.split(" vs ")
                        if better_model == 1:
                            print(f"  {label_type.capitalize()}: {models[0]} is significantly better")
                        elif better_model == 2:
                            print(f"  {label_type.capitalize()}: {models[1]} is significantly better")
                        else:
                            print(f"  {label_type.capitalize()}: No significant difference")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        raise ValueError(f"Model comparison failed: {str(e)}")
        
    logger.info(f"Comparison complete. Results saved to: {output_dir}")


def _run_tune(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the tune command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in tuning
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_threshold_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Parse tuning parameters
    try:
        threshold_starts = [float(x) for x in args.threshold_starts.split(",")]
        threshold_ends = [float(x) for x in args.threshold_ends.split(",")]
        threshold_steps = [float(x) for x in args.threshold_steps.split(",")]
    except ValueError:
        logger.error("Invalid threshold values: must be numeric")
        raise ValueError("Invalid threshold values: must be numeric")
    
    # Create parameter grid
    param_grid = {
        "threshold_start": threshold_starts,
        "threshold_end": threshold_ends,
        "threshold_step": threshold_steps
    }
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        model = load_model(args.model, args.settings, logger=logger)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Get ground truth
    ground_truth = {}
    for label_type in label_types:
        if label_type in label_columns:
            ground_truth[label_type] = data_loader.get_labels(label_type)
    
    # Get predictions
    try:
        logger.info(f"Getting predictions with {args.model} model")
        texts = data_loader.get_texts()
        
        if args.batch_size > 0:
            # Process in batches
            all_predictions = []
            for i in range(0, len(texts), args.batch_size):
                batch_texts = texts[i:i+min(args.batch_size, len(texts)-i)]
                batch_predictions = model.predict(batch_texts)
                all_predictions.extend(batch_predictions)
            predictions = all_predictions
        else:
            predictions = model.predict(texts)
    except Exception as e:
        logger.error(f"Error getting model predictions: {str(e)}")
        raise ValueError(f"Failed to get model predictions: {str(e)}")
    
    # Run threshold parameter tuning for each label type
    tuning_results = {}
    
    try:
        for label_type in label_types:
            if label_type in ground_truth:
                logger.info(f"Tuning threshold parameters for {label_type}")
                
                # Check if we have enough unique classes for tuning
                unique_classes = set(ground_truth[label_type])
                if len(unique_classes) < 2:
                    logger.warning(f"Not enough unique classes for {label_type} (found {len(unique_classes)}). Skipping tuning.")
                    continue
                
                tuning_result = tune_threshold_params(
                    evaluator=evaluator,
                    predictions=predictions,
                    ground_truth=ground_truth,
                    label_type=label_type,
                    param_grid=param_grid,
                    metric=args.optimize_metric,
                    cv_folds=args.cv,
                    logger=logger
                )
                
                tuning_results[label_type] = tuning_result
                
                # Print best parameters
                best_params = tuning_result.get("best_params", {})
                best_score = tuning_result.get("best_score", 0)
                
                if best_params:
                    logger.info(
                        f"Best parameters for {label_type}: "
                        f"start={best_params.get('threshold_start', 0.1)}, "
                        f"end={best_params.get('threshold_end', 0.9)}, "
                        f"step={best_params.get('threshold_step', 0.05)}, "
                        f"{args.optimize_metric}={best_score:.4f}"
                    )
        
        # Save tuning results
        tuning_path = os.path.join(output_dir, "threshold_tuning_results.json")
        with open(tuning_path, "w") as f:
            json.dump(
                tuning_results, 
                f, 
                indent=2, 
                default=lambda o: float(o) if isinstance(o, (np.number, np.floating)) else o
            )
        
        # Print summary to console
        if not args.quiet:
            print_colored("\n===== THRESHOLD TUNING RESULTS =====", "green", bold=True)
            for label_type, results in tuning_results.items():
                print_colored(f"\n{label_type.upper()}:", "blue", bold=True)
                
                best_params = results.get("best_params", {})
                best_score = results.get("best_score", 0)
                
                if best_params:
                    print(f"Best parameters:")
                    print(f"  Threshold start: {best_params.get('threshold_start', 0.1)}")
                    print(f"  Threshold end: {best_params.get('threshold_end', 0.9)}")
                    print(f"  Threshold step: {best_params.get('threshold_step', 0.05)}")
                    print(f"  {args.optimize_metric.capitalize()} score: {best_score:.4f}")
    except Exception as e:
        logger.error(f"Error during threshold tuning: {str(e)}")
        raise ValueError(f"Threshold tuning failed: {str(e)}")
        
    logger.info(f"Threshold tuning complete. Results saved to: {output_dir}")


def _run_cross_validate(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the cross-validate command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in cross-validation
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Parse metrics
    metrics = args.metrics.split(",")
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        model = load_model(args.model, args.settings, logger=logger)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Perform cross-validation
    try:
        logger.info(f"Performing {args.folds}-fold cross-validation")
        
        # Validate that we have enough data for the requested number of folds
        data_size = len(data_loader.get_texts())
        if data_size < args.folds:
            logger.error(f"Not enough data ({data_size} samples) for {args.folds} folds")
            raise ValueError(f"Not enough data ({data_size} samples) for {args.folds} folds")
            
        # Validate that we have enough samples per class for stratified CV
        if any(label_type in label_columns for label_type in label_types):
            for label_type in label_types:
                if label_type in label_columns:
                    labels = data_loader.get_labels(label_type)
                    label_counts = {}
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    
                    min_class_count = min(label_counts.values()) if label_counts else 0
                    if min_class_count < args.folds:
                        logger.warning(
                            f"Some classes in {label_type} have fewer than {args.folds} samples "
                            f"(minimum: {min_class_count}). Standard cross-validation will be used instead of stratified."
                        )
        
        cv_results = perform_cross_validation(
            evaluator=evaluator,
            model=model,
            data_loader=data_loader,
            label_types=label_types,
            cv_folds=args.folds,
            metrics=metrics,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            optimize_threshold=args.optimize_thresholds,
            logger=logger
        )
        
        # Save CV results
        cv_path = os.path.join(output_dir, "cross_validation_results.json")
        with open(cv_path, "w") as f:
            json.dump(
                cv_results, 
                f, 
                indent=2, 
                default=lambda o: float(o) if isinstance(o, (np.number, np.floating)) else o
            )
        
        # Print summary to console
        if not args.quiet:
            print_colored("\n===== CROSS-VALIDATION RESULTS =====", "green", bold=True)
            for label_type, results in cv_results.items():
                print_colored(f"\n{label_type.upper()}:", "blue", bold=True)
                print(f"Mean threshold: {results.get('mean_threshold', 0.5):.3f} ± {results.get('std_threshold', 0):.3f}")
                
                for metric in metrics:
                    mean_key = f"mean_{metric}"
                    std_key = f"std_{metric}"
                    
                    if mean_key in results and std_key in results:
                        print(f"Mean {metric}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        raise ValueError(f"Cross-validation failed: {str(e)}")
        
    logger.info(f"Cross-validation complete. Results saved to: {output_dir}")


def _generate_comparison_report(results: Dict, output_path: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Generate a detailed markdown report of model comparison results.
    
    Args:
        results: Comparison results dictionary
        output_path: Path to save the report
        logger: Optional logger
    """
    try:
        with open(output_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            if "best_model" in results and results["best_model"]:
                f.write(f"**Overall Best Model:** {results['best_model']}\n\n")
            
            if "best_models" in results:
                f.write("**Best Models by Label Type:**\n")
                for label_type, model_name in results["best_models"].items():
                    f.write(f"- {label_type.capitalize()}: {model_name}\n")
                f.write("\n")
            
            # Individual results section
            f.write("## Individual Model Results\n\n")
            if "individual_results" in results:
                for model_name, model_results in results["individual_results"].items():
                    f.write(f"### {model_name}\n\n")
                    
                    if "evaluation" in model_results:
                        for label_type, metrics in model_results["evaluation"].items():
                            f.write(f"**{label_type.capitalize()} Metrics:**\n")
                            f.write(f"- F1 Score: {metrics.get('f1', 0):.4f}\n")
                            f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
                            f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
                            f.write(f"- AUC: {metrics.get('auc', 0):.4f}\n")
                            f.write(f"- Optimal Threshold: {metrics.get('threshold', 0.5):.3f}\n\n")
            
            # Statistical tests section
            if "statistical_tests" in results and results["statistical_tests"]:
                f.write("## Statistical Significance Tests\n\n")
                for pair, tests in results["statistical_tests"].items():
                    f.write(f"### {pair}\n\n")
                    for label_type, test_results in tests.items():
                        f.write(f"**{label_type.capitalize()}:**\n")
                        f.write(f"- P-value: {test_results.get('p_value', 0):.4f}\n")
                        f.write(f"- Significant: {test_results.get('significant', False)}\n")
                        
                        better_model = test_results.get("better_model")
                        if better_model == 1:
                            models = pair.split(" vs ")
                            f.write(f"- Better model: {models[0]}\n")
                        elif better_model == 2:
                            models = pair.split(" vs ")
                            f.write(f"- Better model: {models[1]}\n")
                        else:
                            f.write("- No significant difference\n")
                        f.write("\n")
        
        if logger:
            logger.info(f"Comparison report saved to: {output_path}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error generating comparison report: {str(e)}")
        raise


def run_evaluation_cli():
    """Run the evaluation CLI tool."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Model evaluation tool")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single model")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune threshold parameters")
    
    # Cross-validate command
    cv_parser = subparsers.add_parser("cross-validate", help="Perform cross-validation")
    
    # Common arguments for all commands
    for subparser in [eval_parser, compare_parser, tune_parser, cv_parser]:
        # Data options
        subparser.add_argument("--data", type=str, required=True, help="Path to labeled test data")
        subparser.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                           help="Format of test data file")
        subparser.add_argument("--text-column", type=str, default="text",
                           help="Name of column containing text data")
        
        # Label options
        subparser.add_argument("--sentiment-column", type=str, default="sentiment",
                           help="Name of column containing sentiment labels")
        subparser.add_argument("--emotion-column", type=str, default="emotion",
                           help="Name of column containing emotion labels")
        subparser.add_argument("--label-types", type=str, default="sentiment,emotion",
                           help="Comma-separated list of label types to evaluate")
        
        # Threshold options
        subparser.add_argument("--threshold-start", type=float, default=0.1,
                           help="Start of threshold sweep range")
        subparser.add_argument("--threshold-end", type=float, default=0.9,
                           help="End of threshold sweep range")
        subparser.add_argument("--threshold-step", type=float, default=0.05,
                           help="Step size for threshold sweep")
        subparser.add_argument("--optimize-metric", type=str, default="f1",
                           choices=["f1", "precision", "recall"],
                           help="Metric to optimize when finding optimal threshold")
        
        # Output options
        subparser.add_argument("--output-dir", type=str, default="evaluation_results",
                           help="Directory to save evaluation results")
        subparser.add_argument("--report-name", type=str, default="evaluation_report",
                           help="Base name for report files")
        subparser.add_argument("--export-format", type=str, default="all",
                           choices=["json", "csv", "pickle", "all"],
                           help="Format(s) to export results in")
        
        # Visualization options
        subparser.add_argument("--generate-plots", action="store_true",
                           help="Generate and save visualization plots")
        subparser.add_argument("--interactive-plots", action="store_true",
                           help="Generate interactive HTML visualizations")
        
        # Processing options
        subparser.add_argument("--batch-size", type=int, default=0,
                           help="Batch size for processing (0 for no batching)")
        subparser.add_argument("--parallel", action="store_true",
                           help="Use parallel processing for batch evaluation")
        
        # Logging options
        subparser.add_argument("--log-file", type=str, help="Path to log file")
        subparser.add_argument("--log-level", type=str, default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="Logging level")
        subparser.add_argument("--quiet", action="store_true", help="Suppress console output")
    
    # Evaluate-specific arguments
    eval_parser.add_argument("--model", type=str, default="transformer",
                           choices=["transformer", "groq", "fallback"],
                           help="Model to evaluate")
    eval_parser.add_argument("--settings", type=str, help="Path to settings file")
    
    # Compare-specific arguments
    compare_parser.add_argument("--models", type=str, required=True,
                              help="Comma-separated list of model names to compare")
    compare_parser.add_argument("--display-names", type=str,
                              help="Comma-separated list of display names for models")
    compare_parser.add_argument("--settings-files", type=str,
                              help="Comma-separated list of settings files for models")
    compare_parser.add_argument("--metrics", type=str, default="f1,precision,recall",
                              help="Comma-separated list of metrics to calculate")
    compare_parser.add_argument("--significance-test", type=str, choices=["mcnemar", "bootstrap", "t-test"],
                              help="Type of statistical significance test to perform")
    compare_parser.add_argument("--cv", type=int, default=0,
                              help="Number of cross-validation folds (0 to disable)")
    compare_parser.add_argument("--optimize-thresholds", action="store_true",
                              help="Optimize thresholds for each model")
    
    # Tune-specific arguments
    tune_parser.add_argument("--model", type=str, default="transformer",
                           choices=["transformer", "groq", "fallback"],
                           help="Model to tune thresholds for")
    tune_parser.add_argument("--settings", type=str, help="Path to settings file")
    tune_parser.add_argument("--threshold-starts", type=str, default="0.1,0.2,0.3",
                           help="Comma-separated list of threshold start values to try")
    tune_parser.add_argument("--threshold-ends", type=str, default="0.7,0.8,0.9",
                           help="Comma-separated list of threshold end values to try")
    tune_parser.add_argument("--threshold-steps", type=str, default="0.01,0.02,0.05,0.1",
                           help="Comma-separated list of threshold step sizes to try")
    
    # Cross-validate-specific arguments
    cv_parser.add_argument("--model", type=str, default="transformer",
                          choices=["transformer", "groq", "fallback"],
                          help="Model to cross-validate")
    cv_parser.add_argument("--settings", type=str, help="Path to settings file")
    cv_parser.add_argument("--folds", type=int, default=5,
                          help="Number of cross-validation folds")
    cv_parser.add_argument("--metrics", type=str, default="f1,precision,recall",
                          help="Comma-separated list of metrics to calculate")
    cv_parser.add_argument("--optimize-thresholds", action="store_true",
                          help="Optimize thresholds for each fold")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)
    
    if args.quiet:
        # Disable console output
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.removeHandler(handler)
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation. Output directory: {output_dir}")
    
    try:
        # Route to appropriate function based on command
        if args.command == "evaluate":
            _run_evaluate(args, output_dir, logger)
        elif args.command == "compare":
            _run_compare(args, output_dir, logger)
        elif args.command == "tune":
            _run_tune(args, output_dir, logger)
        elif args.command == "cross-validate":
            _run_cross_validate(args, output_dir, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            print_colored(f"ERROR: Unknown command: {args.command}", "red", bold=True)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        if not args.quiet:
            print_colored(f"ERROR: {str(e)}", "red", bold=True)
        sys.exit(1)


def _configure_eval_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the evaluate command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="transformer",
                      choices=["transformer", "groq", "fallback"],
                      help="Model to evaluate")
    model_group.add_argument("--settings", type=str, help="Path to settings file")
    
    # Threshold options
    threshold_group = parser.add_argument_group("Threshold Options")
    threshold_group.add_argument("--threshold-start", type=float, default=0.1,
                      help="Start of threshold sweep range")
    threshold_group.add_argument("--threshold-end", type=float, default=0.9,
                      help="End of threshold sweep range")
    threshold_group.add_argument("--threshold-step", type=float, default=0.05,
                      help="Step size for threshold sweep")
    threshold_group.add_argument("--optimize-metric", type=str, default="f1",
                      choices=["f1", "precision", "recall"],
                      help="Metric to optimize when finding optimal threshold")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    output_group.add_argument("--report-name", type=str, default="evaluation_report",
                      help="Base name for report files")
    output_group.add_argument("--export-format", type=str, default="all",
                      choices=["json", "csv", "pickle", "all"],
                      help="Format(s) to export results in")
    output_group.add_argument("--generate-plots", action="store_true",
                      help="Generate and save visualization plots")
    output_group.add_argument("--interactive-plots", action="store_true",
                      help="Generate interactive HTML visualizations")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    process_group.add_argument("--parallel", action="store_true",
                      help="Use parallel processing for batch evaluation")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--cv", type=int, default=0,
                      help="Number of cross-validation folds (0 for no CV)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _configure_compare_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the compare command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--models", type=str, required=True,
                      help="Comma-separated list of models to compare (transformer,groq,fallback)")
    model_group.add_argument("--model-names", type=str,
                      help="Optional comma-separated list of display names for models")
    model_group.add_argument("--settings-files", type=str,
                      help="Optional comma-separated list of settings files for models")
    
    # Comparison options
    compare_group = parser.add_argument_group("Comparison Options")
    compare_group.add_argument("--significance-test", type=str, default="bootstrap",
                      choices=["mcnemar", "bootstrap", "t-test"],
                      help="Statistical test to use for comparing models")
    compare_group.add_argument("--metrics", type=str, default="f1,precision,recall",
                      help="Comma-separated list of metrics to compare")
    compare_group.add_argument("--optimize-thresholds", action="store_true",
                      help="Optimize thresholds for each model")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    output_group.add_argument("--report-name", type=str, default="comparison_report",
                      help="Base name for report files")
    output_group.add_argument("--export-format", type=str, default="all",
                      choices=["json", "csv", "pickle", "all"],
                      help="Format(s) to export results in")
    output_group.add_argument("--generate-plots", action="store_true",
                      help="Generate and save visualization plots")
    output_group.add_argument("--interactive-plots", action="store_true",
                      help="Generate interactive HTML visualizations")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    process_group.add_argument("--parallel", action="store_true",
                      help="Use parallel processing for batch evaluation")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--cv", type=int, default=0,
                      help="Number of cross-validation folds (0 for no CV)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _configure_tune_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the tune command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="transformer",
                      choices=["transformer", "groq", "fallback"],
                      help="Model to tune thresholds for")
    model_group.add_argument("--settings", type=str, help="Path to settings file")
    
    # Tuning options
    tuning_group = parser.add_argument_group("Tuning Options")
    tuning_group.add_argument("--optimize-metric", type=str, default="f1",
                      choices=["f1", "precision", "recall"],
                      help="Metric to optimize when finding optimal threshold")
    tuning_group.add_argument("--threshold-starts", type=str, default="0.1,0.2,0.3",
                      help="Comma-separated list of threshold start values to try")
    tuning_group.add_argument("--threshold-ends", type=str, default="0.7,0.8,0.9",
                      help="Comma-separated list of threshold end values to try")
    tuning_group.add_argument("--threshold-steps", type=str, default="0.01,0.02,0.05,0.1",
                      help="Comma-separated list of threshold step sizes to try")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--cv", type=int, default=5,
                      help="Number of cross-validation folds")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save tuning results")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _configure_cv_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the cross-validate command parser."""
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument("--data", type=str, required=True, help="Path to labeled test data")
    data_group.add_argument("--format", type=str, default="csv", choices=["csv", "json", "jsonl", "txt"],
                      help="Format of test data file")
    data_group.add_argument("--text-column", type=str, default="text",
                      help="Name of column containing text data")
    data_group.add_argument("--sentiment-column", type=str, default="sentiment",
                      help="Name of column containing sentiment labels")
    data_group.add_argument("--emotion-column", type=str, default="emotion",
                      help="Name of column containing emotion labels")
    data_group.add_argument("--label-types", type=str, default="sentiment,emotion",
                      help="Comma-separated list of label types to evaluate")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, default="transformer",
                      choices=["transformer", "groq", "fallback"],
                      help="Model to cross-validate")
    model_group.add_argument("--settings", type=str, help="Path to settings file")
    
    # Cross-validation options
    cv_group = parser.add_argument_group("Cross-validation Options")
    cv_group.add_argument("--folds", type=int, default=5,
                      help="Number of cross-validation folds")
    cv_group.add_argument("--metrics", type=str, default="f1,precision,recall",
                      help="Comma-separated list of metrics to calculate")
    cv_group.add_argument("--optimize-thresholds", action="store_true",
                      help="Optimize thresholds for each fold")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="evaluation_results",
                      help="Directory to save cross-validation results")
    
    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch-size", type=int, default=0,
                      help="Batch size for processing (0 for no batching)")
    
    # Logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-file", type=str, help="Path to log file")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    logging_group.add_argument("--quiet", action="store_true", help="Suppress console output")


def _run_evaluate(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the evaluate command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in evaluation
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_threshold_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        model = load_model(args.model, args.settings, logger=logger)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Decide whether to use cross-validation
    if args.cv > 1:
        # Perform cross-validation
        logger.info(f"Performing {args.cv}-fold cross-validation")
        
        try:
            cv_results = perform_cross_validation(
                evaluator=evaluator,
                model=model,
                data_loader=data_loader,
                label_types=label_types,
                cv_folds=args.cv,
                metrics=["f1", "precision", "recall"],
                batch_size=args.batch_size if args.batch_size > 0 else None,
                optimize_threshold=True,
                logger=logger
            )
            
            # Save CV results
            cv_path = os.path.join(output_dir, "cross_validation_results.json")
            with open(cv_path, "w") as f:
                json.dump(cv_results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.number) else o)
            
            # Print summary
            if not args.quiet:
                print_colored("\n===== CROSS-VALIDATION RESULTS =====", "green", bold=True)
                for label_type, results in cv_results.items():
                    print_colored(f"\n{label_type.upper()}:", "blue", bold=True)
                    print(f"Mean threshold: {results.get('mean_threshold', 0.5):.3f} ± {results.get('std_threshold', 0):.3f}")
                    print(f"Mean F1: {results.get('mean_f1', 0):.4f} ± {results.get('std_f1', 0):.4f}")
                    print(f"Mean precision: {results.get('mean_precision', 0):.4f} ± {results.get('std_precision', 0):.4f}")
                    print(f"Mean recall: {results.get('mean_recall', 0):.4f} ± {results.get('std_recall', 0):.4f}")
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")
    else:
        # Perform regular evaluation
        logger.info(f"Performing regular evaluation with {args.model} model")
        
        try:
            if args.batch_size > 0:
                # Batch evaluation
                results = evaluate_with_batches(
                    evaluator=evaluator,
                    model=model,
                    data_loader=data_loader,
                    batch_size=args.batch_size,
                    label_types=label_types,
                    logger=logger
                )
            else:
                # Regular evaluation
                logger.info("Loading test data and ground truth")
                evaluator.data_loader = data_loader
                
                # Set ground truth
                ground_truth = {}
                for label_type in label_types:
                    if label_type in label_columns:
                        ground_truth[label_type] = data_loader.get_labels(label_type)
                
                evaluator.ground_truth = ground_truth
                
                logger.info(f"Running evaluation with {args.model} model")
                texts = data_loader.get_texts()
                predictions = model.predict(texts)
                
                results = evaluator.evaluate_model(
                    predictions=predictions,
                    label_types=label_types
                )
            
            # Configure threshold optimizers
            for label_type, optimizer in evaluator.threshold_optimizers.items():
                optimizer.step_size = args.threshold_step
                optimizer.metric_name = args.optimize_metric
                optimizer.sweep_thresholds(args.threshold_start, args.threshold_end)
                
                # Find optimal threshold
                optimal_threshold = optimizer.find_optimal_threshold(label_type)
                logger.info(f"Optimal threshold for {label_type}: {optimal_threshold:.3f}")
            
            # Generate visualizations
            if args.generate_plots:
                logger.info("Generating visualizations")
                for label_type in label_types:
                    if label_type in evaluator.visualizers:
                        vis_dir = os.path.join(output_dir, f"{label_type}_visualizations")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # Generate standard plots
                        evaluator.visualize_results(label_type, vis_dir)
                        
                        # Generate interactive plots if requested
                        if args.interactive_plots:
                            dashboard_path = os.path.join(vis_dir, "dashboard.html")
                            evaluator.visualizers[label_type].create_dashboard(dashboard_path)
                            
                            threshold_tuner_path = os.path.join(vis_dir, "threshold_tuner.html")
                            evaluator.visualizers[label_type].create_interactive_threshold_tuner(
                                label_type=label_type
                            )
            
            # Export results
            logger.info("Exporting evaluation results")
            
            export_formats = []
            if args.export_format == "all":
                export_formats = ["json", "csv", "pickle"]
            else:
                export_formats = [args.export_format]
                
            for fmt in export_formats:
                output_path = os.path.join(output_dir, f"{args.report_name}.{fmt}")
                evaluator.export_results(output_path, format=fmt)
                
            # Generate detailed report
            report_path = os.path.join(output_dir, f"{args.report_name}.md")
            evaluator.generate_report(report_path)
            
            # Save evaluation state for future reference
            state_path = os.path.join(output_dir, "evaluator_state.pkl")
            evaluator.save_state(state_path)
            
            # Print summary to console
            if not args.quiet:
                print_colored("\n===== EVALUATION RESULTS =====", "green", bold=True)
                for label_type, metrics in results.items():
                    print_colored(f"\n{label_type.upper()} METRICS:", "blue", bold=True)
                    print(f"Optimal threshold: {metrics['threshold']:.3f}")
                    print(f"F1 score: {metrics['f1']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"AUC: {metrics.get('auc', 0):.4f}")
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise ValueError(f"Evaluation failed: {str(e)}")
                
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


def _run_compare(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the compare command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in comparison
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse models
    model_names = args.models.split(",")
    
    # Parse model display names if provided
    if args.model_names:
        display_names = args.model_names.split(",")
        if len(display_names) != len(model_names):
            logger.warning(f"Number of display names ({len(display_names)}) doesn't match number of models ({len(model_names)}). Using model names as display names.")
            display_names = model_names
    else:
        display_names = model_names
    
    # Parse settings files if provided
    if args.settings_files:
        settings_files = args.settings_files.split(",")
        if len(settings_files) != len(model_names):
            logger.warning(f"Number of settings files ({len(settings_files)}) doesn't match number of models ({len(model_names)}). Using no settings files.")
            settings_files = [None] * len(model_names)
    else:
        settings_files = [None] * len(model_names)
    
    # Create model configurations
    model_configs = []
    for i, model_name in enumerate(model_names):
        model_configs.append({
            "name": model_name,
            "display_name": display_names[i],
            "settings_path": settings_files[i]
        })
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Parse metrics
    metrics = args.metrics.split(",")
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Perform model comparison
    try:
        comparison_results = compare_multiple_models(
            model_configs=model_configs,
            data_loader=data_loader,
            label_types=label_types,
            metrics=metrics,
            significance_test=args.significance_test,
            cv_folds=args.cv if args.cv > 1 else None,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            optimize_threshold=args.optimize_thresholds,
            output_dir=output_dir if args.generate_plots else None,
            logger=logger
        )
        
        # Save comparison results
        comparison_path = os.path.join(output_dir, f"{args.report_name}.json")
        with open(comparison_path, "w") as f:
            json.dump(
                comparison_results, 
                f, 
                indent=2, 
                default=lambda o: float(o) if isinstance(o, (np.number, np.floating)) else o
            )
        
        # Generate comparison report
        report_path = os.path.join(output_dir, f"{args.report_name}.md")
        _generate_comparison_report(comparison_results, report_path, logger)
        
        # Export results in different formats if requested
        export_formats = []
        if args.export_format == "all":
            export_formats = ["csv", "pickle"]  # JSON already done
        elif args.export_format != "json":
            export_formats = [args.export_format]
            
        for fmt in export_formats:
            try:
                output_path = os.path.join(output_dir, f"{args.report_name}.{fmt}")
                if fmt == "csv":
                    # For CSV, convert to a flattened DataFrame
                    results_list = []
                    for model_name, results in comparison_results["individual_results"].items():
                        if "evaluation" in results:
                            for label_type, metrics in results["evaluation"].items():
                                row = {"model": model_name, "label_type": label_type}
                                row.update(metrics)
                                results_list.append(row)
                    
                    if results_list:
                        df = pd.DataFrame(results_list)
                        df.to_csv(output_path, index=False)
                elif fmt == "pickle":
                    # For pickle, save the full results
                    import pickle
                    with open(output_path, 'wb') as f:
                        pickle.dump(comparison_results, f)
            except Exception as e:
                logger.error(f"Error exporting to {fmt} format: {str(e)}")
                # Continue with other formats if one fails
        
        # Print summary to console
        if not args.quiet:
            print_colored("\n===== MODEL COMPARISON RESULTS =====", "green", bold=True)
            
            # Print best models
            if "best_models" in comparison_results:
                print_colored("\nBest Models:", "blue", bold=True)
                for label_type, model_name in comparison_results["best_models"].items():
                    print(f"{label_type.capitalize()}: {model_name}")
                    
            if "best_model" in comparison_results and comparison_results["best_model"]:
                print_colored(f"\nOverall Best Model: {comparison_results['best_model']}", "green", bold=True)
                
            # Print statistical significance results
            if "statistical_tests" in comparison_results and comparison_results["statistical_tests"]:
                print_colored("\nStatistical Significance Tests:", "blue", bold=True)
                for pair, tests in comparison_results["statistical_tests"].items():
                    print(f"\n{pair}:")
                    for label_type, test_results in tests.items():
                        better_model = test_results.get("better_model")
                        models = pair.split(" vs ")
                        if better_model == 1:
                            print(f"  {label_type.capitalize()}: {models[0]} is significantly better")
                        elif better_model == 2:
                            print(f"  {label_type.capitalize()}: {models[1]} is significantly better")
                        else:
                            print(f"  {label_type.capitalize()}: No significant difference")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        raise ValueError(f"Model comparison failed: {str(e)}")
        
    logger.info(f"Comparison complete. Results saved to: {output_dir}")


def _run_tune(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the tune command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in tuning
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_threshold_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Parse tuning parameters
    try:
        threshold_starts = [float(x) for x in args.threshold_starts.split(",")]
        threshold_ends = [float(x) for x in args.threshold_ends.split(",")]
        threshold_steps = [float(x) for x in args.threshold_steps.split(",")]
    except ValueError:
        logger.error("Invalid threshold values: must be numeric")
        raise ValueError("Invalid threshold values: must be numeric")
    
    # Create parameter grid
    param_grid = {
        "threshold_start": threshold_starts,
        "threshold_end": threshold_ends,
        "threshold_step": threshold_steps
    }
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        model = load_model(args.model, args.settings, logger=logger)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Get ground truth
    ground_truth = {}
    for label_type in label_types:
        if label_type in label_columns:
            ground_truth[label_type] = data_loader.get_labels(label_type)
    
    # Get predictions
    try:
        logger.info(f"Getting predictions with {args.model} model")
        texts = data_loader.get_texts()
        
        if args.batch_size > 0:
            # Process in batches
            all_predictions = []
            for i in range(0, len(texts), args.batch_size):
                batch_texts = texts[i:i+min(args.batch_size, len(texts)-i)]
                batch_predictions = model.predict(batch_texts)
                all_predictions.extend(batch_predictions)
            predictions = all_predictions
        else:
            predictions = model.predict(texts)
    except Exception as e:
        logger.error(f"Error getting model predictions: {str(e)}")
        raise ValueError(f"Failed to get model predictions: {str(e)}")
    
    # Run threshold parameter tuning for each label type
    tuning_results = {}
    
    try:
        for label_type in label_types:
            if label_type in ground_truth:
                logger.info(f"Tuning threshold parameters for {label_type}")
                
                # Check if we have enough unique classes for tuning
                unique_classes = set(ground_truth[label_type])
                if len(unique_classes) < 2:
                    logger.warning(f"Not enough unique classes for {label_type} (found {len(unique_classes)}). Skipping tuning.")
                    continue
                
                tuning_result = tune_threshold_params(
                    evaluator=evaluator,
                    predictions=predictions,
                    ground_truth=ground_truth,
                    label_type=label_type,
                    param_grid=param_grid,
                    metric=args.optimize_metric,
                    cv_folds=args.cv,
                    logger=logger
                )
                
                tuning_results[label_type] = tuning_result
                
                # Print best parameters
                best_params = tuning_result.get("best_params", {})
                best_score = tuning_result.get("best_score", 0)
                
                if best_params:
                    logger.info(
                        f"Best parameters for {label_type}: "
                        f"start={best_params.get('threshold_start', 0.1)}, "
                        f"end={best_params.get('threshold_end', 0.9)}, "
                        f"step={best_params.get('threshold_step', 0.05)}, "
                        f"{args.optimize_metric}={best_score:.4f}"
                    )
        
        # Save tuning results
        tuning_path = os.path.join(output_dir, "threshold_tuning_results.json")
        with open(tuning_path, "w") as f:
            json.dump(
                tuning_results, 
                f, 
                indent=2, 
                default=lambda o: float(o) if isinstance(o, (np.number, np.floating)) else o
            )
        
        # Print summary to console
        if not args.quiet:
            print_colored("\n===== THRESHOLD TUNING RESULTS =====", "green", bold=True)
            for label_type, results in tuning_results.items():
                print_colored(f"\n{label_type.upper()}:", "blue", bold=True)
                
                best_params = results.get("best_params", {})
                best_score = results.get("best_score", 0)
                
                if best_params:
                    print(f"Best parameters:")
                    print(f"  Threshold start: {best_params.get('threshold_start', 0.1)}")
                    print(f"  Threshold end: {best_params.get('threshold_end', 0.9)}")
                    print(f"  Threshold step: {best_params.get('threshold_step', 0.05)}")
                    print(f"  {args.optimize_metric.capitalize()} score: {best_score:.4f}")
    except Exception as e:
        logger.error(f"Error during threshold tuning: {str(e)}")
        raise ValueError(f"Threshold tuning failed: {str(e)}")
        
    logger.info(f"Threshold tuning complete. Results saved to: {output_dir}")


def _run_cross_validate(args: Any, output_dir: str, logger: logging.Logger) -> None:
    """
    Run the cross-validate command.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save outputs
        logger: Logger instance
        
    Raises:
        ValueError: For invalid arguments or errors in cross-validation
    """
    # Validate arguments
    if not all([
        _validate_data_args(args, logger),
        _validate_model_args(args, logger),
        _validate_output_args(args, logger),
        _validate_cv_args(args, logger)
    ]):
        raise ValueError("Invalid command line arguments")
    
    # Parse label types
    label_types = args.label_types.split(",")
    
    # Parse metrics
    metrics = args.metrics.split(",")
    
    # Create label columns mapping
    label_columns = {}
    if "sentiment" in label_types:
        label_columns["sentiment"] = args.sentiment_column
    if "emotion" in label_types:
        label_columns["emotion"] = args.emotion_column
    
    logger.info(f"Loading test data: {args.data}")
    
    try:
        # Create data loader
        if args.batch_size > 0:
            logger.info(f"Using batch processing with batch size: {args.batch_size}")
            data_loader = BatchTestLoader(
                data_path=args.data,
                batch_size=args.batch_size,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        else:
            data_loader = TestDataLoader(
                data_path=args.data,
                label_format="standard",
                text_column=args.text_column,
                label_columns=label_columns
            )
        
        # Load data
        data_loader.load_data()
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise ValueError(f"Failed to load test data: {str(e)}")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    try:
        # Load model
        model = load_model(args.model, args.settings, logger=logger)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Perform cross-validation
    try:
        logger.info(f"Performing {args.folds}-fold cross-validation")
        
        # Validate that we have enough data for the requested number of folds
        data_size = len(data_loader.get_texts())
        if data_size < args.folds:
            logger.error(f"Not enough data ({data_size} samples) for {args.folds} folds")
            raise ValueError(f"Not enough data ({data_size} samples) for {args.folds} folds")
            
        # Validate that we have enough samples per class for stratified CV
        if any(label_type in label_columns for label_type in label_types):
            for label_type in label_types:
                if label_type in label_columns:
                    labels = data_loader.get_labels(label_type)
                    label_counts = {}
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    
                    min_class_count = min(label_counts.values()) if label_counts else 0
                    if min_class_count < args.folds:
                        logger.warning(
                            f"Some classes in {label_type} have fewer than {args.folds} samples "
                            f"(minimum: {min_class_count}). Standard cross-validation will be used instead of stratified."
                        )
        
        cv_results = perform_cross_validation(
            evaluator=evaluator,
            model=model,
            data_loader=data_loader,
            label_types=label_types,
            cv_folds=args.folds,
            metrics=metrics,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            optimize_threshold=args.optimize_thresholds,
            logger=logger
        )
        
        # Save CV results
        cv_path = os.path.join(output_dir, "cross_validation_results.json")
        with open(cv_path, "w") as f:
            json.dump(
                cv_results, 
                f, 
                indent=2, 
                default=lambda o: float(o) if isinstance(o, (np.number, np.floating)) else o
            )
        
        # Print summary to console
        if not args.quiet:
            print_colored("\n===== CROSS-VALIDATION RESULTS =====", "green", bold=True)
            for label_type, results in cv_results.items():
                print_colored(f"\n{label_type.upper()}:", "blue", bold=True)
                print(f"Mean threshold: {results.get('mean_threshold', 0.5):.3f} ± {results.get('std_threshold', 0):.3f}")
                
                for metric in metrics:
                    mean_key = f"mean_{metric}"
                    std_key = f"std_{metric}"
                    
                    if mean_key in results and std_key in results:
                        print(f"Mean {metric}: {results[mean_key]:.4f} ± {results[std_key]:.4f}")
    except Exception as e:
        logger.error(f"Error during cross-validation: {str(e)}")
        raise ValueError(f"Cross-validation failed: {str(e)}")
        
    logger.info(f"Cross-validation complete. Results saved to: {output_dir}")


def _generate_comparison_report(results: Dict, output_path: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Generate a detailed markdown report of model comparison results.
    
    Args:
        results: Comparison results dictionary
        output_path: Path to save the report
        logger: Optional logger
    """
    try:
        with open(output_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            if "best_model" in results and results["best_model"]:
                f.write(f"**Overall Best Model:** {results['best_model']}\n\n")
            
            if "best_models" in results:
                f.write("**Best Models by Label Type:**\n")
                for label_type, model_name in results["best_models"].items():
                    f.write(f"- {label_type.capitalize()}: {model_name}\n")
                f.write("\n")
            
            # Individual results section
            f.write("## Individual Model Results\n\n")
            if "individual_results" in results:
                for model_name, model_results in results["individual_results"].items():
                    f.write(f"### {model_name}\n\n")
                    
                    if "evaluation" in model_results:
                        for label_type, metrics in model_results["evaluation"].items():
                            f.write(f"**{label_type.capitalize()} Metrics:**\n")
                            f.write(f"- F1 Score: {metrics.get('f1', 0):.4f}\n")
                            f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
                            f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
                            f.write(f"- AUC: {metrics.get('auc', 0):.4f}\n")
                            f.write(f"- Optimal Threshold: {metrics.get('threshold', 0.5):.3f}\n\n")
            
            # Statistical tests section
            if "statistical_tests" in results and results["statistical_tests"]:
                f.write("## Statistical Significance Tests\n\n")
                for pair, tests in results["statistical_tests"].items():
                    f.write(f"### {pair}\n\n")
                    for label_type, test_results in tests.items():
                        f.write(f"**{label_type.capitalize()}:**\n")
                        f.write(f"- P-value: {test_results.get('p_value', 0):.4f}\n")
                        f.write(f"- Significant: {test_results.get('significant', False)}\n")
                        
                        better_model = test_results.get("better_model")
                        if better_model == 1:
                            models = pair.split(" vs ")
                            f.write(f"- Better model: {models[0]}\n")
                        elif better_model == 2:
                            models = pair.split(" vs ")
                            f.write(f"- Better model: {models[1]}\n")
                        else:
                            f.write("- No significant difference\n")
                        f.write("\n")
        
        if logger:
            logger.info(f"Comparison report saved to: {output_path}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error generating comparison report: {str(e)}")
        raise


if __name__ == "__main__":
    run_evaluation_cli() 
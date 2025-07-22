"""
Utilities for model comparison.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple, Union
from src.analyzer.sentiment import analyze_sentiment
from src.analyzer.emotion import analyze_emotion
from src.ui.utils.groq_client import GroqClient
import concurrent.futures
import time
import math


class ModelComparator:
    """
    Utilities for comparing different models.
    """
    
    def __init__(self):
        """
        Initialize the model comparator.
        """
        self.model_names = {
            "standard": "Standard Model",
            "groq_llama3-70b": "Groq (LLaMA-3 70B)",
            "groq_llama3-8b": "Groq (LLaMA-3 8B)",
            "groq_mixtral": "Groq (Mixtral 8x7B)",
            "groq_gemma": "Groq (Gemma 7B)"
        }
        
        self.model_config = {
            "standard": {
                "type": "standard",
                "name": "Standard Model",
                "description": "Local transformer-based model"
            },
            "groq_llama3-70b": {
                "type": "groq",
                "name": "Groq (LLaMA-3 70B)",
                "model": "llama3-70b-8192",
                "description": "LLaMA-3 70B model via Groq API"
            },
            "groq_llama3-8b": {
                "type": "groq",
                "name": "Groq (LLaMA-3 8B)",
                "model": "llama3-8b-8192",
                "description": "LLaMA-3 8B model via Groq API"
            },
            "groq_mixtral": {
                "type": "groq",
                "name": "Groq (Mixtral 8x7B)",
                "model": "mixtral-8x7b-32768",
                "description": "Mixtral 8x7B model via Groq API"
            },
            "groq_gemma": {
                "type": "groq",
                "name": "Groq (Gemma 7B)",
                "model": "gemma-7b-it",
                "description": "Gemma 7B model via Groq API"
            }
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model identifiers.
        
        Returns:
            List[str]: List of available model identifiers
        """
        available_models = ["standard"]
        
        # Check if Groq API is available
        if st.session_state.get("use_groq", False) and st.session_state.get("groq_api_key", ""):
            # Add all Groq models
            available_models.extend([
                "groq_llama3-70b", 
                "groq_llama3-8b", 
                "groq_mixtral", 
                "groq_gemma"
            ])
        
        return available_models
    
    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """
        Get details for a specific model.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Dict: Model details
        """
        return self.model_config.get(model_id, {})
    
    def analyze_with_models(self, text: str, models: List[str], 
                           analysis_type: str = "both") -> Dict[str, Any]:
        """
        Analyze text using multiple models.
        
        Args:
            text (str): Text to analyze
            models (List[str]): List of model identifiers
            analysis_type (str): Type of analysis ("sentiment", "emotion", or "both")
            
        Returns:
            Dict: Results from all models
        """
        if not text or not models:
            return {"error": "No text or models specified"}
        
        results = {}
        
        # Process with each model
        for model_id in models:
            model_name = self.model_names.get(model_id, model_id)
            
            try:
                # Get model configuration
                model_config = self.model_config.get(model_id, {})
                model_type = model_config.get("type", "unknown")
                
                if model_type == "standard":
                    # Use standard model for analysis
                    model_results = self._analyze_with_standard_model(
                        text, 
                        analysis_type=analysis_type
                    )
                elif model_type == "groq":
                    # Use Groq model for analysis
                    groq_model = model_config.get("model", "llama3-70b-8192")
                    model_results = self._analyze_with_groq_model(
                        text, 
                        groq_model=groq_model,
                        analysis_type=analysis_type
                    )
                else:
                    model_results = {"error": f"Unknown model type: {model_type}"}
                
                # Add model info
                model_results["model_id"] = model_id
                model_results["model_name"] = model_name
                model_results["model_type"] = model_type
                
                # Store in results
                results[model_id] = model_results
                
            except Exception as e:
                # Handle errors for individual models
                results[model_id] = {
                    "error": str(e),
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_type": model_config.get("type", "unknown")
                }
        
        # Add overall data
        results["text"] = text
        results["models_compared"] = models
        results["analysis_type"] = analysis_type
        results["comparison_metrics"] = self.calculate_comparison_metrics(results)
        
        return results
    
    def _analyze_with_standard_model(self, text: str, analysis_type: str = "both") -> Dict[str, Any]:
        """
        Analyze text using standard model.
        
        Args:
            text (str): Text to analyze
            analysis_type (str): Type of analysis ("sentiment", "emotion", or "both")
            
        Returns:
            Dict: Analysis results
        """
        result = {"text": text}
        
        # Add timing information
        start_time = time.time()
        
        # Perform sentiment analysis if requested
        if analysis_type in ["sentiment", "both"]:
            sentiment_results = analyze_sentiment(
                text, 
                threshold=st.session_state.get("sentiment_threshold", 0.5),
                use_fallback=st.session_state.get("use_api_fallback", False)
            )
            result["sentiment"] = sentiment_results
        
        # Perform emotion analysis if requested
        if analysis_type in ["emotion", "both"]:
            emotion_results = analyze_emotion(
                text, 
                threshold=st.session_state.get("emotion_threshold", 0.3),
                use_fallback=st.session_state.get("use_api_fallback", False)
            )
            result["emotion"] = emotion_results
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def _analyze_with_groq_model(self, text: str, groq_model: str = "llama3-70b-8192", 
                                analysis_type: str = "both") -> Dict[str, Any]:
        """
        Analyze text using Groq model.
        
        Args:
            text (str): Text to analyze
            groq_model (str): Groq model to use
            analysis_type (str): Type of analysis ("sentiment", "emotion", or "both")
            
        Returns:
            Dict: Analysis results
        """
        # Initialize Groq client
        client = GroqClient(
            api_key=st.session_state.get("groq_api_key", ""),
            model=groq_model
        )
        
        # Add timing information
        start_time = time.time()
        
        # Process with Groq API
        result = client.analyze_text(text, analysis_type)
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        result["groq_model"] = groq_model
        
        return result
    
    def process_batch_comparison(self, texts: List[str], models: List[str], 
                                analysis_type: str = "both",
                                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process a batch of texts using multiple models for comparison.
        
        Args:
            texts (List[str]): List of texts to analyze
            models (List[str]): List of models to compare
            analysis_type (str): Type of analysis ("sentiment", "emotion", or "both")
            progress_callback (callable, optional): Callback for progress updates
            
        Returns:
            Dict: Batch comparison results
        """
        if not texts or not models:
            return {"error": "No texts or models specified"}
        
        # Initialize results
        batch_results = {
            "total_texts": len(texts),
            "models_compared": models,
            "model_names": [self.model_names.get(model_id, model_id) for model_id in models],
            "analysis_type": analysis_type,
            "detailed_results": [],
            "model_performance": {},
            "agreement_stats": {},
            "processing_stats": {}
        }
        
        # Track start time
        start_time = time.time()
        
        # Initialize counters
        processed_count = 0
        model_predictions = {model_id: {"sentiment": {}, "emotion": {}} for model_id in models}
        processing_times = {model_id: [] for model_id in models}
        
        # Process each text
        for idx, text in enumerate(texts):
            # Skip empty texts
            if not text or text.isspace():
                continue
                
            # Analyze with all models
            comparison_result = self.analyze_with_models(text, models, analysis_type)
            
            # Process the results
            simplified_result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "model_results": {}
            }
            
            # Extract model-specific results
            for model_id in models:
                model_result = comparison_result.get(model_id, {})
                
                # Skip if error
                if "error" in model_result:
                    simplified_result["model_results"][model_id] = {
                        "error": model_result["error"]
                    }
                    continue
                
                # Track processing time
                if "processing_time" in model_result:
                    processing_times[model_id].append(model_result["processing_time"])
                
                # Extract sentiment results if available
                if "sentiment" in model_result:
                    sentiment = model_result["sentiment"].get("prediction", "unknown")
                    confidence = model_result["sentiment"].get("confidence", 0.0)
                    
                    # Add to model results
                    simplified_result["model_results"][model_id] = {
                        "sentiment": sentiment,
                        "confidence": confidence
                    }
                    
                    # Update sentiment counts
                    if sentiment not in model_predictions[model_id]["sentiment"]:
                        model_predictions[model_id]["sentiment"][sentiment] = 0
                    model_predictions[model_id]["sentiment"][sentiment] += 1
                
                # Extract emotion results if available
                if "emotion" in model_result:
                    primary_emotion = model_result["emotion"].get("primary_emotion", "unknown")
                    
                    # Add to model results
                    if "model_results" not in simplified_result:
                        simplified_result["model_results"][model_id] = {}
                    simplified_result["model_results"][model_id]["emotion"] = primary_emotion
                    
                    # Update emotion counts
                    if primary_emotion not in model_predictions[model_id]["emotion"]:
                        model_predictions[model_id]["emotion"][primary_emotion] = 0
                    model_predictions[model_id]["emotion"][primary_emotion] += 1
            
            # Add to detailed results
            batch_results["detailed_results"].append(simplified_result)
            
            # Update progress
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count / len(texts))
        
        # Calculate processing stats
        batch_results["processing_stats"] = {
            "total_time": time.time() - start_time,
            "texts_processed": processed_count,
            "model_avg_time": {model_id: np.mean(times) if times else 0 
                               for model_id, times in processing_times.items()},
            "model_total_time": {model_id: sum(times) if times else 0 
                                for model_id, times in processing_times.items()}
        }
        
        # Calculate model performance stats
        batch_results["model_performance"] = {
            model_id: {
                "sentiment_distribution": predictions["sentiment"],
                "emotion_distribution": predictions["emotion"] if "emotion" in analysis_type else {},
                "avg_processing_time": np.mean(processing_times[model_id]) if processing_times[model_id] else 0,
            } for model_id, predictions in model_predictions.items()
        }
        
        # Calculate agreement statistics
        if len(models) > 1:
            batch_results["agreement_stats"] = self.calculate_batch_agreement(
                batch_results["detailed_results"],
                models,
                analysis_type
            )
        
        return batch_results
    
    def calculate_batch_agreement(self, detailed_results: List[Dict[str, Any]], 
                                 models: List[str], analysis_type: str) -> Dict[str, Any]:
        """
        Calculate agreement statistics across all texts.
        
        Args:
            detailed_results (List[Dict]): Detailed results for each text
            models (List[str]): List of models compared
            analysis_type (str): Type of analysis performed
            
        Returns:
            Dict: Agreement statistics
        """
        # Initialize agreement stats
        agreement_stats = {
            "sentiment_agreement": 0,
            "emotion_agreement": 0,
            "full_agreement": 0,
            "disagreement": 0,
            "pairwise_agreement": {}
        }
        
        # Initialize pairwise agreement counters
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                pair_key = f"{model1}_vs_{model2}"
                agreement_stats["pairwise_agreement"][pair_key] = {
                    "sentiment_agreement": 0,
                    "emotion_agreement": 0,
                    "both_agreement": 0,
                    "total_comparisons": 0
                }
        
        # Calculate agreement for each text
        valid_comparisons = 0
        
        for result in detailed_results:
            model_results = result.get("model_results", {})
            
            # Skip if any model has an error
            if any("error" in model_results.get(model, {}) for model in models):
                continue
                
            valid_comparisons += 1
            
            # Check sentiment agreement
            sentiment_values = [model_results.get(model, {}).get("sentiment") 
                              for model in models if model in model_results]
            sentiment_agreement = len(set(sentiment_values)) == 1 if sentiment_values else False
            
            # Check emotion agreement
            emotion_values = [model_results.get(model, {}).get("emotion") 
                            for model in models if model in model_results and "emotion" in model_results[model]]
            emotion_agreement = len(set(emotion_values)) == 1 if emotion_values and "emotion" in analysis_type else True
            
            # Update agreement counters
            if sentiment_agreement:
                agreement_stats["sentiment_agreement"] += 1
            
            if emotion_agreement and "emotion" in analysis_type:
                agreement_stats["emotion_agreement"] += 1
            
            if sentiment_agreement and emotion_agreement:
                agreement_stats["full_agreement"] += 1
            else:
                agreement_stats["disagreement"] += 1
            
            # Update pairwise agreement
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    if model1 in model_results and model2 in model_results:
                        pair_key = f"{model1}_vs_{model2}"
                        
                        # Get results for both models
                        result1 = model_results[model1]
                        result2 = model_results[model2]
                        
                        # Skip if error in either model
                        if "error" in result1 or "error" in result2:
                            continue
                        
                        # Update pairwise counters
                        agreement_stats["pairwise_agreement"][pair_key]["total_comparisons"] += 1
                        
                        # Check sentiment agreement
                        if "sentiment" in result1 and "sentiment" in result2:
                            if result1["sentiment"] == result2["sentiment"]:
                                agreement_stats["pairwise_agreement"][pair_key]["sentiment_agreement"] += 1
                        
                        # Check emotion agreement
                        if "emotion" in analysis_type and "emotion" in result1 and "emotion" in result2:
                            if result1["emotion"] == result2["emotion"]:
                                agreement_stats["pairwise_agreement"][pair_key]["emotion_agreement"] += 1
                        
                        # Check both agreement
                        if "sentiment" in result1 and "sentiment" in result2 and "emotion" in analysis_type:
                            if result1["sentiment"] == result2["sentiment"] and (
                                "emotion" not in analysis_type or 
                                result1.get("emotion") == result2.get("emotion")):
                                agreement_stats["pairwise_agreement"][pair_key]["both_agreement"] += 1
        
        # Convert counts to percentages
        if valid_comparisons > 0:
            agreement_stats["sentiment_agreement"] /= valid_comparisons
            agreement_stats["emotion_agreement"] /= valid_comparisons
            agreement_stats["full_agreement"] /= valid_comparisons
            agreement_stats["disagreement"] /= valid_comparisons
            
            # Convert pairwise agreement to percentages
            for pair_key in agreement_stats["pairwise_agreement"]:
                pair_data = agreement_stats["pairwise_agreement"][pair_key]
                total = pair_data["total_comparisons"]
                
                if total > 0:
                    pair_data["sentiment_agreement"] /= total
                    pair_data["emotion_agreement"] /= total
                    pair_data["both_agreement"] /= total
        
        # Add total valid comparisons
        agreement_stats["valid_comparisons"] = valid_comparisons
        
        return agreement_stats
    
    def calculate_comparison_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comparison metrics between models.
        
        Args:
            results (Dict): Results from different models
            
        Returns:
            Dict: Comparison metrics
        """
        metrics = {
            "agreement_score": None,
            "confidence_variance": None,
            "key_differences": [],
            "recommendation": None
        }
        
        # Get model IDs
        model_ids = [model_id for model_id in results.keys() 
                   if model_id not in ["text", "models_compared", "analysis_type", "comparison_metrics"]]
        
        # Skip if only one model or no valid results
        if len(model_ids) < 2 or any("error" in results.get(model_id, {}) for model_id in model_ids):
            return metrics
        
        # Calculate agreement score
        metrics["agreement_score"] = self.get_agreement_score(results, model_ids)
        
        # Calculate confidence variance
        metrics["confidence_variance"] = self.calculate_confidence_variance(results, model_ids)
        
        # Find key differences
        metrics["key_differences"] = self.find_key_differences(results, model_ids)
        
        # Get recommendation
        metrics["recommendation"] = self.get_recommendation(results, model_ids, metrics)
        
        return metrics
    
    def get_agreement_score(self, results: Dict[str, Any], model_ids: List[str]) -> float:
        """
        Calculate agreement score between models.
        
        Args:
            results (Dict): Results from different models
            model_ids (List[str]): List of model IDs to compare
            
        Returns:
            float: Agreement score (0.0 to 1.0)
        """
        agreement_scores = []
        
        # Check sentiment agreement if available
        if all("sentiment" in results.get(model_id, {}) for model_id in model_ids):
            # Get all sentiment predictions
            sentiment_predictions = [
                results[model_id]["sentiment"]["prediction"] for model_id in model_ids
            ]
            
            # Calculate agreement (1.0 if all match, 0.0 if none match)
            if len(set(sentiment_predictions)) == 1:
                # All models agree
                agreement_scores.append(1.0)
            elif len(set(sentiment_predictions)) == len(model_ids):
                # All models disagree
                agreement_scores.append(0.0)
            else:
                # Partial agreement - calculate as ratio of agreeing pairs
                total_pairs = (len(model_ids) * (len(model_ids) - 1)) / 2
                agreeing_pairs = 0
                
                for i, model1 in enumerate(model_ids):
                    for model2 in model_ids[i+1:]:
                        if results[model1]["sentiment"]["prediction"] == results[model2]["sentiment"]["prediction"]:
                            agreeing_pairs += 1
                
                agreement_scores.append(agreeing_pairs / total_pairs)
        
        # Check emotion agreement if available
        if all("emotion" in results.get(model_id, {}) for model_id in model_ids):
            # Get all primary emotion predictions
            emotion_predictions = [
                results[model_id]["emotion"]["primary_emotion"] for model_id in model_ids
            ]
            
            # Calculate agreement
            if len(set(emotion_predictions)) == 1:
                # All models agree
                agreement_scores.append(1.0)
            elif len(set(emotion_predictions)) == len(model_ids):
                # All models disagree
                agreement_scores.append(0.0)
            else:
                # Partial agreement
                total_pairs = (len(model_ids) * (len(model_ids) - 1)) / 2
                agreeing_pairs = 0
                
                for i, model1 in enumerate(model_ids):
                    for model2 in model_ids[i+1:]:
                        if results[model1]["emotion"]["primary_emotion"] == results[model2]["emotion"]["primary_emotion"]:
                            agreeing_pairs += 1
                
                agreement_scores.append(agreeing_pairs / total_pairs)
        
        # Return average agreement score if any scores were calculated
        if agreement_scores:
            return sum(agreement_scores) / len(agreement_scores)
        else:
            return None
    
    def calculate_confidence_variance(self, results: Dict[str, Any], model_ids: List[str]) -> float:
        """
        Calculate variance in confidence scores between models.
        
        Args:
            results (Dict): Results from different models
            model_ids (List[str]): List of model IDs to compare
            
        Returns:
            float: Confidence variance
        """
        # Check if sentiment results with confidence are available
        if not all(("sentiment" in results.get(model_id, {}) and 
                   "confidence" in results.get(model_id, {}).get("sentiment", {})) 
                  for model_id in model_ids):
            return None
        
        # Get confidence scores
        confidence_scores = [
            results[model_id]["sentiment"]["confidence"] for model_id in model_ids
        ]
        
        # Calculate variance
        return float(np.var(confidence_scores))
    
    def find_key_differences(self, results: Dict[str, Any], model_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Find key differences between models.
        
        Args:
            results (Dict): Results from different models
            model_ids (List[str]): List of model IDs to compare
            
        Returns:
            List[Dict]: List of key differences
        """
        key_differences = []
        
        # Check for sentiment differences
        if all("sentiment" in results.get(model_id, {}) for model_id in model_ids):
            # Get sentiment predictions and confidence
            sentiment_data = {
                model_id: {
                    "prediction": results[model_id]["sentiment"].get("prediction", "unknown"),
                    "confidence": results[model_id]["sentiment"].get("confidence", 0.0)
                } for model_id in model_ids
            }
            
            # Check if predictions differ
            predictions = [data["prediction"] for data in sentiment_data.values()]
            if len(set(predictions)) > 1:
                # Add difference
                key_differences.append({
                    "type": "sentiment",
                    "description": "Models disagree on sentiment prediction",
                    "details": {model_id: sentiment_data[model_id]["prediction"] for model_id in model_ids},
                    "confidence": {model_id: sentiment_data[model_id]["confidence"] for model_id in model_ids}
                })
            
            # Check if confidence differs significantly
            confidences = [data["confidence"] for data in sentiment_data.values()]
            if max(confidences) - min(confidences) > 0.2:  # 20% difference threshold
                key_differences.append({
                    "type": "confidence",
                    "description": "Significant difference in confidence levels",
                    "details": {model_id: f"{sentiment_data[model_id]['confidence']:.1%}" 
                              for model_id in model_ids}
                })
        
        # Check for emotion differences
        if all("emotion" in results.get(model_id, {}) for model_id in model_ids):
            # Get primary emotion predictions
            emotion_data = {
                model_id: {
                    "primary_emotion": results[model_id]["emotion"].get("primary_emotion", "unknown")
                } for model_id in model_ids
            }
            
            # Check if predictions differ
            emotions = [data["primary_emotion"] for data in emotion_data.values()]
            if len(set(emotions)) > 1:
                # Add difference
                key_differences.append({
                    "type": "emotion",
                    "description": "Models disagree on primary emotion",
                    "details": {model_id: emotion_data[model_id]["primary_emotion"] for model_id in model_ids}
                })
        
        # Check for processing time differences
        if all("processing_time" in results.get(model_id, {}) for model_id in model_ids):
            # Get processing times
            times = {model_id: results[model_id]["processing_time"] for model_id in model_ids}
            
            # Check if there's significant difference
            if max(times.values()) > min(times.values()) * 2:  # 2x threshold
                key_differences.append({
                    "type": "processing_time",
                    "description": "Significant difference in processing times",
                    "details": {model_id: f"{times[model_id]:.2f}s" for model_id in model_ids}
                })
        
        return key_differences
    
    def get_recommendation(self, results: Dict[str, Any], model_ids: List[str], 
                          metrics: Dict[str, Any]) -> str:
        """
        Get recommendation based on model comparison.
        
        Args:
            results (Dict): Results from different models
            model_ids (List[str]): List of model IDs to compare
            metrics (Dict): Comparison metrics
            
        Returns:
            str: Recommendation message
        """
        # Skip if missing data
        if "agreement_score" not in metrics or metrics["agreement_score"] is None:
            return "Insufficient data for recommendation"
        
        # Get model names for better messages
        model_names = {model_id: self.model_names.get(model_id, model_id) for model_id in model_ids}
        
        # Check agreement level
        agreement_score = metrics["agreement_score"]
        
        if agreement_score >= 0.9:
            # High agreement case
            return (f"High agreement ({agreement_score:.0%}) between models. "
                   f"Results are consistent across different approaches.")
        
        elif agreement_score >= 0.5:
            # Moderate agreement
            # Find the most confident model for sentiment
            if all(("sentiment" in results.get(model_id, {}) and 
                   "confidence" in results.get(model_id, {}).get("sentiment", {})) 
                  for model_id in model_ids):
                
                confidence_scores = {
                    model_id: results[model_id]["sentiment"]["confidence"] 
                    for model_id in model_ids
                }
                most_confident = max(confidence_scores.items(), key=lambda x: x[1])
                
                return (f"Moderate agreement ({agreement_score:.0%}) between models. "
                       f"Consider {model_names[most_confident[0]]}'s result "
                       f"which has the highest confidence ({most_confident[1]:.0%}).")
        
        else:
            # Low agreement case
            # For low agreement, prefer larger models or ensemble
            groq_models = [m for m in model_ids if m.startswith("groq_")]
            
            if "groq_llama3-70b" in groq_models:
                # Prefer larger Groq model
                return (f"Low agreement ({agreement_score:.0%}) between models. "
                       f"Consider {model_names['groq_llama3-70b']}'s result as it uses the largest model.")
            elif groq_models:
                # Or any Groq model
                return (f"Low agreement ({agreement_score:.0%}) between models. "
                       f"Consider Groq model results as they typically have higher accuracy.")
            else:
                # Or suggest ensemble approach
                return (f"Low agreement ({agreement_score:.0%}) between models. "
                       f"Consider each model's strengths or try an ensemble approach.") 
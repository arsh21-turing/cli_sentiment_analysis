"""
Simple usage example for the evaluation suite.
"""

import pandas as pd
import json
import os

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import ModelEvaluator


# Sample model class for demo purposes
class SampleModel:
    def predict(self, texts):
        # In a real scenario, this would use a trained model
        # Here, we just return dummy predictions
        predictions = []
        for text in texts:
            predictions.append({
                "sentiment": "positive" if "good" in text.lower() else "negative",
                "sentiment_confidence": 0.8 if "good" in text.lower() else 0.7,
                "emotion": "joy" if "happy" in text.lower() else "neutral",
                "emotion_confidence": 0.75 if "happy" in text.lower() else 0.6
            })
        return predictions


def create_sample_data():
    """Create sample test data for demonstration."""
    data = {
        "text": [
            "This is a great product!",
            "I love this service.",
            "This is terrible quality.",
            "I'm so happy with the results.",
            "This makes me sad.",
            "The food was delicious.",
            "I hate this place.",
            "Amazing experience!",
            "Very disappointed.",
            "Wonderful service!"
        ],
        "sentiment": [
            "positive", "positive", "negative", "positive", 
            "negative", "positive", "negative", "positive", 
            "negative", "positive"
        ],
        "emotion": [
            "joy", "joy", "sadness", "joy", "sadness", 
            "joy", "anger", "joy", "sadness", "joy"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv("sample_test_data.csv", index=False)
    return "sample_test_data.csv"


def main():
    """Run the evaluation example."""
    print("Creating sample test data...")
    test_data_path = create_sample_data()
    
    print("Initializing evaluator...")
    # Create evaluator and load test data
    evaluator = ModelEvaluator(data_path=test_data_path)
    
    print("Creating sample model...")
    # Create sample model
    model = SampleModel()
    
    print("Running full evaluation...")
    # Run full evaluation
    results = evaluator.run_full_evaluation(
        model=model,
        output_dir="evaluation_results"
    )
    
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))
    
    # Get optimal thresholds
    thresholds = evaluator.find_optimal_thresholds()
    print(f"\nOptimal thresholds: {thresholds}")
    
    # Generate a report
    report = evaluator.generate_report("evaluation_report.md")
    print(f"\nReport generated: evaluation_report.md")
    
    print("\nEvaluation complete! Check the 'evaluation_results' directory for outputs.")


if __name__ == "__main__":
    main() 
"""
Module for loading labeled test data for model evaluation.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split

from src.utils.file_loader import get_file_iterator


class TestDataLoader:
    """Class for loading and preparing labeled test data for evaluation."""
    
    def __init__(
        self,
        data_path: str,
        label_format: str = "standard",
        text_column: str = "text",
        label_columns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the test data loader.
        
        Args:
            data_path: Path to labeled data file
            label_format: Format of labels ("standard", "binary", "multi-class", "multi-label")
            text_column: Name of column containing text data
            label_columns: Dictionary mapping label types to column names
                e.g., {"sentiment": "sentiment_label", "emotion": "emotion_label"}
        """
        self.data_path = data_path
        self.label_format = label_format
        self.text_column = text_column
        self.label_columns = label_columns or {
            "sentiment": "sentiment",
            "emotion": "emotion"
        }
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load labeled test data.
        
        Returns:
            DataFrame containing the test data
        """
        # Determine file type and load accordingly
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # Handle different JSON formats
            if isinstance(json_data, list):
                self.data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                self.data = pd.DataFrame([json_data])
        elif self.data_path.endswith('.jsonl'):
            self.data = pd.read_json(self.data_path, lines=True)
        elif self.data_path.endswith(('.txt', '.text')):
            # Assume simple text file with lines of format: "text\tlabel"
            texts = []
            labels = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        texts.append(parts[0])
                        labels.append(parts[1])
            self.data = pd.DataFrame({
                self.text_column: texts,
                self.label_columns["sentiment"]: labels
            })
        else:
            raise ValueError(f"Unsupported file type: {self.data_path}")
            
        # Validate that required columns exist
        if self.text_column not in self.data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data")
            
        for label_type, column in self.label_columns.items():
            if column not in self.data.columns:
                raise ValueError(f"Label column '{column}' for {label_type} not found in data")
                
        return self.data
    
    def get_texts(self) -> List[str]:
        """Get text samples from the loaded data."""
        if self.data is None:
            self.load_data()
        return self.data[self.text_column].tolist()
    
    def get_labels(self, label_type: Optional[str] = None) -> Union[Dict[str, List], List]:
        """
        Get ground truth labels, optionally filtered by type.
        
        Args:
            label_type: Optional label type to filter by (e.g., "sentiment", "emotion")
            
        Returns:
            Dictionary of labels by type or list of labels for specific type
        """
        if self.data is None:
            self.load_data()
            
        if label_type:
            if label_type not in self.label_columns:
                raise ValueError(f"Unknown label type: {label_type}")
            column = self.label_columns[label_type]
            return self.data[column].tolist()
        
        # Return all label types
        labels = {}
        for label_type, column in self.label_columns.items():
            labels[label_type] = self.data[column].tolist()
        return labels
    
    def get_sentiment_labels(self) -> List:
        """Get sentiment ground truth labels."""
        return self.get_labels("sentiment")
    
    def get_emotion_labels(self) -> List:
        """Get emotion ground truth labels."""
        return self.get_labels("emotion")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data to use for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        if self.data is None:
            self.load_data()
            
        train_data, test_data = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=random_state
        )
        return train_data, test_data
    
    def validate_labels(self) -> bool:
        """
        Validate that labels are in the expected format.
        
        Returns:
            True if labels are valid, False otherwise
        """
        if self.data is None:
            self.load_data()
            
        # Perform validation based on label_format
        if self.label_format == "binary":
            # Check that sentiment labels are binary (0/1 or positive/negative)
            sentiment_labels = set(self.get_sentiment_labels())
            valid_binary_labels = [
                # Numeric binary
                {0, 1},
                # Text binary
                {"positive", "negative"},
                {"pos", "neg"},
                # Other common formats
                {"0", "1"},
                {"positive", "negative", "neutral"}  # Allow neutral as well
            ]
            return any(sentiment_labels.issubset(s) for s in valid_binary_labels)
            
        elif self.label_format == "multi-class":
            # Check that labels are categorical
            for label_type, column in self.label_columns.items():
                if not isinstance(self.data[column].iloc[0], (str, int)):
                    return False
            return True
            
        elif self.label_format == "multi-label":
            # Check that labels are lists or comma-separated strings
            for label_type, column in self.label_columns.items():
                first_val = self.data[column].iloc[0]
                if not (isinstance(first_val, list) or 
                        (isinstance(first_val, str) and ',' in first_val)):
                    return False
            return True
            
        # Standard format - no specific validation
        return True
    
    def get_label_distribution(self) -> Dict[str, Dict]:
        """
        Get distribution of labels in the dataset.
        
        Returns:
            Dictionary of label distributions by type
        """
        if self.data is None:
            self.load_data()
            
        distributions = {}
        
        for label_type, column in self.label_columns.items():
            if self.label_format == "multi-label":
                # Handle multi-label case (lists or comma-separated strings)
                all_labels = []
                for val in self.data[column]:
                    if isinstance(val, str) and ',' in val:
                        all_labels.extend([l.strip() for l in val.split(',')])
                    elif isinstance(val, list):
                        all_labels.extend(val)
                    else:
                        all_labels.append(val)
                distributions[label_type] = {
                    label: all_labels.count(label) 
                    for label in set(all_labels)
                }
            else:
                # Simple value counts for single-label cases
                distributions[label_type] = self.data[column].value_counts().to_dict()
                
        return distributions
    
    def apply_predictions(self, predictions: List[Dict], label_type: str = "all") -> pd.DataFrame:
        """
        Merge predictions with ground truth data.
        
        Args:
            predictions: List of prediction dictionaries
            label_type: Label type to apply predictions for ("all" or specific type)
            
        Returns:
            DataFrame with both predictions and ground truth
        """
        if self.data is None:
            self.load_data()
            
        if len(predictions) != len(self.data):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) doesn't match "
                f"number of samples ({len(self.data)})"
            )
            
        # Create copy of data to avoid modifying original
        result_df = self.data.copy()
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Add prediction columns to result
        for col in pred_df.columns:
            result_df[f"pred_{col}"] = pred_df[col]
            
        return result_df
    
    def filter_by_confidence(self, min_confidence: float = 0.0, max_confidence: float = 1.0,
                           label_type: str = "all") -> pd.DataFrame:
        """
        Filter predictions by confidence score range.
        
        Args:
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            label_type: Label type to filter by
            
        Returns:
            Filtered DataFrame with predictions in confidence range
        """
        if self.data is None:
            self.load_data()
            
        # Check if prediction columns exist
        confidence_cols = [col for col in self.data.columns if col.endswith('_confidence')]
        if not confidence_cols:
            raise ValueError("No confidence columns found in data")
            
        if label_type == "all":
            # Filter based on all confidence columns
            mask = pd.Series(True, index=self.data.index)
            for col in confidence_cols:
                mask = mask & (self.data[col] >= min_confidence) & (self.data[col] <= max_confidence)
        else:
            # Filter based on specific label type
            conf_col = f"pred_{label_type}_confidence"
            if conf_col not in self.data.columns:
                raise ValueError(f"Confidence column for {label_type} not found")
            mask = (self.data[conf_col] >= min_confidence) & (self.data[conf_col] <= max_confidence)
            
        return self.data[mask]


class BatchTestLoader(TestDataLoader):
    """Extension of TestDataLoader for large datasets with batching."""
    
    def __init__(self, data_path: str, batch_size: int = 100, **kwargs):
        """
        Initialize batch test loader.
        
        Args:
            data_path: Path to labeled data file
            batch_size: Number of samples to process in each batch
            **kwargs: Additional arguments to pass to TestDataLoader
        """
        super().__init__(data_path, **kwargs)
        self.batch_size = batch_size
        
    def get_batch(self, batch_idx: int) -> pd.DataFrame:
        """
        Get specified batch of test data.
        
        Args:
            batch_idx: Index of batch to retrieve
            
        Returns:
            DataFrame with batch of test data
        """
        if self.data is None:
            self.load_data()
            
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))
        
        return self.data.iloc[start_idx:end_idx]
    
    def num_batches(self) -> int:
        """Get total number of batches."""
        if self.data is None:
            self.load_data()
            
        return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    def iterate_batches(self):
        """
        Generator to yield batches of data for sequential processing.
        
        Yields:
            DataFrame containing batch of test data
        """
        for i in range(self.num_batches()):
            yield self.get_batch(i) 
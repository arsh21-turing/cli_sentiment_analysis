import json
import os
import datetime
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


class AnalysisStorage:
    """
    Handles saving and loading of analysis results to local storage.
    """
    
    def __init__(self, storage_dir="saved_analyses"):
        """
        Initialize the storage utility.
        
        Args:
            storage_dir: Directory to use for storing saved analyses
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Create subdirectories for different storage types
        self.data_dir = os.path.join(storage_dir, "data")
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def save_analysis(self, data, name=None, description=None, tags=None) -> str:
        """
        Saves analysis data to local storage.
        
        Args:
            data: Analysis data to save (can be DataFrame, dict, or list)
            name: Optional name for the saved analysis
            description: Optional description
            tags: Optional list of tags
            
        Returns:
            ID of the saved analysis
        """
        # Generate a unique ID if not provided
        analysis_id = str(uuid.uuid4())
        
        # Create timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Generate default name if not provided
        if not name:
            name = f"Analysis {timestamp}"
        
        # Create metadata
        metadata = {
            "id": analysis_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "timestamp": timestamp,
            "data_type": str(type(data)),
            "file_path": f"{analysis_id}.pkl"
        }
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, f"{analysis_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save data using pickle (handles pandas DataFrames and other complex objects)
        data_path = os.path.join(self.data_dir, f"{analysis_id}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
            
        return analysis_id
    
    def load_analysis(self, analysis_id) -> Dict[str, Any]:
        """
        Loads a saved analysis.
        
        Args:
            analysis_id: ID of the analysis to load
            
        Returns:
            Dict with 'metadata' and 'data' keys
        """
        # Load metadata
        metadata_path = os.path.join(self.metadata_dir, f"{analysis_id}.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Analysis with ID {analysis_id} not found")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Load data
        data_path = os.path.join(self.data_dir, metadata["file_path"])
        if not os.path.exists(data_path):
            raise ValueError(f"Data file for analysis {analysis_id} not found")
            
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        return {
            "metadata": metadata,
            "data": data
        }
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """
        Gets metadata for all saved analyses.
        
        Returns:
            List of analysis metadata dictionaries
        """
        analyses = []
        
        for file in os.listdir(self.metadata_dir):
            if file.endswith('.json'):
                file_path = os.path.join(self.metadata_dir, file)
                
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        analyses.append(metadata)
                except:
                    # Skip invalid metadata files
                    continue
                    
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return analyses
    
    def delete_analysis(self, analysis_id) -> bool:
        """
        Deletes a saved analysis.
        
        Args:
            analysis_id: ID of the analysis to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Check if metadata file exists
        metadata_path = os.path.join(self.metadata_dir, f"{analysis_id}.json")
        if not os.path.exists(metadata_path):
            return False
            
        # Load metadata to get data file path
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Get data file path
            data_path = os.path.join(self.data_dir, metadata.get("file_path", f"{analysis_id}.pkl"))
            
            # Delete files
            os.remove(metadata_path)
            if os.path.exists(data_path):
                os.remove(data_path)
                
            return True
        except:
            return False
    
    def search_analyses(self, query=None, tags=None, date_from=None, date_to=None) -> List[Dict[str, Any]]:
        """
        Searches saved analyses by query, tags, or date range.
        
        Args:
            query: Optional search query for name/description
            tags: Optional list of tags to filter by
            date_from: Optional start date for filtering
            date_to: Optional end date for filtering
            
        Returns:
            List of matching analysis metadata
        """
        # Get all analyses
        all_analyses = self.get_all_analyses()
        
        # Apply filters
        filtered_analyses = all_analyses
        
        # Filter by query
        if query:
            query = query.lower()
            filtered_analyses = [
                a for a in filtered_analyses
                if query in a.get('name', '').lower() or query in a.get('description', '').lower()
            ]
        
        # Filter by tags
        if tags:
            filtered_analyses = [
                a for a in filtered_analyses
                if all(tag in a.get('tags', []) for tag in tags)
            ]
            
        # Filter by date range
        if date_from or date_to:
            try:
                # Parse date strings to datetime objects
                if date_from:
                    if isinstance(date_from, str):
                        date_from = datetime.datetime.fromisoformat(date_from)
                    # Convert to ISO format string for comparison
                    date_from = date_from.isoformat()
                
                if date_to:
                    if isinstance(date_to, str):
                        date_to = datetime.datetime.fromisoformat(date_to)
                    # Convert to ISO format string for comparison
                    date_to = date_to.isoformat()
                
                # Filter by date range
                if date_from:
                    filtered_analyses = [
                        a for a in filtered_analyses
                        if a.get('timestamp', '') >= date_from
                    ]
                
                if date_to:
                    filtered_analyses = [
                        a for a in filtered_analyses
                        if a.get('timestamp', '') <= date_to
                    ]
            except:
                # If date parsing fails, skip date filtering
                pass
        
        return filtered_analyses 
import time
import uuid
import json
import pandas as pd
import io
from typing import Dict, List, Callable, Optional, Any, Union
from datetime import datetime, timedelta
from collections import deque
import threading
import os


class RealTimeDataConnector:
    """
    Real-time data connector for streaming sentiment analysis results
    to the analytics dashboard with pub/sub pattern and automatic aggregation.
    """
    
    def __init__(self, max_queue_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the real-time data connector.
        
        Args:
            max_queue_size: Maximum number of items to keep in the queue
            ttl_seconds: Time-to-live in seconds for data items
        """
        self.max_queue_size = max_queue_size
        self.ttl_seconds = ttl_seconds
        self.data_queue = deque(maxlen=max_queue_size)
        self.timestamps = deque(maxlen=max_queue_size)
        self.subscribers = {}  # Maps subscription_id to callback function
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.session_state = None  # Will be set when attached to a session
        self._last_aggregation_time = time.time()
        self._aggregated_data = None
        self._aggregation_interval = 1.0  # Minimum seconds between aggregations
    
    def push(self, result: Dict[str, Any]) -> None:
        """
        Adds a new analysis result to the queue and notifies subscribers.
        
        Args:
            result: Sentiment analysis result to add
        """
        with self.lock:
            # Add timestamp to result if not present
            if 'timestamp' not in result:
                result['timestamp'] = datetime.now().isoformat()
            
            # Add to queue
            self.data_queue.append(result)
            self.timestamps.append(time.time())
            
            # Clean up expired items
            self._cleanup_expired()
            
            # Update aggregated data if needed
            current_time = time.time()
            if current_time - self._last_aggregation_time >= self._aggregation_interval:
                self._update_aggregated_data()
                self._last_aggregation_time = current_time
            
            # Update analytics data in session if attached
            if self.session_state is not None:
                self.update_analytics_data()
            
            # Notify subscribers
            for callback in self.subscribers.values():
                try:
                    callback(result)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Registers a callback for data updates.
        
        Args:
            callback: Function to call when new data is available
            
        Returns:
            Subscription ID
        """
        with self.lock:
            subscription_id = str(uuid.uuid4())
            self.subscribers[subscription_id] = callback
            return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> None:
        """
        Removes a callback subscription.
        
        Args:
            subscription_id: ID of subscription to remove
        """
        with self.lock:
            if subscription_id in self.subscribers:
                del self.subscribers[subscription_id]
    
    def get_latest_data(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns the latest n data items.
        
        Args:
            n: Number of items to return, None for all
            
        Returns:
            List of latest sentiment analysis results
        """
        with self.lock:
            self._cleanup_expired()
            
            if n is None or n >= len(self.data_queue):
                return list(self.data_queue)
            else:
                return list(self.data_queue)[-n:]
    
    def get_aggregated_data(self) -> Dict[str, Any]:
        """
        Returns aggregated statistics for the data in the queue.
        
        Returns:
            Dict with aggregated sentiment statistics
        """
        with self.lock:
            # If we have cached aggregation and it's recent enough, use it
            current_time = time.time()
            if self._aggregated_data is not None and current_time - self._last_aggregation_time < self._aggregation_interval:
                result = self._aggregated_data
            else:
                # Otherwise, compute new aggregation
                result = self._update_aggregated_data()
            
            # Safety check: ensure we always return a dictionary
            if not isinstance(result, dict):
                print(f"WARNING: get_aggregated_data returned {type(result)}, expected dict. Value: {result}")
                return {
                    'count': 0,
                    'sentiment_stats': {},
                    'sentiment_labels': {},
                    'emotion_labels': {},
                    'recent_stats': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            return result
    
    def clear(self) -> None:
        """Clears the data queue."""
        with self.lock:
            self.data_queue.clear()
            self.timestamps.clear()
            self._aggregated_data = None
            
            # Update session if attached
            if self.session_state is not None:
                self.update_analytics_data()
    
    def export_data(self, format: str = 'json') -> io.BytesIO:
        """
        Exports data to specified format.
        
        Args:
            format: Export format (json, csv, excel)
            
        Returns:
            BytesIO containing the exported data
        """
        with self.lock:
            data = list(self.data_queue)
            
            # Convert to DataFrame for easier export
            df = self._convert_to_dataframe(data)
            
            output = io.BytesIO()
            
            if format.lower() == 'json':
                output.write(json.dumps(data, indent=2).encode('utf-8'))
            elif format.lower() == 'csv':
                df.to_csv(output, index=False)
            elif format.lower() == 'excel':
                df.to_excel(output, index=False, engine='xlsxwriter')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            output.seek(0)
            return output
    
    def attach_to_session(self, session_state) -> None:
        """
        Attaches to Streamlit session state for persistence.
        
        Args:
            session_state: Streamlit session state object
        """
        self.session_state = session_state
        
        # Initialize analytics_data if not present
        if not hasattr(session_state, 'analytics_data') or session_state.analytics_data is None:
            self.update_analytics_data()
        
        # Try to restore data if available
        self.restore()
    
    def update_analytics_data(self) -> None:
        """
        Updates analytics_data in session state from the data queue.
        This allows the analytics dashboard to use the latest data.
        """
        if self.session_state is None:
            return
            
        with self.lock:
            data = list(self.data_queue)
            
            if not data:
                # Don't update if no data available
                return
                
            # Convert to DataFrame
            df = self._convert_to_dataframe(data)
            
            # Store in session state
            self.session_state.analytics_data = df
    
    def persist(self) -> None:
        """
        Persists current data for future sessions.
        Uses a simple file-based approach.
        """
        with self.lock:
            data = list(self.data_queue)
            
            if not data:
                return  # Nothing to persist
            
            try:
                # Create tmp directory if it doesn't exist
                os.makedirs("tmp", exist_ok=True)
                
                # Save to file
                with open("tmp/real_time_data.json", "w") as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Error persisting data: {e}")
    
    def restore(self) -> None:
        """
        Restores persisted data from previous sessions.
        """
        try:
            if os.path.exists("tmp/real_time_data.json"):
                with open("tmp/real_time_data.json", "r") as f:
                    data = json.load(f)
                
                # Clear existing data
                self.data_queue.clear()
                self.timestamps.clear()
                
                # Add loaded data
                for item in data:
                    self.data_queue.append(item)
                    self.timestamps.append(time.time())  # Use current time
                
                # Update aggregation
                self._update_aggregated_data()
                
                # Update analytics data in session if attached
                if self.session_state is not None:
                    self.update_analytics_data()
        except Exception as e:
            print(f"Error restoring data: {e}")
    
    def _cleanup_expired(self) -> None:
        """
        Removes expired items from the queue based on TTL.
        """
        if not self.timestamps:
            return
            
        current_time = time.time()
        
        # Find the index of the first non-expired item
        expired_count = 0
        for ts in self.timestamps:
            if current_time - ts > self.ttl_seconds:
                expired_count += 1
            else:
                break
        
        # Remove expired items
        if expired_count > 0:
            for _ in range(expired_count):
                self.data_queue.popleft()
                self.timestamps.popleft()
    
    def _update_aggregated_data(self) -> Dict[str, Any]:
        """
        Updates and returns the aggregated data statistics.
        
        Returns:
            Dict with aggregated sentiment statistics
        """
        data = list(self.data_queue)
        
        if not data:
            self._aggregated_data = {
                'count': 0,
                'last_updated': datetime.now().isoformat()
            }
            return self._aggregated_data
        
        # Extract sentiment scores
        sentiment_scores = []
        sentiment_labels = {}
        emotion_labels = {}
        
        for item in data:
            # Extract sentiment data
            if 'sentiment' in item and isinstance(item['sentiment'], dict):
                sentiment = item['sentiment']
                if 'label' in sentiment and 'score' in sentiment:
                    label = sentiment['label'].lower()
                    score = sentiment['score']
                    
                    # Convert to numeric scale: negative=-1, neutral=0, positive=1
                    if label == 'positive':
                        numeric_score = score
                    elif label == 'negative':
                        numeric_score = -score
                    else:
                        numeric_score = 0
                    
                    sentiment_scores.append(numeric_score)
                    
                    # Count label
                    sentiment_labels[label] = sentiment_labels.get(label, 0) + 1
            
            # Extract emotion data
            if 'emotion' in item and isinstance(item['emotion'], dict):
                emotion = item['emotion']
                if 'label' in emotion:
                    label = emotion['label'].lower()
                    emotion_labels[label] = emotion_labels.get(label, 0) + 1
        
        # Calculate sentiment statistics
        sentiment_stats = {}
        if sentiment_scores:
            sentiment_stats = {
                'mean': sum(sentiment_scores) / len(sentiment_scores),
                'min': min(sentiment_scores),
                'max': max(sentiment_scores),
                'count': len(sentiment_scores),
                'positive_count': sum(1 for s in sentiment_scores if s > 0.2),
                'negative_count': sum(1 for s in sentiment_scores if s < -0.2),
                'neutral_count': sum(1 for s in sentiment_scores if -0.2 <= s <= 0.2)
            }
        
        # Last 5 minutes trend
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_scores = []
        
        for item in data:
            if 'timestamp' in item:
                try:
                    timestamp = datetime.fromisoformat(item['timestamp'])
                    if timestamp >= recent_cutoff:
                        if 'sentiment' in item and isinstance(item['sentiment'], dict):
                            sentiment = item['sentiment']
                            if 'label' in sentiment and 'score' in sentiment:
                                label = sentiment['label'].lower()
                                score = sentiment['score']
                                
                                # Convert to numeric scale
                                if label == 'positive':
                                    numeric_score = score
                                elif label == 'negative':
                                    numeric_score = -score
                                else:
                                    numeric_score = 0
                                
                                recent_scores.append(numeric_score)
                except (ValueError, TypeError):
                    pass
        
        recent_stats = {}
        if recent_scores:
            recent_mean = sum(recent_scores) / len(recent_scores)
            recent_stats = {
                'mean': recent_mean,
                'count': len(recent_scores),
                'positive_ratio': sum(1 for s in recent_scores if s > 0.2) / len(recent_scores),
                'negative_ratio': sum(1 for s in recent_scores if s < -0.2) / len(recent_scores)
            }
        
        # Build aggregated data
        self._aggregated_data = {
            'count': len(data),
            'sentiment_stats': sentiment_stats,
            'sentiment_labels': sentiment_labels,
            'emotion_labels': emotion_labels,
            'recent_stats': recent_stats,
            'last_updated': datetime.now().isoformat()
        }
        
        return self._aggregated_data
    
    def _convert_to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts the list of sentiment analysis results to a DataFrame.
        
        Args:
            data: List of sentiment analysis results
            
        Returns:
            DataFrame with extracted and flattened data
        """
        if not data:
            return pd.DataFrame()
        
        # Extract relevant fields
        rows = []
        
        for item in data:
            row = {}
            
            # Extract text
            if 'text' in item:
                row['text'] = item['text']
            
            # Extract timestamp
            if 'timestamp' in item:
                try:
                    row['timestamp'] = datetime.fromisoformat(item['timestamp'])
                except (ValueError, TypeError):
                    row['timestamp'] = datetime.now()
            else:
                row['timestamp'] = datetime.now()
            
            # Extract sentiment data
            if 'sentiment' in item and isinstance(item['sentiment'], dict):
                sentiment = item['sentiment']
                if 'label' in sentiment:
                    row['sentiment_label'] = sentiment['label']
                if 'score' in sentiment:
                    row['sentiment_confidence'] = sentiment['score']
                    
                    # Calculate numeric sentiment score
                    if row.get('sentiment_label', '').lower() == 'positive':
                        row['sentiment_score'] = sentiment['score']
                    elif row.get('sentiment_label', '').lower() == 'negative':
                        row['sentiment_score'] = -sentiment['score']
                    else:
                        row['sentiment_score'] = 0
                else:
                    # Default sentiment score if no score available
                    row['sentiment_score'] = 0
            else:
                # Default values if no sentiment data
                row['sentiment_label'] = 'neutral'
                row['sentiment_confidence'] = 0.0
                row['sentiment_score'] = 0
            
            # Extract emotion data
            if 'emotion' in item and isinstance(item['emotion'], dict):
                emotion = item['emotion']
                if 'label' in emotion:
                    row['emotion_label'] = emotion['label']
                if 'score' in emotion:
                    row['emotion_score'] = emotion['score']
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Set timestamp as index if available
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        return df
    
    def detect_anomalies(self, detector=None, method='isolation_forest'):
        """
        Applies anomaly detection to the current data.
        
        Args:
            detector: Anomaly detector to use, or None to create a new one
            method: Detection method to use
            
        Returns:
            Tuple of (DataFrame with anomalies, anomaly insights)
        """
        with self.lock:
            # Get latest data
            data = list(self.data_queue)
            
            if not data:
                return None, None
                
            # Convert to DataFrame
            df = self._convert_to_dataframe(data)
            
            if len(df) < 10:
                return df, None  # Not enough data for meaningful detection
            
            # Use provided detector or create a new one
            if detector is None:
                detector = SentimentAnomalyDetector()
            
            # Apply anomaly detection
            df_with_anomalies = detector.detect_anomalies(df, method=method)
            
            # Generate insights
            anomaly_count = df_with_anomalies['anomaly'].sum()
            if anomaly_count > 0:
                insights = detector.generate_insights(df_with_anomalies)
            else:
                insights = None
                
            return df_with_anomalies, insights 
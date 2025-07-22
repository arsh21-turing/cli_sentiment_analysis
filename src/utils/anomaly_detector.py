from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
import pandas as pd
import numpy as np
import calendar

warnings.filterwarnings('ignore')


class SentimentAnomalyDetector:
    """
    Detects unusual patterns and anomalies in sentiment data, providing
    automated insights about significant deviations or trends.
    """
    
    def __init__(self, contamination=0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
        """
        self.contamination = contamination
        self.is_fitted = False
        self.isolation_forest = None
        self.scaler = None
    
    def detect_anomalies(self, data, method='isolation_forest'):
        """
        Detects anomalies in sentiment data.
        
        Args:
            data: DataFrame with sentiment data
            method: Anomaly detection method to use
                ('isolation_forest', 'zscore', 'dbscan', or 'seasonal')
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if len(data) < 5:
            return data.copy()  # Not enough data
        
        result_df = data.copy()
        
        # Extract sentiment scores
        if 'sentiment_score' in result_df.columns:
            sentiment_series = result_df['sentiment_score']
        elif 'sentiment' in result_df.columns:
            sentiment_series = result_df['sentiment']
        else:
            raise ValueError("No sentiment column found in data")
        
        # Apply the selected anomaly detection method
        if method == 'isolation_forest':
            return self._detect_with_isolation_forest(result_df, sentiment_series)
        elif method == 'zscore':
            return self._detect_with_zscore(result_df, sentiment_series)
        elif method == 'dbscan':
            return self._detect_with_dbscan(result_df)
        elif method == 'seasonal':
            return self._detect_seasonal_anomalies(result_df, sentiment_series)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
    
    def _detect_with_isolation_forest(self, data, sentiment_series):
        """
        Detects anomalies using Isolation Forest algorithm.
        
        Args:
            data: DataFrame with sentiment data
            sentiment_series: Series of sentiment scores
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Reshape data for Isolation Forest
        X = sentiment_series.values.reshape(-1, 1)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Fit and predict
        self.is_fitted = True
        anomaly_predictions = self.isolation_forest.fit_predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        # Add predictions and scores to dataframe
        data['anomaly'] = anomaly_predictions == -1  # -1 for anomalies, 1 for normal points
        data['anomaly_score'] = -anomaly_scores  # Invert scores to make anomalies positive
        
        # Add explanations for anomalies
        data['anomaly_explanation'] = data.apply(
            lambda row: self._generate_anomaly_explanation(row, data) if row['anomaly'] else "", 
            axis=1
        )
        
        return data
    
    def _detect_with_zscore(self, data, sentiment_series):
        """
        Detects anomalies using Z-score method.
        
        Args:
            data: DataFrame with sentiment data
            sentiment_series: Series of sentiment scores
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Calculate z-scores
        mean = sentiment_series.mean()
        std = sentiment_series.std()
        
        if std == 0:
            # Can't detect anomalies if there's no variation
            data['anomaly'] = False
            data['anomaly_score'] = 0
            data['anomaly_explanation'] = ""
            return data
        
        z_scores = (sentiment_series - mean) / std
        
        # Mark points with |z| > 3 as anomalies
        data['anomaly'] = abs(z_scores) > 3
        data['anomaly_score'] = abs(z_scores)
        
        # Add explanations for anomalies
        data['anomaly_explanation'] = data.apply(
            lambda row: self._generate_anomaly_explanation(row, data) if row['anomaly'] else "", 
            axis=1
        )
        
        return data
    
    def _detect_with_dbscan(self, data):
        """
        Detects anomalies using DBSCAN clustering.
        
        Args:
            data: DataFrame with sentiment data
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if len(data) < 10:
            # Not enough data for DBSCAN
            return self._detect_with_zscore(data, data['sentiment_score'])
        
        # Extract features for clustering
        features = []
        
        # Always include sentiment
        if 'sentiment_score' in data.columns:
            features.append('sentiment_score')
        elif 'sentiment' in data.columns:
            features.append('sentiment')
        
        # Include hour of day if available
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data['hour'] = data.index.hour
            features.append('hour')
            
            # Include day of week if available
            data['day_of_week'] = data.index.dayofweek
            features.append('day_of_week')
        
        # Include emotion score if available
        if 'emotion_score' in data.columns:
            features.append('emotion_score')
        
        # Check if we have enough features
        if len(features) == 0:
            # Fall back to z-score method
            return self._detect_with_zscore(data, data['sentiment_score'])
        
        # Extract and scale features
        X = data[features].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Points with cluster = -1 are anomalies
        data['anomaly'] = clusters == -1
        
        # Calculate distances to nearest cluster centers for scoring
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) == 1 and unique_clusters[0] == -1:
            # All points are outliers
            data['anomaly_score'] = 1.0
        else:
            # Calculate distance to nearest cluster center
            cluster_centers = {}
            for cluster in unique_clusters:
                if cluster != -1:  # Skip noise points
                    cluster_points = X_scaled[clusters == cluster]
                    cluster_centers[cluster] = np.mean(cluster_points, axis=0)
            
            # Calculate distance to nearest cluster center
            anomaly_scores = []
            for point in X_scaled:
                if len(cluster_centers) > 0:
                    min_distance = min(
                        np.linalg.norm(point - center) 
                        for center in cluster_centers.values()
                    )
                    anomaly_scores.append(min_distance)
                else:
                    anomaly_scores.append(1.0)  # Default if no clusters
            
            # Normalize scores to 0-1 range
            if len(anomaly_scores) > 0:
                max_score = max(anomaly_scores)
                if max_score > 0:
                    anomaly_scores = [s / max_score for s in anomaly_scores]
            
            data['anomaly_score'] = anomaly_scores
        
        # Add explanations for anomalies
        data['anomaly_explanation'] = data.apply(
            lambda row: self._generate_anomaly_explanation(row, data) if row['anomaly'] else "", 
            axis=1
        )
        
        return data
    
    def _detect_seasonal_anomalies(self, data, sentiment_series):
        """
        Detects seasonal anomalies in time series data.
        
        Args:
            data: DataFrame with sentiment data and datetime index
            sentiment_series: Series of sentiment scores
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Check if we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            # Fall back to isolation forest
            return self._detect_with_isolation_forest(data, sentiment_series)
        
        # Need at least 2 periods (days) of data
        if len(data) < 48:  # Assuming hourly data, 48 = 2 days
            # Fall back to isolation forest
            return self._detect_with_isolation_forest(data, sentiment_series)
        
        data = data.copy()
        
        try:
            # Resample to hourly data if we have enough data
            hourly = sentiment_series.resample('H').mean()
            hourly = hourly.interpolate(method='linear')
            
            # Decompose the time series
            result = seasonal_decompose(
                hourly, 
                model='additive', 
                period=24  # 24 hours for daily seasonality
            )
            
            # The residual component contains anomalies
            residual = result.resid
            residual = residual.dropna()
            
            # Calculate z-scores of residuals
            residual_mean = residual.mean()
            residual_std = residual.std()
            
            if residual_std == 0:
                # Can't detect anomalies if there's no variation
                data['anomaly'] = False
                data['anomaly_score'] = 0
                data['anomaly_explanation'] = ""
                return data
            
            residual_z = (residual - residual_mean) / residual_std
            
            # Mark hours with |z| > 3 as anomalies
            anomalous_hours = residual_z[abs(residual_z) > 3].index
            
            # Find the closest hour for each data point
            data['anomaly'] = False
            data['anomaly_score'] = 0
            
            for idx, row in data.iterrows():
                # Find closest hour in anomalous_hours
                if len(anomalous_hours) > 0:
                    closest_hour = min(
                        anomalous_hours,
                        key=lambda h: abs((idx - h).total_seconds())
                    )
                    time_diff = abs((idx - closest_hour).total_seconds())
                    
                    # If within 30 minutes of an anomalous hour
                    if time_diff <= 1800:  # 30 minutes in seconds
                        data.loc[idx, 'anomaly'] = True
                        # Score based on z-score and time proximity
                        z_score = abs(residual_z.loc[closest_hour])
                        time_weight = 1 - (time_diff / 1800)  # 1 at 0 seconds, 0 at 1800 seconds
                        data.loc[idx, 'anomaly_score'] = z_score * time_weight
            
            # Add explanations for anomalies
            data['anomaly_explanation'] = data.apply(
                lambda row: self._generate_anomaly_explanation(row, data) if row['anomaly'] else "", 
                axis=1
            )
            
            return data
            
        except Exception as e:
            # If decomposition fails, fall back to isolation forest
            print(f"Seasonal decomposition failed: {e}. Falling back to isolation forest.")
            return self._detect_with_isolation_forest(data, sentiment_series)
    
    def _generate_anomaly_explanation(self, row, data):
        """
        Generates an explanation for why a point is anomalous.
        
        Args:
            row: DataFrame row containing the anomalous point
            data: Full DataFrame for context
            
        Returns:
            String explanation of the anomaly
        """
        if not row['anomaly']:
            return ""
        
        sentiment_col = 'sentiment_score' if 'sentiment_score' in row else 'sentiment'
        sentiment = row[sentiment_col]
        score = row['anomaly_score']
        
        # Calculate average sentiment for context
        avg_sentiment = data[sentiment_col].mean()
        sentiment_diff = sentiment - avg_sentiment
        
        # Generate explanation based on severity and context
        if score > 2:
            severity = "extreme"
        elif score > 1.5:
            severity = "significant"
        else:
            severity = "unusual"
        
        # Generate explanation based on whether sentiment is unusually positive or negative
        if sentiment_diff > 0:
            explanation = f"{severity.capitalize()} positive sentiment deviation"
            if abs(sentiment_diff) > 0.5:
                explanation += f"; {abs(sentiment_diff):.2f} points above average"
        else:
            explanation = f"{severity.capitalize()} negative sentiment deviation"
            if abs(sentiment_diff) > 0.5:
                explanation += f"; {abs(sentiment_diff):.2f} points below average"
        
        # Add context about unexpected timing if we have datetime data
        if isinstance(data.index, pd.DatetimeIndex) and isinstance(row.name, pd.Timestamp):
            hour = row.name.hour
            day = row.name.dayofweek
            day_name = calendar.day_name[day]
            
            # Check if this hour/day combination typically has different sentiment
            hour_avg = data[data.index.hour == hour][sentiment_col].mean()
            day_avg = data[data.index.dayofweek == day][sentiment_col].mean()
            
            if abs(sentiment - hour_avg) > 0.4:
                explanation += f". Unusual for time of day ({hour}:00)"
            
            if abs(sentiment - day_avg) > 0.4:
                explanation += f". Unexpected for {day_name}"
        
        return explanation
    
    def generate_insights(self, data):
        """
        Generates automated insights about detected anomalies.
        
        Args:
            data: DataFrame with anomaly detection results
            
        Returns:
            Dictionary of insights
        """
        anomalies = data[data['anomaly']]
        
        if len(anomalies) == 0:
            return {
                "found_anomalies": False,
                "message": "No significant anomalies detected in the sentiment data.",
                "details": [],
                "recommendations": ["Continue monitoring sentiment trends for unusual patterns."]
            }
        
        # Group anomalies by time if possible
        anomaly_clusters = []
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Sort anomalies by time
            anomalies = anomalies.sort_index()
            
            # Group anomalies that are close in time
            current_cluster = []
            last_time = None
            
            for idx, row in anomalies.iterrows():
                if last_time is None or (idx - last_time) <= timedelta(hours=2):
                    # Add to current cluster
                    current_cluster.append((idx, row))
                else:
                    # Start a new cluster
                    if current_cluster:
                        anomaly_clusters.append(current_cluster)
                    current_cluster = [(idx, row)]
                
                last_time = idx
            
            # Add the last cluster if it exists
            if current_cluster:
                anomaly_clusters.append(current_cluster)
        else:
            # Just put all anomalies in one cluster
            anomaly_clusters.append([(idx, row) for idx, row in anomalies.iterrows()])
        
        # Generate insights for each cluster
        insights = {
            "found_anomalies": True,
            "message": f"Detected {len(anomalies)} unusual patterns in sentiment data.",
            "details": [],
            "recommendations": []
        }
        
        for i, cluster in enumerate(anomaly_clusters):
            if len(cluster) == 1:
                # Single anomaly
                time, row = cluster[0]
                sentiment_col = 'sentiment_score' if 'sentiment_score' in row else 'sentiment'
                
                if isinstance(time, pd.Timestamp):
                    time_str = time.strftime("%Y-%m-%d %H:%M")
                    detail = f"Anomaly at {time_str}: {row['anomaly_explanation']}"
                else:
                    detail = f"Anomaly at point {time}: {row['anomaly_explanation']}"
                
                insights["details"].append(detail)
            else:
                # Cluster of anomalies
                start_time, _ = cluster[0]
                end_time, _ = cluster[-1]
                
                if isinstance(start_time, pd.Timestamp):
                    start_str = start_time.strftime("%Y-%m-%d %H:%M")
                    end_str = end_time.strftime("%Y-%m-%d %H:%M")
                    duration = end_time - start_time
                    hours = duration.total_seconds() / 3600
                    
                    detail = f"Cluster of {len(cluster)} anomalies from {start_str} to {end_str} ({hours:.1f} hours)"
                else:
                    detail = f"Cluster of {len(cluster)} anomalies from point {start_time} to {end_time}"
                
                insights["details"].append(detail)
                
                # Add sample explanations
                sample_explanations = set()
                for _, row in cluster[:3]:  # Take first 3 as samples
                    if row['anomaly_explanation']:
                        sample_explanations.add(row['anomaly_explanation'].split('.')[0])  # First sentence
                
                if sample_explanations:
                    insights["details"].append("  Examples: " + "; ".join(sample_explanations))
        
        # Generate recommendations based on anomalies
        sentiment_col = 'sentiment_score' if 'sentiment_score' in data.columns else 'sentiment'
        avg_anomaly_sentiment = anomalies[sentiment_col].mean()
        avg_normal_sentiment = data[~data['anomaly']][sentiment_col].mean()
        
        if avg_anomaly_sentiment < avg_normal_sentiment - 0.3:
            insights["recommendations"].append(
                "Investigate potential negative sentiment factors during anomalous periods."
            )
        elif avg_anomaly_sentiment > avg_normal_sentiment + 0.3:
            insights["recommendations"].append(
                "Analyze unusual positive sentiment spikes for potential insights."
            )
        
        # Time-based recommendations
        if isinstance(data.index, pd.DatetimeIndex) and len(anomaly_clusters) > 1:
            # Check if anomalies occur at specific times
            anomaly_hours = [idx.hour for cluster in anomaly_clusters for idx, _ in cluster]
            hour_counts = pd.Series(anomaly_hours).value_counts()
            
            # If there's a pattern in hours
            if max(hour_counts) > len(anomaly_clusters) / 3:
                common_hours = hour_counts[hour_counts > 1].index.tolist()
                if common_hours:
                    hours_str = ", ".join([f"{h}:00" for h in common_hours])
                    insights["recommendations"].append(
                        f"Monitor sentiment especially during hours: {hours_str} when anomalies are more common."
                    )
        
        # Add general recommendations
        if len(anomaly_clusters) > 3:
            insights["recommendations"].append(
                "Consider adjusting anomaly detection sensitivity if too many anomalies are being detected."
            )
        
        # If all anomalies are in the past 24 hours
        recent_anomalies = 0
        if isinstance(data.index, pd.DatetimeIndex):
            cutoff = data.index.max() - timedelta(hours=24)
            recent_anomalies = sum(1 for cluster in anomaly_clusters for idx, _ in cluster if idx > cutoff)
            
            if recent_anomalies > 0:
                insights["recommendations"].append(
                    f"Pay attention to {recent_anomalies} recent anomalies in the past 24 hours."
                )
        
        return insights 
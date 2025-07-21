"""
Advanced analytics and trends utilities for sentiment analysis.
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats


class SentimentAnalytics:
    """
    Analytics utilities for sentiment analysis.
    """
    
    def __init__(self):
        """
        Initialize the analytics utilities.
        """
        # Supported emotions for analysis
        self.supported_emotions = [
            "joy", "sadness", "anger", "fear", 
            "surprise", "disgust", "trust", "anticipation"
        ]
        
        # Supported sentiments for analysis
        self.supported_sentiments = ["positive", "negative", "neutral"]
    
    def analyze_trends(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.
        
        Args:
            data (DataFrame): Data containing sentiment analysis results
            time_column (str): Column name containing time information
            
        Returns:
            Dict: Trend analysis results
        """
        # Check if data is valid
        if data is None or len(data) == 0:
            return {"error": "No data provided for trend analysis"}
        
        # Ensure time column exists
        if time_column not in data.columns:
            return {"error": f"Time column '{time_column}' not found in data"}
        
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(data[time_column]):
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except Exception as e:
                return {"error": f"Error converting time column to datetime: {str(e)}"}
        
        # Initialize results
        results = {
            "time_range": {
                "start": data[time_column].min(),
                "end": data[time_column].max(),
                "duration": (data[time_column].max() - data[time_column].min()).total_seconds()
            },
            "sample_size": len(data),
            "trends": {},
            "seasonal_patterns": {},
            "correlations": {}
        }
        
        # Group by time periods and analyze trends
        try:
            # Daily trends
            daily_data = self._aggregate_by_period(data, time_column, 'D')
            if daily_data is not None:
                results["trends"]["daily"] = self._calculate_trend_metrics(daily_data)
            
            # Weekly trends
            weekly_data = self._aggregate_by_period(data, time_column, 'W')
            if weekly_data is not None:
                results["trends"]["weekly"] = self._calculate_trend_metrics(weekly_data)
            
            # Monthly trends
            monthly_data = self._aggregate_by_period(data, time_column, 'ME')
            if monthly_data is not None:
                results["trends"]["monthly"] = self._calculate_trend_metrics(monthly_data)
            
            # Analyze seasonal patterns
            results["seasonal_patterns"] = self._analyze_seasonality(data, time_column)
            
            # Analyze correlations
            results["correlations"] = self._analyze_correlations(data)
            
        except Exception as e:
            results["error"] = f"Error analyzing trends: {str(e)}"
        
        return results
    
    def _aggregate_by_period(self, data: pd.DataFrame, time_column: str, period: str) -> Optional[pd.DataFrame]:
        """
        Aggregate data by time period.
        
        Args:
            data (DataFrame): Input data
            time_column (str): Time column name
            period (str): Time period ('D', 'W', 'M')
            
        Returns:
            DataFrame: Aggregated data
        """
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Set time column as index
            df.set_index(time_column, inplace=True)
            
            # Resample by period
            resampled = df.resample(period)
            
            # Calculate aggregations
            aggregated = pd.DataFrame({
                'count': resampled.size(),
                'positive_count': resampled['sentiment'].apply(lambda x: (x == 'positive').sum() if 'sentiment' in df.columns else 0),
                'negative_count': resampled['sentiment'].apply(lambda x: (x == 'negative').sum() if 'sentiment' in df.columns else 0),
                'neutral_count': resampled['sentiment'].apply(lambda x: (x == 'neutral').sum() if 'sentiment' in df.columns else 0),
                'avg_confidence': resampled['confidence'].mean() if 'confidence' in df.columns else None,
                'avg_emotion_score': resampled['emotion_score'].mean() if 'emotion_score' in df.columns else None
            })
            
            # Calculate percentages
            total = aggregated['count']
            aggregated['positive_pct'] = (aggregated['positive_count'] / total * 100).fillna(0)
            aggregated['negative_pct'] = (aggregated['negative_count'] / total * 100).fillna(0)
            aggregated['neutral_pct'] = (aggregated['neutral_count'] / total * 100).fillna(0)
            
            return aggregated.dropna()
            
        except Exception as e:
            print(f"Error aggregating by period {period}: {str(e)}")
            return None
    
    def _calculate_trend_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trend metrics for aggregated data.
        
        Args:
            data (DataFrame): Aggregated data
            
        Returns:
            Dict: Trend metrics
        """
        if len(data) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        metrics = {}
        
        # Calculate linear trends
        x = np.arange(len(data))
        
        # Sentiment trends
        for sentiment in ['positive_pct', 'negative_pct', 'neutral_pct']:
            if sentiment in data.columns:
                y = data[sentiment].values
                if len(y) > 1 and not np.all(np.isnan(y)):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    metrics[f"{sentiment}_trend"] = {
                        "slope": slope,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                    }
        
        # Confidence trend
        if 'avg_confidence' in data.columns:
            y = data['avg_confidence'].values
            if len(y) > 1 and not np.all(np.isnan(y)):
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                metrics["confidence_trend"] = {
                    "slope": slope,
                    "r_squared": r_value ** 2,
                    "p_value": p_value,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                }
        
        # Volatility (standard deviation)
        for col in ['positive_pct', 'negative_pct', 'neutral_pct', 'avg_confidence']:
            if col in data.columns:
                metrics[f"{col}_volatility"] = data[col].std()
        
        return metrics
    
    def _analyze_seasonality(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in the data.
        
        Args:
            data (DataFrame): Input data
            time_column (str): Time column name
            
        Returns:
            Dict: Seasonality analysis
        """
        results = {}
        
        try:
            # Extract time components
            data_copy = data.copy()
            data_copy['hour'] = data_copy[time_column].dt.hour
            data_copy['day_of_week'] = data_copy[time_column].dt.dayofweek
            data_copy['month'] = data_copy[time_column].dt.month
            
            # Hourly patterns
            if 'sentiment' in data_copy.columns:
                hourly_sentiment = data_copy.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
                results['hourly_patterns'] = hourly_sentiment.to_dict()
            
            # Daily patterns
            if 'sentiment' in data_copy.columns:
                daily_sentiment = data_copy.groupby(['day_of_week', 'sentiment']).size().unstack(fill_value=0)
                results['daily_patterns'] = daily_sentiment.to_dict()
            
            # Monthly patterns
            if 'sentiment' in data_copy.columns:
                monthly_sentiment = data_copy.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                results['monthly_patterns'] = monthly_sentiment.to_dict()
            
        except Exception as e:
            results['error'] = f"Error analyzing seasonality: {str(e)}"
        
        return results
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between different variables.
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            Dict: Correlation analysis
        """
        results = {}
        
        try:
            # Select numeric columns for correlation analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                correlation_matrix = data[numeric_cols].corr()
                results['correlation_matrix'] = correlation_matrix.to_dict()
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr = correlation_matrix.iloc[i, j]
                        if abs(corr) > 0.7:  # Strong correlation threshold
                            strong_correlations.append({
                                'variable1': numeric_cols[i],
                                'variable2': numeric_cols[j],
                                'correlation': corr
                            })
                
                results['strong_correlations'] = strong_correlations
            
        except Exception as e:
            results['error'] = f"Error analyzing correlations: {str(e)}"
        
        return results
    
    def perform_clustering(self, data: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Perform clustering analysis on sentiment data.
        
        Args:
            data (DataFrame): Input data
            n_clusters (int): Number of clusters
            
        Returns:
            Dict: Clustering results
        """
        results = {
            "n_clusters": n_clusters,
            "cluster_centers": {},
            "cluster_sizes": {},
            "cluster_characteristics": {}
        }
        
        try:
            # Prepare features for clustering
            features = []
            feature_names = []
            
            # Add sentiment features if available
            if 'sentiment' in data.columns:
                sentiment_dummies = pd.get_dummies(data['sentiment'], prefix='sentiment')
                features.append(sentiment_dummies)
                feature_names.extend(sentiment_dummies.columns.tolist())
            
            # Add emotion features if available
            emotion_cols = [col for col in data.columns if col in self.supported_emotions]
            if emotion_cols:
                features.append(data[emotion_cols])
                feature_names.extend(emotion_cols)
            
            # Add confidence if available
            if 'confidence' in data.columns:
                features.append(data[['confidence']])
                feature_names.append('confidence')
            
            if not features:
                return {"error": "No suitable features found for clustering"}
            
            # Combine features
            X = pd.concat(features, axis=1)
            
            # Handle missing values
            X = X.fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to original data
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = cluster_labels
            
            # Analyze clusters
            for cluster_id in range(n_clusters):
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                
                results["cluster_sizes"][f"cluster_{cluster_id}"] = len(cluster_data)
                
                # Calculate cluster characteristics
                characteristics = {}
                
                # Sentiment distribution
                if 'sentiment' in cluster_data.columns:
                    sentiment_dist = cluster_data['sentiment'].value_counts(normalize=True)
                    characteristics['sentiment_distribution'] = sentiment_dist.to_dict()
                
                # Average confidence
                if 'confidence' in cluster_data.columns:
                    characteristics['avg_confidence'] = cluster_data['confidence'].mean()
                
                # Emotion characteristics
                for emotion in self.supported_emotions:
                    if emotion in cluster_data.columns:
                        characteristics[f'avg_{emotion}'] = cluster_data[emotion].mean()
                
                results["cluster_characteristics"][f"cluster_{cluster_id}"] = characteristics
            
            # Store cluster centers in original feature space
            cluster_centers_scaled = kmeans.cluster_centers_
            cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
            
            for i in range(n_clusters):
                results["cluster_centers"][f"cluster_{i}"] = dict(zip(feature_names, cluster_centers_original[i]))
            
            results["data_with_clusters"] = data_with_clusters
            
        except Exception as e:
            results["error"] = f"Error performing clustering: {str(e)}"
        
        return results
    
    def generate_analytics_charts(self, data: pd.DataFrame, analytics_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Generate analytics charts for visualization.
        
        Args:
            data (DataFrame): Input data
            analytics_results (Dict): Results from analytics functions
            
        Returns:
            Dict: Dictionary of Plotly figures
        """
        charts = {}
        
        try:
            # Time series chart
            if 'trends' in analytics_results and 'daily' in analytics_results['trends']:
                charts['time_series'] = self._create_time_series_chart(data)
            
            # Sentiment distribution chart
            if 'sentiment' in data.columns:
                charts['sentiment_distribution'] = self._create_sentiment_distribution_chart(data)
            
            # Emotion heatmap
            emotion_cols = [col for col in data.columns if col in self.supported_emotions]
            if emotion_cols:
                charts['emotion_heatmap'] = self._create_emotion_heatmap(data, emotion_cols)
            
            # Confidence distribution
            if 'confidence' in data.columns:
                charts['confidence_distribution'] = self._create_confidence_chart(data)
            
            # Seasonal patterns
            if 'seasonal_patterns' in analytics_results:
                charts['seasonal_patterns'] = self._create_seasonal_chart(analytics_results['seasonal_patterns'])
            
            # Clustering visualization
            if 'data_with_clusters' in analytics_results:
                charts['clustering'] = self._create_clustering_chart(analytics_results['data_with_clusters'])
            
        except Exception as e:
            charts['error'] = f"Error generating charts: {str(e)}"
        
        return charts
    
    def _create_time_series_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create time series chart for sentiment trends."""
        fig = go.Figure()
        
        # Add sentiment lines if available
        if 'sentiment' in data.columns and 'timestamp' in data.columns:
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_data = data[data['sentiment'] == sentiment]
                if len(sentiment_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=sentiment_data['timestamp'],
                        y=sentiment_data['confidence'] if 'confidence' in sentiment_data.columns else [1] * len(sentiment_data),
                        mode='lines+markers',
                        name=f'{sentiment.capitalize()}',
                        line=dict(width=2)
                    ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_sentiment_distribution_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create sentiment distribution chart."""
        sentiment_counts = data['sentiment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.3
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            showlegend=True
        )
        
        return fig
    
    def _create_emotion_heatmap(self, data: pd.DataFrame, emotion_cols: List[str]) -> go.Figure:
        """Create emotion heatmap."""
        emotion_data = data[emotion_cols].mean()
        
        fig = go.Figure(data=go.Heatmap(
            z=[emotion_data.values],
            x=emotion_data.index,
            y=['Average Emotion Scores'],
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Emotion Analysis Heatmap",
            xaxis_title="Emotions",
            yaxis_title=""
        )
        
        return fig
    
    def _create_confidence_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create confidence distribution chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['confidence'],
            nbinsx=20,
            name='Confidence Distribution'
        ))
        
        fig.update_layout(
            title="Confidence Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def _create_seasonal_chart(self, seasonal_data: Dict[str, Any]) -> go.Figure:
        """Create seasonal patterns chart."""
        fig = go.Figure()
        
        if 'daily_patterns' in seasonal_data:
            daily_data = seasonal_data['daily_patterns']
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in daily_data:
                    values = [daily_data[sentiment].get(i, 0) for i in range(7)]
                    fig.add_trace(go.Bar(
                        x=days,
                        y=values,
                        name=f'{sentiment.capitalize()}'
                    ))
        
        fig.update_layout(
            title="Seasonal Patterns",
            xaxis_title="Day of Week",
            yaxis_title="Count",
            barmode='group'
        )
        
        return fig
    
    def _create_clustering_chart(self, data_with_clusters: pd.DataFrame) -> go.Figure:
        """Create clustering visualization."""
        if 'cluster' not in data_with_clusters.columns:
            return go.Figure()
        
        # Use PCA for dimensionality reduction if we have many features
        numeric_cols = data_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'cluster']
        
        if len(numeric_cols) >= 2:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(data_with_clusters[numeric_cols])
            
            fig = go.Figure()
            
            for cluster_id in data_with_clusters['cluster'].unique():
                cluster_points = X_pca[data_with_clusters['cluster'] == cluster_id]
                fig.add_trace(go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Clustering Visualization (PCA)",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2"
            )
            
            return fig
        
        return go.Figure() 
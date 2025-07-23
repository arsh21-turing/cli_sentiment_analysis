import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime
import scipy.stats as stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re
import warnings


class SentimentAnalytics:
    """
    Advanced analytics utilities for sentiment and emotion data analysis.
    Provides trend analysis, clustering, topic modeling, statistical testing,
    and insight generation for sentiment data.
    """
    
    def __init__(self, transformer=None, preprocessor=None):
        """
        Initialize SentimentAnalytics with optional transformer and preprocessor.
        
        Args:
            transformer: Optional transformer model for sentiment analysis
            preprocessor: Optional text preprocessor for data cleaning
        """
        self.transformer = transformer
        self.preprocessor = preprocessor
        
        # Set up default vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            min_df=2,
            max_df=0.85
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            min_df=2,
            max_df=0.85
        )
        
        # Initialize warnings filter
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def analyze_trends(self, data: Union[pd.DataFrame, List], 
                       time_column: Optional[str] = None, 
                       text_column: str = 'text',
                       window_size: int = 1) -> pd.DataFrame:
        """
        Analyzes sentiment trends over time or document sequence.
        
        Args:
            data: DataFrame or list containing text data and optional timestamps
            time_column: Column containing timestamps (if None, uses sequential order)
            text_column: Column containing text data to analyze
            window_size: Size of sliding window for trend smoothing
            
        Returns:
            DataFrame with trend analysis results
        """
        # Convert list to DataFrame if needed
        if isinstance(data, list):
            if isinstance(data[0], str):
                # List of strings
                df = pd.DataFrame({text_column: data})
            else:
                # List of dictionaries/objects
                df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Analyze sentiment if transformer is provided and sentiment not in data
        if self.transformer and 'sentiment' not in df.columns:
            sentiments = []
            for text in df[text_column]:
                try:
                    result = self.transformer.analyze(text)
                    sentiments.append(result.get('sentiment', 'neutral'))
                except:
                    sentiments.append('neutral')
            df['sentiment'] = sentiments
        
        # Create time index if not provided
        if time_column is None:
            df['time_index'] = range(len(df))
            time_column = 'time_index'
        
        # Convert sentiment to numeric for analysis
        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        df['sentiment_numeric'] = df['sentiment'].map(sentiment_map).fillna(0)
        
        # Calculate rolling statistics
        df['sentiment_rolling_mean'] = df['sentiment_numeric'].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        
        df['sentiment_rolling_std'] = df['sentiment_numeric'].rolling(
            window=window_size, center=True, min_periods=1
        ).std()
        
        # Detect trend changes
        df['trend_change'] = df['sentiment_rolling_mean'].diff()
        
        # Find peaks and valleys
        peaks, _ = find_peaks(df['sentiment_rolling_mean'].fillna(0), height=0.1)
        valleys, _ = find_peaks(-df['sentiment_rolling_mean'].fillna(0), height=0.1)
        
        df['is_peak'] = False
        df['is_valley'] = False
        df.loc[peaks, 'is_peak'] = True
        df.loc[valleys, 'is_valley'] = True
        
        return df
    
    def cluster_sentiments(self, data: Union[pd.DataFrame, List], 
                          text_column: str = 'text',
                          n_clusters: int = 3,
                          method: str = 'kmeans') -> Dict[str, Any]:
        """
        Clusters sentiment data to identify patterns and groups.
        
        Args:
            data: DataFrame or list containing text data
            text_column: Column containing text data
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans', 'dbscan', 'tsne')
            
        Returns:
            Dictionary containing clustering results
        """
        # Prepare data
        if isinstance(data, list):
            if isinstance(data[0], str):
                df = pd.DataFrame({text_column: data})
            else:
                df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Vectorize text
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[text_column])
        except:
            # Fallback to count vectorizer if TF-IDF fails
            tfidf_matrix = self.count_vectorizer.fit_transform(df[text_column])
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = clusterer.fit_predict(tfidf_matrix)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            clusters = clusterer.fit_predict(tfidf_matrix)
            
        elif method == 'tsne':
            # Use t-SNE for dimensionality reduction then K-means
            tsne = TSNE(n_components=2, random_state=42)
            tsne_features = tsne.fit_transform(tfidf_matrix.toarray())
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = clusterer.fit_predict(tsne_features)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Add cluster labels to dataframe
        df['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'sample_texts': cluster_data[text_column].head(5).tolist(),
                'avg_sentiment': cluster_data.get('sentiment_numeric', 0).mean() if 'sentiment_numeric' in cluster_data.columns else 0
            }
        
        return {
            'clusters': clusters,
            'cluster_analysis': cluster_analysis,
            'method': method,
            'n_clusters': n_clusters
        }
    
    def topic_modeling(self, data: Union[pd.DataFrame, List], 
                       text_column: str = 'text',
                       n_topics: int = 5,
                       method: str = 'lda') -> Dict[str, Any]:
        """
        Performs topic modeling on sentiment data.
        
        Args:
            data: DataFrame or list containing text data
            text_column: Column containing text data
            n_topics: Number of topics to extract
            method: Topic modeling method ('lda', 'nmf')
            
        Returns:
            Dictionary containing topic modeling results
        """
        # Prepare data
        if isinstance(data, list):
            if isinstance(data[0], str):
                df = pd.DataFrame({text_column: data})
            else:
                df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Vectorize text
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[text_column])
        except:
            tfidf_matrix = self.count_vectorizer.fit_transform(df[text_column])
        
        # Perform topic modeling
        if method == 'lda':
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
        elif method == 'nmf':
            model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )
        else:
            raise ValueError(f"Unknown topic modeling method: {method}")
        
        # Fit model
        topic_matrix = model.fit_transform(tfidf_matrix)
        
        # Extract topics
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weights': topic[top_words_idx].tolist()
            })
        
        # Assign dominant topics to documents
        dominant_topics = topic_matrix.argmax(axis=1)
        df['dominant_topic'] = dominant_topics
        
        return {
            'topics': topics,
            'topic_matrix': topic_matrix,
            'dominant_topics': dominant_topics,
            'method': method,
            'n_topics': n_topics
        }
    
    def statistical_analysis(self, data: Union[pd.DataFrame, List],
                           sentiment_column: str = 'sentiment',
                           group_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs statistical analysis on sentiment data.
        
        Args:
            data: DataFrame or list containing sentiment data
            sentiment_column: Column containing sentiment labels
            group_column: Optional column for group-wise analysis
            
        Returns:
            Dictionary containing statistical analysis results
        """
        # Prepare data
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Convert sentiment to numeric if needed
        if sentiment_column in df.columns:
            sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            df['sentiment_numeric'] = df[sentiment_column].map(sentiment_map).fillna(0)
        
        results = {}
        
        # Basic statistics
        if 'sentiment_numeric' in df.columns:
            results['basic_stats'] = {
                'mean': df['sentiment_numeric'].mean(),
                'std': df['sentiment_numeric'].std(),
                'median': df['sentiment_numeric'].median(),
                'min': df['sentiment_numeric'].min(),
                'max': df['sentiment_numeric'].max(),
                'count': len(df)
            }
        
        # Sentiment distribution
        if sentiment_column in df.columns:
            sentiment_counts = df[sentiment_column].value_counts()
            results['sentiment_distribution'] = sentiment_counts.to_dict()
            results['sentiment_percentages'] = (sentiment_counts / len(df) * 100).to_dict()
        
        # Group-wise analysis
        if group_column and group_column in df.columns:
            group_stats = {}
            for group in df[group_column].unique():
                group_data = df[df[group_column] == group]
                if 'sentiment_numeric' in group_data.columns:
                    group_stats[group] = {
                        'mean': group_data['sentiment_numeric'].mean(),
                        'std': group_data['sentiment_numeric'].std(),
                        'count': len(group_data)
                    }
            results['group_analysis'] = group_stats
        
        # Statistical tests
        if group_column and group_column in df.columns and 'sentiment_numeric' in df.columns:
            groups = df[group_column].unique()
            if len(groups) == 2:
                group1_data = df[df[group_column] == groups[0]]['sentiment_numeric']
                group2_data = df[df[group_column] == groups[1]]['sentiment_numeric']
                
                # T-test
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                results['t_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return results
    
    def detect_anomalies(self, data: Union[pd.DataFrame, List],
                        sentiment_column: str = 'sentiment',
                        contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detects anomalies in sentiment data.
        
        Args:
            data: DataFrame or list containing sentiment data
            sentiment_column: Column containing sentiment labels
            contamination: Expected proportion of anomalies
            
        Returns:
            Dictionary containing anomaly detection results
        """
        # Prepare data
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Convert sentiment to numeric
        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        df['sentiment_numeric'] = df[sentiment_column].map(sentiment_map).fillna(0)
        
        # Prepare features for anomaly detection
        features = df[['sentiment_numeric']].values
        
        # Detect anomalies using Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(features)
        
        # Mark anomalies (1 for normal, -1 for anomaly)
        df['is_anomaly'] = anomalies == -1
        
        # Get anomaly details
        anomaly_data = df[df['is_anomaly']]
        
        return {
            'anomalies': anomaly_data.to_dict('records'),
            'anomaly_count': len(anomaly_data),
            'total_count': len(df),
            'anomaly_percentage': len(anomaly_data) / len(df) * 100,
            'anomaly_indices': df[df['is_anomaly']].index.tolist()
        }
    
    def generate_insights(self, data: Union[pd.DataFrame, List],
                         text_column: str = 'text',
                         sentiment_column: str = 'sentiment') -> List[str]:
        """
        Generates insights from sentiment analysis data.
        
        Args:
            data: DataFrame or list containing sentiment data
            text_column: Column containing text data
            sentiment_column: Column containing sentiment labels
            
        Returns:
            List of insight strings
        """
        # Prepare data
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        insights = []
        
        # Basic insights
        if sentiment_column in df.columns:
            sentiment_counts = df[sentiment_column].value_counts()
            total = len(df)
            
            # Most common sentiment
            most_common = sentiment_counts.index[0]
            percentage = (sentiment_counts.iloc[0] / total) * 100
            insights.append(f"Most common sentiment: {most_common} ({percentage:.1f}%)")
            
            # Sentiment balance
            if len(sentiment_counts) >= 2:
                top_two = sentiment_counts.head(2)
                ratio = top_two.iloc[0] / top_two.iloc[1]
                if ratio > 2:
                    insights.append(f"Strong dominance of {top_two.index[0]} sentiment")
                elif ratio < 1.5:
                    insights.append("Relatively balanced sentiment distribution")
            
            # Neutral sentiment analysis
            if 'neutral' in sentiment_counts:
                neutral_pct = (sentiment_counts['neutral'] / total) * 100
                if neutral_pct > 50:
                    insights.append(f"High proportion of neutral sentiment ({neutral_pct:.1f}%)")
        
        # Text length insights
        if text_column in df.columns:
            df['text_length'] = df[text_column].str.len()
            avg_length = df['text_length'].mean()
            insights.append(f"Average text length: {avg_length:.0f} characters")
            
            # Length vs sentiment correlation
            if sentiment_column in df.columns:
                sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                df['sentiment_numeric'] = df[sentiment_column].map(sentiment_map).fillna(0)
                correlation = df['text_length'].corr(df['sentiment_numeric'])
                if abs(correlation) > 0.1:
                    direction = "positive" if correlation > 0 else "negative"
                    insights.append(f"Moderate {direction} correlation between text length and sentiment")
        
        # Temporal insights (if time data available)
        time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_columns:
            time_col = time_columns[0]
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(time_col)
                
                # Check for trends
                if sentiment_column in df.columns:
                    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                    df['sentiment_numeric'] = df[sentiment_column].map(sentiment_map).fillna(0)
                    
                    # Simple trend detection
                    first_half = df.head(len(df)//2)['sentiment_numeric'].mean()
                    second_half = df.tail(len(df)//2)['sentiment_numeric'].mean()
                    
                    if second_half > first_half + 0.1:
                        insights.append("Positive sentiment trend over time")
                    elif second_half < first_half - 0.1:
                        insights.append("Negative sentiment trend over time")
                    else:
                        insights.append("Stable sentiment over time")
            except:
                pass
        
        return insights 
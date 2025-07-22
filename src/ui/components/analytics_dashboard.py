"""
Analytics Dashboard Component for Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

from src.ui.utils.analytics import SentimentAnalytics


class AnalyticsDashboard:
    """
    Analytics Dashboard component for advanced sentiment analysis.
    """
    
    def __init__(self):
        """
        Initialize the analytics dashboard.
        """
        self.analytics = SentimentAnalytics()
        self.session_key = "analytics_dashboard"
    
    def render(self) -> None:
        """
        Render the analytics dashboard.
        """
        st.header("📊 Advanced Analytics Dashboard")
        st.markdown("---")
        
        # Check if we have data to analyze
        if not self._has_analysis_data():
            st.warning("No analysis data available. Please perform some sentiment analysis first.")
            return
        
        # Get analysis data
        analysis_data = self._get_analysis_data()
        if not analysis_data:
            st.error("Failed to retrieve analysis data.")
            return
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(analysis_data)
        if df is None or len(df) == 0:
            st.warning("No valid data found for analytics.")
            return
        
        # Display data overview
        self._display_data_overview(df)
        
        # Analytics tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Trend Analysis", 
            "🔍 Clustering Analysis", 
            "📊 Statistical Insights",
            "🎯 Custom Analytics"
        ])
        
        with tab1:
            self._render_trend_analysis(df)
        
        with tab2:
            self._render_clustering_analysis(df)
        
        with tab3:
            self._render_statistical_insights(df)
        
        with tab4:
            self._render_custom_analytics(df)
    
    def _has_analysis_data(self) -> bool:
        """Check if analysis data is available in session state."""
        return (
            'analysis_results' in st.session_state or
            'batch_results' in st.session_state or
            'comparison_results' in st.session_state
        )
    
    def _get_analysis_data(self) -> Optional[List[Dict[str, Any]]]:
        """Get analysis data from session state."""
        data = []
        
        # Get from analysis results
        if 'analysis_results' in st.session_state:
            if isinstance(st.session_state.analysis_results, list):
                data.extend(st.session_state.analysis_results)
            else:
                data.append(st.session_state.analysis_results)
        
        # Get from batch results
        if 'batch_results' in st.session_state:
            if isinstance(st.session_state.batch_results, list):
                data.extend(st.session_state.batch_results)
            else:
                data.append(st.session_state.batch_results)
        
        # Get from comparison results
        if 'comparison_results' in st.session_state:
            if isinstance(st.session_state.comparison_results, list):
                data.extend(st.session_state.comparison_results)
            else:
                data.append(st.session_state.comparison_results)
        
        return data if data else None
    
    def _convert_to_dataframe(self, data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Convert analysis data to DataFrame."""
        try:
            # Normalize the data structure
            normalized_data = []
            
            for item in data:
                if isinstance(item, dict):
                    # Add timestamp if not present
                    if 'timestamp' not in item:
                        item['timestamp'] = datetime.datetime.now()
                    
                    # Normalize sentiment field
                    if 'sentiment' in item:
                        sentiment = item['sentiment']
                        if isinstance(sentiment, dict):
                            # Extract sentiment from dict
                            item['sentiment'] = sentiment.get('label', 'unknown')
                            item['confidence'] = sentiment.get('confidence', 0.0)
                        elif isinstance(sentiment, str):
                            item['sentiment'] = sentiment
                    
                    # Normalize emotion field
                    if 'emotion' in item:
                        emotion = item['emotion']
                        if isinstance(emotion, dict):
                            # Extract emotion details
                            item['emotion_label'] = emotion.get('label', 'unknown')
                            item['emotion_score'] = emotion.get('score', 0.0)
                            # Add individual emotion scores if available
                            for emotion_name in self.analytics.supported_emotions:
                                if emotion_name in emotion:
                                    item[emotion_name] = emotion[emotion_name]
                    
                    normalized_data.append(item)
            
            if normalized_data:
                df = pd.DataFrame(normalized_data)
                return df
            
        except Exception as e:
            st.error(f"Error converting data to DataFrame: {str(e)}")
        
        return None
    
    def _display_data_overview(self, df: pd.DataFrame) -> None:
        """Display overview of the data."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            if 'sentiment' in df.columns:
                unique_sentiments = df['sentiment'].nunique()
                st.metric("Sentiment Categories", unique_sentiments)
            else:
                st.metric("Sentiment Categories", "N/A")
        
        with col3:
            if 'confidence' in df.columns:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        with col4:
            if 'timestamp' in df.columns:
                time_range = df['timestamp'].max() - df['timestamp'].min()
                st.metric("Time Range", f"{time_range.days} days")
            else:
                st.metric("Time Range", "N/A")
        
        st.markdown("---")
    
    def _render_trend_analysis(self, df: pd.DataFrame) -> None:
        """Render trend analysis tab."""
        st.subheader("📈 Trend Analysis")
        
        # Time column selection
        time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if not time_columns and 'timestamp' in df.columns:
            time_columns = ['timestamp']
        
        if not time_columns:
            st.warning("No time column found. Trend analysis requires a time column.")
            return
        
        selected_time_col = st.selectbox(
            "Select Time Column:",
            time_columns,
            index=0
        )
        
        # Analysis period selection
        analysis_period = st.selectbox(
            "Analysis Period:",
            ["Daily", "Weekly", "Monthly"],
            index=0
        )
        
        if st.button("Analyze Trends", type="primary"):
            with st.spinner("Analyzing trends..."):
                try:
                    # Perform trend analysis
                    trend_results = self.analytics.analyze_trends(df, selected_time_col)
                    
                    if "error" in trend_results:
                        st.error(trend_results["error"])
                        return
                    
                    # Display results
                    self._display_trend_results(trend_results)
                    
                    # Generate and display charts
                    charts = self.analytics.generate_analytics_charts(df, trend_results)
                    
                    if "time_series" in charts:
                        st.plotly_chart(charts["time_series"], use_container_width=True)
                    
                    if "seasonal_patterns" in charts:
                        st.plotly_chart(charts["seasonal_patterns"], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing trend analysis: {str(e)}")
    
    def _display_trend_results(self, results: Dict[str, Any]) -> None:
        """Display trend analysis results."""
        st.subheader("Trend Analysis Results")
        
        # Time range information
        if "time_range" in results:
            time_range = results["time_range"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Start Date", time_range["start"].strftime("%Y-%m-%d"))
            
            with col2:
                st.metric("End Date", time_range["end"].strftime("%Y-%m-%d"))
            
            with col3:
                duration_days = time_range["duration"] / (24 * 3600)
                st.metric("Duration", f"{duration_days:.1f} days")
        
        # Trend metrics
        if "trends" in results:
            st.subheader("Trend Metrics")
            
            for period, metrics in results["trends"].items():
                if "error" in metrics:
                    st.warning(f"{period.capitalize()} trends: {metrics['error']}")
                    continue
                
                st.write(f"**{period.capitalize()} Trends:**")
                
                # Create a DataFrame for better display
                trend_data = []
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "trend_direction" in metric_data:
                        trend_data.append({
                            "Metric": metric_name.replace("_", " ").title(),
                            "Trend Direction": metric_data["trend_direction"],
                            "Slope": f"{metric_data['slope']:.4f}",
                            "R²": f"{metric_data['r_squared']:.4f}",
                            "P-value": f"{metric_data['p_value']:.4f}"
                        })
                    elif isinstance(metric_data, (int, float)):
                        trend_data.append({
                            "Metric": metric_name.replace("_", " ").title(),
                            "Value": f"{metric_data:.4f}"
                        })
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    st.dataframe(trend_df, use_container_width=True)
    
    def _render_clustering_analysis(self, df: pd.DataFrame) -> None:
        """Render clustering analysis tab."""
        st.subheader("🔍 Clustering Analysis")
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
        
        with col2:
            clustering_method = st.selectbox(
                "Clustering Method:",
                ["K-Means", "Hierarchical", "DBSCAN"],
                index=0
            )
        
        if st.button("Perform Clustering", type="primary"):
            with st.spinner("Performing clustering analysis..."):
                try:
                    # Perform clustering
                    clustering_results = self.analytics.perform_clustering(df, n_clusters)
                    
                    if "error" in clustering_results:
                        st.error(clustering_results["error"])
                        return
                    
                    # Display results
                    self._display_clustering_results(clustering_results)
                    
                    # Generate and display clustering chart
                    charts = self.analytics.generate_analytics_charts(df, clustering_results)
                    
                    if "clustering" in charts:
                        st.plotly_chart(charts["clustering"], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")
    
    def _display_clustering_results(self, results: Dict[str, Any]) -> None:
        """Display clustering analysis results."""
        st.subheader("Clustering Results")
        
        # Cluster sizes
        if "cluster_sizes" in results:
            st.write("**Cluster Sizes:**")
            cluster_sizes = results["cluster_sizes"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of cluster sizes
                fig = go.Figure(data=[go.Pie(
                    labels=[f"Cluster {i}" for i in range(len(cluster_sizes))],
                    values=list(cluster_sizes.values()),
                    hole=0.3
                )])
                fig.update_layout(title="Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart of cluster sizes
                fig = go.Figure(data=[go.Bar(
                    x=[f"Cluster {i}" for i in range(len(cluster_sizes))],
                    y=list(cluster_sizes.values())
                )])
                fig.update_layout(
                    title="Cluster Sizes",
                    xaxis_title="Cluster",
                    yaxis_title="Number of Records"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        if "cluster_characteristics" in results:
            st.write("**Cluster Characteristics:**")
            
            for cluster_id, characteristics in results["cluster_characteristics"].items():
                with st.expander(f"{cluster_id.replace('_', ' ').title()}"):
                    # Display sentiment distribution
                    if "sentiment_distribution" in characteristics:
                        st.write("**Sentiment Distribution:**")
                        sentiment_dist = characteristics["sentiment_distribution"]
                        
                        # Create bar chart
                        fig = go.Figure(data=[go.Bar(
                            x=list(sentiment_dist.keys()),
                            y=list(sentiment_dist.values())
                        )])
                        fig.update_layout(
                            title=f"Sentiment Distribution - {cluster_id}",
                            xaxis_title="Sentiment",
                            yaxis_title="Proportion"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display other characteristics
                    for key, value in characteristics.items():
                        if key != "sentiment_distribution":
                            if isinstance(value, float):
                                st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    def _render_statistical_insights(self, df: pd.DataFrame) -> None:
        """Render statistical insights tab."""
        st.subheader("📊 Statistical Insights")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Overview:**")
            st.write(f"Total records: {len(df)}")
            st.write(f"Columns: {', '.join(df.columns)}")
            
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                st.write("**Sentiment Distribution:**")
                for sentiment, count in sentiment_counts.items():
                    st.write(f"- {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        with col2:
            # Numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("**Numeric Statistics:**")
                stats_df = df[numeric_cols].describe()
                st.dataframe(stats_df, use_container_width=True)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("Correlation Analysis")
            
            # Calculate correlations
            correlation_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(
                title="Correlation Matrix",
                width=600,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        strong_correlations.append({
                            'Variable 1': numeric_cols[i],
                            'Variable 2': numeric_cols[j],
                            'Correlation': f"{corr:.3f}"
                        })
            
            if strong_correlations:
                st.write("**Strong Correlations (|r| > 0.7):**")
                strong_corr_df = pd.DataFrame(strong_correlations)
                st.dataframe(strong_corr_df, use_container_width=True)
        
        # Distribution charts
        st.subheader("Distribution Analysis")
        
        # Sentiment distribution
        if 'sentiment' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=df['sentiment'].value_counts().index,
                    values=df['sentiment'].value_counts().values
                )])
                fig.update_layout(title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'confidence' in df.columns:
                    fig = go.Figure(data=[go.Histogram(
                        x=df['confidence'],
                        nbinsx=20
                    )])
                    fig.update_layout(
                        title="Confidence Distribution",
                        xaxis_title="Confidence",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_custom_analytics(self, df: pd.DataFrame) -> None:
        """Render custom analytics tab."""
        st.subheader("🎯 Custom Analytics")
        
        # Custom analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "Sentiment by Time Period",
                "Emotion Analysis",
                "Confidence Analysis",
                "Text Length Analysis",
                "Custom Query"
            ]
        )
        
        if analysis_type == "Sentiment by Time Period":
            self._render_sentiment_by_time(df)
        
        elif analysis_type == "Emotion Analysis":
            self._render_emotion_analysis(df)
        
        elif analysis_type == "Confidence Analysis":
            self._render_confidence_analysis(df)
        
        elif analysis_type == "Text Length Analysis":
            self._render_text_length_analysis(df)
        
        elif analysis_type == "Custom Query":
            self._render_custom_query(df)
    
    def _render_sentiment_by_time(self, df: pd.DataFrame) -> None:
        """Render sentiment analysis by time period."""
        if 'timestamp' not in df.columns or 'sentiment' not in df.columns:
            st.warning("Timestamp and sentiment columns required for this analysis.")
            return
        
        # Time period selection
        period = st.selectbox(
            "Time Period:",
            ["Hour", "Day", "Week", "Month"],
            index=1
        )
        
        if st.button("Analyze Sentiment by Time"):
            # Group by time period
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            if period == "Hour":
                df_copy['time_period'] = df_copy['timestamp'].dt.hour
            elif period == "Day":
                df_copy['time_period'] = df_copy['timestamp'].dt.day_name()
            elif period == "Week":
                df_copy['time_period'] = df_copy['timestamp'].dt.isocalendar().week
            elif period == "Month":
                df_copy['time_period'] = df_copy['timestamp'].dt.month_name()
            
            # Calculate sentiment distribution by time period
            sentiment_by_time = df_copy.groupby(['time_period', 'sentiment']).size().unstack(fill_value=0)
            
            # Create visualization
            fig = go.Figure()
            
            for sentiment in sentiment_by_time.columns:
                fig.add_trace(go.Bar(
                    x=sentiment_by_time.index,
                    y=sentiment_by_time[sentiment],
                    name=sentiment.capitalize()
                ))
            
            fig.update_layout(
                title=f"Sentiment Distribution by {period}",
                xaxis_title=period,
                yaxis_title="Count",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_emotion_analysis(self, df: pd.DataFrame) -> None:
        """Render emotion analysis."""
        emotion_cols = [col for col in df.columns if col in self.analytics.supported_emotions]
        
        if not emotion_cols:
            st.warning("No emotion columns found in the data.")
            return
        
        # Emotion analysis options
        analysis_option = st.selectbox(
            "Emotion Analysis Type:",
            ["Emotion Distribution", "Emotion Correlations", "Emotion Trends"]
        )
        
        if analysis_option == "Emotion Distribution":
            # Calculate average emotion scores
            emotion_means = df[emotion_cols].mean()
            
            fig = go.Figure(data=[go.Bar(
                x=emotion_means.index,
                y=emotion_means.values
            )])
            fig.update_layout(
                title="Average Emotion Scores",
                xaxis_title="Emotion",
                yaxis_title="Average Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_option == "Emotion Correlations":
            # Calculate emotion correlations
            emotion_corr = df[emotion_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=emotion_corr.values,
                x=emotion_corr.columns,
                y=emotion_corr.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(
                title="Emotion Correlations",
                width=600,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_option == "Emotion Trends":
            if 'timestamp' in df.columns:
                # Group by time and calculate emotion trends
                df_copy = df.copy()
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
                df_copy['date'] = df_copy['timestamp'].dt.date
                
                emotion_trends = df_copy.groupby('date')[emotion_cols].mean()
                
                fig = go.Figure()
                
                for emotion in emotion_cols:
                    fig.add_trace(go.Scatter(
                        x=emotion_trends.index,
                        y=emotion_trends[emotion],
                        mode='lines+markers',
                        name=emotion.capitalize()
                    ))
                
                fig.update_layout(
                    title="Emotion Trends Over Time",
                    xaxis_title="Date",
                    yaxis_title="Average Emotion Score"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_confidence_analysis(self, df: pd.DataFrame) -> None:
        """Render confidence analysis."""
        if 'confidence' not in df.columns:
            st.warning("Confidence column not found in the data.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig = go.Figure(data=[go.Histogram(
                x=df['confidence'],
                nbinsx=20
            )])
            fig.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence by sentiment
            if 'sentiment' in df.columns:
                confidence_by_sentiment = df.groupby('sentiment')['confidence'].agg(['mean', 'std', 'count'])
                
                fig = go.Figure(data=[go.Bar(
                    x=confidence_by_sentiment.index,
                    y=confidence_by_sentiment['mean']
                )])
                fig.update_layout(
                    title="Average Confidence by Sentiment",
                    xaxis_title="Sentiment",
                    yaxis_title="Average Confidence"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_text_length_analysis(self, df: pd.DataFrame) -> None:
        """Render text length analysis."""
        if 'text' not in df.columns:
            st.warning("Text column not found in the data.")
            return
        
        # Calculate text lengths
        df['text_length'] = df['text'].str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text length distribution
            fig = go.Figure(data=[go.Histogram(
                x=df['text_length'],
                nbinsx=20
            )])
            fig.update_layout(
                title="Text Length Distribution",
                xaxis_title="Text Length (characters)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Text length vs confidence
            if 'confidence' in df.columns:
                fig = go.Figure(data=[go.Scatter(
                    x=df['text_length'],
                    y=df['confidence'],
                    mode='markers',
                    marker=dict(size=6)
                )])
                fig.update_layout(
                    title="Text Length vs Confidence",
                    xaxis_title="Text Length (characters)",
                    yaxis_title="Confidence"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_custom_query(self, df: pd.DataFrame) -> None:
        """Render custom query interface."""
        st.write("**Custom Data Query:**")
        
        # Column selection
        selected_columns = st.multiselect(
            "Select columns to analyze:",
            df.columns.tolist(),
            default=df.columns[:3].tolist()
        )
        
        if selected_columns:
            # Show selected data
            st.write("**Selected Data:**")
            st.dataframe(df[selected_columns], use_container_width=True)
            
            # Basic statistics for selected columns
            numeric_selected = [col for col in selected_columns if col in df.select_dtypes(include=[np.number]).columns]
            
            if numeric_selected:
                st.write("**Statistics for Numeric Columns:**")
                stats_df = df[numeric_selected].describe()
                st.dataframe(stats_df, use_container_width=True)
            
            # Custom filter
            st.write("**Custom Filter:**")
            filter_column = st.selectbox("Filter by column:", selected_columns)
            
            if filter_column in df.columns:
                unique_values = df[filter_column].unique()
                if len(unique_values) <= 20:  # Only show for reasonable number of values
                    selected_values = st.multiselect(
                        f"Select {filter_column} values:",
                        unique_values.tolist()
                    )
                    
                    if selected_values:
                        filtered_df = df[df[filter_column].isin(selected_values)]
                        st.write(f"**Filtered Data ({len(filtered_df)} records):**")
                        st.dataframe(filtered_df[selected_columns], use_container_width=True) 
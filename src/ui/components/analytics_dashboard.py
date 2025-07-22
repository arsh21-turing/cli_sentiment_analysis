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
import altair as alt
from datetime import datetime, timedelta
import io
import json
import base64
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar
from matplotlib.colors import LinearSegmentedColormap
import os
from fpdf import FPDF

from src.utils.analytics import SentimentAnalytics
from src.utils.real_time import RealTimeDataConnector
from src.utils.anomaly_detector import SentimentAnomalyDetector
from src.ui.utils.chart_exporter import ChartExporter, ChartFilterer
from src.ui.utils.analysis_storage import AnalysisStorage


class AnalyticsDashboard:
    """
    Analytics Dashboard component for advanced sentiment analysis.
    """
    
    def __init__(self, session_state=None, key_manager=None):
        """
        Initialize the analytics dashboard.
        
        Args:
            session_state: Streamlit session state for persistence
            key_manager: Widget key manager for unique keys
        """
        self.session_state = session_state or st.session_state
        self.key_manager = key_manager
        if self.key_manager:
            self.key_manager.register_component('analytics_dashboard', 'ad')
        
        # Initialize analytics utilities
        if 'transformer_model' in self.session_state:
            transformer = self.session_state.transformer_model
        else:
            # Create default transformer if one doesn't exist in session
            from src.models.transformer import SentimentEmotionTransformer
            transformer = SentimentEmotionTransformer()
            self.session_state.transformer_model = transformer
        
        # Initialize real-time data connector
        self.setup_real_time_connection()
        
        self.analytics = SentimentAnalytics(transformer=transformer)
        self.session_key = "analytics_dashboard"
        
        # State for real-time alerts
        self.alerts_enabled = False
        self.alert_thresholds = {
            'negative_sentiment': -0.7,  # Alert if sentiment is below this threshold
            'sentiment_drop': 0.3,       # Alert if sentiment drops by this amount
            'negative_ratio': 0.7        # Alert if negative ratio exceeds this threshold
        }
        
        # Real-time visualization placeholders
        self.trend_chart_placeholder = None
        self.stats_placeholder = None
        self.alerts_placeholder = None
        
        # Initialize anomaly detector
        self.anomaly_detector = SentimentAnomalyDetector(contamination=0.05)
        
        # Initialize chart exporter and filterer
        self.chart_exporter = ChartExporter()
        self.chart_filterer = ChartFilterer()
        
        # Initialize storage utility
        self.storage = AnalysisStorage()
    
    def render(self) -> None:
        """
        Render the analytics dashboard.
        """
        st.header("📊 Advanced Analytics Dashboard")
        st.markdown("---")
        
        # Add save/load buttons at the top
        self.render_save_load_buttons()
        
        # Show dialogs if requested
        if st.session_state.get('show_save_dialog', False):
            self.render_save_dialog()
        
        if st.session_state.get('show_load_dialog', False):
            self.render_load_dialog()
        
        if st.session_state.get('show_manage_dialog', False):
            self.render_manage_dialog()
        
        # Show "loaded from saved analysis" indicator if applicable
        if hasattr(self.session_state, 'data_source') and self.session_state.data_source == 'saved_analysis':
            st.info(f"Currently viewing saved analysis: {self.session_state.loaded_analysis_name}")
            
            # Option to return to live data
            if st.button("Return to Live Data"):
                # Clear the loaded analysis
                if 'data_source' in self.session_state:
                    del self.session_state.data_source
                if 'loaded_analysis_id' in self.session_state:
                    del self.session_state.loaded_analysis_id
                if 'loaded_analysis_name' in self.session_state:
                    del self.session_state.loaded_analysis_name
                
                st.rerun()
        
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Trend Analysis", 
            "🔍 Clustering Analysis", 
            "📊 Statistical Insights",
            "🎯 Custom Analytics",
            "🚨 Anomaly Detection",
            "⚡ Real-Time Analytics"
        ])
        
        with tab1:
            self._render_trend_analysis(df)
        
        with tab2:
            self._render_clustering_analysis(df)
        
        with tab3:
            self._render_statistical_insights(df)
        
        with tab4:
            self._render_custom_analytics(df)
        
        with tab5:
            self.render_anomaly_analysis()
        
        with tab6:
            self.render_real_time_dashboard()
    
    def setup_real_time_connection(self):
        """
        Sets up real-time data connector.
        """
        # Check if real-time connector already exists in session state
        if 'real_time_connector' not in self.session_state:
            # Create new connector
            connector = RealTimeDataConnector(max_queue_size=1000, ttl_seconds=3600)
            connector.attach_to_session(self.session_state)
            self.session_state.real_time_connector = connector
        
        # Get connector instance
        self.real_time_connector = self.session_state.real_time_connector
        
        # Subscribe to updates
        if not hasattr(self, 'subscription_id'):
            self.subscription_id = self.real_time_connector.subscribe(self.handle_real_time_update)
    
    def render_real_time_dashboard(self):
        """
        Renders real-time analytics dashboard.
        """
        st.subheader("Real-Time Sentiment Analytics")
        
        # Add save/load buttons to the real-time dashboard as well
        self.render_save_load_buttons_real_time()
        
        # Show dialogs if requested
        if st.session_state.get('show_save_dialog', False):
            self.render_save_dialog()
        
        if st.session_state.get('show_load_dialog', False):
            self.render_load_dialog()
        
        if st.session_state.get('show_manage_dialog', False):
            self.render_manage_dialog()
        
        # Show "loaded from saved analysis" indicator if applicable
        if hasattr(self.session_state, 'data_source') and self.session_state.data_source == 'saved_analysis':
            st.warning(f"Viewing saved analysis: {self.session_state.loaded_analysis_name} (Real-time updates paused)")
            
            # Option to return to live data
            if st.button(
                "Return to Live Data",
                key=self.key_manager.get_key('real_time_analytics', 'return_to_live') if self.key_manager else None
            ):
                # Clear the loaded analysis
                if 'data_source' in self.session_state:
                    del self.session_state.data_source
                if 'loaded_analysis_id' in self.session_state:
                    del self.session_state.loaded_analysis_id
                if 'loaded_analysis_name' in self.session_state:
                    del self.session_state.loaded_analysis_name
                
                st.rerun()
        
        # Settings and controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auto_refresh = st.checkbox(
                "Auto-refresh", 
                value=True, 
                help="Automatically refresh visualizations with new data",
                key=self.key_manager.get_key('real_time_analytics', 'auto_refresh') if self.key_manager else None
            )
            
        with col2:
            # Enable/disable alerts
            self.alerts_enabled = st.checkbox(
                "Enable alerts", 
                value=self.alerts_enabled,
                help="Show alerts for significant sentiment changes",
                key=self.key_manager.get_key('real_time_analytics', 'enable_alerts') if self.key_manager else None
            )
        
        with col3:
            # Export button
            if st.button(
                "Export Charts", 
                help="Export charts with custom options",
                key=self.key_manager.get_key('real_time_analytics', 'export_charts') if self.key_manager else None
            ):
                st.session_state.show_export_options = True
        
        with col4:
            # Clear data button
            if st.button(
                "Clear Data",
                key=self.key_manager.get_key('real_time_analytics', 'clear_data') if self.key_manager else None
            ):
                self.real_time_connector.clear()
                st.success("Real-time data cleared")
                st.rerun()
        
        # Get current data
        data = self.real_time_connector.get_latest_data()
        aggregated_data = self.real_time_connector.get_aggregated_data()
        
        # Safety check for aggregated_data type
        if not isinstance(aggregated_data, dict):
            st.error(f"Invalid aggregated data type: {type(aggregated_data)}. Expected dictionary.")
            aggregated_data = {'count': 0}
        
        data_count = aggregated_data.get('count', 0)
        
        # Show export options if requested
        if st.session_state.get('show_export_options', False):
            # Convert data to DataFrame if needed
            df = self.real_time_connector._convert_to_dataframe(data)
            
            with st.expander("Chart Export Options", expanded=True):
                self.render_export_options_real_time(df)
                
                # Add close button
                if st.button(
                    "Close Export Options",
                    key=self.key_manager.get_key('real_time_analytics', 'close_export') if self.key_manager else None
                ):
                    st.session_state.show_export_options = False
                    st.rerun()
        
        # Detect anomalies in real-time data
        if data_count >= 10:  # Need at least 10 data points for meaningful anomaly detection
            # Convert to DataFrame if not already
            if not isinstance(df, pd.DataFrame):
                df = self.real_time_connector._convert_to_dataframe(data)
                
            # Detect anomalies using isolation forest
            df_with_anomalies = self.anomaly_detector.detect_anomalies(df, method='isolation_forest')
            anomaly_count = df_with_anomalies['anomaly'].sum()
            
            if anomaly_count > 0:
                # Generate insights
                anomaly_insights = self.anomaly_detector.generate_insights(df_with_anomalies)
                
                # Display anomaly alert section
                st.error(f"⚠️ Detected {anomaly_count} unusual sentiment patterns")
                
                with st.expander("Anomaly Details", expanded=True):
                    # Show anomaly insights
                    st.write(anomaly_insights['message'])
                    
                    for detail in anomaly_insights['details']:
                        st.write(f"- {detail}")
                    
                    # Display recommendations
                    if anomaly_insights['recommendations']:
                        st.subheader("Recommendations")
                        for rec in anomaly_insights['recommendations']:
                            st.write(f"- {rec}")
                    
                    # Option to display anomaly visualization
                    if st.button(
                        "Visualize Anomalies",
                        key=self.key_manager.get_key('real_time_analytics', 'visualize_anomalies') if self.key_manager else None
                    ):
                        anomaly_chart = self.create_anomaly_chart(df_with_anomalies)
                        st.altair_chart(anomaly_chart, use_container_width=True)
        
        if data_count == 0:
            st.info("No real-time data available yet. Process some text to see updates here.")
            
            # Show instructions
            st.markdown("""
            ### How to use Real-Time Analytics
            
            1. Use the **Single Text Analysis** tab to analyze text
            2. Results will automatically appear here in real-time
            3. Process multiple texts to see trends and patterns
            4. Upload batch files for larger volume analysis
            
            Real-time analytics works with all text processing methods in the application.
            """)
            return
        
        # Create placeholders for updating components
        if auto_refresh:
            self.trend_chart_placeholder = st.empty()
            self.stats_placeholder = st.empty()
            self.alerts_placeholder = st.empty()
        
        # Display real-time sentiment trend
        st.subheader("Live Sentiment Trend")
        
        # Convert data to DataFrame
        df = self.real_time_connector._convert_to_dataframe(data)
        
        # Create time series chart
        trend_chart = self.create_real_time_trend_chart(df)
        
        if auto_refresh:
            self.trend_chart_placeholder.altair_chart(trend_chart, use_container_width=True)
        else:
            st.altair_chart(trend_chart, use_container_width=True)
        
        # Display real-time statistics
        st.subheader("Live Statistics")
        
        # Get sentiment statistics with type safety
        sentiment_stats = aggregated_data.get('sentiment_stats', {}) if isinstance(aggregated_data, dict) else {}
        sentiment_labels = aggregated_data.get('sentiment_labels', {}) if isinstance(aggregated_data, dict) else {}
        emotion_labels = aggregated_data.get('emotion_labels', {}) if isinstance(aggregated_data, dict) else {}
        
        # Create statistics display
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            mean_sentiment = sentiment_stats.get('mean', 0)
            st.metric(
                "Average Sentiment", 
                f"{mean_sentiment:.2f}",
                delta=None
            )
        
        with stats_cols[1]:
            positive_count = sentiment_stats.get('positive_count', 0)
            positive_pct = (positive_count / data_count) * 100 if data_count > 0 else 0
            st.metric(
                "Positive",
                f"{positive_pct:.1f}%",
                f"{positive_count} texts"
            )
        
        with stats_cols[2]:
            neutral_count = sentiment_stats.get('neutral_count', 0)
            neutral_pct = (neutral_count / data_count) * 100 if data_count > 0 else 0
            st.metric(
                "Neutral",
                f"{neutral_pct:.1f}%",
                f"{neutral_count} texts"
            )
        
        with stats_cols[3]:
            negative_count = sentiment_stats.get('negative_count', 0)
            negative_pct = (negative_count / data_count) * 100 if data_count > 0 else 0
            st.metric(
                "Negative",
                f"{negative_pct:.1f}%",
                f"{negative_count} texts"
            )
        
        # Display recent sentiment shift
        recent_stats = aggregated_data.get('recent_stats', {}) if isinstance(aggregated_data, dict) else {}
        if recent_stats and 'mean' in recent_stats:
            recent_mean = recent_stats.get('mean', 0)
            overall_mean = sentiment_stats.get('mean', 0)
            
            st.subheader("Recent Trend (Last 5 Minutes)")
            
            # Calculate change
            sentiment_shift = recent_mean - overall_mean
            shift_pct = (sentiment_shift / abs(overall_mean)) * 100 if overall_mean != 0 else 0
            
            # Show with appropriate color
            if abs(sentiment_shift) < 0.05:  # Small change
                st.info(f"Sentiment is stable at {recent_mean:.2f} (change: {sentiment_shift:+.2f})")
            elif sentiment_shift > 0:  # Positive shift
                st.success(f"Sentiment is trending positive at {recent_mean:.2f} (change: {sentiment_shift:+.2f}, {shift_pct:+.1f}%)")
            else:  # Negative shift
                st.error(f"Sentiment is trending negative at {recent_mean:.2f} (change: {sentiment_shift:+.2f}, {shift_pct:+.1f}%)")
        
        # Display alerts if enabled
        if self.alerts_enabled:
            self.show_alerts(aggregated_data)
        
        # Display sentiment heatmap visualization
        st.subheader("Sentiment Heatmap")
        
        # Create tabs for different heatmap views
        heatmap_tabs = st.tabs(["Time-based Heatmap", "Hourly Pattern", "Daily Pattern", "Day/Hour Heatmap"])
        
        with heatmap_tabs[0]:
            # Time-based sentiment heatmap
            time_heatmap = self.create_sentiment_time_heatmap(df)
            st.pyplot(time_heatmap)
        
        with heatmap_tabs[1]:
            # Hourly pattern heatmap
            hourly_heatmap = self.create_hourly_sentiment_heatmap(df)
            st.pyplot(hourly_heatmap)
            
        with heatmap_tabs[2]:
            # Daily pattern heatmap
            daily_heatmap = self.create_daily_sentiment_heatmap(df)
            st.pyplot(daily_heatmap)
        
        with heatmap_tabs[3]:
            # 2D heatmap of sentiment by hour and day
            day_hour_heatmap = self.create_day_hour_heatmap(df)
            st.pyplot(day_hour_heatmap)
        
        # Live sentiment distribution
        st.subheader("Live Sentiment Distribution")
        
        # Create sentiment distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Create sentiment label distribution chart
            if sentiment_labels:
                sentiment_data = pd.DataFrame({
                    'Label': list(sentiment_labels.keys()),
                    'Count': list(sentiment_labels.values())
                })
                
                # Sort by common order
                sentiment_order = ['positive', 'neutral', 'negative']
                sentiment_data['Order'] = sentiment_data['Label'].map(
                    {label: i for i, label in enumerate(sentiment_order)}
                ).fillna(len(sentiment_order))
                sentiment_data = sentiment_data.sort_values('Order')
                
                # Create chart
                label_chart = alt.Chart(sentiment_data).mark_bar().encode(
                    x='Label:N',
                    y='Count:Q',
                    color=alt.Color('Label:N', scale=alt.Scale(
                        domain=['positive', 'neutral', 'negative'],
                        range=['green', 'gray', 'red']
                    )),
                    tooltip=['Label', 'Count']
                ).properties(
                    title='Sentiment Label Distribution'
                )
                
                st.altair_chart(label_chart, use_container_width=True)
        
        with col2:
            # Create emotion distribution chart if available
            if emotion_labels:
                emotion_data = pd.DataFrame({
                    'Emotion': list(emotion_labels.keys()),
                    'Count': list(emotion_labels.values())
                })
                
                # Create pie chart for emotions
                emotion_chart = alt.Chart(emotion_data).mark_arc().encode(
                    theta='Count:Q',
                    color='Emotion:N',
                    tooltip=['Emotion', 'Count']
                ).properties(
                    title='Emotion Distribution'
                )
                
                st.altair_chart(emotion_chart, use_container_width=True)
        
        # Recent texts
        st.subheader("Recent Analyses")
        
        # Get the most recent 10 items
        recent_items = self.real_time_connector.get_latest_data(10)
        recent_items.reverse()  # Show newest first
        
        for i, item in enumerate(recent_items):
            # Type safety check for item
            if not isinstance(item, dict):
                st.error(f"Invalid item type: {type(item)}. Expected dictionary.")
                continue
                
            with st.expander(f"Text {i+1}: {item.get('text', '')[:50]}...", expanded=i==0):
                # Display text
                st.markdown(f"**Text:** {item.get('text', 'N/A')}")
                
                # Display sentiment
                sentiment = item.get('sentiment', {})
                if isinstance(sentiment, dict) and sentiment:
                    label = sentiment.get('label', 'N/A')
                    score = sentiment.get('score', 0)
                    
                    # Color based on label
                    if label.lower() == 'positive':
                        st.markdown(f"**Sentiment:** :green[{label} ({score:.2f})]")
                    elif label.lower() == 'negative':
                        st.markdown(f"**Sentiment:** :red[{label} ({score:.2f})]")
                    else:
                        st.markdown(f"**Sentiment:** {label} ({score:.2f})")
                
                # Display emotion if available
                emotion = item.get('emotion', {})
                if isinstance(emotion, dict) and emotion:
                    label = emotion.get('label', 'N/A')
                    score = emotion.get('score', 0)
                    st.markdown(f"**Emotion:** {label} ({score:.2f})")
                
                # Display timestamp
                if 'timestamp' in item:
                    st.text(f"Analyzed at: {item['timestamp']}")
        
        # Add export button
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Real-Time Data (JSON)", use_container_width=True):
                json_bytes = self.real_time_connector.export_data('json')
                st.download_button(
                    label="Download JSON",
                    data=json_bytes,
                    file_name=f"real_time_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📈 Export Real-Time Data (CSV)", use_container_width=True):
                csv_bytes = self.real_time_connector.export_data('csv')
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name=f"real_time_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    def handle_real_time_update(self, data: Dict[str, Any]) -> None:
        """
        Callback for real-time data updates.
        
        Args:
            data: New data received from the connector
        """
        # This method is called when new data is pushed to the connector
        # We could perform custom processing here if needed
        pass
    
    def enable_alerts(self, alert_thresholds: Optional[Dict[str, float]] = None) -> None:
        """
        Enables real-time alerting.
        
        Args:
            alert_thresholds: Custom thresholds for alerts
        """
        self.alerts_enabled = True
        
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
    
    def get_live_sentiment_stats(self) -> Dict[str, Any]:
        """
        Returns current sentiment statistics from live data.
        
        Returns:
            Aggregated statistics from live data stream
        """
        return self.real_time_connector.get_aggregated_data()
    
    def show_alerts(self, aggregated_data: Dict[str, Any]) -> None:
        """
        Shows alerts based on real-time sentiment data.
        
        Args:
            aggregated_data: Aggregated sentiment data
        """
        # Type safety check
        if not isinstance(aggregated_data, dict):
            st.error(f"Invalid aggregated data type: {type(aggregated_data)}. Expected dictionary.")
            return
            
        sentiment_stats = aggregated_data.get('sentiment_stats', {}) if isinstance(aggregated_data.get('sentiment_stats', {}), dict) else {}
        recent_stats = aggregated_data.get('recent_stats', {}) if isinstance(aggregated_data.get('recent_stats', {}), dict) else {}
        
        alerts = []
        
        # Check for very negative sentiment
        mean_sentiment = sentiment_stats.get('mean', 0) if isinstance(sentiment_stats, dict) else 0
        if mean_sentiment < self.alert_thresholds['negative_sentiment']:
            alerts.append({
                'type': 'critical',
                'message': f"Overall sentiment is very negative ({mean_sentiment:.2f})",
                'details': "Consider investigating the cause of negative sentiment."
            })
        
        # Check for significant sentiment drop
        if isinstance(recent_stats, dict) and isinstance(sentiment_stats, dict) and 'mean' in recent_stats and 'mean' in sentiment_stats:
            recent_mean = recent_stats.get('mean', 0)
            overall_mean = sentiment_stats.get('mean', 0)
            
            sentiment_shift = recent_mean - overall_mean
            
            if sentiment_shift < -self.alert_thresholds['sentiment_drop']:
                alerts.append({
                    'type': 'warning',
                    'message': f"Significant negative shift in sentiment detected ({sentiment_shift:.2f})",
                    'details': f"Recent sentiment average: {recent_mean:.2f}, Overall average: {overall_mean:.2f}"
                })
        
        # Check for high negative ratio
        if isinstance(recent_stats, dict) and 'negative_ratio' in recent_stats:
            negative_ratio = recent_stats.get('negative_ratio', 0)
            
            if negative_ratio > self.alert_thresholds['negative_ratio']:
                alerts.append({
                    'type': 'warning',
                    'message': f"High proportion of negative texts ({negative_ratio*100:.1f}%)",
                    'details': f"Recent texts are predominantly negative."
                })
        
        # Display alerts
        for alert in alerts:
            if alert['type'] == 'critical':
                st.error(f"🚨 ALERT: {alert['message']}")
            else:
                st.warning(f"⚠️ ALERT: {alert['message']}")
            
            # Show details in expander
            with st.expander("Details"):
                st.write(alert['details'])
        
        if not alerts and self.alerts_enabled:
            st.success("✓ No alerts - sentiment metrics are within normal ranges.")
    
    def create_real_time_trend_chart(self, data: pd.DataFrame) -> alt.Chart:
        """
        Creates real-time trend visualization.
        
        Args:
            data: Time series data to visualize
            
        Returns:
            Altair chart object
        """
        if len(data) == 0:
            # Create empty chart if no data
            empty_df = pd.DataFrame({'timestamp': [datetime.now()], 'value': [0]})
            return alt.Chart(empty_df).mark_line().encode(
                x='timestamp:T',
                y='value:Q'
            ).properties(
                title='No data available'
            )
        
        # Reset index to get timestamp as column
        plot_data = data.reset_index()
        
        # Create chart with sentiment score
        line = alt.Chart(plot_data).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('sentiment_score:Q', title='Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
            tooltip=['timestamp:T', 'sentiment_score:Q', 'sentiment_label:N']
        ).properties(
            title='Sentiment Score Over Time'
        )
        
        # Add points colored by sentiment label
        points = alt.Chart(plot_data).mark_circle(size=60).encode(
            x='timestamp:T',
            y='sentiment_score:Q',
            color=alt.Color('sentiment_label:N', scale=alt.Scale(
                domain=['positive', 'neutral', 'negative'],
                range=['green', 'gray', 'red']
            )),
            tooltip=['timestamp:T', 'sentiment_score:Q', 'sentiment_label:N', 'text:N']
        )
        
        # Combine layers
        return (line + points)
    
    def create_sentiment_time_heatmap(self, data):
        """
        Creates a sentiment heatmap visualization showing intensity over time.
        
        Args:
            data: DataFrame with sentiment data and timestamp index
            
        Returns:
            Matplotlib figure
        """
        if len(data) < 2:
            # Not enough data for a meaningful heatmap
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.text(0.5, 0.5, "Not enough data for heatmap visualization", 
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        # Make sure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            # If no timestamp index, create a dummy one
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                # Create artificial timestamps if none exist
                timestamps = pd.date_range(
                    start=datetime.now() - pd.Timedelta(minutes=len(data)),
                    periods=len(data),
                    freq='1min'
                )
                data.index = timestamps
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Define time intervals for binning
        # If we have data spanning more than a day, use hours as bins
        # Otherwise use minutes
        time_span = data.index.max() - data.index.min()
        
        if time_span.total_seconds() > 86400:  # More than a day
            # Group by hour
            data = data.copy()
            data['hour'] = data.index.floor('H')
            grouped = data.groupby('hour')['sentiment_score'].mean()
            x = grouped.index
            y = grouped.values
            time_format = '%b %d %H:%M'
            title = "Hourly Sentiment Heatmap"
            
        elif time_span.total_seconds() > 3600:  # More than an hour
            # Group by 5-minute intervals
            data = data.copy()
            data['interval'] = data.index.floor('5min')
            grouped = data.groupby('interval')['sentiment_score'].mean()
            x = grouped.index
            y = grouped.values
            time_format = '%H:%M'
            title = "5-Minute Sentiment Heatmap"
            
        else:
            # Group by minute
            data = data.copy()
            data['minute'] = data.index.floor('min')
            grouped = data.groupby('minute')['sentiment_score'].mean()
            x = grouped.index
            y = grouped.values
            time_format = '%H:%M:%S'
            title = "Minute-by-Minute Sentiment Heatmap"
        
        # Create custom colormap: red for negative, white for neutral, green for positive
        cmap = LinearSegmentedColormap.from_list(
            "sentiment_cmap", ["#ff3333", "#ffffff", "#33cc33"]
        )
        
        # Plot the heatmap as a scatter plot with colored markers
        scatter = ax.scatter(
            x, [0] * len(x),
            c=y,
            cmap=cmap,
            s=150,  # marker size
            vmin=-1, vmax=1,  # value range
            marker='s'  # square markers
        )
        
        # Set x-axis to format the dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        plt.xticks(rotation=45)
        
        # Add color bar
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.01)
        cbar.set_label('Sentiment Intensity')
        
        # Set labels and title
        ax.set_title(title)
        ax.set_yticks([])  # Hide y-axis
        
        # Add timeline connecter
        ax.plot(x, [0] * len(x), '-', color='grey', alpha=0.3, zorder=0)
        
        # Add line for neutral sentiment
        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.2, zorder=0)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def create_hourly_sentiment_heatmap(self, data):
        """
        Creates a sentiment heatmap showing patterns by hour of day.
        
        Args:
            data: DataFrame with sentiment data and timestamp index
            
        Returns:
            Matplotlib figure
        """
        # Make sure we have datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                # Not enough data for a meaningful heatmap
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.text(0.5, 0.5, "No timestamp data available for hourly heatmap", 
                        ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                return fig
        
        # Extract hour of day
        data = data.copy()
        data['hour'] = data.index.hour
        
        # Group by hour and calculate average sentiment
        hourly_sentiment = data.groupby('hour')['sentiment_score'].agg(['mean', 'count']).reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create custom colormap: red for negative, white for neutral, green for positive
        cmap = LinearSegmentedColormap.from_list(
            "sentiment_cmap", ["#ff3333", "#ffffff", "#33cc33"]
        )
        
        # Plot the heatmap as a scatter plot with colored markers
        scatter = ax.scatter(
            hourly_sentiment['hour'], 
            [0] * len(hourly_sentiment),
            c=hourly_sentiment['mean'],
            cmap=cmap,
            s=hourly_sentiment['count'] * 10 + 50,  # Size based on count
            vmin=-1, vmax=1,  # value range
            marker='o'  # circular markers
        )
        
        # Set x-axis ticks and labels
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
        
        # Add color bar
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.01)
        cbar.set_label('Sentiment Intensity')
        
        # Set labels and title
        ax.set_title('Hourly Sentiment Pattern')
        ax.set_xlabel('Hour of Day (24h)')
        ax.set_yticks([])  # Hide y-axis
        
        # Add horizontal line for neutral sentiment
        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.2, zorder=0)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add text labels showing the sentiment value
        for _, row in hourly_sentiment.iterrows():
            ax.text(row['hour'], 0, f"{row['mean']:.2f}", 
                    ha='center', va='bottom', fontsize=8)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def create_daily_sentiment_heatmap(self, data):
        """
        Creates a sentiment heatmap showing patterns by day of week.
        
        Args:
            data: DataFrame with sentiment data and timestamp index
            
        Returns:
            Matplotlib figure
        """
        # Make sure we have datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                # Not enough data for a meaningful heatmap
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.text(0.5, 0.5, "No timestamp data available for daily heatmap", 
                        ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                return fig
        
        # Extract day of week
        data = data.copy()
        data['day_of_week'] = data.index.dayofweek  # 0 = Monday, 6 = Sunday
        
        # Group by day of week and calculate average sentiment
        daily_sentiment = data.groupby('day_of_week')['sentiment_score'].agg(['mean', 'count']).reset_index()
        
        # Add any missing days with NaN values
        all_days = pd.DataFrame({'day_of_week': range(7)})
        daily_sentiment = pd.merge(all_days, daily_sentiment, on='day_of_week', how='left')
        daily_sentiment = daily_sentiment.fillna({'mean': 0, 'count': 0})
        
        # Sort by day of week
        daily_sentiment = daily_sentiment.sort_values('day_of_week')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create custom colormap: red for negative, white for neutral, green for positive
        cmap = LinearSegmentedColormap.from_list(
            "sentiment_cmap", ["#ff3333", "#ffffff", "#33cc33"]
        )
        
        # Plot the heatmap as a scatter plot with colored markers
        scatter = ax.scatter(
            daily_sentiment['day_of_week'], 
            [0] * len(daily_sentiment),
            c=daily_sentiment['mean'],
            cmap=cmap,
            s=daily_sentiment['count'] * 10 + 50,  # Size based on count
            vmin=-1, vmax=1,  # value range
            marker='o'  # circular markers
        )
        
        # Set x-axis ticks and labels
        ax.set_xlim(-0.5, 6.5)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Add color bar
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.01)
        cbar.set_label('Sentiment Intensity')
        
        # Set labels and title
        ax.set_title('Daily Sentiment Pattern')
        ax.set_xlabel('Day of Week')
        ax.set_yticks([])  # Hide y-axis
        
        # Add horizontal line for neutral sentiment
        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.2, zorder=0)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add text labels showing the sentiment value and count
        for _, row in daily_sentiment.iterrows():
            if row['count'] > 0:
                ax.text(row['day_of_week'], 0, f"{row['mean']:.2f}\n(n={int(row['count'])})", 
                        ha='center', va='bottom', fontsize=8)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def create_day_hour_heatmap(self, data):
        """
        Creates a 2D heatmap showing sentiment by hour of day and day of week.
        
        Args:
            data: DataFrame with sentiment data and timestamp index
            
        Returns:
            Matplotlib figure
        """
        # Make sure we have datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                # Not enough data for a meaningful heatmap
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.text(0.5, 0.5, "No timestamp data available for heatmap", 
                        ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                return fig
        
        # Extract day of week and hour
        data = data.copy()
        data['day_of_week'] = data.index.dayofweek  # 0 = Monday, 6 = Sunday
        data['hour'] = data.index.hour
        
        # Group by day of week and hour, calculate average sentiment
        heatmap_data = data.groupby(['day_of_week', 'hour'])['sentiment_score'].agg(['mean', 'count']).reset_index()
        
        # Create pivot table for heatmap
        try:
            pivot_table = heatmap_data.pivot_table(index='day_of_week', columns='hour', values='mean')
            count_table = heatmap_data.pivot_table(index='day_of_week', columns='hour', values='count')
        except ValueError:
            # Not enough data points
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.text(0.5, 0.5, "Not enough data for day/hour heatmap", 
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Create custom colormap: red for negative, white for neutral, green for positive
        cmap = LinearSegmentedColormap.from_list(
            "sentiment_cmap", ["#ff3333", "#ffffff", "#33cc33"]
        )
        
        # Create heatmap
        im = ax.imshow(pivot_table, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
        cbar.set_label('Sentiment Intensity')
        
        # Configure axes
        ax.set_xticks(np.arange(24))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(24)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Add a grid
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add text annotations with sentiment values
        for i in range(pivot_table.shape[0]):
            for j in range(pivot_table.shape[1]):
                if not pd.isna(pivot_table.iloc[i, j]):
                    sentiment = pivot_table.iloc[i, j]
                    count = count_table.iloc[i, j] if not pd.isna(count_table.iloc[i, j]) else 0
                    
                    # Only show annotation if we have data
                    if count > 0:
                        text_color = 'black' if -0.3 < sentiment < 0.3 else 'white'
                        text = f"{sentiment:.2f}\n(n={int(count)})" if count > 1 else f"{sentiment:.2f}"
                        ax.text(j, i, text, ha="center", va="center", color=text_color, 
                               fontsize=8, fontweight="bold")
        
        # Add title and labels
        ax.set_title('Sentiment by Day of Week and Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        
        plt.tight_layout()
        
        return fig
    
    def create_anomaly_chart(self, data):
        """
        Creates a visualization of sentiment data with anomalies highlighted.
        
        Args:
            data: DataFrame with anomaly detection results
            
        Returns:
            Altair chart
        """
        # Reset index to get timestamp as column if needed
        if isinstance(data.index, pd.DatetimeIndex):
            plot_data = data.reset_index()
        else:
            plot_data = data.copy()
            if 'timestamp' not in plot_data.columns:
                # Create a dummy timestamp column
                plot_data['timestamp'] = range(len(plot_data))
        
        # Determine time/x-axis field
        x_field = 'timestamp' if 'timestamp' in plot_data.columns else 'index'
        
        # Determine sentiment field
        sentiment_field = 'sentiment_score' if 'sentiment_score' in plot_data.columns else 'sentiment'
        
        # Create base chart with all points
        base = alt.Chart(plot_data).encode(
            x=alt.X(f'{x_field}:T' if isinstance(data.index, pd.DatetimeIndex) else f'{x_field}:Q', 
                    title='Time'),
            y=alt.Y(f'{sentiment_field}:Q', title='Sentiment Score',
                   scale=alt.Scale(domain=[-1, 1]))
        )
        
        # Create line chart for all data
        line = base.mark_line(color='blue').encode(
            tooltip=[x_field, sentiment_field, 'sentiment_label:N']
        )
        
        # Create scatter plot for normal points
        normal_points = base.mark_circle(size=60).encode(
            color=alt.Color('sentiment_label:N', scale=alt.Scale(
                domain=['positive', 'neutral', 'negative'],
                range=['green', 'gray', 'red']
            )),
            tooltip=[x_field, sentiment_field, 'sentiment_label:N']
        ).transform_filter(
            alt.datum.anomaly != True
        )
        
        # Create scatter plot for anomalies
        anomalies = base.mark_circle(size=120, color='black', opacity=0.7).encode(
            tooltip=[
                x_field,
                sentiment_field,
                'sentiment_label:N',
                'anomaly_score:Q',
                'anomaly_explanation:N'
            ]
        ).transform_filter(
            alt.datum.anomaly == True
        )
        
        # Add a stroke to anomalies to make them stand out
        anomaly_strokes = base.mark_circle(size=150, fill=None, stroke='red', strokeWidth=2).encode(
            tooltip=[
                x_field,
                sentiment_field,
                'sentiment_label:N',
                'anomaly_score:Q',
                'anomaly_explanation:N'
            ]
        ).transform_filter(
            alt.datum.anomaly == True
        )
        
        # Combine all layers
        chart = alt.layer(line, normal_points, anomalies, anomaly_strokes).properties(
            title='Sentiment with Anomaly Detection',
            width='container',
            height=400
        )
        
        return chart
    
    def render_export_options(self, data):
        """
        Renders export options for charts.
        
        Args:
            data: DataFrame with sentiment data
        """
        st.subheader("Export Analytics Charts")
        
        # Create columns for different sections
        col1, col2 = st.columns(2)
        
        with col1:
            # Date range selection
            st.subheader("Date Range")
            
            # Get date range options based on the data
            date_ranges = {}
            custom_dates = False
            
            if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                date_ranges = self.chart_filterer.get_date_range_options(data)
                
                # Date range selector
                range_options = list(date_ranges.keys()) + ["Custom Range"]
                selected_range = st.radio("Select Date Range", range_options)
                
                if selected_range == "Custom Range":
                    custom_dates = True
                    
                    # Custom date range inputs
                    min_date = data.index.min().to_pydatetime()
                    max_date = data.index.max().to_pydatetime()
                    
                    from_date = st.date_input(
                        "From Date", 
                        min_date, 
                        min_value=min_date, 
                        max_value=max_date,
                        key=self.key_manager.get_key('analytics_dashboard', 'load_from_date') if self.key_manager else None
                    )
                    to_date = st.date_input(
                        "To Date", 
                        max_date, 
                        min_value=min_date, 
                        max_value=max_date,
                        key=self.key_manager.get_key('analytics_dashboard', 'load_to_date') if self.key_manager else None
                    )
                    
                    # Add time selectors for more precision
                    from_time = st.time_input("From Time", min_date.time())
                    to_time = st.time_input("To Time", max_date.time())
                    
                    # Combine date and time
                    from_datetime = datetime.combine(from_date, from_time)
                    to_datetime = datetime.combine(to_date, to_time)
                    
                else:
                    # Use predefined range
                    from_datetime, to_datetime = date_ranges.get(selected_range, (None, None))
            else:
                st.info("No time-based data available for date filtering.")
                from_datetime, to_datetime = None, None
        
        with col2:
            # Filtering options
            st.subheader("Data Filtering")
            
            # Sentiment filter
            if 'sentiment_label' in data.columns:
                unique_sentiments = data['sentiment_label'].unique()
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=unique_sentiments,
                    default=unique_sentiments
                )
            else:
                sentiment_filter = None
                
            # Emotion filter
            if 'emotion_label' in data.columns:
                unique_emotions = data['emotion_label'].unique()
                emotion_filter = st.multiselect(
                    "Filter by Emotion",
                    options=unique_emotions,
                    default=unique_emotions
                )
            else:
                emotion_filter = None
                
            # Confidence threshold
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        # Apply filters
        filtered_data = self.chart_filterer.filter_data(
            data,
            from_date=from_datetime,
            to_date=to_datetime,
            sentiment_filter=sentiment_filter,
            emotion_filter=emotion_filter,
            min_confidence=min_confidence if min_confidence > 0 else None
        )
        
        # Show filter summary
        if len(filtered_data) != len(data):
            st.info(f"Filter applied: {len(filtered_data)} out of {len(data)} data points selected")
        
        # Export format selection
        st.subheader("Export Format")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.radio("Format", ["PNG", "PDF"])
            
        with col2:
            chart_size = st.radio("Chart Size", ["Standard", "Large", "Custom"])
            
            if chart_size == "Custom":
                width = st.number_input("Width (pixels)", min_value=400, max_value=2000, value=800, step=100)
                height = st.number_input("Height (pixels)", min_value=300, max_value=1500, value=600, step=100)
            else:
                width = 800 if chart_size == "Standard" else 1200
                height = 600 if chart_size == "Standard" else 900
        
        with col3:
            # Add custom title/notes
            include_title = st.checkbox("Include Custom Title", value=True)
            
            if include_title:
                chart_title = st.text_input("Chart Title", "Sentiment Analysis Report")
            else:
                chart_title = "Sentiment Analysis Report"
                
            include_notes = st.checkbox("Include Notes", value=False)
            
            if include_notes:
                notes = st.text_area("Notes", height=100)
            else:
                notes = None
        
        # Select which charts to export
        st.subheader("Charts to Include")
        
        include_trend = st.checkbox("Sentiment Trend Chart", value=True)
        include_distribution = st.checkbox("Sentiment Distribution", value=True)
        include_heatmap = st.checkbox("Sentiment Heatmap", value=True)
        
        # Preview and export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Preview Charts", use_container_width=True):
                # Show preview of the charts
                self.show_chart_preview(filtered_data, include_trend, include_distribution, include_heatmap)
        
        with col2:
            if st.button("Export Charts", use_container_width=True):
                # Generate and export the charts
                self.export_charts(
                    filtered_data, 
                    format=export_format.lower(),
                    width=width,
                    height=height,
                    charts={
                        'trend': include_trend,
                        'distribution': include_distribution,
                        'heatmap': include_heatmap
                    },
                    title=chart_title,
                    notes=notes
                )
    
    def render_export_options_real_time(self, data):
        """
        Renders export options for real-time analytics with unique keys.
        
        Args:
            data: DataFrame to export
        """
        st.subheader("Export Configuration")
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=data.index.min().date() if hasattr(data.index, 'min') and data.index.min() else datetime.now().date(),
                key=self.key_manager.get_key('real_time_analytics', 'export_start_date') if self.key_manager else None
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=data.index.max().date() if hasattr(data.index, 'max') and data.index.max() else datetime.now().date(),
                key=self.key_manager.get_key('real_time_analytics', 'export_end_date') if self.key_manager else None
            )
        
        # Data filtering options
        st.subheader("Data Filtering")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.multiselect(
                "Sentiment Filter",
                options=['positive', 'neutral', 'negative'],
                default=['positive', 'neutral', 'negative'],
                key=self.key_manager.get_key('real_time_analytics', 'export_sentiment_filter') if self.key_manager else None
            )
        
        with col2:
            emotion_filter = st.multiselect(
                "Emotion Filter",
                options=['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                default=['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                key=self.key_manager.get_key('real_time_analytics', 'export_emotion_filter') if self.key_manager else None
            )
        
        with col3:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key=self.key_manager.get_key('real_time_analytics', 'export_confidence') if self.key_manager else None
            )
        
        # Apply filters
        filtered_data = data.copy()
        
        # Filter by date range
        if hasattr(filtered_data.index, 'date'):
            filtered_data = filtered_data[
                (filtered_data.index.date >= start_date) & 
                (filtered_data.index.date <= end_date)
            ]
        
        # Filter by sentiment
        if sentiment_filter and 'sentiment_label' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['sentiment_label'].isin(sentiment_filter)]
        
        # Filter by emotion
        if emotion_filter and 'emotion_label' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['emotion_label'].isin(emotion_filter)]
        
        # Filter by confidence
        if 'sentiment_confidence' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['sentiment_confidence'] >= confidence_threshold]
        
        # Export format and settings
        st.subheader("Export Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=["PNG", "PDF"],
                index=0,
                key=self.key_manager.get_key('real_time_analytics', 'export_format') if self.key_manager else None
            )
        
        with col2:
            chart_size = st.selectbox(
                "Chart Size",
                options=["Small (600x400)", "Medium (800x600)", "Large (1200x800)"],
                index=1,
                key=self.key_manager.get_key('real_time_analytics', 'export_size') if self.key_manager else None
            )
        
        # Parse chart size
        size_map = {
            "Small (600x400)": (600, 400),
            "Medium (800x600)": (800, 600),
            "Large (1200x800)": (1200, 800)
        }
        width, height = size_map[chart_size]
        
        # Custom title and notes
        chart_title = st.text_input(
            "Chart Title",
            value="Sentiment Analysis Report",
            key=self.key_manager.get_key('real_time_analytics', 'export_title') if self.key_manager else None
        )
        
        notes = st.text_area(
            "Additional Notes (optional)",
            height=100,
            key=self.key_manager.get_key('real_time_analytics', 'export_notes') if self.key_manager else None
        )
        
        # Select which charts to export
        st.subheader("Charts to Include")
        
        include_trend = st.checkbox(
            "Sentiment Trend Chart", 
            value=True,
            key=self.key_manager.get_key('real_time_analytics', 'export_trend') if self.key_manager else None
        )
        include_distribution = st.checkbox(
            "Sentiment Distribution", 
            value=True,
            key=self.key_manager.get_key('real_time_analytics', 'export_distribution') if self.key_manager else None
        )
        include_heatmap = st.checkbox(
            "Sentiment Heatmap", 
            value=True,
            key=self.key_manager.get_key('real_time_analytics', 'export_heatmap') if self.key_manager else None
        )
        
        # Preview and export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "Preview Charts", 
                use_container_width=True,
                key=self.key_manager.get_key('real_time_analytics', 'preview_charts') if self.key_manager else None
            ):
                # Show preview of the charts
                self.show_chart_preview(filtered_data, include_trend, include_distribution, include_heatmap)
        
        with col2:
            if st.button(
                "Export Charts", 
                use_container_width=True,
                key=self.key_manager.get_key('real_time_analytics', 'export_charts_btn') if self.key_manager else None
            ):
                # Generate and export the charts
                self.export_charts(
                    filtered_data, 
                    format=export_format.lower(),
                    width=width,
                    height=height,
                    charts={
                        'trend': include_trend,
                        'distribution': include_distribution,
                        'heatmap': include_heatmap
                    },
                    title=chart_title,
                    notes=notes
                )
    
    def show_chart_preview(self, data, include_trend, include_distribution, include_heatmap):
        """
        Shows a preview of the charts that will be exported.
        
        Args:
            data: Filtered DataFrame
            include_trend: Whether to include trend chart
            include_distribution: Whether to include distribution chart
            include_heatmap: Whether to include heatmap
        """
        st.subheader("Chart Preview")
        
        if len(data) == 0:
            st.warning("No data available for the selected filters.")
            return
        
        # Create and display the charts
        if include_trend:
            st.subheader("Sentiment Trend")
            trend_chart = self.create_real_time_trend_chart(data)
            st.altair_chart(trend_chart, use_container_width=True)
        
        if include_distribution:
            st.subheader("Sentiment Distribution")
            
            # Create sentiment distribution chart
            if 'sentiment_label' in data.columns:
                sentiment_counts = data['sentiment_label'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                dist_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                    x='Sentiment:N',
                    y='Count:Q',
                    color=alt.Color('Sentiment:N', scale=alt.Scale(
                        domain=['positive', 'neutral', 'negative'],
                        range=['green', 'gray', 'red']
                    )),
                    tooltip=['Sentiment', 'Count']
                ).properties(
                    title='Sentiment Distribution'
                )
                
                st.altair_chart(dist_chart, use_container_width=True)
        
        if include_heatmap:
            st.subheader("Sentiment Heatmap")
            
            # Create time-based heatmap using matplotlib
            fig = self.create_sentiment_time_heatmap(data)
            st.pyplot(fig)
    
    def export_charts(self, data, format="png", width=800, height=600, charts=None,
                     title="Sentiment Analysis Report", notes=None):
        """
        Exports the selected charts in the specified format.
        
        Args:
            data: Filtered DataFrame
            format: Export format ('png' or 'pdf')
            width: Chart width in pixels
            height: Chart height in pixels
            charts: Dict indicating which charts to include
            title: Chart title
            notes: Optional notes to include
        """
        if len(data) == 0:
            st.error("No data available to export with the current filters.")
            return
        
        # Prepare charts to export
        charts_to_export = []
        
        # Create trend chart if requested
        if charts.get('trend', False):
            trend_chart = self.create_real_time_trend_chart(data)
            charts_to_export.append((trend_chart, "Sentiment Trend Over Time"))
        
        # Create distribution chart if requested
        if charts.get('distribution', False) and 'sentiment_label' in data.columns:
            sentiment_counts = data['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            dist_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                x='Sentiment:N',
                y='Count:Q',
                color=alt.Color('Sentiment:N', scale=alt.Scale(
                    domain=['positive', 'neutral', 'negative'],
                    range=['green', 'gray', 'red']
                )),
                tooltip=['Sentiment', 'Count']
            ).properties(
                title='Sentiment Distribution'
            )
            
            charts_to_export.append((dist_chart, "Sentiment Distribution"))
            
            # Add emotion distribution if available
            if 'emotion_label' in data.columns:
                emotion_counts = data['emotion_label'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Count']
                
                emotion_chart = alt.Chart(emotion_counts).mark_bar().encode(
                    x='Emotion:N',
                    y='Count:Q',
                    color='Emotion:N',
                    tooltip=['Emotion', 'Count']
                ).properties(
                    title='Emotion Distribution'
                )
                
                charts_to_export.append((emotion_chart, "Emotion Distribution"))
        
        # For heatmap, we need to use the chartExporter's method directly
        # since it's a matplotlib figure rather than an Altair chart
        
        # Generate subtitle with date range info
        subtitle = None
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
            min_date = data.index.min().strftime("%Y-%m-%d %H:%M")
            max_date = data.index.max().strftime("%Y-%m-%d %H:%M")
            subtitle = f"Date range: {min_date} to {max_date}"
        
        # Export based on format
        if len(charts_to_export) > 0:
            try:
                # For single chart export
                if len(charts_to_export) == 1:
                    chart, chart_title = charts_to_export[0]
                    
                    # Add the heatmap to the PDF separately if needed
                    if format == "pdf" and charts.get('heatmap', False):
                        # First export the Altair chart
                        output = self.chart_exporter.export_chart(
                            chart, 
                            format=format,
                            width=width,
                            height=height,
                            filename=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
                        )
                        
                        # Create a dashboard with both the Altair chart and heatmap
                        heatmap_fig = self.create_sentiment_time_heatmap(data)
                        
                        # For PDF, we can't directly add the matplotlib figure
                        # Save it as a temp file
                        temp_heatmap = os.path.join(self.chart_exporter.temp_dir, "heatmap.png")
                        heatmap_fig.savefig(temp_heatmap, bbox_inches='tight')
                        
                        # Create a PDF with both visualizations
                        pdf = FPDF()
                        pdf.add_page()
                        
                        # Add title
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(0, 10, title, ln=True, align='C')
                        
                        if subtitle:
                            pdf.set_font("Arial", 'I', 12)
                            pdf.cell(0, 10, subtitle, ln=True, align='C')
                        
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, chart_title, ln=True)
                        
                        # Add the Altair chart from the temp file
                        temp_chart = os.path.join(self.chart_exporter.temp_dir, "chart_temp.png")
                        with open(temp_chart, "wb") as f:
                            f.write(self.chart_exporter.export_chart(chart, format="png").getvalue())
                        
                        pdf.image(temp_chart, x=10, w=190)
                        pdf.add_page()
                        
                        # Add the heatmap
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Sentiment Heatmap", ln=True)
                        pdf.image(temp_heatmap, x=10, w=190)
                        
                        # Add notes if provided
                        if notes:
                            pdf.add_page()
                            pdf.set_font("Arial", 'B', 12)
                            pdf.cell(0, 10, "Notes", ln=True)
                            pdf.set_font("Arial", size=10)
                            pdf.multi_cell(0, 10, notes)
                        
                        # Save to BytesIO
                        output = io.BytesIO()
                        output.write(pdf.output(dest='S').encode('latin1'))
                        output.seek(0)
                        
                        # Clean up temp files
                        try:
                            os.remove(temp_chart)
                            os.remove(temp_heatmap)
                        except:
                            pass
                    
                    else:
                        # Regular single chart export
                        output = self.chart_exporter.export_chart(
                            chart, 
                            format=format,
                            width=width,
                            height=height,
                            filename=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
                        )
                
                # For multi-chart export
                else:
                    # Export as dashboard
                    output = self.chart_exporter.export_dashboard(
                        charts_to_export,
                        title=title,
                        format=format,
                        subtitle=subtitle,
                        notes=notes
                    )
                
                # Create download button
                if format == "pdf":
                    mime = "application/pdf"
                    extension = "pdf"
                else:  # png
                    mime = "image/png"
                    extension = "png"
                
                # Generate filename
                filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.{extension}"
                
                # Provide download button
                st.download_button(
                    label=f"Download {format.upper()}",
                    data=output,
                    file_name=filename,
                    mime=mime
                )
                
                st.success(f"Charts exported successfully! Click the download button to save.")
                
            except Exception as e:
                st.error(f"Error exporting charts: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    def render_save_load_buttons(self):
        """
        Renders save and load buttons for analysis persistence.
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "💾 Save Analysis", 
                help="Save current analysis results to local storage",
                key=self.key_manager.get_key('analytics_dashboard', 'save_analysis') if self.key_manager else None
            ):
                self.show_save_dialog()
        
        with col2:
            if st.button(
                "📂 Load Saved Analysis", 
                help="Load previously saved analysis results",
                key=self.key_manager.get_key('analytics_dashboard', 'load_analysis') if self.key_manager else None
            ):
                self.show_load_dialog()
        
        with col3:
            if st.button(
                "📊 Manage Saved Analyses", 
                help="View and manage all saved analyses",
                key=self.key_manager.get_key('analytics_dashboard', 'manage_analyses') if self.key_manager else None
            ):
                self.show_manage_dialog()
    
    def render_save_load_buttons_real_time(self):
        """
        Renders save and load buttons for real-time analytics persistence.
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "💾 Save Analysis", 
                help="Save current analysis results to local storage",
                key=self.key_manager.get_key('real_time_analytics', 'save_analysis') if self.key_manager else None
            ):
                self.show_save_dialog()
        
        with col2:
            if st.button(
                "📂 Load Saved Analysis", 
                help="Load previously saved analysis results",
                key=self.key_manager.get_key('real_time_analytics', 'load_analysis') if self.key_manager else None
            ):
                self.show_load_dialog()
        
        with col3:
            if st.button(
                "📊 Manage Saved Analyses", 
                help="View and manage all saved analyses",
                key=self.key_manager.get_key('real_time_analytics', 'manage_analyses') if self.key_manager else None
            ):
                self.show_manage_dialog()
    
    def show_save_dialog(self):
        """
        Shows a dialog for saving the current analysis.
        """
        st.session_state.show_save_dialog = True
    
    def show_load_dialog(self):
        """
        Shows a dialog for loading a saved analysis.
        """
        st.session_state.show_load_dialog = True
    
    def show_manage_dialog(self):
        """
        Shows a dialog for managing saved analyses.
        """
        st.session_state.show_manage_dialog = True
    
    def render_save_dialog(self):
        """
        Renders the save analysis dialog.
        """
        with st.form(key=self.key_manager.get_key('analytics_dashboard', 'save_form') if self.key_manager else "save_analysis_form"):
            st.subheader("Save Analysis")
            
            # Get current data
            data = None
            
            # Try to get analytics data
            if hasattr(self.session_state, 'analytics_data') and self.session_state.analytics_data is not None:
                data = self.session_state.analytics_data
            
            # If no analytics data, try real-time connector
            elif hasattr(self.session_state, 'real_time_connector'):
                real_time_data = self.session_state.real_time_connector.get_latest_data()
                if real_time_data:
                    data = self.session_state.real_time_connector._convert_to_dataframe(real_time_data)
            
            if data is None or len(data) == 0:
                st.warning("No analysis data available to save.")
                if st.form_submit_button("Close"):
                    st.session_state.show_save_dialog = False
                return
            
            # Input fields for save metadata
            name = st.text_input(
                "Analysis Name", 
                value=f"Analysis {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                key=self.key_manager.get_key('analytics_dashboard', 'save_name') if self.key_manager else None
            )
            description = st.text_area(
                "Description (optional)", 
                height=100,
                key=self.key_manager.get_key('analytics_dashboard', 'save_description') if self.key_manager else None
            )
            tags_input = st.text_input(
                "Tags (comma-separated, optional)",
                key=self.key_manager.get_key('analytics_dashboard', 'save_tags') if self.key_manager else None
            )
            
            tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
            
            # Show data summary
            st.subheader("Data Summary")
            st.write(f"Number of data points: {len(data)}")
            
            if isinstance(data, pd.DataFrame):
                if isinstance(data.index, pd.DatetimeIndex):
                    st.write(f"Date range: {data.index.min()} to {data.index.max()}")
                
                # Show column information
                st.write(f"Columns: {', '.join(data.columns)}")
                
                # Show sentiment distribution if available
                if 'sentiment_label' in data.columns:
                    sentiment_counts = data['sentiment_label'].value_counts()
                    st.write("Sentiment distribution:")
                    for label, count in sentiment_counts.items():
                        st.write(f"- {label}: {count} ({(count/len(data))*100:.1f}%)")
            
            # Save and cancel buttons
            col1, col2 = st.columns(2)
            
            with col1:
                save_button = st.form_submit_button(
                    "Save Analysis",
                    key=self.key_manager.get_key('analytics_dashboard', 'save_submit') if self.key_manager else None
                )
            
            with col2:
                cancel_button = st.form_submit_button(
                    "Cancel",
                    key=self.key_manager.get_key('analytics_dashboard', 'save_cancel') if self.key_manager else None
                )
            
            if save_button:
                try:
                    # Save the analysis
                    analysis_id = self.storage.save_analysis(data, name, description, tags)
                    
                    st.success(f"Analysis saved successfully with ID: {analysis_id}")
                    st.session_state.show_save_dialog = False
                    
                    # Store the last saved ID for reference
                    st.session_state.last_saved_analysis_id = analysis_id
                except Exception as e:
                    st.error(f"Error saving analysis: {str(e)}")
            
            if cancel_button:
                st.session_state.show_save_dialog = False
    
    def render_load_dialog(self):
        """
        Renders the load saved analysis dialog.
        """
        st.subheader("Load Saved Analysis")
        
        # Get all saved analyses
        analyses = self.storage.get_all_analyses()
        
        if not analyses:
            st.info("No saved analyses found.")
            if st.button("Close"):
                st.session_state.show_load_dialog = False
            return
        
        # Search and filter options
        with st.expander("Search and Filter Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                search_query = st.text_input("Search by name or description")
            
            with col2:
                tags_input = st.text_input("Filter by tags (comma-separated)")
                tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
            
            # Apply filters
            if search_query or tags:
                analyses = self.storage.search_analyses(query=search_query, tags=tags)
                
                if not analyses:
                    st.info("No analyses match the search criteria.")
                else:
                    st.success(f"Found {len(analyses)} matching analyses.")
        
        # Display analyses as cards
        for analysis in analyses:
            with st.expander(f"{analysis['name']} ({analysis['timestamp'][:10]})", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display analysis metadata
                    st.write(f"**Description:** {analysis['description'] or 'N/A'}")
                    st.write(f"**Created:** {analysis['timestamp']}")
                    
                    if analysis.get('tags'):
                        st.write(f"**Tags:** {', '.join(analysis['tags'])}")
                
                with col2:
                    # Load button
                    if st.button("Load", key=f"load_{analysis['id']}"):
                        try:
                            # Load the analysis
                            loaded = self.storage.load_analysis(analysis['id'])
                            
                            # Store in session state
                            self.session_state.analytics_data = loaded['data']
                            
                            # Set flag to indicate data is from saved analysis
                            self.session_state.data_source = 'saved_analysis'
                            self.session_state.loaded_analysis_id = analysis['id']
                            self.session_state.loaded_analysis_name = analysis['name']
                            
                            st.success(f"Analysis '{analysis['name']}' loaded successfully!")
                            st.session_state.show_load_dialog = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading analysis: {str(e)}")
        
        # Close button
        if st.button("Close", key="close_load_dialog"):
            st.session_state.show_load_dialog = False
    
    def render_manage_dialog(self):
        """
        Renders the manage saved analyses dialog.
        """
        st.subheader("Manage Saved Analyses")
        
        # Get all saved analyses
        analyses = self.storage.get_all_analyses()
        
        if not analyses:
            st.info("No saved analyses found.")
            if st.button("Close", key="close_manage_empty"):
                st.session_state.show_manage_dialog = False
            return
        
        # Create a table of analyses
        table_data = []
        for analysis in analyses:
            # Format date
            try:
                timestamp = datetime.fromisoformat(analysis['timestamp'])
                date_str = timestamp.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = analysis['timestamp']
            
            # Create table row
            table_data.append({
                "ID": analysis['id'],
                "Name": analysis['name'],
                "Description": analysis['description'] or 'N/A',
                "Created": date_str,
                "Tags": ", ".join(analysis['tags']) if analysis['tags'] else 'N/A'
            })
        
        # Display as dataframe
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        # Export/Import options
        st.subheader("Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export all analyses
            if st.button("Export All Analyses", help="Export all analyses to a file", use_container_width=True):
                try:
                    # Create export data
                    export_data = []
                    for analysis in analyses:
                        try:
                            # Load the full analysis
                            loaded = self.storage.load_analysis(analysis['id'])
                            export_data.append({
                                "metadata": analysis,
                                "data": loaded['data'].to_dict() if isinstance(loaded['data'], pd.DataFrame) else loaded['data']
                            })
                        except:
                            # Skip analyses that fail to load
                            continue
                    
                    # Convert to JSON
                    export_json = json.dumps(export_data, default=str)
                    
                    # Create download button
                    st.download_button(
                        label="Download Export",
                        data=export_json,
                        file_name=f"sentiment_analyses_export_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting analyses: {str(e)}")
        
        with col2:
            # Import analyses
            uploaded_file = st.file_uploader("Import Analyses", type="json")
            
            if uploaded_file is not None:
                try:
                    # Read the file
                    import_data = json.load(uploaded_file)
                    
                    # Validate format
                    if not isinstance(import_data, list):
                        st.error("Invalid import format. Expected a list of analyses.")
                    else:
                        # Import each analysis
                        imported_count = 0
                        for item in import_data:
                            try:
                                if 'metadata' in item and 'data' in item:
                                    # Extract metadata
                                    metadata = item['metadata']
                                    name = metadata.get('name', f"Imported {datetime.now().strftime('%Y-%m-%d')}")
                                    description = metadata.get('description', "Imported analysis")
                                    tags = metadata.get('tags', [])
                                    
                                    # Convert data back to DataFrame if needed
                                    data = item['data']
                                    if isinstance(data, dict) and 'index' in data:
                                        data = pd.DataFrame(data)
                                    
                                    # Save with new ID
                                    self.storage.save_analysis(data, name, description, tags)
                                    imported_count += 1
                            except:
                                # Skip items that fail
                                continue
                        
                        st.success(f"Successfully imported {imported_count} analyses.")
                        
                        # Refresh the dialog
                        st.rerun()
                except Exception as e:
                    st.error(f"Error importing analyses: {str(e)}")
        
        # Delete options
        st.subheader("Delete Options")
        
        # Select analysis to delete
        selected_analysis = st.selectbox(
            "Select analysis to delete",
            options=[f"{a['name']} ({a['timestamp'][:10]})" for a in analyses],
            format_func=lambda x: x
        )
        
        # Find the selected analysis
        selected_index = [f"{a['name']} ({a['timestamp'][:10]})" for a in analyses].index(selected_analysis)
        analysis_to_delete = analyses[selected_index]
        
        # Show delete confirmation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Delete Selected", use_container_width=True):
                st.session_state.confirm_delete = analysis_to_delete['id']
        
        with col2:
            if st.button("Delete All", help="Delete all saved analyses", use_container_width=True):
                st.session_state.confirm_delete_all = True
        
        # Handle delete confirmation
        if 'confirm_delete' in st.session_state and st.session_state.confirm_delete:
            st.warning(f"Are you sure you want to delete '{analysis_to_delete['name']}'? This cannot be undone.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, Delete", use_container_width=True):
                    try:
                        # Delete the analysis
                        if self.storage.delete_analysis(st.session_state.confirm_delete):
                            st.success(f"Analysis deleted successfully.")
                            # Clear the confirmation
                            st.session_state.confirm_delete = None
                            # Refresh the dialog
                            st.rerun()
                        else:
                            st.error(f"Error deleting analysis.")
                    except Exception as e:
                        st.error(f"Error deleting analysis: {str(e)}")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    # Clear the confirmation
                    st.session_state.confirm_delete = None
                    st.rerun()
        
        # Handle delete all confirmation
        if 'confirm_delete_all' in st.session_state and st.session_state.confirm_delete_all:
            st.warning("Are you sure you want to delete ALL saved analyses? This cannot be undone.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, Delete All", use_container_width=True):
                    try:
                        # Delete all analyses
                        deleted_count = 0
                        for analysis in analyses:
                            if self.storage.delete_analysis(analysis['id']):
                                deleted_count += 1
                        
                        st.success(f"Successfully deleted {deleted_count} analyses.")
                        # Clear the confirmation
                        st.session_state.confirm_delete_all = None
                        # Refresh the dialog
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting analyses: {str(e)}")
            
            with col2:
                if st.button("Cancel Delete", use_container_width=True):
                    # Clear the confirmation
                    st.session_state.confirm_delete_all = None
                    st.rerun()
        
        # Close button
        if st.button("Close", key="close_manage_dialog"):
            st.session_state.show_manage_dialog = False
    
    def render_anomaly_analysis(self):
        """
        Renders the anomaly analysis section with various detection methods.
        """
        st.header("Sentiment Anomaly Detection")
        
        # Load data
        analysis_data = self._get_analysis_data()
        if not analysis_data:
            st.warning("No analysis data available for anomaly detection.")
            return
            
        data = self._convert_to_dataframe(analysis_data)
        if data is None or len(data) < 10:
            st.warning("Not enough data available for anomaly detection. Need at least 10 data points.")
            return
        
        # Detection method selection
        col1, col2 = st.columns(2)
        
        with col1:
            detection_method = st.selectbox(
                "Anomaly Detection Method",
                options=['isolation_forest', 'zscore', 'dbscan', 'seasonal'],
                index=0,
                help=("isolation_forest: Good for complex anomalies, "
                      "zscore: Simple statistical outliers, "
                      "dbscan: Density-based clustering anomalies, "
                      "seasonal: Time-series specific anomalies")
            )
        
        with col2:
            contamination = st.slider(
                "Anomaly Ratio",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Expected proportion of anomalies (lower = fewer anomalies detected)"
            )
        
        # Set up detector with new parameters
        self.anomaly_detector = SentimentAnomalyDetector(contamination=contamination)
        
        # Run anomaly detection
        with st.spinner("Detecting anomalies..."):
            df_with_anomalies = self.anomaly_detector.detect_anomalies(data, method=detection_method)
            anomaly_count = df_with_anomalies['anomaly'].sum()
        
        # Show results
        if anomaly_count > 0:
            st.success(f"Detected {anomaly_count} anomalies ({(anomaly_count/len(data))*100:.1f}% of data)")
            
            # Generate and display insights
            anomaly_insights = self.anomaly_detector.generate_insights(df_with_anomalies)
            
            # Create visualization
            anomaly_chart = self.create_anomaly_chart(df_with_anomalies)
            st.altair_chart(anomaly_chart, use_container_width=True)
            
            # Display anomaly insights
            with st.expander("Anomaly Insights", expanded=True):
                st.write(anomaly_insights['message'])
                
                st.subheader("Details")
                for detail in anomaly_insights['details']:
                    st.write(f"- {detail}")
                
                if anomaly_insights['recommendations']:
                    st.subheader("Recommendations")
                    for rec in anomaly_insights['recommendations']:
                        st.write(f"- {rec}")
            
            # Display table of detected anomalies
            st.subheader("Anomaly Details")
            anomaly_df = df_with_anomalies[df_with_anomalies['anomaly']].copy()
            
            # Format for display
            display_cols = ['timestamp', 'sentiment_score', 'sentiment_label', 
                           'anomaly_score', 'anomaly_explanation']
            
            if isinstance(anomaly_df.index, pd.DatetimeIndex):
                anomaly_df = anomaly_df.reset_index()
                if 'index' in anomaly_df.columns:
                    anomaly_df.rename(columns={'index': 'timestamp'}, inplace=True)
            
            display_df = anomaly_df[
                [col for col in display_cols if col in anomaly_df.columns]
            ].sort_values('anomaly_score', ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Option to download anomaly data
            csv = display_df.to_csv(index=False)
            st.download_button(
                "Download Anomaly Data",
                csv,
                "sentiment_anomalies.csv",
                "text/csv",
                key='download-anomalies'
            )
            
        else:
            st.info("No anomalies detected with the current settings.")
            
            # Suggest adjustments
            st.write("Suggestions if you expected to find anomalies:")
            st.write("- Increase the 'Anomaly Ratio' to detect more potential anomalies")
            st.write("- Try a different detection method (e.g., 'zscore' for simple outliers)")
            st.write("- Check if your data contains unusual patterns that would qualify as anomalies")
    
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
                        item['timestamp'] = datetime.now()
                    
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
        
        # Text column selection
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
        if not text_columns:
            text_columns = ['text']  # Default fallback
        
        selected_text_col = st.selectbox(
            "Select Text Column:",
            text_columns,
            index=0,
            key=self.key_manager.get_key('analytics_dashboard', 'trend_text_col') if self.key_manager else 'trend_text_col'
        )
        
        selected_time_col = None
        if time_columns:
            selected_time_col = st.selectbox(
                "Select Time Column (optional):",
                ['None'] + time_columns,
                index=0,
                key=self.key_manager.get_key('analytics_dashboard', 'trend_time_col') if self.key_manager else 'trend_time_col'
            )
            if selected_time_col == 'None':
                selected_time_col = None
        
        # Window size for trend smoothing
        window_size = st.slider(
            "Trend Smoothing Window:",
            min_value=1,
            max_value=10,
            value=3,
            help="Size of sliding window for trend smoothing"
        )
        
        if st.button("Analyze Trends", type="primary"):
            with st.spinner("Analyzing trends..."):
                try:
                    # Perform trend analysis using new comprehensive method
                    trend_results = self.analytics.analyze_trends(
                        data=df,
                        time_column=selected_time_col if selected_time_col != 'None' else None
                    )
                    
                    # Display results
                    self._display_trend_results(trend_results)
                    
                except Exception as e:
                    st.error(f"Error performing trend analysis: {str(e)}")
    
    def _display_trend_results(self, results: Dict[str, Any]) -> None:
        """Display trend analysis results."""
        st.subheader("Trend Analysis Results")
        
        # Check for errors
        if "error" in results:
            st.error(f"❌ {results['error']}")
            return
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "sample_size" in results:
                st.metric("Sample Size", results["sample_size"])
        
        with col2:
            if "time_range" in results:
                duration_days = results["time_range"]["duration"] / (24 * 3600)
                st.metric("Duration (Days)", f"{duration_days:.1f}")
        
        with col3:
            if "trends" in results and results["trends"]:
                trend_count = len(results["trends"])
                st.metric("Trend Periods", trend_count)
        
        # Display trends
        if "trends" in results and results["trends"]:
            st.subheader("Trend Analysis")
            
            for period, trend_data in results["trends"].items():
                with st.expander(f"{period.title()} Trends", expanded=True):
                    if "trend_direction" in trend_data:
                        st.metric("Trend Direction", trend_data["trend_direction"])
                    if "trend_strength" in trend_data:
                        st.metric("Trend Strength", f"{trend_data['trend_strength']:.3f}")
                    if "volatility" in trend_data:
                        st.metric("Volatility", f"{trend_data['volatility']:.3f}")
        
        # Display seasonal patterns
        if "seasonal_patterns" in results and results["seasonal_patterns"]:
            st.subheader("Seasonal Patterns")
            
            for pattern_type, pattern_data in results["seasonal_patterns"].items():
                with st.expander(f"{pattern_type.title()} Patterns", expanded=False):
                    if isinstance(pattern_data, dict):
                        for key, value in pattern_data.items():
                            if isinstance(value, (int, float)):
                                st.metric(key.replace("_", " ").title(), f"{value:.3f}")
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Display correlations
        if "correlations" in results and results["correlations"]:
            st.subheader("Correlations")
            
            for corr_type, corr_data in results["correlations"].items():
                with st.expander(f"{corr_type.title()} Correlations", expanded=False):
                    if isinstance(corr_data, dict):
                        for key, value in corr_data.items():
                            if isinstance(value, (int, float)):
                                st.metric(key.replace("_", " ").title(), f"{value:.3f}")
                            else:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Display time range info
        if "time_range" in results:
            st.subheader("Time Range")
            time_range = results["time_range"]
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Start:** {time_range['start']}")
            
            with col2:
                st.write(f"**End:** {time_range['end']}")
            st.write("**Sentiment Distribution:**")
            
            # Create sentiment distribution chart
            fig = go.Figure(data=[
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    marker_color=['green', 'red', 'gray']
                )
            ])
            
            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_clustering_analysis(self, df: pd.DataFrame) -> None:
        """Render clustering analysis tab."""
        st.subheader("🔍 Clustering Analysis")
        
        # Text column selection
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
        if not text_columns:
            text_columns = ['text']  # Default fallback
        
        selected_text_col = st.selectbox(
            "Select Text Column:",
            text_columns,
            index=0,
            key=self.key_manager.get_key('analytics_dashboard', 'cluster_text_col') if self.key_manager else 'cluster_text_col'
        )
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
        
        with col2:
            clustering_method = st.selectbox(
                "Clustering Method:",
                ["kmeans", "dbscan", "tsne"],
                index=0,
                key=self.key_manager.get_key('analytics_dashboard', 'cluster_method') if self.key_manager else 'cluster_method'
            )
        
        if st.button("Perform Clustering", type="primary"):
            with st.spinner("Performing clustering analysis..."):
                try:
                    # Perform clustering using new comprehensive method
                    clustering_results = self.analytics.cluster_sentiments(
                        data=df,
                        text_column=selected_text_col,
                        n_clusters=n_clusters,
                        method=clustering_method
                    )
                    
                    # Display results
                    self._display_clustering_results(clustering_results)
                    
                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")
    
    def _display_clustering_results(self, results: Dict[str, Any]) -> None:
        """Display clustering analysis results."""
        st.subheader("Clustering Results")
        
        # Display clustering method and parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Clustering Method", results.get('method', 'Unknown').upper())
        
        with col2:
            st.metric("Number of Clusters", results.get('n_clusters', 0))
        
        # Cluster analysis
        if "cluster_analysis" in results:
            st.subheader("Cluster Analysis")
            
            cluster_analysis = results["cluster_analysis"]
            
            # Create cluster summary
            cluster_data = []
            for cluster_id, analysis in cluster_analysis.items():
                cluster_data.append({
                    "Cluster ID": cluster_id,
                    "Size": analysis.get('size', 0),
                    "Avg Sentiment": f"{analysis.get('avg_sentiment', 0):.3f}",
                    "Sample Texts": len(analysis.get('sample_texts', []))
                })
            
            if cluster_data:
                cluster_df = pd.DataFrame(cluster_data)
                st.write("**Cluster Summary:**")
                st.dataframe(cluster_df, use_container_width=True)
                
                # Create cluster size visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Cluster {d['Cluster ID']}" for d in cluster_data],
                        y=[d['Size'] for d in cluster_data],
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title="Cluster Sizes",
                    xaxis_title="Cluster",
                    yaxis_title="Number of Records"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed cluster information
            st.subheader("Detailed Cluster Information")
            
            for cluster_id, analysis in cluster_analysis.items():
                with st.expander(f"Cluster {cluster_id} (Size: {analysis.get('size', 0)})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Cluster Statistics:**")
                        st.write(f"- Size: {analysis.get('size', 0)}")
                        st.write(f"- Average Sentiment: {analysis.get('avg_sentiment', 0):.3f}")
                    
                    with col2:
                        st.write("**Sample Texts:**")
                        sample_texts = analysis.get('sample_texts', [])
                        for i, text in enumerate(sample_texts[:3]):  # Show first 3
                            st.write(f"{i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
                    
                    if sample_texts:
                        st.write("**All Sample Texts:**")
                        for text in sample_texts:
                            st.text_area(f"Text {sample_texts.index(text)+1}:", text, height=100, disabled=True)
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
        
        # Sentiment column selection
        sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
        if not sentiment_columns:
            sentiment_columns = ['sentiment']  # Default fallback
        
        selected_sentiment_col = st.selectbox(
            "Select Sentiment Column:",
            sentiment_columns,
            index=0,
            key=self.key_manager.get_key('analytics_dashboard', 'stats_sentiment_col') if self.key_manager else 'stats_sentiment_col'
        )
        
        # Group column selection (optional)
        group_columns = [col for col in df.columns if col != selected_sentiment_col]
        group_column = None
        if group_columns:
            group_option = st.selectbox(
                "Select Group Column (optional):",
                ['None'] + group_columns,
                index=0,
                key=self.key_manager.get_key('analytics_dashboard', 'stats_group_col') if self.key_manager else 'stats_group_col'
            )
            if group_option != 'None':
                group_column = group_option
        
        if st.button("Generate Statistical Analysis", type="primary"):
            with st.spinner("Performing statistical analysis..."):
                try:
                    # Perform statistical analysis using new comprehensive method
                    stats_results = self.analytics.statistical_analysis(
                        data=df,
                        sentiment_column=selected_sentiment_col,
                        group_column=group_column
                    )
                    
                    # Display results
                    self._display_statistical_results(stats_results)
                    
                except Exception as e:
                    st.error(f"Error performing statistical analysis: {str(e)}")
    
    def _display_statistical_results(self, results: Dict[str, Any]) -> None:
        """Display statistical analysis results."""
        st.subheader("Statistical Analysis Results")
        
        # Basic statistics
        if "basic_stats" in results:
            st.write("**Basic Statistics:**")
            basic_stats = results["basic_stats"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean", f"{basic_stats.get('mean', 0):.3f}")
                st.metric("Median", f"{basic_stats.get('median', 0):.3f}")
            
            with col2:
                st.metric("Standard Deviation", f"{basic_stats.get('std', 0):.3f}")
                st.metric("Count", basic_stats.get('count', 0))
            
            with col3:
                st.metric("Minimum", f"{basic_stats.get('min', 0):.3f}")
                st.metric("Maximum", f"{basic_stats.get('max', 0):.3f}")
        
        # Sentiment distribution
        if "sentiment_distribution" in results:
            st.subheader("Sentiment Distribution")
            
            sentiment_dist = results["sentiment_distribution"]
            sentiment_percentages = results.get("sentiment_percentages", {})
            
            # Create distribution chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sentiment_dist.keys()),
                    y=list(sentiment_dist.values()),
                    text=[f"{sentiment_percentages.get(k, 0):.1f}%" for k in sentiment_dist.keys()],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display percentages
            st.write("**Sentiment Percentages:**")
            for sentiment, percentage in sentiment_percentages.items():
                st.write(f"- {sentiment}: {percentage:.1f}%")
        
        # Group analysis
        if "group_analysis" in results:
            st.subheader("Group Analysis")
            
            group_stats = results["group_analysis"]
            
            # Create group comparison chart
            groups = list(group_stats.keys())
            means = [group_stats[g]['mean'] for g in groups]
            counts = [group_stats[g]['count'] for g in groups]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=groups,
                    y=means,
                    text=[f"n={counts[i]}" for i in range(len(groups))],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Average Sentiment by Group",
                xaxis_title="Group",
                yaxis_title="Average Sentiment"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display group statistics
            st.write("**Group Statistics:**")
            group_data = []
            for group, stats in group_stats.items():
                group_data.append({
                    "Group": group,
                    "Mean": f"{stats['mean']:.3f}",
                    "Std": f"{stats['std']:.3f}",
                    "Count": stats['count']
                })
            
            if group_data:
                group_df = pd.DataFrame(group_data)
                st.dataframe(group_df, use_container_width=True)
        
        # Statistical tests
        if "t_test" in results:
            st.subheader("Statistical Tests")
            
            t_test = results["t_test"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("T-statistic", f"{t_test['t_statistic']:.3f}")
            
            with col2:
                st.metric("P-value", f"{t_test['p_value']:.4f}")
            
            with col3:
                significance = "Significant" if t_test['significant'] else "Not Significant"
                st.metric("Significance", significance)
            
            # Interpretation
            if t_test['significant']:
                st.success("The difference between groups is statistically significant (p < 0.05)")
            else:
                st.info("No statistically significant difference between groups (p ≥ 0.05)")
        
        # Generate insights
        if st.button("Generate Insights"):
            with st.spinner("Generating insights..."):
                try:
                    insights = self.analytics.generate_insights(
                        data=df,
                        text_column='text',
                        sentiment_column='sentiment'
                    )
                    
                    st.subheader("Generated Insights")
                    for i, insight in enumerate(insights, 1):
                        st.write(f"{i}. {insight}")
                        
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
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
            ],
            key=self.key_manager.get_key('analytics_dashboard', 'custom_analysis_type') if self.key_manager else 'custom_analysis_type'
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
            index=1,
            key=self.key_manager.get_key('analytics_dashboard', 'time_period') if self.key_manager else 'time_period'
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
            ["Emotion Distribution", "Emotion Correlations", "Emotion Trends"],
            key=self.key_manager.get_key('analytics_dashboard', 'emotion_analysis_type') if self.key_manager else 'emotion_analysis_type'
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
            filter_column = st.selectbox("Filter by column:", selected_columns,
                key=self.key_manager.get_key('analytics_dashboard', 'custom_filter_col') if self.key_manager else 'custom_filter_col'
            )
            
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
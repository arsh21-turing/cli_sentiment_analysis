"""
Display utilities for the Streamlit application.
Provides formatting and styling functions for analysis results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ResultsFormatter:
    """
    Formats analysis results for display in the Streamlit UI.
    """
    
    def format_sentiment_result(self, sentiment_data):
        """
        Format sentiment analysis results for display.
        
        Args:
            sentiment_data (dict): Raw sentiment analysis data
            
        Returns:
            dict: Formatted display data
        """
        # Handle None or empty data
        if not sentiment_data:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "scores": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
            }
        
        # Extract the prediction and confidence
        prediction = sentiment_data.get("label", "Unknown")
        confidence = sentiment_data.get("score", 0.0)
        
        # Format scores for visualization
        scores = {
            "Positive": sentiment_data.get("positive", 0.0),
            "Negative": sentiment_data.get("negative", 0.0),
            "Neutral": sentiment_data.get("neutral", 0.0)
        }
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "scores": scores
        }
    
    def format_emotion_result(self, emotion_data):
        """
        Format emotion analysis results for display.
        
        Args:
            emotion_data (dict): Raw emotion analysis data
            
        Returns:
            dict: Formatted display data
        """
        # Handle None or empty data
        if not emotion_data:
            return {
                "scores": {},
                "top_emotions": []
            }
        
        # Get emotion scores
        scores = emotion_data.get("scores", {})
        
        # Sort emotions by score (descending)
        sorted_emotions = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        # Get top emotions (up to 5)
        top_emotions = sorted_emotions[:5]
        
        return {
            "scores": scores,
            "top_emotions": top_emotions
        }
    
    def create_bar_chart(self, data, title):
        """
        Create and display a bar chart for results.
        
        Args:
            data (dict): Data to display in chart
            title (str): Title for the chart
        """
        # Handle empty data
        if not data or len(data) == 0:
            st.warning(f"No data available for {title}")
            return
            
        try:
            # Convert dictionary to DataFrame for plotting
            df = pd.DataFrame({
                "Category": list(data.keys()),
                "Score": list(data.values())
            })
            
            # Color scheme based on chart type
            color_discrete_map = None
            if "Positive" in data:
                # For sentiment charts
                color_discrete_map = {
                    "Positive": "#4CAF50",  # Green
                    "Negative": "#F44336",  # Red
                    "Neutral": "#9E9E9E"    # Gray
                }
            
            # Create bar chart with Plotly
            fig = px.bar(
                df,
                x="Category",
                y="Score",
                title=title,
                color="Category",
                color_discrete_map=color_discrete_map,
                text_auto=".2%",
                height=300
            )
            
            # Format y-axis as percentage
            fig.update_yaxes(tickformat=".0%")
            
            # Improve appearance
            fig.update_layout(
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=12),
                xaxis_title=None,
                yaxis_title="Score",
                legend_title_text=None
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.write("Data:", data)
    
    def create_results_table(self, data):
        """
        Create and display a table of results.
        
        Args:
            data (list): List of dictionaries containing results data
        """
        # Handle empty data
        if not data or len(data) == 0:
            st.warning("No results data available to display.")
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Format confidence column if it exists
            if 'sentiment_confidence' in df.columns:
                df['sentiment_confidence'] = df['sentiment_confidence'].apply(lambda x: f"{x:.2%}")
            
            # Apply styling
            def style_sentiment(val):
                if val == 'positive':
                    return 'background-color: rgba(76, 175, 80, 0.2)'
                elif val == 'negative':
                    return 'background-color: rgba(244, 67, 54, 0.2)'
                else:
                    return ''
            
            # Apply styling if sentiment column exists
            if 'sentiment' in df.columns:
                styled_df = df.style.applymap(style_sentiment, subset=['sentiment'])
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                # Display without styling
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
        
        except Exception as e:
            st.error(f"Error displaying results table: {str(e)}")
            # Display raw data as fallback
            st.write("Raw data:", data) 
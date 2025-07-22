"""
Groq API-specific analysis component for the Streamlit application.
"""

import streamlit as st
import time
import pandas as pd
from src.ui.utils.groq_client import GroqClient
from src.ui.utils.display import ResultsFormatter


class GroqAnalysisComponent:
    """
    Creates and manages Groq analysis interface.
    """
    
    def __init__(self, key_manager=None):
        """
        Initialize the Groq analysis component.
        
        Args:
            key_manager: Widget key manager for unique keys
        """
        self.formatter = ResultsFormatter()
        self.key_manager = key_manager
        if self.key_manager:
            self.key_manager.register_component('groq_analysis', 'ga')
    
    def render(self):
        """
        Render the Groq analysis UI components.
        """
        st.header("Groq API Text Analysis")
        
        # Check if Groq API is enabled and properly configured
        if not st.session_state.get("use_groq", False):
            st.warning("⚠️ Groq API is not enabled. Enable it in the sidebar under 'Groq API Configuration'.")
            
            st.markdown("""
            ### Why use Groq API?
            
            Groq offers state-of-the-art LLMs with extremely fast inference speeds. Using Groq API provides:
            
            - More nuanced sentiment analysis
            - Richer emotional insights
            - Faster processing times
            - Support for multiple models
            
            Enable Groq API in the sidebar settings to access these features.
            """)
            return
        
        # Check if API key is set
        if not st.session_state.get("groq_api_key", ""):
            st.error("❌ Groq API key is not set. Please enter your API key in the sidebar.")
            
            st.markdown("""
            ### Getting a Groq API Key
            
            1. Sign up at [console.groq.com](https://console.groq.com)
            2. Navigate to the API Keys section
            3. Create a new API key
            4. Copy the key and paste it in the sidebar
            """)
            return
        
        # Display current model
        st.info(f"🤖 Using Groq model: **{st.session_state.get('groq_model', 'llama3-70b-8192')}**")
        
        # Text input area
        text = st.text_area(
            "Enter text to analyze with Groq:",
            height=200,
            placeholder="Type or paste your text here for Groq-powered analysis...",
            key=self.key_manager.get_key('groq_analysis', 'text_input') if self.key_manager else None
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_sentiment = st.checkbox(
                "Analyze Sentiment", 
                value=True,
                key=self.key_manager.get_key('groq_analysis', 'analyze_sentiment') if self.key_manager else None
            )
        
        with col2:
            analyze_emotion = st.checkbox(
                "Analyze Emotions", 
                value=True,
                key=self.key_manager.get_key('groq_analysis', 'analyze_emotion') if self.key_manager else None
            )
        
        if not analyze_sentiment and not analyze_emotion:
            st.warning("Please select at least one analysis type.")
        
        # Analysis button
        if st.button(
            "Analyze with Groq", 
            use_container_width=False, 
            disabled=not (analyze_sentiment or analyze_emotion),
            key=self.key_manager.get_key('groq_analysis', 'analyze_button') if self.key_manager else None
        ):
            if not text or text.isspace():
                st.error("❌ Error: Please enter text to analyze. The text field cannot be empty.")
            else:
                # Create an analysis status area
                status_area = st.empty()
                status_area.info("⏳ Initializing Groq analysis...")
                
                try:
                    # Get current parameters from session state
                    params = {
                        "groq_api_key": st.session_state.groq_api_key,
                        "groq_model": st.session_state.groq_model,
                        "analyze_sentiment": analyze_sentiment,
                        "analyze_emotion": analyze_emotion
                    }
                    
                    # Process text with Groq API
                    with st.spinner("🔍 Analyzing text with Groq API..."):
                        results = self.process_text(text, params, status_area)
                    
                    if results:
                        # Update status
                        status_area.success("✅ Groq analysis completed successfully!")
                        
                        # Store results in session state
                        st.session_state.groq_results = results
                        st.session_state.groq_processing_complete = True
                    else:
                        status_area.error("❌ Groq analysis failed. Please see error details above.")
                except Exception as e:
                    status_area.error(f"❌ Unexpected error: {str(e)}")
                    st.exception(e)
        
        # Display results if processing is complete
        if hasattr(st.session_state, "groq_processing_complete") and st.session_state.groq_processing_complete:
            if hasattr(st.session_state, "groq_results") and st.session_state.groq_results:
                self.display_results(st.session_state.groq_results)
    
    def process_text(self, text, params, status_area=None):
        """
        Process text using Groq API.
        
        Args:
            text (str): Text to analyze
            params (dict): Parameters including API key and model
            status_area (st.empty, optional): Empty placeholder for status updates
            
        Returns:
            dict: Analysis results
        """
        if not text or text.isspace():
            st.error("❌ Error: Empty text provided. Please enter valid text to analyze.")
            return None
            
        try:
            # Initialize Groq client
            client = GroqClient(
                api_key=params["groq_api_key"],
                model=params["groq_model"]
            )
            
            # Determine analysis type
            analysis_type = None
            if params["analyze_sentiment"] and params["analyze_emotion"]:
                analysis_type = "both"
            elif params["analyze_sentiment"]:
                analysis_type = "sentiment"
            elif params["analyze_emotion"]:
                analysis_type = "emotion"
            else:
                st.error("❌ No analysis type selected.")
                return None
            
            # Update status if status_area exists
            if status_area:
                status_area.info(f"⏳ Requesting analysis from Groq API ({params['groq_model']})...")
            
            # Analyze text using Groq API
            start_time = time.time()
            results = client.analyze_text(text, analysis_type)
            end_time = time.time()
            
            # Add processing time
            results["processing_time"] = end_time - start_time
            
            # Add parameters
            results["parameters"] = params
            
            # Update session statistics
            from src.ui.utils.session import SessionManager
            session_manager = SessionManager()
            session_manager.increment_texts_analyzed(1, is_groq=True)
            
            return results
        
        except Exception as e:
            st.error(f"❌ Error analyzing text with Groq API: {str(e)}")
            st.info("💡 Tip: Check your API key and ensure it's valid for the selected model.")
            return None
    
    def display_results(self, results):
        """
        Displays Groq analysis results.
        
        Args:
            results (dict): Analysis results to display
        """
        st.subheader("Groq API Analysis Results")
        
        # Display model info and processing time
        model_col, time_col = st.columns(2)
        
        with model_col:
            st.info(f"🤖 Model: **{results['model']}**")
        
        with time_col:
            st.info(f"⏱️ Processing Time: **{results['processing_time']:.3f}** seconds")
        
        # Create columns for results
        if "sentiment" in results and "emotion" in results:
            col1, col2 = st.columns(2)
        else:
            col1 = st.container()
        
        # Display sentiment results
        if "sentiment" in results:
            with col1:
                st.markdown("### Sentiment Analysis")
                formatted_sentiment = self.formatter.format_sentiment_result(results["sentiment"])
                self.formatter.create_bar_chart(
                    data=formatted_sentiment["scores"], 
                    title="Sentiment Scores"
                )
                
                # Colorful display of sentiment
                sentiment = formatted_sentiment['prediction']
                conf = formatted_sentiment['confidence']
                
                if sentiment == "positive":
                    st.markdown(f"**Predicted Sentiment:** 🟢 **{sentiment.title()}**")
                elif sentiment == "negative":
                    st.markdown(f"**Predicted Sentiment:** 🔴 **{sentiment.title()}**")
                else:
                    st.markdown(f"**Predicted Sentiment:** ⚪ **{sentiment.title()}**")
                
                # Confidence indicator
                st.markdown(f"**Confidence:** {conf:.2%}")
                
                # Confidence gauge
                confidence_color = "green" if conf > 0.8 else "orange" if conf > 0.6 else "red"
                st.markdown(f"""
<div>
<div style='width:{conf*100}%; background-color:{confidence_color}; height:10px; border-radius:5px'></div>
</div>""", unsafe_allow_html=True)
        
        # Display emotion results
        if "emotion" in results:
            with col2 if "sentiment" in results else col1:
                st.markdown("### Emotion Analysis")
                formatted_emotion = self.formatter.format_emotion_result(results["emotion"])
                self.formatter.create_bar_chart(
                    data=formatted_emotion["scores"], 
                    title="Emotion Scores"
                )
                
                # Display primary emotion
                primary_emotion = results["emotion"].get("primary_emotion", "unknown")
                st.markdown(f"**Primary Emotion:** **{primary_emotion.title()}**")
                
                # Enhanced display for top emotions
                st.markdown("**Top Emotions:**")
                emotion_emojis = {
                    "joy": "😊", "sadness": "😢", "anger": "😠", 
                    "fear": "😨", "surprise": "😲", "disgust": "🤢",
                    "trust": "🤝", "anticipation": "🤔"
                }
                
                for emotion, score in formatted_emotion["top_emotions"]:
                    emoji = emotion_emojis.get(emotion.lower(), "•")
                    st.markdown(f"- {emoji} **{emotion}**: {score:.2%}")
        
        # Display the analyzed text
        st.markdown("### Analyzed Text")
        st.text_area("Text", value=results["text"], height=100, disabled=True)
        
        # Provide option to compare with regular models
        st.markdown("### Compare with Other Models")
        if st.button("Run Standard Analysis on Same Text"):
            # Store text in session state for regular analysis
            st.session_state.compare_text = results["text"]
            st.session_state.active_tab = "Single Text Analysis"
            st.rerun()
        
        # Export functionality
        st.markdown("---")
        if st.button("📥 Export Results", key="export_groq_results"):
            # Use the ExportManager
            from src.ui.components.export_manager import ExportManager
            export_manager = ExportManager()
            export_manager.render_export_panel("groq", results)
        
        # Advanced: Raw API response
        with st.expander("Advanced: Raw API Response", expanded=False):
            st.json(results)
    
    def display_batch_results(self, results):
        """
        Displays Groq batch processing results.
        
        Args:
            results (dict): Batch processing results to display
        """
        # Collapsible results section
        with st.expander("📊 Groq Batch Processing Results", expanded=True):
            st.subheader("Summary")
            
            # Display summary
            st.markdown(f"**File:** `{results['file_name']}`")
            st.markdown(f"**Total Texts Processed:** {results['total_texts']}")
            st.markdown(f"**Model Used:** `{results['batch_results']['model']}`")
            st.markdown(f"**Processing Time:** {results['batch_results']['processing_time']:.2f} seconds")
            
            # Create metrics for overall results
            batch_results = results["batch_results"]
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            # Sentiment metrics with emojis
            with col1:
                st.metric("🟢 Positive Texts", batch_results["sentiment_counts"].get("positive", 0))
            
            with col2:
                st.metric("🔴 Negative Texts", batch_results["sentiment_counts"].get("negative", 0))
            
            with col3:
                st.metric("⚪ Neutral Texts", batch_results["sentiment_counts"].get("neutral", 0))
            
            with col4:
                st.metric("🎯 Avg. Confidence", f"{batch_results['average_confidence']:.2%}")
            
            # Display emotion distribution
            st.markdown("### Emotion Distribution")
            self.formatter.create_bar_chart(
                data=batch_results["emotion_distribution"], 
                title="Emotion Distribution Across All Texts"
            )
            
            # Display full results table with filtering
            st.markdown("### Detailed Results")
            
            # Convert to DataFrame for filtering
            df_results = pd.DataFrame(batch_results["detailed_results"])
            
            # Add filter options
            filter_container = st.container()
            with filter_container:
                filter_col1, filter_col2 = st.columns(2)
                
                # Filter by sentiment if sentiment column exists
                if "sentiment" in df_results.columns:
                    with filter_col1:
                        sentiment_filter = st.multiselect(
                            "Filter by Sentiment",
                            options=sorted(df_results["sentiment"].unique()),
                            default=sorted(df_results["sentiment"].unique())
                        )
                
                # Filter by confidence if confidence column exists
                if "sentiment_confidence" in df_results.columns:
                    with filter_col2:
                        min_confidence = st.slider(
                            "Minimum Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.05
                        )
            
            # Apply filters
            filtered_df = df_results
            
            if "sentiment" in df_results.columns and sentiment_filter:
                filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]
            
            if "sentiment_confidence" in df_results.columns:
                filtered_df = filtered_df[filtered_df["sentiment_confidence"] >= min_confidence]
            
            # Display filtered results
            st.write(f"Showing {len(filtered_df)} of {len(df_results)} results")
            self.formatter.create_results_table(filtered_df.to_dict("records"))
            
            # Export functionality
            st.markdown("### Export Results")
            
            # Enhanced export with ExportManager
            if st.button("📥 Export Results", key="export_groq_batch_results"):
                # Use the ExportManager
                from src.ui.components.export_manager import ExportManager
                export_manager = ExportManager()
                export_manager.render_export_panel("groq_batch", results)
            
            # Quick download buttons for immediate access
            download_col1, download_col2 = st.columns(2)
            
            # Download all results
            with download_col1:
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "Quick Download (CSV)",
                    csv,
                    file_name=f"groq_batch_analysis_results_{results['file_name'].split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Download filtered results
            with download_col2:
                filtered_csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Quick Download Filtered (CSV)",
                    filtered_csv,
                    file_name=f"groq_filtered_results_{results['file_name'].split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Display the parameters used with better formatting
            st.markdown("### Parameters Used")
            
            # Check if parameters exist and handle missing keys
            if results and 'parameters' in results and results['parameters']:
                params = results['parameters']
                params_col1, params_col2, params_col3 = st.columns(3)
                
                with params_col1:
                    groq_model = params.get('groq_model', 'llama3-70b-8192')
                    st.metric("Model", groq_model)
                
                with params_col2:
                    batch_size = params.get('batch_size', 5)
                    st.metric("Batch Size", batch_size)
                
                with params_col3:
                    st.metric("Processing Type", "Groq API")
            else:
                st.info("Parameters information not available")
            
            # Add report generation section
            from src.ui.components.report_generator import ReportGenerator
            
            st.markdown("### Generate Report")
            report_generator = ReportGenerator()
            report_generator.render_options("groq", results) 
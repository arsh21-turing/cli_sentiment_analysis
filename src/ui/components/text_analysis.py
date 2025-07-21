"""
Text analysis component for the Streamlit application.
Provides UI for analyzing individual text inputs.
"""

import streamlit as st
import time
from src.models.transformer import SentimentEmotionTransformer
from src.ui.utils.display import ResultsFormatter


class TextAnalysisComponent:
    """
    Creates and manages single text analysis interface.
    """
    
    def __init__(self):
        """
        Initialize the text analysis component.
        """
        self.formatter = ResultsFormatter()
        self.model = None
    
    def get_model(self):
        """
        Get or create the transformer model instance.
        
        Returns:
            SentimentEmotionTransformer: Model instance
        """
        if self.model is None:
            self.model = SentimentEmotionTransformer()
        return self.model
    
    def render(self):
        """
        Render the text analysis UI components.
        """
        st.header("Analyze Single Text")
        
        # Check if we have prefilled text from comparison
        prefill_text = ""
        if hasattr(st.session_state, "prefill_text") and st.session_state.prefill_text:
            prefill_text = st.session_state.prefill_text
            # Clear it after reading to avoid persistence
            st.session_state.prefill_text = ""
        
        # Text input area with better guidance
        text = st.text_area(
            "Enter text to analyze:",
            value=prefill_text,
            height=200,
            placeholder="Type or paste your text here... (minimum 10 characters recommended for accurate analysis)"
        )
        
        # Analysis button with progress handling
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("Analyze Text", use_container_width=True)
        
        with col2:
            # Information about expected time
            st.caption("Analysis typically takes a few seconds.")
        
        # Text length validation and feedback
        if text and len(text) < 10:
            st.warning("⚠️ Text is very short. Analysis may be less accurate with limited content.")
        
        # Process text when button is clicked
        if analyze_button:
            if not text or text.isspace():
                st.error("❌ Error: Please enter text to analyze. The text field cannot be empty.")
            else:
                # Create an analysis status area
                status_area = st.empty()
                status_area.info("⏳ Initializing analysis...")
                
                try:
                    # Get current parameters from session state
                    params = {
                        "sentiment_threshold": st.session_state.sentiment_threshold,
                        "emotion_threshold": st.session_state.emotion_threshold,
                        "use_api_fallback": st.session_state.use_api_fallback
                    }
                    
                    # Process text with current parameters
                    with st.spinner("🔍 Analyzing text..."):
                        results = self.process_text(text, params, status_area)
                    
                    if results:
                        # Update status
                        status_area.success("✅ Analysis completed successfully!")
                        
                        # Store results in session state
                        st.session_state.analysis_results = results
                        st.session_state.processing_complete = True
                    else:
                        status_area.error("❌ Analysis failed. Please see error details above.")
                except Exception as e:
                    status_area.error(f"❌ Unexpected error: {str(e)}")
                    st.exception(e)
        
        # Display results if processing is complete
        if hasattr(st.session_state, "processing_complete") and st.session_state.processing_complete:
            if st.session_state.analysis_results:
                self.display_results(st.session_state.analysis_results)
    
    def process_text(self, text, params, status_area=None):
        """
        Process text with current parameters.
        
        Args:
            text (str): Text to analyze
            params (dict): Parameters like thresholds and API fallback setting
            status_area (st.empty, optional): Empty placeholder for status updates
            
        Returns:
            dict: Analysis results
        """
        if not text or text.isspace():
            st.error("❌ Error: Empty text provided. Please enter valid text to analyze.")
            return None
            
        try:
            # Update status if status_area exists
            if status_area:
                status_area.info("⏳ Analyzing sentiment...")
            
            # Get model and set thresholds
            model = self.get_model()
            model.set_thresholds(
                sentiment_threshold=params["sentiment_threshold"],
                emotion_threshold=params["emotion_threshold"]
            )
            
            # Analyze text
            analysis_result = model.analyze(text, use_fallback=params["use_api_fallback"])
            
            # Update status if status_area exists
            if status_area:
                status_area.info("⏳ Analyzing emotions...")
            
            # Extract sentiment and emotion results
            sentiment_results = analysis_result.get("sentiment", {})
            emotion_results = analysis_result.get("emotion", {})
            
            # Combine results
            results = {
                "text": text,
                "sentiment": sentiment_results,
                "emotion": emotion_results,
                "parameters": params
            }
            
            # Update session statistics
            from src.ui.utils.session import SessionManager
            session_manager = SessionManager()
            session_manager.increment_texts_analyzed(1, is_groq=False)
            
            return results
        
        except Exception as e:
            st.error(f"❌ Error analyzing text: {str(e)}")
            st.info("💡 Tip: Check if your text contains valid content or try adjusting the threshold values.")
            if params["use_api_fallback"] == False:
                st.info("💡 Tip: Try enabling API fallback in the sidebar for alternative processing.")
            return None
    
    def display_results(self, results):
        """
        Format and display analysis results.
        
        Args:
            results (dict): Analysis results to display
        """
        st.subheader("Analysis Results")
        
        # Create two columns for the results
        col1, col2 = st.columns(2)
        
        # Display sentiment results in first column
        with col1:
            st.markdown("### Sentiment Analysis")
            formatted_sentiment = self.formatter.format_sentiment_result(results["sentiment"])
            self.formatter.create_bar_chart(
                data=formatted_sentiment["scores"], 
                title="Sentiment Scores"
            )
            
            # Enhanced display with formatting based on sentiment
            sentiment = formatted_sentiment['prediction']
            conf = formatted_sentiment['confidence']
            
            # Colorful display of sentiment
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
        
        # Display emotion results in second column
        with col2:
            st.markdown("### Emotion Analysis")
            formatted_emotion = self.formatter.format_emotion_result(results["emotion"])
            self.formatter.create_bar_chart(
                data=formatted_emotion["scores"], 
                title="Emotion Scores"
            )
            
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
        
        # Display the parameters used with better formatting
        st.markdown("### Parameters Used")
        params_col1, params_col2, params_col3 = st.columns(3)
        with params_col1:
            st.metric("Sentiment Threshold", f"{results['parameters']['sentiment_threshold']:.2f}")
        with params_col2:
            st.metric("Emotion Threshold", f"{results['parameters']['emotion_threshold']:.2f}")
        with params_col3:
            st.metric("API Fallback", "Enabled" if results['parameters']['use_api_fallback'] else "Disabled")
        
        # Export functionality
        st.markdown("---")
        if st.button("📥 Export Results", key="export_text_results"):
            # Use the ExportManager
            from src.ui.components.export_manager import ExportManager
            export_manager = ExportManager()
            export_manager.render_export_panel("text", results) 
"""
Sidebar component for the Streamlit application.
Provides parameter controls for the text analysis functionality.
"""

import streamlit as st
import re
from src.ui.utils.session import SessionManager


class SidebarComponent:
    """
    Creates and manages sidebar controls for parameter configuration.
    """
    
    def __init__(self, title="Parameters"):
        """
        Initialize the sidebar component.
        
        Args:
            title (str): Title displayed at top of sidebar
        """
        self.title = title
        self.groq_models = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        self.session_manager = SessionManager()
    
    def create_sidebar(self):
        """
        Create all sidebar elements and return current values.
        
        Returns:
            dict: Dictionary containing all configured parameters
        """
        st.sidebar.title(self.title)
        
        # Create usage metrics at the top
        self.create_usage_metrics()
        
        # Create collapsible sections
        with st.sidebar.expander("Threshold Settings", expanded=True):
            # Create threshold sliders
            sentiment_threshold, emotion_threshold = self.create_threshold_sliders()
        
        with st.sidebar.expander("API Settings", expanded=True):
            # Create API fallback toggle
            use_api_fallback = self.create_api_toggle()
        
        with st.sidebar.expander("Groq API Configuration", expanded=True):
            # Create Groq API configuration
            groq_api_key, groq_model, use_groq = self.create_groq_section()
        
        # Return all parameters as a dictionary
        return {
            "sentiment_threshold": sentiment_threshold,
            "emotion_threshold": emotion_threshold,
            "use_api_fallback": use_api_fallback,
            "groq_api_key": groq_api_key,
            "groq_model": groq_model,
            "use_groq": use_groq
        }
    
    def create_threshold_sliders(self):
        """
        Create sliders for sentiment and emotion thresholds.
        
        Returns:
            tuple: (sentiment_threshold, emotion_threshold)
        """
        sentiment_threshold = st.sidebar.slider(
            "Sentiment Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Confidence threshold for sentiment classification"
        )
        
        emotion_threshold = st.sidebar.slider(
            "Emotion Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Confidence threshold for emotion detection"
        )
        
        return sentiment_threshold, emotion_threshold
    
    def create_api_toggle(self):
        """
        Create toggle for API fallback option.
        
        Returns:
            bool: True if API fallback is enabled, False otherwise
        """
        use_api_fallback = st.sidebar.toggle(
            "Use API Fallback",
            value=False,
            help="Fall back to external API if local model fails"
        )
        
        return use_api_fallback
    
    def create_groq_section(self):
        """
        Creates Groq API configuration section.
        
        Returns:
            tuple: (groq_api_key, groq_model, use_groq)
        """
        # Get current values from session state
        current_groq_api_key = st.session_state.get("groq_api_key", "")
        current_groq_model = st.session_state.get("groq_model", "llama3-70b-8192")
        current_use_groq = st.session_state.get("use_groq", False)
        
        # Create checkbox to enable/disable Groq
        use_groq = st.sidebar.checkbox(
            "Use Groq API",
            value=current_use_groq,
            help="Enable Groq API for analysis"
        )
        
        # API key input
        groq_api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            value=current_groq_api_key,
            disabled=not use_groq,
            help="Enter your Groq API key (stored securely)",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
        )
        
        # Model selection
        groq_model = st.sidebar.selectbox(
            "Groq Model",
            options=self.groq_models,
            index=self.groq_models.index(current_groq_model) if current_groq_model in self.groq_models else 0,
            disabled=not use_groq,
            help="Select the Groq model to use for analysis"
        )
        
        # Display API key status
        if use_groq:
            if self._validate_groq_api_key(groq_api_key):
                st.sidebar.success("API key format is valid.")
                
                # Test connection button
                if st.sidebar.button("Test Connection", disabled=not groq_api_key):
                    self._test_groq_connection(groq_api_key, groq_model)
            else:
                st.sidebar.warning("API key format is invalid or empty.")
                st.sidebar.markdown("""
                ℹ️ Groq API keys typically start with `gsk_` followed by a string of characters.
                """)
        
        # Add a link to get Groq API key
        st.sidebar.markdown("""
        [Get a Groq API key](https://console.groq.com/keys) if you don't have one.
        """)
        
        return groq_api_key, groq_model, use_groq
    
    def _validate_groq_api_key(self, api_key):
        """
        Validates Groq API key format.
        
        Args:
            api_key (str): API key to validate
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        if not api_key:
            return False
        
        # Basic format check for Groq API keys
        # Groq API keys typically start with "gsk_" followed by a string of characters
        return bool(re.match(r'^gsk_[a-zA-Z0-9_]+$', api_key))
    
    def _test_groq_connection(self, api_key, model):
        """
        Tests connection to Groq API.
        
        Args:
            api_key (str): Groq API key
            model (str): Groq model name
        """
        # Import here to avoid circular imports
        from src.ui.utils.groq_client import GroqClient
        
        with st.sidebar.spinner("Testing Groq API connection..."):
            try:
                client = GroqClient(api_key=api_key, model=model)
                if client.test_connection():
                    st.sidebar.success(f"✅ Successfully connected to Groq API with model '{model}'")
                else:
                    st.sidebar.error("❌ Failed to connect to Groq API. Check your API key and try again.")
            except Exception as e:
                st.sidebar.error(f"❌ Error connecting to Groq API: {str(e)}")
    
    def create_usage_metrics(self):
        """
        Create usage metrics display.
        """
        stats = self.session_manager.get_session_stats()
        
        # Create a container with background styling
        with st.sidebar.container():
            st.markdown(
                """
<div>
<h3>
Session Statistics

</h3>
</div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display metrics in columns
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric(
                label="Texts Analyzed",
                value=stats["texts_analyzed"],
                delta=None
            )
        
        with col2:
            requests = stats["standard_requests"] + stats["groq_requests"]
            st.metric(
                label="Total Requests",
                value=requests,
                delta=None
            )
        
        # If duration is available, show it
        if stats["session_duration"]:
            duration_mins = stats["session_duration"].total_seconds() / 60
            if duration_mins < 1:
                duration_str = f"{int(duration_mins * 60)}s"
            else:
                duration_str = f"{int(duration_mins)}m"
            
            st.sidebar.caption(f"Session time: {duration_str}") 
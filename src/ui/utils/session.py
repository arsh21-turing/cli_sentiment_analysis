"""
Session state management for the Streamlit application.
Provides persistence of parameters between interactions.
"""

import streamlit as st
import re


class SessionManager:
    """
    Manages Streamlit session state for parameter persistence.
    """
    
    def init_session(self):
        """
        Initialize session state with default values if not already present.
        """
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            
            # Analysis parameters
            st.session_state.sentiment_threshold = 0.5
            st.session_state.emotion_threshold = 0.3
            st.session_state.use_api_fallback = False
            
            # Results storage
            st.session_state.analysis_results = None
            st.session_state.batch_results = None
            st.session_state.processing_complete = False
            st.session_state.groq_results = None
            st.session_state.groq_processing_complete = False
            
            # Groq API settings
            st.session_state.groq_api_key = ""
            st.session_state.groq_model = "llama3-70b-8192"
            st.session_state.use_groq = False
            
            # UI state
            st.session_state.active_tab = "Single Text Analysis"
            st.session_state.compare_text = ""
            
            # Usage statistics
            st.session_state.texts_analyzed = 0
            st.session_state.standard_requests = 0
            st.session_state.groq_requests = 0
            st.session_state.session_start_time = None
    
    def get_params(self):
        """
        Get current parameter values from session state.
        
        Returns:
            dict: Dictionary containing all parameters
        """
        return {
            # Analysis parameters
            "sentiment_threshold": st.session_state.sentiment_threshold,
            "emotion_threshold": st.session_state.emotion_threshold,
            "use_api_fallback": st.session_state.use_api_fallback,
            
            # Groq API settings
            "groq_api_key": st.session_state.groq_api_key,
            "groq_model": st.session_state.groq_model,
            "use_groq": st.session_state.use_groq
        }
    
    def update_params(self, params):
        """
        Update session state with new parameter values.
        
        Args:
            params (dict): New parameter values to store
        """
        for key, value in params.items():
            if key in st.session_state:
                st.session_state[key] = value
    
    def reset_session(self, keep_api_keys=True):
        """
        Reset session state to default values.
        
        Args:
            keep_api_keys (bool): Whether to preserve API keys
        """
        # Store API key if needed
        api_key = st.session_state.groq_api_key if keep_api_keys else ""
        
        # Reset analysis parameters
        st.session_state.sentiment_threshold = 0.5
        st.session_state.emotion_threshold = 0.3
        st.session_state.use_api_fallback = False
        
        # Reset results
        st.session_state.analysis_results = None
        st.session_state.batch_results = None
        st.session_state.processing_complete = False
        st.session_state.groq_results = None
        st.session_state.groq_processing_complete = False
        
        # Reset Groq settings (except API key if keeping)
        st.session_state.groq_api_key = api_key
        st.session_state.groq_model = "llama3-70b-8192"
        st.session_state.use_groq = False if not api_key else st.session_state.use_groq
        
        # Reset UI state
        st.session_state.active_tab = "Single Text Analysis"
        st.session_state.compare_text = ""
        
        # Preserve session statistics
        texts_analyzed = st.session_state.texts_analyzed
        standard_requests = st.session_state.standard_requests
        groq_requests = st.session_state.groq_requests
        session_start_time = st.session_state.session_start_time
        
        # Restore session statistics
        st.session_state.texts_analyzed = texts_analyzed
        st.session_state.standard_requests = standard_requests
        st.session_state.groq_requests = groq_requests
        st.session_state.session_start_time = session_start_time
    
    def validate_groq_api_key(self, api_key):
        """
        Validates Groq API key format.
        
        Args:
            api_key (str): API key to validate
            
        Returns:
            bool: True if key format is valid, False otherwise
        """
        if not api_key:
            return False
        
        # Basic format check for Groq API keys
        # Groq API keys typically start with "gsk_" followed by a string of characters
        return bool(re.match(r'^gsk_[a-zA-Z0-9_]+$', api_key))
    
    def clear_api_keys(self):
        """
        Securely clears stored API keys from session.
        """
        st.session_state.groq_api_key = ""
        st.warning("API keys have been cleared from the session.")
    
    def increment_texts_analyzed(self, count: int = 1, is_groq: bool = False) -> None:
        """
        Increment the count of texts analyzed.
        
        Args:
            count (int): Number of texts to add to the counter
            is_groq (bool): Whether the analysis was done with Groq API
        """
        # Initialize session start time if not set
        if st.session_state.session_start_time is None:
            import datetime
            st.session_state.session_start_time = datetime.datetime.now()
        
        # Update counters
        st.session_state.texts_analyzed += count
        
        # Update request counters
        if is_groq:
            st.session_state.groq_requests += 1
        else:
            st.session_state.standard_requests += 1
    
    def get_session_stats(self) -> dict:
        """
        Get usage statistics for the current session.
        
        Returns:
            dict: Dictionary containing session statistics
        """
        # Calculate session duration
        duration = None
        if st.session_state.session_start_time:
            import datetime
            duration = datetime.datetime.now() - st.session_state.session_start_time
        
        return {
            "texts_analyzed": st.session_state.texts_analyzed,
            "standard_requests": st.session_state.standard_requests,
            "groq_requests": st.session_state.groq_requests,
            "session_duration": duration
        } 
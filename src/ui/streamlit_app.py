"""
Streamlit application for text analysis and batch processing.
Provides a UI for analyzing both single texts and uploaded files.
"""

import streamlit as st
import sys
from pathlib import Path
import traceback

# Add project root to Python path to enable imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from src.ui.components.sidebar import SidebarComponent
from src.ui.components.text_analysis import TextAnalysisComponent
from src.ui.components.file_upload import FileUploadComponent
from src.ui.components.groq_analysis import GroqAnalysisComponent
from src.ui.components.comparison import ComparisonComponent
from src.ui.components.analytics_dashboard import AnalyticsDashboard
from src.ui.components.log_dashboard import LogDashboard
from src.ui.utils.session import SessionManager
from src.ui.utils.comparison_utils import ModelComparator
from src.utils.real_time import RealTimeDataConnector
from src.utils.logging_system import LoggingSystem, LogLevel, LogCategory
from src.utils.widget_key_manager import WidgetKeyManager
from datetime import datetime


def setup_page():
    """Configure page title, layout, and basic appearance."""
    st.set_page_config(
        page_title="Text Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #0D47A1;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
        border-bottom-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

    # App header
    st.title("📊 Text Analysis Tool")
    st.markdown("Analyze sentiment and emotions in text with configurable thresholds.")


def check_groq_availability():
    """
    Checks if Groq API is properly configured.
    
    Returns:
        bool: True if Groq API is available, False otherwise
    """
    if not st.session_state.get("use_groq", False):
        return False
        
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        return False
        
    # Validate API key format
    session_manager = SessionManager()
    if not session_manager.validate_groq_api_key(api_key):
        return False
        
    return True


def check_comparison_availability():
    """
    Check if comparison features are available.
    
    Returns:
        bool: True if at least two different models are available for comparison
    """
    comparator = ModelComparator()
    available_models = comparator.get_available_models()
    return len(available_models) >= 2


def initialize_data_connector():
    """
    Initializes the real-time data connector.
    """
    # Check if real-time connector already exists in session state
    if 'real_time_connector' not in st.session_state:
        # Create new connector
        connector = RealTimeDataConnector(max_queue_size=1000, ttl_seconds=3600)
        connector.attach_to_session(st.session_state)
        st.session_state.real_time_connector = connector


def process_with_real_time_updates(text: str, model=None) -> dict:
    """
    Process text with real-time updates to the data connector.
    
    Args:
        text: Text to analyze
        model: Model to use for analysis, defaults to transformer_model
        
    Returns:
        Analysis result
    """
    # Use specified model or default transformer
    if model is None:
        if 'transformer_model' in st.session_state:
            model = st.session_state.transformer_model
        else:
            # Create default transformer if one doesn't exist
            from src.models.transformer import SentimentEmotionTransformer
            model = SentimentEmotionTransformer()
            st.session_state.transformer_model = model
    
    # Process text
    result = model.analyze(text)
    
    # Add timestamp
    result['timestamp'] = datetime.now().isoformat()
    
    # Add parameters to the result
    if 'transformer_model' in st.session_state:
        # Get current parameters from session state
        result['parameters'] = {
            'sentiment_threshold': st.session_state.get('sentiment_threshold', 0.5),
            'emotion_threshold': st.session_state.get('emotion_threshold', 0.3),
            'use_api_fallback': st.session_state.get('use_api_fallback', False)
        }
    
    # Push to real-time connector
    if 'real_time_connector' in st.session_state:
        st.session_state.real_time_connector.push(result)
    
    # Store in session state analysis results
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.session_state.analysis_results = []
    st.session_state.analysis_results.append(result)
    
    return result


def init_logging_system():
    """Initializes the logging system."""
    # Create logs directory in the application directory
    import os
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    
    # Initialize LoggingSystem
    logger = LoggingSystem(
        log_dir=logs_dir,
        retention_days=30,
        log_level=LogLevel.INFO,
        enable_file=True,
        enable_console=True,
        enable_audit=True,
        enable_metrics=True
    )
    
    return logger


def main():
    """Main application entry point."""
    # Add error recovery mechanism
    if 'app_error' in st.session_state:
        st.error("Previous error detected. Attempting to recover...")
        del st.session_state.app_error
        st.rerun()
    
    try:
        # Setup page configuration
        setup_page()
        
        # Initialize key manager
        if 'key_manager' not in st.session_state:
            st.session_state.key_manager = WidgetKeyManager()
        
        # Initialize logging system
        logger = init_logging_system()
        
        # Log session start
        logger.log_system_event(
            "New user session started", 
            LogLevel.INFO, 
            LogCategory.SYSTEM
        )
        
        # Initialize session state with proper error handling
        try:
            session_manager = SessionManager()
            session_manager.init_session()
            # Clean up any stale session state
            session_manager.cleanup_stale_session_state()
        except Exception as e:
            st.error(f"Session initialization error: {str(e)}")
            logger.log_system_event(
                f"Session initialization failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
            return
        
        # Initialize export options state
        if 'show_export_options' not in st.session_state:
            st.session_state.show_export_options = False
        
        # Initialize save/load dialog states
        if 'show_save_dialog' not in st.session_state:
            st.session_state.show_save_dialog = False
        if 'show_load_dialog' not in st.session_state:
            st.session_state.show_load_dialog = False
        if 'show_manage_dialog' not in st.session_state:
            st.session_state.show_manage_dialog = False
        if 'confirm_delete' not in st.session_state:
            st.session_state.confirm_delete = None
        if 'confirm_delete_all' not in st.session_state:
            st.session_state.confirm_delete_all = False
        
        # Initialize real-time data connector
        try:
            initialize_data_connector()
        except Exception as e:
            st.error(f"Real-time connector initialization error: {str(e)}")
            logger.log_system_event(
                f"Real-time connector initialization failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
        
        # Ensure batch_results is initialized
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
        
        # Make real-time processing function available
        st.process_with_real_time_updates = process_with_real_time_updates
        
        # Reset key counters for main components
        st.session_state.key_manager.reset_component('main')
        
        # Create sidebar with configuration parameters
        try:
            sidebar = SidebarComponent(title="Analysis Parameters", key_manager=st.session_state.key_manager)
            params = sidebar.create_sidebar()
            session_manager.update_params(params)
        except Exception as e:
            st.error(f"Sidebar initialization error: {str(e)}")
            logger.log_system_event(
                f"Sidebar initialization failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
            return
        
        # Handle text from quick analysis or comparison
        compare_text = st.session_state.get("compare_text", None)
        
        # Create tabs with error handling
        try:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Single Text Analysis", 
                "File Upload", 
                "Groq Analysis", 
                "Model Comparison", 
                "Analytics Dashboard",
                "Real-Time Analytics",
                "Logs & Audit"
            ])
        except Exception as e:
            st.error(f"Tab creation error: {str(e)}")
            logger.log_system_event(
                f"Tab creation failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
            return
        
        # Handle Single Text Analysis tab
        try:
            with tab1:
                st.session_state.key_manager.reset_component('text_analysis')
                text_analysis = TextAnalysisComponent(key_manager=st.session_state.key_manager)
                
                # If we have text to compare or analyze, prefill it
                if compare_text:
                    st.session_state.prefill_text = compare_text
                    if st.session_state.get("compare_text"):
                        st.info("Text copied from Groq Analysis for comparison")
                    else:
                        st.info("Text from Quick Analysis loaded")
                
                text_analysis.render()
        except Exception as e:
            st.error(f"Text analysis tab error: {str(e)}")
            logger.log_system_event(
                f"Text analysis tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
            
        # Handle File Upload tab  
        try:
            with tab2:
                st.session_state.key_manager.reset_component('file_upload')
                file_upload = FileUploadComponent(key_manager=st.session_state.key_manager)
                file_upload.render()
        except Exception as e:
            st.error(f"File upload tab error: {str(e)}")
            logger.log_system_event(
                f"File upload tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
            
        # Handle Groq Analysis tab
        try:
            with tab3:
                st.session_state.key_manager.reset_component('groq_analysis')
                groq_analysis = GroqAnalysisComponent(key_manager=st.session_state.key_manager)
                groq_analysis.render()
        except Exception as e:
            st.error(f"Groq analysis tab error: {str(e)}")
            logger.log_system_event(
                f"Groq analysis tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
        
        # Handle Model Comparison tab
        try:
            with tab4:
                st.session_state.key_manager.reset_component('comparison')
                comparison = ComparisonComponent(key_manager=st.session_state.key_manager)
                comparison.render()
        except Exception as e:
            st.error(f"Model comparison tab error: {str(e)}")
            logger.log_system_event(
                f"Model comparison tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
        
        # Handle Analytics Dashboard tab
        try:
            with tab5:
                st.session_state.key_manager.reset_component('analytics_dashboard')
                analytics_dashboard = AnalyticsDashboard(
                    session_state=st.session_state,
                    key_manager=st.session_state.key_manager
                )
                analytics_dashboard.render()
        except Exception as e:
            st.error(f"Analytics dashboard tab error: {str(e)}")
            logger.log_system_event(
                f"Analytics dashboard tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
        
        # Handle Real-Time Analytics tab
        try:
            with tab6:
                st.session_state.key_manager.reset_component('real_time_analytics')
                analytics_dashboard = AnalyticsDashboard(
                    session_state=st.session_state,
                    key_manager=st.session_state.key_manager
                )
                analytics_dashboard.render_real_time_dashboard()
        except Exception as e:
            st.error(f"Real-time analytics tab error: {str(e)}")
            logger.log_system_event(
                f"Real-time analytics tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
        
        # Handle Logs & Audit tab
        try:
            with tab7:
                st.session_state.key_manager.reset_component('logs')
                logs_dashboard = LogDashboard(
                    session_state=st.session_state,
                    key_manager=st.session_state.key_manager
                )
                logs_dashboard.render()
        except Exception as e:
            st.error(f"Logs dashboard tab error: {str(e)}")
            logger.log_system_event(
                f"Logs dashboard tab failed: {str(e)}", 
                LogLevel.ERROR, 
                LogCategory.SYSTEM,
                exception=e
            )
        
        # Handle rerun request from sidebar actions
        if 'should_rerun' in st.session_state and st.session_state.should_rerun:
            del st.session_state.should_rerun
            st.rerun()
            
    except Exception as e:
        st.error(f"Application initialization error: {str(e)}")
        # Set error flag for recovery
        st.session_state.app_error = True
        # Log the error
        try:
            logger = LoggingSystem()
            logger.log_system_event(
                f"Application initialization failed: {str(e)}", 
                LogLevel.CRITICAL, 
                LogCategory.SYSTEM,
                exception=e
            )
        except:
            pass  # If logging fails, just show the error
        st.stop()


if __name__ == "__main__":
    main()

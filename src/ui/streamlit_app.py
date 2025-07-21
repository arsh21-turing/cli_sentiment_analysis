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
from src.ui.utils.session import SessionManager
from src.ui.utils.comparison_utils import ModelComparator


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


def main():
    """Main application entry point."""
    try:
        # Setup page configuration
        setup_page()
        
        # Initialize session state
        session_manager = SessionManager()
        session_manager.init_session()
        
        # Create sidebar with configuration parameters
        sidebar = SidebarComponent(title="Analysis Parameters")
        params = sidebar.create_sidebar()
        session_manager.update_params(params)
        
        # Sidebar actions
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Reset Settings", use_container_width=True):
                session_manager.reset_session(keep_api_keys=True)
                st.experimental_rerun()
        
        with col2:
            if st.button("Clear API Keys", use_container_width=True):
                session_manager.clear_api_keys()
                st.experimental_rerun()
        
        # Add Groq status indicator
        if check_groq_availability():
            st.sidebar.success("✅ Groq API is ready to use")
        else:
            st.sidebar.warning("⚠️ Groq API is not configured")
        
        # Add comparison status indicator
        if check_comparison_availability():
            st.sidebar.success("✅ Model comparison is available")
        else:
            st.sidebar.warning("⚠️ Model comparison requires Groq API")
        
        # Add global export options
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Global Export Options")
        
        if st.sidebar.button("Export Session Statistics", use_container_width=True):
            # Create the export manager
            from src.ui.components.export_manager import ExportManager
            export_manager = ExportManager()
            
            # Get session stats
            stats = session_manager.get_session_stats()
            formatted_stats = {"session_stats": stats}
            
            # Render export panel
            export_manager.render_export_panel("session", formatted_stats)
        
        # Check if we're comparing text from Groq analysis
        if st.session_state.get("compare_text", ""):
            compare_text = st.session_state.compare_text
            st.session_state.compare_text = ""  # Clear it after reading
            st.session_state.active_tab = "Single Text Analysis"
        else:
            compare_text = None
        
        # Create tabs for different analysis modes
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Single Text Analysis", 
            "📁 File Upload", 
            "🚀 Groq Analysis",
            "🔄 Model Comparison",
            "📊 Analytics Dashboard"
        ])
        
        # Handle Single Text Analysis tab
        with tab1:
            text_analysis = TextAnalysisComponent()
            
            # If we have text to compare, prefill it
            if compare_text:
                st.session_state.prefill_text = compare_text
                st.info("Text copied from Groq Analysis for comparison")
            
            text_analysis.render()
            
        # Handle File Upload tab  
        with tab2:
            file_upload = FileUploadComponent()
            file_upload.render()
            
        # Handle Groq Analysis tab
        with tab3:
            groq_analysis = GroqAnalysisComponent()
            groq_analysis.render()
        
        # Handle Model Comparison tab
        with tab4:
            comparison = ComparisonComponent()
            comparison.render()
        
        # Handle Analytics Dashboard tab
        with tab5:
            analytics_dashboard = AnalyticsDashboard()
            analytics_dashboard.render()
        
        # Footer with app info
        st.markdown("---")
        st.markdown(
            """
<div>
            Text Analysis Tool | Version 1.3 | Built with Streamlit | Featuring Advanced Analytics
</div>
            """, 
            unsafe_allow_html=True
        )

    except Exception as e:
        # Global error handling
        st.error("An unexpected error occurred in the application.")
        
        # Technical details in expander
        with st.expander("Technical Details"):
            st.write("Error details:")
            st.code(traceback.format_exc())
            
            # Suggestion for recovery
            st.info("""
            Suggestions:
            1. Try refreshing the page
            2. Check your input data
            3. Reset all settings using the button in the sidebar
            """)


if __name__ == "__main__":
    main()

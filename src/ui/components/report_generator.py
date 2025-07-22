"""
Report generator component for the Streamlit application.
"""

import streamlit as st
import io
from datetime import datetime
from src.ui.utils.report_builder import ReportBuilder
from typing import Dict, Any, Optional, List


class ReportGenerator:
    """
    Creates and manages report generation interface.
    """
    
    def __init__(self):
        """
        Initialize the report generator component.
        """
        self.report_builder = ReportBuilder()
    
    def render_options(self, results_type: str, results: Dict[str, Any]) -> None:
        """
        Render report generation options UI.
        
        Args:
            results_type (str): Type of results ("standard" or "groq")
            results (Dict): Results data to include in the report
        """
        if not results:
            st.warning("No results available to generate a report.")
            return
        
        st.subheader("Generate Analysis Report")
        st.markdown("Create a downloadable report containing the analysis results.")
        
        # Report options
        with st.expander("Report Options", expanded=True):
            # Report title
            report_title = st.text_input(
                "Report Title", 
                value=f"Text Analysis Report - {results.get('file_name', 'Batch Analysis')}"
            )
            
            # Report sections
            st.markdown("**Include Sections:**")
            col1, col2 = st.columns(2)
            
            with col1:
                include_summary = st.checkbox("Summary", value=True, 
                                              help="Include summary statistics")
                include_charts = st.checkbox("Charts", value=True, 
                                             help="Include visualization charts")
            
            with col2:
                include_details = st.checkbox("Detailed Results", value=True, 
                                              help="Include detailed results table")
                include_metadata = st.checkbox("Analysis Parameters", value=True, 
                                               help="Include analysis parameters used")
            
            # Row limit for detailed results
            if include_details:
                max_rows = st.slider(
                    "Maximum Rows in Detailed Results", 
                    min_value=10, 
                    max_value=1000, 
                    value=100,
                    step=10,
                    help="Limit the number of rows in the detailed results table"
                )
            else:
                max_rows = 100
        
        # Gather options
        options = {
            "title": report_title,
            "sections": [],
            "max_rows": max_rows
        }
        
        if include_summary:
            options["sections"].append("summary")
        if include_charts:
            options["sections"].append("charts")
        if include_details:
            options["sections"].append("details")
        if include_metadata:
            options["sections"].append("metadata")
        
        # Display download buttons
        st.markdown("### Download Report")
        
        # Arrange download buttons in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # PDF download button
            pdf_button = st.download_button(
                label="Download PDF Report",
                data=self._generate_report_pdf(results, options),
                file_name=self._generate_filename(report_title, "pdf"),
                mime="application/pdf",
                use_container_width=True
            )
        
        with col2:
            # CSV download button
            csv_button = st.download_button(
                label="Download CSV Report",
                data=self._generate_report_csv(results, options),
                file_name=self._generate_filename(report_title, "csv"),
                mime="text/csv",
                use_container_width=True
            )
        
        # Show preview option
        if st.checkbox("Show Report Preview"):
            self.preview_report(options, results)
    
    def _generate_report_pdf(self, results: Dict[str, Any], options: Dict[str, Any]) -> io.BytesIO:
        """
        Generate PDF report.
        
        Args:
            results (Dict): Results data to include in report
            options (Dict): Report generation options
            
        Returns:
            io.BytesIO: PDF report as BytesIO
        """
        # Set report title
        self.report_builder.title = options.get("title", "Text Analysis Report")
        
        # Create PDF report
        return self.report_builder.create_pdf_report(results, options)
    
    def _generate_report_csv(self, results: Dict[str, Any], options: Dict[str, Any]) -> io.BytesIO:
        """
        Generate CSV report.
        
        Args:
            results (Dict): Results data to include in report
            options (Dict): Report generation options
            
        Returns:
            io.BytesIO: CSV report as BytesIO
        """
        # Set report title
        self.report_builder.title = options.get("title", "Text Analysis Report")
        
        # Create CSV report
        return self.report_builder.create_csv_report(results, options)
    
    def _generate_filename(self, title: str, format: str) -> str:
        """
        Generate filename for the report.
        
        Args:
            title (str): Report title
            format (str): File format
            
        Returns:
            str: Filename
        """
        # Clean title to make it suitable for a filename
        clean_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        clean_title = clean_title.replace(" ", "_")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{clean_title}_{timestamp}.{format}"
    
    def preview_report(self, options: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Display a preview of the report.
        
        Args:
            options (Dict): Report generation options
            results (Dict): Results data to include in report
        """
        # Set report title
        self.report_builder.title = options.get("title", "Text Analysis Report")
        
        # Generate HTML for preview
        html = self.report_builder.generate_report_html(results, options)
        
        # Display HTML in an expander
        with st.expander("Report Preview", expanded=True):
            st.components.v1.html(html, height=700, scrolling=True) 
"""
Export manager component for the Streamlit application.
"""

import streamlit as st
import io
import json
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from io import BytesIO
from src.ui.utils.export_utils import ExportFormatter
from src.ui.utils.session import SessionManager


class ExportManager:
    """
    Creates and manages export functionality.
    """
    
    def __init__(self):
        """
        Initialize export manager component.
        """
        self.formatter = ExportFormatter()
        self.session_manager = SessionManager()
        
        # Available export formats
        self.formats = {
            "pdf": {
                "name": "PDF Report",
                "description": "Comprehensive report with charts and tables",
                "mime": "application/pdf",
                "icon": "📄"
            },
            "csv": {
                "name": "CSV",
                "description": "Detailed data in CSV format for spreadsheets",
                "mime": "text/csv",
                "icon": "📊"
            },
            "json": {
                "name": "JSON",
                "description": "Raw data in structured JSON format",
                "mime": "application/json",
                "icon": "🔍"
            },
            "excel": {
                "name": "Excel",
                "description": "Excel workbook with multiple sheets and charts",
                "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "icon": "📈"
            }
        }
        
        # Available templates
        self.templates = {
            "default": {
                "name": "Default Template",
                "description": "Standard report layout with sections for summary, details, and charts",
                "preview_img": "default_template.png"
            },
            "compact": {
                "name": "Compact Template",
                "description": "Condensed report format focusing on essential information",
                "preview_img": "compact_template.png"
            },
            "detailed": {
                "name": "Detailed Template",
                "description": "Comprehensive report with expanded details and explanations",
                "preview_img": "detailed_template.png"
            },
            "presentation": {
                "name": "Presentation Template",
                "description": "Visual report designed for presentations with larger charts",
                "preview_img": "presentation_template.png"
            },
            "custom1": {
                "name": "Custom Template 1",
                "description": "User-defined template with customizable sections",
                "preview_img": "custom1_template.png"
            },
            "custom2": {
                "name": "Custom Template 2",
                "description": "User-defined template with alternative layout",
                "preview_img": "custom2_template.png"
            }
        }
    
    def render_export_panel(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """
        Render export UI panel.
        
        Args:
            analysis_type (str): Type of analysis ("text", "batch", "groq", "comparison", "session")
            results (Dict): Results data to export
        """
        if not results:
            st.info("No results available to export.")
            return
        
        st.markdown("## Export Results")
        st.markdown("Export your analysis results in different formats.")
        
        # Create tabs for different export options
        export_tabs = st.tabs([
            f"{self.formats['pdf']['icon']} PDF Report", 
            f"{self.formats['csv']['icon']} CSV Export", 
            f"{self.formats['json']['icon']} JSON Export",
            f"{self.formats['excel']['icon']} Excel Export"
        ])
        
        # PDF Export Tab
        with export_tabs[0]:
            self._render_pdf_export(analysis_type, results)
        
        # CSV Export Tab
        with export_tabs[1]:
            self._render_csv_export(analysis_type, results)
        
        # JSON Export Tab
        with export_tabs[2]:
            self._render_json_export(analysis_type, results)
        
        # Excel Export Tab
        with export_tabs[3]:
            self._render_excel_export(analysis_type, results)
        
        # Add session stats export option
        st.markdown("---")
        self._render_session_stats_export()
    
    def _render_pdf_export(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """
        Render PDF export options.
        
        Args:
            analysis_type (str): Type of analysis
            results (Dict): Results data
        """
        st.markdown("### PDF Report Export")
        st.markdown("Create a comprehensive PDF report of your analysis results.")
        
        # Template selection
        template = st.selectbox(
            "Report Template",
            options=list(self.templates.keys()),
            format_func=lambda x: self.templates[x]["name"],
            help="Select a template for your report"
        )
        
        # Display template description
        st.caption(self.templates[template]["description"])
        
        # Report title
        default_title = f"{analysis_type.capitalize()} Analysis Report"
        if "file_name" in results:
            default_title = f"Analysis Report - {results['file_name']}"
        
        report_title = st.text_input("Report Title", value=default_title)
        
        # Content options
        st.markdown("#### Content Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_summary = st.checkbox("Include Summary", value=True)
            include_charts = st.checkbox("Include Charts", value=True)
        
        with col2:
            include_details = st.checkbox("Include Detailed Results", value=True)
            include_metadata = st.checkbox("Include Analysis Parameters", value=True)
        
        # Row limit for detailed results
        if include_details:
            max_rows = st.slider(
                "Maximum Rows in Detailed Results", 
                min_value=10, 
                max_value=500, 
                value=100,
                step=10
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
        
        # Preview button
        if st.button("Preview PDF Report", key="preview_pdf"):
            self._preview_report(results, options, template)
        
        # Download button
        pdf_data = self.export_results(results, "pdf", template, options)
        filename = self.generate_filename(analysis_type, "pdf", report_title)
        
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name=filename,
            mime=self.formats["pdf"]["mime"],
            use_container_width=True
        )
    
    def _render_csv_export(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """
        Render CSV export options.
        
        Args:
            analysis_type (str): Type of analysis
            results (Dict): Results data
        """
        st.markdown("### CSV Export")
        st.markdown("Export detailed results in CSV format for use in spreadsheets.")
        
        # CSV Export Options
        st.markdown("#### Export Options")
        
        include_metadata = st.checkbox("Include Metadata Row", value=True, 
                                      help="Include a metadata row with analysis details")
        
        detailed_only = st.checkbox("Export Detailed Results Only", value=False,
                                   help="Export only the detailed results without summary data")
        
        # Gather options
        options = {
            "include_metadata": include_metadata,
            "detailed_only": detailed_only
        }
        
        # Download button
        csv_data = self.export_results(results, "csv", None, options)
        filename = self.generate_filename(analysis_type, "csv")
        
        st.download_button(
            label="Download CSV Data",
            data=csv_data,
            file_name=filename,
            mime=self.formats["csv"]["mime"],
            use_container_width=True
        )
        
        # Show preview
        with st.expander("Preview CSV Data", expanded=False):
            # Read the CSV data into a pandas DataFrame
            csv_io = io.StringIO(csv_data.decode('utf-8'))
            try:
                df = pd.read_csv(csv_io)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error previewing CSV data: {e}")
                st.code(csv_data.decode('utf-8')[:1000])
    
    def _render_json_export(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """
        Render JSON export options.
        
        Args:
            analysis_type (str): Type of analysis
            results (Dict): Results data
        """
        st.markdown("### JSON Export")
        st.markdown("Export raw data in JSON format for programmatic use.")
        
        # JSON Export Options
        st.markdown("#### Export Options")
        
        pretty_print = st.checkbox("Pretty Print JSON", value=True,
                                  help="Format the JSON with indentation for readability")
        
        include_all = st.checkbox("Include All Data", value=True,
                                 help="Include all data in the export (may be large)")
        
        # Gather options
        options = {
            "pretty_print": pretty_print,
            "include_all": include_all
        }
        
        # Download button
        json_data = self.export_results(results, "json", None, options)
        filename = self.generate_filename(analysis_type, "json")
        
        st.download_button(
            label="Download JSON Data",
            data=json_data,
            file_name=filename,
            mime=self.formats["json"]["mime"],
            use_container_width=True
        )
        
        # Show preview
        with st.expander("Preview JSON Data", expanded=False):
            st.json(json.loads(json_data))
    
    def _render_excel_export(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """
        Render Excel export options.
        
        Args:
            analysis_type (str): Type of analysis
            results (Dict): Results data
        """
        st.markdown("### Excel Export")
        st.markdown("Export results as an Excel workbook with multiple sheets and charts.")
        
        # Excel Export Options
        st.markdown("#### Export Options")
        
        include_charts = st.checkbox("Include Charts", value=True,
                                    help="Include charts in the Excel file")
        
        include_formulas = st.checkbox("Include Formulas", value=True,
                                      help="Include formulas for calculations")
        
        # Gather options
        options = {
            "include_charts": include_charts,
            "include_formulas": include_formulas
        }
        
        # Download button
        excel_data = self.export_results(results, "excel", None, options)
        filename = self.generate_filename(analysis_type, "xlsx")
        
        st.download_button(
            label="Download Excel Workbook",
            data=excel_data,
            file_name=filename,
            mime=self.formats["excel"]["mime"],
            use_container_width=True
        )
        
        # Show info about the Excel file
        st.info("""
        The Excel workbook contains multiple sheets:
        - Metadata: Information about the analysis
        - Summary: Summary of results
        - Detailed Results: Detailed analysis data
        - Charts: Visualizations of the analysis results
        """)
    
    def _render_session_stats_export(self) -> None:
        """
        Render session statistics export options.
        """
        st.subheader("Export Session Statistics")
        st.markdown("Export statistics about your current analysis session.")
        
        # Get session stats
        stats = self.session_manager.get_session_stats()
        
        # Display session stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Texts Analyzed", stats.get("texts_analyzed", 0))
        
        with col2:
            requests = stats.get("standard_requests", 0) + stats.get("groq_requests", 0)
            st.metric("Total Requests", requests)
        
        with col3:
            if stats.get("session_duration"):
                duration_mins = stats["session_duration"].total_seconds() / 60
                st.metric("Session Time", f"{int(duration_mins)} min")
        
        # Export options in columns
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # PDF Export
            session_pdf = self.export_session_stats("pdf")
            st.download_button(
                label="Session Report (PDF)",
                data=session_pdf,
                file_name=f"session_stats_{self.formatter.format_timestamp()}.pdf",
                mime=self.formats["pdf"]["mime"]
            )
        
        with export_col2:
            # JSON Export
            session_json = self.export_session_stats("json")
            st.download_button(
                label="Session Stats (JSON)",
                data=session_json,
                file_name=f"session_stats_{self.formatter.format_timestamp()}.json",
                mime=self.formats["json"]["mime"]
            )
        
        with export_col3:
            # Excel Export
            session_excel = self.export_session_stats("excel")
            st.download_button(
                label="Session Stats (Excel)",
                data=session_excel,
                file_name=f"session_stats_{self.formatter.format_timestamp()}.xlsx",
                mime=self.formats["excel"]["mime"]
            )
    
    def export_results(self, results: Dict[str, Any], export_format: str, 
                      template: Optional[str] = None, options: Dict[str, Any] = {}) -> BytesIO:
        """
        Export results in the specified format.
        
        Args:
            results (Dict): Results data to export
            export_format (str): Format to export ("pdf", "csv", "json", "excel")
            template (str, optional): Report template to use
            options (Dict): Export options like sections to include
            
        Returns:
            BytesIO: File-like object containing the exported data
        """
        if export_format == "pdf":
            return self._export_pdf(results, template, options)
        elif export_format == "csv":
            return self._export_csv(results, options)
        elif export_format == "json":
            return self._export_json(results, options)
        elif export_format == "excel":
            return self._export_excel(results, options)
        else:
            raise ValueError(f"Unknown export format: {export_format}")
    
    def _export_pdf(self, results: Dict[str, Any], template: str, options: Dict[str, Any]) -> BytesIO:
        """
        Export results as PDF.
        
        Args:
            results (Dict): Results data
            template (str): Template to use
            options (Dict): Export options
            
        Returns:
            BytesIO: PDF file
        """
        # For now, create a simple text-based PDF
        # In a real implementation, you would use a proper PDF library
        pdf_content = f"""
        Analysis Report
        ===============
        
        Title: {options.get('title', 'Analysis Report')}
        Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Results Summary:
        {json.dumps(results, indent=2, default=str)}
        """
        
        # Convert to bytes
        return io.BytesIO(pdf_content.encode('utf-8'))
    
    def _export_csv(self, results: Dict[str, Any], options: Dict[str, Any]) -> BytesIO:
        """
        Export results as CSV.
        
        Args:
            results (Dict): Results data
            options (Dict): Export options
            
        Returns:
            BytesIO: CSV file
        """
        # Handle different result types
        csv_file = io.BytesIO()
        
        if options.get("detailed_only", False):
            # Export only detailed results
            detailed_results = None
            
            if "batch_results" in results and "detailed_results" in results["batch_results"]:
                detailed_results = results["batch_results"]["detailed_results"]
            elif "detailed_results" in results:
                detailed_results = results["detailed_results"]
            
            if detailed_results:
                # Convert to DataFrame and export
                df = pd.DataFrame(detailed_results)
                df.to_csv(csv_file, index=False)
            else:
                # No detailed results, export empty file
                csv_file.write(b"No detailed results available")
        else:
            # Export with metadata if requested
            if options.get("include_metadata", True):
                # Write metadata header
                metadata = [
                    ["Export Date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["Analysis Type", results.get("analysis_type", "Unknown")]
                ]
                
                # Add file information if available
                if "file_name" in results:
                    metadata.append(["File Name", results["file_name"]])
                
                if "total_texts" in results:
                    metadata.append(["Total Texts", results["total_texts"]])
                
                # Add parameters if available
                if "parameters" in results:
                    for key, value in results["parameters"].items():
                        metadata.append([f"Parameter: {key}", value])
                
                # Write metadata rows
                writer = pd.DataFrame(metadata).to_csv(csv_file, header=False, index=False)
                
                # Add separator
                csv_file.write(b"\n")
            
            # Write main data
            if "batch_results" in results and "detailed_results" in results["batch_results"]:
                pd.DataFrame(results["batch_results"]["detailed_results"]).to_csv(csv_file, index=False)
            elif "detailed_results" in results:
                pd.DataFrame(results["detailed_results"]).to_csv(csv_file, index=False)
            else:
                # Fallback to flat representation of whatever data we have
                pd.json_normalize(results).to_csv(csv_file, index=False)
        
        csv_file.seek(0)
        return csv_file
    
    def _export_json(self, results: Dict[str, Any], options: Dict[str, Any]) -> BytesIO:
        """
        Export results as JSON.
        
        Args:
            results (Dict): Results data
            options (Dict): Export options
            
        Returns:
            BytesIO: JSON file
        """
        # Add export metadata
        export_data = {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "export_version": "1.0"
        }
        
        # Include all results or just the detailed ones
        if options.get("include_all", True):
            export_data["results"] = results
        else:
            # Extract just the detailed results
            detailed_results = None
            
            if "batch_results" in results and "detailed_results" in results["batch_results"]:
                detailed_results = results["batch_results"]["detailed_results"]
            elif "detailed_results" in results:
                detailed_results = results["detailed_results"]
            
            if detailed_results:
                export_data["detailed_results"] = detailed_results
            else:
                export_data["results"] = results
        
        # Format as JSON
        json_str = self.formatter.format_to_json(export_data, options.get("pretty_print", True))
        
        # Convert to bytes
        json_bytes = json_str.encode('utf-8')
        
        # Return as BytesIO
        json_file = io.BytesIO(json_bytes)
        return json_file
    
    def _export_excel(self, results: Dict[str, Any], options: Dict[str, Any]) -> BytesIO:
        """
        Export results as Excel.
        
        Args:
            results (Dict): Results data
            options (Dict): Export options
            
        Returns:
            BytesIO: Excel file
        """
        # Use the ExportFormatter to create Excel
        excel_data = self.formatter.format_to_excel(results)
        
        return excel_data
    
    def export_session_stats(self, format_type: str = "pdf") -> BytesIO:
        """
        Export current session statistics.
        
        Args:
            format_type (str): Format to export ("pdf", "json", "excel")
            
        Returns:
            BytesIO: File containing session statistics
        """
        # Get session stats
        stats = self.session_manager.get_session_stats()
        
        # Format the stats
        formatted_stats = self.formatter.format_session_stats(stats)
        
        if format_type == "pdf":
            # Create PDF report
            pdf_content = f"""
            Session Statistics Report
            =========================
            
            Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Statistics:
            {json.dumps(formatted_stats, indent=2, default=str)}
            """
            
            return io.BytesIO(pdf_content.encode('utf-8'))
        
        elif format_type == "json":
            # Format as JSON
            json_str = self.formatter.format_to_json(formatted_stats, True)
            json_bytes = json_str.encode('utf-8')
            return io.BytesIO(json_bytes)
        
        elif format_type == "excel":
            # Format as Excel
            return self.formatter.format_to_excel(formatted_stats)
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def generate_filename(self, analysis_type: str, export_format: str, 
                         title: Optional[str] = None) -> str:
        """
        Generate filename for export.
        
        Args:
            analysis_type (str): Type of analysis
            export_format (str): Export format 
            title (str, optional): Custom title
            
        Returns:
            str: Generated filename with timestamp
        """
        # Get timestamp
        timestamp = self.formatter.format_timestamp()
        
        # Clean and format the title if provided
        if title:
            # Remove special characters and spaces
            clean_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)
            clean_title = clean_title.replace(" ", "_")
            base_name = f"{clean_title}_{timestamp}"
        else:
            # Use analysis type as base name
            base_name = f"{analysis_type}_analysis_{timestamp}"
        
        # Add appropriate extension
        if export_format == "excel":
            return f"{base_name}.xlsx"
        else:
            return f"{base_name}.{export_format}"
    
    def _preview_report(self, results: Dict[str, Any], options: Dict[str, Any], template: str) -> None:
        """
        Display a preview of the report.
        
        Args:
            results (Dict): Results data
            options (Dict): Report options
            template (str): Template to use
        """
        st.info("Preview functionality will be implemented in a future update.")
        st.json(results)
    
    def get_available_templates(self) -> List[str]:
        """
        Get list of available templates.
        
        Returns:
            List[str]: List of available template names
        """
        return list(self.templates.keys())
    
    def get_template_preview(self, template_name: str) -> str:
        """
        Get preview of a template.
        
        Args:
            template_name (str): Name of template
            
        Returns:
            str: HTML preview of the template
        """
        # In a real implementation, this would load the actual template
        # For now, return a placeholder
        return f"""
        <h1>Preview of {template_name}</h1>
        <p>This is a preview of the template.</p>
        """ 
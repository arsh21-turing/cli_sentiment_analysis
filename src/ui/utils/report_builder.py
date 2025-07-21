"""
Report builder utilities for PDF and CSV report generation.
"""

import io
import csv
import datetime
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import pandas as pd
import base64
import jinja2
import weasyprint
from src.ui.utils.chart_generator import ChartGenerator


class ReportBuilder:
    """
    Creates PDF and CSV reports from analysis results.
    """
    
    def __init__(self, title="Text Analysis Report"):
        """
        Initialize the report builder.
        
        Args:
            title (str): Title for the generated report
        """
        self.title = title
        self.chart_generator = ChartGenerator(output_format="png")
        self.template_loader = jinja2.FileSystemLoader(searchpath="./src/ui/templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
    
    def create_pdf_report(self, results: Dict[str, Any], options: Dict[str, Any] = {}) -> BytesIO:
        """
        Create a PDF report.
        
        Args:
            results (Dict): Analysis results data
            options (Dict): Options for report generation
            
        Returns:
            BytesIO: PDF report as BytesIO
        """
        # Generate HTML for the report
        html = self.generate_report_html(results, options)
        
        # Convert HTML to PDF
        pdf_file = io.BytesIO()
        weasyprint.HTML(string=html).write_pdf(pdf_file)
        pdf_file.seek(0)
        
        return pdf_file
    
    def create_csv_report(self, results: Dict[str, Any], options: Dict[str, Any] = {}) -> BytesIO:
        """
        Create a CSV report.
        
        Args:
            results (Dict): Analysis results data
            options (Dict): Options for report generation
            
        Returns:
            BytesIO: CSV report as BytesIO
        """
        csv_file = io.BytesIO()
        
        # Get batch results
        batch_results = results.get("batch_results", {})
        detailed_results = batch_results.get("detailed_results", [])
        
        # Check if we have detailed results
        if not detailed_results:
            # Create a simple CSV with summary data
            writer = csv.writer(csv_file)
            
            # Write header
            writer.writerow(["Report Type", "File Name", "Generated At", "Total Texts"])
            
            # Write summary data
            writer.writerow([
                "Text Analysis Report",
                results.get("file_name", "Unknown"),
                self.get_timestamp(),
                results.get("total_texts", 0)
            ])
            
            # Write sentiment counts if available
            if "sentiment_counts" in batch_results:
                writer.writerow([])  # Empty row
                writer.writerow(["Sentiment Counts"])
                for sentiment, count in batch_results["sentiment_counts"].items():
                    writer.writerow([sentiment, count])
            
            # Write emotion distribution if available
            if "emotion_distribution" in batch_results:
                writer.writerow([])  # Empty row
                writer.writerow(["Emotion Distribution"])
                for emotion, count in batch_results["emotion_distribution"].items():
                    writer.writerow([emotion, count])
        else:
            # Create a detailed CSV from the results DataFrame
            df = pd.DataFrame(detailed_results)
            
            # Add metadata
            metadata = pd.DataFrame({
                "Report Info": ["Generated At", "File Name", "Total Texts", "Processing Type"],
                "Value": [
                    self.get_timestamp(),
                    results.get("file_name", "Unknown"),
                    results.get("total_texts", 0),
                    results.get("processing_type", "standard")
                ]
            })
            
            # Write to CSV
            metadata.to_csv(csv_file, index=False)
            csv_file.write(b"\n")  # Add empty row
            df.to_csv(csv_file, index=False)
        
        csv_file.seek(0)
        return csv_file
    
    def generate_report_html(self, results: Dict[str, Any], options: Dict[str, Any] = {}) -> str:
        """
        Generate HTML for the PDF report.
        
        Args:
            results (Dict): Analysis results data
            options (Dict): Options for report generation
            
        Returns:
            str: HTML string for the report
        """
        try:
            # Try to load the template
            template = self.template_env.get_template("report_template.html")
            
            # Initialize HTML with template
            html = template.render(
                title=self.title,
                timestamp=self.get_timestamp(),
                filename=results.get("file_name", "Unknown"),
                total_texts=results.get("total_texts", 0),
                processing_type=results.get("processing_type", "standard")
            )
        except jinja2.exceptions.TemplateNotFound:
            # If template not found, create basic HTML
            html = f"""
            <!DOCTYPE html>
<html>
<head>
<title>
{self.title}

</title>
<style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .chart {{ width: 100%; margin: 20px 0; text-align: center; }}
                .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
</style>
</head>
<body>
<div>
<h1>
{self.title}

</h1>
<p>
Generated on {self.get_timestamp()}

</p>
</div>
        """
    
        # Add sections based on options
        sections = options.get("sections", ["summary", "charts", "details", "metadata"])
        
        # Add summary section
        if "summary" in sections:
            html = self.add_summary_section(results, html)
        
        # Add charts section
        if "charts" in sections:
            html = self.add_charts_section(results, html)
        
        # Add detailed results section
        if "details" in sections:
            max_rows = options.get("max_rows", 100)
            html = self.add_detailed_results_section(results, html, max_rows)
        
        # Add metadata section
        if "metadata" in sections:
            html = self.add_metadata_section(results, html)
        
        # Close HTML if using basic template
        if "body" not in html:
            html += """
<div>
<p>
Generated by Text Analysis Tool

</p>
</div>
</body>
</html>
        """
    
        return html

    def get_timestamp(self) -> str:
        """
        Get formatted timestamp for the report.
        
        Returns:
            str: Formatted timestamp
        """
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def add_summary_section(self, results: Dict[str, Any], html: str) -> str:
        """
        Add summary section to the report.
        
        Args:
            results (Dict): Analysis results data
            html (str): Current HTML content
            
        Returns:
            str: Updated HTML with summary section
        """
        batch_results = results.get("batch_results", {})
        
        # Create summary section content
        summary_content = f"""
<div>
<h2>
Summary

</h2>
<table>
<tr>
<th>
Metric

</th>
<th>
Value

</th>
</tr>
<tr>
<td>
File Name

</td>
<td>
{results.get("file_name", "Unknown")}

</td>
</tr>
<tr>
<td>
Total Texts Analyzed

</td>
<td>
{results.get("total_texts", 0)}

</td>
</tr>
        """
        
        # Add processing time if available (Groq)
        if "processing_time" in batch_results:
            summary_content += f"""
<tr>
<td>
Processing Time

</td>
<td>
{batch_results.get("processing_time", 0):.2f} seconds

</td>
</tr>
            """
        
        # Add model info if available (Groq)
        if "model" in batch_results:
            summary_content += f"""
<tr>
<td>
Model Used

</td>
<td>
{batch_results.get("model", "Unknown")}

</td>
</tr>
            """
        
        # Add sentiment counts if available
        if "sentiment_counts" in batch_results:
            sentiment_counts = batch_results["sentiment_counts"]
            for sentiment, count in sentiment_counts.items():
                sentiment_label = sentiment.capitalize()
                summary_content += f"""
<tr>
<td>
{sentiment_label} Texts

</td>
<td>
{count}

</td>
</tr>
                """
        
        # Add average confidence if available
        if "average_confidence" in batch_results:
            avg_conf = batch_results["average_confidence"]
            summary_content += f"""
<tr>
<td>
Average Confidence

</td>
<td>
{avg_conf:.2%}

</td>
</tr>
            """
        
        summary_content += """
</table>
</div>
        """
        
        # Add to HTML before the closing body tag
        if "</body>" in html:
            html = html.replace("</body>", summary_content + "</body>")
        else:
            html += summary_content
        
        return html

    def add_charts_section(self, results: Dict[str, Any], html: str) -> str:
        """
        Add charts section to the report.
        
        Args:
            results (Dict): Analysis results data
            html (str): Current HTML content
            
        Returns:
            str: Updated HTML with charts section
        """
        batch_results = results.get("batch_results", {})
        
        # Start charts section
        charts_content = """
<div>
<h2>
Charts

</h2>
        """
        
        # Sentiment distribution chart
        if "sentiment_counts" in batch_results and batch_results["sentiment_counts"]:
            sentiment_data = batch_results["sentiment_counts"]
            chart_img = self.chart_generator.create_bar_chart(
                sentiment_data, "Sentiment Distribution", width=800, height=400
            )
            # Convert to base64 for embedding in HTML
            chart_base64 = base64.b64encode(chart_img.getvalue()).decode()
            
            charts_content += f"""
            <div class="chart">
<h3>
Sentiment Distribution

</h3>
<img src="data:image/png;base64,{chart_base64}" alt="Sentiment Distribution" />
</div>
            """
        
        # Emotion distribution chart
        if "emotion_distribution" in batch_results and batch_results["emotion_distribution"]:
            emotion_data = batch_results["emotion_distribution"]
            
            # Create pie chart for emotions
            chart_img = self.chart_generator.create_pie_chart(
                emotion_data, "Emotion Distribution", width=800, height=500
            )
            # Convert to base64 for embedding in HTML
            chart_base64 = base64.b64encode(chart_img.getvalue()).decode()
            
            charts_content += f"""
<div>
<h3>
Emotion Distribution

</h3>
<img src="data:image/png;base64,{chart_base64}" alt="Emotion Distribution" />
</div>
            """
        
        # Close charts section
        charts_content += """
        </div>
        """
        
        # Add to HTML before the closing body tag
        if "</body>" in html:
            html = html.replace("</body>", charts_content + "</body>")
        else:
            html += charts_content
        
        return html

    def add_detailed_results_section(self, results: Dict[str, Any], html: str, 
                                     max_rows: int = 100) -> str:
        """
        Add detailed results section to the report.
        
        Args:
            results (Dict): Analysis results data
            html (str): Current HTML content
            max_rows (int): Maximum number of rows to include
            
        Returns:
            str: Updated HTML with detailed results section
        """
        batch_results = results.get("batch_results", {})
        detailed_results = batch_results.get("detailed_results", [])
        
        # Start detailed results section
        detailed_content = f"""
<div>
<h2>
Detailed Results

</h2>
        """
        
        if detailed_results:
            # Limit the number of rows
            limited_results = detailed_results[:max_rows]
            
            # Create table of detailed results
            detailed_content += """
<table>
<tr>
            """
            
            # Add column headers
            columns = list(limited_results[0].keys())
            for col in columns:
                detailed_content += f'<th>{col.capitalize().replace("_", " ")}</th>'
            detailed_content += '</tr>'

            # Add rows
            for result in limited_results:
                detailed_content += '<tr>'
                for col in columns:
                    value = result.get(col, "")

                    # Format specific values
                    if col == "confidence" and isinstance(value, (int, float)):
                        value = f"{value:.2%}"
                    elif col == "text":
                        value = value[:50] + "..." if len(value) > 50 else value
                    elif col == "emotion_scores" and isinstance(value, dict):
                        value = ", ".join([f"{k}: {v:.2f}" for k, v in value.items() 
                                          if v > 0.1])
                    
                    detailed_content += f'<td>{value}</td>'
                detailed_content += '</tr>'

            detailed_content += """
</table>
            """
            
            # Add note if results were limited
            if len(detailed_results) > max_rows:
                detailed_content += f"""
<p>
<em>
Note: Showing {max_rows} of {len(detailed_results)} results.

</em>
</p>
                """
        else:
            detailed_content += """
<p>
No detailed results available.

</p>
            """
        
        # Close detailed results section
        detailed_content += """
</div>
        """
        
        # Add to HTML before the closing body tag
        if "</body>" in html:
            html = html.replace("</body>", detailed_content + "</body>")
        else:
            html += detailed_content
        
        return html

    def add_metadata_section(self, results: Dict[str, Any], html: str) -> str:
        """
        Add metadata section to the report.
        
        Args:
            results (Dict): Analysis results data
            html (str): Current HTML content
            
        Returns:
            str: Updated HTML with metadata section
        """
        processing_type = results.get("processing_type", "standard")
        parameters = results.get("parameters", {})
        
        # Start metadata section
        metadata_content = """
<div>
<h2>
Analysis Parameters

</h2>
<table>
<tr>
<th>
Parameter

</th>
<th>
Value

</th>
</tr>
        """
        
        # Add common parameters
        if processing_type == "standard":
            # Add standard parameters
            metadata_content += f"""
<tr>
<td>
Sentiment Threshold

</td>
<td>
{parameters.get("sentiment_threshold", 0.5):.2f}

</td>
</tr>
<tr>
<td>
Emotion Threshold

</td>
<td>
{parameters.get("emotion_threshold", 0.3):.2f}

</td>
</tr>
<tr>
<td>
API Fallback

</td>
<td>
{"Enabled" if parameters.get("use_api_fallback", False) else "Disabled"}

</td>
</tr>
            """
        else:
            # Add Groq parameters
            metadata_content += f"""
<tr>
<td>
Processing Type

</td>
<td>
Groq API

</td>
</tr>
<tr>
<td>
Model

</td>
<td>
{parameters.get("groq_model", "Unknown")}

</td>
</tr>
<tr>
<td>
Sentiment Analysis

</td>
<td>
{"Enabled" if parameters.get("analyze_sentiment", True) else "Disabled"}

</td>
</tr>
<tr>
<td>
Emotion Analysis

</td>
<td>
{"Enabled" if parameters.get("analyze_emotion", True) else "Disabled"}

</td>
</tr>
<tr>
<td>
Batch Size

</td>
<td>
{parameters.get("batch_size", 5)}

</td>
</tr>
            """
        
        # Close metadata section
        metadata_content += """
</table>
</div>
        """
        
        # Add to HTML before the closing body tag
        if "</body>" in html:
            html = html.replace("</body>", metadata_content + "</body>")
        else:
            html += metadata_content
        
        return html 
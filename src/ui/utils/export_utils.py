"""
Export formatting utilities for different export formats.
"""

import io
import json
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from io import BytesIO
import base64


class ExportFormatter:
    """
    Formats data for different export formats.
    """
    
    def __init__(self):
        """
        Initialize the export formatter.
        """
        pass
    
    def format_to_json(self, data: Dict[str, Any], pretty_print: bool = True) -> str:
        """
        Format data as JSON.
        
        Args:
            data (Dict): Data to format
            pretty_print (bool): Whether to pretty print the JSON
            
        Returns:
            str: JSON string
        """
        if pretty_print:
            return json.dumps(data, indent=2, default=self._json_serializer)
        else:
            return json.dumps(data, default=self._json_serializer)
    
    def _json_serializer(self, obj):
        """
        Custom JSON serializer to handle non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serializable representation of the object
        """
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)
    
    def format_to_excel(self, data: Dict[str, Any]) -> BytesIO:
        """
        Format data as Excel file.
        
        Args:
            data (Dict): Data to format
            
        Returns:
            BytesIO: Excel file
        """
        # Create an Excel file in memory
        excel_file = io.BytesIO()
        
        # Try to determine the type of data and create appropriate sheets
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            # Add metadata sheet
            self._add_metadata_sheet(writer, data)
            
            # Add main results sheet
            self._add_results_sheet(writer, data)
            
            # Add detailed results if available
            self._add_detailed_results_sheet(writer, data)
            
            # Generate charts if supported
            workbook = writer.book
            self.generate_excel_charts(workbook, data)
        
        excel_file.seek(0)
        return excel_file
    
    def _add_metadata_sheet(self, writer: Any, data: Dict[str, Any]) -> None:
        """
        Add metadata sheet to Excel file.
        
        Args:
            writer: Excel writer
            data (Dict): Data to add
        """
        # Extract metadata
        metadata = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Analysis Type": data.get("analysis_type", "Unknown")
        }
        
        # Add file information if available
        if "file_name" in data:
            metadata["File Name"] = data["file_name"]
        
        if "total_texts" in data:
            metadata["Total Texts"] = data["total_texts"]
        
        if "processing_type" in data:
            metadata["Processing Type"] = data["processing_type"]
        
        # Add parameters if available
        if "parameters" in data:
            for key, value in data["parameters"].items():
                metadata[f"Parameter: {key}"] = value
        
        # Convert to DataFrame
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Property", "Value"])
        
        # Write to Excel
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        
        # Format the sheet
        workbook = writer.book
        worksheet = writer.sheets["Metadata"]
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3E0EA',
            'border': 1
        })
        
        # Apply formatting to header row
        for col_num, value in enumerate(metadata_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 50)
    
    def _add_results_sheet(self, writer: Any, data: Dict[str, Any]) -> None:
        """
        Add main results sheet to Excel file.
        
        Args:
            writer: Excel writer
            data (Dict): Data to add
        """
        # Different handling based on analysis type
        if "batch_results" in data:
            # Batch analysis results
            batch_results = data["batch_results"]
            
            # Create summary data
            summary_data = {}
            
            # Add sentiment counts if available
            if "sentiment_counts" in batch_results:
                for sentiment, count in batch_results["sentiment_counts"].items():
                    summary_data[f"{sentiment.capitalize()} Texts"] = count
            
            # Add confidence if available
            if "average_confidence" in batch_results:
                summary_data["Average Confidence"] = batch_results["average_confidence"]
            
            # Add processing info if available
            if "processing_time" in batch_results:
                summary_data["Processing Time (s)"] = batch_results["processing_time"]
                summary_data["Texts/Second"] = data.get("total_texts", 0) / batch_results["processing_time"]
            
            # Convert to DataFrame
            summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
            
            # Write to Excel
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Format the sheet
            workbook = writer.book
            worksheet = writer.sheets["Summary"]
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3E0EA',
                'border': 1
            })
            
            # Apply formatting to header row
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Adjust column widths
            worksheet.set_column('A:A', 25)
            worksheet.set_column('B:B', 15)
            
        elif "comparison_metrics" in data:
            # Comparison results
            metrics = data["comparison_metrics"]
            
            # Create summary data
            if metrics:
                summary_data = {}
                
                if "agreement_score" in metrics:
                    summary_data["Agreement Score"] = metrics["agreement_score"]
                
                if "confidence_variance" in metrics:
                    summary_data["Confidence Variance"] = metrics["confidence_variance"]
                
                if "recommendation" in metrics:
                    summary_data["Recommendation"] = metrics["recommendation"]
                
                # Convert to DataFrame
                summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
                
                # Write to Excel
                summary_df.to_excel(writer, sheet_name="Comparison", index=False)
                
                # Format the sheet
                workbook = writer.book
                worksheet = writer.sheets["Comparison"]
                
                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D3E0EA',
                    'border': 1
                })
                
                # Apply formatting to header row
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Adjust column widths
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:B', 50)
        
        elif "session_stats" in data:
            # Session statistics
            stats = data["session_stats"]
            
            # Create summary data
            summary_data = {}
            
            if "texts_analyzed" in stats:
                summary_data["Texts Analyzed"] = stats["texts_analyzed"]
            
            if "standard_requests" in stats:
                summary_data["Standard Requests"] = stats["standard_requests"]
            
            if "groq_requests" in stats:
                summary_data["Groq Requests"] = stats["groq_requests"]
            
            if "session_duration" in stats and stats["session_duration"]:
                duration_seconds = stats["session_duration"].total_seconds()
                summary_data["Session Duration (minutes)"] = duration_seconds / 60
            
            # Convert to DataFrame
            summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
            
            # Write to Excel
            summary_df.to_excel(writer, sheet_name="Session Stats", index=False)
            
            # Format the sheet
            workbook = writer.book
            worksheet = writer.sheets["Session Stats"]
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3E0EA',
                'border': 1
            })
            
            # Apply formatting to header row
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Adjust column widths
            worksheet.set_column('A:A', 25)
            worksheet.set_column('B:B', 15)
    
    def _add_detailed_results_sheet(self, writer: Any, data: Dict[str, Any]) -> None:
        """
        Add detailed results sheet to Excel file.
        
        Args:
            writer: Excel writer
            data (Dict): Data to add
        """
        detailed_results = None
        
        # Extract detailed results based on data type
        if "batch_results" in data and "detailed_results" in data["batch_results"]:
            detailed_results = data["batch_results"]["detailed_results"]
        elif "detailed_results" in data:
            detailed_results = data["detailed_results"]
        
        if detailed_results:
            # Convert to DataFrame
            df = pd.DataFrame(detailed_results)
            
            # Write to Excel
            df.to_excel(writer, sheet_name="Detailed Results", index=False)
            
            # Format the sheet
            workbook = writer.book
            worksheet = writer.sheets["Detailed Results"]
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3E0EA',
                'border': 1
            })
            
            # Apply formatting to header row
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Adjust column widths based on content
            for i, col in enumerate(df.columns):
                # Set column width based on column name and content
                max_len = max(
                    df[col].astype(str).map(len).max(),  # max length of values
                    len(str(col))  # length of column name
                )
                # Add a little extra space
                worksheet.set_column(i, i, min(max_len + 2, 50))
    
    def generate_excel_charts(self, workbook: Any, data: Dict[str, Any]) -> None:
        """
        Generate charts in Excel workbook.
        
        Args:
            workbook: Excel workbook
            data (Dict): Data to create charts from
        """
        # Check if we can add charts
        if "batch_results" in data:
            # Try to add sentiment distribution chart
            try:
                self._add_sentiment_chart(workbook, data)
            except Exception:
                pass
            
            # Try to add emotion distribution chart
            try:
                self._add_emotion_chart(workbook, data)
            except Exception:
                pass
        
        elif "comparison_metrics" in data:
            # Try to add agreement chart
            try:
                self._add_agreement_chart(workbook, data)
            except Exception:
                pass
        
        elif "session_stats" in data:
            # Try to add session stats chart
            try:
                self._add_session_stats_chart(workbook, data)
            except Exception:
                pass
    
    def _add_sentiment_chart(self, workbook: Any, data: Dict[str, Any]) -> None:
        """
        Add sentiment distribution chart to the workbook.
        
        Args:
            workbook: Excel workbook
            data (Dict): Data containing sentiment distribution
        """
        if "batch_results" not in data or "sentiment_counts" not in data["batch_results"]:
            return
        
        # Get sentiment counts
        sentiment_counts = data["batch_results"]["sentiment_counts"]
        
        # Create a new sheet for charts
        if "Charts" not in workbook.sheetnames:
            workbook.add_worksheet("Charts")
        
        chart_sheet = workbook.get_worksheet_by_name("Charts")
        
        # Write data for chart
        chart_sheet.write_column('A1', ["Sentiment"])
        chart_sheet.write_column('B1', ["Count"])
        
        row = 1
        for sentiment, count in sentiment_counts.items():
            chart_sheet.write(row, 0, sentiment.capitalize())
            chart_sheet.write(row, 1, count)
            row += 1
        
        # Create chart
        chart = workbook.add_chart({'type': 'pie'})
        
        # Configure the series
        chart.add_series({
            'name': 'Sentiment Distribution',
            'categories': ['Charts', 1, 0, row - 1, 0],
            'values': ['Charts', 1, 1, row - 1, 1],
            'data_labels': {'percentage': True},
        })
        
        # Add title
        chart.set_title({'name': 'Sentiment Distribution'})
        
        # Set chart size
        chart.set_size({'width': 500, 'height': 300})
        
        # Insert the chart into the worksheet
        chart_sheet.insert_chart('D1', chart)
    
    def _add_emotion_chart(self, workbook: Any, data: Dict[str, Any]) -> None:
        """
        Add emotion distribution chart to the workbook.
        
        Args:
            workbook: Excel workbook
            data (Dict): Data containing emotion distribution
        """
        if ("batch_results" not in data or 
            "emotion_distribution" not in data["batch_results"] or 
            not data["batch_results"]["emotion_distribution"]):
            return
        
        # Get emotion distribution
        emotion_dist = data["batch_results"]["emotion_distribution"]
        
        # Create a new sheet for charts if it doesn't exist
        if "Charts" not in workbook.sheetnames:
            workbook.add_worksheet("Charts")
        
        chart_sheet = workbook.get_worksheet_by_name("Charts")
        
        # Write data for chart
        chart_sheet.write_column('E1', ["Emotion"])
        chart_sheet.write_column('F1', ["Count"])
        
        row = 1
        for emotion, count in emotion_dist.items():
            chart_sheet.write(row, 4, emotion.capitalize())
            chart_sheet.write(row, 5, count)
            row += 1
        
        # Create chart
        chart = workbook.add_chart({'type': 'column'})
        
        # Configure the series
        chart.add_series({
            'name': 'Emotion Distribution',
            'categories': ['Charts', 1, 4, row - 1, 4],
            'values': ['Charts', 1, 5, row - 1, 5],
        })
        
        # Add title
        chart.set_title({'name': 'Emotion Distribution'})
        
        # Set chart style
        chart.set_style(11)  # Use a predefined style
        
        # Set chart size
        chart.set_size({'width': 500, 'height': 300})
        
        # Insert the chart into the worksheet
        chart_sheet.insert_chart('D15', chart)
    
    def _add_agreement_chart(self, workbook: Any, data: Dict[str, Any]) -> None:
        """
        Add agreement chart to the workbook.
        
        Args:
            workbook: Excel workbook
            data (Dict): Data containing agreement metrics
        """
        if "agreement_stats" not in data or not data["agreement_stats"]:
            return
        
        # Get agreement stats
        agreement_stats = data["agreement_stats"]
        
        # Create a new sheet for charts
        if "Charts" not in workbook.sheetnames:
            workbook.add_worksheet("Charts")
        
        chart_sheet = workbook.get_worksheet_by_name("Charts")
        
        # Write data for chart
        chart_sheet.write_column('A1', ["Metric"])
        chart_sheet.write_column('B1', ["Value"])
        
        metrics = [
            ("Sentiment Agreement", agreement_stats.get("sentiment_agreement", 0)),
            ("Emotion Agreement", agreement_stats.get("emotion_agreement", 0)),
            ("Full Agreement", agreement_stats.get("full_agreement", 0))
        ]
        
        for i, (metric, value) in enumerate(metrics, 1):
            chart_sheet.write(i, 0, metric)
            chart_sheet.write(i, 1, value)
        
        # Create chart
        chart = workbook.add_chart({'type': 'column'})
        
        # Configure the series
        chart.add_series({
            'name': 'Agreement Levels',
            'categories': ['Charts', 1, 0, 3, 0],
            'values': ['Charts', 1, 1, 3, 1],
        })
        
        # Add title
        chart.set_title({'name': 'Model Agreement'})
        
        # Set y-axis to percentage format
        chart.set_y_axis({'num_format': '0.0%'})
        
        # Set chart size
        chart.set_size({'width': 500, 'height': 300})
        
        # Insert the chart into the worksheet
        chart_sheet.insert_chart('D1', chart)
    
    def _add_session_stats_chart(self, workbook: Any, data: Dict[str, Any]) -> None:
        """
        Add session statistics chart to the workbook.
        
        Args:
            workbook: Excel workbook
            data (Dict): Data containing session statistics
        """
        if "session_stats" not in data:
            return
        
        # Get session stats
        stats = data["session_stats"]
        
        # Create a new sheet for charts
        if "Charts" not in workbook.sheetnames:
            workbook.add_worksheet("Charts")
        
        chart_sheet = workbook.get_worksheet_by_name("Charts")
        
        # Write data for chart
        chart_sheet.write_column('A1', ["Request Type"])
        chart_sheet.write_column('B1', ["Count"])
        
        chart_sheet.write(1, 0, "Standard Requests")
        chart_sheet.write(1, 1, stats.get("standard_requests", 0))
        
        chart_sheet.write(2, 0, "Groq Requests")
        chart_sheet.write(2, 1, stats.get("groq_requests", 0))
        
        # Create chart
        chart = workbook.add_chart({'type': 'pie'})
        
        # Configure the series
        chart.add_series({
            'name': 'Request Distribution',
            'categories': ['Charts', 1, 0, 2, 0],
            'values': ['Charts', 1, 1, 2, 1],
            'data_labels': {'percentage': True},
        })
        
        # Add title
        chart.set_title({'name': 'Request Distribution'})
        
        # Set chart size
        chart.set_size({'width': 500, 'height': 300})
        
        # Insert the chart into the worksheet
        chart_sheet.insert_chart('D1', chart)
    
    def format_session_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format session statistics for export.
        
        Args:
            stats (Dict): Session statistics to format
            
        Returns:
            Dict: Formatted statistics ready for export
        """
        formatted_stats = {
            "session_stats": stats,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Add duration in readable format if available
        if "session_duration" in stats and stats["session_duration"]:
            duration_seconds = stats["session_duration"].total_seconds()
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            formatted_stats["readable_duration"] = f"{minutes}m {seconds}s"
        
        # Calculate averages and ratios
        total_requests = stats.get("standard_requests", 0) + stats.get("groq_requests", 0)
        
        if total_requests > 0:
            formatted_stats["standard_ratio"] = stats.get("standard_requests", 0) / total_requests
            formatted_stats["groq_ratio"] = stats.get("groq_requests", 0) / total_requests
        
        return formatted_stats
    
    def format_timestamp(self) -> str:
        """
        Format current timestamp for filenames.
        
        Returns:
            str: Formatted timestamp string
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
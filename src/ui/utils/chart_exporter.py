import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fpdf import FPDF
import pytz
from PIL import Image
import altair as alt
import os
from typing import Dict, List, Optional, Union, Tuple


class ChartExporter:
    """
    Handles exporting of charts and analytics data to various formats
    with customization options.
    """
    
    def __init__(self):
        """Initialize the chart exporter."""
        self.temp_dir = "tmp"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def export_chart(self, chart, format="png", width=800, height=600, 
                     scale_factor=2, filename=None):
        """
        Exports an Altair chart to the specified format.
        
        Args:
            chart: Altair chart to export
            format: Export format ("png", "svg", or "pdf")
            width: Width of the exported chart in pixels
            height: Height of the exported chart in pixels
            scale_factor: Resolution scale factor
            filename: Optional filename for the exported chart
            
        Returns:
            BytesIO object containing the exported chart or path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_export_{timestamp}"
            
        # Make sure the filename has no extension yet
        if "." in filename:
            filename = filename.split(".")[0]
        
        # For SVG/PNG formats, use Altair's built-in save method
        if format.lower() in ["png", "svg"]:
            # Set chart properties
            chart = chart.properties(width=width, height=height)
            
            # Save to a temporary file
            temp_path = os.path.join(self.temp_dir, f"{filename}.{format}")
            
            # Save with appropriate mime type and scale factor
            chart.save(temp_path, scale_factor=scale_factor)
            
            # Read the file into BytesIO
            with open(temp_path, "rb") as f:
                output = io.BytesIO(f.read())
            
            # Clean up
            try:
                os.remove(temp_path)
            except:
                pass
                
            return output
            
        # For PDF, we need to create a PDF document and embed the chart
        elif format.lower() == "pdf":
            # First export as PNG
            png_data = self.export_chart(chart, format="png", width=width, 
                                         height=height, scale_factor=scale_factor)
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Add title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Analytics Chart Export", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
            
            # Save PNG to temp file
            temp_png = os.path.join(self.temp_dir, f"{filename}_temp.png")
            with open(temp_png, "wb") as f:
                f.write(png_data.getvalue())
            
            # Add the chart
            pdf.image(temp_png, x=10, y=pdf.get_y(), w=190)
            
            # Clean up
            try:
                os.remove(temp_png)
            except:
                pass
            
            # Save to BytesIO
            output = io.BytesIO()
            output.write(pdf.output(dest='S').encode('latin1'))
            output.seek(0)
            
            return output
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_dashboard(self, charts, title="Analytics Dashboard", format="pdf",
                        subtitle=None, notes=None, include_timestamp=True):
        """
        Creates a multi-chart dashboard export.
        
        Args:
            charts: List of (chart, title) tuples to include
            title: Main dashboard title
            format: Export format ("pdf" or "png")
            subtitle: Optional subtitle
            notes: Optional notes to include at the bottom
            include_timestamp: Whether to include generation timestamp
            
        Returns:
            BytesIO object containing the exported dashboard
        """
        if format.lower() == "pdf":
            return self._export_dashboard_pdf(charts, title, subtitle, notes, include_timestamp)
        elif format.lower() == "png":
            return self._export_dashboard_png(charts, title, subtitle, notes, include_timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_dashboard_pdf(self, charts, title, subtitle, notes, include_timestamp):
        """Creates a PDF dashboard with multiple charts."""
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        
        # Add subtitle if provided
        if subtitle:
            pdf.set_font("Arial", 'I', 12)
            pdf.cell(0, 10, subtitle, ln=True, align='C')
        
        # Add timestamp if requested
        if include_timestamp:
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        
        pdf.ln(5)
        
        # Add each chart
        for i, (chart, chart_title) in enumerate(charts):
            # Add chart title
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, chart_title, ln=True)
            
            # Export chart as PNG
            chart_png = self.export_chart(chart, format="png", scale_factor=2)
            
            # Save to temp file
            temp_png = os.path.join(self.temp_dir, f"chart_{i}.png")
            with open(temp_png, "wb") as f:
                f.write(chart_png.getvalue())
            
            # Add the chart
            pdf.image(temp_png, x=10, y=pdf.get_y(), w=190)
            
            # Add spacing
            pdf.ln(5)
            
            # Clean up
            try:
                os.remove(temp_png)
            except:
                pass
            
            # Add a new page if not the last chart
            if i < len(charts) - 1:
                pdf.add_page()
        
        # Add notes if provided
        if notes:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Notes", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 10, notes)
        
        # Save to BytesIO
        output = io.BytesIO()
        output.write(pdf.output(dest='S').encode('latin1'))
        output.seek(0)
        
        return output
    
    def _export_dashboard_png(self, charts, title, subtitle, notes, include_timestamp):
        """Creates a PNG dashboard with multiple charts stacked vertically."""
        # Calculate total height needed
        header_height = 100  # Space for title, subtitle, timestamp
        footer_height = 150 if notes else 50  # Space for notes or just padding
        chart_height = 500  # Height per chart
        spacing = 50  # Spacing between charts
        
        total_height = header_height + (chart_height * len(charts)) + (spacing * (len(charts) - 1)) + footer_height
        width = 1000  # Fixed width
        
        # Create a large figure
        fig, axes = plt.subplots(len(charts) + 2, 1, figsize=(width/100, total_height/100), dpi=100)
        fig.subplots_adjust(hspace=0.3)
        
        # If there's only one chart, make axes a list
        if len(charts) == 1:
            axes = [axes]
            
        # Add title and subtitle
        axes[0].axis('off')
        axes[0].set_title(title, fontsize=20, fontweight='bold', pad=20)
        
        if subtitle:
            axes[0].text(0.5, 0.5, subtitle, fontsize=14, 
                         horizontalalignment='center', verticalalignment='center')
            
        if include_timestamp:
            timestamp = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            axes[0].text(0., 0., timestamp, fontsize=10, 
                        horizontalalignment='left', verticalalignment='bottom')
        
        # Export each chart and add to the figure
        for i, (chart, chart_title) in enumerate(charts):
            # Export chart as PNG
            chart_png = self.export_chart(chart, format="png", scale_factor=1.5)
            
            # Save to temp file
            temp_png = os.path.join(self.temp_dir, f"chart_{i}.png")
            with open(temp_png, "wb") as f:
                f.write(chart_png.getvalue())
            
            # Load image
            img = plt.imread(temp_png)
            
            # Display in the appropriate axis
            axes[i+1].imshow(img)
            axes[i+1].set_title(chart_title, fontsize=16)
            axes[i+1].axis('off')
            
            # Clean up
            try:
                os.remove(temp_png)
            except:
                pass
        
        # Add notes if provided
        axes[-1].axis('off')
        if notes:
            axes[-1].text(0.5, 0.5, notes, fontsize=12, 
                         horizontalalignment='center', verticalalignment='center',
                         wrap=True)
        
        # Save to BytesIO
        output = io.BytesIO()
        plt.tight_layout()
        fig.savefig(output, format='png', bbox_inches='tight')
        plt.close(fig)
        output.seek(0)
        
        return output


class ChartFilterer:
    """
    Applies filtering and date range selection to chart data.
    """
    
    def __init__(self):
        """Initialize the chart filterer."""
        pass
    
    def filter_data(self, data, 
                   from_date=None, to_date=None,
                   sentiment_filter=None, 
                   emotion_filter=None,
                   min_confidence=None):
        """
        Filters DataFrame based on specified criteria.
        
        Args:
            data: DataFrame to filter
            from_date: Start date/time for filtering
            to_date: End date/time for filtering
            sentiment_filter: List of sentiment labels to include
            emotion_filter: List of emotion labels to include
            min_confidence: Minimum confidence score to include
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        
        # Apply date filtering if date index is available
        if isinstance(filtered_data.index, pd.DatetimeIndex):
            if from_date:
                filtered_data = filtered_data[filtered_data.index >= from_date]
            if to_date:
                filtered_data = filtered_data[filtered_data.index <= to_date]
        
        # Apply sentiment label filtering
        if sentiment_filter and 'sentiment_label' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['sentiment_label'].isin(sentiment_filter)]
            
        # Apply emotion label filtering
        if emotion_filter and 'emotion_label' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['emotion_label'].isin(emotion_filter)]
        
        # Apply confidence filtering
        if min_confidence is not None:
            # Apply to sentiment confidence
            if 'sentiment_confidence' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['sentiment_confidence'] >= min_confidence]
            elif 'sentiment_score' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['sentiment_score'].abs() >= min_confidence]
                
            # Apply to emotion confidence if available
            if 'emotion_score' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['emotion_score'] >= min_confidence]
        
        return filtered_data
    
    def get_date_range_options(self, data):
        """
        Generates common date range options for the given data.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            Dict of date range options
        """
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) == 0:
            return {}
            
        # Get min and max dates
        min_date = data.index.min()
        max_date = data.index.max()
        
        # Calculate common ranges
        now = pd.Timestamp.now(tz=min_date.tz)
        
        ranges = {
            "All Data": (min_date, max_date),
            "Last Hour": (now - timedelta(hours=1), now),
            "Last 24 Hours": (now - timedelta(days=1), now),
            "Last 7 Days": (now - timedelta(days=7), now),
            "Last 30 Days": (now - timedelta(days=30), now),
        }
        
        # Filter to ranges that make sense for the data
        data_range = max_date - min_date
        
        # Only return ranges that include some data
        filtered_ranges = {}
        for name, (start, end) in ranges.items():
            # Always include "All Data"
            if name == "All Data":
                filtered_ranges[name] = (start, end)
                continue
                
            # For others, check if there's data in this range
            if min_date <= end and max_date >= start:
                filtered_ranges[name] = (start, end)
        
        return filtered_ranges 
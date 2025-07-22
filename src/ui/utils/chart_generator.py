"""
Chart generation utilities for PDF reports.
Provides static chart generation in image formats.
"""

import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Union, Optional, Any
from io import BytesIO


class ChartGenerator:
    """
    Generates static charts for reports.
    """
    
    def __init__(self, output_format="png"):
        """
        Initialize the chart generator.
        
        Args:
            output_format (str): Output format for charts ('png' or 'svg')
        """
        self.output_format = output_format
        self.set_chart_style()
    
    def create_bar_chart(self, data: Dict[str, float], title: str, width: int = 600, 
                          height: int = 400) -> BytesIO:
        """
        Create a bar chart as an image.
        
        Args:
            data (Dict): Data for the bar chart
            title (str): Chart title
            width (int): Chart width in pixels
            height (int): Chart height in pixels
            
        Returns:
            BytesIO: Chart image as BytesIO
        """
        # Set figure size (convert pixels to inches)
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Get color palette
        colors = self.get_color_palette(len(data))
        
        # Create chart
        if data:
            # Sort data for better visualization
            sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
            
            plt.bar(sorted_data.keys(), sorted_data.values(), color=colors)
            
            # Add values on top of bars
            for i, (key, value) in enumerate(sorted_data.items()):
                if isinstance(value, float):
                    # Format as percentage if value is between 0 and 1
                    if 0 <= value <= 1:
                        plt.text(i, value + 0.01, f"{value:.1%}", ha='center')
                    else:
                        plt.text(i, value + (max(sorted_data.values()) * 0.01), 
                                 f"{value:.1f}", ha='center')
            
            # Add labels and title
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Categories', fontsize=12)
            plt.ylabel('Values', fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            plt.xticks(rotation=45, ha='right')
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
        
        # Convert to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format=self.output_format, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf
    
    def create_pie_chart(self, data: Dict[str, float], title: str, width: int = 600, 
                          height: int = 400) -> BytesIO:
        """
        Create a pie chart as an image.
        
        Args:
            data (Dict): Data for the pie chart
            title (str): Chart title
            width (int): Chart width in pixels
            height (int): Chart height in pixels
            
        Returns:
            BytesIO: Chart image as BytesIO
        """
        # Set figure size
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Get color palette
        colors = self.get_color_palette(len(data))
        
        # Create chart
        if data and sum(data.values()) > 0:
            # Sort data for better visualization
            sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
            
            # Format percentage labels
            def autopct_format(values):
                def my_format(pct):
                    total = sum(values)
                    val = int(round(pct * total / 100.0))
                    return f'{pct:.1f}%\n({val})'
                return my_format
            
            # Create pie chart
            plt.pie(sorted_data.values(), labels=sorted_data.keys(), 
                    autopct=autopct_format(sorted_data.values()), 
                    colors=colors, shadow=False, startangle=90)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            plt.axis('equal')
            
            # Add title
            plt.title(title, fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
        
        # Convert to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format=self.output_format, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf
    
    def create_distribution_chart(self, data: Dict[str, float], title: str, width: int = 600, 
                                   height: int = 400) -> BytesIO:
        """
        Create a distribution chart as an image.
        
        Args:
            data (Dict): Data for the distribution chart
            title (str): Chart title
            width (int): Chart width in pixels
            height (int): Chart height in pixels
            
        Returns:
            BytesIO: Chart image as BytesIO
        """
        # Set figure size
        plt.figure(figsize=(width/100, height/100), dpi=100)
        
        # Create chart
        if data and len(data) > 0:
            # Convert to arrays
            values = np.array(list(data.values()))
            
            # Create histogram or KDE based on number of data points
            if len(values) >= 10:
                sns.histplot(values, kde=True)
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
            else:
                # For small datasets, use a rug plot with markers
                plt.plot(values, np.zeros_like(values), 'o', markersize=10)
                plt.xlabel('Value', fontsize=12)
                plt.yticks([])  # Hide y-axis
                
                # Add a bit of jitter to y for better visualization
                for i, val in enumerate(values):
                    plt.text(val, 0.1, f"{val:.2f}", ha='center')
            
            # Add title
            plt.title(title, fontsize=14, fontweight='bold')
            
            # Add grid for better readability
            plt.grid(linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
        
        # Convert to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format=self.output_format, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf
    
    def set_chart_style(self):
        """
        Set the default chart style.
        """
        # Use seaborn styles for better looking charts
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        
        # Set color palette
        sns.set_palette("colorblind")
        
        # Additional matplotlib customizations
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.5
    
    def get_color_palette(self, num_colors: int) -> List[str]:
        """
        Get a color palette for charts.
        
        Args:
            num_colors (int): Number of colors needed
            
        Returns:
            List[str]: List of color hex codes
        """
        # Use seaborn colorblind palette for accessibility
        colors = sns.color_palette("colorblind", num_colors)
        
        # Convert to hex codes using matplotlib's to_hex method
        return [plt.matplotlib.colors.to_hex(color) for color in colors] 
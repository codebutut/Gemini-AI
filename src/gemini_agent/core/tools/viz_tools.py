import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import os
from gemini_agent.utils.logger import get_logger
from gemini_agent.core.tools import tool

logger = get_logger(__name__)

@tool
def generate_chart(data: List[Dict[str, Any]], chart_type: str, title: str, output_file: str) -> str:
    """
    Generates a chart from a list of dictionaries and saves it to a file.
    
    Args:
        data: List of dictionaries containing the data.
        chart_type: Type of chart ('line', 'bar', 'scatter', 'pie', 'hist').
        title: Title of the chart.
        output_file: Path to save the generated image.
    """
    try:
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'line':
            sns.lineplot(data=df)
        elif chart_type == 'bar':
            cols = df.columns
            sns.barplot(x=cols[0], y=cols[1], data=df)
        elif chart_type == 'scatter':
            cols = df.columns
            sns.scatterplot(x=cols[0], y=cols[1], data=df)
        elif chart_type == 'pie':
            cols = df.columns
            plt.pie(df[cols[1]], labels=df[cols[0]], autopct='%1.1f%%')
        elif chart_type == 'hist':
            sns.histplot(data=df)
        else:
            return f"Error: Unsupported chart type '{chart_type}'"

        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return f"Chart saved to {output_file}"
    except Exception as e:
        logger.error(f"Failed to generate chart: {e}")
        return f"Error: {e}"

@tool
def plot_data(csv_path: str, x_col: str, y_col: str, chart_type: str, output_file: str, title: Optional[str] = None) -> str:
    """
    Plots data directly from a CSV file.
    """
    try:
        if not os.path.exists(csv_path):
            return f"Error: File {csv_path} not found."
            
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'line':
            sns.lineplot(x=x_col, y=y_col, data=df)
        elif chart_type == 'bar':
            sns.barplot(x=x_col, y=y_col, data=df)
        elif chart_type == 'scatter':
            sns.scatterplot(x=x_col, y=y_col, data=df)
        else:
            return f"Error: Unsupported chart type '{chart_type}' for CSV plotting."

        plt.title(title or f"{chart_type.capitalize()} plot of {y_col} vs {x_col}")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return f"Plot saved to {output_file}"
    except Exception as e:
        logger.error(f"Failed to plot data from CSV: {e}")
        return f"Error: {e}"

#!/usr/bin/env python3
"""
Script to run the Streamlit application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    # Get the path to the streamlit app
    app_path = os.path.join("src", "ui", "streamlit_app.py")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main() 
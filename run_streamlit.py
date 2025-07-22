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
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Virtual environment not detected")
        print("💡 Tip: Activate your virtual environment with: source venv/bin/activate")
    
    # Check if streamlit is available
    try:
        import streamlit
        print(f"✅ Streamlit version {streamlit.__version__} is available")
    except ImportError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        return
    
    print("🚀 Starting Streamlit application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("💡 Tip: Make sure all dependencies are installed with: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 
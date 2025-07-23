#!/bin/bash

# Script to start the Streamlit application

echo "🚀 Starting Sentiment Analysis Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the application
echo "🚀 Starting Streamlit application..."
echo "📱 Open your browser to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

python run_streamlit.py 
#!/bin/bash

# Smart Surveillance System - Ultra Enhanced Run Script
# This script runs the system with all enhanced features enabled

echo "🚀 Starting Smart Surveillance System - Ultra Enhanced Version"
echo "================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if enhanced dependencies are installed
echo "Checking enhanced dependencies..."
python -c "import easyocr" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Enhanced dependencies not found. Installing..."
    pip install -r requirements_enhanced.txt
fi

echo ""
echo "Starting ultra-enhanced surveillance system..."
echo "Dashboard will be available at: http://localhost:8082"
echo ""
echo "Press Ctrl+C to stop the system"
echo "================================================================"
echo ""

# Run the ultra-enhanced system
python main_ultra_enhanced.py

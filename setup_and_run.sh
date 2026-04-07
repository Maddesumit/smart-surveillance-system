#!/bin/bash

# Smart Surveillance System - Setup and Run Script
# This script sets up the environment and runs the surveillance system

echo "🚀 Smart Surveillance System - Setup Script"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Check Python version
echo "📌 Python version:"
python --version
echo ""

# Install/Update dependencies
echo "📦 Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt

# Install optional dependencies for advanced features
echo ""
echo "🔧 Installing optional advanced features..."
echo "Installing face_recognition (for facial recognition)..."
pip install face_recognition || echo "⚠️  face_recognition installation failed (optional)"

echo "Installing mediapipe (for behavior analysis)..."
pip install mediapipe || echo "⚠️  mediapipe installation failed (optional)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "=============================================="
echo "🎯 How to run the system:"
echo "=============================================="
echo ""
echo "Option 1 - Basic System (Recommended for first run):"
echo "  python main.py"
echo ""
echo "Option 2 - Enhanced System (All features):"
echo "  python main_enhanced.py"
echo ""
echo "Option 3 - Professional System:"
echo "  python main_enhanced_professional.py"
echo ""
echo "Dashboard will be available at: http://localhost:8082"
echo ""
echo "Press Ctrl+C to stop the system"
echo "=============================================="

#!/bin/bash

# Smart Surveillance System - Quick Run Script
# This script activates the virtual environment and runs the system

echo "🎥 Starting Smart Surveillance System..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Check which version to run
if [ "$1" == "basic" ]; then
    echo "Running BASIC system (core features only)..."
    python main.py
elif [ "$1" == "professional" ]; then
    echo "Running PROFESSIONAL system (enhanced UI)..."
    python main_enhanced_professional.py
else
    echo "Running ENHANCED system (all features)..."
    python main_enhanced.py
fi

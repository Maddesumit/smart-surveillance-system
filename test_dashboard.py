#!/usr/bin/env python3
"""
Simple dashboard server for testing behavior analysis
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask
from dashboard import create_app

if __name__ == '__main__':
    try:
        print("Starting simple dashboard server...")
        app = create_app()
        print("Dashboard app created successfully")
        
        # Run on localhost for camera access compatibility
        print("Starting server on http://127.0.0.1:8081")
        app.run(host='127.0.0.1', port=8081, debug=True)
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()

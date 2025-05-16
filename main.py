#!/usr/bin/env python3
"""
Smart Surveillance System - Main Application

This is the main entry point for the Smart Surveillance System.
It initializes all components and starts the main processing loop.
"""

import os
import sys
import time
import logging
import signal
import threading
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import components from our project
from video_processing.video_stream import VideoStream
from object_detection.detector import ObjectDetector
from object_detection.tracker import ObjectTracker
from anomaly_detection.analyzer import AnomalyDetector
from alert_system.notifier import AlertSystem

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'surveillance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('smart_surveillance')

# Global flag to control the main loop
running = True

# Dashboard thread
dashboard_thread = None

def signal_handler(sig, frame):
    """
    Handle termination signals to gracefully shut down the system.
    """
    global running
    logger.info("Shutdown signal received. Stopping system...")
    running = False

def start_dashboard():
    """
    Start the dashboard in a separate thread.
    """
    from dashboard.app import app
    app.run(host='0.0.0.0', port=8082, debug=False, use_reloader=False)

def main():
    """
    Main function to initialize and run the surveillance system.
    """
    global dashboard_thread, running
    
    logger.info("Starting Smart Surveillance System")
    
    try:
        # Initialize components
        # Change this line in the main() function
        # In the main() function, replace the video stream initialization with:
        logger.info("Initializing video stream...")
        video_stream = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                video_stream = VideoStream(source=0)  # Use webcam
                # Test if we can read a frame
                success, _ = video_stream.read_frame()
                if success:
                    logger.info(f"Successfully connected to video source on attempt {attempt+1}")
                    break
                else:
                    logger.warning(f"Could not read frame on attempt {attempt+1}, retrying...")
                    if video_stream:
                        video_stream.release()
                    time.sleep(1)  # Wait before retrying
            except Exception as e:
                logger.error(f"Error initializing video stream on attempt {attempt+1}: {str(e)}")
                time.sleep(1)  # Wait before retrying
        
        if video_stream is None or not hasattr(video_stream, 'cap') or not video_stream.cap.isOpened():
            logger.error("Failed to initialize video stream after multiple attempts")
            # You could exit here or continue with limited functionality
            # sys.exit(1)  # Uncomment to exit if video is required
        
        logger.info("Initializing object detector and tracker...")
        detector = ObjectDetector()
        tracker = ObjectTracker()
        
        logger.info("Initializing anomaly detector...")
        analyzer = AnomalyDetector()
        
        # Add a restricted area (example: bottom half of the frame)
        frame_height, frame_width = 480, 640  # Default size, adjust if needed
        restricted_area = [
            (50, frame_height//2),
            (frame_width-50, frame_height//2),
            (frame_width-50, frame_height-50),
            (50, frame_height-50)
        ]
        analyzer.add_restricted_area(restricted_area)
        
        logger.info("Initializing alert system...")
        notifier = AlertSystem()
        
        # Start dashboard in a separate thread
        logger.info("Starting dashboard...")
        dashboard_thread = threading.Thread(target=start_dashboard)
        dashboard_thread.daemon = True  # This ensures the thread will exit when the main program exits
        dashboard_thread.start()
        
        logger.info("System initialized successfully")
        
        # Main processing loop
        while running:
            # Read a frame from the video stream
            success, frame = video_stream.read_frame()
            
            if not success:
                logger.warning("Failed to read frame from video stream")
                time.sleep(0.1)  # Short delay before trying again
                continue
            
            # Detect objects in the frame
            detections = detector.detect(frame)
            
            # Track objects across frames
            tracked_objects = tracker.update(detections)
            
            # Process frame for anomalies
            anomalies = analyzer.detect_anomalies(frame, tracked_objects)
            
            # Generate alerts for detected anomalies
            for anomaly in anomalies:
                notifier.generate_alert(anomaly)
            
            # Short delay to control processing rate
            time.sleep(0.01)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        if 'video_stream' in locals():
            video_stream.release()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Start the main function
    main()
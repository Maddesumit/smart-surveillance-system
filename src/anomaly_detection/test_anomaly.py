"""
Test script for anomaly detection.

This script demonstrates the use of the AnomalyDetector class
with object detection and tracking.
"""

import os
import sys
import logging
import cv2
import time
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.video_processing.video_stream import VideoStream
from src.object_detection.detector import ObjectDetector
from src.object_detection.tracker import ObjectTracker
from src.anomaly_detection.analyzer import AnomalyDetector
from src.alert_system.notifier import AlertSystem

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test anomaly detection.
    """
    # Initialize video stream (use 0 for webcam or provide a video file path)
    video_stream = VideoStream(source=0, width=640, height=480, fps_target=30)
    
    # If webcam fails, try using a video file instead
    if not video_stream.connect():
        logger.warning("Webcam access failed, trying to use a sample video file instead")
        
        # Try to use a sample video file
        sample_video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                        'data', 'sample_video.mp4')
        
        if os.path.exists(sample_video_path):
            video_stream = VideoStream(source=sample_video_path, width=640, height=480, fps_target=30)
            if not video_stream.connect():
                logger.error("Could not open sample video either. Exiting.")
                return
        else:
            logger.error(f"Sample video not found at {sample_video_path}. Exiting.")
            return
    
    # Initialize object detector
    try:
        detector = ObjectDetector(model_size='s', confidence_threshold=0.25)
        logger.info("Object detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize object detector: {str(e)}")
        video_stream.release()
        return
    
    # Initialize object tracker
    tracker = ObjectTracker(max_disappeared=30, max_distance=50)
    
    # Initialize anomaly detector
    anomaly_detector = AnomalyDetector(stationary_threshold=30, confidence_threshold=0.6)
    
    # Define a restricted area (example: center rectangle)
    frame_width, frame_height = 640, 480  # Adjust based on your video size
    restricted_area = [
        (int(frame_width * 0.3), int(frame_height * 0.3)),  # Top-left
        (int(frame_width * 0.7), int(frame_height * 0.3)),  # Top-right
        (int(frame_width * 0.7), int(frame_height * 0.7)),  # Bottom-right
        (int(frame_width * 0.3), int(frame_height * 0.7))   # Bottom-left
    ]
    anomaly_detector.add_restricted_area(restricted_area)
    
    # Create window for display
    cv2.namedWindow('Anomaly Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Anomaly Detection', 800, 600)
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    logger.info("Starting anomaly detection loop")
    
    try:
        while True:
            # Read frame from video stream
            success, frame = video_stream.read_frame()
            
            if not success or frame is None:
                logger.warning("Received None frame, stopping")
                break
            
            # Calculate FPS
            fps_frame_count += 1
            if (time.time() - fps_start_time) > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Detect objects in the frame
            detections = detector.detect(frame)
            
            # Update object tracking
            tracked_objects = tracker.update(detections)
            
            # Detect anomalies
            anomalies = anomaly_detector.detect_anomalies(frame, tracked_objects)
            
            # Draw detections, tracks, and anomalies on the frame
            frame_with_detections = detector.draw_detections(frame, detections)
            frame_with_tracks = tracker.draw_tracks(frame_with_detections)
            result_frame = anomaly_detector.draw_anomalies(frame_with_tracks, anomalies)
            
            # Display FPS and anomaly count on the frame
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Anomalies: {len(anomalies)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Anomaly Detection', result_frame)
            
            # Check for key press (q to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User requested quit")
                break
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping")
    except Exception as e:
        logger.error(f"Error in detection loop: {str(e)}")
    finally:
        # Clean up
        video_stream.release()
        cv2.destroyAllWindows()
        logger.info("Resources released, exiting")

if __name__ == "__main__":
    main()
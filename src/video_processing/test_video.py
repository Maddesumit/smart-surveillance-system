#!/usr/bin/env python3
"""
Video Processing Test Script

This script demonstrates the video processing capabilities
by capturing video from a webcam and applying the processing pipeline.
"""

import cv2
import numpy as np
import time
import logging
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from src.video_processing.video_stream import VideoStream
from src.video_processing.processing_pipeline import ProcessingPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_video')

def main():
    """
    Main function to test video processing.
    """
    # Initialize video stream (use webcam by default)
    video_stream = VideoStream(source=0, width=640, height=480, fps_target=30)
    
    # Initialize processing pipeline
    pipeline = ProcessingPipeline(buffer_size=30)
    
    # Define a sample region of interest (a rectangle in the center)
    # Format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    center_roi = np.array([
        [160, 120],  # Top-left
        [480, 120],  # Top-right
        [480, 360],  # Bottom-right
        [160, 360]   # Bottom-left
    ])
    pipeline.add_roi(center_roi)
    
    # Create windows for displaying results
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    
    logger.info("Starting video processing test. Press 'q' to quit.")
    
    try:
        while True:
            # Get a frame from the video stream
            ret, frame = video_stream.get_frame()
            if not ret:
                logger.error("Failed to get frame from video stream")
                break
            
            # Process the frame
            results = pipeline.process_frame(frame)
            
            # Display results
            cv2.imshow('Original', results['original'])
            cv2.imshow('Foreground Mask', results['fg_mask'])
            cv2.imshow('ROI', results['roi_frame'])
            cv2.imshow('Final Result', results['final_frame'])
            
            # Display FPS and processing time
            fps = video_stream.get_fps()
            proc_time = results['processing_time'] * 1000  # Convert to ms
            logger.info(f"FPS: {fps:.1f}, Processing time: {proc_time:.1f} ms")
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User requested exit")
                break
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Clean up resources
        video_stream.release()
        cv2.destroyAllWindows()
        logger.info("Resources released")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Video Stream Module

This module handles video capture from various sources (webcam, IP camera, video file)
and provides methods for frame extraction and basic preprocessing.
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional, Union
import os

# Set up logging
logger = logging.getLogger('smart_surveillance.video_stream')

class VideoStream:
    """
    A class to handle video streaming from different sources.
    
    This class provides methods to capture video from webcams, IP cameras,
    or video files, and perform basic preprocessing on the frames.
    """
    
    def __init__(self, source: Union[int, str] = 0, 
                 width: int = 640, 
                 height: int = 480,
                 fps_target: int = 30):
        """
        Initialize the VideoStream object.
        
        Args:
            source: Camera index (0 for webcam) or path/URL to video source
            width: Target width for resizing frames
            height: Target height for resizing frames
            fps_target: Target frames per second
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.cap = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        
        # Connect to the video source
        self.connect()
    
    def connect(self) -> bool:
        """
        Connect to the video source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Release any existing capture
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set properties if it's a camera (not a video file)
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            
            # Get actual properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Connected to video source: {self.source}")
            logger.info(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to video source: {str(e)}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple containing:
                bool: True if frame was successfully read
                np.ndarray or None: The frame if successful, None otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video source is not open")
            return False, None
        
        # Calculate time since last frame for FPS control
        current_time = time.time()
        time_delta = current_time - self.last_frame_time
        
        # Limit frame rate to target FPS
        if self.fps_target > 0 and time_delta < 1.0/self.fps_target:
            time.sleep(1.0/self.fps_target - time_delta)
        
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            # Generate a test pattern as fallback
            test_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Add some visual elements to the test frame
            cv2.putText(test_frame, "No Camera Feed", (50, self.height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Draw a rectangle
            cv2.rectangle(test_frame, (100, 100), (self.width-100, self.height-100), (0, 255, 0), 2)
            return True, test_frame  # Return the test pattern instead of failing
        
        # Update frame count and timing
        self.frame_count += 1
        current_time = time.time()
        if self.last_frame_time > 0:
            # Calculate actual FPS
            self.fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        return True, frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame (resize, normalize, etc.).
        
        Args:
            frame: The input frame to preprocess
            
        Returns:
            np.ndarray: The preprocessed frame
        """
        if frame is None:
            return None
        
        # Resize frame if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        return frame
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get a preprocessed frame from the video source.
        
        Returns:
            Tuple containing:
                bool: True if frame was successfully read and processed
                np.ndarray or None: The processed frame if successful, None otherwise
        """
        ret, frame = self.read_frame()
        if not ret:
            return False, None
        
        processed_frame = self.preprocess_frame(frame)
        return True, processed_frame
    
    def get_fps(self) -> float:
        """
        Get the current frames per second rate.
        
        Returns:
            float: Current FPS
        """
        return self.fps
    
    def release(self) -> None:
        """
        Release the video capture resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Released video capture resources")


class ThreadedVideoStream(VideoStream):
    """
    Threaded video stream for improved performance.
    
    Uses a dedicated capture thread to continuously read frames
    into a buffer, allowing the main thread to process frames
    without waiting for camera I/O.
    """
    
    def __init__(self, source: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps_target: int = 30,
                 buffer_size: int = 5):
        """
        Initialize the threaded video stream.
        
        Args:
            source: Camera index or video path
            width: Target width
            height: Target height  
            fps_target: Target FPS
            buffer_size: Max frames to buffer
        """
        super().__init__(source, width, height, fps_target)
        
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.buffer_lock = None
        self.capture_thread = None
        self.running = False
        
        # Import threading here to avoid issues
        import threading
        from queue import Queue
        
        self.buffer_lock = threading.Lock()
        self.frame_queue = Queue(maxsize=buffer_size)
        
    def start(self) -> bool:
        """
        Start the threaded capture.
        
        Returns:
            bool: True if started successfully
        """
        import threading
        
        if self.running:
            return True
            
        if not self.connect():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Threaded video stream started")
        return True
    
    def stop(self) -> None:
        """Stop the threaded capture."""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.release()
        logger.info("Threaded video stream stopped")
    
    def _capture_loop(self) -> None:
        """Background capture loop."""
        from queue import Empty
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Resize if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Update FPS
                current_time = time.time()
                if self.last_frame_time > 0:
                    self.fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
                self.last_frame_time = current_time
                self.frame_count += 1
                
                # Add to queue (drop oldest if full)
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
                
                # Maintain target FPS
                target_interval = 1.0 / self.fps_target
                elapsed = time.time() - current_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the buffer (non-blocking).
        
        Returns:
            Tuple of (success, frame)
        """
        from queue import Empty
        
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except Empty:
            # Return cached frame or test pattern
            return super().read_frame()
    
    def get_buffer_size(self) -> int:
        """Get current buffer fill level."""
        return self.frame_queue.qsize()
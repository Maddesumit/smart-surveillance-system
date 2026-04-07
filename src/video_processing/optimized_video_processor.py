#!/usr/bin/env python3
"""
Optimized Video Processor

This module provides optimized video processing with:
1. Frame skipping - process detection every Nth frame
2. Threaded video capture - separate thread for smooth capture
3. GPU/CUDA acceleration - use GPU if available

Author: Smart Surveillance System
"""

import cv2
import numpy as np
import time
import logging
import threading
from queue import Queue, Empty
from typing import Tuple, Optional, Union, Dict, List, Any
from collections import deque

# Set up logging
logger = logging.getLogger('smart_surveillance.optimized_video')


class OptimizedVideoProcessor:
    """
    High-performance video processor with threading, frame skipping, and GPU support.
    """
    
    def __init__(self, 
                 source: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps_target: int = 30,
                 frame_skip: int = 2,
                 buffer_size: int = 5,
                 use_gpu: bool = True):
        """
        Initialize the optimized video processor.
        
        Args:
            source: Camera index or video source path/URL
            width: Target frame width
            height: Target frame height
            fps_target: Target FPS for capture
            frame_skip: Process detection every Nth frame (1 = every frame, 2 = every other, etc.)
            buffer_size: Max frames to buffer for smooth playback
            use_gpu: Whether to attempt GPU acceleration
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps_target = fps_target
        self.frame_skip = max(1, frame_skip)  # Minimum 1
        self.buffer_size = buffer_size
        
        # Video capture
        self.cap = None
        self.running = False
        
        # Threading
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.lock = threading.Lock()
        
        # Frame counting and timing
        self.frame_count = 0
        self.detection_frame_count = 0
        self.last_frame_time = 0
        self.fps_actual = 0
        self.fps_detection = 0
        
        # Cached detection results (for skipped frames)
        self.cached_detections = []
        self.cached_tracked_objects = {}
        self.cached_face_detections = []
        
        # GPU/CUDA support
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.cuda_enabled = False
        self._check_gpu_support()
        
        # Performance metrics
        self.metrics = {
            'capture_fps': 0,
            'display_fps': 0,
            'detection_fps': 0,
            'gpu_enabled': self.gpu_available,
            'frame_skip': self.frame_skip,
            'buffer_fill': 0
        }
        
        logger.info(f"OptimizedVideoProcessor initialized: GPU={self.gpu_available}, frame_skip={frame_skip}")
    
    def _check_gpu_support(self) -> None:
        """Check for GPU/CUDA availability and configure accordingly."""
        if not self.use_gpu:
            logger.info("GPU usage disabled by configuration")
            return
            
        try:
            # Check CUDA availability in OpenCV
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                self.gpu_available = True
                self.cuda_enabled = True
                
                # Get GPU info
                device_info = cv2.cuda.getDevice()
                logger.info(f"CUDA enabled: {cuda_devices} device(s) found, using device {device_info}")
                
                # Set CUDA device
                cv2.cuda.setDevice(0)
                
        except cv2.error as e:
            logger.info(f"OpenCV CUDA not available: {e}")
            self.gpu_available = False
        except Exception as e:
            logger.warning(f"Error checking GPU support: {e}")
            self.gpu_available = False
        
        # Also check for PyTorch CUDA (for YOLO)
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"PyTorch CUDA available: {gpu_name}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error checking PyTorch CUDA: {e}")
    
    def start(self) -> bool:
        """
        Start the video capture and processing threads.
        
        Returns:
            bool: True if started successfully
        """
        if self.running:
            logger.warning("Video processor already running")
            return True
        
        # Initialize video capture
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set capture properties
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                
                # Enable hardware acceleration if available
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video capture started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            return False
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Capture thread started")
        return True
    
    def stop(self) -> None:
        """Stop the video processor and release resources."""
        self.running = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Release video capture
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Video processor stopped")
    
    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture."""
        capture_times = deque(maxlen=30)  # For FPS calculation
        
        while self.running:
            try:
                start_time = time.time()
                
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.01)
                    continue
                
                # Resize if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Try to put in queue (non-blocking)
                try:
                    # If queue is full, remove oldest frame
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put_nowait((frame, time.time()))
                except:
                    pass  # Skip frame if queue issues
                
                # Calculate capture FPS
                capture_times.append(time.time() - start_time)
                if len(capture_times) > 5:
                    avg_time = sum(capture_times) / len(capture_times)
                    self.metrics['capture_fps'] = 1.0 / max(avg_time, 0.001)
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                target_time = 1.0 / self.fps_target
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        logger.info("Capture loop ended")
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Get the next frame from the buffer.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        try:
            frame, timestamp = self.frame_queue.get(timeout=1.0)
            self.frame_count += 1
            self.metrics['buffer_fill'] = self.frame_queue.qsize()
            return True, frame, timestamp
        except Empty:
            return False, None, 0.0
    
    def should_process_detection(self) -> bool:
        """
        Determine if this frame should have detection run on it.
        
        Returns:
            bool: True if detection should be run
        """
        return (self.frame_count % self.frame_skip) == 0
    
    def cache_detection_results(self, 
                                 detections: List[Dict],
                                 tracked_objects: Dict,
                                 face_detections: List[Dict] = None) -> None:
        """
        Cache detection results for use on skipped frames.
        
        Args:
            detections: List of object detections
            tracked_objects: Dict of tracked objects
            face_detections: Optional list of face detections
        """
        with self.lock:
            self.cached_detections = detections.copy() if detections else []
            self.cached_tracked_objects = tracked_objects.copy() if tracked_objects else {}
            if face_detections:
                self.cached_face_detections = face_detections.copy()
            self.detection_frame_count += 1
    
    def get_cached_results(self) -> Tuple[List[Dict], Dict, List[Dict]]:
        """
        Get cached detection results for display on skipped frames.
        
        Returns:
            Tuple of (detections, tracked_objects, face_detections)
        """
        with self.lock:
            return (
                self.cached_detections.copy(),
                self.cached_tracked_objects.copy(),
                self.cached_face_detections.copy()
            )
    
    def update_detection_fps(self, processing_time: float) -> None:
        """Update detection FPS metric."""
        if processing_time > 0:
            self.metrics['detection_fps'] = 1.0 / processing_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        self.metrics['buffer_fill'] = self.frame_queue.qsize()
        return self.metrics.copy()
    
    def preprocess_for_gpu(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for GPU acceleration if available.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame (may be GPU mat if CUDA available)
        """
        if self.cuda_enabled:
            try:
                # Upload to GPU memory
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                return gpu_frame
            except Exception as e:
                logger.debug(f"GPU preprocessing failed, using CPU: {e}")
        
        return frame
    
    def is_running(self) -> bool:
        """Check if the video processor is running."""
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to the video source."""
        self.stop()
        time.sleep(0.5)
        return self.start()
    
    def set_frame_skip(self, skip: int) -> None:
        """Update frame skip value."""
        self.frame_skip = max(1, skip)
        self.metrics['frame_skip'] = self.frame_skip
        logger.info(f"Frame skip updated to: {self.frame_skip}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available."""
        info = {
            'gpu_available': self.gpu_available,
            'cuda_enabled': self.cuda_enabled,
            'device_name': 'N/A'
        }
        
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    info['device_name'] = torch.cuda.get_device_name(0)
                    info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**2  # MB
                    info['memory_cached'] = torch.cuda.memory_reserved(0) / 1024**2  # MB
            except:
                pass
        
        return info


# Singleton instance for easy access
_video_processor: Optional[OptimizedVideoProcessor] = None


def get_optimized_processor(source: Union[int, str] = 0, **kwargs) -> OptimizedVideoProcessor:
    """
    Get or create the singleton video processor.
    
    Args:
        source: Video source
        **kwargs: Additional arguments for OptimizedVideoProcessor
        
    Returns:
        OptimizedVideoProcessor instance
    """
    global _video_processor
    
    if _video_processor is None:
        _video_processor = OptimizedVideoProcessor(source=source, **kwargs)
    
    return _video_processor


def reset_processor() -> None:
    """Reset the singleton video processor."""
    global _video_processor
    
    if _video_processor:
        _video_processor.stop()
        _video_processor = None

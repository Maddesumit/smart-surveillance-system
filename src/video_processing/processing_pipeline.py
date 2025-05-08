#!/usr/bin/env python3
"""
Video Processing Pipeline

This module implements the video processing pipeline for the surveillance system,
including frame buffering, background subtraction, and region of interest selection.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
import time

# Set up logging
logger = logging.getLogger('smart_surveillance.processing_pipeline')

class ProcessingPipeline:
    """
    A pipeline for processing video frames.
    
    This class implements various video processing techniques such as
    frame buffering, background subtraction, and region of interest selection.
    """
    
    def __init__(self, buffer_size: int = 30, 
                 bg_subtractor_history: int = 500,
                 bg_subtractor_threshold: float = 16,
                 roi_areas: List[np.ndarray] = None):
        """
        Initialize the processing pipeline.
        
        Args:
            buffer_size: Number of frames to keep in buffer
            bg_subtractor_history: History length for background subtractor
            bg_subtractor_threshold: Threshold for background subtractor
            roi_areas: List of polygons defining regions of interest
        """
        self.buffer_size = buffer_size
        self.frame_buffer = []
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_subtractor_history,
            varThreshold=bg_subtractor_threshold,
            detectShadows=True
        )
        
        # Initialize regions of interest
        self.roi_areas = roi_areas if roi_areas is not None else []
        
        # Performance metrics
        self.processing_times = {}
        self.last_processing_time = 0
    
    def add_roi(self, points: np.ndarray) -> None:
        """
        Add a region of interest defined by a polygon.
        
        Args:
            points: Numpy array of points defining the polygon, shape (n, 2)
        """
        if points.shape[0] < 3 or points.shape[1] != 2:
            logger.error("ROI must be defined by at least 3 points with x,y coordinates")
            return
        
        self.roi_areas.append(points)
        logger.info(f"Added ROI with {points.shape[0]} points")
    
    def clear_roi(self) -> None:
        """
        Clear all regions of interest.
        """
        self.roi_areas = []
        logger.info("Cleared all ROIs")
    
    def add_to_buffer(self, frame: np.ndarray) -> None:
        """
        Add a frame to the buffer, maintaining buffer size.
        
        Args:
            frame: The frame to add to the buffer
        """
        if frame is None:
            return
        
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Keep buffer at specified size
        while len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
    
    def apply_background_subtraction(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background subtraction to detect moving objects.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple containing:
                np.ndarray: Foreground mask
                np.ndarray: Frame with foreground highlighted
        """
        start_time = time.time()
        
        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (gray pixels) and noise
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        # Apply morphological operations to remove noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create a copy of the frame with foreground highlighted
        highlighted_frame = frame.copy()
        highlighted_frame[fg_mask == 255] = [0, 0, 255]  # Red color for foreground
        
        # Record processing time
        self.processing_times['background_subtraction'] = time.time() - start_time
        
        return fg_mask, highlighted_frame
    
    def apply_roi_mask(self, frame: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply regions of interest mask to the frame or another mask.
        
        Args:
            frame: Input frame
            mask: Optional mask to apply ROI to (if None, creates a new mask)
            
        Returns:
            Tuple containing:
                np.ndarray: ROI mask
                np.ndarray: Frame with ROI outlined
        """
        start_time = time.time()
        
        # Create a blank mask if none provided
        if mask is None:
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            roi_mask = mask.copy()
        
        # Create a copy of the frame for visualization
        roi_frame = frame.copy()
        
        # If no ROIs defined, use the entire frame
        if not self.roi_areas:
            if mask is None:
                roi_mask.fill(255)  # Set all pixels as ROI
            return roi_mask, roi_frame
        
        # Apply each ROI
        for roi_points in self.roi_areas:
            # Convert points to the format expected by fillPoly
            roi_points_reshaped = roi_points.reshape((-1, 1, 2)).astype(np.int32)
            
            # Fill the ROI in the mask
            if mask is None:
                cv2.fillPoly(roi_mask, [roi_points_reshaped], 255)
            
            # Draw the ROI outline on the frame
            cv2.polylines(roi_frame, [roi_points_reshaped], True, (0, 255, 0), 2)
        
        # If a mask was provided, apply the ROI mask to it
        if mask is not None:
            roi_mask = cv2.bitwise_and(mask, roi_mask)
        
        # Record processing time
        self.processing_times['roi_masking'] = time.time() - start_time
        
        return roi_mask, roi_frame
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a frame through the entire pipeline.
        
        Args:
            frame: Input frame to process
            
        Returns:
            Dict containing processed outputs:
                'original': Original frame
                'fg_mask': Foreground mask from background subtraction
                'fg_highlighted': Frame with foreground highlighted
                'roi_mask': Region of interest mask
                'roi_frame': Frame with ROI outlined
                'final_mask': Combined foreground and ROI mask
                'final_frame': Final processed frame
                'processing_time': Total processing time in seconds
        """
        if frame is None:
            logger.warning("Cannot process None frame")
            return {}
        
        start_time = time.time()
        
        # Add frame to buffer
        self.add_to_buffer(frame)
        
        # Apply background subtraction
        fg_mask, fg_highlighted = self.apply_background_subtraction(frame)
        
        # Apply ROI masking
        roi_mask, roi_frame = self.apply_roi_mask(frame)
        
        # Combine foreground mask with ROI mask
        final_mask = cv2.bitwise_and(fg_mask, roi_mask)
        
        # Create final frame with both foreground and ROI visualization
        final_frame = frame.copy()
        final_frame[final_mask == 255] = [0, 0, 255]  # Red for detected motion
        
        # Draw ROI outlines on final frame
        for roi_points in self.roi_areas:
            roi_points_reshaped = roi_points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(final_frame, [roi_points_reshaped], True, (0, 255, 0), 2)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        self.last_processing_time = total_time
        self.processing_times['total'] = total_time
        
        # Return all processed outputs
        return {
            'original': frame,
            'fg_mask': fg_mask,
            'fg_highlighted': fg_highlighted,
            'roi_mask': roi_mask,
            'roi_frame': roi_frame,
            'final_mask': final_mask,
            'final_frame': final_frame,
            'processing_time': total_time
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the processing pipeline.
        
        Returns:
            Dict containing processing times for different stages
        """
        return self.processing_times.copy()
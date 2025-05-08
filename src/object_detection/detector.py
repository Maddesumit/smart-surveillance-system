"""
Object Detection Module

This module provides functionality for detecting objects in video frames
using the YOLOv5 model.
"""

import os
import logging
import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Class for detecting objects in images using YOLOv5.
    
    This class loads a pre-trained YOLOv5 model and provides methods
    to detect objects in images with confidence scores and bounding boxes.
    """
    
    def __init__(self, model_size='s', confidence_threshold=0.25, device=None):
        """
        Initialize the ObjectDetector with a YOLOv5 model.
        
        Args:
            model_size (str): Size of YOLOv5 model ('n', 's', 'm', 'l', 'x')
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.confidence_threshold = confidence_threshold
        
        # Determine device (use CUDA if available)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load YOLOv8 model (newer version of YOLO from Ultralytics)
        try:
            # Use YOLOv8 instead of YOLOv5 as it's the current version in Ultralytics package
            model_name = f"yolov8{model_size}"
            self.model = YOLO(model_name)
            logger.info(f"Loaded {model_name} model successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            # Try alternative approach - download the model first
            try:
                logger.info("Attempting to download the model explicitly...")
                # This will download the model if it doesn't exist
                self.model = YOLO("yolov8s.pt")
                logger.info("Successfully loaded yolov8s.pt model")
            except Exception as e2:
                logger.error(f"Failed to load alternative model: {str(e2)}")
                raise
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame (numpy.ndarray): Image frame to process
            
        Returns:
            list: List of detection results with format:
                 [class_id, class_name, confidence, [x1, y1, x2, y2]]
        """
        if frame is None:
            logger.warning("Received None frame for detection")
            return []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            # Process results
            detections = []
            for detection in results.boxes.data.tolist():
                # Format: [x1, y1, x2, y2, confidence, class_id]
                x1, y1, x2, y2, confidence, class_id = detection
                
                if confidence >= self.confidence_threshold:
                    class_id = int(class_id)
                    class_name = results.names[class_id]
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Draw detection bounding boxes and labels on a frame.
        
        Args:
            frame (numpy.ndarray): Image frame to draw on
            detections (list): List of detection results
            
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        result_frame = frame.copy()
        
        for detection in detections:
            # Extract detection information
            class_name = detection['class_name']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Generate random color for this class (consistent for same class)
            color = (int(hash(class_name) % 255), 
                     int(hash(class_name + '1') % 255),
                     int(hash(class_name + '2') % 255))
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, 
                          (x1, y1 - text_size[1] - 5),
                          (x1 + text_size[0], y1), 
                          color, 
                          -1)
            
            # Draw label text
            cv2.putText(result_frame, 
                        label, 
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2)
        
        return result_frame
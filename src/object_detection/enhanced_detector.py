#!/usr/bin/env python3
"""
Enhanced Object Detector with Custom Model Support

This is an improved version of the object detector that supports
custom trained models and provides better accuracy for surveillance scenarios.
"""

import cv2
import numpy as np
import logging
import torch
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedObjectDetector:
    """
    Enhanced object detector with support for custom trained models
    and surveillance-optimized configurations.
    """
    
    def __init__(self, 
                 model_path: str = 'yolov8s.pt',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 surveillance_mode: bool = True):
        """
        Initialize the Enhanced Object Detector.
        
        Args:
            model_path: Path to model file (pre-trained or custom)
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            surveillance_mode: Enable surveillance-specific optimizations
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.surveillance_mode = surveillance_mode
        
        # Surveillance-specific classes (high priority for detection)
        self.surveillance_classes = {
            'person', 'backpack', 'handbag', 'suitcase', 
            'bottle', 'laptop', 'cell phone', 'bag',
            'knife', 'scissors', 'bicycle', 'motorbike', 
            'car', 'truck', 'sports ball'
        }
        
        # Load model
        self.model = self._load_model()
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'surveillance_objects': 0,
            'avg_confidence': 0.0,
            'frames_processed': 0
        }
        
        logger.info(f"Enhanced detector initialized with model: {model_path}")
        logger.info(f"Surveillance mode: {surveillance_mode}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model with error handling and validation."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Attempting to download pre-trained model...")
                
                # Try to download if it's a standard model name
                if 'yolov8' in self.model_path and self.model_path.endswith('.pt'):
                    model = YOLO(self.model_path)  # This will download if needed
                else:
                    raise FileNotFoundError(f"Custom model not found: {self.model_path}")
            else:
                model = YOLO(self.model_path)
            
            # Validate model
            if model is None:
                raise ValueError("Failed to load model")
            
            # Set model parameters for better surveillance performance
            if self.surveillance_mode:
                # Configure model for surveillance scenarios
                model.overrides.update({
                    'conf': self.confidence_threshold,
                    'iou': self.iou_threshold,
                    'agnostic_nms': False,  # Class-aware NMS
                    'max_det': 300,  # Allow more detections for crowded scenes
                    'augment': True,  # Test time augmentation
                })
            
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            
            # Log model info
            if hasattr(model, 'names'):
                logger.info(f"Model classes: {len(model.names)}")
                surveillance_classes_in_model = [
                    name for name in model.names.values() 
                    if name in self.surveillance_classes
                ]
                logger.info(f"Surveillance-relevant classes: {surveillance_classes_in_model}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray, return_raw: bool = False) -> List[Dict]:
        """
        Detect objects in a frame with enhanced accuracy.
        
        Args:
            frame: Input frame for detection
            return_raw: Return raw model results if True
            
        Returns:
            List of detection dictionaries or raw results
        """
        if self.model is None:
            logger.warning("Model not loaded. Returning empty detections.")
            return []
        
        try:
            # Preprocess frame for better detection
            processed_frame = self._preprocess_frame(frame)
            
            # Run detection with surveillance optimizations
            results = self.model(
                processed_frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                augment=self.surveillance_mode,  # Test time augmentation
                verbose=False
            )
            
            if return_raw:
                return results
            
            # Process results
            detections = self._process_results(results, frame.shape)
            
            # Apply surveillance-specific filtering
            if self.surveillance_mode:
                detections = self._apply_surveillance_filtering(detections)
            
            # Update statistics
            self._update_stats(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal detection performance."""
        if not self.surveillance_mode:
            return frame
        
        # Apply surveillance-specific preprocessing
        processed = frame.copy()
        
        # Enhance contrast for better small object detection
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        # Denoise slightly to reduce false positives
        processed = cv2.bilateralFilter(processed, 5, 50, 50)
        
        return processed
    
    def _process_results(self, results, frame_shape: Tuple) -> List[Dict]:
        """Process raw model results into standardized detection format."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Get detection data
                xyxy = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = result.names.get(class_id, f'class_{class_id}')
                
                # Filter by confidence
                if confidence < self.confidence_threshold:
                    continue
                
                # Create detection dictionary
                detection = {
                    'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'is_surveillance_object': class_name in self.surveillance_classes
                }
                
                # Add additional metadata for surveillance
                if self.surveillance_mode:
                    detection.update(self._calculate_detection_metadata(detection, frame_shape))
                
                detections.append(detection)
        
        return detections
    
    def _calculate_detection_metadata(self, detection: Dict, frame_shape: Tuple) -> Dict:
        """Calculate additional metadata for surveillance analysis."""
        bbox = detection['bbox']
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate relative position
        center_x = (bbox[0] + bbox[2]) / 2 / frame_width
        center_y = (bbox[1] + bbox[3]) / 2 / frame_height
        
        # Calculate object size relative to frame
        obj_width = (bbox[2] - bbox[0]) / frame_width
        obj_height = (bbox[3] - bbox[1]) / frame_height
        obj_area = obj_width * obj_height
        
        # Determine object zone (for spatial analysis)
        zone = self._get_spatial_zone(center_x, center_y)
        
        metadata = {
            'center_normalized': (center_x, center_y),
            'size_normalized': (obj_width, obj_height),
            'area_normalized': obj_area,
            'spatial_zone': zone,
            'is_small_object': obj_area < 0.01,  # Less than 1% of frame
            'is_large_object': obj_area > 0.25,  # More than 25% of frame
        }
        
        return metadata
    
    def _get_spatial_zone(self, center_x: float, center_y: float) -> str:
        """Determine spatial zone of object for surveillance analysis."""
        if center_y < 0.33:
            if center_x < 0.33:
                return 'top_left'
            elif center_x > 0.67:
                return 'top_right'
            else:
                return 'top_center'
        elif center_y > 0.67:
            if center_x < 0.33:
                return 'bottom_left'
            elif center_x > 0.67:
                return 'bottom_right'
            else:
                return 'bottom_center'
        else:
            if center_x < 0.33:
                return 'middle_left'
            elif center_x > 0.67:
                return 'middle_right'
            else:
                return 'center'
    
    def _apply_surveillance_filtering(self, detections: List[Dict]) -> List[Dict]:
        """Apply surveillance-specific filtering to improve accuracy."""
        if not detections:
            return detections
        
        filtered_detections = []
        
        for detection in detections:
            # Boost confidence for surveillance-relevant objects
            if detection['is_surveillance_object']:
                # Slightly boost confidence for important objects
                detection['confidence'] = min(detection['confidence'] * 1.1, 1.0)
            
            # Filter out very small objects that might be noise (unless they're important)
            if detection.get('is_small_object', False):
                if not detection['is_surveillance_object']:
                    # Skip small non-surveillance objects with low confidence
                    if detection['confidence'] < 0.4:
                        continue
            
            # Filter out detections at frame edges (often partial/false detections)
            bbox = detection['bbox']
            frame_margin = 5  # pixels
            if (bbox[0] <= frame_margin or bbox[1] <= frame_margin or 
                bbox[2] >= detection.get('frame_width', 1000) - frame_margin or
                bbox[3] >= detection.get('frame_height', 1000) - frame_margin):
                if detection['confidence'] < 0.6:  # Higher threshold for edge detections
                    continue
            
            filtered_detections.append(detection)
        
        return filtered_detections
    
    def _update_stats(self, detections: List[Dict]):
        """Update detection statistics for monitoring."""
        self.detection_stats['frames_processed'] += 1
        self.detection_stats['total_detections'] += len(detections)
        
        surveillance_count = sum(1 for d in detections if d['is_surveillance_object'])
        self.detection_stats['surveillance_objects'] += surveillance_count
        
        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
            # Running average
            frames = self.detection_stats['frames_processed']
            current_avg = self.detection_stats['avg_confidence']
            self.detection_stats['avg_confidence'] = (current_avg * (frames - 1) + avg_conf) / frames
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics."""
        stats = self.detection_stats.copy()
        if stats['frames_processed'] > 0:
            stats['detections_per_frame'] = stats['total_detections'] / stats['frames_processed']
            stats['surveillance_ratio'] = stats['surveillance_objects'] / max(stats['total_detections'], 1)
        return stats
    
    def draw_detections(self, 
                       frame: np.ndarray, 
                       detections: List[Dict],
                       show_metadata: bool = False) -> np.ndarray:
        """
        Draw detections on frame with enhanced visualization.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_metadata: Show additional surveillance metadata
            
        Returns:
            Frame with detections drawn
        """
        result_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            is_surveillance = detection['is_surveillance_object']
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on object type
            if is_surveillance:
                color = (0, 255, 0)  # Green for surveillance objects
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            if is_surveillance:
                label += " [S]"  # Mark surveillance objects
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            
            # Draw label background
            cv2.rectangle(result_frame, 
                         (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0] + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(result_frame, label, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw additional metadata if requested
            if show_metadata and 'spatial_zone' in detection:
                zone = detection['spatial_zone']
                cv2.putText(result_frame, zone, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw statistics overlay
        if self.surveillance_mode:
            stats = self.get_detection_stats()
            stats_text = f"Detections: {len(detections)} | Surveillance: {sum(1 for d in detections if d['is_surveillance_object'])}"
            cv2.putText(result_frame, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def load_custom_model(self, model_path: str) -> bool:
        """
        Load a custom trained model.
        
        Args:
            model_path: Path to custom model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading custom model: {model_path}")
            
            self.model_path = model_path
            self.model = self._load_model()
            
            # Reset statistics
            self.detection_stats = {
                'total_detections': 0,
                'surveillance_objects': 0,
                'avg_confidence': 0.0,
                'frames_processed': 0
            }
            
            logger.info("Custom model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {str(e)}")
            return False


def main():
    """Example usage of the Enhanced Object Detector."""
    try:
        # Initialize enhanced detector
        detector = EnhancedObjectDetector(
            model_path='yolov8s.pt',
            confidence_threshold=0.3,
            surveillance_mode=True
        )
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return
        
        logger.info("Starting enhanced detection test...")
        logger.info("Press 'q' to quit, 's' to show stats")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            detections = detector.detect(frame)
            
            # Draw results
            result_frame = detector.draw_detections(frame, detections, show_metadata=True)
            
            # Display
            cv2.imshow('Enhanced Object Detection', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = detector.get_detection_stats()
                logger.info(f"Detection Statistics: {stats}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        final_stats = detector.get_detection_stats()
        logger.info(f"Final Statistics: {final_stats}")
        
    except Exception as e:
        logger.error(f"Error in enhanced detector test: {str(e)}")


if __name__ == "__main__":
    main()

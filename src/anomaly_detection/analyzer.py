"""
Anomaly Detection Module

This module provides functionality for detecting suspicious activities
based on object detection and tracking results.
"""

import logging
import time
import numpy as np
import cv2
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class for detecting anomalies in video frames based on object tracking data.
    
    This class analyzes tracked objects to detect suspicious activities such as:
    - Unattended objects (objects left behind)
    - Restricted area violations
    - Objects that remain stationary for too long
    """
    
    def __init__(self, stationary_threshold=30, confidence_threshold=0.6):
        """
        Initialize the AnomalyDetector.
        
        Args:
            stationary_threshold (int): Number of frames an object must be stationary to trigger an alert
            confidence_threshold (float): Minimum confidence score for anomaly detection
        """
        # Configuration parameters
        self.stationary_threshold = stationary_threshold
        self.confidence_threshold = confidence_threshold
        
        # State tracking
        self.stationary_objects = {}  # Dictionary: object_id -> frames_stationary
        self.restricted_areas = []    # List of polygons defining restricted areas
        self.events = []              # List to store detected anomaly events
        
        logger.info("Anomaly detector initialized")
    
    def add_restricted_area(self, points):
        """
        Add a restricted area polygon.
        
        Args:
            points (list): List of (x, y) tuples defining the polygon vertices
        """
        if len(points) >= 3:  # Need at least 3 points to define a polygon
            self.restricted_areas.append(np.array(points, dtype=np.int32))
            logger.info(f"Added restricted area with {len(points)} points")
        else:
            logger.warning("Restricted area needs at least 3 points")
    
    def detect_anomalies(self, frame, tracked_objects):
        """
        Detect anomalies in the current frame based on tracked objects.
        
        Args:
            frame (numpy.ndarray): Current video frame
            tracked_objects (dict): Dictionary of tracked objects from ObjectTracker
            
        Returns:
            list: List of detected anomalies
        """
        anomalies = []
        
        # Process each tracked object
        for object_id, obj in tracked_objects.items():
            # Check for unattended/stationary objects
            anomaly = self._check_stationary_object(object_id, obj)
            if anomaly:
                anomalies.append(anomaly)
            
            # Check for restricted area violations
            anomaly = self._check_restricted_area(object_id, obj)
            if anomaly:
                anomalies.append(anomaly)
        
        # Clean up objects that are no longer being tracked
        self._cleanup_stationary_objects(tracked_objects)
        
        # Log and store events
        for anomaly in anomalies:
            self._log_event(anomaly)
        
        return anomalies
    
    def _check_stationary_object(self, object_id, obj):
        """
        Check if an object has been stationary for too long.
        
        Args:
            object_id (int): ID of the tracked object
            obj (dict): Object data from tracker
            
        Returns:
            dict or None: Anomaly data if detected, None otherwise
        """
        # Get trajectory points
        trajectory = obj.get('trajectory', [])
        
        # Need at least 2 points to check if stationary
        if len(trajectory) < 2:
            return None
        
        # Calculate movement between last two positions
        last_pos = trajectory[-1]
        prev_pos = trajectory[-2]
        movement = np.sqrt((last_pos[0] - prev_pos[0])**2 + (last_pos[1] - prev_pos[1])**2)
        
        # Check if object is stationary (very little movement)
        is_stationary = movement < 5  # 5 pixels threshold
        
        # Update stationary counter
        if is_stationary:
            if object_id in self.stationary_objects:
                self.stationary_objects[object_id] += 1
            else:
                self.stationary_objects[object_id] = 1
        else:
            # Reset counter if object moved
            self.stationary_objects[object_id] = 0
        
        # Check if object has been stationary for too long
        if self.stationary_objects.get(object_id, 0) >= self.stationary_threshold:
            # Only trigger for certain object types (e.g., bags, suitcases)
            if obj['class_name'] in ['backpack', 'handbag', 'suitcase', 'sports ball', 'bottle']:
                return {
                    'type': 'unattended_object',
                    'object_id': object_id,
                    'class_name': obj['class_name'],
                    'confidence': obj['confidence'],
                    'location': obj['centroid'],
                    'frames_stationary': self.stationary_objects[object_id],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        return None
    
    def _check_restricted_area(self, object_id, obj):
        """
        Check if an object is in a restricted area.
        
        Args:
            object_id (int): ID of the tracked object
            obj (dict): Object data from tracker
            
        Returns:
            dict or None: Anomaly data if detected, None otherwise
        """
        if not self.restricted_areas:
            return None
        
        # Get object centroid
        centroid = obj.get('centroid')
        if not centroid:
            return None
        
        # Check if centroid is inside any restricted area
        for area_idx, area in enumerate(self.restricted_areas):
            if cv2.pointPolygonTest(area, centroid, False) >= 0:  # Point is inside polygon
                return {
                    'type': 'restricted_area_violation',
                    'object_id': object_id,
                    'class_name': obj['class_name'],
                    'confidence': obj['confidence'],
                    'location': centroid,
                    'area_id': area_idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        return None
    
    def _cleanup_stationary_objects(self, tracked_objects):
        """
        Remove stationary object records for objects that are no longer tracked.
        
        Args:
            tracked_objects (dict): Currently tracked objects
        """
        # Get list of objects to remove
        to_remove = [obj_id for obj_id in self.stationary_objects 
                    if obj_id not in tracked_objects]
        
        # Remove objects
        for obj_id in to_remove:
            self.stationary_objects.pop(obj_id, None)
    
    def _log_event(self, anomaly):
        """
        Log an anomaly event and add it to the events list.
        
        Args:
            anomaly (dict): Anomaly event data
        """
        # Apply confidence threshold
        if anomaly['confidence'] < self.confidence_threshold:
            return  # Skip low-confidence anomalies
        
        # Log the event
        logger.warning(f"Anomaly detected: {anomaly['type']} - {anomaly['class_name']} "
                      f"at location {anomaly['location']}")
        
        # Add to events list (with deduplication)
        # Check if this is a duplicate of a recent event
        is_duplicate = False
        for event in self.events[-10:]:  # Check last 10 events
            if (event['type'] == anomaly['type'] and 
                event['object_id'] == anomaly['object_id'] and
                (datetime.now() - datetime.strptime(event['timestamp'], "%Y-%m-%d %H:%M:%S")).seconds < 5):
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.events.append(anomaly)
    
    def draw_anomalies(self, frame, anomalies):
        """
        Draw anomaly indicators on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            anomalies (list): List of detected anomalies
            
        Returns:
            numpy.ndarray: Frame with anomalies highlighted
        """
        result_frame = frame.copy()
        
        # Draw restricted areas
        for area in self.restricted_areas:
            cv2.polylines(result_frame, [area], True, (0, 0, 255), 2)
            cv2.fillPoly(result_frame, [area], (0, 0, 255, 30))  # Semi-transparent fill
        
        # Draw anomalies
        for anomaly in anomalies:
            # Get location
            x, y = anomaly['location']
            
            # Different visualization based on anomaly type
            if anomaly['type'] == 'unattended_object':
                # Draw pulsing circle for unattended object
                pulse = int(time.time() * 3) % 20 + 20  # Pulsing effect
                cv2.circle(result_frame, (x, y), pulse, (0, 165, 255), 2)
                cv2.circle(result_frame, (x, y), 10, (0, 165, 255), -1)
                
                # Add text
                text = f"Unattended {anomaly['class_name']}"
                cv2.putText(result_frame, text, (x - 10, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
            elif anomaly['type'] == 'restricted_area_violation':
                # Draw warning triangle for restricted area violation
                triangle_size = 30
                pts = np.array([
                    [x, y - triangle_size],
                    [x - triangle_size//2, y + triangle_size//2],
                    [x + triangle_size//2, y + triangle_size//2]
                ], np.int32)
                cv2.fillPoly(result_frame, [pts], (0, 0, 255))
                
                # Add exclamation mark
                cv2.putText(result_frame, "!", (x - 5, y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Add text
                text = f"Restricted Area: {anomaly['class_name']}"
                cv2.putText(result_frame, text, (x - 10, y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_frame
    
    def get_recent_events(self, count=10):
        """
        Get the most recent anomaly events.
        
        Args:
            count (int): Number of events to return
            
        Returns:
            list: List of recent events
        """
        return self.events[-count:]
"""
Object Tracking Module

This module provides functionality for tracking detected objects
across video frames using simple tracking algorithms.
"""

import logging
import cv2
import numpy as np
from scipy.spatial import distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectTracker:
    """
    Class for tracking objects across video frames.
    
    This class implements a simple tracking algorithm based on IoU (Intersection over Union)
    and centroid distance to associate detections across frames.
    """
    
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Initialize the ObjectTracker.
        
        Args:
            max_disappeared (int): Maximum number of frames an object can be missing
            max_distance (int): Maximum pixel distance for associating detections
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary: object_id -> {centroid, bbox, class_id, disappeared}
        self.disappeared = {}  # Dictionary: object_id -> count of frames disappeared
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        logger.info("Object tracker initialized")
    
    def _calculate_centroid(self, bbox):
        """
        Calculate the centroid of a bounding box.
        
        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (cx, cy) centroid coordinates
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU score between 0 and 1
        """
        # Determine intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate area of intersection rectangle
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate IoU
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0
        
        return intersection_area / union_area
    
    def update(self, detections):
        """
        Update object tracking with new detections.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            dict: Dictionary of tracked objects with their IDs
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] += 1
                
                # Remove object if it has been missing for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.objects.pop(object_id)
                    self.disappeared.pop(object_id)
            
            return self.objects
        
        # If we have no existing objects, register all detections as new objects
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
        
        # Otherwise, match existing objects with new detections
        else:
            self._match_and_update(detections)
        
        return self.objects
    
    def _register(self, detection):
        """
        Register a new object with a unique ID.
        
        Args:
            detection (dict): Detection dictionary
        """
        object_id = self.next_object_id
        centroid = self._calculate_centroid(detection['bbox'])
        
        self.objects[object_id] = {
            'centroid': centroid,
            'bbox': detection['bbox'],
            'class_id': detection['class_id'],
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'trajectory': [centroid]  # Store trajectory for movement analysis
        }
        
        self.disappeared[object_id] = 0
        self.next_object_id += 1
    
    def _match_and_update(self, detections):
        """
        Match existing objects with new detections and update tracking.
        
        Args:
            detections (list): List of detection dictionaries
        """
        # Get centroids of existing objects and new detections
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[object_id]['centroid'] for object_id in object_ids]
        object_bboxes = [self.objects[object_id]['bbox'] for object_id in object_ids]
        object_classes = [self.objects[object_id]['class_id'] for object_id in object_ids]
        
        detection_centroids = [self._calculate_centroid(d['bbox']) for d in detections]
        detection_bboxes = [d['bbox'] for d in detections]
        detection_classes = [d['class_id'] for d in detections]
        
        # Calculate distance matrix between all objects and detections
        D = np.zeros((len(object_centroids), len(detection_centroids)))
        
        # Fill distance matrix with IoU scores and centroid distances
        for i, (oc, ob, ocls) in enumerate(zip(object_centroids, object_bboxes, object_classes)):
            for j, (dc, db, dcls) in enumerate(zip(detection_centroids, detection_bboxes, detection_classes)):
                # Only match objects of the same class
                if ocls != dcls:
                    D[i, j] = float('inf')
                    continue
                
                # Calculate IoU between bounding boxes
                iou = self._calculate_iou(ob, db)
                
                # Calculate centroid distance
                cent_dist = distance.euclidean(oc, dc)
                
                # Combine metrics (higher IoU and lower distance is better)
                if iou > 0.3:  # If IoU is good enough, prioritize it
                    D[i, j] = (1.0 - iou) * 100  # Convert to distance (lower is better)
                else:
                    D[i, j] = cent_dist
        
        # Mark all objects as disappeared initially
        for object_id in object_ids:
            self.disappeared[object_id] += 1
        
        # Match objects to detections using greedy algorithm
        matched_indices = []
        
        # Sort distances and match closest pairs first
        for _ in range(min(len(object_centroids), len(detection_centroids))):
            # Find the smallest distance
            if np.min(D) == float('inf'):
                break
            
            # Find indices of minimum distance
            i, j = np.unravel_index(np.argmin(D), D.shape)
            
            # If the distance is too large, don't match
            if D[i, j] > self.max_distance:
                break
            
            # Add match to results
            matched_indices.append((i, j))
            
            # Set rows and columns to infinity to prevent re-matching
            D[i, :] = float('inf')
            D[:, j] = float('inf')
        
        # Update matched objects
        for i, j in matched_indices:
            object_id = object_ids[i]
            detection = detections[j]
            
            # Update object with new detection
            centroid = self._calculate_centroid(detection['bbox'])
            self.objects[object_id]['centroid'] = centroid
            self.objects[object_id]['bbox'] = detection['bbox']
            self.objects[object_id]['confidence'] = detection['confidence']
            self.objects[object_id]['trajectory'].append(centroid)
            
            # Reset disappeared counter
            self.disappeared[object_id] = 0
        
        # Register unmatched detections as new objects
        matched_detection_indices = [j for _, j in matched_indices]
        unmatched_detections = [d for idx, d in enumerate(detections) 
                               if idx not in matched_detection_indices]
        
        for detection in unmatched_detections:
            self._register(detection)
        
        # Remove objects that have disappeared for too long
        for object_id in list(self.disappeared.keys()):
            if self.disappeared[object_id] > self.max_disappeared:
                self.objects.pop(object_id, None)
                self.disappeared.pop(object_id, None)
    
    def draw_tracks(self, frame):
        """
        Draw object tracks on a frame.
        
        Args:
            frame (numpy.ndarray): Image frame to draw on
            
        Returns:
            numpy.ndarray: Frame with tracks drawn
        """
        result_frame = frame.copy()
        
        for object_id, obj in self.objects.items():
            # Extract object information
            bbox = obj['bbox']
            class_name = obj['class_name']
            trajectory = obj['trajectory']
            
            # Generate color based on object ID
            color = (int(hash(object_id) % 255), 
                     int(hash(object_id + 10) % 255),
                     int(hash(object_id + 20) % 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and class
            label = f"ID:{object_id} {class_name}"
            cv2.putText(result_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory (last 20 points)
            if len(trajectory) > 1:
                for i in range(1, min(20, len(trajectory))):
                    # Draw line between consecutive points
                    pt1 = trajectory[i - 1]
                    pt2 = trajectory[i]
                    cv2.line(result_frame, pt1, pt2, color, 2)
        
        return result_frame
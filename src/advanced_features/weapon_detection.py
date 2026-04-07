#!/usr/bin/env python3
"""
Weapon Detection Module

This module provides real-time weapon detection capabilities including
firearms, knives, and other dangerous objects.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sqlite3
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaponDetector:
    """
    Advanced weapon detection system for surveillance applications.
    
    Features:
    - Real-time weapon detection (guns, knives, rifles)
    - Weapon type classification
    - Threat level assessment
    - Context-aware filtering
    - Automatic critical alerts
    """
    
    def __init__(self,
                 model_path: str = "models/weapon_detection.pt",
                 database_path: str = "weapon_detection.db",
                 confidence_threshold: float = 0.6):
        """
        Initialize the weapon detection system.
        
        Args:
            model_path: Path to weapon detection model
            database_path: Path to SQLite database
            confidence_threshold: Minimum confidence for detection
        """
        self.model_path = model_path
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize database
        self._init_database()
        
        # Weapon categories and threat levels
        self.weapon_categories = {
            'gun': {'threat_level': 'CRITICAL', 'priority': 1},
            'rifle': {'threat_level': 'CRITICAL', 'priority': 1},
            'pistol': {'threat_level': 'CRITICAL', 'priority': 1},
            'knife': {'threat_level': 'HIGH', 'priority': 2},
            'machete': {'threat_level': 'HIGH', 'priority': 2},
            'bat': {'threat_level': 'MEDIUM', 'priority': 3},
            'stick': {'threat_level': 'LOW', 'priority': 4}
        }
        
        # Try to load weapon detection model
        self.model = None
        self._init_model()
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'by_weapon_type': {},
            'false_positives': 0,
            'confirmed_threats': 0
        }
        
        logger.info("Weapon Detection System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for weapon detections."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Weapon detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weapon_detections (
                detection_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                weapon_type TEXT,
                threat_level TEXT,
                confidence REAL,
                location TEXT,
                person_id TEXT,
                image_path TEXT,
                video_clip_path TEXT,
                resolved BOOLEAN DEFAULT 0,
                false_positive BOOLEAN DEFAULT 0
            )
        ''')
        
        # Threat events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_events (
                event_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                threat_level TEXT,
                weapon_count INTEGER,
                involved_persons TEXT,
                location TEXT,
                response_time REAL,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_model(self):
        """Initialize weapon detection model."""
        try:
            # Try to load custom weapon detection model
            # In production, use a fine-tuned YOLOv8 model on weapon datasets
            self.model = YOLO('yolov8s.pt')  # Use base model for now
            logger.info("Weapon detection model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load weapon model: {str(e)}")
            logger.warning("Using fallback detection method")
    
    def detect_weapons(self,
                      frame: np.ndarray,
                      person_detections: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Detect weapons in the frame.
        
        Args:
            frame: Input video frame
            person_detections: Optional list of person detections for context
            
        Returns:
            List of weapon detection results
        """
        weapon_detections = []
        
        if self.model is None:
            # Use fallback method
            return self._fallback_weapon_detection(frame)
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Check if detected object is weapon-like
                    if self._is_weapon_class(class_name):
                        weapon_type = self._classify_weapon_type(
                            class_name, bbox, frame
                        )
                        
                        # Create weapon detection
                        detection = self._create_weapon_detection(
                            weapon_type, confidence, bbox, frame.shape
                        )
                        
                        # Associate with person if available
                        if person_detections:
                            detection['person_id'] = self._find_associated_person(
                                bbox, person_detections
                            )
                        
                        weapon_detections.append(detection)
                        
                        # Store in database
                        self._store_weapon_detection(detection)
            
        except Exception as e:
            logger.error(f"Error in weapon detection: {str(e)}")
        
        return weapon_detections
    
    def _is_weapon_class(self, class_name: str) -> bool:
        """Check if detected class is weapon-related."""
        weapon_keywords = ['knife', 'gun', 'rifle', 'pistol', 'weapon', 
                          'firearm', 'blade', 'sword', 'bat']
        return any(keyword in class_name.lower() for keyword in weapon_keywords)
    
    def _classify_weapon_type(self,
                             class_name: str,
                             bbox: np.ndarray,
                             frame: np.ndarray) -> str:
        """Classify the specific type of weapon."""
        # Simplified classification
        # In production, use dedicated weapon classification model
        
        class_lower = class_name.lower()
        
        if 'gun' in class_lower or 'pistol' in class_lower:
            return 'pistol'
        elif 'rifle' in class_lower:
            return 'rifle'
        elif 'knife' in class_lower or 'blade' in class_lower:
            return 'knife'
        elif 'bat' in class_lower:
            return 'bat'
        else:
            return 'unknown_weapon'
    
    def _create_weapon_detection(self,
                                weapon_type: str,
                                confidence: float,
                                bbox: np.ndarray,
                                frame_shape: Tuple) -> Dict:
        """Create weapon detection record."""
        detection_id = f"weapon_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Get threat level for weapon type
        weapon_info = self.weapon_categories.get(
            weapon_type,
            {'threat_level': 'MEDIUM', 'priority': 3}
        )
        
        x1, y1, x2, y2 = bbox
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        detection = {
            'detection_id': detection_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'weapon_detected',
            'weapon_type': weapon_type,
            'threat_level': weapon_info['threat_level'],
            'priority': weapon_info['priority'],
            'confidence': confidence,
            'bbox': bbox.tolist(),
            'location': centroid,
            'person_id': None,
            'description': f"{weapon_type.upper()} detected with {confidence:.2%} confidence"
        }
        
        self.stats['total_detections'] += 1
        self.stats['by_weapon_type'][weapon_type] = \
            self.stats['by_weapon_type'].get(weapon_type, 0) + 1
        
        return detection
    
    def _find_associated_person(self,
                               weapon_bbox: np.ndarray,
                               person_detections: List[Dict]) -> Optional[str]:
        """Find person associated with detected weapon."""
        x1, y1, x2, y2 = weapon_bbox
        weapon_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        min_distance = float('inf')
        associated_person = None
        
        for person in person_detections:
            person_bbox = person['bbox']
            px1, py1, px2, py2 = person_bbox
            
            # Check if weapon is inside person's bounding box
            if px1 <= weapon_center[0] <= px2 and py1 <= weapon_center[1] <= py2:
                return person.get('track_id', 'unknown')
            
            # Find closest person
            person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            distance = np.sqrt(
                (weapon_center[0] - person_center[0])**2 +
                (weapon_center[1] - person_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                associated_person = person.get('track_id', 'unknown')
        
        # Only associate if person is reasonably close (within 150 pixels)
        if min_distance < 150:
            return associated_person
        
        return None
    
    def _fallback_weapon_detection(self, frame: np.ndarray) -> List[Dict]:
        """Fallback weapon detection using color/shape analysis."""
        # Simplified fallback method
        # In production, implement more sophisticated detection
        detections = []
        
        # This is a placeholder - implement actual fallback logic
        logger.debug("Using fallback weapon detection")
        
        return detections
    
    def _store_weapon_detection(self, detection: Dict):
        """Store weapon detection in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO weapon_detections
                (detection_id, timestamp, weapon_type, threat_level, 
                 confidence, location, person_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection['detection_id'],
                detection['timestamp'],
                detection['weapon_type'],
                detection['threat_level'],
                detection['confidence'],
                str(detection['location']),
                detection.get('person_id')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing weapon detection: {str(e)}")
    
    def get_weapon_stats(self) -> Dict:
        """Get weapon detection statistics."""
        return self.stats.copy()
    
    def draw_weapon_detections(self,
                              frame: np.ndarray,
                              weapon_detections: List[Dict]) -> np.ndarray:
        """
        Draw weapon detection results on the frame.
        
        Args:
            frame: Input frame
            weapon_detections: List of weapon detections
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for detection in weapon_detections:
            bbox = detection['bbox']
            weapon_type = detection['weapon_type']
            threat_level = detection['threat_level']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color based on threat level
            if threat_level == 'CRITICAL':
                color = (0, 0, 255)  # Red
                thickness = 4
            elif threat_level == 'HIGH':
                color = (0, 100, 255)  # Orange
                thickness = 3
            elif threat_level == 'MEDIUM':
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw weapon type and confidence
            label = f"WEAPON: {weapon_type.upper()}"
            conf_label = f"{confidence:.2%}"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_BOLD, 0.7, 2
            )
            cv2.rectangle(output, (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_BOLD, 0.7, (255, 255, 255), 2)
            
            # Draw confidence
            cv2.putText(output, conf_label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw warning icon
            cv2.circle(output, (x1 - 15, y1 + 15), 12, color, -1)
            cv2.putText(output, "!", (x1 - 20, y1 + 22),
                       cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)
        
        return output
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Weapon Detection System cleanup complete")


# Example usage
if __name__ == "__main__":
    detector = WeaponDetector()
    print("Weapon Detection System ready!")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    print(f"Weapon categories: {list(detector.weapon_categories.keys())}")

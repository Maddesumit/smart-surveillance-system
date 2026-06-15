#!/usr/bin/env python3
"""
Enhanced Abandoned Object Detection Module

Detects objects that have been left unattended for a configurable
duration. Uses background subtraction and object tracking to
distinguish between stationary objects and abandoned ones.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import sqlite3

logger = logging.getLogger(__name__)


class AbandonedObjectDetector:
    """
    Enhanced abandoned object detection system.
    
    Detection logic:
    1. Track all non-person objects (bags, suitcases, etc.)
    2. Monitor how long each object remains stationary
    3. Check if the object's owner (nearest person) has moved away
    4. Trigger alert if object is stationary AND no person is nearby
    
    Features:
    - Owner association (links objects to nearest person)
    - Owner departure detection
    - Configurable time thresholds
    - Object type classification
    - Zone-based sensitivity
    """
    
    SUSPICIOUS_OBJECTS = [
        'backpack', 'handbag', 'suitcase', 'bag', 'briefcase',
        'box', 'package', 'sports ball', 'bottle'
    ]
    
    def __init__(self,
                 database_path: str = "abandoned_objects.db",
                 stationary_threshold: float = 30.0,
                 owner_distance_threshold: float = 200.0,
                 alert_threshold: float = 60.0,
                 movement_tolerance: float = 15.0):
        """
        Initialize abandoned object detection.
        
        Args:
            database_path: Path to SQLite database
            stationary_threshold: Seconds object must be still before monitoring
            owner_distance_threshold: Pixels - if owner moves this far, object is "abandoned"
            alert_threshold: Seconds after owner leaves before alert
            movement_tolerance: Pixels - max movement to still count as stationary
        """
        self.database_path = database_path
        self.stationary_threshold = stationary_threshold
        self.owner_distance_threshold = owner_distance_threshold
        self.alert_threshold = alert_threshold
        self.movement_tolerance = movement_tolerance
        
        # Object tracking state
        self.tracked_objects = {}  # object_key -> state
        
        # Stats
        self.stats = {
            'total_abandoned_alerts': 0,
            'objects_monitored': 0,
            'false_positives': 0,
            'active_suspicious': 0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Abandoned Object Detection initialized")
        logger.info(f"  Stationary threshold: {stationary_threshold}s")
        logger.info(f"  Owner distance: {owner_distance_threshold}px")
        logger.info(f"  Alert threshold: {alert_threshold}s")
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS abandoned_object_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                object_type TEXT,
                location_x REAL,
                location_y REAL,
                duration_seconds REAL,
                owner_id TEXT,
                owner_distance REAL,
                alert_level TEXT,
                bbox TEXT,
                resolved BOOLEAN DEFAULT 0,
                false_positive BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_abandoned_objects(self,
                                 frame: np.ndarray,
                                 all_detections: List[Dict],
                                 person_detections: List[Dict]) -> List[Dict]:
        """
        Detect abandoned objects in the frame.
        
        Args:
            frame: Input video frame
            all_detections: All object detections (including non-person)
            person_detections: Person-only detections for owner tracking
            
        Returns:
            List of abandoned object alerts
        """
        current_time = datetime.now()
        abandoned_alerts = []
        
        # Get suspicious objects (non-person objects that could be abandoned)
        suspicious_objects = [
            d for d in all_detections 
            if d.get('class_name') in self.SUSPICIOUS_OBJECTS
        ]
        
        # Get person positions for owner tracking
        person_positions = []
        for person in person_detections:
            if person.get('class_name') == 'person':
                bbox = person['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                person_positions.append({
                    'id': person.get('track_id', 'unknown'),
                    'position': (cx, cy),
                    'bbox': bbox
                })
        
        # Track each suspicious object
        active_object_keys = set()
        
        for obj in suspicious_objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            obj_class = obj.get('class_name', 'unknown')
            
            # Create unique key for this object (based on position)
            obj_key = f"{obj_class}_{int(cx//50)}_{int(cy//50)}"
            active_object_keys.add(obj_key)
            
            # Initialize tracking for new object
            if obj_key not in self.tracked_objects:
                # Find nearest person as potential owner
                nearest_person = self._find_nearest_person(cx, cy, person_positions)
                
                self.tracked_objects[obj_key] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'position': (cx, cy),
                    'initial_position': (cx, cy),
                    'class_name': obj_class,
                    'bbox': bbox,
                    'owner_id': nearest_person['id'] if nearest_person else None,
                    'owner_initial_pos': nearest_person['position'] if nearest_person else None,
                    'is_stationary': False,
                    'is_abandoned': False,
                    'owner_departed_time': None,
                    'alert_issued': False
                }
                self.stats['objects_monitored'] += 1
                continue
            
            state = self.tracked_objects[obj_key]
            state['last_seen'] = current_time
            state['bbox'] = bbox
            
            # Check if object has moved
            displacement = np.sqrt(
                (cx - state['initial_position'][0]) ** 2 +
                (cy - state['initial_position'][1]) ** 2
            )
            
            if displacement > self.movement_tolerance:
                # Object moved - reset
                state['first_seen'] = current_time
                state['initial_position'] = (cx, cy)
                state['is_stationary'] = False
                state['is_abandoned'] = False
                state['owner_departed_time'] = None
                state['alert_issued'] = False
                continue
            
            # Object is stationary
            stationary_time = (current_time - state['first_seen']).total_seconds()
            
            if stationary_time >= self.stationary_threshold:
                state['is_stationary'] = True
                
                # Check if owner has moved away
                owner_distance = self._get_owner_distance(state, person_positions)
                
                if owner_distance is not None and owner_distance > self.owner_distance_threshold:
                    # Owner has departed
                    if state['owner_departed_time'] is None:
                        state['owner_departed_time'] = current_time
                    
                    abandoned_time = (current_time - state['owner_departed_time']).total_seconds()
                    
                    if abandoned_time >= self.alert_threshold and not state['alert_issued']:
                        # Generate abandoned object alert
                        state['is_abandoned'] = True
                        state['alert_issued'] = True
                        
                        alert = {
                            'type': 'abandoned_object',
                            'object_type': obj_class,
                            'timestamp': current_time.isoformat(),
                            'location': (int(cx), int(cy)),
                            'bbox': bbox,
                            'stationary_time': stationary_time,
                            'abandoned_time': abandoned_time,
                            'owner_distance': owner_distance,
                            'priority': 'high',
                            'threat_level': 'HIGH',
                            'message': f"Abandoned {obj_class} detected! Unattended for {abandoned_time:.0f}s",
                            'owner_id': state['owner_id']
                        }
                        
                        abandoned_alerts.append(alert)
                        self._store_event(alert)
                        self.stats['total_abandoned_alerts'] += 1
                        
                        logger.warning(f"ABANDONED OBJECT: {obj_class} at ({cx:.0f}, {cy:.0f})")
                
                elif owner_distance is None or owner_distance <= self.owner_distance_threshold:
                    # Owner is still nearby - reset departure timer
                    state['owner_departed_time'] = None
        
        # Cleanup objects no longer visible
        to_remove = [k for k in self.tracked_objects if k not in active_object_keys]
        for key in to_remove:
            state = self.tracked_objects[key]
            if (current_time - state['last_seen']).total_seconds() > 5:
                del self.tracked_objects[key]
        
        # Update stats
        self.stats['active_suspicious'] = sum(
            1 for s in self.tracked_objects.values() if s['is_abandoned']
        )
        
        return abandoned_alerts
    
    def _find_nearest_person(self, x: float, y: float, 
                             person_positions: List[Dict]) -> Optional[Dict]:
        """Find the nearest person to a given point."""
        if not person_positions:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for person in person_positions:
            px, py = person['position']
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = person
        
        # Only associate if person is reasonably close
        if min_dist < self.owner_distance_threshold:
            return nearest
        return None
    
    def _get_owner_distance(self, state: Dict, person_positions: List[Dict]) -> Optional[float]:
        """Get distance between object and its owner."""
        owner_id = state.get('owner_id')
        obj_pos = state['initial_position']
        
        if owner_id:
            # Find owner by ID
            for person in person_positions:
                if person['id'] == owner_id:
                    px, py = person['position']
                    return np.sqrt((obj_pos[0] - px) ** 2 + (obj_pos[1] - py) ** 2)
        
        # If owner not found by ID, find nearest person
        nearest = self._find_nearest_person(obj_pos[0], obj_pos[1], person_positions)
        if nearest:
            px, py = nearest['position']
            return np.sqrt((obj_pos[0] - px) ** 2 + (obj_pos[1] - py) ** 2)
        
        # No persons visible - consider object abandoned
        return self.owner_distance_threshold + 1
    
    def _store_event(self, event: Dict):
        """Store abandoned object event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO abandoned_object_events 
                (object_type, location_x, location_y, duration_seconds, 
                 owner_id, owner_distance, alert_level, bbox)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event['object_type'],
                event['location'][0],
                event['location'][1],
                event['stationary_time'],
                event.get('owner_id'),
                event.get('owner_distance'),
                event['priority'],
                str(event['bbox'])
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing abandoned object event: {e}")
    
    def draw_abandoned_objects(self, frame: np.ndarray) -> np.ndarray:
        """Draw abandoned object indicators on frame."""
        output = frame.copy()
        current_time = datetime.now()
        
        for obj_key, state in self.tracked_objects.items():
            if not state['is_stationary']:
                continue
            
            bbox = state['bbox']
            x1, y1, x2, y2 = bbox
            stationary_time = (current_time - state['first_seen']).total_seconds()
            
            if state['is_abandoned']:
                # Red for abandoned
                color = (0, 0, 255)
                label = f"ABANDONED: {state['class_name']} ({stationary_time:.0f}s)"
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            elif state['is_stationary']:
                # Yellow for stationary/monitoring
                color = (0, 255, 255)
                label = f"Monitoring: {state['class_name']} ({stationary_time:.0f}s)"
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            else:
                continue
            
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    def get_stats(self) -> Dict:
        """Get abandoned object detection statistics."""
        return self.stats.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Abandoned Object Detection cleanup complete")

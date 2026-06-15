#!/usr/bin/env python3
"""
Fall Detection Module

Detects when a person falls using aspect ratio analysis,
vertical velocity tracking, and pose estimation.
Useful for elderly care, workplace safety, and public spaces.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque
import sqlite3

logger = logging.getLogger(__name__)


class FallDetector:
    """
    Fall detection system using bounding box aspect ratio changes,
    vertical velocity, and optional pose estimation.
    
    Detection logic:
    1. Track person's bounding box aspect ratio over time
    2. A fall is indicated when:
       - Aspect ratio changes from tall (standing) to wide (fallen)
       - Rapid downward movement of centroid
       - Person remains in horizontal position
    """
    
    def __init__(self,
                 database_path: str = "fall_detection.db",
                 aspect_ratio_threshold: float = 1.0,
                 velocity_threshold: float = 50.0,
                 confirmation_frames: int = 10,
                 cooldown_seconds: float = 30.0):
        """
        Initialize fall detection system.
        
        Args:
            database_path: Path to SQLite database
            aspect_ratio_threshold: Ratio below which person is considered fallen
                                   (width/height > threshold means horizontal)
            velocity_threshold: Minimum downward velocity (pixels/frame) for fall
            confirmation_frames: Frames person must stay down to confirm fall
            cooldown_seconds: Seconds before same person can trigger another fall alert
        """
        self.database_path = database_path
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.velocity_threshold = velocity_threshold
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds
        
        # Tracking state per person
        self.person_state = {}  # person_id -> state dict
        
        # Stats
        self.stats = {
            'total_falls_detected': 0,
            'false_alarms': 0,
            'persons_monitored': 0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Fall Detection System initialized")
        logger.info(f"  Aspect ratio threshold: {aspect_ratio_threshold}")
        logger.info(f"  Velocity threshold: {velocity_threshold}")
        logger.info(f"  Confirmation frames: {confirmation_frames}")
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                person_id TEXT,
                location_x REAL,
                location_y REAL,
                confidence REAL,
                fall_duration_seconds REAL,
                bbox TEXT,
                resolved BOOLEAN DEFAULT 0,
                false_positive BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_falls(self,
                     frame: np.ndarray,
                     person_detections: List[Dict]) -> List[Dict]:
        """
        Detect falls in the current frame.
        
        Args:
            frame: Input video frame
            person_detections: List of person detections with bbox and track_id
            
        Returns:
            List of fall detection events
        """
        fall_events = []
        current_time = datetime.now()
        
        for person in person_detections:
            if person.get('class_name') != 'person':
                continue
            
            person_id = person.get('track_id', str(id(person)))
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            width = x2 - x1
            height = y2 - y1
            
            if height <= 0 or width <= 0:
                continue
            
            # Calculate metrics
            aspect_ratio = width / height  # >1 means wider than tall
            centroid_y = (y1 + y2) / 2
            centroid_x = (x1 + x2) / 2
            
            # Initialize state for new person
            if person_id not in self.person_state:
                self.person_state[person_id] = {
                    'positions_y': deque(maxlen=30),
                    'aspect_ratios': deque(maxlen=30),
                    'is_fallen': False,
                    'fall_start_time': None,
                    'fallen_frames': 0,
                    'last_alert_time': None,
                    'standing_height': height  # Initial height as reference
                }
                self.stats['persons_monitored'] += 1
            
            state = self.person_state[person_id]
            state['positions_y'].append(centroid_y)
            state['aspect_ratios'].append(aspect_ratio)
            
            # Update standing height reference (use max height seen)
            if height > state['standing_height'] * 0.8 and aspect_ratio < 0.8:
                state['standing_height'] = max(state['standing_height'], height)
            
            # Calculate vertical velocity
            velocity_y = 0
            if len(state['positions_y']) >= 2:
                velocity_y = state['positions_y'][-1] - state['positions_y'][-2]
            
            # Fall detection logic
            is_horizontal = aspect_ratio > self.aspect_ratio_threshold
            height_reduced = height < state['standing_height'] * 0.6
            rapid_descent = velocity_y > self.velocity_threshold
            
            # Check for fall conditions
            if is_horizontal or (height_reduced and rapid_descent):
                state['fallen_frames'] += 1
                
                if not state['is_fallen'] and state['fallen_frames'] >= self.confirmation_frames:
                    # Check cooldown
                    if (state['last_alert_time'] is None or 
                        (current_time - state['last_alert_time']).total_seconds() > self.cooldown_seconds):
                        
                        # Confirmed fall
                        state['is_fallen'] = True
                        state['fall_start_time'] = current_time
                        state['last_alert_time'] = current_time
                        
                        confidence = min(1.0, state['fallen_frames'] / (self.confirmation_frames * 2))
                        
                        fall_event = {
                            'type': 'fall_detected',
                            'person_id': person_id,
                            'timestamp': current_time.isoformat(),
                            'location': (int(centroid_x), int(centroid_y)),
                            'bbox': bbox,
                            'confidence': confidence,
                            'aspect_ratio': aspect_ratio,
                            'velocity': velocity_y,
                            'threat_level': 'HIGH',
                            'message': f"Fall detected! Person {person_id} may need assistance",
                            'priority': 'high'
                        }
                        
                        fall_events.append(fall_event)
                        self._store_fall_event(fall_event)
                        self.stats['total_falls_detected'] += 1
                        
                        logger.warning(f"FALL DETECTED: Person {person_id} at ({centroid_x:.0f}, {centroid_y:.0f})")
            else:
                # Person is upright - reset fall state
                if state['fallen_frames'] > 0:
                    state['fallen_frames'] = max(0, state['fallen_frames'] - 2)
                
                if state['is_fallen'] and state['fallen_frames'] == 0:
                    state['is_fallen'] = False
                    state['fall_start_time'] = None
        
        # Cleanup old person states
        self._cleanup_old_states()
        
        return fall_events
    
    def _cleanup_old_states(self):
        """Remove states for persons not seen recently."""
        current_time = datetime.now()
        to_remove = []
        
        for person_id, state in self.person_state.items():
            if len(state['positions_y']) == 0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_state[person_id]
    
    def _store_fall_event(self, event: Dict):
        """Store fall event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fall_events (person_id, location_x, location_y, confidence, bbox)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                event['person_id'],
                event['location'][0],
                event['location'][1],
                event['confidence'],
                str(event['bbox'])
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing fall event: {e}")
    
    def draw_fall_detections(self, frame: np.ndarray, fall_events: List[Dict]) -> np.ndarray:
        """Draw fall detection results on frame."""
        output = frame.copy()
        
        # Draw status for all tracked persons
        for person_id, state in self.person_state.items():
            if state['is_fallen']:
                # Draw alert for fallen person
                if state['positions_y']:
                    # Use last known position
                    pass  # Will be drawn from fall_events
        
        # Draw fall events
        for event in fall_events:
            bbox = event['bbox']
            x1, y1, x2, y2 = bbox
            location = event['location']
            
            # Red pulsing border for fall
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Alert text
            cv2.putText(output, "FALL DETECTED!", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_BOLD, 0.8, (0, 0, 255), 2)
            cv2.putText(output, f"Conf: {event['confidence']:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Alert circle
            cv2.circle(output, location, 30, (0, 0, 255), 3)
            cv2.putText(output, "!", (location[0] - 8, location[1] + 8),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (0, 0, 255), 3)
        
        return output
    
    def get_stats(self) -> Dict:
        """Get fall detection statistics."""
        active_fallen = sum(1 for s in self.person_state.values() if s['is_fallen'])
        return {
            **self.stats,
            'currently_fallen': active_fallen,
            'active_tracking': len(self.person_state)
        }
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Fall Detection System cleanup complete")

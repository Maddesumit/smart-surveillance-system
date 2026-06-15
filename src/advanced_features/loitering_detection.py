#!/usr/bin/env python3
"""
Loitering Detection Module

Detects when a person stays in a specific area for too long.
Tracks person positions over time and triggers alerts when
dwell time exceeds configurable thresholds.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import sqlite3

logger = logging.getLogger(__name__)


class LoiteringDetector:
    """
    Loitering detection system that monitors person dwell time in zones.
    
    Features:
    - Per-person time tracking in defined zones
    - Configurable time thresholds per zone
    - Visual indicators showing dwell time
    - Progressive alert levels (warning -> alert -> critical)
    - Zone-based monitoring with custom boundaries
    """
    
    def __init__(self,
                 database_path: str = "loitering_detection.db",
                 warning_threshold: float = 60.0,
                 alert_threshold: float = 180.0,
                 critical_threshold: float = 300.0,
                 movement_tolerance: float = 50.0,
                 alert_cooldown: float = 60.0):
        """
        Initialize loitering detection system.
        
        Args:
            database_path: Path to SQLite database
            warning_threshold: Seconds before warning (default 1 min)
            alert_threshold: Seconds before alert (default 3 min)
            critical_threshold: Seconds before critical alert (default 5 min)
            movement_tolerance: Max pixel movement to still count as loitering
            alert_cooldown: Seconds between repeated alerts for same person
        """
        self.database_path = database_path
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.movement_tolerance = movement_tolerance
        self.alert_cooldown = alert_cooldown
        
        # Person tracking state
        self.person_state = {}  # person_id -> tracking state
        
        # Monitored zones (list of polygons)
        self.zones = []
        
        # Stats
        self.stats = {
            'total_loitering_events': 0,
            'active_loiterers': 0,
            'warnings_issued': 0,
            'alerts_issued': 0,
            'critical_alerts': 0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info("Loitering Detection System initialized")
        logger.info(f"  Warning: {warning_threshold}s, Alert: {alert_threshold}s, Critical: {critical_threshold}s")
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS loitering_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                person_id TEXT,
                duration_seconds REAL,
                location_x REAL,
                location_y REAL,
                zone_id TEXT,
                alert_level TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_zone(self, zone_id: str, polygon: List[Tuple[int, int]], 
                 custom_threshold: Optional[float] = None):
        """
        Add a monitored zone.
        
        Args:
            zone_id: Unique zone identifier
            polygon: List of (x, y) points defining the zone boundary
            custom_threshold: Override alert threshold for this zone
        """
        self.zones.append({
            'id': zone_id,
            'polygon': np.array(polygon, dtype=np.int32),
            'threshold': custom_threshold or self.alert_threshold
        })
        logger.info(f"Added monitoring zone: {zone_id}")
    
    def detect_loitering(self,
                         frame: np.ndarray,
                         person_detections: List[Dict]) -> List[Dict]:
        """
        Detect loitering behavior.
        
        Args:
            frame: Input video frame
            person_detections: List of person detections
            
        Returns:
            List of loitering events/alerts
        """
        current_time = datetime.now()
        loitering_events = []
        active_person_ids = set()
        
        for person in person_detections:
            if person.get('class_name') != 'person':
                continue
            
            person_id = person.get('track_id', str(id(person)))
            active_person_ids.add(person_id)
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            # Person centroid
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Initialize state for new person
            if person_id not in self.person_state:
                self.person_state[person_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'initial_position': (cx, cy),
                    'current_position': (cx, cy),
                    'max_displacement': 0.0,
                    'is_loitering': False,
                    'alert_level': None,
                    'last_alert_time': None,
                    'zone_id': None
                }
            
            state = self.person_state[person_id]
            state['last_seen'] = current_time
            state['current_position'] = (cx, cy)
            
            # Calculate displacement from initial position
            displacement = np.sqrt(
                (cx - state['initial_position'][0]) ** 2 +
                (cy - state['initial_position'][1]) ** 2
            )
            state['max_displacement'] = max(state['max_displacement'], displacement)
            
            # Check if person has moved significantly
            if displacement > self.movement_tolerance:
                # Person moved - reset tracking
                state['first_seen'] = current_time
                state['initial_position'] = (cx, cy)
                state['max_displacement'] = 0.0
                state['is_loitering'] = False
                state['alert_level'] = None
                continue
            
            # Calculate dwell time
            dwell_time = (current_time - state['first_seen']).total_seconds()
            
            # Determine alert level
            new_alert_level = None
            if dwell_time >= self.critical_threshold:
                new_alert_level = 'critical'
            elif dwell_time >= self.alert_threshold:
                new_alert_level = 'high'
            elif dwell_time >= self.warning_threshold:
                new_alert_level = 'medium'
            
            # Generate alert if level changed or cooldown expired
            if new_alert_level and new_alert_level != state['alert_level']:
                should_alert = True
                if state['last_alert_time']:
                    time_since_alert = (current_time - state['last_alert_time']).total_seconds()
                    if time_since_alert < self.alert_cooldown:
                        should_alert = False
                
                if should_alert:
                    state['alert_level'] = new_alert_level
                    state['is_loitering'] = True
                    state['last_alert_time'] = current_time
                    
                    # Determine zone
                    zone_id = self._get_zone_for_point(cx, cy)
                    state['zone_id'] = zone_id
                    
                    event = {
                        'type': 'loitering_detected',
                        'person_id': person_id,
                        'timestamp': current_time.isoformat(),
                        'dwell_time': dwell_time,
                        'location': (int(cx), int(cy)),
                        'bbox': bbox,
                        'alert_level': new_alert_level,
                        'priority': new_alert_level,
                        'zone_id': zone_id,
                        'message': f"Loitering: Person {person_id} stationary for {dwell_time:.0f}s",
                        'displacement': displacement
                    }
                    
                    loitering_events.append(event)
                    self._store_event(event)
                    
                    # Update stats
                    self.stats['total_loitering_events'] += 1
                    if new_alert_level == 'medium':
                        self.stats['warnings_issued'] += 1
                    elif new_alert_level == 'high':
                        self.stats['alerts_issued'] += 1
                    elif new_alert_level == 'critical':
                        self.stats['critical_alerts'] += 1
        
        # Cleanup persons no longer visible
        to_remove = [pid for pid in self.person_state if pid not in active_person_ids]
        for pid in to_remove:
            state = self.person_state[pid]
            if (current_time - state['last_seen']).total_seconds() > 10:
                del self.person_state[pid]
        
        # Update active loiterers count
        self.stats['active_loiterers'] = sum(
            1 for s in self.person_state.values() if s['is_loitering']
        )
        
        return loitering_events
    
    def _get_zone_for_point(self, x: float, y: float) -> Optional[str]:
        """Check which zone a point belongs to."""
        point = (int(x), int(y))
        for zone in self.zones:
            if cv2.pointPolygonTest(zone['polygon'], point, False) >= 0:
                return zone['id']
        return None
    
    def _store_event(self, event: Dict):
        """Store loitering event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO loitering_events 
                (person_id, duration_seconds, location_x, location_y, zone_id, alert_level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event['person_id'],
                event['dwell_time'],
                event['location'][0],
                event['location'][1],
                event.get('zone_id'),
                event['alert_level']
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing loitering event: {e}")
    
    def draw_loitering_indicators(self, frame: np.ndarray, person_detections: List[Dict]) -> np.ndarray:
        """Draw loitering indicators on frame."""
        output = frame.copy()
        
        # Draw zones
        for zone in self.zones:
            cv2.polylines(output, [zone['polygon']], True, (255, 255, 0), 2)
            # Zone label
            centroid = zone['polygon'].mean(axis=0).astype(int)
            cv2.putText(output, zone['id'], tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw loitering indicators for tracked persons
        current_time = datetime.now()
        for person_id, state in self.person_state.items():
            if not state['is_loitering']:
                continue
            
            cx, cy = state['current_position']
            cx, cy = int(cx), int(cy)
            dwell_time = (current_time - state['first_seen']).total_seconds()
            
            # Color based on alert level
            if state['alert_level'] == 'critical':
                color = (0, 0, 255)  # Red
            elif state['alert_level'] == 'high':
                color = (0, 128, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw circle around loiterer
            radius = int(30 + (dwell_time / 60) * 10)  # Grows with time
            cv2.circle(output, (cx, cy), radius, color, 2)
            
            # Draw dwell time
            time_text = f"{dwell_time:.0f}s"
            cv2.putText(output, time_text, (cx - 20, cy - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw "LOITERING" label
            cv2.putText(output, "LOITERING", (cx - 40, cy + radius + 20),
                       cv2.FONT_HERSHEY_BOLD, 0.5, color, 2)
        
        return output
    
    def get_stats(self) -> Dict:
        """Get loitering detection statistics."""
        return self.stats.copy()
    
    def get_active_loiterers(self) -> List[Dict]:
        """Get list of currently loitering persons."""
        current_time = datetime.now()
        loiterers = []
        
        for person_id, state in self.person_state.items():
            if state['is_loitering']:
                loiterers.append({
                    'person_id': person_id,
                    'dwell_time': (current_time - state['first_seen']).total_seconds(),
                    'position': state['current_position'],
                    'alert_level': state['alert_level'],
                    'zone_id': state['zone_id']
                })
        
        return loiterers
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Loitering Detection System cleanup complete")

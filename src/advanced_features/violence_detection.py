#!/usr/bin/env python3
"""
Violence Detection Module

This module provides real-time violence and fight detection capabilities
using pose estimation and action recognition techniques.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import deque
import sqlite3
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViolenceDetector:
    """
    Advanced violence detection system for surveillance applications.
    
    Features:
    - Real-time fight detection
    - Aggressive behavior recognition
    - Multi-person interaction analysis
    - Temporal pattern analysis
    - Threat level assessment
    """
    
    def __init__(self, 
                 database_path: str = "violence_detection.db",
                 sensitivity: float = 0.7,
                 temporal_window: int = 30):
        """
        Initialize the violence detection system.
        
        Args:
            database_path: Path to SQLite database
            sensitivity: Detection sensitivity (0.0-1.0)
            temporal_window: Number of frames to analyze
        """
        self.database_path = database_path
        self.sensitivity = sensitivity
        self.temporal_window = temporal_window
        
        # Initialize database
        self._init_database()
        
        # Temporal buffers for each person
        self.person_buffers = {}
        
        # Violence detection thresholds
        self.thresholds = {
            'rapid_movement': 0.6,
            'aggressive_pose': 0.7,
            'interaction_proximity': 100,  # pixels
            'sustained_duration': 15  # frames
        }
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'false_positives': 0,
            'confirmed_incidents': 0
        }
        
        logger.info("Violence Detection System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for violence events."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Violence events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violence_events (
                event_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                location TEXT,
                threat_level TEXT,
                involved_persons TEXT,
                duration_seconds REAL,
                confidence REAL,
                description TEXT,
                video_clip_path TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        # Person involvement table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_involvement (
                involvement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                person_id TEXT,
                role TEXT,
                aggression_score REAL,
                timestamp DATETIME,
                FOREIGN KEY (event_id) REFERENCES violence_events(event_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_violence(self,
                       frame: np.ndarray,
                       person_detections: List[Dict],
                       pose_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Detect violence and aggressive behavior in the frame.
        
        Args:
            frame: Input video frame
            person_detections: List of person detection objects
            pose_data: Optional pose estimation data
            
        Returns:
            List of violence detection results
        """
        violence_events = []
        
        if len(person_detections) < 2:
            # Violence typically involves multiple people
            return violence_events
        
        # Analyze each person's behavior
        person_behaviors = []
        for detection in person_detections:
            person_id = detection.get('track_id', 'unknown')
            bbox = detection['bbox']
            
            # Extract behavior features
            behavior = self._analyze_person_behavior(
                frame, bbox, person_id, pose_data
            )
            person_behaviors.append(behavior)
        
        # Analyze interactions between people
        interactions = self._analyze_interactions(
            person_behaviors, person_detections
        )
        
        # Detect violence based on interactions
        for interaction in interactions:
            if interaction['violence_score'] > self.sensitivity:
                event = self._create_violence_event(
                    interaction, frame.shape
                )
                violence_events.append(event)
                
                # Store in database
                self._store_violence_event(event)
        
        return violence_events
    
    def _analyze_person_behavior(self,
                                 frame: np.ndarray,
                                 bbox: List[int],
                                 person_id: str,
                                 pose_data: Optional[List[Dict]]) -> Dict:
        """Analyze individual person's behavior."""
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        
        # Initialize buffer for this person
        if person_id not in self.person_buffers:
            self.person_buffers[person_id] = {
                'positions': deque(maxlen=self.temporal_window),
                'movements': deque(maxlen=self.temporal_window),
                'poses': deque(maxlen=self.temporal_window)
            }
        
        buffer = self.person_buffers[person_id]
        
        # Calculate movement features
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        buffer['positions'].append(centroid)
        
        # Calculate movement speed and acceleration
        movement_speed = 0.0
        if len(buffer['positions']) >= 2:
            prev_pos = buffer['positions'][-2]
            movement_speed = np.sqrt(
                (centroid[0] - prev_pos[0])**2 + 
                (centroid[1] - prev_pos[1])**2
            )
        
        buffer['movements'].append(movement_speed)
        
        # Analyze pose if available
        aggressive_pose_score = 0.0
        if pose_data:
            aggressive_pose_score = self._detect_aggressive_pose(
                pose_data, person_id
            )
        
        # Calculate behavior metrics
        avg_movement = np.mean(list(buffer['movements'])) if buffer['movements'] else 0
        max_movement = np.max(list(buffer['movements'])) if buffer['movements'] else 0
        movement_variance = np.var(list(buffer['movements'])) if buffer['movements'] else 0
        
        return {
            'person_id': person_id,
            'bbox': bbox,
            'centroid': centroid,
            'movement_speed': movement_speed,
            'avg_movement': avg_movement,
            'max_movement': max_movement,
            'movement_variance': movement_variance,
            'aggressive_pose_score': aggressive_pose_score,
            'rapid_movement': max_movement > self.thresholds['rapid_movement'] * 100
        }
    
    def _detect_aggressive_pose(self,
                               pose_data: List[Dict],
                               person_id: str) -> float:
        """Detect aggressive poses (raised arms, lunging, etc.)."""
        # Simplified aggressive pose detection
        # In production, use trained pose classification model
        
        score = 0.0
        
        # Look for raised arms, wide stance, forward lean
        # This is a placeholder - implement actual pose analysis
        
        return score
    
    def _analyze_interactions(self,
                             person_behaviors: List[Dict],
                             person_detections: List[Dict]) -> List[Dict]:
        """Analyze interactions between people."""
        interactions = []
        
        # Check all pairs of people
        for i in range(len(person_behaviors)):
            for j in range(i + 1, len(person_behaviors)):
                person_a = person_behaviors[i]
                person_b = person_behaviors[j]
                
                # Calculate proximity
                distance = np.sqrt(
                    (person_a['centroid'][0] - person_b['centroid'][0])**2 +
                    (person_a['centroid'][1] - person_b['centroid'][1])**2
                )
                
                # Check if people are close enough to interact
                if distance < self.thresholds['interaction_proximity']:
                    # Calculate violence score
                    violence_score = self._calculate_violence_score(
                        person_a, person_b, distance
                    )
                    
                    if violence_score > 0.3:  # Minimum threshold
                        interactions.append({
                            'person_a': person_a['person_id'],
                            'person_b': person_b['person_id'],
                            'distance': distance,
                            'violence_score': violence_score,
                            'location': person_a['centroid'],
                            'behaviors': {
                                'person_a': person_a,
                                'person_b': person_b
                            }
                        })
        
        return interactions
    
    def _calculate_violence_score(self,
                                  person_a: Dict,
                                  person_b: Dict,
                                  distance: float) -> float:
        """Calculate violence probability score."""
        score = 0.0
        
        # Factor 1: Rapid movements (30% weight)
        if person_a['rapid_movement'] or person_b['rapid_movement']:
            score += 0.3
        
        # Factor 2: High movement variance (20% weight)
        avg_variance = (person_a['movement_variance'] + person_b['movement_variance']) / 2
        if avg_variance > 50:
            score += 0.2
        
        # Factor 3: Aggressive poses (30% weight)
        avg_aggression = (person_a['aggressive_pose_score'] + person_b['aggressive_pose_score']) / 2
        score += avg_aggression * 0.3
        
        # Factor 4: Close proximity (20% weight)
        proximity_score = max(0, 1 - (distance / self.thresholds['interaction_proximity']))
        score += proximity_score * 0.2
        
        return min(score, 1.0)
    
    def _create_violence_event(self,
                              interaction: Dict,
                              frame_shape: Tuple) -> Dict:
        """Create violence event record."""
        event_id = f"violence_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Determine threat level
        violence_score = interaction['violence_score']
        if violence_score > 0.9:
            threat_level = 'CRITICAL'
        elif violence_score > 0.75:
            threat_level = 'HIGH'
        elif violence_score > 0.6:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'LOW'
        
        event = {
            'event_id': event_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'violence_detected',
            'threat_level': threat_level,
            'violence_score': violence_score,
            'involved_persons': [
                interaction['person_a'],
                interaction['person_b']
            ],
            'location': interaction['location'],
            'distance': interaction['distance'],
            'description': f"Potential violence detected between {interaction['person_a']} and {interaction['person_b']}",
            'behaviors': interaction['behaviors']
        }
        
        self.stats['total_detections'] += 1
        
        return event
    
    def _store_violence_event(self, event: Dict):
        """Store violence event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO violence_events 
                (event_id, timestamp, location, threat_level, involved_persons, 
                 confidence, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event['event_id'],
                event['timestamp'],
                str(event['location']),
                event['threat_level'],
                ','.join(event['involved_persons']),
                event['violence_score'],
                event['description']
            ))
            
            # Store person involvement
            for person_id in event['involved_persons']:
                cursor.execute('''
                    INSERT INTO person_involvement
                    (event_id, person_id, role, aggression_score, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    event['event_id'],
                    person_id,
                    'participant',
                    event['violence_score'],
                    event['timestamp']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing violence event: {str(e)}")
    
    def get_violence_stats(self) -> Dict:
        """Get violence detection statistics."""
        return self.stats.copy()
    
    def draw_violence_detection(self,
                               frame: np.ndarray,
                               violence_events: List[Dict]) -> np.ndarray:
        """
        Draw violence detection results on the frame.
        
        Args:
            frame: Input frame
            violence_events: List of violence events
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for event in violence_events:
            location = event['location']
            threat_level = event['threat_level']
            violence_score = event['violence_score']
            
            # Color based on threat level
            if threat_level == 'CRITICAL':
                color = (0, 0, 255)  # Red
            elif threat_level == 'HIGH':
                color = (0, 100, 255)  # Orange
            elif threat_level == 'MEDIUM':
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw alert circle
            cv2.circle(output, location, 50, color, 3)
            cv2.circle(output, location, 10, color, -1)
            
            # Draw warning text
            text = f"VIOLENCE: {threat_level}"
            cv2.putText(output, text, (location[0] - 80, location[1] - 60),
                       cv2.FONT_HERSHEY_BOLD, 0.7, color, 2)
            
            # Draw confidence
            conf_text = f"Conf: {violence_score:.2f}"
            cv2.putText(output, conf_text, (location[0] - 60, location[1] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Violence Detection System cleanup complete")


# Example usage
if __name__ == "__main__":
    detector = ViolenceDetector()
    print("Violence Detection System ready!")
    print(f"Sensitivity: {detector.sensitivity}")
    print(f"Temporal window: {detector.temporal_window} frames")

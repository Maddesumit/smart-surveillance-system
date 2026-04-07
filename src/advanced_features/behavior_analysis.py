#!/usr/bin/env python3
"""
Behavior Analysis Module

This module provides advanced behavior analysis capabilities for surveillance systems,
including activity recognition, pose estimation, crowd behavior analysis, and anomaly detection.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import threading
import time
from collections import deque, defaultdict

# Try to import pose estimation libraries
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Install with: pip install mediapipe")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorAnalyzer:
    """
    Advanced behavior analysis system for surveillance applications.
    
    Features:
    - Human pose estimation and tracking
    - Activity recognition (walking, running, falling, fighting)
    - Crowd behavior analysis
    - Suspicious behavior detection
    - Loitering detection
    - Gesture recognition
    - Social distancing monitoring
    """
    
    def __init__(self, 
                 database_path: str = "behavior_analysis.db",
                 pose_confidence: float = 0.5,
                 activity_window_size: int = 30):
        """
        Initialize the behavior analysis system.
        
        Args:
            database_path: Path to SQLite database for behavior data
            pose_confidence: Minimum confidence for pose detection
            activity_window_size: Number of frames to analyze for activity recognition
        """
        self.database_path = database_path
        self.pose_confidence = pose_confidence
        self.activity_window_size = activity_window_size
        
        # Pose estimation components
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose_detector = self.mp_pose.Pose(
                min_detection_confidence=pose_confidence,
                min_tracking_confidence=pose_confidence
            )
        else:
            self.mp_pose = None
            self.mp_drawing = None
            self.pose_detector = None
        
        # Tracking data
        self.person_tracks = {}  # person_id -> track data
        self.activity_history = defaultdict(deque)  # person_id -> activity sequence
        self.pose_history = defaultdict(deque)  # person_id -> pose sequence
        
        # Behavior patterns
        self.behavior_patterns = {
            'walking': {'speed_range': (0.5, 2.0), 'pose_stability': 0.3},
            'running': {'speed_range': (2.0, 8.0), 'pose_stability': 0.4},
            'standing': {'speed_range': (0.0, 0.3), 'pose_stability': 0.1},
            'sitting': {'height_ratio': 0.6, 'pose_stability': 0.1},
            'falling': {'speed_change': 3.0, 'height_change': 0.4},
            'fighting': {'movement_chaos': 0.7, 'proximity': 50},
            'loitering': {'stationary_time': 300}  # 5 minutes
        }
        
        # Analytics data
        self.behavior_stats = {
            'total_persons_analyzed': 0,
            'activities_detected': defaultdict(int),
            'suspicious_behaviors': 0,
            'crowd_events': 0
        }
        
        # Threading
        self.analysis_lock = threading.Lock()
        
        # Enable/disable flag
        self.enabled = True
        
        # Behavior rules configuration
        self.behavior_rules_config = {
            'loitering_threshold': 300,  # seconds
            'running_speed_threshold': 2.0,  # m/s
            'falling_detection_enabled': True,
            'fighting_detection_enabled': True,
            'crowd_density_threshold': 0.7,
            'suspicious_movement_enabled': True,
            'restricted_areas': [],
            'activity_thresholds': {
                'walking': {'min_speed': 0.5, 'max_speed': 2.0},
                'running': {'min_speed': 2.0, 'max_speed': 8.0},
                'standing': {'max_speed': 0.3, 'max_movement': 0.1}
            }
        }
        
        # Recent events cache
        self.recent_events = deque(maxlen=100)
        
        # Initialize database
        self._init_database()
        
        # Load saved rules
        self._load_rules_from_database()
        
        logger.info("Behavior Analysis System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for behavior data."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Behavior events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    activity_type TEXT,
                    confidence REAL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds REAL,
                    location_x REAL,
                    location_y REAL,
                    bounding_box TEXT,
                    is_suspicious BOOLEAN DEFAULT FALSE,
                    notes TEXT
                )
            ''')
            
            # Pose data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pose_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pose_landmarks TEXT,
                    pose_confidence REAL,
                    frame_number INTEGER
                )
            ''')
            
            # Crowd events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crowd_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    person_count INTEGER,
                    density REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    location TEXT,
                    duration_seconds REAL,
                    severity TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Behavior analysis database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def analyze_behavior(self, 
                        frame: np.ndarray,
                        person_detections: List[Dict],
                        frame_number: int = 0) -> List[Dict]:
        """
        Analyze behavior in the current frame.
        
        Args:
            frame: Input video frame
            person_detections: List of person detection objects
            frame_number: Current frame number
            
        Returns:
            List of behavior analysis results
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available for pose estimation")
            return []
        
        behavior_results = []
        
        try:
            with self.analysis_lock:
                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                for person in person_detections:
                    # Handle different input formats (dict, tuple, or object)
                    if isinstance(person, dict):
                        if person.get('class_name') != 'person':
                            continue
                        person_id = person.get('track_id', f"person_{len(self.person_tracks)}")
                        bbox = person['bbox']  # [x, y, w, h]
                    elif hasattr(person, 'class_name'):
                        # Handle tracking object format
                        if getattr(person, 'class_name', '') != 'person':
                            continue
                        person_id = getattr(person, 'track_id', f"person_{len(self.person_tracks)}")
                        bbox = getattr(person, 'bbox', [0, 0, 0, 0])
                    else:
                        # Skip unsupported formats
                        continue
                    
                    # Extract person region - ensure bbox is valid
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                    else:
                        continue
                    
                    # Ensure valid bounding box
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    
                    if x2 <= x or y2 <= y:
                        continue
                    
                    person_region = rgb_frame[y:y2, x:x2]
                    
                    # Pose estimation
                    pose_results = self.pose_detector.process(person_region)
                    
                    if pose_results.pose_landmarks:
                        # Analyze pose and behavior
                        behavior_result = self._analyze_person_behavior(
                            person_id, pose_results, bbox, frame_number
                        )
                        
                        if behavior_result:
                            behavior_results.append(behavior_result)
                
                # Analyze crowd behavior
                crowd_analysis = self._analyze_crowd_behavior(person_detections, frame.shape)
                if crowd_analysis:
                    behavior_results.append(crowd_analysis)
                
        except Exception as e:
            logger.error(f"Error analyzing behavior: {str(e)}")
        
        return behavior_results
    
    def _analyze_person_behavior(self, 
                               person_id: str,
                               pose_results,
                               bbox: List[int],
                               frame_number: int) -> Optional[Dict]:
        """Analyze individual person behavior."""
        try:
            # Extract pose landmarks
            landmarks = pose_results.pose_landmarks.landmark
            pose_data = self._extract_pose_features(landmarks)
            
            # Update person track
            current_time = datetime.now()
            
            if person_id not in self.person_tracks:
                self.person_tracks[person_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': deque(maxlen=self.activity_window_size),
                    'poses': deque(maxlen=self.activity_window_size),
                    'activities': deque(maxlen=self.activity_window_size),
                    'suspicious_count': 0
                }
            
            track = self.person_tracks[person_id]
            track['last_seen'] = current_time
            track['positions'].append(bbox)
            track['poses'].append(pose_data)
            
            # Activity recognition
            activity = self._recognize_activity(person_id, pose_data, bbox)
            track['activities'].append(activity)
            
            # Suspicious behavior detection
            is_suspicious = self._detect_suspicious_behavior(person_id, activity, pose_data)
            
            if is_suspicious:
                track['suspicious_count'] += 1
            
            # Create behavior result
            behavior_result = {
                'person_id': person_id,
                'activity': activity,
                'pose_confidence': pose_results.pose_world_landmarks is not None,
                'is_suspicious': is_suspicious,
                'bbox': bbox,
                'timestamp': current_time.isoformat(),
                'frame_number': frame_number,
                'pose_features': pose_data
            }
            
            # Update statistics
            self.behavior_stats['activities_detected'][activity] += 1
            if is_suspicious:
                self.behavior_stats['suspicious_behaviors'] += 1
            
            # Store in database
            self._store_behavior_event(behavior_result)
            
            return behavior_result
            
        except Exception as e:
            logger.error(f"Error analyzing person behavior: {str(e)}")
            return None
    
    def _extract_pose_features(self, landmarks) -> Dict:
        """Extract relevant features from pose landmarks."""
        try:
            # Key landmarks indices (MediaPipe Pose)
            key_points = {
                'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14,
                'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24,
                'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28
            }
            
            # Extract coordinates
            pose_points = {}
            for name, idx in key_points.items():
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    pose_points[name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
            
            # Calculate derived features
            features = {
                'pose_points': pose_points,
                'body_height': self._calculate_body_height(pose_points),
                'body_width': self._calculate_body_width(pose_points),
                'body_lean': self._calculate_body_lean(pose_points),
                'limb_movement': self._calculate_limb_movement(pose_points),
                'pose_stability': self._calculate_pose_stability(pose_points)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pose features: {str(e)}")
            return {}
    
    def _calculate_pose_stability(self, pose_points: Dict) -> float:
        """Calculate pose stability score."""
        try:
            # Check if key joints are visible and confident
            key_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            visible_joints = 0
            total_visibility = 0
            
            for joint in key_joints:
                if joint in pose_points:
                    visibility = pose_points[joint].get('visibility', 0)
                    if visibility > 0.5:
                        visible_joints += 1
                    total_visibility += visibility
            
            if len(key_joints) == 0:
                return 0.0
                
            # Calculate stability based on visibility and joint presence
            visibility_score = total_visibility / len(key_joints)
            presence_score = visible_joints / len(key_joints)
            
            return (visibility_score + presence_score) / 2.0
            
        except Exception as e:
            logger.error(f"Error calculating pose stability: {str(e)}")
            return 0.5

    def _calculate_body_height(self, pose_points: Dict) -> float:
        """Calculate body height from pose points."""
        try:
            if 'nose' in pose_points and 'left_ankle' in pose_points:
                nose_y = pose_points['nose']['y']
                ankle_y = pose_points['left_ankle']['y']
                return abs(ankle_y - nose_y)
            elif 'nose' in pose_points and 'right_ankle' in pose_points:
                nose_y = pose_points['nose']['y']
                ankle_y = pose_points['right_ankle']['y']
                return abs(ankle_y - nose_y)
        except:
            pass
        return 0.8  # Default height

    def _calculate_body_width(self, pose_points: Dict) -> float:
        """Calculate approximate body width from pose."""
        try:
            if 'left_shoulder' in pose_points and 'right_shoulder' in pose_points:
                left_x = pose_points['left_shoulder']['x']
                right_x = pose_points['right_shoulder']['x']
                return abs(right_x - left_x)
        except:
            pass
        return 0.2  # Default width

    def _calculate_body_lean(self, pose_points: Dict) -> float:
        """Calculate body lean angle."""
        try:
            if 'left_shoulder' in pose_points and 'right_shoulder' in pose_points:
                left_shoulder = pose_points['left_shoulder']
                right_shoulder = pose_points['right_shoulder']
                
                # Calculate shoulder line angle
                dx = right_shoulder['x'] - left_shoulder['x']
                dy = right_shoulder['y'] - left_shoulder['y']
                
                angle = np.arctan2(dy, dx) * 180 / np.pi
                return abs(angle)
        except:
            pass
        return 0.0

    def _calculate_limb_movement(self, pose_points: Dict) -> float:
        """Calculate limb movement intensity."""
        # This would require comparing with previous poses
        # For now, return a placeholder based on limb positions
        try:
            movement_score = 0.0
            limb_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
            
            for joint in limb_joints:
                if joint in pose_points:
                    # Simple movement estimation based on joint position variance
                    movement_score += pose_points[joint].get('visibility', 0)
            
            return movement_score / len(limb_joints) if limb_joints else 0.5
        except:
            return 0.5

    def _recognize_activity(self, person_id: str, pose_data: Dict, bbox: List[int]) -> str:
        """Recognize current activity based on pose and movement."""
        try:
            track = self.person_tracks.get(person_id)
            if not track or len(track['positions']) < 2:
                return 'unknown'
            
            # Calculate movement speed
            prev_pos = track['positions'][-2]
            curr_pos = bbox
            
            # Calculate center points
            prev_center = [prev_pos[0] + prev_pos[2]//2, prev_pos[1] + prev_pos[3]//2]
            curr_center = [curr_pos[0] + curr_pos[2]//2, curr_pos[1] + curr_pos[3]//2]
            
            # Calculate distance moved
            distance = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            
            # Estimate speed (pixels per frame)
            speed = distance
            
            # Get pose features
            body_height = pose_data.get('body_height', 0)
            pose_stability = pose_data.get('pose_stability', 0)
            
            # Activity classification
            if speed < 5:
                if body_height < 0.6:  # Sitting/crouching
                    return 'sitting'
                else:
                    return 'standing'
            elif speed < 20:
                return 'walking'
            else:
                return 'running'
                
        except Exception as e:
            logger.error(f"Error recognizing activity: {str(e)}")
            return 'unknown'
    
    def _detect_suspicious_behavior(self, person_id: str, activity: str, pose_data: Dict) -> bool:
        """Detect if current behavior is suspicious."""
        try:
            track = self.person_tracks.get(person_id)
            if not track:
                return False
            
            # Check for loitering
            time_in_area = (datetime.now() - track['first_seen']).total_seconds()
            if activity == 'standing' and time_in_area > self.behavior_patterns['loitering']['stationary_time']:
                return True
            
            # Check for unusual pose patterns
            pose_stability = pose_data.get('pose_stability', 1.0)
            if pose_stability < 0.3:  # Very unstable pose
                return True
            
            # Check for rapid activity changes
            if len(track['activities']) >= 5:
                recent_activities = list(track['activities'])[-5:]
                unique_activities = len(set(recent_activities))
                if unique_activities >= 4:  # Too many activity changes
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting suspicious behavior: {str(e)}")
            return False
    
    def _analyze_crowd_behavior(self, person_detections: List[Dict], frame_shape: Tuple) -> Optional[Dict]:
        """Analyze crowd behavior patterns."""
        try:
            person_count = len([p for p in person_detections if p.get('class_name') == 'person'])
            
            if person_count < 3:  # Not a crowd
                return None
            
            # Calculate crowd density
            frame_area = frame_shape[0] * frame_shape[1]
            density = person_count / frame_area * 10000  # persons per 10k pixels
            
            # Determine crowd event type
            event_type = 'normal_crowd'
            severity = 'low'
            
            if density > 0.01:  # High density threshold
                event_type = 'high_density_crowd'
                severity = 'medium'
            
            if person_count > 20:  # Large crowd
                event_type = 'large_gathering'
                severity = 'high'
            
            crowd_result = {
                'event_type': 'crowd_analysis',
                'crowd_event_type': event_type,
                'person_count': person_count,
                'density': density,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'frame_area': frame_area
            }
            
            # Update statistics
            self.behavior_stats['crowd_events'] += 1
            
            # Store in database
            self._store_crowd_event(crowd_result)
            
            return crowd_result
            
        except Exception as e:
            logger.error(f"Error analyzing crowd behavior: {str(e)}")
            return None
    
    def _store_behavior_event(self, behavior_result: Dict):
        """Store behavior event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO behavior_events 
                (person_id, activity_type, confidence, start_time, location_x, location_y, 
                 bounding_box, is_suspicious, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                behavior_result['person_id'],
                behavior_result['activity'],
                behavior_result['pose_confidence'],
                behavior_result['timestamp'],
                behavior_result['bbox'][0],
                behavior_result['bbox'][1],
                json.dumps(behavior_result['bbox']),
                behavior_result['is_suspicious'],
                json.dumps(behavior_result.get('pose_features', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing behavior event: {str(e)}")
    
    def _store_crowd_event(self, crowd_result: Dict):
        """Store crowd event in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO crowd_events 
                (event_type, person_count, density, location, severity)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                crowd_result['crowd_event_type'],
                crowd_result['person_count'],
                crowd_result['density'],
                json.dumps({'frame_area': crowd_result['frame_area']}),
                crowd_result['severity']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing crowd event: {str(e)}")
    
    def get_behavior_stats(self) -> Dict:
        """Get behavior analysis statistics."""
        stats = self.behavior_stats.copy()
        stats['active_persons'] = len(self.person_tracks)
        return stats
    
    def get_person_activities(self, person_id: str, limit: int = 10) -> List[Dict]:
        """Get recent activities for a specific person."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT activity_type, confidence, start_time, is_suspicious, notes
                FROM behavior_events 
                WHERE person_id = ?
                ORDER BY start_time DESC
                LIMIT ?
            ''', (person_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'activity': row[0],
                    'confidence': row[1],
                    'timestamp': row[2],
                    'is_suspicious': bool(row[3]),
                    'notes': json.loads(row[4]) if row[4] else {}
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting person activities: {str(e)}")
            return []
    
    def draw_behavior_analysis(self, frame: np.ndarray, behavior_results: List[Dict]) -> np.ndarray:
        """
        Draw behavior analysis results on the frame.
        
        Args:
            frame: Input frame
            behavior_results: List of behavior analysis results
            
        Returns:
            Frame with drawn analysis
        """
        result_frame = frame.copy()
        
        for result in behavior_results:
            if result.get('event_type') == 'crowd_analysis':
                # Draw crowd information
                text = f"Crowd: {result['person_count']} people, Density: {result['density']:.3f}"
                cv2.putText(result_frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                continue
            
            # Draw individual behavior
            bbox = result.get('bbox', [0, 0, 100, 100])
            x, y, w, h = bbox
            
            activity = result.get('activity', 'unknown')
            is_suspicious = result.get('is_suspicious', False)
            person_id = result.get('person_id', 'unknown')
            
            # Choose color based on suspicion level
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw activity label
            label = f"ID:{person_id} - {activity}"
            if is_suspicious:
                label += " [SUSPICIOUS]"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame,
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         color, -1)
            
            # Label text
            cv2.putText(result_frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw pose if available
            if MEDIAPIPE_AVAILABLE and result.get('pose_features'):
                self._draw_pose_landmarks(result_frame, result['pose_features'], (x, y, w, h))
        
        return result_frame
    
    def _draw_pose_landmarks(self, frame: np.ndarray, pose_features: Dict, bbox: Tuple):
        """Draw pose landmarks on the frame."""
        try:
            if not pose_features.get('pose_points'):
                return
            
            x, y, w, h = bbox
            pose_points = pose_features['pose_points']
            
            # Draw key connections
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle')
            ]
            
            for start_point, end_point in connections:
                if start_point in pose_points and end_point in pose_points:
                    start = pose_points[start_point]
                    end = pose_points[end_point]
                    
                    if start.get('visibility', 0) > 0.5 and end.get('visibility', 0) > 0.5:
                        start_pos = (
                            int(x + start['x'] * w),
                            int(y + start['y'] * h)
                        )
                        end_pos = (
                            int(x + end['x'] * w),
                            int(y + end['y'] * h)
                        )
                        
                        cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error drawing pose landmarks: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.pose_detector:
                self.pose_detector.close()
            logger.info("Behavior analysis system cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_behavior_rules(self) -> Dict:
        """Get current behavior analysis rules."""
        return {
            'enabled': self.enabled,
            'rules': self.behavior_rules_config.copy(),
            'patterns': self.behavior_patterns.copy()
        }
    
    def update_behavior_rules(self, rules: Dict) -> bool:
        """Update behavior analysis rules."""
        try:
            if 'enabled' in rules:
                self.enabled = rules['enabled']
            
            if 'rules' in rules:
                self.behavior_rules_config.update(rules['rules'])
            
            if 'patterns' in rules:
                self.behavior_patterns.update(rules['patterns'])
            
            # Save to database
            self._save_rules_to_database()
            
            logger.info("Behavior rules updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating behavior rules: {str(e)}")
            return False
    
    def get_recent_behavior_events(self, limit: int = 20, suspicious_only: bool = False) -> List[Dict]:
        """Get recent behavior events."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, person_id, activity_type, confidence, start_time, 
                       end_time, duration_seconds, location_x, location_y, 
                       is_suspicious, notes
                FROM behavior_events
            '''
            
            if suspicious_only:
                query += ' WHERE is_suspicious = 1'
            
            query += ' ORDER BY start_time DESC LIMIT ?'
            
            cursor.execute(query, (limit,))
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    'id': row[0],
                    'person_id': row[1],
                    'activity_type': row[2],
                    'confidence': row[3],
                    'start_time': row[4],
                    'end_time': row[5],
                    'duration_seconds': row[6],
                    'location': {'x': row[7], 'y': row[8]},
                    'is_suspicious': bool(row[9]),
                    'notes': json.loads(row[10]) if row[10] else {}
                })
            
            conn.close()
            return events
            
        except Exception as e:
            logger.error(f"Error getting recent behavior events: {str(e)}")
            return []
    
    def get_activity_patterns(self, time_range: str = '24h') -> Dict:
        """Get activity patterns for analytics."""
        try:
            # Convert time range to hours
            hours = {'1h': 1, '6h': 6, '24h': 24, '7d': 168, '30d': 720}.get(time_range, 24)
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get activity counts by type
            cursor.execute('''
                SELECT activity_type, COUNT(*) as count, 
                       AVG(confidence) as avg_confidence,
                       SUM(CASE WHEN is_suspicious = 1 THEN 1 ELSE 0 END) as suspicious_count
                FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours')
                GROUP BY activity_type
                ORDER BY count DESC
            '''.format(hours))
            
            activity_counts = {}
            for row in cursor.fetchall():
                activity_counts[row[0]] = {
                    'count': row[1],
                    'avg_confidence': row[2],
                    'suspicious_count': row[3]
                }
            
            # Get hourly activity distribution
            cursor.execute('''
                SELECT strftime('%H', start_time) as hour, 
                       activity_type, COUNT(*) as count
                FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours')
                GROUP BY hour, activity_type
                ORDER BY hour
            '''.format(hours))
            
            hourly_distribution = defaultdict(lambda: defaultdict(int))
            for row in cursor.fetchall():
                hourly_distribution[row[0]][row[1]] = row[2]
            
            conn.close()
            
            return {
                'activity_counts': activity_counts,
                'hourly_distribution': dict(hourly_distribution),
                'time_range': time_range,
                'total_events': sum(data['count'] for data in activity_counts.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting activity patterns: {str(e)}")
            return {}
    
    def set_enabled(self, enabled: bool):
        """Enable or disable behavior analysis."""
        self.enabled = enabled
        logger.info(f"Behavior analysis {'enabled' if enabled else 'disabled'}")
    
    def configure_rule(self, rule_name: str, config: Dict) -> bool:
        """Configure a specific behavior rule."""
        try:
            if rule_name in self.behavior_patterns:
                self.behavior_patterns[rule_name].update(config)
            elif rule_name in self.behavior_rules_config:
                if isinstance(self.behavior_rules_config[rule_name], dict):
                    self.behavior_rules_config[rule_name].update(config)
                else:
                    self.behavior_rules_config[rule_name] = config.get('value', self.behavior_rules_config[rule_name])
            else:
                # New rule
                self.behavior_rules_config[rule_name] = config
            
            # Save to database
            self._save_rules_to_database()
            
            logger.info(f"Rule {rule_name} configured successfully")
            return True
        except Exception as e:
            logger.error(f"Error configuring rule {rule_name}: {str(e)}")
            return False
    
    def get_detailed_stats(self, time_range: str = '24h') -> Dict:
        """Get detailed behavior analysis statistics."""
        try:
            hours = {'1h': 1, '6h': 6, '24h': 24, '7d': 168, '30d': 720}.get(time_range, 24)
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Total events
            cursor.execute('''
                SELECT COUNT(*) FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours')
            '''.format(hours))
            total_events = cursor.fetchone()[0]
            
            # Suspicious events
            cursor.execute('''
                SELECT COUNT(*) FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours') AND is_suspicious = 1
            '''.format(hours))
            suspicious_events = cursor.fetchone()[0]
            
            # Unique persons
            cursor.execute('''
                SELECT COUNT(DISTINCT person_id) FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours')
            '''.format(hours))
            unique_persons = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence) FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours')
            '''.format(hours))
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Most common activities
            cursor.execute('''
                SELECT activity_type, COUNT(*) as count
                FROM behavior_events 
                WHERE start_time >= datetime('now', '-{} hours')
                GROUP BY activity_type
                ORDER BY count DESC
                LIMIT 5
            '''.format(hours))
            
            top_activities = []
            for row in cursor.fetchall():
                top_activities.append({'activity': row[0], 'count': row[1]})
            
            conn.close()
            
            base_stats = self.get_behavior_stats()
            
            return {
                **base_stats,
                'time_range': time_range,
                'total_events': total_events,
                'suspicious_events': suspicious_events,
                'unique_persons': unique_persons,
                'avg_confidence': avg_confidence,
                'top_activities': top_activities,
                'enabled': self.enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed stats: {str(e)}")
            return self.get_behavior_stats()
    
    def _save_rules_to_database(self):
        """Save behavior rules to database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create rules table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_rules (
                    id INTEGER PRIMARY KEY,
                    rules_data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Save current rules
            rules_data = json.dumps({
                'enabled': self.enabled,
                'rules_config': self.behavior_rules_config,
                'patterns': self.behavior_patterns
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO behavior_rules (id, rules_data, updated_at)
                VALUES (1, ?, datetime('now'))
            ''', (rules_data,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving rules to database: {str(e)}")
    
    def _load_rules_from_database(self):
        """Load behavior rules from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT rules_data FROM behavior_rules WHERE id = 1')
            result = cursor.fetchone()
            
            if result:
                rules_data = json.loads(result[0])
                self.enabled = rules_data.get('enabled', True)
                self.behavior_rules_config.update(rules_data.get('rules_config', {}))
                self.behavior_patterns.update(rules_data.get('patterns', {}))
                
                logger.info("Behavior rules loaded from database")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading rules from database: {str(e)}")

    # ...existing code...
def main():
    """Example usage of the Behavior Analysis System."""
    # Initialize system
    behavior_analyzer = BehaviorAnalyzer()
    
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available. Please install: pip install mediapipe")
        return
    
    # Example: Process video stream
    cap = cv2.VideoCapture(0)
    frame_number = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simulate person detections (in real usage, this comes from object detector)
            dummy_detections = [
                {
                    'class_name': 'person',
                    'bbox': [100, 100, 200, 400],  # x, y, w, h
                    'confidence': 0.8,
                    'track_id': 'person_1'
                }
            ]
            
            # Analyze behavior
            behavior_results = behavior_analyzer.analyze_behavior(
                frame, dummy_detections, frame_number
            )
            
            # Draw results
            result_frame = behavior_analyzer.draw_behavior_analysis(frame, behavior_results)
            
            # Display
            cv2.imshow('Behavior Analysis', result_frame)
            
            frame_number += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        behavior_analyzer.cleanup()


if __name__ == "__main__":
    main()

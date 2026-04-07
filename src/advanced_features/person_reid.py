#!/usr/bin/env python3
"""
Person Re-Identification Module

This module provides person re-identification capabilities for tracking individuals
across multiple cameras and time periods in surveillance systems.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
import sqlite3
import pickle
import json
from pathlib import Path
import threading
from collections import defaultdict
import hashlib

# Try to import deep learning libraries
try:
    import torch
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch torchvision")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonReID:
    """
    Person Re-Identification system for multi-camera surveillance.
    
    Features:
    - Person feature extraction and matching
    - Cross-camera person tracking
    - Appearance-based person search
    - Person gallery management
    - Similarity scoring and ranking
    - Temporal tracking across sessions
    """
    
    def __init__(self,
                 database_path: str = "person_reid.db",
                 features_path: str = "person_features.pkl",
                 similarity_threshold: float = 0.7,
                 max_gallery_size: int = 1000):
        """
        Initialize the Person Re-ID system.
        
        Args:
            database_path: Path to SQLite database
            features_path: Path to save person features
            similarity_threshold: Threshold for person matching
            max_gallery_size: Maximum number of persons in gallery
        """
        self.database_path = database_path
        self.features_path = features_path
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        
        # Person gallery and features
        self.person_gallery = {}  # person_id -> person data
        self.person_features = {}  # person_id -> feature vector
        self.feature_index = {}   # for fast similarity search
        
        # Tracking data
        self.active_tracks = {}   # track_id -> person_data
        self.track_history = defaultdict(list)  # person_id -> track history
        
        # Camera management
        self.camera_info = {}     # camera_id -> camera metadata
        self.cross_camera_links = defaultdict(list)  # person_id -> camera appearances
        
        # Statistics
        self.reid_stats = {
            'total_persons': 0,
            'successful_matches': 0,
            'new_persons_detected': 0,
            'cross_camera_tracks': 0
        }
        
        # Threading
        self.processing_lock = threading.Lock()
        
        # Initialize components
        self._init_database()
        self._load_person_gallery()
        
        # Feature extractor (simplified for now)
        self.feature_extractor = self._init_feature_extractor()
        
        logger.info("Person Re-ID System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for person re-ID data."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Person gallery table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_gallery (
                    person_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_detections INTEGER DEFAULT 0,
                    cameras_seen TEXT,
                    feature_vector TEXT,
                    appearance_descriptor TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Person tracks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    camera_id TEXT,
                    track_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    bounding_boxes TEXT,
                    confidence_scores TEXT,
                    feature_vector TEXT,
                    appearance_changes TEXT
                )
            ''')
            
            # Cross-camera matches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cross_camera_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    camera1_id TEXT,
                    camera2_id TEXT,
                    match_confidence REAL,
                    time_difference REAL,
                    match_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Camera information table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_info (
                    camera_id TEXT PRIMARY KEY,
                    camera_name TEXT,
                    location TEXT,
                    view_area TEXT,
                    calibration_data TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Person Re-ID database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def _init_feature_extractor(self):
        """Initialize feature extraction model."""
        # Simplified feature extractor using color histograms and basic features
        # In a real implementation, you would use a trained deep learning model
        class SimpleFeatureExtractor:
            def extract_features(self, person_image: np.ndarray) -> np.ndarray:
                """Extract simple appearance features from person image."""
                try:
                    # Resize image to standard size
                    img_resized = cv2.resize(person_image, (64, 128))
                    
                    # Color histogram features
                    hist_features = []
                    for i in range(3):  # BGR channels
                        hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                        hist_features.extend(hist.flatten())
                    
                    # Texture features (simplified)
                    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Local Binary Pattern (simplified)
                    lbp_features = []
                    for y in range(1, gray.shape[0]-1, 4):
                        for x in range(1, gray.shape[1]-1, 4):
                            center = gray[y, x]
                            pattern = 0
                            pattern |= (gray[y-1, x-1] > center) << 0
                            pattern |= (gray[y-1, x] > center) << 1
                            pattern |= (gray[y-1, x+1] > center) << 2
                            pattern |= (gray[y, x+1] > center) << 3
                            pattern |= (gray[y+1, x+1] > center) << 4
                            pattern |= (gray[y+1, x] > center) << 5
                            pattern |= (gray[y+1, x-1] > center) << 6
                            pattern |= (gray[y, x-1] > center) << 7
                            lbp_features.append(pattern)
                    
                    # Combine features
                    features = np.array(hist_features + lbp_features[:100])  # Limit size
                    
                    # Normalize
                    if np.linalg.norm(features) > 0:
                        features = features / np.linalg.norm(features)
                    
                    return features
                    
                except Exception as e:
                    logger.error(f"Error extracting features: {str(e)}")
                    return np.zeros(396)  # 96 (hist) + 300 (lbp subset)
        
        return SimpleFeatureExtractor()
    
    def _load_person_gallery(self):
        """Load person gallery from file."""
        try:
            if Path(self.features_path).exists():
                with open(self.features_path, 'rb') as f:
                    data = pickle.load(f)
                    self.person_gallery = data.get('gallery', {})
                    self.person_features = data.get('features', {})
                
                logger.info(f"Loaded {len(self.person_gallery)} persons from gallery")
            
        except Exception as e:
            logger.error(f"Error loading person gallery: {str(e)}")
    
    def _save_person_gallery(self):
        """Save person gallery to file."""
        try:
            data = {
                'gallery': self.person_gallery,
                'features': self.person_features
            }
            
            with open(self.features_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info("Person gallery saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving person gallery: {str(e)}")
    
    def register_camera(self, camera_id: str, camera_name: str, location: str = ""):
        """Register a new camera in the system."""
        try:
            self.camera_info[camera_id] = {
                'name': camera_name,
                'location': location,
                'registered_time': datetime.now(),
                'active_tracks': {}
            }
            
            # Store in database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO camera_info 
                (camera_id, camera_name, location, status)
                VALUES (?, ?, ?, 'active')
            ''', (camera_id, camera_name, location))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Camera registered: {camera_id} - {camera_name}")
            
        except Exception as e:
            logger.error(f"Error registering camera: {str(e)}")
    
    def process_detections(self, 
                          frame: np.ndarray,
                          person_detections: List[Dict],
                          camera_id: str,
                          timestamp: Optional[datetime] = None) -> List[Dict]:
        """
        Process person detections for re-identification.
        
        Args:
            frame: Input video frame
            person_detections: List of person detection objects
            camera_id: ID of the source camera
            timestamp: Detection timestamp
            
        Returns:
            List of re-identification results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        reid_results = []
        
        try:
            with self.processing_lock:
                for detection in person_detections:
                    if detection.get('class_name') != 'person':
                        continue
                    
                    # Extract person region
                    bbox = detection['bbox']  # [x, y, w, h]
                    person_image = self._extract_person_image(frame, bbox)
                    
                    if person_image is None:
                        continue
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(person_image)
                    
                    # Find matching person in gallery
                    match_result = self._find_matching_person(features, detection, camera_id)
                    
                    # Create re-ID result
                    reid_result = {
                        'detection': detection,
                        'camera_id': camera_id,
                        'timestamp': timestamp.isoformat(),
                        'person_id': match_result['person_id'],
                        'is_new_person': match_result['is_new'],
                        'match_confidence': match_result['confidence'],
                        'features': features,
                        'appearance_hash': self._calculate_appearance_hash(person_image)
                    }
                    
                    # Update tracking data
                    self._update_person_track(reid_result, person_image)
                    
                    reid_results.append(reid_result)
                
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
        
        return reid_results
    
    def _extract_person_image(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract person image from frame using bounding box."""
        try:
            # Handle both [x1, y1, x2, y2] and [x, y, w, h] formats
            if len(bbox) == 4:
                # Check if it's [x1, y1, x2, y2] format (x2 > x1 + reasonable_width)
                x1, y1, x2_or_w, y2_or_h = bbox
                if x2_or_w > x1 + 50:  # Likely [x1, y1, x2, y2] format
                    x, y, x2, y2 = x1, y1, x2_or_w, y2_or_h
                else:  # [x, y, w, h] format
                    x, y, w, h = bbox
                    x2, y2 = x + w, y + h
            else:
                return None
            
            # Ensure valid bounding box
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x or y2 <= y:
                return None
            
            person_image = frame[y:y2, x:x2]
            
            # Minimum size check
            if person_image.shape[0] < 32 or person_image.shape[1] < 16:
                return None
            
            return person_image
            
        except Exception as e:
            logger.error(f"Error extracting person image: {str(e)}")
            return None
    
    def _find_matching_person(self, features: np.ndarray, detection: Dict, camera_id: str) -> Dict:
        """Find matching person in the gallery."""
        try:
            best_match_id = None
            best_similarity = 0.0
            
            # Compare with existing persons
            for person_id, person_features in self.person_features.items():
                similarity = self._calculate_feature_similarity(features, person_features)
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = person_id
            
            if best_match_id:
                # Found matching person
                self.reid_stats['successful_matches'] += 1
                return {
                    'person_id': best_match_id,
                    'confidence': best_similarity,
                    'is_new': False
                }
            else:
                # New person
                new_person_id = self._create_new_person(features, detection, camera_id)
                self.reid_stats['new_persons_detected'] += 1
                return {
                    'person_id': new_person_id,
                    'confidence': 1.0,
                    'is_new': True
                }
                
        except Exception as e:
            logger.error(f"Error finding matching person: {str(e)}")
            return {
                'person_id': f"unknown_{int(datetime.now().timestamp())}",
                'confidence': 0.0,
                'is_new': True
            }
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors."""
        try:
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _create_new_person(self, features: np.ndarray, detection: Dict, camera_id: str) -> str:
        """Create a new person entry in the gallery."""
        try:
            # Generate unique person ID
            person_id = f"person_{len(self.person_gallery):06d}_{int(datetime.now().timestamp())}"
            
            # Create person entry
            person_data = {
                'person_id': person_id,
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'cameras_seen': {camera_id},
                'total_detections': 1,
                'appearance_descriptors': [],
                'track_history': []
            }
            
            # Store in gallery
            self.person_gallery[person_id] = person_data
            self.person_features[person_id] = features
            
            # Manage gallery size
            if len(self.person_gallery) > self.max_gallery_size:
                self._cleanup_old_persons()
            
            # Store in database
            self._store_person_in_database(person_data, features)
            
            logger.info(f"Created new person: {person_id}")
            return person_id
            
        except Exception as e:
            logger.error(f"Error creating new person: {str(e)}")
            return f"error_{int(datetime.now().timestamp())}"
    
    def _update_person_track(self, reid_result: Dict, person_image: np.ndarray):
        """Update person tracking information."""
        try:
            person_id = reid_result['person_id']
            camera_id = reid_result['camera_id']
            
            # Update person data
            if person_id in self.person_gallery:
                person_data = self.person_gallery[person_id]
                person_data['last_seen'] = datetime.now()
                person_data['total_detections'] += 1
                person_data['cameras_seen'].add(camera_id)
                
                # Check for cross-camera tracking
                if len(person_data['cameras_seen']) > 1:
                    self.reid_stats['cross_camera_tracks'] += 1
                    self._record_cross_camera_match(person_id, camera_id)
            
            # Update appearance hash
            appearance_hash = reid_result['appearance_hash']
            
            # Store track information
            track_info = {
                'camera_id': camera_id,
                'timestamp': reid_result['timestamp'],
                'bbox': reid_result['detection']['bbox'],
                'confidence': reid_result['detection']['confidence'],
                'appearance_hash': appearance_hash
            }
            
            self.track_history[person_id].append(track_info)
            
            # Limit track history size
            if len(self.track_history[person_id]) > 100:
                self.track_history[person_id] = self.track_history[person_id][-100:]
            
        except Exception as e:
            logger.error(f"Error updating person track: {str(e)}")
    
    def _calculate_appearance_hash(self, person_image: np.ndarray) -> str:
        """Calculate a hash for person appearance."""
        try:
            # Resize to standard size
            img_resized = cv2.resize(person_image, (32, 64))
            
            # Convert to bytes and hash
            img_bytes = img_resized.tobytes()
            hash_obj = hashlib.md5(img_bytes)
            
            return hash_obj.hexdigest()[:16]  # First 16 characters
            
        except Exception as e:
            logger.error(f"Error calculating appearance hash: {str(e)}")
            return "unknown"
    
    def _record_cross_camera_match(self, person_id: str, camera_id: str):
        """Record cross-camera person match."""
        try:
            person_data = self.person_gallery[person_id]
            cameras_seen = list(person_data['cameras_seen'])
            
            if len(cameras_seen) < 2:
                return
            
            # Record match between current camera and previous cameras
            for prev_camera in cameras_seen:
                if prev_camera != camera_id:
                    conn = sqlite3.connect(self.database_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO cross_camera_matches 
                        (person_id, camera1_id, camera2_id, match_confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (person_id, prev_camera, camera_id, 0.8))  # Default confidence
                    
                    conn.commit()
                    conn.close()
                    break
            
        except Exception as e:
            logger.error(f"Error recording cross-camera match: {str(e)}")
    
    def _store_person_in_database(self, person_data: Dict, features: np.ndarray):
        """Store person data in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO person_gallery 
                (person_id, first_seen, last_seen, total_detections, 
                 cameras_seen, feature_vector, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')
            ''', (
                person_data['person_id'],
                person_data['first_seen'].isoformat(),
                person_data['last_seen'].isoformat(),
                person_data['total_detections'],
                json.dumps(list(person_data['cameras_seen'])),
                json.dumps(features.tolist())
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing person in database: {str(e)}")
    
    def _cleanup_old_persons(self):
        """Remove old persons from gallery to manage memory."""
        try:
            # Sort by last seen time
            persons_by_time = sorted(
                self.person_gallery.items(),
                key=lambda x: x[1]['last_seen']
            )
            
            # Remove oldest 10% of persons
            num_to_remove = len(persons_by_time) // 10
            
            for i in range(num_to_remove):
                person_id = persons_by_time[i][0]
                
                # Remove from memory
                del self.person_gallery[person_id]
                if person_id in self.person_features:
                    del self.person_features[person_id]
                
                # Update database status
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE person_gallery SET status = 'archived' 
                    WHERE person_id = ?
                ''', (person_id,))
                
                conn.commit()
                conn.close()
            
            logger.info(f"Cleaned up {num_to_remove} old persons from gallery")
            
        except Exception as e:
            logger.error(f"Error cleaning up old persons: {str(e)}")
    
    def search_person(self, query_image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar persons in the gallery.
        
        Args:
            query_image: Query person image
            top_k: Number of top matches to return
            
        Returns:
            List of matching persons with similarity scores
        """
        try:
            # Extract features from query image
            query_features = self.feature_extractor.extract_features(query_image)
            
            # Calculate similarities
            similarities = []
            for person_id, person_features in self.person_features.items():
                similarity = self._calculate_feature_similarity(query_features, person_features)
                similarities.append((person_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top matches
            results = []
            for person_id, similarity in similarities[:top_k]:
                person_data = self.person_gallery.get(person_id, {})
                results.append({
                    'person_id': person_id,
                    'similarity': similarity,
                    'total_detections': person_data.get('total_detections', 0),
                    'cameras_seen': list(person_data.get('cameras_seen', [])),
                    'last_seen': person_data.get('last_seen', '').isoformat() if person_data.get('last_seen') else ''
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching person: {str(e)}")
            return []
    
    def get_person_history(self, person_id: str) -> Dict:
        """Get complete history for a person."""
        try:
            person_data = self.person_gallery.get(person_id, {})
            track_history = self.track_history.get(person_id, [])
            
            return {
                'person_id': person_id,
                'person_data': person_data,
                'track_history': track_history,
                'total_tracks': len(track_history),
                'cameras_count': len(person_data.get('cameras_seen', set()))
            }
            
        except Exception as e:
            logger.error(f"Error getting person history: {str(e)}")
            return {}
    
    def get_reid_stats(self) -> Dict:
        """Get re-identification statistics."""
        stats = self.reid_stats.copy()
        stats['total_persons'] = len(self.person_gallery)
        stats['active_cameras'] = len(self.camera_info)
        return stats
    
    def draw_reid_results(self, frame: np.ndarray, reid_results: List[Dict]) -> np.ndarray:
        """
        Draw re-identification results on the frame.
        
        Args:
            frame: Input frame
            reid_results: List of re-ID results
            
        Returns:
            Frame with drawn results
        """
        result_frame = frame.copy()
        
        for result in reid_results:
            detection = result['detection']
            bbox = detection['bbox']
            
            # Handle both [x1, y1, x2, y2] and [x, y, w, h] formats
            if len(bbox) == 4:
                x1, y1, x2_or_w, y2_or_h = bbox
                if x2_or_w > x1 + 50:  # Likely [x1, y1, x2, y2] format
                    x, y, x2, y2 = x1, y1, x2_or_w, y2_or_h
                else:  # [x, y, w, h] format
                    x, y, w, h = bbox
                    x2, y2 = x + w, y + h
            else:
                continue
            
            person_id = result['person_id']
            is_new = result['is_new_person']
            confidence = result['match_confidence']
            
            # Choose color based on person status
            if is_new:
                color = (0, 255, 255)  # Yellow for new persons
            else:
                color = (0, 255, 0)    # Green for known persons
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x2, y2), color, 2)
            
            # Create label
            label = f"ID: {person_id[-8:]}"  # Show last 8 chars of ID
            if not is_new:
                label += f" ({confidence:.2f})"
            else:
                label += " [NEW]"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame,
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         color, -1)
            
            # Draw label text
            cv2.putText(result_frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result_frame
    
    def save_gallery(self):
        """Save current person gallery."""
        self._save_person_gallery()
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self._save_person_gallery()
            logger.info("Person Re-ID system cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Example usage of the Person Re-ID System."""
    # Initialize system
    reid_system = PersonReID()
    
    # Register cameras
    reid_system.register_camera("cam_001", "Main Entrance", "Building A")
    reid_system.register_camera("cam_002", "Corridor", "Building A")
    
    # Example: Process video stream
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simulate person detections (in real usage, this comes from object detector)
            dummy_detections = [
                {
                    'class_name': 'person',
                    'bbox': [100, 100, 150, 300],  # x, y, w, h
                    'confidence': 0.8
                }
            ]
            
            # Process for re-identification
            reid_results = reid_system.process_detections(
                frame, dummy_detections, "cam_001"
            )
            
            # Draw results
            result_frame = reid_system.draw_reid_results(frame, reid_results)
            
            # Display
            cv2.imshow('Person Re-ID', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        reid_system.cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Facial Recognition System

This module provides facial recognition capabilities for surveillance systems,
including face detection, encoding, matching, and person identification.
"""

import cv2
import numpy as np
import logging
import os
import pickle
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime
import sqlite3
import threading
import time

# Try to import face_recognition library
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition library not available. Install with: pip install face_recognition")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialRecognitionSystem:
    """
    Advanced facial recognition system for surveillance applications.
    
    Features:
    - Face detection and encoding
    - Known face database management
    - Real-time face matching
    - Person identification and tracking
    - Face quality assessment
    - Multi-face processing
    """
    
    def __init__(self, 
                 database_path: str = "face_database.db",
                 encodings_path: str = "face_encodings.pkl",
                 known_faces_dir: str = "known_faces",
                 tolerance: float = 0.6,
                 model: str = "hog"):
        """
        Initialize the facial recognition system.
        
        Args:
            database_path: Path to SQLite database for face metadata
            encodings_path: Path to save/load face encodings
            known_faces_dir: Directory containing known face images
            tolerance: Face matching tolerance (lower = stricter)
            model: Face detection model ('hog' or 'cnn')
        """
        self.database_path = database_path
        self.encodings_path = encodings_path
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.model = model
        
        # Face data storage
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = {}
        
        # Performance tracking
        self.recognition_stats = {
            'total_faces_detected': 0,
            'known_faces_identified': 0,
            'unknown_faces_detected': 0,
            'processing_time_avg': 0.0
        }
        
        # Threading for performance
        self.processing_lock = threading.Lock()
        
        # Initialize components
        self._init_database()
        self._load_known_faces()
        
        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("Face recognition not available. Please install face_recognition library.")
            return
            
        logger.info("Facial Recognition System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for face metadata."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    encoding_path TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_detections INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    notes TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT,
                    confidence REAL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    is_known BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encoding_hash TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    detection_count INTEGER DEFAULT 1,
                    sample_image_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Face recognition database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def _load_known_faces(self):
        """Load known face encodings from file and directory."""
        try:
            # Load from pickle file if exists
            if os.path.exists(self.encodings_path):
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_metadata = data.get('metadata', {})
                
                logger.info(f"Loaded {len(self.known_face_encodings)} known face encodings")
                return
            
            # Load from known faces directory
            self.known_faces_dir.mkdir(exist_ok=True)
            
            for person_dir in self.known_faces_dir.iterdir():
                if person_dir.is_dir():
                    person_name = person_dir.name
                    
                    # Process all images in person directory
                    for img_file in person_dir.glob("*.jpg"):
                        self._add_face_from_image(str(img_file), person_name)
            
            # Save encodings
            self._save_encodings()
            
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")
    
    def _add_face_from_image(self, image_path: str, person_name: str):
        """Add a face encoding from an image file."""
        if not FACE_RECOGNITION_AVAILABLE:
            return False
            
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Use the first face found
                encoding = face_encodings[0]
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person_name)
                
                # Store metadata
                self.known_face_metadata[person_name] = {
                    'source_image': image_path,
                    'added_date': datetime.now().isoformat(),
                    'encoding_quality': self._assess_face_quality(image, face_encodings[0])
                }
                
                logger.info(f"Added face encoding for {person_name}")
                return True
            else:
                logger.warning(f"No faces found in {image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding face from {image_path}: {str(e)}")
            return False
    
    def _assess_face_quality(self, image: np.ndarray, encoding: np.ndarray) -> float:
        """Assess the quality of a face encoding."""
        try:
            # Simple quality assessment based on image properties
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Check sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Check brightness
            brightness = np.mean(gray)
            
            # Normalize quality score (0-1)
            quality_score = min(1.0, (sharpness / 1000.0) * (brightness / 255.0) * 2)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing face quality: {str(e)}")
            return 0.5  # Default medium quality
    
    def _save_encodings(self):
        """Save face encodings to pickle file."""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'metadata': self.known_face_metadata
            }
            
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info("Face encodings saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving encodings: {str(e)}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of face detection dictionaries
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []
        
        start_time = time.time()
        detections = []
        
        try:
            with self.processing_lock:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Process each detected face
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                    # Match against known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=self.tolerance
                    )
                    
                    # Find the best match
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        is_known = True
                    else:
                        name = "Unknown"
                        confidence = 0.0
                        is_known = False
                    
                    # Create detection dictionary
                    top, right, bottom, left = face_location
                    detection = {
                        'name': name,
                        'confidence': confidence,
                        'is_known': is_known,
                        'bbox': [left, top, right - left, bottom - top],  # x, y, w, h
                        'face_location': face_location,
                        'face_encoding': face_encoding,
                        'detection_id': f"face_{int(time.time() * 1000)}_{i}",
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    detections.append(detection)
                    
                    # Update statistics
                    self.recognition_stats['total_faces_detected'] += 1
                    if is_known:
                        self.recognition_stats['known_faces_identified'] += 1
                        self._update_person_record(name)
                    else:
                        self.recognition_stats['unknown_faces_detected'] += 1
                        self._handle_unknown_face(face_encoding, detection)
                
                # Update processing time
                processing_time = time.time() - start_time
                self.recognition_stats['processing_time_avg'] = (
                    self.recognition_stats['processing_time_avg'] * 0.9 + 
                    processing_time * 0.1
                )
                
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
        
        return detections
    
    def _update_person_record(self, person_name: str):
        """Update database record for a known person."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE known_faces 
                SET last_seen = CURRENT_TIMESTAMP, 
                    total_detections = total_detections + 1
                WHERE name = ?
            ''', (person_name,))
            
            if cursor.rowcount == 0:
                # Insert new record if person doesn't exist
                cursor.execute('''
                    INSERT INTO known_faces (name, total_detections)
                    VALUES (?, 1)
                ''', (person_name,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating person record: {str(e)}")
    
    def _handle_unknown_face(self, face_encoding: np.ndarray, detection: Dict):
        """Handle detection of an unknown face."""
        try:
            # Create hash for the encoding
            encoding_hash = hash(face_encoding.tobytes())
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check if this unknown face has been seen before
            cursor.execute('''
                SELECT id, detection_count FROM unknown_faces 
                WHERE encoding_hash = ?
            ''', (str(encoding_hash),))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing unknown face record
                cursor.execute('''
                    UPDATE unknown_faces 
                    SET last_seen = CURRENT_TIMESTAMP,
                        detection_count = detection_count + 1
                    WHERE id = ?
                ''', (result[0],))
            else:
                # Insert new unknown face record
                cursor.execute('''
                    INSERT INTO unknown_faces (encoding_hash, detection_count)
                    VALUES (?, 1)
                ''', (str(encoding_hash),))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error handling unknown face: {str(e)}")
    
    def add_known_person(self, name: str, image_path: str) -> bool:
        """
        Add a new known person to the database.
        
        Args:
            name: Person's name
            image_path: Path to a clear image of the person
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add face encoding
            success = self._add_face_from_image(image_path, name)
            
            if success:
                # Update database
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO known_faces 
                    (name, encoding_path, status) 
                    VALUES (?, ?, 'active')
                ''', (name, image_path))
                
                conn.commit()
                conn.close()
                
                # Save encodings
                self._save_encodings()
                
                logger.info(f"Successfully added known person: {name}")
                return True
            
        except Exception as e:
            logger.error(f"Error adding known person {name}: {str(e)}")
        
        return False
    
    def remove_known_person(self, name: str) -> bool:
        """
        Remove a known person from the database.
        
        Args:
            name: Person's name to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from memory
            indices_to_remove = []
            for i, known_name in enumerate(self.known_face_names):
                if known_name == name:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                del self.known_face_encodings[i]
                del self.known_face_names[i]
            
            # Remove metadata
            if name in self.known_face_metadata:
                del self.known_face_metadata[name]
            
            # Update database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE known_faces SET status = 'removed' WHERE name = ?
            ''', (name,))
            
            conn.commit()
            conn.close()
            
            # Save encodings
            self._save_encodings()
            
            logger.info(f"Removed known person: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing known person {name}: {str(e)}")
            return False
    
    def get_recognition_stats(self) -> Dict:
        """Get facial recognition statistics."""
        return self.recognition_stats.copy()
    
    def get_known_persons(self) -> List[str]:
        """Get list of known person names."""
        return list(set(self.known_face_names))
    
    def draw_face_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw face detection results on the frame.
        
        Args:
            frame: Input frame
            detections: List of face detections
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            name = detection['name']
            confidence = detection['confidence']
            is_known = detection['is_known']
            
            # Choose color based on recognition status
            if is_known:
                color = (0, 255, 0)  # Green for known faces
                label = f"{name} ({confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown faces
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, 
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         color, -1)
            
            # Draw label text
            cv2.putText(result_frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Save final encodings
            self._save_encodings()
            logger.info("Facial recognition system cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def enroll_person_from_base64(self, name: str, base64_image: str) -> bool:
        """
        Enroll a new person using base64 encoded image data.
        
        Args:
            name: Person's name
            base64_image: Base64 encoded image string
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import base64
            import tempfile
            
            # Decode base64 image
            if base64_image.startswith('data:image'):
                # Remove data URL prefix if present
                base64_image = base64_image.split(',')[1]
            
            image_data = base64.b64decode(base64_image)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name
            
            try:
                # Use existing add_known_person method
                success = self.add_known_person(name, temp_file_path)
                return success
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error enrolling person from base64: {str(e)}")
            return False


def main():
    """Example usage of the Facial Recognition System."""
    # Initialize system
    face_system = FacialRecognitionSystem()
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition not available. Please install: pip install face_recognition")
        return
    
    # Example: Add a known person
    # face_system.add_known_person("John Doe", "path/to/john_image.jpg")
    
    # Example: Process video stream
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            detections = face_system.detect_faces(frame)
            
            # Draw results
            result_frame = face_system.draw_face_detections(frame, detections)
            
            # Display
            cv2.imshow('Facial Recognition', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_system.cleanup()


if __name__ == "__main__":
    main()

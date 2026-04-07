#!/usr/bin/env python3
"""
Fire and Smoke Detection Module

This module provides real-time fire and smoke detection capabilities
for early warning and emergency response.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sqlite3
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FireSmokeDetector:
    """
    Advanced fire and smoke detection system for surveillance applications.
    
    Features:
    - Real-time fire detection
    - Smoke pattern recognition
    - Spread rate estimation
    - Emergency alert generation
    - Multi-frame verification
    """
    
    def __init__(self,
                 database_path: str = "fire_smoke_detection.db",
                 sensitivity: float = 0.7,
                 verification_frames: int = 5):
        """
        Initialize the fire and smoke detection system.
        
        Args:
            database_path: Path to SQLite database
            sensitivity: Detection sensitivity (0.0-1.0)
            verification_frames: Number of frames for verification
        """
        self.database_path = database_path
        self.sensitivity = sensitivity
        self.verification_frames = verification_frames
        
        # Initialize database
        self._init_database()
        
        # Detection buffers
        self.fire_buffer = deque(maxlen=verification_frames)
        self.smoke_buffer = deque(maxlen=verification_frames)
        
        # Color ranges for fire detection (HSV)
        self.fire_lower = np.array([0, 100, 100])
        self.fire_upper = np.array([35, 255, 255])
        
        # Statistics
        self.stats = {
            'total_fire_detections': 0,
            'total_smoke_detections': 0,
            'false_positives': 0,
            'confirmed_incidents': 0
        }
        
        logger.info("Fire and Smoke Detection System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for fire/smoke events."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Fire/smoke events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fire_smoke_events (
                event_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                event_type TEXT,
                severity TEXT,
                location TEXT,
                spread_rate REAL,
                confidence REAL,
                area_affected REAL,
                image_path TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_fire_smoke(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect fire and smoke in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of fire/smoke detection results
        """
        detections = []
        
        # Detect fire
        fire_detections = self._detect_fire(frame)
        if fire_detections:
            detections.extend(fire_detections)
        
        # Detect smoke
        smoke_detections = self._detect_smoke(frame)
        if smoke_detections:
            detections.extend(smoke_detections)
        
        # Store verified detections
        for detection in detections:
            if detection['verified']:
                self._store_fire_smoke_event(detection)
        
        return detections
    
    def _detect_fire(self, frame: np.ndarray) -> List[Dict]:
        """Detect fire using color-based analysis."""
        detections = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for fire colors
        fire_mask = cv2.inRange(hsv, self.fire_lower, self.fire_upper)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small areas
            if area > 500:  # Minimum fire area
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate fire characteristics
                fire_roi = frame[y:y+h, x:x+w]
                fire_score = self._calculate_fire_score(fire_roi, fire_mask[y:y+h, x:x+w])
                
                if fire_score > self.sensitivity:
                    # Create detection
                    detection = {
                        'type': 'fire',
                        'bbox': [x, y, x+w, y+h],
                        'location': (x + w//2, y + h//2),
                        'area': area,
                        'confidence': fire_score,
                        'timestamp': datetime.now().isoformat(),
                        'verified': False
                    }
                    
                    # Add to buffer for verification
                    self.fire_buffer.append(detection)
                    
                    # Verify if we have enough frames
                    if len(self.fire_buffer) >= self.verification_frames:
                        if self._verify_fire_detection():
                            detection['verified'] = True
                            detection['severity'] = self._assess_fire_severity(area, fire_score)
                            detections.append(detection)
                            self.stats['total_fire_detections'] += 1
        
        return detections
    
    def _detect_smoke(self, frame: np.ndarray) -> List[Dict]:
        """Detect smoke using texture and color analysis."""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect smoke-like regions (grayish, low contrast)
        # This is a simplified approach
        # In production, use trained CNN model for smoke detection
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Calculate difference from mean
        mean_val = np.mean(blurred)
        smoke_mask = cv2.inRange(blurred, mean_val - 30, mean_val + 30)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small areas
            if area > 1000:  # Minimum smoke area
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate smoke characteristics
                smoke_roi = frame[y:y+h, x:x+w]
                smoke_score = self._calculate_smoke_score(smoke_roi)
                
                if smoke_score > self.sensitivity:
                    # Create detection
                    detection = {
                        'type': 'smoke',
                        'bbox': [x, y, x+w, y+h],
                        'location': (x + w//2, y + h//2),
                        'area': area,
                        'confidence': smoke_score,
                        'timestamp': datetime.now().isoformat(),
                        'verified': False
                    }
                    
                    # Add to buffer for verification
                    self.smoke_buffer.append(detection)
                    
                    # Verify if we have enough frames
                    if len(self.smoke_buffer) >= self.verification_frames:
                        if self._verify_smoke_detection():
                            detection['verified'] = True
                            detection['severity'] = self._assess_smoke_severity(area, smoke_score)
                            detections.append(detection)
                            self.stats['total_smoke_detections'] += 1
        
        return detections
    
    def _calculate_fire_score(self, roi: np.ndarray, mask: np.ndarray) -> float:
        """Calculate fire probability score."""
        if roi.size == 0:
            return 0.0
        
        # Calculate percentage of fire-colored pixels
        fire_percentage = np.sum(mask > 0) / mask.size
        
        # Calculate brightness
        brightness = np.mean(roi)
        
        # Combine metrics
        score = (fire_percentage * 0.7) + (min(brightness / 255, 1.0) * 0.3)
        
        return score
    
    def _calculate_smoke_score(self, roi: np.ndarray) -> float:
        """Calculate smoke probability score."""
        if roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance (smoke has low variance)
        variance = np.var(gray)
        
        # Calculate grayness
        mean_color = np.mean(roi, axis=(0, 1))
        color_diff = np.std(mean_color)
        
        # Low variance and low color difference indicate smoke
        texture_score = max(0, 1 - (variance / 1000))
        gray_score = max(0, 1 - (color_diff / 50))
        
        score = (texture_score * 0.6) + (gray_score * 0.4)
        
        return score
    
    def _verify_fire_detection(self) -> bool:
        """Verify fire detection across multiple frames."""
        if len(self.fire_buffer) < self.verification_frames:
            return False
        
        # Check if fire is detected in majority of frames
        verified_count = sum(1 for d in self.fire_buffer if d['confidence'] > self.sensitivity)
        
        return verified_count >= (self.verification_frames * 0.6)
    
    def _verify_smoke_detection(self) -> bool:
        """Verify smoke detection across multiple frames."""
        if len(self.smoke_buffer) < self.verification_frames:
            return False
        
        # Check if smoke is detected in majority of frames
        verified_count = sum(1 for d in self.smoke_buffer if d['confidence'] > self.sensitivity)
        
        return verified_count >= (self.verification_frames * 0.6)
    
    def _assess_fire_severity(self, area: float, confidence: float) -> str:
        """Assess fire severity level."""
        severity_score = (area / 10000) * confidence
        
        if severity_score > 0.8:
            return 'CRITICAL'
        elif severity_score > 0.5:
            return 'HIGH'
        elif severity_score > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_smoke_severity(self, area: float, confidence: float) -> str:
        """Assess smoke severity level."""
        severity_score = (area / 15000) * confidence
        
        if severity_score > 0.7:
            return 'CRITICAL'
        elif severity_score > 0.5:
            return 'HIGH'
        elif severity_score > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _store_fire_smoke_event(self, detection: Dict):
        """Store fire/smoke event in database."""
        try:
            event_id = f"{detection['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fire_smoke_events
                (event_id, timestamp, event_type, severity, location, 
                 confidence, area_affected)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                detection['timestamp'],
                detection['type'],
                detection.get('severity', 'UNKNOWN'),
                str(detection['location']),
                detection['confidence'],
                detection['area']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing fire/smoke event: {str(e)}")
    
    def get_fire_smoke_stats(self) -> Dict:
        """Get fire and smoke detection statistics."""
        return self.stats.copy()
    
    def draw_fire_smoke_detections(self,
                                   frame: np.ndarray,
                                   detections: List[Dict]) -> np.ndarray:
        """
        Draw fire and smoke detection results on the frame.
        
        Args:
            frame: Input frame
            detections: List of fire/smoke detections
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for detection in detections:
            if not detection.get('verified', False):
                continue
            
            bbox = detection['bbox']
            event_type = detection['type']
            severity = detection.get('severity', 'UNKNOWN')
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color based on type and severity
            if event_type == 'fire':
                if severity == 'CRITICAL':
                    color = (0, 0, 255)  # Red
                elif severity == 'HIGH':
                    color = (0, 100, 255)  # Orange
                else:
                    color = (0, 165, 255)  # Light orange
            else:  # smoke
                if severity == 'CRITICAL':
                    color = (128, 128, 128)  # Dark gray
                elif severity == 'HIGH':
                    color = (169, 169, 169)  # Gray
                else:
                    color = (192, 192, 192)  # Light gray
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{event_type.upper()}: {severity}"
            conf_label = f"{confidence:.2%}"
            
            # Background for text
            cv2.rectangle(output, (x1, y1 - 30), (x2, y1), color, -1)
            
            # Draw text
            cv2.putText(output, label, (x1 + 5, y1 - 15),
                       cv2.FONT_HERSHEY_BOLD, 0.6, (255, 255, 255), 2)
            cv2.putText(output, conf_label, (x1 + 5, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw emergency icon
            cv2.circle(output, (x1 - 20, y1 + 20), 15, color, -1)
            cv2.putText(output, "!", (x1 - 27, y1 + 28),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 2)
        
        return output
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Fire and Smoke Detection System cleanup complete")


# Example usage
if __name__ == "__main__":
    detector = FireSmokeDetector()
    print("Fire and Smoke Detection System ready!")
    print(f"Sensitivity: {detector.sensitivity}")
    print(f"Verification frames: {detector.verification_frames}")

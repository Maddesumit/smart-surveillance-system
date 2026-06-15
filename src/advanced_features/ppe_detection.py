#!/usr/bin/env python3
"""
PPE (Personal Protective Equipment) Compliance Detection Module

Detects if workers are wearing required safety equipment:
helmets, vests, masks, gloves, safety glasses.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)


class PPEDetector:
    """
    PPE compliance detection system for workplace safety monitoring.
    
    Uses YOLO object detection to identify safety equipment on persons
    and flag non-compliance violations.
    """
    
    # PPE categories and their detection regions relative to person bbox
    PPE_CATEGORIES = {
        'helmet': {
            'region': 'head',  # top 25% of person bbox
            'required_zones': ['construction', 'warehouse', 'factory'],
            'color_ranges': {
                'yellow': ([20, 100, 100], [30, 255, 255]),
                'white': ([0, 0, 200], [180, 30, 255]),
                'orange': ([10, 100, 100], [20, 255, 255]),
                'blue': ([100, 100, 100], [130, 255, 255])
            },
            'priority': 'high'
        },
        'vest': {
            'region': 'torso',  # middle 50% of person bbox
            'required_zones': ['construction', 'warehouse', 'road'],
            'color_ranges': {
                'yellow': ([20, 100, 100], [35, 255, 255]),
                'orange': ([5, 150, 150], [20, 255, 255])
            },
            'priority': 'high'
        },
        'mask': {
            'region': 'face',  # top 30% of person bbox, center
            'required_zones': ['hospital', 'lab', 'cleanroom'],
            'priority': 'medium'
        },
        'gloves': {
            'region': 'hands',  # lower sides of person bbox
            'required_zones': ['lab', 'hospital', 'factory'],
            'priority': 'low'
        },
        'safety_glasses': {
            'region': 'face',
            'required_zones': ['lab', 'factory', 'workshop'],
            'priority': 'medium'
        }
    }
    
    def __init__(self,
                 database_path: str = "ppe_detection.db",
                 confidence_threshold: float = 0.5,
                 zone_type: str = "construction",
                 required_ppe: Optional[List[str]] = None):
        """
        Initialize PPE detection system.
        
        Args:
            database_path: Path to SQLite database
            confidence_threshold: Minimum confidence for detection
            zone_type: Type of work zone (determines required PPE)
            required_ppe: Override list of required PPE items
        """
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.zone_type = zone_type
        
        # Determine required PPE based on zone
        if required_ppe:
            self.required_ppe = required_ppe
        else:
            self.required_ppe = [
                ppe for ppe, config in self.PPE_CATEGORIES.items()
                if zone_type in config['required_zones']
            ]
        
        # Detection state
        self.person_ppe_status = {}  # person_id -> {ppe_item: detected}
        self.violation_history = defaultdict(list)
        
        # Stats
        self.stats = {
            'total_checks': 0,
            'compliant': 0,
            'violations': 0,
            'by_ppe_type': defaultdict(lambda: {'detected': 0, 'missing': 0})
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"PPE Detection initialized for zone: {zone_type}")
        logger.info(f"Required PPE: {self.required_ppe}")
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppe_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                person_id TEXT,
                missing_ppe TEXT,
                zone_type TEXT,
                confidence REAL,
                bbox TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ppe_compliance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_persons INTEGER,
                compliant_count INTEGER,
                violation_count INTEGER,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_ppe(self,
                   frame: np.ndarray,
                   person_detections: List[Dict]) -> List[Dict]:
        """
        Detect PPE compliance for all detected persons.
        
        Args:
            frame: Input video frame
            person_detections: List of person detections with bbox
            
        Returns:
            List of PPE compliance results per person
        """
        results = []
        
        for person in person_detections:
            if person.get('class_name') != 'person':
                continue
            
            person_id = person.get('track_id', f"person_{len(results)}")
            bbox = person['bbox']
            
            # Check each required PPE item
            ppe_status = {}
            for ppe_item in self.required_ppe:
                detected = self._check_ppe_item(frame, bbox, ppe_item)
                ppe_status[ppe_item] = detected
            
            # Determine compliance
            missing_ppe = [item for item, detected in ppe_status.items() if not detected]
            is_compliant = len(missing_ppe) == 0
            
            # Update stats
            self.stats['total_checks'] += 1
            if is_compliant:
                self.stats['compliant'] += 1
            else:
                self.stats['violations'] += 1
            
            for item, detected in ppe_status.items():
                if detected:
                    self.stats['by_ppe_type'][item]['detected'] += 1
                else:
                    self.stats['by_ppe_type'][item]['missing'] += 1
            
            result = {
                'person_id': person_id,
                'bbox': bbox,
                'ppe_status': ppe_status,
                'is_compliant': is_compliant,
                'missing_ppe': missing_ppe,
                'timestamp': datetime.now().isoformat(),
                'type': 'ppe_violation' if not is_compliant else 'ppe_compliant'
            }
            
            results.append(result)
            
            # Store violation
            if not is_compliant:
                self._store_violation(result)
        
        return results
    
    def _check_ppe_item(self, frame: np.ndarray, bbox: List[int], ppe_item: str) -> bool:
        """
        Check if a specific PPE item is detected on a person.
        
        Uses color-based detection for helmets and vests,
        and region analysis for other items.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        
        if h <= 0 or w <= 0:
            return False
        
        config = self.PPE_CATEGORIES.get(ppe_item, {})
        region = config.get('region', 'full')
        
        # Extract region of interest
        if region == 'head':
            roi = frame[y1:y1 + h // 4, x1:x2]
        elif region == 'torso':
            roi = frame[y1 + h // 4:y1 + 3 * h // 4, x1:x2]
        elif region == 'face':
            roi = frame[y1:y1 + h // 3, x1 + w // 4:x2 - w // 4]
        elif region == 'hands':
            roi = frame[y1 + 2 * h // 3:y2, x1:x2]
        else:
            roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Color-based detection for helmets and vests
        color_ranges = config.get('color_ranges', {})
        if color_ranges:
            return self._detect_by_color(roi, color_ranges)
        
        # For mask detection, check skin exposure in face region
        if ppe_item == 'mask':
            return self._detect_mask(roi)
        
        # Default: assume not detected
        return False
    
    def _detect_by_color(self, roi: np.ndarray, color_ranges: Dict) -> bool:
        """Detect PPE by color in HSV space."""
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels == 0:
            return False
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # If significant portion of region matches PPE color
            color_ratio = cv2.countNonZero(mask) / total_pixels
            if color_ratio > 0.15:  # 15% threshold
                return True
        
        return False
    
    def _detect_mask(self, face_roi: np.ndarray) -> bool:
        """Detect face mask by checking skin exposure."""
        if face_roi.size == 0:
            return False
        
        # Convert to HSV and detect skin color
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        total_pixels = face_roi.shape[0] * face_roi.shape[1]
        if total_pixels == 0:
            return False
        
        skin_ratio = cv2.countNonZero(skin_mask) / total_pixels
        
        # If less skin is visible in lower face, mask is likely present
        # Lower face region
        lower_half = skin_mask[skin_mask.shape[0] // 2:, :]
        lower_pixels = lower_half.shape[0] * lower_half.shape[1]
        
        if lower_pixels == 0:
            return False
        
        lower_skin_ratio = cv2.countNonZero(lower_half) / lower_pixels
        
        # Mask detected if lower face has less skin visible
        return lower_skin_ratio < 0.3
    
    def _store_violation(self, result: Dict):
        """Store PPE violation in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ppe_violations (person_id, missing_ppe, zone_type, confidence, bbox)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                result['person_id'],
                ','.join(result['missing_ppe']),
                self.zone_type,
                self.confidence_threshold,
                str(result['bbox'])
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing PPE violation: {e}")
    
    def draw_ppe_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw PPE detection results on frame."""
        output = frame.copy()
        
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox
            is_compliant = result['is_compliant']
            
            # Green for compliant, red for violation
            color = (0, 255, 0) if is_compliant else (0, 0, 255)
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            if is_compliant:
                label = "PPE: OK"
            else:
                missing = ', '.join(result['missing_ppe'])
                label = f"MISSING: {missing}"
            
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    def get_stats(self) -> Dict:
        """Get PPE detection statistics."""
        return {
            **self.stats,
            'compliance_rate': (self.stats['compliant'] / max(self.stats['total_checks'], 1)) * 100,
            'zone_type': self.zone_type,
            'required_ppe': self.required_ppe
        }
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("PPE Detection System cleanup complete")

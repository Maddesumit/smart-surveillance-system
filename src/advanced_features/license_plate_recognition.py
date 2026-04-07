#!/usr/bin/env python3
"""
License Plate Recognition (LPR/ANPR) Module

This module provides automatic license plate detection and recognition
for vehicle identification and access control.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sqlite3
import re

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateRecognizer:
    """
    Advanced license plate recognition system for surveillance applications.
    
    Features:
    - Real-time plate detection
    - OCR-based plate reading
    - Vehicle tracking integration
    - Whitelist/blacklist checking
    - Multi-country plate format support
    """
    
    def __init__(self,
                 database_path: str = "license_plates.db",
                 confidence_threshold: float = 0.5,
                 languages: List[str] = ['en']):
        """
        Initialize the license plate recognition system.
        
        Args:
            database_path: Path to SQLite database
            confidence_threshold: Minimum confidence for OCR
            languages: List of languages for OCR
        """
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.languages = languages
        
        # Initialize database
        self._init_database()
        
        # Initialize OCR reader
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(languages, gpu=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}")
        
        # Plate detection cascade (using Haar Cascade as fallback)
        self.plate_cascade = None
        self._init_plate_detector()
        
        # Whitelist and blacklist
        self.whitelist = set()
        self.blacklist = set()
        self._load_lists()
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_reads': 0,
            'whitelist_matches': 0,
            'blacklist_matches': 0
        }
        
        logger.info("License Plate Recognition System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for license plates."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # License plate detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plate_detections (
                detection_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                plate_number TEXT,
                confidence REAL,
                vehicle_type TEXT,
                location TEXT,
                camera_id TEXT,
                image_path TEXT,
                in_whitelist BOOLEAN,
                in_blacklist BOOLEAN
            )
        ''')
        
        # Whitelist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS whitelist (
                plate_number TEXT PRIMARY KEY,
                owner_name TEXT,
                vehicle_type TEXT,
                added_date DATETIME,
                notes TEXT
            )
        ''')
        
        # Blacklist table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blacklist (
                plate_number TEXT PRIMARY KEY,
                reason TEXT,
                added_date DATETIME,
                severity TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_plate_detector(self):
        """Initialize license plate detector."""
        # Try to load Haar Cascade for plate detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        try:
            self.plate_cascade = cv2.CascadeClassifier(cascade_path)
            if self.plate_cascade.empty():
                logger.warning("Plate cascade not loaded properly")
                self.plate_cascade = None
        except Exception as e:
            logger.warning(f"Could not load plate cascade: {str(e)}")
    
    def _load_lists(self):
        """Load whitelist and blacklist from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Load whitelist
            cursor.execute("SELECT plate_number FROM whitelist")
            self.whitelist = set(row[0] for row in cursor.fetchall())
            
            # Load blacklist
            cursor.execute("SELECT plate_number FROM blacklist")
            self.blacklist = set(row[0] for row in cursor.fetchall())
            
            conn.close()
            
            logger.info(f"Loaded {len(self.whitelist)} whitelist and {len(self.blacklist)} blacklist entries")
            
        except Exception as e:
            logger.error(f"Error loading lists: {str(e)}")
    
    def detect_and_recognize_plates(self,
                                    frame: np.ndarray,
                                    vehicle_detections: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Detect and recognize license plates in the frame.
        
        Args:
            frame: Input video frame
            vehicle_detections: Optional list of vehicle detections
            
        Returns:
            List of license plate recognition results
        """
        plate_results = []
        
        # Detect license plate regions
        plate_regions = self._detect_plate_regions(frame)
        
        # Recognize text in each plate region
        for region in plate_regions:
            x, y, w, h = region['bbox']
            plate_roi = frame[y:y+h, x:x+w]
            
            # Preprocess plate image
            preprocessed = self._preprocess_plate(plate_roi)
            
            # Recognize plate number
            plate_text, confidence = self._recognize_plate_text(preprocessed)
            
            if plate_text and confidence > self.confidence_threshold:
                # Clean and validate plate number
                plate_number = self._clean_plate_number(plate_text)
                
                if self._validate_plate_format(plate_number):
                    # Create detection result
                    result = {
                        'detection_id': f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        'timestamp': datetime.now().isoformat(),
                        'plate_number': plate_number,
                        'confidence': confidence,
                        'bbox': [x, y, x+w, y+h],
                        'location': (x + w//2, y + h//2),
                        'in_whitelist': plate_number in self.whitelist,
                        'in_blacklist': plate_number in self.blacklist
                    }
                    
                    # Associate with vehicle if available
                    if vehicle_detections:
                        result['vehicle_id'] = self._find_associated_vehicle(
                            result['bbox'], vehicle_detections
                        )
                    
                    plate_results.append(result)
                    
                    # Store in database
                    self._store_plate_detection(result)
                    
                    # Update statistics
                    self.stats['total_detections'] += 1
                    self.stats['successful_reads'] += 1
                    if result['in_whitelist']:
                        self.stats['whitelist_matches'] += 1
                    if result['in_blacklist']:
                        self.stats['blacklist_matches'] += 1
        
        return plate_results
    
    def _detect_plate_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect license plate regions in the frame."""
        regions = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Haar Cascade detection
        if self.plate_cascade is not None:
            plates = self.plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 15)
            )
            for (x, y, w, h) in plates:
                regions.append({'bbox': (x, y, w, h), 'method': 'cascade'})
        
        # Method 2: Contour-based detection (fallback)
        if not regions:
            regions.extend(self._detect_plates_by_contours(gray))
        
        return regions
    
    def _detect_plates_by_contours(self, gray: np.ndarray) -> List[Dict]:
        """Detect plates using contour analysis."""
        regions = []
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # License plates typically have 4 corners
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (plates are wider than tall)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                    regions.append({'bbox': (x, y, w, h), 'method': 'contour'})
        
        return regions
    
    def _preprocess_plate(self, plate_roi: np.ndarray) -> np.ndarray:
        """Preprocess plate image for better OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def _recognize_plate_text(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Recognize text from plate image using OCR."""
        if self.reader is None:
            return "", 0.0
        
        try:
            # Run OCR
            results = self.reader.readtext(plate_image)
            
            if results:
                # Get result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2]
                
                return text, confidence
            
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
        
        return "", 0.0
    
    def _clean_plate_number(self, text: str) -> str:
        """Clean and format plate number."""
        # Remove special characters and spaces
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        return cleaned
    
    def _validate_plate_format(self, plate_number: str) -> bool:
        """Validate plate number format."""
        # Basic validation: 4-10 characters, alphanumeric
        if not plate_number:
            return False
        
        if len(plate_number) < 4 or len(plate_number) > 10:
            return False
        
        # Must contain at least one letter and one number
        has_letter = any(c.isalpha() for c in plate_number)
        has_number = any(c.isdigit() for c in plate_number)
        
        return has_letter and has_number
    
    def _find_associated_vehicle(self,
                                plate_bbox: List[int],
                                vehicle_detections: List[Dict]) -> Optional[str]:
        """Find vehicle associated with detected plate."""
        px1, py1, px2, py2 = plate_bbox
        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)
        
        for vehicle in vehicle_detections:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            
            # Check if plate center is inside vehicle bbox
            if vx1 <= plate_center[0] <= vx2 and vy1 <= plate_center[1] <= vy2:
                return vehicle.get('track_id', 'unknown')
        
        return None
    
    def _store_plate_detection(self, result: Dict):
        """Store plate detection in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO plate_detections
                (detection_id, timestamp, plate_number, confidence, 
                 location, in_whitelist, in_blacklist)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['detection_id'],
                result['timestamp'],
                result['plate_number'],
                result['confidence'],
                str(result['location']),
                result['in_whitelist'],
                result['in_blacklist']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing plate detection: {str(e)}")
    
    def add_to_whitelist(self, plate_number: str, owner_name: str = "", notes: str = ""):
        """Add plate to whitelist."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO whitelist
                (plate_number, owner_name, added_date, notes)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, owner_name, datetime.now().isoformat(), notes))
            
            conn.commit()
            conn.close()
            
            self.whitelist.add(plate_number)
            logger.info(f"Added {plate_number} to whitelist")
            
        except Exception as e:
            logger.error(f"Error adding to whitelist: {str(e)}")
    
    def add_to_blacklist(self, plate_number: str, reason: str = "", severity: str = "MEDIUM"):
        """Add plate to blacklist."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO blacklist
                (plate_number, reason, added_date, severity)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, reason, datetime.now().isoformat(), severity))
            
            conn.commit()
            conn.close()
            
            self.blacklist.add(plate_number)
            logger.info(f"Added {plate_number} to blacklist")
            
        except Exception as e:
            logger.error(f"Error adding to blacklist: {str(e)}")
    
    def get_lpr_stats(self) -> Dict:
        """Get license plate recognition statistics."""
        return self.stats.copy()
    
    def draw_plate_detections(self,
                             frame: np.ndarray,
                             plate_results: List[Dict]) -> np.ndarray:
        """
        Draw license plate detection results on the frame.
        
        Args:
            frame: Input frame
            plate_results: List of plate detection results
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for result in plate_results:
            bbox = result['bbox']
            plate_number = result['plate_number']
            confidence = result['confidence']
            in_whitelist = result['in_whitelist']
            in_blacklist = result['in_blacklist']
            
            x1, y1, x2, y2 = bbox
            
            # Color based on list status
            if in_blacklist:
                color = (0, 0, 255)  # Red
                status = "BLACKLIST"
            elif in_whitelist:
                color = (0, 255, 0)  # Green
                status = "WHITELIST"
            else:
                color = (255, 255, 0)  # Cyan
                status = "DETECTED"
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw plate number
            label = f"{plate_number} ({status})"
            conf_label = f"{confidence:.2%}"
            
            # Background for text
            cv2.rectangle(output, (x1, y1 - 50), (x2, y1), color, -1)
            
            # Draw text
            cv2.putText(output, label, (x1 + 5, y1 - 30),
                       cv2.FONT_HERSHEY_BOLD, 0.6, (255, 255, 255), 2)
            cv2.putText(output, conf_label, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("License Plate Recognition System cleanup complete")


# Example usage
if __name__ == "__main__":
    recognizer = LicensePlateRecognizer()
    print("License Plate Recognition System ready!")
    print(f"OCR available: {EASYOCR_AVAILABLE}")
    print(f"Confidence threshold: {recognizer.confidence_threshold}")

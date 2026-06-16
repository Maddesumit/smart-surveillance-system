#!/usr/bin/env python3
"""
Facial Recognition System

This module provides facial recognition capabilities for surveillance systems,
including face detection, encoding, matching, and person identification.

Backends (auto-selected, preferred first):
  * "opencv" : OpenCV DNN face detection (YuNet) + recognition (SFace).
               Works without dlib, ideal for Apple Silicon / CPU-only setups.
  * "dlib"   : the `face_recognition` library (HOG/CNN + 128-D encodings).

In addition to recognition, the system can automatically capture unique
cropped snapshots of detected faces (deduplicated by embedding) so the
dashboard can show a gallery of everyone it has seen and let the operator
assign a name to any captured face.
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

# Project root (…/smart_surviance) so storage is independent of the CWD.
BASE_DIR = Path(__file__).resolve().parents[2]

# Try to import the dlib-based face_recognition library (optional).
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition (dlib) not available; will use OpenCV backend if models are present.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenCV DNN model locations (downloaded from the OpenCV Zoo).
YUNET_MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")
OPENCV_BACKEND_AVAILABLE = (
    hasattr(cv2, "FaceDetectorYN")
    and hasattr(cv2, "FaceRecognizerSF")
    and os.path.exists(YUNET_MODEL)
    and os.path.exists(SFACE_MODEL)
)

# SFace cosine similarity thresholds (higher = more similar).
SFACE_MATCH_THRESHOLD = 0.363      # recognise as same known identity
SFACE_DEDUP_THRESHOLD = 0.45       # treat as an already-captured face


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return an L2-normalized 1-D float32 vector."""
    vec = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two already-normalized vectors."""
    try:
        return float(np.dot(a, b))
    except Exception:
        return 0.0


class FacialRecognitionSystem:
    """
    Advanced facial recognition system for surveillance applications.

    Features:
    - Face detection and encoding (OpenCV YuNet/SFace or dlib)
    - Known face database management
    - Real-time face matching
    - Automatic unique face snapshot capture + gallery
    - Person identification and tracking
    - Face quality assessment
    - Multi-face processing
    """

    def __init__(self,
                 database_path: str = "face_database.db",
                 encodings_path: str = "face_encodings.pkl",
                 known_faces_dir: str = "known_faces",
                 captured_faces_dir: str = "captured_faces",
                 tolerance: float = 0.6,
                 model: str = "hog"):
        """
        Initialize the facial recognition system.

        All relative paths are resolved against the project root so that every
        component in the process shares the same storage regardless of CWD.
        """
        self.database_path = str(self._resolve(database_path))
        self.encodings_path = str(self._resolve(encodings_path))
        self.known_faces_dir = Path(self._resolve(known_faces_dir))
        self.captured_faces_dir = Path(self._resolve(captured_faces_dir))
        self.tolerance = tolerance
        self.model = model

        # Choose backend: OpenCV preferred (no dlib dependency), then dlib.
        if OPENCV_BACKEND_AVAILABLE:
            self.backend = "opencv"
        elif FACE_RECOGNITION_AVAILABLE:
            self.backend = "dlib"
        else:
            self.backend = None

        # Known-face data storage
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.known_face_metadata: Dict[str, Dict] = {}

        # Captured (gallery) faces: list of dicts with embedding + metadata
        self.captured_faces: List[Dict] = []
        self._capture_counter = 0

        # Performance tracking
        self.recognition_stats = {
            'total_faces_detected': 0,
            'known_faces_identified': 0,
            'unknown_faces_detected': 0,
            'processing_time_avg': 0.0,
            'backend': self.backend or 'none'
        }

        self.processing_lock = threading.Lock()

        # OpenCV DNN handles (created lazily / here when available)
        self._detector = None
        self._recognizer = None
        self._detector_size = (320, 320)
        if self.backend == "opencv":
            self._init_opencv_models()

        # Initialize storage
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.captured_faces_dir.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._load_known_faces()
        self._load_captured_faces()

        if self.backend is None:
            logger.error("No face recognition backend available. Install opencv models or face_recognition.")
        else:
            logger.info(f"Facial Recognition System initialized (backend={self.backend})")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve(path: str) -> Path:
        """Resolve a possibly-relative path against the project root."""
        p = Path(path)
        return p if p.is_absolute() else (BASE_DIR / p)

    def _init_opencv_models(self):
        """Create the YuNet detector and SFace recognizer."""
        try:
            self._detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL, "", self._detector_size, 0.7, 0.3, 5000
            )
            self._recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
            logger.info("OpenCV YuNet + SFace models loaded")
        except Exception as e:
            logger.error(f"Failed to load OpenCV face models: {e}")
            self.backend = "dlib" if FACE_RECOGNITION_AVAILABLE else None

    @property
    def available(self) -> bool:
        return self.backend is not None

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    def _init_database(self):
        """Initialize SQLite database for face metadata."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

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

            # Gallery of automatically captured unique faces
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS captured_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    image_path TEXT,
                    embedding BLOB,
                    is_known INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    detection_count INTEGER DEFAULT 1
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("Face recognition database initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    # ------------------------------------------------------------------ #
    # Known faces loading / persistence
    # ------------------------------------------------------------------ #
    def _load_known_faces(self):
        """Load known face encodings from pickle and/or the known_faces dir."""
        try:
            if os.path.exists(self.encodings_path):
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                # Only reuse encodings produced by the current backend.
                if data.get('backend') == self.backend:
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_metadata = data.get('metadata', {})
                    logger.info(f"Loaded {len(self.known_face_encodings)} known face encodings")
                    if self.known_face_encodings:
                        return
                else:
                    logger.warning("Stored encodings use a different backend; rebuilding from images.")

            # Rebuild from images on disk
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_dir_scan()
            self._save_encodings()
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")

    def known_face_dir_scan(self):
        """Walk known_faces/<person>/*.jpg and add each face."""
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        for person_dir in self.known_faces_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                for img_file in list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")):
                    self._add_face_from_image(str(img_file), person_name)

    def _save_encodings(self):
        """Save face encodings to pickle file (tagged with backend)."""
        try:
            data = {
                'backend': self.backend,
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'metadata': self.known_face_metadata
            }
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info("Face encodings saved successfully")
        except Exception as e:
            logger.error(f"Error saving encodings: {str(e)}")

    # ------------------------------------------------------------------ #
    # Embedding computation (backend specific)
    # ------------------------------------------------------------------ #
    def _detect_opencv(self, frame: np.ndarray):
        """Run YuNet, returning the raw Nx15 face rows for ``frame``."""
        h, w = frame.shape[:2]
        if (w, h) != self._detector_size:
            self._detector_size = (w, h)
            self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def _embed_opencv(self, frame: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
        """Align + embed a single face row into a normalized 128-D vector."""
        try:
            aligned = self._recognizer.alignCrop(frame, face_row)
            feat = self._recognizer.feature(aligned)
            return _normalize(feat)
        except Exception as e:
            logger.error(f"Error computing OpenCV embedding: {e}")
            return None

    def _embed_image_best_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return the embedding for the largest face in an image (enrollment)."""
        if self.backend == "opencv":
            faces = self._detect_opencv(image_bgr)
            if len(faces) == 0:
                return None
            # Pick the largest face (w*h)
            best = max(faces, key=lambda r: float(r[2]) * float(r[3]))
            return self._embed_opencv(image_bgr, best)
        elif self.backend == "dlib":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model=self.model)
            encs = face_recognition.face_encodings(rgb, locs)
            if not encs:
                return None
            return np.asarray(encs[0], dtype=np.float32)
        return None

    def _match_known(self, embedding: np.ndarray) -> Tuple[str, float, bool]:
        """Match an embedding against known faces.

        Returns (name, confidence, is_known).
        """
        if embedding is None or len(self.known_face_encodings) == 0:
            return "Unknown", 0.0, False

        if self.backend == "opencv":
            sims = [_cosine_similarity(embedding, k) for k in self.known_face_encodings]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= SFACE_MATCH_THRESHOLD:
                return self.known_face_names[best_idx], best_sim, True
            return "Unknown", best_sim, False

        # dlib: use face distance / tolerance
        distances = face_recognition.face_distance(self.known_face_encodings, embedding)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        if best_dist <= self.tolerance:
            return self.known_face_names[best_idx], 1.0 - best_dist, True
        return "Unknown", max(0.0, 1.0 - best_dist), False

    def _add_face_from_image(self, image_path: str, person_name: str) -> bool:
        """Add a face encoding from an image file."""
        if not self.available:
            return False
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image {image_path}")
                return False

            embedding = self._embed_image_best_face(image)
            if embedding is None:
                logger.warning(f"No faces found in {image_path}")
                return False

            self.known_face_encodings.append(embedding)
            self.known_face_names.append(person_name)
            self.known_face_metadata[person_name] = {
                'source_image': image_path,
                'added_date': datetime.now().isoformat(),
                'encoding_quality': self._assess_face_quality(image)
            }
            logger.info(f"Added face encoding for {person_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding face from {image_path}: {str(e)}")
            return False

    def _assess_face_quality(self, image: np.ndarray) -> float:
        """Assess the quality of a face image (sharpness * brightness)."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            return float(min(1.0, (sharpness / 1000.0) * (brightness / 255.0) * 2))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------ #
    # Detection / recognition
    # ------------------------------------------------------------------ #
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect and recognize faces in a frame."""
        if not self.available or frame is None:
            return []

        start_time = time.time()
        detections: List[Dict] = []

        try:
            with self.processing_lock:
                if self.backend == "opencv":
                    rows = self._detect_opencv(frame)
                    for i, row in enumerate(rows):
                        x, y, w, h = [int(v) for v in row[0:4]]
                        embedding = self._embed_opencv(frame, row)
                        name, confidence, is_known = self._match_known(embedding)
                        detections.append(self._make_detection(
                            name, confidence, is_known, [x, y, w, h], embedding, i,
                            score=float(row[-1])
                        ))
                else:  # dlib
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb, model=self.model)
                    encs = face_recognition.face_encodings(rgb, locs)
                    for i, (enc, loc) in enumerate(zip(encs, locs)):
                        embedding = np.asarray(enc, dtype=np.float32)
                        name, confidence, is_known = self._match_known(embedding)
                        top, right, bottom, left = loc
                        detections.append(self._make_detection(
                            name, confidence, is_known,
                            [left, top, right - left, bottom - top], embedding, i
                        ))

                # Update stats / records
                for det in detections:
                    self.recognition_stats['total_faces_detected'] += 1
                    if det['is_known']:
                        self.recognition_stats['known_faces_identified'] += 1
                        self._update_person_record(det['name'])
                    else:
                        self.recognition_stats['unknown_faces_detected'] += 1
                        self._handle_unknown_face(det.get('face_encoding'), det)

                processing_time = time.time() - start_time
                self.recognition_stats['processing_time_avg'] = (
                    self.recognition_stats['processing_time_avg'] * 0.9 +
                    processing_time * 0.1
                )
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")

        return detections

    def _make_detection(self, name, confidence, is_known, bbox, embedding, idx, score=None) -> Dict:
        return {
            'name': name,
            'confidence': float(confidence),
            'is_known': bool(is_known),
            'bbox': [int(v) for v in bbox],  # x, y, w, h
            'face_encoding': embedding,
            'detection_score': score,
            'detection_id': f"face_{int(time.time() * 1000)}_{idx}",
            'timestamp': datetime.now().isoformat()
        }

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
                cursor.execute('''
                    INSERT INTO known_faces (name, total_detections)
                    VALUES (?, 1)
                ''', (person_name,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating person record: {str(e)}")

    def _handle_unknown_face(self, face_encoding, detection: Dict):
        """Track recurrence of an unknown face in the database."""
        try:
            if face_encoding is None:
                return
            encoding_hash = hash(np.asarray(face_encoding, dtype=np.float32).round(2).tobytes())
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM unknown_faces WHERE encoding_hash = ?', (str(encoding_hash),))
            result = cursor.fetchone()
            if result:
                cursor.execute('''
                    UPDATE unknown_faces
                    SET last_seen = CURRENT_TIMESTAMP, detection_count = detection_count + 1
                    WHERE id = ?
                ''', (result[0],))
            else:
                cursor.execute('''
                    INSERT INTO unknown_faces (encoding_hash, detection_count) VALUES (?, 1)
                ''', (str(encoding_hash),))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error handling unknown face: {str(e)}")

    # ------------------------------------------------------------------ #
    # Enrollment / known-person management
    # ------------------------------------------------------------------ #
    def add_known_person(self, name: str, image_path: str) -> bool:
        """Add a new known person from an image, persisting the image."""
        try:
            # Copy the source image into known_faces/<name>/ so it survives.
            person_dir = self.known_faces_dir / self._safe_name(name)
            person_dir.mkdir(parents=True, exist_ok=True)
            stored_path = person_dir / f"{self._safe_name(name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    cv2.imwrite(str(stored_path), img)
                else:
                    stored_path = Path(image_path)
            except Exception:
                stored_path = Path(image_path)

            success = self._add_face_from_image(str(stored_path), name)
            if success:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO known_faces (name, encoding_path, status)
                    VALUES (?, ?, 'active')
                ''', (name, str(stored_path)))
                conn.commit()
                conn.close()
                self._save_encodings()
                logger.info(f"Successfully added known person: {name}")
                return True
        except Exception as e:
            logger.error(f"Error adding known person {name}: {str(e)}")
        return False

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name).strip().replace(' ', '_')

    def remove_known_person(self, name: str) -> bool:
        """Remove a known person from memory, disk and the database."""
        try:
            indices = [i for i, n in enumerate(self.known_face_names) if n == name]
            for i in reversed(indices):
                del self.known_face_encodings[i]
                del self.known_face_names[i]
            self.known_face_metadata.pop(name, None)

            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE known_faces SET status = 'removed' WHERE name = ?", (name,))
            conn.commit()
            conn.close()

            # Remove stored images
            person_dir = self.known_faces_dir / self._safe_name(name)
            if person_dir.exists():
                import shutil
                shutil.rmtree(person_dir, ignore_errors=True)

            self._save_encodings()
            logger.info(f"Removed known person: {name}")
            return True
        except Exception as e:
            logger.error(f"Error removing known person {name}: {str(e)}")
            return False

    def get_recognition_stats(self) -> Dict:
        return self.recognition_stats.copy()

    def get_known_persons(self) -> List[str]:
        return sorted(set(self.known_face_names))

    def enroll_person_from_base64(self, name: str, base64_image: str) -> bool:
        """Enroll a new person using base64 encoded image data."""
        try:
            import base64
            import tempfile
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            image_data = base64.b64decode(base64_image)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
                tf.write(image_data)
                temp_path = tf.name
            try:
                return self.add_known_person(name, temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error enrolling person from base64: {str(e)}")
            return False

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw_face_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw face detection results on the frame."""
        result_frame = frame.copy()
        for detection in detections:
            x, y, w, h = detection['bbox']
            name = detection['name']
            confidence = detection['confidence']
            is_known = detection['is_known']

            if is_known:
                color = (0, 255, 0)
                label = f"{name} ({confidence:.2f})"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(result_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return result_frame

    def cleanup(self):
        try:
            self._save_encodings()
            logger.info("Facial recognition system cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    # ------------------------------------------------------------------ #
    # Unique face snapshot capture + gallery
    # ------------------------------------------------------------------ #
    def _load_captured_faces(self):
        """Load captured-face metadata + embeddings from the database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, label, image_path, embedding, is_known, confidence,
                       first_seen, last_seen, detection_count
                FROM captured_faces ORDER BY id
            ''')
            rows = cursor.fetchall()
            conn.close()
            self.captured_faces = []
            for r in rows:
                emb = None
                if r[3] is not None:
                    try:
                        emb = np.frombuffer(r[3], dtype=np.float32)
                    except Exception:
                        emb = None
                self.captured_faces.append({
                    'id': r[0], 'label': r[1], 'image_path': r[2], 'embedding': emb,
                    'is_known': bool(r[4]), 'confidence': r[5],
                    'first_seen': r[6], 'last_seen': r[7], 'detection_count': r[8]
                })
            logger.info(f"Loaded {len(self.captured_faces)} captured faces")
        except Exception as e:
            logger.error(f"Error loading captured faces: {str(e)}")

    def _find_similar_capture(self, embedding: np.ndarray) -> Optional[Dict]:
        """Return an existing captured face matching the embedding, if any."""
        if embedding is None:
            return None
        best, best_sim = None, 0.0
        for cap in self.captured_faces:
            if cap.get('embedding') is None:
                continue
            sim = _cosine_similarity(embedding, cap['embedding'])
            if sim > best_sim:
                best, best_sim = cap, sim
        threshold = SFACE_DEDUP_THRESHOLD if self.backend == 'opencv' else (1.0 - self.tolerance)
        return best if best is not None and best_sim >= threshold else None

    def capture_unique_faces(self, frame: np.ndarray, detections: List[Dict],
                             min_score: float = 0.85, min_size: int = 60) -> List[Dict]:
        """Save cropped snapshots of newly-seen faces (deduplicated).

        Known faces are stored once per identity; unknown faces are stored once
        per distinct embedding. Returns the list of newly captured face dicts.
        """
        if not self.available or frame is None:
            return []
        newly_captured = []
        for det in detections:
            embedding = det.get('face_encoding')
            if embedding is None:
                continue
            x, y, w, h = det['bbox']
            if w < min_size or h < min_size:
                continue
            score = det.get('detection_score')
            if score is not None and score < min_score:
                continue

            is_known = det['is_known']
            name = det['name'] if is_known else 'Unknown'

            # Dedup: known -> by name; unknown -> by embedding similarity
            existing = None
            if is_known:
                existing = next((c for c in self.captured_faces if c['label'] == name), None)
            else:
                existing = self._find_similar_capture(embedding)

            if existing is not None:
                self._touch_capture(existing)
                continue

            crop = self._crop_face(frame, det['bbox'])
            if crop is None:
                continue
            new_cap = self._save_capture(crop, name, embedding, is_known, det['confidence'])
            if new_cap:
                newly_captured.append(new_cap)
        return newly_captured

    def _crop_face(self, frame: np.ndarray, bbox, pad: float = 0.25):
        """Crop a face region with padding, clamped to frame bounds."""
        try:
            x, y, w, h = bbox
            px, py = int(w * pad), int(h * pad)
            x1 = max(0, x - px); y1 = max(0, y - py)
            x2 = min(frame.shape[1], x + w + px); y2 = min(frame.shape[0], y + h + py)
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2].copy()
        except Exception:
            return None

    def _save_capture(self, crop: np.ndarray, label: str, embedding: np.ndarray,
                      is_known: bool, confidence: float) -> Optional[Dict]:
        """Persist a new captured face image + metadata."""
        try:
            self._capture_counter += 1
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{self._safe_name(label)}_{ts}.jpg"
            image_path = self.captured_faces_dir / filename
            cv2.imwrite(str(image_path), crop)

            emb_blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO captured_faces (label, image_path, embedding, is_known, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (label, str(image_path), emb_blob, 1 if is_known else 0, float(confidence)))
            face_id = cursor.lastrowid
            conn.commit()
            conn.close()

            cap = {
                'id': face_id, 'label': label, 'image_path': str(image_path),
                'embedding': np.asarray(embedding, dtype=np.float32) if embedding is not None else None,
                'is_known': is_known, 'confidence': float(confidence),
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(), 'detection_count': 1
            }
            self.captured_faces.append(cap)
            logger.info(f"Captured new face snapshot: {label} ({filename})")
            return cap
        except Exception as e:
            logger.error(f"Error saving captured face: {str(e)}")
            return None

    def _touch_capture(self, cap: Dict):
        """Increment detection count / last_seen for an existing capture."""
        try:
            cap['detection_count'] = cap.get('detection_count', 1) + 1
            cap['last_seen'] = datetime.now().isoformat()
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE captured_faces
                SET last_seen = CURRENT_TIMESTAMP, detection_count = detection_count + 1
                WHERE id = ?
            ''', (cap['id'],))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating captured face: {str(e)}")

    def get_captured_faces(self) -> List[Dict]:
        """Return gallery entries (most recent first) for the dashboard."""
        items = []
        for cap in sorted(self.captured_faces, key=lambda c: c.get('last_seen') or '', reverse=True):
            items.append({
                'id': cap['id'],
                'label': cap['label'],
                'is_known': cap['is_known'],
                'confidence': round(float(cap.get('confidence') or 0), 3),
                'image': os.path.basename(cap['image_path']) if cap.get('image_path') else None,
                'first_seen': cap.get('first_seen'),
                'last_seen': cap.get('last_seen'),
                'detection_count': cap.get('detection_count', 1)
            })
        return items

    def get_captured_face_path(self, filename: str) -> Optional[str]:
        """Return the absolute path of a captured face image by filename."""
        safe = os.path.basename(filename)
        path = self.captured_faces_dir / safe
        return str(path) if path.exists() else None

    def label_captured_face(self, face_id: int, name: str) -> bool:
        """Assign a name to a captured face and enroll it as a known person."""
        cap = next((c for c in self.captured_faces if c['id'] == face_id), None)
        if cap is None:
            logger.warning(f"label_captured_face: id {face_id} not found")
            return False
        try:
            # Enroll using the stored snapshot so the person is recognised live.
            enrolled = False
            if cap.get('image_path') and os.path.exists(cap['image_path']):
                enrolled = self.add_known_person(name, cap['image_path'])
            elif cap.get('embedding') is not None:
                # Fall back to using the stored embedding directly.
                self.known_face_encodings.append(np.asarray(cap['embedding'], dtype=np.float32))
                self.known_face_names.append(name)
                self._save_encodings()
                enrolled = True

            # Update the capture record itself.
            cap['label'] = name
            cap['is_known'] = True
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE captured_faces SET label = ?, is_known = 1 WHERE id = ?', (name, face_id))
            conn.commit()
            conn.close()
            return enrolled
        except Exception as e:
            logger.error(f"Error labeling captured face: {str(e)}")
            return False

    def delete_captured_face(self, face_id: int) -> bool:
        """Remove a captured face snapshot from gallery + disk."""
        cap = next((c for c in self.captured_faces if c['id'] == face_id), None)
        if cap is None:
            return False
        try:
            if cap.get('image_path') and os.path.exists(cap['image_path']):
                os.remove(cap['image_path'])
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM captured_faces WHERE id = ?', (face_id,))
            conn.commit()
            conn.close()
            self.captured_faces = [c for c in self.captured_faces if c['id'] != face_id]
            return True
        except Exception as e:
            logger.error(f"Error deleting captured face: {str(e)}")
            return False


# ---------------------------------------------------------------------- #
# Shared singleton accessor
# ---------------------------------------------------------------------- #
_shared_instance: Optional[FacialRecognitionSystem] = None
_shared_lock = threading.Lock()


def get_facial_recognition_system() -> FacialRecognitionSystem:
    """Return a process-wide shared FacialRecognitionSystem instance.

    Using a single shared instance ensures that faces enrolled through the
    dashboard API are immediately available to the live video pipeline.
    """
    global _shared_instance
    if _shared_instance is None:
        with _shared_lock:
            if _shared_instance is None:
                _shared_instance = FacialRecognitionSystem()
    return _shared_instance


def main():
    """Example usage of the Facial Recognition System."""
    face_system = FacialRecognitionSystem()
    if not face_system.available:
        print("No face recognition backend available.")
        return

    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = face_system.detect_faces(frame)
            face_system.capture_unique_faces(frame, detections)
            result_frame = face_system.draw_face_detections(frame, detections)
            cv2.imshow('Facial Recognition', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_system.cleanup()


if __name__ == "__main__":
    main()

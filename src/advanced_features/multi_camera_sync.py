#!/usr/bin/env python3
"""
Multi-Camera Synchronization Module

This module provides multi-camera management and synchronization capabilities
for comprehensive surveillance coverage and cross-camera tracking.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
import threading
import time
import json
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
import queue
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCameraManager:
    """
    Multi-camera management system for synchronized surveillance.
    
    Features:
    - Multiple camera stream management
    - Camera calibration and synchronization
    - Cross-camera object tracking
    - Coverage area optimization
    - Distributed processing coordination
    - Camera failure detection and recovery
    - Real-time stream aggregation
    """
    
    def __init__(self,
                 database_path: str = "multi_camera.db",
                 sync_tolerance: float = 0.1,  # seconds
                 max_cameras: int = 16):
        """
        Initialize the multi-camera manager.
        
        Args:
            database_path: Path to SQLite database
            sync_tolerance: Time synchronization tolerance
            max_cameras: Maximum number of cameras
        """
        self.database_path = database_path
        self.sync_tolerance = sync_tolerance
        self.max_cameras = max_cameras
        
        # Camera management
        self.cameras = {}  # camera_id -> camera info
        self.camera_streams = {}  # camera_id -> video capture
        self.camera_threads = {}  # camera_id -> processing thread
        self.camera_queues = {}  # camera_id -> frame queue
        
        # Synchronization
        self.sync_master = None  # Master camera for synchronization
        self.time_offsets = {}  # camera_id -> time offset from master
        self.sync_lock = threading.Lock()
        
        # Cross-camera tracking
        self.global_objects = {}  # global_id -> object data across cameras
        self.camera_overlaps = {}  # (cam1, cam2) -> overlap data
        self.handoff_zones = {}  # camera_id -> handoff zone definitions
        
        # Performance monitoring
        self.camera_stats = defaultdict(lambda: {
            'frames_processed': 0,
            'fps': 0.0,
            'latency': 0.0,
            'errors': 0,
            'last_frame_time': None
        })
        
        # System state
        self.is_running = False
        self.master_thread = None
        
        # Initialize components
        self._init_database()
        
        logger.info("Multi-Camera Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for multi-camera data."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Camera configuration table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_config (
                    camera_id TEXT PRIMARY KEY,
                    camera_name TEXT,
                    stream_url TEXT,
                    location TEXT,
                    view_area TEXT,
                    calibration_matrix TEXT,
                    distortion_coeffs TEXT,
                    status TEXT DEFAULT 'inactive',
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP,
                    fps_target REAL DEFAULT 30.0,
                    resolution TEXT DEFAULT '640x480'
                )
            ''')
            
            # Camera synchronization table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_sync (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    master_camera_id TEXT,
                    slave_camera_id TEXT,
                    time_offset REAL,
                    sync_quality REAL,
                    calibration_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Cross-camera objects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cross_camera_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_object_id TEXT,
                    camera_id TEXT,
                    local_object_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    confidence REAL,
                    track_data TEXT
                )
            ''')
            
            # Camera overlap zones table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_overlaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera1_id TEXT,
                    camera2_id TEXT,
                    overlap_area TEXT,
                    transformation_matrix TEXT,
                    overlap_percentage REAL,
                    calibrated BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Multi-camera database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def add_camera(self,
                   camera_id: str,
                   camera_name: str,
                   stream_source,  # Can be int (webcam) or str (IP stream)
                   location: str = "",
                   view_area: Dict = None,
                   fps_target: float = 30.0) -> bool:
        """
        Add a new camera to the system.
        
        Args:
            camera_id: Unique camera identifier
            camera_name: Human-readable camera name
            stream_source: Video source (webcam index or stream URL)
            location: Physical location description
            view_area: Dictionary describing camera view area
            fps_target: Target FPS for this camera
            
        Returns:
            True if camera added successfully
        """
        try:
            if len(self.cameras) >= self.max_cameras:
                logger.error(f"Maximum number of cameras ({self.max_cameras}) reached")
                return False
            
            if camera_id in self.cameras:
                logger.warning(f"Camera {camera_id} already exists")
                return False
            
            # Initialize video capture
            cap = cv2.VideoCapture(stream_source)
            if not cap.isOpened():
                logger.error(f"Failed to open camera stream: {stream_source}")
                return False
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FPS, fps_target)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Get actual camera properties
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create camera configuration
            camera_config = {
                'camera_id': camera_id,
                'name': camera_name,
                'stream_source': stream_source,
                'location': location,
                'view_area': view_area or {},
                'fps_target': fps_target,
                'fps_actual': actual_fps,
                'resolution': (width, height),
                'calibration_matrix': None,
                'distortion_coeffs': None,
                'status': 'active',
                'added_time': datetime.now()
            }
            
            # Store camera info
            self.cameras[camera_id] = camera_config
            self.camera_streams[camera_id] = cap
            self.camera_queues[camera_id] = queue.Queue(maxsize=10)
            
            # Set as master camera if it's the first one
            if self.sync_master is None:
                self.sync_master = camera_id
                self.time_offsets[camera_id] = 0.0
                logger.info(f"Set {camera_id} as sync master camera")
            
            # Store in database
            self._store_camera_config(camera_config)
            
            logger.info(f"Camera added: {camera_id} - {camera_name} ({width}x{height} @ {actual_fps}fps)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera {camera_id}: {str(e)}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the system."""
        try:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False
            
            # Stop camera processing if running
            if camera_id in self.camera_threads:
                self._stop_camera_thread(camera_id)
            
            # Release resources
            if camera_id in self.camera_streams:
                self.camera_streams[camera_id].release()
                del self.camera_streams[camera_id]
            
            if camera_id in self.camera_queues:
                del self.camera_queues[camera_id]
            
            # Remove from tracking
            del self.cameras[camera_id]
            
            # Update sync master if needed
            if self.sync_master == camera_id:
                remaining_cameras = list(self.cameras.keys())
                self.sync_master = remaining_cameras[0] if remaining_cameras else None
                if self.sync_master:
                    self.time_offsets[self.sync_master] = 0.0
                    logger.info(f"Set new sync master: {self.sync_master}")
            
            # Update database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE camera_config SET status = 'removed' WHERE camera_id = ?
            ''', (camera_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Camera removed: {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing camera {camera_id}: {str(e)}")
            return False
    
    def start_synchronized_capture(self):
        """Start synchronized capture from all cameras."""
        try:
            if self.is_running:
                logger.warning("Multi-camera capture already running")
                return
            
            if not self.cameras:
                logger.warning("No cameras available to start")
                return
            
            self.is_running = True
            
            # Start individual camera threads
            for camera_id in self.cameras:
                thread = threading.Thread(
                    target=self._camera_capture_thread,
                    args=(camera_id,),
                    daemon=True
                )
                thread.start()
                self.camera_threads[camera_id] = thread
            
            # Start master synchronization thread
            self.master_thread = threading.Thread(
                target=self._master_sync_thread,
                daemon=True
            )
            self.master_thread.start()
            
            logger.info(f"Started synchronized capture for {len(self.cameras)} cameras")
            
        except Exception as e:
            logger.error(f"Error starting synchronized capture: {str(e)}")
            self.is_running = False
    
    def stop_synchronized_capture(self):
        """Stop synchronized capture from all cameras."""
        try:
            self.is_running = False
            
            # Stop all camera threads
            for camera_id in list(self.camera_threads.keys()):
                self._stop_camera_thread(camera_id)
            
            # Clear queues
            for camera_queue in self.camera_queues.values():
                while not camera_queue.empty():
                    try:
                        camera_queue.get_nowait()
                    except queue.Empty:
                        break
            
            logger.info("Stopped synchronized capture")
            
        except Exception as e:
            logger.error(f"Error stopping synchronized capture: {str(e)}")
    
    def _camera_capture_thread(self, camera_id: str):
        """Individual camera capture thread."""
        try:
            cap = self.camera_streams[camera_id]
            camera_queue = self.camera_queues[camera_id]
            stats = self.camera_stats[camera_id]
            
            frame_count = 0
            start_time = time.time()
            
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    stats['errors'] += 1
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Add timestamp
                timestamp = time.time()
                frame_data = {
                    'frame': frame,
                    'timestamp': timestamp,
                    'camera_id': camera_id,
                    'frame_number': frame_count
                }
                
                # Add to queue (non-blocking)
                try:
                    camera_queue.put_nowait(frame_data)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        camera_queue.get_nowait()
                        camera_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
                # Update statistics
                frame_count += 1
                stats['frames_processed'] = frame_count
                stats['last_frame_time'] = timestamp
                
                # Calculate FPS
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    stats['fps'] = frame_count / elapsed
                
                # Control frame rate
                target_fps = self.cameras[camera_id]['fps_target']
                if target_fps > 0:
                    time.sleep(max(0, 1.0/target_fps - (time.time() - timestamp)))
            
        except Exception as e:
            logger.error(f"Error in camera thread {camera_id}: {str(e)}")
            self.camera_stats[camera_id]['errors'] += 1
    
    def _stop_camera_thread(self, camera_id: str):
        """Stop individual camera thread."""
        try:
            if camera_id in self.camera_threads:
                thread = self.camera_threads[camera_id]
                if thread.is_alive():
                    # Thread will stop when is_running becomes False
                    thread.join(timeout=2.0)
                del self.camera_threads[camera_id]
        except Exception as e:
            logger.error(f"Error stopping camera thread {camera_id}: {str(e)}")
    
    def _master_sync_thread(self):
        """Master synchronization thread."""
        try:
            while self.is_running:
                if self.sync_master and len(self.cameras) > 1:
                    self._synchronize_cameras()
                
                time.sleep(1.0)  # Sync check interval
                
        except Exception as e:
            logger.error(f"Error in master sync thread: {str(e)}")
    
    def _synchronize_cameras(self):
        """Synchronize all cameras with the master camera."""
        try:
            with self.sync_lock:
                master_time = self.camera_stats[self.sync_master].get('last_frame_time')
                
                if master_time is None:
                    return
                
                for camera_id in self.cameras:
                    if camera_id == self.sync_master:
                        continue
                    
                    camera_time = self.camera_stats[camera_id].get('last_frame_time')
                    
                    if camera_time is not None:
                        # Calculate time offset
                        offset = camera_time - master_time
                        self.time_offsets[camera_id] = offset
                        
                        # Calculate latency
                        current_time = time.time()
                        latency = current_time - camera_time
                        self.camera_stats[camera_id]['latency'] = latency
            
        except Exception as e:
            logger.error(f"Error synchronizing cameras: {str(e)}")
    
    def get_synchronized_frames(self, timeout: float = 0.1) -> Dict[str, Dict]:
        """
        Get synchronized frames from all cameras.
        
        Args:
            timeout: Timeout for frame retrieval
            
        Returns:
            Dictionary of camera_id -> frame_data
        """
        synchronized_frames = {}
        
        try:
            # Get frames from all cameras
            for camera_id, camera_queue in self.camera_queues.items():
                try:
                    frame_data = camera_queue.get(timeout=timeout)
                    synchronized_frames[camera_id] = frame_data
                except queue.Empty:
                    # No frame available within timeout
                    continue
            
            # Apply time synchronization corrections
            if self.sync_master and len(synchronized_frames) > 1:
                master_frame = synchronized_frames.get(self.sync_master)
                if master_frame:
                    master_timestamp = master_frame['timestamp']
                    
                    # Filter frames within sync tolerance
                    synced_frames = {}
                    for camera_id, frame_data in synchronized_frames.items():
                        offset = self.time_offsets.get(camera_id, 0.0)
                        adjusted_timestamp = frame_data['timestamp'] - offset
                        
                        time_diff = abs(adjusted_timestamp - master_timestamp)
                        if time_diff <= self.sync_tolerance:
                            synced_frames[camera_id] = frame_data
                    
                    return synced_frames
            
            return synchronized_frames
            
        except Exception as e:
            logger.error(f"Error getting synchronized frames: {str(e)}")
            return {}
    
    def calibrate_camera_overlap(self, camera1_id: str, camera2_id: str) -> bool:
        """
        Calibrate overlap between two cameras.
        
        Args:
            camera1_id: First camera ID
            camera2_id: Second camera ID
            
        Returns:
            True if calibration successful
        """
        try:
            if camera1_id not in self.cameras or camera2_id not in self.cameras:
                logger.error("Both cameras must be registered")
                return False
            
            # Get synchronized frames from both cameras
            frames = self.get_synchronized_frames(timeout=1.0)
            
            if camera1_id not in frames or camera2_id not in frames:
                logger.error("Could not get frames from both cameras")
                return False
            
            frame1 = frames[camera1_id]['frame']
            frame2 = frames[camera2_id]['frame']
            
            # Detect overlap using feature matching
            overlap_data = self._detect_camera_overlap(frame1, frame2)
            
            if overlap_data['overlap_percentage'] > 0.1:  # At least 10% overlap
                # Store overlap information
                self.camera_overlaps[(camera1_id, camera2_id)] = overlap_data
                
                # Store in database
                self._store_camera_overlap(camera1_id, camera2_id, overlap_data)
                
                logger.info(f"Calibrated overlap between {camera1_id} and {camera2_id}: "
                           f"{overlap_data['overlap_percentage']:.1%}")
                return True
            else:
                logger.warning(f"No significant overlap found between {camera1_id} and {camera2_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error calibrating camera overlap: {str(e)}")
            return False
    
    def _detect_camera_overlap(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict:
        """Detect overlap between two camera views using feature matching."""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and descriptors
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return {'overlap_percentage': 0.0, 'transformation_matrix': None}
            
            # Match features
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            
            # Filter good matches
            good_matches = [m for m in matches if m.distance < 50]
            
            overlap_percentage = 0.0
            transformation_matrix = None
            
            if len(good_matches) > 10:
                # Extract matched points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Calculate overlap area
                    h1, w1 = frame1.shape[:2]
                    h2, w2 = frame2.shape[:2]
                    
                    # Transform corners of frame1 to frame2 space
                    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
                    corners1_transformed = cv2.perspectiveTransform(corners1, H)
                    
                    # Calculate intersection area
                    # This is a simplified calculation
                    overlap_percentage = len(good_matches) / max(len(kp1), len(kp2))
                    transformation_matrix = H.tolist()
            
            return {
                'overlap_percentage': min(overlap_percentage, 1.0),
                'transformation_matrix': transformation_matrix,
                'feature_matches': len(good_matches),
                'total_features_1': len(kp1),
                'total_features_2': len(kp2)
            }
            
        except Exception as e:
            logger.error(f"Error detecting camera overlap: {str(e)}")
            return {'overlap_percentage': 0.0, 'transformation_matrix': None}
    
    def track_across_cameras(self, 
                           object_detections: Dict[str, List[Dict]],
                           timestamp: Optional[datetime] = None) -> Dict[str, List[Dict]]:
        """
        Track objects across multiple cameras.
        
        Args:
            object_detections: Dictionary of camera_id -> list of detections
            timestamp: Detection timestamp
            
        Returns:
            Dictionary of global tracking results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        cross_camera_tracks = {}
        
        try:
            # Process each camera's detections
            for camera_id, detections in object_detections.items():
                camera_tracks = []
                
                for detection in detections:
                    # Check if this detection matches any global object
                    global_id = self._match_to_global_object(detection, camera_id)
                    
                    if global_id is None:
                        # Create new global object
                        global_id = self._create_global_object(detection, camera_id, timestamp)
                    else:
                        # Update existing global object
                        self._update_global_object(global_id, detection, camera_id, timestamp)
                    
                    # Add to camera tracks
                    track_result = {
                        'local_detection': detection,
                        'global_id': global_id,
                        'camera_id': camera_id,
                        'timestamp': timestamp.isoformat()
                    }
                    camera_tracks.append(track_result)
                
                cross_camera_tracks[camera_id] = camera_tracks
            
            # Clean up old global objects
            self._cleanup_old_global_objects()
            
            return cross_camera_tracks
            
        except Exception as e:
            logger.error(f"Error tracking across cameras: {str(e)}")
            return {}
    
    def _match_to_global_object(self, detection: Dict, camera_id: str) -> Optional[str]:
        """Match detection to existing global object."""
        try:
            # Simple matching based on spatial proximity and time
            detection_center = self._get_detection_center(detection)
            
            for global_id, global_obj in self.global_objects.items():
                # Check if object was recently seen in nearby cameras
                for cam_id, cam_data in global_obj['camera_tracks'].items():
                    if cam_id == camera_id:
                        # Same camera - check spatial proximity
                        last_detection = cam_data['detections'][-1] if cam_data['detections'] else None
                        if last_detection:
                            last_center = self._get_detection_center(last_detection)
                            distance = np.linalg.norm(np.array(detection_center) - np.array(last_center))
                            
                            # Simple distance threshold
                            if distance < 100:  # pixels
                                return global_id
                    else:
                        # Different camera - check if cameras have overlap
                        overlap_key = (cam_id, camera_id)
                        reverse_overlap_key = (camera_id, cam_id)
                        
                        if overlap_key in self.camera_overlaps or reverse_overlap_key in self.camera_overlaps:
                            # Cameras have overlap - use transformation to match
                            if self._check_cross_camera_match(detection, cam_data, camera_id, cam_id):
                                return global_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching to global object: {str(e)}")
            return None
    
    def _get_detection_center(self, detection: Dict) -> Tuple[float, float]:
        """Get center point of detection bounding box."""
        bbox = detection.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
        x, y, w, h = bbox
        return (x + w/2, y + h/2)
    
    def _check_cross_camera_match(self, detection: Dict, cam_data: Dict, 
                                 camera1_id: str, camera2_id: str) -> bool:
        """Check if detection matches across cameras using transformation."""
        try:
            overlap_key = (camera2_id, camera1_id)
            if overlap_key not in self.camera_overlaps:
                overlap_key = (camera1_id, camera2_id)
            
            if overlap_key not in self.camera_overlaps:
                return False
            
            overlap_data = self.camera_overlaps[overlap_key]
            transformation_matrix = overlap_data.get('transformation_matrix')
            
            if transformation_matrix is None:
                return False
            
            # Transform detection coordinates
            detection_center = self._get_detection_center(detection)
            point = np.array([[detection_center]], dtype=np.float32)
            
            H = np.array(transformation_matrix)
            transformed_point = cv2.perspectiveTransform(point, H)
            
            # Compare with last detection in other camera
            if cam_data['detections']:
                last_detection = cam_data['detections'][-1]
                last_center = self._get_detection_center(last_detection)
                
                distance = np.linalg.norm(transformed_point[0][0] - np.array(last_center))
                
                # Threshold for cross-camera matching
                return distance < 150  # pixels
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cross-camera match: {str(e)}")
            return False
    
    def _create_global_object(self, detection: Dict, camera_id: str, timestamp: datetime) -> str:
        """Create new global object."""
        try:
            global_id = f"global_{len(self.global_objects):06d}_{int(timestamp.timestamp())}"
            
            global_object = {
                'global_id': global_id,
                'created_time': timestamp,
                'last_updated': timestamp,
                'camera_tracks': {
                    camera_id: {
                        'detections': [detection],
                        'first_seen': timestamp,
                        'last_seen': timestamp
                    }
                },
                'object_class': detection.get('class_name', 'unknown'),
                'total_detections': 1
            }
            
            self.global_objects[global_id] = global_object
            
            return global_id
            
        except Exception as e:
            logger.error(f"Error creating global object: {str(e)}")
            return f"error_{int(timestamp.timestamp())}"
    
    def _update_global_object(self, global_id: str, detection: Dict, 
                            camera_id: str, timestamp: datetime):
        """Update existing global object."""
        try:
            if global_id not in self.global_objects:
                return
            
            global_obj = self.global_objects[global_id]
            global_obj['last_updated'] = timestamp
            global_obj['total_detections'] += 1
            
            if camera_id not in global_obj['camera_tracks']:
                global_obj['camera_tracks'][camera_id] = {
                    'detections': [],
                    'first_seen': timestamp,
                    'last_seen': timestamp
                }
            
            cam_track = global_obj['camera_tracks'][camera_id]
            cam_track['detections'].append(detection)
            cam_track['last_seen'] = timestamp
            
            # Limit detection history
            if len(cam_track['detections']) > 50:
                cam_track['detections'] = cam_track['detections'][-50:]
            
        except Exception as e:
            logger.error(f"Error updating global object: {str(e)}")
    
    def _cleanup_old_global_objects(self):
        """Remove old global objects to manage memory."""
        try:
            current_time = datetime.now()
            max_age = timedelta(hours=1)  # Objects older than 1 hour
            
            objects_to_remove = []
            for global_id, global_obj in self.global_objects.items():
                age = current_time - global_obj['last_updated']
                if age > max_age:
                    objects_to_remove.append(global_id)
            
            for global_id in objects_to_remove:
                del self.global_objects[global_id]
            
            if objects_to_remove:
                logger.info(f"Cleaned up {len(objects_to_remove)} old global objects")
                
        except Exception as e:
            logger.error(f"Error cleaning up global objects: {str(e)}")
    
    def _store_camera_config(self, camera_config: Dict):
        """Store camera configuration in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO camera_config 
                (camera_id, camera_name, stream_url, location, view_area, 
                 status, fps_target, resolution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                camera_config['camera_id'],
                camera_config['name'],
                str(camera_config['stream_source']),
                camera_config['location'],
                json.dumps(camera_config['view_area']),
                camera_config['status'],
                camera_config['fps_target'],
                f"{camera_config['resolution'][0]}x{camera_config['resolution'][1]}"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing camera config: {str(e)}")
    
    def _store_camera_overlap(self, camera1_id: str, camera2_id: str, overlap_data: Dict):
        """Store camera overlap data in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO camera_overlaps 
                (camera1_id, camera2_id, overlap_area, transformation_matrix, 
                 overlap_percentage, calibrated)
                VALUES (?, ?, ?, ?, ?, TRUE)
            ''', (
                camera1_id,
                camera2_id,
                json.dumps(overlap_data),
                json.dumps(overlap_data.get('transformation_matrix')),
                overlap_data['overlap_percentage']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing camera overlap: {str(e)}")
    
    def get_camera_stats(self) -> Dict:
        """Get statistics for all cameras."""
        return {
            'total_cameras': len(self.cameras),
            'active_cameras': len([c for c in self.cameras.values() if c['status'] == 'active']),
            'sync_master': self.sync_master,
            'camera_details': dict(self.camera_stats),
            'global_objects': len(self.global_objects),
            'camera_overlaps': len(self.camera_overlaps)
        }
    
    def get_camera_list(self) -> List[Dict]:
        """Get list of all cameras with their details."""
        camera_list = []
        for camera_id, config in self.cameras.items():
            stats = self.camera_stats[camera_id]
            camera_info = {
                'camera_id': camera_id,
                'name': config['name'],
                'location': config['location'],
                'resolution': config['resolution'],
                'fps_target': config['fps_target'],
                'fps_actual': stats['fps'],
                'status': config['status'],
                'frames_processed': stats['frames_processed'],
                'latency': stats['latency'],
                'errors': stats['errors']
            }
            camera_list.append(camera_info)
        
        return camera_list
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.stop_synchronized_capture()
            
            # Release all camera streams
            for cap in self.camera_streams.values():
                cap.release()
            
            logger.info("Multi-camera manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Example usage of the Multi-Camera Manager."""
    # Initialize system
    camera_manager = MultiCameraManager()
    
    # Add cameras (assuming webcams are available)
    camera_manager.add_camera("cam_001", "Main Camera", 0, "Main entrance")
    # camera_manager.add_camera("cam_002", "Side Camera", 1, "Side entrance")
    
    # Start synchronized capture
    camera_manager.start_synchronized_capture()
    
    try:
        # Process frames
        for i in range(100):  # Process 100 frame sets
            frames = camera_manager.get_synchronized_frames()
            
            if frames:
                print(f"Frame set {i}: {len(frames)} cameras")
                
                # Display frames (if available)
                for camera_id, frame_data in frames.items():
                    frame = frame_data['frame']
                    cv2.imshow(f"Camera {camera_id}", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)
    
    finally:
        camera_manager.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

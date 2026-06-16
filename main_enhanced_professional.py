#!/usr/bin/env python3
"""
# Import advanced features
try:
    from advanced_features.facial_recognition import FacialRecognitionSystem
    from advanced_features.behavior_analysis import BehaviorAnalyzer
    from advanced_features.person_reid import PersonReID
    from advanced_features.multi_camera_sync import MultiCameraManager
    from advanced_features.real_time_analytics import AnalyticsEngine
    from advanced_features.advanced_alerts import AdvancedAlertSystem
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Import dashboard components
try:
    from dashboard import create_app, socketio
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dashboard not available: {e}")
    DASHBOARD_AVAILABLE = False
    socketio = Nonert Surveillance System - Main Application with Professional Dashboard

This enhanced version integrates all advanced features with a professional dashboard
and standardized alert system to reduce noise and improve monitoring.
"""

import os
import sys
import time
import logging
import signal
import threading
from datetime import datetime
import json

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core components
from video_processing.video_stream import VideoStream
from object_detection.detector import ObjectDetector
from object_detection.tracker import ObjectTracker
from anomaly_detection.analyzer import AnomalyDetector
from alert_system.notifier import AlertSystem

# Import advanced features
try:
    from advanced_features.facial_recognition import FacialRecognitionSystem, get_facial_recognition_system
    from advanced_features.behavior_analysis import BehaviorAnalyzer
    from advanced_features.person_reid import PersonReID
    from advanced_features.multi_camera_sync import MultiCameraManager
    from advanced_features.real_time_analytics import AnalyticsEngine
    from advanced_features.advanced_alerts import AdvancedAlertSystem
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Import dashboard
from dashboard import create_app, socketio

# Set up enhanced logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'surveillance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('enhanced_smart_surveillance')

# Global system state
running = True
system_components = {}
dashboard_app = None
dashboard_thread = None
selected_camera_index = 0  # Default camera

def enumerate_cameras(max_cameras=10):
    """
    Enumerate all available cameras on the system.
    
    Returns:
        list: List of dicts with camera info (index, name, working status)
    """
    import cv2
    available_cameras = []
    
    print("\n🔍 Scanning for available cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify the camera works
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Try to get camera name (backend dependent)
                backend = cap.getBackendName()
                
                camera_info = {
                    'index': i,
                    'resolution': f"{width}x{height}",
                    'fps': fps if fps > 0 else "Unknown",
                    'backend': backend,
                    'working': True
                }
                available_cameras.append(camera_info)
                print(f"  ✅ Found camera {i}: {width}x{height} @ {fps}fps ({backend})")
            cap.release()
    
    if not available_cameras:
        print("  ❌ No working cameras found!")
    
    return available_cameras

def select_camera():
    """
    Interactive camera selection at startup.
    
    Returns:
        int: Selected camera index
    """
    global selected_camera_index
    
    print("\n" + "="*60)
    print("📹 CAMERA SELECTION")
    print("="*60)
    
    cameras = enumerate_cameras()
    
    if not cameras:
        print("\n⚠️  No cameras detected. Using default (index 0).")
        return 0
    
    if len(cameras) == 1:
        print(f"\n✅ Only one camera found. Using camera {cameras[0]['index']}.")
        selected_camera_index = cameras[0]['index']
        return cameras[0]['index']
    
    # Display available cameras
    print("\n📋 Available Cameras:")
    print("-" * 50)
    for cam in cameras:
        print(f"  [{cam['index']}] Camera {cam['index']}: {cam['resolution']} @ {cam['fps']}fps")
        print(f"      Backend: {cam['backend']}")
    print("-" * 50)
    
    # Get user selection
    while True:
        try:
            choice = input(f"\n👉 Select camera index (0-{len(cameras)-1}) [default: 0]: ").strip()
            
            if choice == "":
                selected_camera_index = cameras[0]['index']
                break
            
            choice = int(choice)
            valid_indices = [cam['index'] for cam in cameras]
            
            if choice in valid_indices:
                selected_camera_index = choice
                break
            else:
                print(f"❌ Invalid choice. Please select from: {valid_indices}")
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Selection cancelled. Using default camera (index 0).")
            selected_camera_index = 0
            break
    
    print(f"\n✅ Selected camera: {selected_camera_index}")
    return selected_camera_index

# Global flags
DASHBOARD_AVAILABLE = False

# Shared data store for real-time statistics
import threading
shared_stats_lock = threading.Lock()
shared_stats = {
    'system_start_time': datetime.now(),
    'objects_detected_today': 0,
    'faces_recognized_today': 0,
    'alerts_generated_today': 0,
    'live_stats': {
        'objects_in_view': 0,
        'faces_detected': 0,
        'tracked_objects': 0
    },
    'last_update': datetime.now(),
    'detection_sessions': {
        'person_detections': set(),  # Track unique detection sessions
        'face_recognitions': set(),  # Track unique recognition sessions
        'last_person_detection': None,
        'last_face_recognition': None
    }
}

def update_shared_stats(key, value=None, increment=False):
    """Thread-safe update of shared statistics."""
    global shared_stats
    with shared_stats_lock:
        if increment and key in shared_stats:
            shared_stats[key] += 1
        elif not increment and value is not None:
            if '.' in key:  # nested key like 'live_stats.objects_in_view'
                keys = key.split('.')
                current = shared_stats
                for k in keys[:-1]:
                    current = current[k]
                current[keys[-1]] = value
            else:
                shared_stats[key] = value
        shared_stats['last_update'] = datetime.now()

def get_shared_stats():
    """Thread-safe access to shared statistics."""
    with shared_stats_lock:
        return shared_stats.copy()

class StandardizedAlertManager:
    """Enhanced alert management with reduced noise and better categorization."""
    
    def __init__(self):
        self.alert_history = []
        self.alert_stats = {
            'total': 0,
            'high_priority': 0,
            'medium_priority': 0,
            'low_priority': 0,
            'types': {}
        }
        self.last_alerts = {}
        self.throttle_times = {
            'unknown_person': 30,  # 30 seconds
            'object_detection': 10,  # 10 seconds
            'suspicious_behavior': 20,  # 20 seconds
            'intrusion': 5,  # 5 seconds (critical)
            'restricted_area_violation': 180,  # 3 minutes (increased to reduce spam)
            'unattended_object': 60,  # 1 minute
            'system_status': 60  # 1 minute
        }
    
    def create_alert(self, alert_type, message, priority='medium', data=None):
        """Create a standardized alert with intelligent throttling."""
        current_time = datetime.now()
        
        # Check throttling
        throttle_time = self.throttle_times.get(alert_type, 15)
        alert_key = f"{alert_type}_{message[:50]}"
        
        if alert_key in self.last_alerts:
            time_diff = (current_time - self.last_alerts[alert_key]).total_seconds()
            if time_diff < throttle_time:
                return None  # Throttled
        
        self.last_alerts[alert_key] = current_time
        
        # Create alert
        alert = {
            'id': f"alert_{int(current_time.timestamp())}",
            'type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': current_time.isoformat(),
            'data': data or {}
        }
        
        # Update statistics
        self.alert_stats['total'] += 1
        self.alert_stats[f'{priority}_priority'] += 1
        self.alert_stats['types'][alert_type] = self.alert_stats['types'].get(alert_type, 0) + 1
        
        # Add to history (keep last 1000)
        self.alert_history.append(alert)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Save to database
        self._save_alert_to_db(alert)
        
        # Emit to dashboard if available
        if socketio and DASHBOARD_AVAILABLE:
            try:
                socketio.emit('new_alert', alert)
                logger.info(f"✅ Alert emitted to dashboard via Socket.IO: {alert['type']} - {alert['message'][:50]}...")
            except Exception as e:
                logger.error(f"❌ Error emitting alert to dashboard: {e}")
        
        # Log alert
        logger.info(f"ALERT [{priority.upper()}] {alert_type}: {message}")
        
        return alert
    
    def _save_alert_to_db(self, alert):
        """Save alert to database."""
        try:
            import sqlite3
            import json
            
            conn = sqlite3.connect('alerts.db')
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    message TEXT,
                    priority TEXT,
                    timestamp TEXT,
                    data TEXT
                )
            ''')
            
            # Insert alert
            cursor.execute('''
                INSERT OR REPLACE INTO alerts (id, type, message, priority, timestamp, data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert['id'],
                alert['type'],
                alert['message'],
                alert['priority'],
                alert['timestamp'],
                json.dumps(alert['data'])
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving alert to database: {e}")

# Global alert manager
alert_manager = StandardizedAlertManager()

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    global running
    logger.info("Shutdown signal received. Stopping system...")
    running = False

def initialize_core_components():
    """Initialize core surveillance components."""
    global system_components
    
    logger.info("Initializing core surveillance components...")
    
    try:
        # Video stream with retry logic
        logger.info("Initializing video stream...")
        video_stream = None
        for attempt in range(3):
            try:
                video_stream = VideoStream(source=selected_camera_index)
                success, _ = video_stream.read_frame()
                if success:
                    logger.info(f"Video stream initialized successfully on attempt {attempt+1}")
                    break
                else:
                    logger.warning(f"Video stream test failed on attempt {attempt+1}")
                    if video_stream:
                        video_stream.release()
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Video stream initialization error (attempt {attempt+1}): {e}")
                time.sleep(1)
        
        if not video_stream:
            logger.error("Failed to initialize video stream")
            alert_manager.create_alert('system_status', 'Video stream initialization failed', 'high')
        
        # Object detection and tracking
        logger.info("Initializing object detection and tracking...")
        detector = ObjectDetector()
        tracker = ObjectTracker()
        
        # Anomaly detection
        logger.info("Initializing anomaly detection...")
        analyzer = AnomalyDetector()
        
        # Add example restricted area - small corner area only
        frame_height, frame_width = 480, 640
        # Create a small restricted area in the top-right corner (20% of frame)
        restricted_area = [
            (int(frame_width * 0.8), 0),
            (frame_width, 0),
            (frame_width, int(frame_height * 0.3)),
            (int(frame_width * 0.8), int(frame_height * 0.3))
        ]
        analyzer.add_restricted_area(restricted_area)
        
        # Alert system
        logger.info("Initializing alert system...")
        notifier = AlertSystem()
        
        # Store components
        system_components.update({
            'video_stream': video_stream,
            'detector': detector,
            'tracker': tracker,
            'analyzer': analyzer,
            'notifier': notifier
        })
        
        logger.info("Core components initialized successfully")
        alert_manager.create_alert('system_status', 'Core surveillance system online', 'low')
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing core components: {e}")
        alert_manager.create_alert('system_status', f'Core system initialization failed: {str(e)}', 'high')
        return False

def initialize_advanced_features():
    """Initialize advanced surveillance features."""
    global system_components
    
    if not ADVANCED_FEATURES_AVAILABLE:
        logger.warning("Advanced features not available")
        return False
    
    logger.info("Initializing advanced surveillance features...")
    
    try:
        # Facial recognition
        logger.info("Initializing facial recognition...")
        facial_recognition = get_facial_recognition_system()
        
        # Behavior analysis
        logger.info("Initializing behavior analysis...")
        behavior_analyzer = BehaviorAnalyzer()
        
        # Person re-identification
        logger.info("Initializing person re-identification...")
        person_reid = PersonReID()
        
        # Multi-camera synchronization
        logger.info("Initializing multi-camera sync...")
        multi_camera = MultiCameraManager()
        
        # Real-time analytics
        logger.info("Initializing real-time analytics...")
        analytics = AnalyticsEngine()
        
        # Advanced alerts
        logger.info("Initializing advanced alert system...")
        advanced_alerts = AdvancedAlertSystem()
        
        # Store advanced components
        system_components.update({
            'facial_recognition': facial_recognition,
            'behavior_analyzer': behavior_analyzer,
            'person_reid': person_reid,
            'multi_camera': multi_camera,
            'analytics': analytics,
            'advanced_alerts': advanced_alerts
        })
        
        logger.info("Advanced features initialized successfully")
        alert_manager.create_alert('system_status', 'Advanced features online', 'low')
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing advanced features: {e}")
        alert_manager.create_alert('system_status', f'Advanced features initialization failed: {str(e)}', 'medium')
        return False

def start_dashboard():
    """Start the professional dashboard."""
    global dashboard_app, DASHBOARD_AVAILABLE
    
    try:
        logger.info("Starting professional dashboard...")
        dashboard_app = create_app()
        
        # Make alert manager available to dashboard
        dashboard_app.alert_manager = alert_manager
        dashboard_app.system_components = system_components
        dashboard_app.get_shared_stats = get_shared_stats
        
        # Set dashboard as available
        DASHBOARD_AVAILABLE = True
        
        # Start dashboard server on localhost for camera access compatibility
        socketio.run(dashboard_app, host='127.0.0.1', port=8082, debug=False)
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        alert_manager.create_alert('system_status', f'Dashboard failed to start: {str(e)}', 'medium')

def process_frame_with_all_features(frame, frame_count=0):
    """Process a single frame with all available features."""
    results = {
        'detections': [],
        'tracked_objects': {},
        'face_detections': [],
        'anomalies': [],
        'behavior_analysis': {},
        'alerts_generated': []
    }
    
    try:
        # Core object detection
        if 'detector' in system_components:
            detections = system_components['detector'].detect(frame)
            results['detections'] = detections
            
            # Update live stats
            update_shared_stats('live_stats.objects_in_view', len(detections))
            
            # Count person detections for daily stats (use time-based sessions to avoid over-counting)
            person_count = sum(1 for d in detections if d['class_name'] == 'person')
            if person_count > 0:
                current_time = time.time()
                with shared_stats_lock:
                    if 'detection_sessions' not in shared_stats:
                        shared_stats['detection_sessions'] = {}
                    detection_sessions = shared_stats['detection_sessions']
                    last_detection = detection_sessions.get('last_person_detection', 0)
                    # Only count if it's been more than 5 seconds since last detection
                    if isinstance(last_detection, (int, float)) and current_time - last_detection > 5.0:
                        shared_stats['objects_detected_today'] += 1
                        shared_stats['detection_sessions']['last_person_detection'] = current_time
        
        # Object tracking
        if 'tracker' in system_components and results['detections']:
            tracked_objects = system_components['tracker'].update(results['detections'])
            results['tracked_objects'] = tracked_objects
            
            # Update live stats
            update_shared_stats('live_stats.tracked_objects', len(tracked_objects))
        
        # Facial recognition (process before person alerts)
        if 'facial_recognition' in system_components:
            try:
                face_detections = system_components['facial_recognition'].detect_faces(frame)
                results['face_detections'] = face_detections

                # Automatically capture unique face snapshots for the gallery
                try:
                    system_components['facial_recognition'].capture_unique_faces(frame, face_detections)
                except Exception as cap_err:
                    logger.debug(f"Face capture error: {cap_err}")
                
                # Update live stats
                update_shared_stats('live_stats.faces_detected', len(face_detections))
                
                # Count recognized faces for daily stats (use time-based sessions to avoid over-counting)
                recognized_count = sum(1 for f in face_detections if f['is_known'])
                if recognized_count > 0:
                    current_time = time.time()
                    with shared_stats_lock:
                        if 'detection_sessions' not in shared_stats:
                            shared_stats['detection_sessions'] = {}
                        detection_sessions = shared_stats['detection_sessions']
                        last_recognition = detection_sessions.get('last_face_recognition', 0)
                        # Only count if it's been more than 10 seconds since last recognition
                        if isinstance(last_recognition, (int, float)) and current_time - last_recognition > 10.0:
                            shared_stats['faces_recognized_today'] += 1
                            shared_stats['detection_sessions']['last_face_recognition'] = current_time
                    
            except Exception as e:
                logger.error(f"Error in facial recognition: {e}")
        
        # Generate intelligent person detection alerts (after face recognition)
        if results['detections']:
            for detection in results['detections']:
                if detection['class_name'] == 'person' and detection['confidence'] > 0.8:
                    # Check if this person is identified through face recognition
                    person_name = None
                    face_confidence = 0.0
                    
                    # Look for matching face detection
                    for face in results.get('face_detections', []):
                        if face['is_known'] and _bboxes_overlap(detection['bbox'], face['bbox']):
                            person_name = face['name']
                            face_confidence = face['confidence']
                            break
                    
                    # Generate appropriate alert based on identification
                    if person_name:
                        alert = alert_manager.create_alert(
                            'face_recognition',
                            f"Known person {person_name} detected (confidence: {face_confidence:.2f})",
                            'low',
                            {
                                'person_name': person_name,
                                'face_confidence': face_confidence,
                                'detection_confidence': detection['confidence'],
                                'bbox': detection['bbox']
                            }
                        )
                    else:
                        # Check for unknown faces
                        unknown_face_detected = any(
                            not face['is_known'] and _bboxes_overlap(detection['bbox'], face['bbox'])
                            for face in results.get('face_detections', [])
                        )
                        
                        if unknown_face_detected:
                            alert = alert_manager.create_alert(
                                'unknown_person',
                                f"Unknown person detected (confidence: {detection['confidence']:.2f})",
                                'high',
                                {
                                    'class_name': detection['class_name'],
                                    'confidence': detection['confidence'],
                                    'bbox': detection['bbox']
                                }
                            )
                        else:
                            # Generic person detection (no face visible or processed)
                            alert = alert_manager.create_alert(
                                'object_detection',
                                f"Person detected in area (confidence: {detection['confidence']:.2f})",
                                'low',
                                {
                                    'class_name': detection['class_name'],
                                    'confidence': detection['confidence'],
                                    'bbox': detection['bbox']
                                }
                            )
                    
                    if alert:
                        results['alerts_generated'].append(alert)
                        update_shared_stats('alerts_generated_today', increment=True)
                        
                elif detection['class_name'] in ['knife', 'scissors'] and detection['confidence'] > 0.6:
                    # Alert for potentially dangerous objects
                    alert = alert_manager.create_alert(
                        'object_detection',
                        f"Potential weapon detected: {detection['class_name']} (confidence: {detection['confidence']:.2f})",
                        'high',
                        {
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'bbox': detection['bbox']
                        }
                    )
                    if alert:
                        results['alerts_generated'].append(alert)
                
        # Behavior analysis
        if 'behavior_analyzer' in system_components and results['tracked_objects']:
            try:
                behavior_results = system_components['behavior_analyzer'].analyze_behavior(
                    frame, list(results['tracked_objects'].values())
                )
                results['behavior_analysis'] = behavior_results
                
                # Generate alerts for suspicious behavior
                if behavior_results.get('suspicious_activities'):
                    for activity in behavior_results['suspicious_activities']:
                        alert = alert_manager.create_alert(
                            'suspicious_behavior',
                            f"Suspicious behavior detected: {activity['type']}",
                            'medium',
                            activity
                        )
                        if alert:
                            results['alerts_generated'].append(alert)
            
            except Exception as e:
                logger.error(f"Error in behavior analysis: {e}")
        
        # Anomaly detection
        if 'analyzer' in system_components:
            anomalies = system_components['analyzer'].detect_anomalies(
                frame, results['tracked_objects']
            )
            results['anomalies'] = anomalies
            
            # Generate alerts for anomalies (with better filtering)
            for anomaly in anomalies:
                # Skip restricted area violations for non-person objects in low confidence
                if (anomaly['type'] == 'restricted_area_violation' and 
                    anomaly['class_name'] != 'person' and 
                    anomaly.get('confidence', 0) < 0.7):
                    continue
                
                # Improved priority classification
                if anomaly['type'] == 'intrusion_detected':
                    priority = 'high'
                elif anomaly['type'] == 'restricted_area_violation':
                    # Only high priority for people in restricted areas
                    if anomaly['class_name'] == 'person':
                        priority = 'high'
                    else:
                        # Lower priority for objects, vary by confidence
                        confidence = anomaly.get('confidence', 0.5)
                        if confidence > 0.8:
                            priority = 'medium'
                        else:
                            priority = 'low'
                elif anomaly['type'] == 'unattended_object':
                    frames_stationary = anomaly.get('frames_stationary', 0)
                    if frames_stationary > 300:  # Object stationary for more than 10 seconds at 30fps
                        priority = 'high'
                    else:
                        priority = 'medium'
                else:
                    priority = 'medium'
                
                # Create message based on anomaly type
                if anomaly['type'] == 'restricted_area_violation':
                    message = f"Restricted area violation detected: {anomaly['class_name']} at {anomaly['location']}"
                elif anomaly['type'] == 'unattended_object':
                    message = f"Unattended object detected: {anomaly['class_name']} (stationary for {anomaly.get('frames_stationary', 0)} frames)"
                else:
                    message = f"Anomaly detected: {anomaly['type']}"
                
                alert = alert_manager.create_alert(
                    anomaly['type'],
                    message,
                    priority,
                    anomaly
                )
                if alert:
                    results['alerts_generated'].append(alert)
        
        # Person re-identification
        if 'person_reid' in system_components and results['tracked_objects']:
            try:
                reid_results = system_components['person_reid'].process_detections(
                    frame, list(results['tracked_objects'].values()), 'main_camera'
                )
                results['reid_results'] = reid_results
            except Exception as e:
                logger.error(f"Error in person re-identification: {e}")
        
        # Real-time analytics update
        if 'analytics' in system_components:
            try:
                # Record individual metrics using the correct method
                system_components['analytics'].record_metric(
                    'objects_detected', len(results['detections']), 'count', 'surveillance'
                )
                system_components['analytics'].record_metric(
                    'faces_detected', len(results['face_detections']), 'count', 'surveillance'
                )
                system_components['analytics'].record_metric(
                    'anomalies_detected', len(results['anomalies']), 'count', 'surveillance'
                )
                system_components['analytics'].record_metric(
                    'alerts_generated', len(results['alerts_generated']), 'count', 'surveillance'
                )
            except Exception as e:
                logger.error(f"Error updating analytics: {e}")
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing frame: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Continue processing with empty results to prevent crash
        results = {
            'detections': [],
            'tracked_objects': {},
            'face_detections': [],
            'anomalies': [],
            'behavior_analysis': {},
            'alerts_generated': []
        }
    
    return results

def main_surveillance_loop():
    """Main surveillance processing loop with all features."""
    logger.info("Starting main surveillance loop...")
    
    frame_count = 0
    last_stats_time = time.time()
    
    while running:
        try:
            if 'video_stream' not in system_components:
                logger.error("Video stream not available")
                time.sleep(1)
                continue
            
            # Read frame
            success, frame = system_components['video_stream'].read_frame()
            if not success:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Process frame with all features
            results = process_frame_with_all_features(frame, frame_count)
            
            # Update frame counter
            frame_count += 1
            
            # Log statistics every 100 frames
            if frame_count % 100 == 0:
                current_time = time.time()
                if last_stats_time is not None:
                    fps = 100 / (current_time - last_stats_time)
                else:
                    fps = 0.0
                last_stats_time = current_time
                
                logger.info(f"Processed {frame_count} frames - FPS: {fps:.2f} - "
                          f"Objects: {len(results['detections'])} - "
                          f"Faces: {len(results['face_detections'])} - "
                          f"Alerts: {len(results['alerts_generated'])}")
            
            # Occasionally generate test alerts for demonstration (every 30 seconds)
            import random
            if frame_count % (30 * 30) == 0:  # Every 30 seconds at 30fps
                test_alerts = [
                    {
                        'type': 'unknown_person',
                        'message': 'Unknown person detected at entrance',
                        'priority': 'high',
                        'data': {'location': 'main_entrance', 'confidence': 0.95}
                    },
                    {
                        'type': 'suspicious_behavior',
                        'message': 'Person loitering detected',
                        'priority': 'medium',
                        'data': {'location': 'parking_lot', 'duration': 300}
                    },
                    {
                        'type': 'face_recognition',
                        'message': f'Known person {random.choice(["John Doe", "Jane Smith", "Mike Johnson"])} detected',
                        'priority': 'low',
                        'data': {'confidence': round(random.uniform(0.8, 0.95), 2)}
                    },
                    {
                        'type': 'behavior_analysis',
                        'message': f'{random.choice(["Running", "Crowding", "Unusual movement"])} detected',
                        'priority': random.choice(['low', 'medium']),
                        'data': {'activity': 'unusual_behavior'}
                    }
                ]
                
                # Randomly select and send one test alert
                if random.random() < 0.8:  # 80% chance for more frequent testing
                    test_alert = random.choice(test_alerts)
                    alert = alert_manager.create_alert(
                        test_alert['type'],
                        test_alert['message'],
                        test_alert['priority'],
                        test_alert['data']
                    )
                    if alert:
                        results['alerts_generated'].append(alert)
                        update_shared_stats('alerts_generated_today', increment=True)
            
            # Brief pause to control processing rate
            time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(0.1)

def main():
    """Enhanced main function with all features and professional dashboard."""
    global dashboard_thread, running
    
    logger.info("Starting Enhanced Smart Surveillance System v2.0")
    logger.info("Features: Core Surveillance + Advanced AI + Professional Dashboard")
    
    try:
        # Select camera before initializing (skip if already set via command line)
        if selected_camera_index == 0 and '--camera' not in sys.argv and '-c' not in sys.argv:
            select_camera()
        
        # Initialize core components
        if not initialize_core_components():
            logger.error("Failed to initialize core components")
            return 1
        
        # Initialize advanced features
        advanced_success = initialize_advanced_features()
        if advanced_success:
            logger.info("All features initialized successfully")
        else:
            logger.warning("Running with core features only")
        
        # Start professional dashboard
        logger.info("Starting professional dashboard on http://localhost:8082")
        dashboard_thread = threading.Thread(target=start_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Wait a moment for dashboard to start
        time.sleep(2)
        
        # Generate startup alert
        feature_count = len([k for k in system_components.keys() if system_components[k] is not None])
        alert_manager.create_alert(
            'system_status',
            f'Enhanced surveillance system started with {feature_count} active features',
            'low'
        )
        
        logger.info("="*60)
        logger.info("Enhanced Smart Surveillance System is now running!")
        logger.info("Dashboard: http://localhost:8082")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*60)
        
        # Start main surveillance loop
        main_surveillance_loop()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up system resources...")
        
        # Generate shutdown alert
        alert_manager.create_alert(
            'system_status',
            'Enhanced surveillance system shutting down',
            'low'
        )
        
        # Release video stream
        if 'video_stream' in system_components and system_components['video_stream']:
            system_components['video_stream'].release()
        
        # Cleanup advanced features
        if 'facial_recognition' in system_components and system_components['facial_recognition']:
            system_components['facial_recognition'].cleanup()
        
        logger.info("Enhanced Smart Surveillance System shutdown complete")
        return 0

def _bboxes_overlap(bbox1, bbox2, overlap_threshold=0.3):
    """
    Check if two bounding boxes overlap significantly.
    
    Args:
        bbox1, bbox2: [x, y, w, h] format bounding boxes
        overlap_threshold: Minimum overlap ratio to consider as overlap
    
    Returns:
        bool: True if boxes overlap significantly
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left >= right or top >= bottom:
        return False
    
    # Calculate intersection area
    intersection_area = (right - left) * (bottom - top)
    
    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    # Calculate overlap ratio
    overlap_ratio = intersection_area / union_area if union_area > 0 else 0
    
    return overlap_ratio >= overlap_threshold

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Smart Surveillance System')
    parser.add_argument('--camera', '-c', type=int, default=None,
                       help='Camera index to use (skips interactive selection)')
    parser.add_argument('--list-cameras', '-l', action='store_true',
                       help='List available cameras and exit')
    args = parser.parse_args()
    
    # List cameras mode
    if args.list_cameras:
        print("\n🎥 Smart Surveillance System - Camera List\n")
        cameras = enumerate_cameras()
        if cameras:
            print("\n📋 Available Cameras:")
            for cam in cameras:
                print(f"  Camera {cam['index']}: {cam['resolution']} @ {cam['fps']}fps ({cam['backend']})")
        sys.exit(0)
    
    # Set camera from command line if provided
    if args.camera is not None:
        selected_camera_index = args.camera
        print(f"\n✅ Using camera {selected_camera_index} (from command line)")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the enhanced system
    exit_code = main()
    sys.exit(exit_code)


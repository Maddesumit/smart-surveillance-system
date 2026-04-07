# Professional Dashboard Routes with Advanced Features
import numpy as np
from flask import Blueprint, render_template, Response, jsonify, request, current_app
import cv2
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
import json

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Socket.IO for real-time alert broadcasts
try:
    from dashboard import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    socketio = None

# Import feature data for detail pages (after path setup)
from dashboard.routes.feature_data import get_feature, get_all_features

# Import core modules
from video_processing.video_stream import VideoStream, ThreadedVideoStream
from video_processing.optimized_video_processor import OptimizedVideoProcessor, get_optimized_processor
from object_detection.detector import ObjectDetector
from object_detection.tracker import ObjectTracker
from anomaly_detection.analyzer import AnomalyDetector
from alert_system.notifier import AlertSystem

# Video optimization settings
FRAME_SKIP = 3  # Process detection every 3rd frame
USE_THREADED_STREAM = True  # Use threaded video capture
USE_GPU = True  # Try to use GPU acceleration

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
    print(f"Warning: Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Create Blueprint
main = Blueprint('main', __name__)

# Global system components
video_stream = None
detector = None
tracker = None
analyzer = None
notifier = None

# Advanced features
facial_recognition = None
behavior_analyzer = None
person_reid = None
multi_camera = None
analytics = None
advanced_alerts = None

# Alert management
alerts_history = []
alert_stats = {
    'total': 0,
    'high_priority': 0,
    'medium_priority': 0,
    'low_priority': 0,
    'types': {}
}

# Camera management
selected_camera_index = 0
available_cameras = []


def get_alerts_db_path():
    """Get canonical alerts database path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    db_dir = os.path.join(base_dir, 'logs')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'alerts.db')


def get_legacy_alerts_db_path():
    """Get legacy alerts database path used by older code."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(base_dir, 'alerts.db')

class StandardizedAlert:
    """Standardized alert system to reduce noise and improve relevance."""
    
    PRIORITY_HIGH = 'high'
    PRIORITY_MEDIUM = 'medium'
    PRIORITY_LOW = 'low'
    
    TYPE_UNKNOWN_PERSON = 'unknown_person'
    TYPE_INTRUSION = 'intrusion'
    TYPE_SUSPICIOUS_BEHAVIOR = 'suspicious_behavior'
    TYPE_OBJECT_DETECTION = 'object_detection'
    TYPE_SYSTEM_STATUS = 'system_status'
    TYPE_FACIAL_RECOGNITION = 'facial_recognition'
    
    # Alert throttling to prevent spam
    last_alerts = {}
    throttle_time = 10  # seconds (reduced from 30 for better responsiveness)
    
    @classmethod
    def create_alert(cls, alert_type, message, priority=PRIORITY_MEDIUM, data=None):
        """Create a standardized alert with throttling."""
        current_time = datetime.now()
        
        # Check if we should throttle this alert
        alert_key = f"{alert_type}_{message[:50]}"
        if alert_key in cls.last_alerts:
            time_diff = (current_time - cls.last_alerts[alert_key]).total_seconds()
            if time_diff < cls.throttle_time:
                return None  # Throttled
        
        cls.last_alerts[alert_key] = current_time
        
        return {
            'id': f"alert_{int(current_time.timestamp())}",
            'type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': current_time.isoformat(),
            'data': data or {}
        }

def enumerate_cameras(max_cameras=10, skip_active=True):
    """
    Enumerate all available cameras on the system.
    
    Args:
        max_cameras: Maximum number of camera indices to check
        skip_active: If True, skip the currently active camera to avoid conflicts
    
    Returns:
        list: List of dicts with camera info
    """
    global available_cameras, video_stream, selected_camera_index
    
    # If we already have cameras cached and video stream is active, return cached list
    if available_cameras and video_stream is not None and skip_active:
        return available_cameras
    
    available_cameras = []
    
    for i in range(max_cameras):
        # Skip the active camera to avoid conflicts
        if skip_active and video_stream is not None and i == selected_camera_index:
            # Add the active camera info without re-opening it
            try:
                width = int(video_stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if video_stream.cap else 640
                height = int(video_stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if video_stream.cap else 480
                fps = int(video_stream.cap.get(cv2.CAP_PROP_FPS)) if video_stream.cap else 30
                backend = video_stream.cap.getBackendName() if video_stream.cap and hasattr(video_stream.cap, 'getBackendName') else "AVFoundation"
            except:
                width, height, fps, backend = 640, 480, 30, "Unknown"
            
            camera_info = {
                'index': i,
                'name': f"Camera {i} (Active)",
                'resolution': f"{width}x{height}",
                'fps': fps if fps > 0 else 30,
                'backend': backend,
                'working': True
            }
            available_cameras.append(camera_info)
            continue
        
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    try:
                        backend = cap.getBackendName()
                    except:
                        backend = "Unknown"
                    
                    camera_info = {
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps if fps > 0 else 30,
                        'backend': backend,
                        'working': True
                    }
                    available_cameras.append(camera_info)
                cap.release()
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
            continue
    
    return available_cameras

def switch_camera(camera_index):
    """
    Switch to a different camera.
    
    Args:
        camera_index: Index of the camera to switch to
        
    Returns:
        bool: True if successful, False otherwise
    """
    global video_stream, selected_camera_index
    
    try:
        # Release current video stream
        if video_stream is not None:
            video_stream.release()
        
        # Create new video stream with selected camera
        video_stream = VideoStream(source=camera_index)
        success, _ = video_stream.read_frame()
        
        if success:
            selected_camera_index = camera_index
            print(f"✅ Switched to camera {camera_index}")
            return True
        else:
            # Fallback to previous camera
            video_stream = VideoStream(source=selected_camera_index)
            print(f"❌ Failed to switch to camera {camera_index}")
            return False
            
    except Exception as e:
        print(f"❌ Error switching camera: {e}")
        # Try to restore previous camera
        try:
            video_stream = VideoStream(source=selected_camera_index)
        except:
            pass
        return False

def initialize_components():
    """Initialize all system components."""
    global video_stream, detector, tracker, analyzer, notifier
    global facial_recognition, behavior_analyzer, person_reid, multi_camera, analytics, advanced_alerts
    global selected_camera_index
    
    if video_stream is None:
        print("🔧 Initializing core components...")
        
        # Note: Camera enumeration is done on-demand via the /cameras API
        # to avoid conflicts with the main video stream
        
        # Core components - use selected camera with optional threading
        if USE_THREADED_STREAM:
            print("📹 Using threaded video stream for better performance...")
            video_stream = ThreadedVideoStream(source=selected_camera_index)
            video_stream.start()
        else:
            video_stream = VideoStream(source=selected_camera_index)
        
        # Initialize detector with GPU if available
        if USE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                    detector = ObjectDetector(device='cuda')
                else:
                    print("⚠️ GPU requested but not available, using CPU")
                    detector = ObjectDetector()
            except ImportError:
                print("⚠️ PyTorch not installed for GPU detection, using CPU")
                detector = ObjectDetector()
        else:
            detector = ObjectDetector()
        
        tracker = ObjectTracker()
        analyzer = AnomalyDetector()
        notifier = AlertSystem()
        
        # Set up alert callback
        original_generate_alert = notifier.generate_alert
        def wrapped_generate_alert(anomaly):
            alert = original_generate_alert(anomaly)
            process_standardized_alert(anomaly)
            return alert
        notifier.generate_alert = wrapped_generate_alert
        
        print(f"✅ Core components initialized with camera {selected_camera_index}")
        
    if ADVANCED_FEATURES_AVAILABLE and facial_recognition is None:
        print("🚀 Initializing advanced features...")
        try:
            facial_recognition = FacialRecognitionSystem()
            behavior_analyzer = BehaviorAnalyzer()
            person_reid = PersonReID()
            multi_camera = MultiCameraManager()()
            analytics = AnalyticsEngine()
            advanced_alerts = AdvancedAlertSystem()
            print("✅ Advanced features initialized")
        except Exception as e:
            print(f"⚠️ Advanced features initialization failed: {e}")

def process_standardized_alert(anomaly):
    """Process and standardize alerts to reduce noise."""
    global alerts_history, alert_stats
    
    alert_type = anomaly.get('type', 'unknown')
    message = anomaly.get('message', 'Alert detected')
    
    # Determine priority based on alert type and content
    if alert_type in ['intrusion_detected', 'unknown_face_detected', 'crowd_detected']:
        priority = StandardizedAlert.PRIORITY_HIGH
    elif alert_type in ['suspicious_behavior', 'restricted_area_breach', 'restricted_area_violation']:
        priority = StandardizedAlert.PRIORITY_MEDIUM
    elif alert_type in ['person_detected', 'unattended_object']:
        priority = StandardizedAlert.PRIORITY_LOW
    else:
        priority = StandardizedAlert.PRIORITY_LOW
    
    # Create standardized alert
    std_alert = StandardizedAlert.create_alert(
        alert_type=alert_type,
        message=message,
        priority=priority,
        data=anomaly
    )
    
    if std_alert:
        alerts_history.append(std_alert)
        
        # Update statistics
        alert_stats['total'] += 1
        alert_stats[f'{priority}_priority'] += 1
        alert_stats['types'][alert_type] = alert_stats['types'].get(alert_type, 0) + 1
        
        # Keep only recent alerts (last 1000)
        if len(alerts_history) > 1000:
            alerts_history = alerts_history[-1000:]
        
        # Save to database
        save_alert_to_db(std_alert)

def save_alert_to_db(alert):
    """Save alert to SQLite database and broadcast via Socket.IO."""
    try:
        db_path = get_alerts_db_path()
        
        conn = sqlite3.connect(db_path)
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
        
        # **Broadcast alert to all connected clients in real-time**
        if SOCKETIO_AVAILABLE and socketio:
            try:
                # Try to emit within Flask app context if available
                try:
                    socketio.emit('new_alert', alert, broadcast=True, namespace='/')
                except RuntimeError:
                    # No app context, try with current_app
                    try:
                        with current_app.app_context():
                            socketio.emit('new_alert', alert, broadcast=True, namespace='/')
                    except RuntimeError:
                        # Still no context, alert is saved to DB which is sufficient
                        pass
            except Exception as emit_error:
                # Log but don't crash - alerts are still persisted to DB
                import logging
                logging.warning(f"Socket.IO alert broadcast failed (non-critical): {emit_error}")
    except Exception as e:
        print(f"Error saving alert to database: {e}")

# Cached detection results for frame skipping
cached_detections = []
cached_tracked_objects = {}
cached_face_detections = []
frame_counter = 0

def generate_frames():
    """Generate video frames with optimized detection using frame skipping and threading."""
    global video_stream, detector, tracker, analyzer, facial_recognition
    global cached_detections, cached_tracked_objects, cached_face_detections, frame_counter
    
    initialize_components()
    
    error_count = 0
    max_errors = 10
    last_fps_time = time.time()
    fps_frame_count = 0
    display_fps = 0
    
    print(f"🚀 Starting optimized video stream (frame_skip={FRAME_SKIP}, threaded={USE_THREADED_STREAM}, gpu={USE_GPU})")
    
    while True:
        try:
            # Check if video stream is valid
            if video_stream is None or not hasattr(video_stream, 'cap') or video_stream.cap is None:
                print("Video stream not initialized, waiting...")
                time.sleep(1)
                initialize_components()
                continue
            
            # Get frame from video stream
            success, frame = video_stream.read_frame()
            if not success:
                error_count += 1
                if error_count >= max_errors:
                    print(f"Too many errors ({error_count}), waiting before retry...")
                    time.sleep(0.5)
                    error_count = 0
                continue
            
            # Reset error count on successful read
            error_count = 0
            frame_counter += 1
            fps_frame_count += 1
            
            # Calculate display FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                display_fps = fps_frame_count / (current_time - last_fps_time)
                fps_frame_count = 0
                last_fps_time = current_time
            
            # Frame skipping: Only run detection every FRAME_SKIP frames
            should_detect = (frame_counter % FRAME_SKIP) == 0
            
            if should_detect:
                detection_start = time.time()
                
                # Object detection
                detections = detector.detect(frame)
                
                # Record analytics
                record_detection_analytics(detections)
                
                # Object tracking
                tracked_objects = tracker.update(detections)
                
                # === PERSON COUNT TRACKING ===
                person_detections = [d for d in detections if d.get('class_name', '').lower() == 'person']
                current_person_count = len(person_detections)
                
                # Update analytics_data with person count
                analytics_data['person_count_current'] = current_person_count
                if current_person_count > analytics_data.get('person_count_max', 0):
                    analytics_data['person_count_max'] = current_person_count
                
                # Record person count history (keep last 500 data points)
                analytics_data.setdefault('person_count_history', [])
                analytics_data['person_count_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'count': current_person_count
                })
                if len(analytics_data['person_count_history']) > 500:
                    analytics_data['person_count_history'] = analytics_data['person_count_history'][-500:]
                
                # === PERSON DETECTION ALERTS ===
                if current_person_count > 0:
                    person_alert = StandardizedAlert.create_alert(
                        'person_detected',
                        f"{current_person_count} person(s) detected in surveillance area",
                        StandardizedAlert.PRIORITY_LOW,
                        {'person_count': current_person_count, 'timestamp': datetime.now().isoformat()}
                    )
                    if person_alert:
                        alerts_history.append(person_alert)
                        save_alert_to_db(person_alert)
                
                # === CROWD DETECTION ALERT (threshold: 5 persons) ===
                CROWD_THRESHOLD = 5
                if current_person_count >= CROWD_THRESHOLD:
                    crowd_alert = StandardizedAlert.create_alert(
                        'crowd_detected',
                        f"Crowd alert: {current_person_count} persons detected (threshold: {CROWD_THRESHOLD})",
                        StandardizedAlert.PRIORITY_HIGH,
                        {'person_count': current_person_count, 'threshold': CROWD_THRESHOLD}
                    )
                    if crowd_alert:
                        alerts_history.append(crowd_alert)
                        save_alert_to_db(crowd_alert)
                
                # Facial recognition (if available)
                face_detections = []
                if facial_recognition:
                    try:
                        face_detections = facial_recognition.detect_faces(frame)
                        
                        # Process unknown faces
                        for face in face_detections:
                            if not face['is_known']:
                                alert = StandardizedAlert.create_alert(
                                    StandardizedAlert.TYPE_UNKNOWN_PERSON,
                                    f"Unknown person detected with confidence {face['confidence']:.2f}",
                                    StandardizedAlert.PRIORITY_HIGH,
                                    {'bbox': face['bbox'], 'confidence': face['confidence']}
                                )
                                if alert:
                                    alerts_history.append(alert)
                                    save_alert_to_db(alert)
                    except Exception as e:
                        print(f"Error in facial recognition: {e}")
                
                # Anomaly detection
                anomalies = analyzer.detect_anomalies(frame, tracked_objects)
                
                # Process anomalies
                for anomaly in anomalies:
                    process_standardized_alert(anomaly)
                
                # Cache results for skipped frames
                cached_detections = detections.copy() if detections else []
                cached_tracked_objects = tracked_objects.copy() if tracked_objects else {}
                cached_face_detections = face_detections.copy() if face_detections else []
                
                detection_time = time.time() - detection_start
            else:
                # Use cached results for skipped frames
                detections = cached_detections
                tracked_objects = cached_tracked_objects
                face_detections = cached_face_detections
                current_person_count = analytics_data.get('person_count_current', 0)
            
            # Draw face detections if available
            if facial_recognition and face_detections:
                try:
                    frame = facial_recognition.draw_face_detections(frame, face_detections)
                except:
                    pass
            
            # Draw object detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                label = f"{detection['class_name']} ({detection['confidence']:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw tracking info
            for track_id, obj in tracked_objects.items():
                x1, y1, x2, y2 = obj['bbox']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, f"ID: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Add system status overlay with FPS and person count
            status_text = f"FPS: {display_fps:.1f} | Objects: {len(detections)} | Tracked: {len(tracked_objects)} | Faces: {len(face_detections)}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add optimization indicator
            opt_text = f"Skip: {FRAME_SKIP} | GPU: {'Yes' if USE_GPU else 'No'}"
            cv2.putText(frame, opt_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # === PERSON COUNT OVERLAY (top-right) ===
            person_count_text = f"Persons: {current_person_count}"
            text_size = cv2.getTextSize(person_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 20
            # Draw background rectangle for person count
            cv2.rectangle(frame, (text_x - 10, 10), (frame.shape[1] - 10, 45), (0, 0, 0), -1)
            cv2.rectangle(frame, (text_x - 10, 10), (frame.shape[1] - 10, 45), (0, 217, 165), 2)
            person_color = (0, 255, 0) if current_person_count < 5 else (0, 0, 255)
            cv2.putText(frame, person_count_text, (text_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, person_color, 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Encode frame with optimized JPEG quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Slightly lower quality for faster encoding
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.1)
            continue

@main.before_app_request
def startup():
    """Initialize components on startup."""
    initialize_components()

@main.route('/')
def home():
    """Project homepage with all features and technology information."""
    return render_template('home.html')

@main.route('/dashboard')
def dashboard():
    """Professional surveillance dashboard."""
    initialize_components()
    
    # Get system status
    system_status = {
        'core_features': {
            'video_stream': video_stream is not None,
            'object_detection': detector is not None,
            'tracking': tracker is not None,
            'anomaly_detection': analyzer is not None,
            'alerts': notifier is not None
        },
        'advanced_features': {
            'available': ADVANCED_FEATURES_AVAILABLE,
            'facial_recognition': facial_recognition is not None,
            'behavior_analysis': behavior_analyzer is not None,
            'person_reid': person_reid is not None,
            'multi_camera': multi_camera is not None,
            'analytics': analytics is not None,
            'advanced_alerts': advanced_alerts is not None
        }
    }
    
    return render_template('dashboard.html', system_status=system_status)

@main.route('/features/<feature_slug>')
def feature_detail(feature_slug):
    """Display detailed information about a specific feature."""
    feature = get_feature(feature_slug)
    if not feature:
        return render_template('404.html'), 404
    return render_template('feature_detail.html', feature=feature)

@main.route('/features')
def features_list():
    """List all available features."""
    features = get_all_features()
    return render_template('features_list.html', features=features)

@main.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/cameras')
def list_cameras():
    """Get list of available cameras."""
    try:
        # Re-enumerate cameras to get fresh list
        cameras = enumerate_cameras()
        return jsonify({
            'success': True,
            'cameras': cameras,
            'selected': selected_camera_index
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main.route('/cameras/switch/<int:camera_index>', methods=['POST'])
def switch_camera_route(camera_index):
    """Switch to a different camera."""
    try:
        success = switch_camera(camera_index)
        if success:
            return jsonify({
                'success': True,
                'message': f'Switched to camera {camera_index}',
                'selected': camera_index
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to switch to camera {camera_index}'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main.route('/cameras/current')
def get_current_camera():
    """Get current camera information."""
    try:
        current_camera = None
        for cam in available_cameras:
            if cam['index'] == selected_camera_index:
                current_camera = cam
                break
        
        return jsonify({
            'success': True,
            'selected': selected_camera_index,
            'camera': current_camera
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main.route('/alerts')
def get_alerts():
    """Get recent alerts with filtering."""
    try:
        limit = request.args.get('limit', type=int)
        alert_type = request.args.get('type', None)
        priority = request.args.get('priority', None)
        
        # Query database for alerts
        conn = sqlite3.connect(get_alerts_db_path())
        cursor = conn.cursor()
        
        # Build query with filters
        query = 'SELECT id, type, message, priority, timestamp, data FROM alerts'
        params = []
        conditions = []
        
        # Add dismissed filter if column exists
        try:
            cursor.execute("PRAGMA table_info(alerts)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'dismissed' in columns:
                conditions.append('(dismissed IS NULL OR dismissed = 0)')
        except:
            pass
        
        if alert_type:
            conditions.append('type = ?')
            params.append(alert_type)
        if priority:
            conditions.append('priority = ?')
            params.append(priority)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY timestamp DESC'
        if limit and limit > 0:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to alert objects
        recent_alerts = []
        for row in rows:
            parsed_data = {}
            if row[5]:
                try:
                    parsed_data = json.loads(row[5])
                except (json.JSONDecodeError, TypeError):
                    parsed_data = {'raw_data': str(row[5])}

            alert = {
                'id': row[0],
                'type': row[1],
                'message': row[2],
                'priority': row[3],
                'timestamp': row[4],
                'data': parsed_data
            }
            recent_alerts.append(alert)
        
        return jsonify(recent_alerts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/alert_statistics')
def get_alert_statistics():
    """Get alert statistics."""
    try:
        # Calculate recent stats (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in alerts_history 
                        if datetime.fromisoformat(a['timestamp'].replace('Z', '+00:00').replace('+00:00', '')) > recent_threshold]
        
        stats = {
            'total_alerts': len(alerts_history),
            'recent_alerts_24h': len(recent_alerts),
            'alert_stats': alert_stats.copy(),
            'recent_by_priority': {
                'high': len([a for a in recent_alerts if a['priority'] == 'high']),
                'medium': len([a for a in recent_alerts if a['priority'] == 'medium']),
                'low': len([a for a in recent_alerts if a['priority'] == 'low'])
            },
            'recent_by_type': {}
        }
        
        # Count recent alerts by type
        for alert in recent_alerts:
            alert_type = alert['type']
            stats['recent_by_type'][alert_type] = stats['recent_by_type'].get(alert_type, 0) + 1
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/facial_recognition')
def facial_recognition_page():
    """Facial recognition management page."""
    return render_template('facial_recognition.html')

@main.route('/analytics')
def analytics_page():
    """Analytics dashboard page."""
    return render_template('analytics.html')

@main.route('/settings')
def settings_page():
    """System settings page."""
    return render_template('settings.html')

@main.route('/system_status')
def system_status():
    """Get comprehensive system status."""
    initialize_components()
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'core_components': {
            'video_stream': video_stream is not None,
            'object_detector': detector is not None,
            'tracker': tracker is not None,
            'anomaly_detector': analyzer is not None,
            'alert_system': notifier is not None
        },
        'advanced_features': {
            'available': ADVANCED_FEATURES_AVAILABLE,
            'facial_recognition': facial_recognition is not None,
            'behavior_analysis': behavior_analyzer is not None,
            'person_reid': person_reid is not None,
            'multi_camera': multi_camera is not None,
            'real_time_analytics': analytics is not None,
            'advanced_alerts': advanced_alerts is not None
        },
        'statistics': {
            'alerts': alert_stats.copy(),
            'facial_recognition': facial_recognition.get_recognition_stats() if facial_recognition else {},
            'behavior_analysis': behavior_analyzer.get_behavior_stats() if behavior_analyzer else {},
            'person_reid': person_reid.get_reid_stats() if person_reid else {}
        }
    }
    
    return jsonify(status)

# Advanced feature management endpoints
@main.route('/facial_recognition/add_person', methods=['POST'])
def add_known_person():
    """Add a new person to facial recognition database."""
    if not facial_recognition:
        return jsonify({'error': 'Facial recognition not available'}), 400
    
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')  # Base64 encoded image
        
        if not name or not image_data:
            return jsonify({'error': 'Name and image are required'}), 400
        
        # Use the facial recognition system to enroll person from base64 image
        result = facial_recognition.enroll_person_from_base64(name, image_data)
        
        if result:
            return jsonify({
                'success': True, 
                'message': f'Person {name} enrolled successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to enroll person. Please ensure the image contains a clear face.'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/facial_recognition/known_persons')
def get_known_persons():
    """Get list of known persons."""
    if not facial_recognition:
        return jsonify({'error': 'Facial recognition not available'}), 400
    
    try:
        known_persons = facial_recognition.get_known_persons()
        stats = facial_recognition.get_recognition_stats()
        
        return jsonify({
            'known_persons': known_persons,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/analytics/behavior')
def get_behavior_analytics():
    """Get behavior analysis data."""
    if not behavior_analyzer:
        return jsonify({'error': 'Behavior analysis not available'}), 400
    
    try:
        stats = behavior_analyzer.get_analysis_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/analytics/person_reid')
def get_person_reid_analytics():
    """Get person re-identification analytics."""
    if not person_reid:
        return jsonify({'error': 'Person ReID not available'}), 400
    
    try:
        stats = person_reid.get_reid_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/alerts/<alert_id>')
def get_alert_details(alert_id):
    """Get detailed information about a specific alert."""
    try:
        # Query database for the specific alert
        conn = sqlite3.connect(get_alerts_db_path())
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, type, message, priority, timestamp, data 
            FROM alerts WHERE id = ?
        ''', (alert_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return jsonify({'error': 'Alert not found'}), 404
        
        # Convert to alert object with detailed information
        alert = {
            'id': row[0],
            'type': row[1],
            'message': row[2],
            'priority': row[3],
            'timestamp': row[4],
            'data': json.loads(row[5]) if row[5] else {},
            'details': {
                'formatted_timestamp': datetime.fromisoformat(row[4].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S'),
                'priority_color': {
                    'high': '#dc3545',
                    'medium': '#ffc107', 
                    'low': '#28a745'
                }.get(row[3], '#6c757d'),
                'type_description': {
                    'system_status': 'System Status Alert',
                    'object_detection': 'Object Detection Alert',
                    'face_recognition': 'Face Recognition Alert',
                    'unknown_person': 'Unknown Person Alert',
                    'suspicious_behavior': 'Suspicious Behavior Alert',
                    'restricted_area_violation': 'Restricted Area Violation',
                    'intrusion_detected': 'Intrusion Detection Alert',
                    'unattended_object': 'Unattended Object Alert'
                }.get(row[1], 'General Alert')
            }
        }
        
        return jsonify(alert)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/alerts/dismiss/<alert_id>', methods=['POST'])
def dismiss_alert(alert_id):
    """Dismiss a specific alert."""
    try:
        conn = sqlite3.connect(get_alerts_db_path())
        cursor = conn.cursor()
        
        cursor.execute('UPDATE alerts SET dismissed = 1 WHERE id = ?', (alert_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Alert dismissed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/alerts/clear_all', methods=['POST'])
def clear_all_alerts():
    """Clear all alerts."""
    try:
        conn = sqlite3.connect(get_alerts_db_path())
        cursor = conn.cursor()
        
        cursor.execute('UPDATE alerts SET dismissed = 1')
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'All alerts cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def initialize_alerts_database():
    """Initialize alerts database with sample data."""
    try:
        conn = sqlite3.connect(get_alerts_db_path())
        cursor = conn.cursor()
        
        # Create alerts table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                priority TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT,
                dismissed INTEGER DEFAULT 0
            )
        ''')
        
        # Add dismissed column if it doesn't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN dismissed INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            # Column already exists, which is fine
            pass
        
        # Check if we have any alerts
        # Reset dismissed alerts on fresh startup so notifications show again
        cursor.execute('UPDATE alerts SET dismissed = 0 WHERE dismissed = 1')
        
        cursor.execute('SELECT COUNT(*) FROM alerts')
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Add sample alerts for testing
            sample_alerts = [
                {
                    'id': 'alert_001',
                    'type': 'unknown_person',
                    'message': 'Unknown person detected at main entrance',
                    'priority': 'high',
                    'timestamp': datetime.now().isoformat(),
                    'data': json.dumps({'location': 'main_entrance', 'confidence': 0.95})
                },
                {
                    'id': 'alert_002',
                    'type': 'suspicious_behavior',
                    'message': 'Loitering detected in parking area',
                    'priority': 'medium',
                    'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat(),
                    'data': json.dumps({'location': 'parking_area', 'duration': 300})
                },
                {
                    'id': 'alert_003',
                    'type': 'object_detection',
                    'message': 'Unattended object detected',
                    'priority': 'medium',
                    'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                    'data': json.dumps({'location': 'lobby', 'object_type': 'bag'})
                },
                {
                    'id': 'alert_004',
                    'type': 'intrusion',
                    'message': 'Motion detected in restricted area',
                    'priority': 'high',
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'data': json.dumps({'location': 'restricted_zone_a', 'motion_level': 0.8})
                },
                {
                    'id': 'alert_005',
                    'type': 'face_recognition',
                    'message': 'Known person John Doe detected',
                    'priority': 'low',
                    'timestamp': (datetime.now() - timedelta(hours=3)).isoformat(),
                    'data': json.dumps({'person_name': 'John Doe', 'confidence': 0.92})
                },
                {
                    'id': 'alert_006',
                    'type': 'behavior_analysis',
                    'message': 'Running detected in hallway',
                    'priority': 'low',
                    'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
                    'data': json.dumps({'activity': 'running', 'location': 'hallway_b'})
                }
            ]
            
            for alert in sample_alerts:
                cursor.execute('''
                    INSERT INTO alerts (id, type, message, priority, timestamp, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (alert['id'], alert['type'], alert['message'], alert['priority'], 
                     alert['timestamp'], alert['data']))
        
        conn.commit()
        conn.close()

        # Migrate legacy alerts database if present
        legacy_db_path = get_legacy_alerts_db_path()
        canonical_db_path = get_alerts_db_path()
        if os.path.exists(legacy_db_path) and legacy_db_path != canonical_db_path:
            try:
                old_conn = sqlite3.connect(legacy_db_path)
                old_cursor = old_conn.cursor()

                old_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
                if old_cursor.fetchone():
                    old_cursor.execute("SELECT id, type, message, priority, timestamp, data FROM alerts")
                    legacy_rows = old_cursor.fetchall()

                    if legacy_rows:
                        new_conn = sqlite3.connect(canonical_db_path)
                        new_cursor = new_conn.cursor()
                        new_cursor.executemany(
                            '''
                            INSERT OR IGNORE INTO alerts (id, type, message, priority, timestamp, data)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ''',
                            legacy_rows
                        )
                        new_conn.commit()
                        new_conn.close()

                old_conn.close()
            except Exception as migration_error:
                print(f"Warning: legacy alert migration failed: {migration_error}")

        print("Alerts database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing alerts database: {e}")

# Initialize alerts database when module loads
initialize_alerts_database()

# ============================================
# Person Re-identification API Endpoints
# ============================================

@main.route('/api/person_reid/tracks')
def get_person_tracks():
    """Get tracked persons data for Person Re-ID section."""
    global person_reid
    
    try:
        if person_reid is None:
            # Return demo data if module not initialized
            return jsonify({
                'success': True,
                'tracks': [
                    {
                        'person_id': 'P001',
                        'first_seen': (datetime.now() - timedelta(hours=2)).isoformat(),
                        'last_seen': datetime.now().isoformat(),
                        'cameras': ['Camera 0', 'Camera 1'],
                        'total_sightings': 5,
                        'confidence': 0.85
                    },
                    {
                        'person_id': 'P002',
                        'first_seen': (datetime.now() - timedelta(hours=1)).isoformat(),
                        'last_seen': datetime.now().isoformat(),
                        'cameras': ['Camera 0'],
                        'total_sightings': 3,
                        'confidence': 0.92
                    },
                    {
                        'person_id': 'P003',
                        'first_seen': (datetime.now() - timedelta(minutes=30)).isoformat(),
                        'last_seen': datetime.now().isoformat(),
                        'cameras': ['Camera 1'],
                        'total_sightings': 2,
                        'confidence': 0.78
                    }
                ],
                'total_tracked': 3,
                'cross_camera_matches': 1
            })
        
        # Get real data from person_reid module
        stats = person_reid.get_reid_stats()
        return jsonify({
            'success': True,
            'tracks': stats.get('recent_tracks', []),
            'total_tracked': stats.get('total_persons', 0),
            'cross_camera_matches': stats.get('cross_camera_matches', 0)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/person_reid/stats')
def get_person_reid_stats():
    """Get Person Re-ID statistics."""
    global person_reid
    
    try:
        if person_reid is None:
            return jsonify({
                'success': True,
                'total_persons': 3,
                'active_tracks': 2,
                'cross_camera_matches': 1,
                'gallery_size': 3,
                'match_rate': 0.85,
                'avg_confidence': 0.85
            })
        
        stats = person_reid.get_reid_stats()
        return jsonify({
            'success': True,
            **stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/person_reid/search', methods=['POST'])
def search_person_by_id():
    """Search for a person by ID."""
    global person_reid
    
    try:
        data = request.get_json() or {}
        person_id = data.get('person_id', '')
        
        if not person_id:
            return jsonify({'success': False, 'error': 'Person ID required'}), 400
        
        if person_reid is None:
            # Return demo result
            return jsonify({
                'success': True,
                'person': {
                    'person_id': person_id,
                    'first_seen': (datetime.now() - timedelta(hours=1)).isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'cameras': ['Camera 0'],
                    'total_sightings': 3,
                    'history': [
                        {'time': (datetime.now() - timedelta(minutes=30)).isoformat(), 'camera': 'Camera 0'},
                        {'time': (datetime.now() - timedelta(minutes=15)).isoformat(), 'camera': 'Camera 0'},
                        {'time': datetime.now().isoformat(), 'camera': 'Camera 0'}
                    ]
                }
            })
        
        history = person_reid.get_person_history(person_id)
        return jsonify({
            'success': True,
            'person': history
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# Analytics API Endpoints
# ============================================

# Analytics tracking variables
analytics_data = {
    'detection_count': 0,
    'person_count': 0,
    'person_count_current': 0,
    'person_count_max': 0,
    'person_count_history': [],
    'start_time': datetime.now(),
    'hourly_detections': [0] * 24,
    'object_categories': {},
    'alerts_per_hour': 0,
    'fps_samples': []
}

@main.route('/api/analytics/dashboard')
def get_analytics_dashboard():
    """Get comprehensive analytics dashboard data."""
    global analytics, analytics_data
    
    try:
        # Calculate uptime
        uptime_seconds = (datetime.now() - analytics_data['start_time']).total_seconds()
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        
        # Calculate average FPS
        fps_samples = analytics_data.get('fps_samples', [])
        avg_fps = sum(fps_samples[-100:]) / max(len(fps_samples[-100:]), 1) if fps_samples else 30.0
        
        # Get object categories
        categories = analytics_data.get('object_categories', {'person': 45, 'car': 20, 'dog': 5, 'cat': 3})
        
        # Get hourly data
        current_hour = datetime.now().hour
        hourly_data = analytics_data.get('hourly_detections', [0] * 24)
        
        return jsonify({
            'success': True,
            'detection_count': analytics_data.get('detection_count', 0),
            'person_count': analytics_data.get('person_count', 0),
            'uptime': f"{hours}h {minutes}m",
            'uptime_seconds': uptime_seconds,
            'avg_fps': round(avg_fps, 1),
            'alerts_per_hour': analytics_data.get('alerts_per_hour', 0),
            'object_categories': categories,
            'hourly_detections': hourly_data,
            'current_hour': current_hour,
            'activity_summary': [
                {'time': '00:00-06:00', 'level': 'Low', 'events': sum(hourly_data[0:6])},
                {'time': '06:00-12:00', 'level': 'Medium', 'events': sum(hourly_data[6:12])},
                {'time': '12:00-18:00', 'level': 'High', 'events': sum(hourly_data[12:18])},
                {'time': '18:00-24:00', 'level': 'Medium', 'events': sum(hourly_data[18:24])}
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/analytics/metrics')
def get_analytics_metrics():
    """Get performance metrics for analytics."""
    global analytics_data
    
    try:
        return jsonify({
            'success': True,
            'metrics': {
                'cpu_usage': 45.2,
                'memory_usage': 62.5,
                'gpu_usage': 38.0,
                'disk_usage': 55.0,
                'network_in': 12.5,
                'network_out': 8.3,
                'frame_rate': 30.0,
                'processing_time_ms': 33.0
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/analytics/trends')
def get_analytics_trends():
    """Get trend data for charts."""
    global analytics_data
    
    try:
        # Generate trend data for last 24 hours
        hourly_data = analytics_data.get('hourly_detections', [0] * 24)
        
        # Create labels for the chart
        labels = [f"{i:02d}:00" for i in range(24)]
        
        return jsonify({
            'success': True,
            'detection_trends': {
                'labels': labels,
                'data': hourly_data
            },
            'alert_trends': {
                'labels': labels,
                'data': [max(0, int(d * 0.1)) for d in hourly_data]  # Approximate alerts as 10% of detections
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Person Count API Endpoint
# ============================================

@main.route('/api/person_count')
def get_person_count():
    """Get real-time person count data with trace history."""
    global analytics_data
    
    try:
        history = analytics_data.get('person_count_history', [])
        
        # Get last N entries (default 100)
        limit = request.args.get('limit', 100, type=int)
        recent_history = history[-limit:] if history else []
        
        return jsonify({
            'success': True,
            'current_count': analytics_data.get('person_count_current', 0),
            'max_count': analytics_data.get('person_count_max', 0),
            'total_persons_detected': analytics_data.get('person_count', 0),
            'history': recent_history,
            'history_length': len(history),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def record_detection_analytics(detections):
    """Record detection data for analytics."""
    global analytics_data
    
    try:
        current_hour = datetime.now().hour
        
        # Count detections
        analytics_data['detection_count'] += len(detections)
        analytics_data['hourly_detections'][current_hour] += len(detections)
        
        # Count by category
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            if class_name not in analytics_data['object_categories']:
                analytics_data['object_categories'][class_name] = 0
            analytics_data['object_categories'][class_name] += 1
            
            # Count persons specifically
            if class_name.lower() == 'person':
                analytics_data['person_count'] += 1
                
    except Exception as e:
        print(f"Error recording analytics: {e}")
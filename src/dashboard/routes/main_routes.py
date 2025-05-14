# Main routes for the dashboard
import numpy as np
from flask import Blueprint, render_template, Response, jsonify
import cv2
import os
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import necessary modules from our project
from video_processing.video_stream import VideoStream
from anomaly_detection.analyzer import AnomalyDetector
from alert_system.notifier import AlertSystem  # Changed from AlertNotifier

# Create a Blueprint for our main routes
main = Blueprint('main', __name__)

# Add this import at the top with other imports
from object_detection.detector import ObjectDetector
from object_detection.tracker import ObjectTracker

# Update global variables
video_stream = None
analyzer = None
notifier = None
detector = None
tracker = None
alerts_history = []  # Add this line to define alerts_history as a global variable

# Add a test alert
test_alert = {
    'type': 'test_alert',
    'message': 'This is a test alert',
    'timestamp': '2023-01-01 00:00:00'
}
alerts_history.append(test_alert)
print(f"Added test alert to history: {test_alert}")

# Update initialize_components function
@main.before_app_request
def initialize_components():
    """Initialize all components before the first request"""
    global video_stream, analyzer, notifier, detector, tracker
    
    # Only initialize once
    if video_stream is None:
        # Use webcam instead of video file
        video_stream = VideoStream(source=0)  # 0 is typically the default webcam
        
        # Initialize object detector and tracker
        detector = ObjectDetector()  # Add this line
        tracker = ObjectTracker()    # Add this line
        
        # Initialize anomaly analyzer
        analyzer = AnomalyDetector()
        
        # Initialize alert notifier
        notifier = AlertSystem()
        
        # Instead of using subscribe method, set up the callback manually
        original_generate_alert = notifier.generate_alert
        
        def wrapped_generate_alert(anomaly):
            alert = original_generate_alert(anomaly)
            alert_callback(alert)
            return alert
            
        notifier.generate_alert = wrapped_generate_alert
        
        # Add a restricted area (bottom half of the frame)
        frame_height, frame_width = 480, 640  # Default size, adjust if needed
        restricted_area = [
            (50, frame_height//2),
            (frame_width-50, frame_height//2),
            (frame_width-50, frame_height-50),
            (50, frame_height-50)
        ]
        analyzer.add_restricted_area(restricted_area)
        print(f"Added restricted area: {restricted_area}")

def alert_callback(alert_data):
    """Callback function for new alerts"""
    global alerts_history
    print(f"Alert callback received: {alert_data}")
    alerts_history.append(alert_data)
    print(f"Current alerts history size: {len(alerts_history)}")
    # Keep only the last 100 alerts
    if len(alerts_history) > 100:
        alerts_history = alerts_history[-100:]

# Update generate_frames function
def generate_frames():
    """Generate video frames with detection overlays"""
    global video_stream, analyzer, detector, tracker
    
    while True:
        # Get frame from video stream
        success, frame = video_stream.read_frame()
        if not success:
            print("Failed to read frame from video stream")
            # Create a blank frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No webcam feed available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Detect objects in the frame
            detections = detector.detect(frame)
            print(f"Detected {len(detections)} objects")
            
            # Add test detection if no detections found
            if not detections and success:  # If no detections but frame is valid
                print("No detections from model, adding a test detection")
                # Create a test detection in the center of the frame
                h, w = frame.shape[:2]
                test_bbox = [w//4, h//4, w*3//4, h*3//4]  # Center box
                test_detection = {
                    'bbox': test_bbox,
                    'class_id': 0,
                    'class_name': 'test_object',
                    'confidence': 0.9
                }
                detections = [test_detection]
            
            # Track objects across frames
            tracked_objects = tracker.update(detections)
            print(f"Tracking {len(tracked_objects)} objects")
            
            # Process frame for anomalies
            anomalies = analyzer.detect_anomalies(frame, tracked_objects)
            print(f"Detected anomalies: {anomalies}")
            
            # If anomalies are detected, generate alerts
            if anomalies:
                for anomaly in anomalies:
                    print(f"Generating alert for anomaly: {anomaly['type']}")
                    notifier.generate_alert(anomaly)
            
            # Draw bounding boxes and anomaly indicators
            for obj_id, obj in tracked_objects.items():
                bbox = obj['bbox']
                class_name = obj['class_name']
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Draw label
                cv2.putText(frame, f"{class_name} #{obj_id}", (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw restricted areas if analyzer is initialized
            if analyzer and hasattr(analyzer, 'restricted_areas'):
                for area in analyzer.restricted_areas:
                    pts = area.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                    cv2.putText(frame, "Restricted Area", (area[0][0], area[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the format expected by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@main.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@main.route('/video_feed')
def video_feed():
    """Route for streaming video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/alerts')
def get_alerts():
    """API endpoint to get recent alerts"""
    global alerts_history
    return jsonify(alerts_history)
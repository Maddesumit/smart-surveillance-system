#!/usr/bin/env python3
"""
Enhanced Smart Surveillance System - Main Application with Advanced Features

This is the enhanced main entry point that integrates all advanced surveillance features
including facial recognition, behavior analysis, person re-ID, and multi-camera sync.
"""

import os
import sys
import time
import logging
import signal
import threading
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

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
    from advanced_features.facial_recognition import FacialRecognitionSystem
    from advanced_features.behavior_analysis import BehaviorAnalyzer
    from advanced_features.person_reid import PersonReIdentification
    from advanced_features.multi_camera_sync import MultiCameraManager
    from advanced_features.real_time_analytics import AnalyticsEngine
    from advanced_features.advanced_alerts import AdvancedAlertSystem, AlertPriority
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'enhanced_surveillance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
dashboard_thread = None
display_window = None

class EnhancedSurveillanceSystem:
    """Enhanced surveillance system with all advanced features integrated."""
    
    def __init__(self):
        """Initialize the enhanced surveillance system."""
        self.video_stream = None
        self.detector = None
        self.tracker = None
        self.analyzer = None
        self.notifier = None
        
        # Advanced features (if available)
        self.face_recognition = None
        self.behavior_analyzer = None
        self.person_reid = None
        self.camera_manager = None
        self.analytics_engine = None
        self.advanced_alerts = None
        
        # System statistics
        self.stats = {
            'frames_processed': 0,
            'objects_detected': 0,
            'faces_recognized': 0,
            'behaviors_analyzed': 0,
            'alerts_generated': 0,
            'start_time': datetime.now()
        }
        
        # Display settings
        self.show_display = True
        self.display_fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
    
    def initialize_core_components(self):
        """Initialize core surveillance components."""
        logger.info("Initializing core surveillance components...")
        
        # Initialize video stream
        self.video_stream = self._initialize_video_stream()
        
        # Initialize object detection and tracking
        logger.info("Initializing object detector and tracker...")
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        
        # Initialize anomaly detection
        logger.info("Initializing anomaly detector...")
        self.analyzer = AnomalyDetector()
        self._setup_restricted_areas()
        
        # Initialize alert system
        logger.info("Initializing alert system...")
        self.notifier = AlertSystem()
        
        logger.info("Core components initialized successfully")
    
    def _initialize_video_stream(self):
        """Initialize video stream with fallback handling."""
        logger.info("Initializing video stream...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                video_stream = VideoStream(source=0)  # Use webcam
                success, _ = video_stream.read_frame()
                if success:
                    logger.info(f"Successfully connected to video source on attempt {attempt+1}")
                    return video_stream
                else:
                    logger.warning(f"Could not read frame on attempt {attempt+1}")
                    if video_stream:
                        video_stream.release()
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error initializing video stream on attempt {attempt+1}: {str(e)}")
                time.sleep(1)
        
        logger.error("Failed to initialize video stream after multiple attempts")
        return None
    
    def _setup_restricted_areas(self):
        """Setup restricted areas for anomaly detection."""
        # Add default restricted area (example: bottom half of frame)
        frame_height, frame_width = 480, 640
        restricted_area = [
            (50, frame_height//2),
            (frame_width-50, frame_height//2),
            (frame_width-50, frame_height-50),
            (50, frame_height-50)
        ]
        self.analyzer.add_restricted_area(restricted_area)
        logger.info("Default restricted area configured")
    
    def initialize_advanced_features(self):
        """Initialize advanced surveillance features."""
        if not ADVANCED_FEATURES_AVAILABLE:
            logger.warning("Advanced features not available - running with core features only")
            return
        
        logger.info("Initializing advanced surveillance features...")
        
        try:
            # Initialize facial recognition
            logger.info("Initializing facial recognition system...")
            self.face_recognition = FacialRecognitionSystem()
            
            # Initialize behavior analysis
            logger.info("Initializing behavior analysis system...")
            self.behavior_analyzer = BehaviorAnalyzer()
            
            # Initialize person re-identification
            logger.info("Initializing person re-identification system...")
            self.person_reid = PersonReIdentification()
            self.person_reid.register_camera("main_camera", "Main Surveillance Camera")
            
            # Initialize multi-camera management
            logger.info("Initializing multi-camera management...")
            self.camera_manager = MultiCameraManager()
            self.camera_manager.add_camera("main_camera", "rtsp://main_camera", (0, 0))
            
            # Initialize analytics engine
            logger.info("Initializing real-time analytics...")
            self.analytics_engine = AnalyticsEngine()
            self.analytics_engine.start_analytics()
            
            # Initialize advanced alerts
            logger.info("Initializing advanced alert system...")
            self.advanced_alerts = AdvancedAlertSystem()
            self.advanced_alerts.start_processing()
            
            logger.info("Advanced features initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing advanced features: {str(e)}")
            self.face_recognition = None
            self.behavior_analyzer = None
            self.person_reid = None
            self.camera_manager = None
            self.analytics_engine = None
            self.advanced_alerts = None
    
    def start_dashboard(self):
        """Start the web dashboard in a separate thread."""
        try:
            from dashboard.app import app
            dashboard_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=8082, debug=False, use_reloader=False)
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            logger.info("Web dashboard started on port 8082")
            return dashboard_thread
        except Exception as e:
            logger.error(f"Failed to start dashboard: {str(e)}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the entire surveillance pipeline."""
        processed_frame = frame.copy()
        
        try:
            # Core detection and tracking
            detections = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detections)
            
            # Update statistics
            self.stats['objects_detected'] += len(detections)
            
            # Core anomaly detection
            anomalies = self.analyzer.detect_anomalies(frame, tracked_objects)
            
            # Process with advanced features if available
            if ADVANCED_FEATURES_AVAILABLE:
                processed_frame = self._process_advanced_features(
                    frame, detections, tracked_objects, anomalies
                )
            
            # Generate alerts
            self._generate_alerts(anomalies, detections)
            
            # Draw visualizations
            processed_frame = self._draw_visualizations(
                processed_frame, detections, tracked_objects, anomalies
            )
            
            # Update system statistics
            self._update_statistics()
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
        
        return processed_frame
    
    def _process_advanced_features(self, frame: np.ndarray, detections: List, 
                                 tracked_objects: Dict, anomalies: List) -> np.ndarray:
        """Process frame through advanced features."""
        processed_frame = frame.copy()
        
        try:
            # Facial recognition
            if self.face_recognition:
                face_detections = self.face_recognition.detect_faces(frame)
                self.stats['faces_recognized'] += len([f for f in face_detections if f['is_known']])
                processed_frame = self.face_recognition.draw_face_detections(processed_frame, face_detections)
            
            # Behavior analysis
            if self.behavior_analyzer:
                person_detections = [d for d in detections if d.get('class_name') == 'person']
                behavior_results = self.behavior_analyzer.analyze_behavior(
                    frame, person_detections, self.stats['frames_processed']
                )
                self.stats['behaviors_analyzed'] += len(behavior_results)
                processed_frame = self.behavior_analyzer.draw_behavior_analysis(processed_frame, behavior_results)
                
                # Generate behavior-based alerts
                for behavior in behavior_results:
                    if behavior.get('is_suspicious') and self.advanced_alerts:
                        self.advanced_alerts.create_alert(
                            alert_type="suspicious_behavior",
                            title=f"Suspicious Behavior Detected",
                            description=f"Person {behavior.get('person_id')} showing suspicious {behavior.get('activity')} behavior",
                            priority=AlertPriority.HIGH,
                            source_location="main_camera",
                            confidence=0.8,
                            image=frame
                        )
            
            # Person re-identification
            if self.person_reid:
                person_detections = [d for d in detections if d.get('class_name') == 'person']
                reid_results = self.person_reid.process_detections(
                    frame, person_detections, "main_camera"
                )
            
            # Real-time analytics
            if self.analytics_engine:
                # Record metrics
                self.analytics_engine.record_metric("objects_detected", len(detections))
                self.analytics_engine.record_metric("anomalies_detected", len(anomalies))
                self.analytics_engine.record_metric("frames_processed", 1)
                
                if face_detections:
                    self.analytics_engine.record_metric("faces_detected", len(face_detections))
        
        except Exception as e:
            logger.error(f"Error in advanced features processing: {str(e)}")
        
        return processed_frame
    
    def _generate_alerts(self, anomalies: List, detections: List):
        """Generate alerts for detected anomalies."""
        try:
            # Core alerts
            for anomaly in anomalies:
                self.notifier.generate_alert(anomaly)
                self.stats['alerts_generated'] += 1
                
                # Enhanced alerts if available
                if self.advanced_alerts:
                    priority = AlertPriority.MEDIUM
                    if anomaly.get('type') == 'restricted_area_violation':
                        priority = AlertPriority.HIGH
                    elif anomaly.get('type') == 'unattended_object':
                        priority = AlertPriority.CRITICAL
                    
                    self.advanced_alerts.create_alert(
                        alert_type=anomaly.get('type', 'unknown'),
                        title=f"Security Alert: {anomaly.get('type', 'Unknown').replace('_', ' ').title()}",
                        description=anomaly.get('message', 'Anomaly detected'),
                        priority=priority,
                        source_location="main_camera",
                        confidence=anomaly.get('confidence', 0.5),
                        metadata=anomaly
                    )
        
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
    
    def _draw_visualizations(self, frame: np.ndarray, detections: List, 
                           tracked_objects: Dict, anomalies: List) -> np.ndarray:
        """Draw visualizations on the frame."""
        try:
            # Draw object detections
            for detection in detections:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{detection.get('class_name', 'Unknown')} ({detection.get('confidence', 0):.2f})"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw tracking IDs
            for obj_id, obj_data in tracked_objects.items():
                bbox = obj_data.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    cv2.putText(frame, f"ID: {obj_id}", (x, y + h + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw anomaly indicators
            for anomaly in anomalies:
                bbox = anomaly.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame, "ANOMALY", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw system info
            self._draw_system_info(frame)
        
        except Exception as e:
            logger.error(f"Error drawing visualizations: {str(e)}")
        
        return frame
    
    def _draw_system_info(self, frame: np.ndarray):
        """Draw system information on the frame."""
        try:
            # Update FPS calculation
            current_time = time.time()
            self.fps_counter += 1
            if current_time - self.last_fps_time >= 1.0:
                self.display_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = current_time
            
            # System info text
            info_lines = [
                f"FPS: {self.display_fps}",
                f"Frames: {self.stats['frames_processed']}",
                f"Objects: {self.stats['objects_detected']}",
                f"Alerts: {self.stats['alerts_generated']}"
            ]
            
            if ADVANCED_FEATURES_AVAILABLE:
                info_lines.extend([
                    f"Faces: {self.stats['faces_recognized']}",
                    f"Behaviors: {self.stats['behaviors_analyzed']}"
                ])
            
            # Draw info box
            box_height = len(info_lines) * 25 + 10
            cv2.rectangle(frame, (10, 10), (250, box_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (250, box_height), (255, 255, 255), 2)
            
            # Draw text
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (15, 30 + i * 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        except Exception as e:
            logger.error(f"Error drawing system info: {str(e)}")
    
    def _update_statistics(self):
        """Update system statistics."""
        self.stats['frames_processed'] += 1
        
        # Log stats periodically
        if self.stats['frames_processed'] % 100 == 0:
            runtime = (datetime.now() - self.stats['start_time']).total_seconds()
            avg_fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
            
            logger.info(f"System stats - Frames: {self.stats['frames_processed']}, "
                       f"Objects: {self.stats['objects_detected']}, "
                       f"Alerts: {self.stats['alerts_generated']}, "
                       f"Avg FPS: {avg_fps:.2f}")
    
    def run_main_loop(self):
        """Run the main surveillance processing loop."""
        logger.info("Starting main surveillance loop...")
        
        global running
        
        # Initialize display window if enabled
        if self.show_display:
            cv2.namedWindow("Enhanced Smart Surveillance", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Smart Surveillance", 1280, 720)
        
        try:
            while running:
                if not self.video_stream:
                    logger.error("Video stream not available")
                    break
                
                # Read frame
                success, frame = self.video_stream.read_frame()
                if not success:
                    logger.warning("Failed to read frame from video stream")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                if self.show_display:
                    cv2.imshow("Enhanced Smart Surveillance", processed_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit key pressed")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"screenshots/enhanced_surveillance_{timestamp}.jpg"
                        os.makedirs("screenshots", exist_ok=True)
                        cv2.imwrite(screenshot_path, processed_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Short delay to control processing rate
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up system resources."""
        logger.info("Cleaning up system resources...")
        
        global running
        running = False
        
        # Cleanup video stream
        if self.video_stream:
            self.video_stream.release()
        
        # Cleanup advanced features
        if self.analytics_engine:
            self.analytics_engine.stop_analytics()
        
        if self.advanced_alerts:
            self.advanced_alerts.stop_processing()
        
        # Close display windows
        cv2.destroyAllWindows()
        
        # Final statistics
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info(f"System shutdown - Runtime: {runtime:.2f}s, "
                   f"Total frames: {self.stats['frames_processed']}, "
                   f"Total alerts: {self.stats['alerts_generated']}")

def signal_handler(sig, frame):
    """Handle termination signals."""
    global running
    logger.info("Shutdown signal received. Stopping system...")
    running = False

def main():
    """Main function to run the enhanced surveillance system."""
    logger.info("🧠 Starting Enhanced Smart Surveillance System")
    
    # Create and initialize system
    surveillance_system = EnhancedSurveillanceSystem()
    
    try:
        # Initialize core components
        surveillance_system.initialize_core_components()
        
        # Initialize advanced features
        surveillance_system.initialize_advanced_features()
        
        # Start web dashboard
        dashboard_thread = surveillance_system.start_dashboard()
        
        logger.info("System initialized successfully")
        logger.info("Press 'q' in the video window to quit, or 's' to save screenshot")
        logger.info("Web dashboard available at: http://localhost:8082")
        
        # Run main processing loop
        surveillance_system.run_main_loop()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        logger.info("Enhanced Smart Surveillance System shutdown complete")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the enhanced system
    main()

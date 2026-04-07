#!/usr/bin/env python3
"""
Smart Surveillance System - Ultra Enhanced Version

This is the main entry point for the enhanced surveillance system with
all advanced features enabled including the new safety and recognition modules.
"""

import os
import sys
import time
import logging
import signal
import threading
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core components
from video_processing.video_stream import VideoStream
from object_detection.detector import ObjectDetector
from object_detection.tracker import ObjectTracker
from anomaly_detection.analyzer import AnomalyDetector
from alert_system.notifier import AlertSystem

# Import existing advanced features
try:
    from advanced_features.facial_recognition import FacialRecognitionSystem
    FACIAL_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Facial recognition not available: {e}")
    FACIAL_RECOGNITION_AVAILABLE = False

try:
    from advanced_features.person_reid import PersonReID
    PERSON_REID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Person Re-ID not available: {e}")
    PERSON_REID_AVAILABLE = False

try:
    from advanced_features.behavior_analysis import BehaviorAnalyzer
    BEHAVIOR_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Behavior analysis not available: {e}")
    BEHAVIOR_ANALYSIS_AVAILABLE = False

# Import NEW enhanced features
try:
    from advanced_features.violence_detection import ViolenceDetector
    VIOLENCE_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Violence detection not available: {e}")
    VIOLENCE_DETECTION_AVAILABLE = False

try:
    from advanced_features.weapon_detection import WeaponDetector
    WEAPON_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Weapon detection not available: {e}")
    WEAPON_DETECTION_AVAILABLE = False

try:
    from advanced_features.fire_smoke_detection import FireSmokeDetector
    FIRE_SMOKE_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Fire/smoke detection not available: {e}")
    FIRE_SMOKE_DETECTION_AVAILABLE = False

try:
    from advanced_features.license_plate_recognition import LicensePlateRecognizer
    LICENSE_PLATE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: License plate recognition not available: {e}")
    LICENSE_PLATE_AVAILABLE = False

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'ultra_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('ultra_enhanced_surveillance')

# Global flag to control the main loop
running = True

# Dashboard thread
dashboard_thread = None

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down the system."""
    global running
    logger.info("Shutdown signal received. Stopping system...")
    running = False

def start_dashboard():
    """Start the dashboard in a separate thread."""
    from dashboard.app import app
    app.run(host='0.0.0.0', port=8082, debug=False, use_reloader=False)

def print_system_status():
    """Print system status and available features."""
    print("\n" + "="*70)
    print("🚀 SMART SURVEILLANCE SYSTEM - ULTRA ENHANCED VERSION")
    print("="*70)
    print("\n📊 SYSTEM STATUS:")
    print("-" * 70)
    
    # Core features
    print("\n✅ CORE FEATURES:")
    print("  • Object Detection (YOLOv8)")
    print("  • Object Tracking")
    print("  • Anomaly Detection")
    print("  • Alert System")
    print("  • Web Dashboard")
    
    # Existing advanced features
    print("\n✅ EXISTING ADVANCED FEATURES:")
    features_count = 0
    if FACIAL_RECOGNITION_AVAILABLE:
        print("  • Facial Recognition")
        features_count += 1
    if PERSON_REID_AVAILABLE:
        print("  • Person Re-Identification")
        features_count += 1
    if BEHAVIOR_ANALYSIS_AVAILABLE:
        print("  • Behavior Analysis")
        features_count += 1
    
    # NEW enhanced features
    print("\n🆕 NEW ENHANCED FEATURES:")
    new_features_count = 0
    if VIOLENCE_DETECTION_AVAILABLE:
        print("  • Violence & Fight Detection")
        new_features_count += 1
    if WEAPON_DETECTION_AVAILABLE:
        print("  • Weapon Detection")
        new_features_count += 1
    if FIRE_SMOKE_DETECTION_AVAILABLE:
        print("  • Fire & Smoke Detection")
        new_features_count += 1
    if LICENSE_PLATE_AVAILABLE:
        print("  • License Plate Recognition")
        new_features_count += 1
    
    total_features = features_count + new_features_count
    print(f"\n📈 TOTAL ADVANCED FEATURES: {total_features}")
    print(f"   • Existing: {features_count}")
    print(f"   • New: {new_features_count}")
    
    print("\n" + "="*70)
    print("🌐 Dashboard: http://localhost:8082")
    print("📝 Logs: " + log_file)
    print("="*70 + "\n")

def main():
    """Main function to initialize and run the enhanced surveillance system."""
    global dashboard_thread, running
    
    # Print system status
    print_system_status()
    
    logger.info("Starting Ultra Enhanced Smart Surveillance System")
    
    try:
        # Initialize video stream
        logger.info("Initializing video stream...")
        video_stream = VideoStream(source=0)
        
        # Initialize core components
        logger.info("Initializing core components...")
        detector = ObjectDetector()
        tracker = ObjectTracker()
        analyzer = AnomalyDetector()
        notifier = AlertSystem()
        
        # Initialize existing advanced features
        face_recognition = None
        person_reid = None
        behavior_analyzer = None
        
        if FACIAL_RECOGNITION_AVAILABLE:
            logger.info("Initializing facial recognition...")
            try:
                face_recognition = FacialRecognitionSystem()
            except Exception as e:
                logger.error(f"Failed to initialize facial recognition: {str(e)}")
        
        if PERSON_REID_AVAILABLE:
            logger.info("Initializing person re-identification...")
            try:
                person_reid = PersonReID()
            except Exception as e:
                logger.error(f"Failed to initialize person re-ID: {str(e)}")
        
        if BEHAVIOR_ANALYSIS_AVAILABLE:
            logger.info("Initializing behavior analysis...")
            try:
                behavior_analyzer = BehaviorAnalyzer()
            except Exception as e:
                logger.error(f"Failed to initialize behavior analysis: {str(e)}")
        
        # Initialize NEW enhanced features
        violence_detector = None
        weapon_detector = None
        fire_smoke_detector = None
        plate_recognizer = None
        
        if VIOLENCE_DETECTION_AVAILABLE:
            logger.info("Initializing violence detection...")
            try:
                violence_detector = ViolenceDetector()
                logger.info("✅ Violence detection ready")
            except Exception as e:
                logger.error(f"Failed to initialize violence detection: {str(e)}")
        
        if WEAPON_DETECTION_AVAILABLE:
            logger.info("Initializing weapon detection...")
            try:
                weapon_detector = WeaponDetector()
                logger.info("✅ Weapon detection ready")
            except Exception as e:
                logger.error(f"Failed to initialize weapon detection: {str(e)}")
        
        if FIRE_SMOKE_DETECTION_AVAILABLE:
            logger.info("Initializing fire/smoke detection...")
            try:
                fire_smoke_detector = FireSmokeDetector()
                logger.info("✅ Fire/smoke detection ready")
            except Exception as e:
                logger.error(f"Failed to initialize fire/smoke detection: {str(e)}")
        
        if LICENSE_PLATE_AVAILABLE:
            logger.info("Initializing license plate recognition...")
            try:
                plate_recognizer = LicensePlateRecognizer()
                logger.info("✅ License plate recognition ready")
            except Exception as e:
                logger.error(f"Failed to initialize license plate recognition: {str(e)}")
        
        # Start dashboard
        logger.info("Starting dashboard...")
        dashboard_thread = threading.Thread(target=start_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logger.info("🎉 System initialized successfully!")
        logger.info("Press Ctrl+C to stop the system")
        
        # Main processing loop
        frame_count = 0
        while running:
            # Read frame
            success, frame = video_stream.read_frame()
            
            if not success:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Core detection and tracking
            detections = detector.detect(frame)
            tracked_objects = tracker.update(detections)
            
            # Filter person detections
            person_detections = [d for d in tracked_objects.values() 
                               if d.get('class_name') == 'person']
            
            # Filter vehicle detections
            vehicle_detections = [d for d in tracked_objects.values() 
                                if d.get('class_name') in ['car', 'truck', 'bus', 'motorcycle']]
            
            # Process with existing advanced features
            if face_recognition and person_detections:
                try:
                    face_detections = face_recognition.detect_faces(frame)
                except Exception as e:
                    logger.error(f"Face recognition error: {str(e)}")
            
            if behavior_analyzer and person_detections:
                try:
                    behavior_results = behavior_analyzer.analyze_behavior(
                        frame, person_detections, frame_count
                    )
                except Exception as e:
                    logger.error(f"Behavior analysis error: {str(e)}")
            
            # Process with NEW enhanced features
            if violence_detector and len(person_detections) >= 2:
                try:
                    violence_events = violence_detector.detect_violence(
                        frame, person_detections
                    )
                    for event in violence_events:
                        logger.warning(f"⚠️  VIOLENCE DETECTED: {event['threat_level']}")
                        notifier.generate_alert({
                            'type': 'violence_detected',
                            'severity': event['threat_level'],
                            'message': event['description']
                        })
                except Exception as e:
                    logger.error(f"Violence detection error: {str(e)}")
            
            if weapon_detector:
                try:
                    weapon_detections = weapon_detector.detect_weapons(
                        frame, person_detections
                    )
                    for weapon in weapon_detections:
                        logger.critical(f"🚨 WEAPON DETECTED: {weapon['weapon_type']} - {weapon['threat_level']}")
                        notifier.generate_alert({
                            'type': 'weapon_detected',
                            'severity': 'CRITICAL',
                            'message': weapon['description']
                        })
                except Exception as e:
                    logger.error(f"Weapon detection error: {str(e)}")
            
            if fire_smoke_detector:
                try:
                    fire_smoke_events = fire_smoke_detector.detect_fire_smoke(frame)
                    for event in fire_smoke_events:
                        if event.get('verified'):
                            logger.critical(f"🔥 {event['type'].upper()} DETECTED: {event.get('severity', 'UNKNOWN')}")
                            notifier.generate_alert({
                                'type': event['type'],
                                'severity': 'CRITICAL',
                                'message': f"{event['type'].upper()} detected - Emergency response required!"
                            })
                except Exception as e:
                    logger.error(f"Fire/smoke detection error: {str(e)}")
            
            if plate_recognizer and vehicle_detections:
                try:
                    plate_results = plate_recognizer.detect_and_recognize_plates(
                        frame, vehicle_detections
                    )
                    for plate in plate_results:
                        logger.info(f"🚗 License Plate: {plate['plate_number']}")
                        if plate['in_blacklist']:
                            logger.warning(f"⚠️  BLACKLISTED VEHICLE: {plate['plate_number']}")
                            notifier.generate_alert({
                                'type': 'blacklisted_vehicle',
                                'severity': 'HIGH',
                                'message': f"Blacklisted vehicle detected: {plate['plate_number']}"
                            })
                except Exception as e:
                    logger.error(f"License plate recognition error: {str(e)}")
            
            # Anomaly detection
            anomalies = analyzer.detect_anomalies(frame, tracked_objects)
            for anomaly in anomalies:
                notifier.generate_alert(anomaly)
            
            # Short delay
            time.sleep(0.01)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if 'video_stream' in locals():
            video_stream.release()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the main function
    main()

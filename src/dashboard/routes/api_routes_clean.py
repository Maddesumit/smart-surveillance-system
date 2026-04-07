"""
Professional API Routes for Smart Surveillance Dashboard
Advanced features integration with RESTful API endpoints
"""

from flask import Blueprint, jsonify, request
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import advanced features
try:
    from advanced_features.facial_recognition import FacialRecognitionSystem
    from advanced_features.behavior_analysis import BehaviorAnalyzer
    from advanced_features.person_reid import PersonReID
    from advanced_features.multi_camera_sync import MultiCameraSync
    from advanced_features.real_time_analytics import RealTimeAnalytics
    from advanced_features.advanced_alerts import AdvancedAlertSystem
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Create API Blueprint
api = Blueprint('api', __name__)

# Global references to advanced features
facial_recognition = None
behavior_analyzer = None
person_reid = None
multi_camera = None
analytics = None
advanced_alerts = None

def initialize_advanced_features():
    """Initialize advanced features if available."""
    global facial_recognition, behavior_analyzer, person_reid, multi_camera, analytics, advanced_alerts
    
    if not ADVANCED_FEATURES_AVAILABLE:
        return False
    
    try:
        if facial_recognition is None:
            facial_recognition = FacialRecognitionSystem()
        if behavior_analyzer is None:
            behavior_analyzer = BehaviorAnalyzer()
        if person_reid is None:
            person_reid = PersonReID()
        if multi_camera is None:
            multi_camera = MultiCameraSync()
        if analytics is None:
            analytics = RealTimeAnalytics()
        if advanced_alerts is None:
            advanced_alerts = AdvancedAlertSystem()
        return True
    except Exception as e:
        print(f"Error initializing advanced features: {e}")
        return False

# Facial Recognition API Endpoints
@api.route('/facial_recognition/system_status')
def facial_recognition_system_status():
    """Get facial recognition system status."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'available': False, 'error': 'Facial recognition not available'})
    
    try:
        stats = facial_recognition.get_recognition_stats()
        known_persons = facial_recognition.get_known_persons()
        
        return jsonify({
            'available': True,
            'stats': stats,
            'known_persons_count': len(known_persons),
            'known_persons': known_persons
        })
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})

@api.route('/analytics/dashboard_data')
def get_dashboard_data():
    """Get comprehensive dashboard analytics data."""
    initialize_advanced_features()
    
    try:
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0,
                'uptime': '0:00:00'
            },
            'detection_stats': {
                'objects_detected_today': 0,
                'faces_recognized_today': 0,
                'anomalies_detected_today': 0,
                'alerts_generated_today': 0
            },
            'advanced_features_status': {
                'facial_recognition': facial_recognition is not None,
                'behavior_analysis': behavior_analyzer is not None,
                'person_reid': person_reid is not None,
                'multi_camera': multi_camera is not None,
                'real_time_analytics': analytics is not None
            }
        }
        
        # Add facial recognition stats if available
        if facial_recognition:
            dashboard_data['facial_recognition'] = facial_recognition.get_recognition_stats()
        
        # Add behavior analysis stats if available
        if behavior_analyzer:
            dashboard_data['behavior_analysis'] = behavior_analyzer.get_analysis_stats()
        
        # Add person ReID stats if available
        if person_reid:
            dashboard_data['person_reid'] = person_reid.get_reid_stats()
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/behavior_analysis/status')
def behavior_analysis_status():
    """Get behavior analysis system status."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'available': False, 'error': 'Behavior analysis not available'})
    
    try:
        stats = behavior_analyzer.get_analysis_stats()
        return jsonify({
            'available': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})

@api.route('/person_reid/status')
def person_reid_status():
    """Get person re-identification system status."""
    initialize_advanced_features()
    
    if not person_reid:
        return jsonify({'available': False, 'error': 'Person ReID not available'})
    
    try:
        stats = person_reid.get_reid_stats()
        return jsonify({
            'available': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})

@api.route('/system/config')
def get_system_config():
    """Get system configuration."""
    try:
        config = {
            'detection_confidence_threshold': 0.5,
            'tracking_max_disappeared': 50,
            'facial_recognition_tolerance': 0.6,
            'alert_throttle_time': 30,
            'video_fps': 30,
            'video_resolution': '1280x720',
            'advanced_features_enabled': ADVANCED_FEATURES_AVAILABLE
        }
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health Check Endpoint
@api.route('/health')
def system_health_check():
    """System health check endpoint."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'core_surveillance': True,
                'advanced_features': ADVANCED_FEATURES_AVAILABLE,
                'database': True,
                'api': True
            },
            'version': '2.0.0'
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

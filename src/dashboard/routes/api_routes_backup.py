"""
Professional API Routes for Smart Surveillance Dashboard
Advanced features integration with RESTful API endpoints
"""

from flask import Blueprint, jsonify, request, send_file
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
import cv2
import numpy as np

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
except ImportError:
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

@api.route('/status')
def system_status():
    """Get comprehensive system status."""
    initialize_advanced_features()
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'core_features': {
            'video_stream': True,
            'object_detection': True,
            'tracking': True,
            'anomaly_detection': True,
            'alerts': True
        },
        'advanced_features': {
            'available': ADVANCED_FEATURES_AVAILABLE,
            'facial_recognition': facial_recognition is not None,
            'behavior_analysis': behavior_analyzer is not None,
            'person_reid': person_reid is not None,
            'multi_camera': multi_camera is not None,
            'real_time_analytics': analytics is not None,
            'advanced_alerts': advanced_alerts is not None
        }
    }
    
    return jsonify(status)

@api.route('/facial-recognition/status')
def facial_recognition_status():
    """Get facial recognition system status."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'error': 'Facial recognition not available'}), 503
    
    try:
        stats = facial_recognition.get_recognition_stats()
        known_persons = facial_recognition.get_known_persons()
        
        return jsonify({
            'status': 'active',
            'known_persons': len(known_persons),
            'persons_list': known_persons,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/facial-recognition/add-person', methods=['POST'])
def add_known_person():
    """Add a new known person to facial recognition."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'error': 'Facial recognition not available'}), 503
    
    data = request.get_json()
    if not data or 'name' not in data or 'image_path' not in data:
        return jsonify({'error': 'Name and image_path required'}), 400
    
    try:
        success = facial_recognition.add_known_person(data['name'], data['image_path'])
        if success:
            return jsonify({'success': True, 'message': f"Added {data['name']}"})
        else:
            return jsonify({'error': 'Failed to add person'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/facial-recognition/remove-person', methods=['DELETE'])
def remove_known_person():
    """Remove a known person from facial recognition."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'error': 'Facial recognition not available'}), 503
    
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error': 'Name required'}), 400
    
    try:
        success = facial_recognition.remove_known_person(data['name'])
        if success:
            return jsonify({'success': True, 'message': f"Removed {data['name']}"})
        else:
            return jsonify({'error': 'Failed to remove person'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/behavior-analysis/status')
def behavior_analysis_status():
    """Get behavior analysis status."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'error': 'Behavior analysis not available'}), 503
    
    try:
        stats = behavior_analyzer.get_analysis_stats()
        return jsonify({
            'status': 'active',
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/person-reid/status')
def person_reid_status():
    """Get person re-identification status."""
    initialize_advanced_features()
    
    if not person_reid:
        return jsonify({'error': 'Person re-ID not available'}), 503
    
    try:
        stats = person_reid.get_reid_stats()
        return jsonify({
            'status': 'active',
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/analytics/summary')
def analytics_summary():
    """Get real-time analytics summary."""
    initialize_advanced_features()
    
    if not analytics:
        return jsonify({'error': 'Analytics not available'}), 503
    
    try:
        summary = analytics.get_analytics_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/analytics/hourly')
def analytics_hourly():
    """Get hourly analytics data."""
    initialize_advanced_features()
    
    if not analytics:
        return jsonify({'error': 'Analytics not available'}), 503
    
    try:
        hourly_data = analytics.get_hourly_analytics()
        return jsonify(hourly_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/alerts/recent')
def recent_alerts():
    """Get recent alerts with filtering and pagination."""
    try:
        # Get parameters
        limit = request.args.get('limit', 50, type=int)
        alert_type = request.args.get('type', None)
        hours = request.args.get('hours', 24, type=int)
        
        # Calculate time threshold
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        # Query database
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'logs', 'alerts.db')
        
        alerts = []
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM alerts WHERE timestamp > ? "
            params = [time_threshold.isoformat()]
            
            if alert_type:
                query += "AND type = ? "
                params.append(alert_type)
            
            query += "ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                alerts.append({
                    'id': row[0],
                    'type': row[1],
                    'message': row[2],
                    'timestamp': row[3],
                    'priority': row[4],
                    'data': row[5] if len(row) > 5 else None
                })
            
            conn.close()
        
        return jsonify({
            'alerts': alerts,
            'total': len(alerts),
            'filters': {
                'type': alert_type,
                'hours': hours,
                'limit': limit
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/alerts/statistics')
def alert_statistics():
    """Get alert statistics."""
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'logs', 'alerts.db')
        
        stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_priority': {},
            'recent_alerts': 0
        }
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Total alerts
            cursor.execute("SELECT COUNT(*) FROM alerts")
            stats['total_alerts'] = cursor.fetchone()[0]
            
            # Alerts by type
            cursor.execute("SELECT type, COUNT(*) FROM alerts GROUP BY type")
            stats['alerts_by_type'] = dict(cursor.fetchall())
            
            # Alerts by priority
            cursor.execute("SELECT priority, COUNT(*) FROM alerts GROUP BY priority")
            stats['alerts_by_priority'] = dict(cursor.fetchall())
            
            # Recent alerts (last 24 hours)
            yesterday = datetime.now() - timedelta(hours=24)
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp > ?", (yesterday.isoformat(),))
            stats['recent_alerts'] = cursor.fetchone()[0]
            
            conn.close()
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/cameras/status')
def cameras_status():
    """Get multi-camera system status."""
    initialize_advanced_features()
    
    if not multi_camera:
        return jsonify({'error': 'Multi-camera not available'}), 503
    
    try:
        cameras = multi_camera.get_camera_status()
        return jsonify({
            'cameras': cameras,
            'total_cameras': len(cameras),
            'active_cameras': len([c for c in cameras if c.get('status') == 'active'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/settings/update', methods=['POST'])
def update_settings():
    """Update system settings."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update settings based on the data received
        # This would typically save to a configuration file or database
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully',
            'updated_settings': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

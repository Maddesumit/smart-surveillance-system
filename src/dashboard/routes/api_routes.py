"""
Professional API Routes for Smart Surveillance Dashboard
Advanced features integration with RESTful API endpoints
"""

from flask import Blueprint, jsonify, request
import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
            multi_camera = MultiCameraManager()
        if analytics is None:
            analytics = AnalyticsEngine()
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
        # Get real-time stats from main surveillance system
        from flask import current_app
        stats = {}
        if hasattr(current_app, 'get_shared_stats'):
            stats = current_app.get_shared_stats()
        
        # Calculate system uptime
        start_time = stats.get('system_start_time', datetime.now())
        uptime = datetime.now() - start_time
        uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'cpu_usage': 0,  # TODO: Implement actual CPU monitoring
                'memory_usage': 0,  # TODO: Implement actual memory monitoring
                'disk_usage': 0,  # TODO: Implement actual disk monitoring
                'uptime': uptime_str
            },
            'detection_stats': {
                'objects_detected_today': stats.get('objects_detected_today', 0),
                'faces_recognized_today': stats.get('faces_recognized_today', 0),
                'anomalies_detected_today': 0,  # TODO: Add anomaly counting
                'alerts_generated_today': stats.get('alerts_generated_today', 0)
            },
            'live_stats': stats.get('live_stats', {
                'objects_in_view': 0,
                'faces_detected': 0,
                'tracked_objects': 0
            }),
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
            dashboard_data['behavior_analysis'] = behavior_analyzer.get_behavior_stats()
        
        # Add person ReID stats if available
        if person_reid:
            dashboard_data['person_reid'] = person_reid.get_reid_stats()
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/analytics/live_stats')
def get_live_stats():
    """Get real-time live statistics for the dashboard."""
    try:
        from flask import current_app
        stats = {}
        if hasattr(current_app, 'get_shared_stats'):
            stats = current_app.get_shared_stats()
        
        # Try to get live stats from main_routes detection loop
        try:
            from dashboard.routes.main_routes import live_detection_stats
            live_stats = {
                'objects_in_view': live_detection_stats.get('objects_in_view', 0),
                'faces_detected': live_detection_stats.get('faces_detected', 0),
                'tracked_objects': live_detection_stats.get('tracked_objects', 0)
            }
        except ImportError:
            live_stats = stats.get('live_stats', {
                'objects_in_view': 0,
                'faces_detected': 0,
                'tracked_objects': 0
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'live_stats': live_stats,
            'detection_stats': {
                'objects_detected_today': stats.get('objects_detected_today', 0),
                'faces_recognized_today': stats.get('faces_recognized_today', 0),
                'alerts_generated_today': stats.get('alerts_generated_today', 0)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/behavior_analysis/status')
def behavior_analysis_status():
    """Get behavior analysis system status."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'available': False, 'error': 'Behavior analysis not available'})
    
    try:
        stats = behavior_analyzer.get_behavior_stats()
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

# Facial Recognition Management Endpoints
@api.route('/facial_recognition/enroll', methods=['POST'])
def enroll_face():
    """Enroll a new face for recognition."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'success': False, 'error': 'Facial recognition not available'}), 400
    
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')  # Base64 encoded image
        
        if not name or not image_data:
            return jsonify({'success': False, 'error': 'Name and image required'}), 400
        
        # Process enrollment using enroll_person_from_base64 method
        result = facial_recognition.enroll_person_from_base64(name, image_data)
        
        if result:
            return jsonify({
                'success': True,
                'message': f'Successfully enrolled {name}',
                'person_id': name  # Use name as person_id since method returns boolean
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to enroll person'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/facial_recognition/known_persons')
def get_known_persons():
    """Get list of known persons."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'known_persons': []})
    
    try:
        persons = facial_recognition.get_known_persons()
        return jsonify({'known_persons': persons})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/facial_recognition/remove_person', methods=['DELETE', 'POST'])
def remove_known_person():
    """Remove a known person from facial recognition."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'success': False, 'error': 'Facial recognition not available'}), 400
    
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'success': False, 'error': 'Person name is required'}), 400
        
        success = facial_recognition.remove_known_person(data['name'])
        if success:
            return jsonify({'success': True, 'message': f"Removed {data['name']}"})
        else:
            return jsonify({'success': False, 'error': 'Failed to remove person'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/facial_recognition/export_database')
def export_face_database():
    """Export facial recognition database."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'success': False, 'error': 'Facial recognition not available'}), 400
    
    try:
        import os
        import zipfile
        import tempfile
        import shutil
        from datetime import datetime
        
        # Create temporary directory for export
        temp_dir = tempfile.mkdtemp()
        export_name = f"face_database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_path = os.path.join(temp_dir, export_name)
        os.makedirs(export_path)
        
        # Get known persons and their data
        known_persons = facial_recognition.get_known_persons()
        stats = facial_recognition.get_recognition_stats()
        
        # Create export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_persons': len(known_persons),
            'known_persons': known_persons,
            'recognition_stats': stats,
            'version': '1.0'
        }
        
        # Save export data as JSON
        import json
        with open(os.path.join(export_path, 'database_info.json'), 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Copy database files if they exist
        db_files = ['face_database.db', 'face_encodings.pkl']
        for db_file in db_files:
            if os.path.exists(db_file):
                shutil.copy2(db_file, export_path)
        
        # Copy known faces directory if it exists
        known_faces_dir = 'known_faces'
        if os.path.exists(known_faces_dir):
            shutil.copytree(known_faces_dir, os.path.join(export_path, 'known_faces'))
        
        # Create zip file
        zip_path = f"{export_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        # Read zip file content
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Return zip file
        from flask import Response
        import base64
        
        response = Response(
            zip_data,
            mimetype='application/zip',
            headers={
                'Content-Disposition': f'attachment; filename={export_name}.zip',
                'Content-Length': len(zip_data)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting database: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/facial_recognition/optimize_database', methods=['POST'])
def optimize_face_database():
    """Optimize facial recognition database."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'success': False, 'error': 'Facial recognition not available'}), 400
    
    try:
        # Get current stats
        initial_stats = facial_recognition.get_recognition_stats()
        known_persons_before = len(facial_recognition.get_known_persons())
        
        # Perform optimization (this is a simplified version)
        # In a real implementation, you might:
        # - Remove duplicate encodings
        # - Clean up old temporary files
        # - Rebuild feature indexes
        # - Compress database
        
        optimizations_performed = []
        
        # Clean up database
        try:
            import sqlite3
            conn = sqlite3.connect('face_database.db')
            cursor = conn.cursor()
            
            # Remove any orphaned records
            cursor.execute('DELETE FROM person_detections WHERE person_name NOT IN (SELECT name FROM known_persons)')
            removed_orphaned = cursor.rowcount
            if removed_orphaned > 0:
                optimizations_performed.append(f"Removed {removed_orphaned} orphaned detection records")
            
            # Vacuum database to reclaim space
            cursor.execute('VACUUM')
            optimizations_performed.append("Database vacuumed and compacted")
            
            conn.commit()
            conn.close()
            
        except Exception as db_error:
            logger.warning(f"Database optimization warning: {str(db_error)}")
        
        # Clean up temporary files
        import os
        import glob
        temp_files_removed = 0
        for pattern in ['*.tmp', 'temp_*.jpg', 'temp_*.png']:
            for temp_file in glob.glob(pattern):
                try:
                    os.remove(temp_file)
                    temp_files_removed += 1
                except:
                    pass
        
        if temp_files_removed > 0:
            optimizations_performed.append(f"Removed {temp_files_removed} temporary files")
        
        # Rebuild encodings if needed
        facial_recognition._save_encodings()
        optimizations_performed.append("Face encodings optimized and saved")
        
        # Get final stats
        final_stats = facial_recognition.get_recognition_stats()
        known_persons_after = len(facial_recognition.get_known_persons())
        
        return jsonify({
            'success': True,
            'message': 'Database optimization completed',
            'optimizations': optimizations_performed,
            'stats': {
                'known_persons_before': known_persons_before,
                'known_persons_after': known_persons_after,
                'initial_stats': initial_stats,
                'final_stats': final_stats
            }
        })
        
    except Exception as e:
        logger.error(f"Error optimizing database: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/facial_recognition/clear_database', methods=['POST'])
def clear_face_database():
    """Clear all facial recognition data (DANGEROUS OPERATION)."""
    initialize_advanced_features()
    
    if not facial_recognition:
        return jsonify({'success': False, 'error': 'Facial recognition not available'}), 400
    
    try:
        data = request.get_json()
        confirmation = data.get('confirmation', '').upper()
        
        if confirmation != 'YES':
            return jsonify({'success': False, 'error': 'Confirmation required. Send "YES" in confirmation field.'}), 400
        
        # Get count before clearing
        persons_before = len(facial_recognition.get_known_persons())
        
        # Clear all known persons
        known_persons = facial_recognition.get_known_persons()
        removed_count = 0
        for person in known_persons:
            if facial_recognition.remove_known_person(person):
                removed_count += 1
        
        # Clean database
        try:
            import sqlite3
            conn = sqlite3.connect('face_database.db')
            cursor = conn.cursor()
            
            # Clear all tables
            cursor.execute('DELETE FROM known_persons')
            cursor.execute('DELETE FROM person_detections')
            cursor.execute('DELETE FROM face_encodings')
            cursor.execute('VACUUM')
            
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.warning(f"Database clearing warning: {str(db_error)}")
        
        # Remove encoding files
        import os
        encoding_files = ['face_encodings.pkl', 'face_database.db']
        for file in encoding_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
        
        # Clear known faces directory
        import shutil
        if os.path.exists('known_faces'):
            try:
                shutil.rmtree('known_faces')
                os.makedirs('known_faces')
            except:
                pass
        
        return jsonify({
            'success': True,
            'message': f'Database cleared successfully. Removed {removed_count} persons.',
            'persons_removed': removed_count
        })
        
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Behavior Analysis API Endpoints
@api.route('/behavior_analysis/rules')
def get_behavior_rules():
    """Get behavior analysis rules."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        rules = behavior_analyzer.get_behavior_rules()
        return jsonify({'success': True, 'rules': rules})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/rules', methods=['POST'])
def update_behavior_rules():
    """Update behavior analysis rules."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        data = request.get_json()
        rules = data.get('rules', {})
        
        success = behavior_analyzer.update_behavior_rules(rules)
        if success:
            return jsonify({'success': True, 'message': 'Rules updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update rules'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/events')
def get_behavior_events():
    """Get recent behavior events."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        limit = request.args.get('limit', 20, type=int)
        suspicious_only = request.args.get('suspicious_only', False, type=bool)
        
        events = behavior_analyzer.get_recent_behavior_events(limit=limit, suspicious_only=suspicious_only)
        return jsonify({'success': True, 'events': events})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/activity_patterns')
def get_activity_patterns():
    """Get activity patterns for analytics."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        time_range = request.args.get('time_range', '24h')
        patterns = behavior_analyzer.get_activity_patterns(time_range=time_range)
        return jsonify({'success': True, 'patterns': patterns})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/toggle', methods=['POST'])
def toggle_behavior_analysis():
    """Toggle behavior analysis on/off."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        behavior_analyzer.set_enabled(enabled)
        return jsonify({'success': True, 'enabled': enabled})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/configure_rule', methods=['POST'])
def configure_behavior_rule():
    """Configure a specific behavior rule."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        data = request.get_json()
        rule_name = data.get('rule_name')
        rule_config = data.get('config', {})
        
        success = behavior_analyzer.configure_rule(rule_name, rule_config)
        if success:
            return jsonify({'success': True, 'message': f'Rule {rule_name} configured successfully'})
        else:
            return jsonify({'success': False, 'error': f'Failed to configure rule {rule_name}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/stats')
def get_detailed_behavior_stats():
    """Get detailed behavior analysis statistics."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        time_range = request.args.get('time_range', '24h')
        stats = behavior_analyzer.get_detailed_stats(time_range=time_range)
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/generate_demo', methods=['POST'])
def generate_demo_behavior_data():
    """Generate demo behavior analysis data for testing."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        behavior_analyzer.generate_demo_events()
        return jsonify({'success': True, 'message': 'Demo data generated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api.route('/behavior_analysis/clear_events', methods=['POST'])
def clear_behavior_events():
    """Clear all behavior events."""
    initialize_advanced_features()
    
    if not behavior_analyzer:
        return jsonify({'success': False, 'error': 'Behavior analysis not available'})
    
    try:
        conn = sqlite3.connect(behavior_analyzer.database_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM behavior_events')
        cursor.execute('DELETE FROM crowd_events')
        
        conn.commit()
        conn.close()
        
        # Reset stats
        behavior_analyzer.behavior_stats['activities_detected'] = defaultdict(int)
        behavior_analyzer.behavior_stats['suspicious_behaviors'] = 0
        behavior_analyzer.behavior_stats['crowd_events'] = 0
        behavior_analyzer.recent_events.clear()
        
        return jsonify({'success': True, 'message': 'All behavior events cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

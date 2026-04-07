#!/usr/bin/env python3
"""
Advanced Features Integration Test

This script tests all advanced features to check their implementation status
and integration with the main surveillance system.
"""

import os
import sys
import logging
import time
import numpy as np
import cv2
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('advanced_features_test')

def test_feature_imports():
    """Test if all advanced features can be imported."""
    results = {}
    
    # Test each advanced feature module
    features_to_test = [
        ('facial_recognition', 'FacialRecognitionSystem'),
        ('behavior_analysis', 'BehaviorAnalyzer'),
        ('person_reid', 'PersonReIdentification'),
        ('multi_camera_sync', 'MultiCameraManager'),
        ('real_time_analytics', 'AnalyticsEngine'),
        ('advanced_alerts', 'AdvancedAlertSystem')
    ]
    
    for module_name, class_name in features_to_test:
        try:
            module = __import__(f'advanced_features.{module_name}', fromlist=[class_name])
            feature_class = getattr(module, class_name)
            results[module_name] = {
                'import_status': 'SUCCESS',
                'class_available': True,
                'error': None
            }
            logger.info(f"✅ {module_name} - Import successful")
        except ImportError as e:
            results[module_name] = {
                'import_status': 'FAILED',
                'class_available': False,
                'error': f"Import error: {str(e)}"
            }
            logger.error(f"❌ {module_name} - Import failed: {str(e)}")
        except Exception as e:
            results[module_name] = {
                'import_status': 'FAILED',
                'class_available': False,
                'error': f"Other error: {str(e)}"
            }
            logger.error(f"❌ {module_name} - Error: {str(e)}")
    
    return results

def test_feature_initialization():
    """Test if advanced features can be initialized."""
    results = {}
    
    # Test Facial Recognition
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        face_system = FacialRecognitionSystem()
        results['facial_recognition'] = {
            'initialization': 'SUCCESS',
            'dependencies_available': hasattr(face_system, 'known_face_encodings'),
            'error': None
        }
        logger.info("✅ Facial Recognition - Initialization successful")
    except Exception as e:
        results['facial_recognition'] = {
            'initialization': 'FAILED',
            'dependencies_available': False,
            'error': str(e)
        }
        logger.error(f"❌ Facial Recognition - Initialization failed: {str(e)}")
    
    # Test Behavior Analysis
    try:
        from advanced_features.behavior_analysis import BehaviorAnalyzer
        behavior_analyzer = BehaviorAnalyzer()
        results['behavior_analysis'] = {
            'initialization': 'SUCCESS',
            'dependencies_available': hasattr(behavior_analyzer, 'person_tracks'),
            'error': None
        }
        logger.info("✅ Behavior Analysis - Initialization successful")
    except Exception as e:
        results['behavior_analysis'] = {
            'initialization': 'FAILED',
            'dependencies_available': False,
            'error': str(e)
        }
        logger.error(f"❌ Behavior Analysis - Initialization failed: {str(e)}")
    
    # Test Person Re-ID
    try:
        from advanced_features.person_reid import PersonReIdentification
        reid_system = PersonReIdentification()
        results['person_reid'] = {
            'initialization': 'SUCCESS',
            'dependencies_available': hasattr(reid_system, 'person_gallery'),
            'error': None
        }
        logger.info("✅ Person Re-ID - Initialization successful")
    except Exception as e:
        results['person_reid'] = {
            'initialization': 'FAILED',
            'dependencies_available': False,
            'error': str(e)
        }
        logger.error(f"❌ Person Re-ID - Initialization failed: {str(e)}")
    
    # Test Multi-Camera Sync
    try:
        from advanced_features.multi_camera_sync import MultiCameraManager
        camera_manager = MultiCameraManager()
        results['multi_camera_sync'] = {
            'initialization': 'SUCCESS',
            'dependencies_available': hasattr(camera_manager, 'cameras'),
            'error': None
        }
        logger.info("✅ Multi-Camera Sync - Initialization successful")
    except Exception as e:
        results['multi_camera_sync'] = {
            'initialization': 'FAILED',
            'dependencies_available': False,
            'error': str(e)
        }
        logger.error(f"❌ Multi-Camera Sync - Initialization failed: {str(e)}")
    
    # Test Real-time Analytics
    try:
        from advanced_features.real_time_analytics import AnalyticsEngine
        analytics = AnalyticsEngine()
        results['real_time_analytics'] = {
            'initialization': 'SUCCESS',
            'dependencies_available': hasattr(analytics, 'metrics_buffer'),
            'error': None
        }
        logger.info("✅ Real-time Analytics - Initialization successful")
    except Exception as e:
        results['real_time_analytics'] = {
            'initialization': 'FAILED',
            'dependencies_available': False,
            'error': str(e)
        }
        logger.error(f"❌ Real-time Analytics - Initialization failed: {str(e)}")
    
    # Test Advanced Alerts
    try:
        from advanced_features.advanced_alerts import AdvancedAlertSystem
        alert_system = AdvancedAlertSystem()
        results['advanced_alerts'] = {
            'initialization': 'SUCCESS',
            'dependencies_available': hasattr(alert_system, 'active_alerts'),
            'error': None
        }
        logger.info("✅ Advanced Alerts - Initialization successful")
    except Exception as e:
        results['advanced_alerts'] = {
            'initialization': 'FAILED',
            'dependencies_available': False,
            'error': str(e)
        }
        logger.error(f"❌ Advanced Alerts - Initialization failed: {str(e)}")
    
    return results

def test_basic_functionality():
    """Test basic functionality of advanced features with dummy data."""
    results = {}
    
    # Create dummy frame for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (300, 400), (255, 255, 255), -1)  # Dummy person
    
    # Dummy person detection
    dummy_detection = {
        'class_name': 'person',
        'bbox': [100, 100, 200, 300],  # x, y, w, h
        'confidence': 0.85,
        'track_id': 'person_1'
    }
    
    # Test Facial Recognition functionality
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        face_system = FacialRecognitionSystem()
        detections = face_system.detect_faces(dummy_frame)
        results['facial_recognition_function'] = {
            'status': 'SUCCESS',
            'output': f"Processed frame, found {len(detections)} faces",
            'error': None
        }
        logger.info(f"✅ Facial Recognition Function - Works: {len(detections)} faces detected")
    except Exception as e:
        results['facial_recognition_function'] = {
            'status': 'FAILED',
            'output': None,
            'error': str(e)
        }
        logger.error(f"❌ Facial Recognition Function - Failed: {str(e)}")
    
    # Test Behavior Analysis functionality
    try:
        from advanced_features.behavior_analysis import BehaviorAnalyzer
        behavior_analyzer = BehaviorAnalyzer()
        behavior_results = behavior_analyzer.analyze_behavior(dummy_frame, [dummy_detection], 1)
        results['behavior_analysis_function'] = {
            'status': 'SUCCESS',
            'output': f"Analyzed {len(behavior_results)} behaviors",
            'error': None
        }
        logger.info(f"✅ Behavior Analysis Function - Works: {len(behavior_results)} behaviors analyzed")
    except Exception as e:
        results['behavior_analysis_function'] = {
            'status': 'FAILED',
            'output': None,
            'error': str(e)
        }
        logger.error(f"❌ Behavior Analysis Function - Failed: {str(e)}")
    
    # Test Person Re-ID functionality
    try:
        from advanced_features.person_reid import PersonReIdentification
        reid_system = PersonReIdentification()
        reid_system.register_camera("camera_1", "Test Camera")
        reid_results = reid_system.process_detections(dummy_frame, [dummy_detection], "camera_1")
        results['person_reid_function'] = {
            'status': 'SUCCESS',
            'output': f"Processed {len(reid_results)} person detections",
            'error': None
        }
        logger.info(f"✅ Person Re-ID Function - Works: {len(reid_results)} person detections processed")
    except Exception as e:
        results['person_reid_function'] = {
            'status': 'FAILED',
            'output': None,
            'error': str(e)
        }
        logger.error(f"❌ Person Re-ID Function - Failed: {str(e)}")
    
    # Test Advanced Alerts functionality
    try:
        from advanced_features.advanced_alerts import AdvancedAlertSystem, AlertPriority
        alert_system = AdvancedAlertSystem()
        alert_id = alert_system.create_alert(
            alert_type="test_alert",
            title="Test Alert",
            description="This is a test alert",
            priority=AlertPriority.MEDIUM,
            source_location="test_camera",
            confidence=0.9
        )
        results['advanced_alerts_function'] = {
            'status': 'SUCCESS',
            'output': f"Created alert with ID: {alert_id}",
            'error': None
        }
        logger.info(f"✅ Advanced Alerts Function - Works: Created alert {alert_id}")
    except Exception as e:
        results['advanced_alerts_function'] = {
            'status': 'FAILED',
            'output': None,
            'error': str(e)
        }
        logger.error(f"❌ Advanced Alerts Function - Failed: {str(e)}")
    
    return results

def check_dependencies():
    """Check availability of external dependencies."""
    dependencies = {}
    
    # Check face_recognition library
    try:
        import face_recognition
        dependencies['face_recognition'] = 'AVAILABLE'
        logger.info("✅ face_recognition library - Available")
    except ImportError:
        dependencies['face_recognition'] = 'MISSING'
        logger.warning("⚠️ face_recognition library - Missing (pip install face_recognition)")
    
    # Check MediaPipe
    try:
        import mediapipe
        dependencies['mediapipe'] = 'AVAILABLE'
        logger.info("✅ mediapipe library - Available")
    except ImportError:
        dependencies['mediapipe'] = 'MISSING'
        logger.warning("⚠️ mediapipe library - Missing (pip install mediapipe)")
    
    # Check Twilio
    try:
        import twilio
        dependencies['twilio'] = 'AVAILABLE'
        logger.info("✅ twilio library - Available")
    except ImportError:
        dependencies['twilio'] = 'MISSING'
        logger.warning("⚠️ twilio library - Missing (pip install twilio)")
    
    # Check requests
    try:
        import requests
        dependencies['requests'] = 'AVAILABLE'
        logger.info("✅ requests library - Available")
    except ImportError:
        dependencies['requests'] = 'MISSING'
        logger.warning("⚠️ requests library - Missing (pip install requests)")
    
    # Check torch (for advanced models)
    try:
        import torch
        dependencies['torch'] = 'AVAILABLE'
        logger.info("✅ torch library - Available")
    except ImportError:
        dependencies['torch'] = 'MISSING'
        logger.warning("⚠️ torch library - Missing (pip install torch)")
    
    return dependencies

def check_main_system_integration():
    """Check if advanced features are integrated into main.py."""
    integration_status = {}
    
    try:
        with open('main.py', 'r') as f:
            main_content = f.read()
        
        # Check for advanced features imports
        advanced_imports = [
            'from advanced_features',
            'facial_recognition',
            'behavior_analysis',
            'person_reid',
            'multi_camera_sync',
            'real_time_analytics',
            'advanced_alerts'
        ]
        
        found_imports = []
        for import_check in advanced_imports:
            if import_check in main_content:
                found_imports.append(import_check)
        
        integration_status['main_py_integration'] = {
            'advanced_imports_found': found_imports,
            'total_imports_checked': len(advanced_imports),
            'integration_level': 'PARTIAL' if found_imports else 'NONE'
        }
        
        if found_imports:
            logger.info(f"✅ Main System Integration - Found {len(found_imports)} advanced feature imports")
        else:
            logger.warning("⚠️ Main System Integration - No advanced features integrated in main.py")
            
    except Exception as e:
        integration_status['main_py_integration'] = {
            'error': str(e),
            'integration_level': 'ERROR'
        }
        logger.error(f"❌ Main System Integration - Error checking: {str(e)}")
    
    return integration_status

def generate_integration_report(import_results, init_results, func_results, deps, integration):
    """Generate a comprehensive report."""
    print("\n" + "="*80)
    print("🧠 SMART SURVEILLANCE SYSTEM - ADVANCED FEATURES STATUS REPORT")
    print("="*80)
    
    print(f"\n📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import Status
    print(f"\n🔍 IMPORT STATUS:")
    print("-" * 40)
    for feature, result in import_results.items():
        status_icon = "✅" if result['import_status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {feature}: {result['import_status']}")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Initialization Status
    print(f"\n🚀 INITIALIZATION STATUS:")
    print("-" * 40)
    for feature, result in init_results.items():
        status_icon = "✅" if result['initialization'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {feature}: {result['initialization']}")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Functionality Status
    print(f"\n⚙️ FUNCTIONALITY STATUS:")
    print("-" * 40)
    for feature, result in func_results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {feature}: {result['status']}")
        if result['output']:
            print(f"   Result: {result['output']}")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Dependencies Status
    print(f"\n📦 DEPENDENCIES STATUS:")
    print("-" * 40)
    for dep, status in deps.items():
        status_icon = "✅" if status == 'AVAILABLE' else "⚠️"
        print(f"{status_icon} {dep}: {status}")
    
    # Integration Status
    print(f"\n🔗 MAIN SYSTEM INTEGRATION:")
    print("-" * 40)
    main_integration = integration.get('main_py_integration', {})
    level = main_integration.get('integration_level', 'UNKNOWN')
    level_icon = "✅" if level == 'FULL' else "⚠️" if level == 'PARTIAL' else "❌"
    print(f"{level_icon} Integration Level: {level}")
    
    if 'advanced_imports_found' in main_integration:
        imports_found = main_integration['advanced_imports_found']
        print(f"   Found imports: {len(imports_found)}")
        for imp in imports_found:
            print(f"   - {imp}")
    
    # Summary and Recommendations
    print(f"\n📊 SUMMARY:")
    print("-" * 40)
    
    total_features = len(import_results)
    working_features = sum(1 for r in init_results.values() if r['initialization'] == 'SUCCESS')
    functional_features = sum(1 for r in func_results.values() if r['status'] == 'SUCCESS')
    available_deps = sum(1 for status in deps.values() if status == 'AVAILABLE')
    total_deps = len(deps)
    
    print(f"📈 Working Features: {working_features}/{total_features}")
    print(f"⚡ Functional Features: {functional_features}/{len(func_results)}")
    print(f"📦 Available Dependencies: {available_deps}/{total_deps}")
    print(f"🔗 Main Integration: {level}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if available_deps < total_deps:
        missing_deps = [dep for dep, status in deps.items() if status == 'MISSING']
        print(f"1. Install missing dependencies: {', '.join(missing_deps)}")
    
    if level in ['NONE', 'PARTIAL']:
        print("2. Integrate advanced features into main.py for full functionality")
    
    if working_features < total_features:
        print("3. Fix initialization issues for non-working features")
    
    print(f"\n🎯 NEXT STEPS:")
    print("-" * 40)
    print("1. Run: pip install face_recognition mediapipe twilio requests torch")
    print("2. Update main.py to import and use advanced features")
    print("3. Test integration with real video input")
    print("4. Configure notification channels in advanced alerts")
    
    print("\n" + "="*80)

def main():
    """Main test function."""
    print("🧠 Starting Advanced Features Integration Test...")
    
    # Run all tests
    print("\n🔍 Testing imports...")
    import_results = test_feature_imports()
    
    print("\n🚀 Testing initialization...")
    init_results = test_feature_initialization()
    
    print("\n⚙️ Testing basic functionality...")
    func_results = test_basic_functionality()
    
    print("\n📦 Checking dependencies...")
    deps = check_dependencies()
    
    print("\n🔗 Checking main system integration...")
    integration = check_main_system_integration()
    
    # Generate comprehensive report
    generate_integration_report(import_results, init_results, func_results, deps, integration)

if __name__ == "__main__":
    main()

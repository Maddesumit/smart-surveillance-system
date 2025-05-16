#!/usr/bin/env python3
"""
Test Script for Smart Surveillance System

This script tests the core functionality of the surveillance system
by simulating common scenarios.
"""

import os
import sys
import time
import logging
import cv2
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import components from our project
from object_detection.detector import ObjectDetector
from object_detection.tracker import ObjectTracker
from anomaly_detection.analyzer import AnomalyDetector
from alert_system.notifier import AlertSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_surveillance')

def test_unattended_object():
    """
    Test the detection of unattended objects.
    """
    logger.info("Testing unattended object detection...")
    
    # Create test components
    detector = ObjectDetector()
    tracker = ObjectTracker()
    analyzer = AnomalyDetector(stationary_threshold=5)  # Lower threshold for testing
    notifier = AlertSystem()
    
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate an object that stays in the same place
    alerts_received = []
    
    def alert_callback(alert):
        alerts_received.append(alert)
        logger.info(f"Alert received: {alert['type']} - {alert['message']}")
    
    # Override the generate_alert method to use our callback
    original_generate_alert = notifier.generate_alert
    notifier.generate_alert = lambda anomaly: alert_callback(original_generate_alert(anomaly))
    
    # Simulate an object that stays in the same place for multiple frames
    for i in range(10):
        # Create a test detection (simulated object)
        test_bbox = [100, 100, 200, 200]  # [x1, y1, x2, y2]
        test_detection = {
            'bbox': test_bbox,
            'class_id': 0,
            'class_name': 'backpack',  # Changed from 'test_object' to 'backpack'
            'confidence': 0.9
        }
        
        # Update tracker with this detection
        tracked_objects = tracker.update([test_detection])
        
        # Check for anomalies
        anomalies = analyzer.detect_anomalies(frame, tracked_objects)
        
        # Generate alerts for any anomalies
        for anomaly in anomalies:
            notifier.generate_alert(anomaly)
        
        logger.info(f"Frame {i+1}: {len(anomalies)} anomalies detected")
    
    # Check if we received any unattended object alerts
    unattended_alerts = [a for a in alerts_received if a['type'] == 'unattended_object']
    if unattended_alerts:
        logger.info("✅ Unattended object detection test PASSED")
    else:
        logger.error("❌ Unattended object detection test FAILED")
    
    return len(unattended_alerts) > 0

def test_restricted_area():
    """
    Test the detection of restricted area violations.
    """
    logger.info("Testing restricted area violation detection...")
    
    # Create test components
    detector = ObjectDetector()
    tracker = ObjectTracker()
    analyzer = AnomalyDetector()
    notifier = AlertSystem()
    
    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Define a restricted area in the center of the frame
    restricted_area = [
        (200, 200),
        (400, 200),
        (400, 400),
        (200, 400)
    ]
    analyzer.add_restricted_area(np.array(restricted_area))
    
    # Simulate alerts
    alerts_received = []
    
    def alert_callback(alert):
        alerts_received.append(alert)
        logger.info(f"Alert received: {alert['type']} - {alert['message']}")
    
    # Override the generate_alert method to use our callback
    original_generate_alert = notifier.generate_alert
    notifier.generate_alert = lambda anomaly: alert_callback(original_generate_alert(anomaly))
    
    # Simulate an object entering the restricted area
    test_bbox = [250, 250, 350, 350]  # Inside the restricted area
    test_detection = {
        'bbox': test_bbox,
        'class_id': 0,
        'class_name': 'person',
        'confidence': 0.9
    }
    
    # Update tracker with this detection
    tracked_objects = tracker.update([test_detection])
    
    # Check for anomalies
    anomalies = analyzer.detect_anomalies(frame, tracked_objects)
    
    # Generate alerts for any anomalies
    for anomaly in anomalies:
        notifier.generate_alert(anomaly)
    
    # Check if we received any restricted area alerts
    restricted_alerts = [a for a in alerts_received if a['type'] == 'restricted_area_violation']
    if restricted_alerts:
        logger.info("✅ Restricted area detection test PASSED")
    else:
        logger.error("❌ Restricted area detection test FAILED")
    
    return len(restricted_alerts) > 0

def run_all_tests():
    """
    Run all test cases and report results.
    """
    logger.info("Starting system tests...")
    
    tests = [
        ("Unattended Object Detection", test_unattended_object),
        ("Restricted Area Violation", test_restricted_area),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}\nRunning test: {test_name}\n{'='*50}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Error in test {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    logger.info("\n\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed {passed}/{len(results)} tests")

if __name__ == "__main__":
    run_all_tests()
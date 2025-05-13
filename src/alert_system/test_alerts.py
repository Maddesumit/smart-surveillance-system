"""
Test script for the alert system.

This script demonstrates the use of the AlertSystem class
with simulated anomaly detections.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alert_system.notifier import AlertSystem

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the alert system.
    """
    # Create a log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize the alert system
    alert_log_path = os.path.join(log_dir, 'alerts.log')
    alert_system = AlertSystem(alert_history_size=50, alert_log_path=alert_log_path)
    
    # Add a custom template
    alert_system.add_template('motion_detected', "ALERT: Motion detected in {area_name} at {timestamp}")
    
    # Simulate some anomaly detections
    logger.info("Simulating anomaly detections...")
    
    # Simulate an unattended object
    unattended_object = {
        'type': 'unattended_object',
        'object_id': 1,
        'class_name': 'backpack',
        'confidence': 0.85,
        'location': (320, 240),
        'frames_stationary': 150,  # 5 seconds at 30 fps
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate an alert for the unattended object
    alert1 = alert_system.generate_alert(unattended_object)
    
    # Simulate a restricted area violation
    restricted_area_violation = {
        'type': 'restricted_area_violation',
        'object_id': 2,
        'class_name': 'person',
        'confidence': 0.92,
        'location': (150, 200),
        'area_id': 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate an alert for the restricted area violation
    alert2 = alert_system.generate_alert(restricted_area_violation)
    
    # Simulate a custom alert type
    custom_anomaly = {
        'type': 'motion_detected',
        'confidence': 0.78,
        'area_name': 'Entrance',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Generate an alert for the custom anomaly
    alert3 = alert_system.generate_alert(custom_anomaly)
    
    # Wait a moment
    time.sleep(1)
    
    # Get recent alerts
    recent_alerts = alert_system.get_recent_alerts()
    logger.info(f"Recent alerts: {len(recent_alerts)}")
    
    # Display the alerts
    for i, alert in enumerate(recent_alerts):
        logger.info(f"Alert {i+1}: {alert['message']}")
    
    # Acknowledge an alert
    if recent_alerts:
        alert_id = recent_alerts[0]['id']
        alert_system.acknowledge_alert(alert_id)
        logger.info(f"Acknowledged alert with ID: {alert_id}")
    
    # Get unacknowledged alerts
    unacknowledged = alert_system.get_recent_alerts(acknowledged=False)
    logger.info(f"Unacknowledged alerts: {len(unacknowledged)}")
    
    logger.info("Alert system test completed")

if __name__ == "__main__":
    main()
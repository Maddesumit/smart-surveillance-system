"""
Alert System Module

This module provides functionality for generating and managing alerts
based on detected anomalies in the surveillance system.
"""

import logging
import time
from datetime import datetime
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Class for generating and managing alerts based on detected anomalies.
    
    This class handles:
    - Creating alerts from anomaly data
    - Managing alert templates
    - Tracking alert history
    - Displaying alerts in the application
    """
    
    def __init__(self, alert_history_size=100, alert_log_path=None):
        """
        Initialize the AlertSystem.
        
        Args:
            alert_history_size (int): Maximum number of alerts to keep in history
            alert_log_path (str): Path to save alert logs (None for no file logging)
        """
        # Configuration parameters
        self.alert_history_size = alert_history_size
        self.alert_log_path = alert_log_path
        
        # Alert storage
        self.alert_history = []  # List to store alert history
        self.alert_templates = {
            'unattended_object': "ALERT: Unattended {object_type} detected at location ({x}, {y}) for {duration} seconds",
            'restricted_area_violation': "ALERT: {object_type} entered restricted area {area_id} at location ({x}, {y})",
            'general': "ALERT: {message}"
        }
        
        logger.info("Alert system initialized")
        
        # Create log directory if needed
        if self.alert_log_path:
            os.makedirs(os.path.dirname(self.alert_log_path), exist_ok=True)
    
    def add_template(self, alert_type, template):
        """
        Add or update an alert message template.
        
        Args:
            alert_type (str): Type of alert (e.g., 'unattended_object')
            template (str): Message template with placeholders
        """
        self.alert_templates[alert_type] = template
        logger.info(f"Added template for alert type: {alert_type}")
    
    def generate_alert(self, anomaly):
        """
        Generate an alert from an anomaly detection.
        
        Args:
            anomaly (dict): Anomaly data from AnomalyDetector
            
        Returns:
            dict: Alert data
        """
        # Create base alert data
        alert = {
            'id': int(time.time() * 1000),  # Unique ID based on timestamp
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': anomaly.get('type', 'unknown'),
            'source_data': anomaly,
            'acknowledged': False
        }
        
        # Generate message from template
        alert_type = anomaly.get('type', 'general')
        template = self.alert_templates.get(alert_type, self.alert_templates['general'])
        
        # Prepare template variables
        template_vars = {
            'object_type': anomaly.get('class_name', 'unknown object'),
            'message': f"Anomaly of type {alert_type} detected"
        }
        
        # Add location if available
        if 'location' in anomaly:
            x, y = anomaly['location']
            template_vars['x'] = x
            template_vars['y'] = y
        
        # Add area_id if available
        if 'area_id' in anomaly:
            template_vars['area_id'] = anomaly['area_id']
            
        # Add duration for stationary objects
        if 'frames_stationary' in anomaly:
            # Assuming 30 fps, convert frames to seconds
            duration = anomaly['frames_stationary'] / 30.0
            template_vars['duration'] = round(duration, 1)
        
        # Format the message
        try:
            alert['message'] = template.format(**template_vars)
        except KeyError as e:
            # Fallback if template has missing variables
            logger.warning(f"Template error: {e}. Using fallback message.")
            alert['message'] = f"Alert: {alert_type} detected"
        
        # Add to history
        self._add_to_history(alert)
        
        # Log the alert
        logger.warning(f"Alert generated: {alert['message']}")
        
        return alert
    
    def _add_to_history(self, alert):
        """
        Add an alert to the history and optionally log to file.
        
        Args:
            alert (dict): Alert data
        """
        # Add to in-memory history
        self.alert_history.append(alert)
        
        # Trim history if it exceeds the maximum size
        if len(self.alert_history) > self.alert_history_size:
            self.alert_history = self.alert_history[-self.alert_history_size:]
        
        # Log to file if configured
        if self.alert_log_path:
            try:
                with open(self.alert_log_path, 'a') as f:
                    f.write(json.dumps(alert) + '\n')
            except Exception as e:
                logger.error(f"Failed to write alert to log file: {e}")
    
    def get_recent_alerts(self, count=10, alert_type=None, acknowledged=None):
        """
        Get recent alerts with optional filtering.
        
        Args:
            count (int): Number of alerts to return
            alert_type (str): Filter by alert type
            acknowledged (bool): Filter by acknowledged status
            
        Returns:
            list: List of alert dictionaries
        """
        # Start with all alerts
        alerts = self.alert_history.copy()
        
        # Apply filters
        if alert_type is not None:
            alerts = [a for a in alerts if a['type'] == alert_type]
            
        if acknowledged is not None:
            alerts = [a for a in alerts if a['acknowledged'] == acknowledged]
        
        # Return most recent alerts (last items in the list)
        return alerts[-count:]
    
    def acknowledge_alert(self, alert_id):
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            bool: True if alert was found and acknowledged, False otherwise
        """
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        
        logger.warning(f"Alert {alert_id} not found for acknowledgement")
        return False
    
    def clear_history(self):
        """
        Clear the alert history.
        """
        self.alert_history = []
        logger.info("Alert history cleared")
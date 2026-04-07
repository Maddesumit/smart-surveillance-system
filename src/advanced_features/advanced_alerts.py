#!/usr/bin/env python3
"""
Advanced Alert System

This module provides sophisticated alerting capabilities for surveillance systems,
including intelligent alert prioritization, notification routing, and advanced response mechanisms.
"""

import logging
import time
import json
import smtplib
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Set
from enum import Enum
from pathlib import Path
import sqlite3
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    from email.mime.image import MimeImage
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = MimeMultipart = MimeImage = None
import cv2
import numpy as np
from collections import defaultdict, deque

# Try to import additional notification services
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("Twilio not available. Install with: pip install twilio")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests not available. Install with: pip install requests")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class AlertStatus(Enum):
    """Alert status types."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PUSH = "push"
    DISCORD = "discord"
    SLACK = "slack"

class AdvancedAlertSystem:
    """
    Advanced alert management system for surveillance applications.
    
    Features:
    - Multi-channel alert routing
    - Intelligent alert prioritization
    - Alert correlation and deduplication
    - Escalation workflows
    - Custom alert rules and filters
    - Rich alert content (images, videos, metadata)
    - Real-time alert dashboard
    - Alert analytics and reporting
    """
    
    def __init__(self,
                 database_path: str = "advanced_alerts.db",
                 config_path: str = "alert_config.json",
                 max_alert_history: int = 10000):
        """
        Initialize the Advanced Alert System.
        
        Args:
            database_path: Path to SQLite database for alert storage
            config_path: Path to alert configuration file
            max_alert_history: Maximum number of alerts to keep in history
        """
        self.database_path = database_path
        self.config_path = config_path
        self.max_alert_history = max_alert_history
        
        # Alert storage
        self.active_alerts = {}  # alert_id -> alert_data
        self.alert_queue = deque()  # Processing queue
        self.alert_rules = {}  # Custom alert rules
        
        # Notification channels
        self.notification_channels = {}
        self.notification_rules = defaultdict(list)  # priority -> [channels]
        
        # Alert correlation
        self.correlation_rules = []
        self.correlation_windows = defaultdict(deque)  # event_type -> recent_alerts
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_priority': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'false_positives': 0,
            'response_times': deque(maxlen=1000)
        }
        
        # Threading
        self.processing_lock = threading.Lock()
        self.processing_thread = None
        self.running = False
        
        # Initialize components
        self._init_database()
        self._load_configuration()
        self._setup_default_rules()
        
        logger.info("Advanced Alert System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for alert storage."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    source_location TEXT,
                    confidence REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    image_path TEXT,
                    video_path TEXT,
                    false_positive BOOLEAN DEFAULT 0
                )
            ''')
            
            # Create notification log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT,
                    channel TEXT,
                    recipient TEXT,
                    status TEXT,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts (id)
                )
            ''')
            
            # Create alert rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Advanced alert database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def _load_configuration(self):
        """Load alert configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Load notification channels
                for channel_config in config.get('notification_channels', []):
                    self._setup_notification_channel(channel_config)
                
                # Load notification rules
                self.notification_rules = defaultdict(list, config.get('notification_rules', {}))
                
                logger.info("Alert configuration loaded")
            else:
                self._create_default_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default alert configuration."""
        default_config = {
            "notification_channels": [
                {
                    "type": "email",
                    "name": "admin_email",
                    "config": {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "your_email@gmail.com",
                        "password": "your_password",
                        "recipient": "admin@example.com"
                    }
                }
            ],
            "notification_rules": {
                "5": ["admin_email"],  # EMERGENCY
                "4": ["admin_email"],  # CRITICAL
                "3": ["admin_email"],  # HIGH
                "2": [],               # MEDIUM
                "1": []                # LOW
            },
            "correlation_rules": [
                {
                    "name": "repeated_intrusion",
                    "pattern": "restricted_area_violation",
                    "count": 3,
                    "window_seconds": 300,
                    "action": "escalate_priority"
                }
            ]
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info("Default alert configuration created")
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
    
    def _setup_notification_channel(self, channel_config: Dict):
        """Setup a notification channel."""
        try:
            channel_type = channel_config['type']
            channel_name = channel_config['name']
            config = channel_config['config']
            
            if channel_type == 'email':
                self.notification_channels[channel_name] = {
                    'type': 'email',
                    'smtp_server': config['smtp_server'],
                    'smtp_port': config['smtp_port'],
                    'username': config['username'],
                    'password': config['password'],
                    'recipient': config['recipient']
                }
            elif channel_type == 'sms' and TWILIO_AVAILABLE:
                self.notification_channels[channel_name] = {
                    'type': 'sms',
                    'client': TwilioClient(config['account_sid'], config['auth_token']),
                    'from_number': config['from_number'],
                    'to_number': config['to_number']
                }
            elif channel_type == 'webhook' and REQUESTS_AVAILABLE:
                self.notification_channels[channel_name] = {
                    'type': 'webhook',
                    'url': config['url'],
                    'headers': config.get('headers', {}),
                    'auth': config.get('auth')
                }
            
            logger.info(f"Notification channel '{channel_name}' configured")
            
        except Exception as e:
            logger.error(f"Error setting up notification channel: {str(e)}")
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            {
                'name': 'high_confidence_intrusion',
                'conditions': {
                    'type': 'restricted_area_violation',
                    'confidence': {'min': 0.8}
                },
                'actions': {
                    'priority': AlertPriority.HIGH,
                    'notify': True,
                    'capture_image': True
                }
            },
            {
                'name': 'unattended_object_critical',
                'conditions': {
                    'type': 'unattended_object',
                    'duration': {'min': 300}  # 5 minutes
                },
                'actions': {
                    'priority': AlertPriority.CRITICAL,
                    'notify': True,
                    'capture_video': True
                }
            }
        ]
        
        for rule in default_rules:
            self.alert_rules[rule['name']] = rule
    
    def start_processing(self):
        """Start alert processing thread."""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_alerts)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Alert processing started")
    
    def stop_processing(self):
        """Stop alert processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Alert processing stopped")
    
    def _process_alerts(self):
        """Main alert processing loop."""
        while self.running:
            try:
                if self.alert_queue:
                    with self.processing_lock:
                        alert = self.alert_queue.popleft()
                    
                    self._process_single_alert(alert)
                else:
                    time.sleep(0.1)  # Brief pause when no alerts
                    
            except Exception as e:
                logger.error(f"Error in alert processing: {str(e)}")
                time.sleep(1)
    
    def _process_single_alert(self, alert: Dict):
        """Process a single alert."""
        try:
            alert_id = alert['id']
            
            # Apply custom rules
            self._apply_alert_rules(alert)
            
            # Check for correlation
            self._check_correlation(alert)
            
            # Store in database
            self._store_alert(alert)
            
            # Send notifications
            if alert.get('notify', True):
                self._send_notifications(alert)
            
            # Update statistics
            self._update_statistics(alert)
            
            logger.info(f"Processed alert: {alert_id}")
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.get('id', 'unknown')}: {str(e)}")
    
    def create_alert(self,
                    alert_type: str,
                    title: str,
                    description: str = "",
                    priority: AlertPriority = AlertPriority.MEDIUM,
                    source_location: str = "",
                    confidence: float = 1.0,
                    metadata: Optional[Dict] = None,
                    image: Optional[np.ndarray] = None,
                    capture_video: bool = False) -> str:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            title: Alert title
            description: Alert description
            priority: Alert priority level
            source_location: Location where alert originated
            confidence: Confidence score (0-1)
            metadata: Additional metadata
            image: Associated image
            capture_video: Whether to capture video
            
        Returns:
            Generated alert ID
        """
        try:
            # Generate unique alert ID
            alert_id = f"alert_{int(time.time() * 1000)}_{len(self.active_alerts)}"
            
            # Create alert object
            alert = {
                'id': alert_id,
                'type': alert_type,
                'title': title,
                'description': description,
                'priority': priority.value,
                'status': AlertStatus.PENDING.value,
                'source_location': source_location,
                'confidence': confidence,
                'metadata': metadata or {},
                'created_at': datetime.now(),
                'image_path': None,
                'video_path': None,
                'notify': True
            }
            
            # Save image if provided
            if image is not None:
                image_path = self._save_alert_image(alert_id, image)
                alert['image_path'] = image_path
            
            # Queue for processing
            with self.processing_lock:
                self.alert_queue.append(alert)
                self.active_alerts[alert_id] = alert
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            return ""
    
    def _save_alert_image(self, alert_id: str, image: np.ndarray) -> str:
        """Save alert image to disk."""
        try:
            # Create alerts directory
            alerts_dir = Path("alerts")
            alerts_dir.mkdir(exist_ok=True)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = alerts_dir / f"{alert_id}_{timestamp}.jpg"
            cv2.imwrite(str(image_path), image)
            
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Error saving alert image: {str(e)}")
            return ""
    
    def _apply_alert_rules(self, alert: Dict):
        """Apply custom alert rules."""
        try:
            for rule_name, rule in self.alert_rules.items():
                conditions = rule['conditions']
                actions = rule['actions']
                
                # Check if conditions match
                if self._check_conditions(alert, conditions):
                    # Apply actions
                    if 'priority' in actions:
                        alert['priority'] = actions['priority'].value
                    if 'notify' in actions:
                        alert['notify'] = actions['notify']
                    if 'capture_image' in actions and actions['capture_image']:
                        alert['capture_image'] = True
                    if 'capture_video' in actions and actions['capture_video']:
                        alert['capture_video'] = True
                    
                    logger.info(f"Applied rule '{rule_name}' to alert {alert['id']}")
                    
        except Exception as e:
            logger.error(f"Error applying alert rules: {str(e)}")
    
    def _check_conditions(self, alert: Dict, conditions: Dict) -> bool:
        """Check if alert matches conditions."""
        try:
            # Check alert type
            if 'type' in conditions and alert['type'] != conditions['type']:
                return False
            
            # Check confidence range
            if 'confidence' in conditions:
                conf_conditions = conditions['confidence']
                alert_conf = alert['confidence']
                
                if 'min' in conf_conditions and alert_conf < conf_conditions['min']:
                    return False
                if 'max' in conf_conditions and alert_conf > conf_conditions['max']:
                    return False
            
            # Check metadata conditions
            if 'metadata' in conditions:
                for key, value in conditions['metadata'].items():
                    if key not in alert.get('metadata', {}) or alert['metadata'][key] != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking conditions: {str(e)}")
            return False
    
    def _check_correlation(self, alert: Dict):
        """Check for alert correlation patterns."""
        try:
            alert_type = alert['type']
            current_time = datetime.now()
            
            # Add to correlation window
            self.correlation_windows[alert_type].append({
                'alert': alert,
                'timestamp': current_time
            })
            
            # Check correlation rules
            for rule in self.correlation_rules:
                if rule['pattern'] == alert_type:
                    window_seconds = rule['window_seconds']
                    required_count = rule['count']
                    
                    # Count recent alerts of this type
                    cutoff_time = current_time - timedelta(seconds=window_seconds)
                    recent_alerts = [
                        item for item in self.correlation_windows[alert_type]
                        if item['timestamp'] > cutoff_time
                    ]
                    
                    if len(recent_alerts) >= required_count:
                        # Apply correlation action
                        if rule['action'] == 'escalate_priority':
                            alert['priority'] = min(alert['priority'] + 1, AlertPriority.EMERGENCY.value)
                            alert['description'] += f" [ESCALATED: {len(recent_alerts)} similar alerts in {window_seconds}s]"
                        
                        logger.warning(f"Alert correlation triggered: {rule['name']}")
            
            # Clean old entries
            for alert_type_key in self.correlation_windows:
                cutoff_time = current_time - timedelta(seconds=3600)  # Keep 1 hour
                self.correlation_windows[alert_type_key] = deque([
                    item for item in self.correlation_windows[alert_type_key]
                    if item['timestamp'] > cutoff_time
                ])
            
        except Exception as e:
            logger.error(f"Error checking correlation: {str(e)}")
    
    def _store_alert(self, alert: Dict):
        """Store alert in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (
                    id, type, priority, status, title, description,
                    source_location, confidence, metadata, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert['id'], alert['type'], alert['priority'],
                alert['status'], alert['title'], alert['description'],
                alert['source_location'], alert['confidence'],
                json.dumps(alert['metadata']), alert.get('image_path')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
    
    def _send_notifications(self, alert: Dict):
        """Send notifications for alert."""
        try:
            priority = alert['priority']
            channels = self.notification_rules.get(str(priority), [])
            
            for channel_name in channels:
                if channel_name in self.notification_channels:
                    self._send_notification(alert, channel_name)
            
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
    
    def _send_notification(self, alert: Dict, channel_name: str):
        """Send notification through specific channel."""
        try:
            channel = self.notification_channels[channel_name]
            channel_type = channel['type']
            
            if channel_type == 'email':
                self._send_email_notification(alert, channel)
            elif channel_type == 'sms':
                self._send_sms_notification(alert, channel)
            elif channel_type == 'webhook':
                self._send_webhook_notification(alert, channel)
            
            # Log notification
            self._log_notification(alert['id'], channel_type, channel_name, 'sent')
            
        except Exception as e:
            logger.error(f"Error sending notification to {channel_name}: {str(e)}")
            self._log_notification(alert['id'], channel_name, '', 'failed', str(e))
    
    def _send_email_notification(self, alert: Dict, channel: Dict):
        """Send email notification."""
        try:
            if not EMAIL_AVAILABLE:
                raise Exception("Email libraries not available")
                
            msg = MimeMultipart()
            msg['From'] = channel['username']
            msg['To'] = channel['recipient']
            msg['Subject'] = f"Security Alert: {alert['title']}"
            
            # Create email body
            body = f"""
Security Alert Notification

Alert ID: {alert['id']}
Type: {alert['type']}
Priority: {AlertPriority(alert['priority']).name}
Location: {alert['source_location']}
Time: {alert['created_at']}
Confidence: {alert['confidence']:.2f}

Description:
{alert['description']}

Please take appropriate action.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Attach image if available
            if alert.get('image_path') and Path(alert['image_path']).exists():
                with open(alert['image_path'], 'rb') as f:
                    img = MimeImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename='alert_image.jpg')
                    msg.attach(img)
            
            # Send email
            server = smtplib.SMTP(channel['smtp_server'], channel['smtp_port'])
            server.starttls()
            server.login(channel['username'], channel['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            raise
    
    def _send_sms_notification(self, alert: Dict, channel: Dict):
        """Send SMS notification."""
        try:
            if not TWILIO_AVAILABLE:
                raise Exception("Twilio not available")
            
            message_body = f"Security Alert: {alert['title']}\nLocation: {alert['source_location']}\nTime: {alert['created_at']}\nPriority: {AlertPriority(alert['priority']).name}"
            
            message = channel['client'].messages.create(
                body=message_body,
                from_=channel['from_number'],
                to=channel['to_number']
            )
            
            logger.info(f"SMS notification sent for alert {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {str(e)}")
            raise
    
    def _send_webhook_notification(self, alert: Dict, channel: Dict):
        """Send webhook notification."""
        try:
            if not REQUESTS_AVAILABLE:
                raise Exception("Requests library not available")
            
            payload = {
                'alert_id': alert['id'],
                'type': alert['type'],
                'title': alert['title'],
                'description': alert['description'],
                'priority': AlertPriority(alert['priority']).name,
                'location': alert['source_location'],
                'confidence': alert['confidence'],
                'timestamp': alert['created_at'].isoformat(),
                'metadata': alert['metadata']
            }
            
            response = requests.post(
                channel['url'],
                json=payload,
                headers=channel.get('headers', {}),
                auth=channel.get('auth'),
                timeout=10
            )
            
            response.raise_for_status()
            logger.info(f"Webhook notification sent for alert {alert['id']}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
            raise
    
    def _log_notification(self, alert_id: str, channel: str, recipient: str, status: str, error_message: str = ""):
        """Log notification attempt."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO notification_log (alert_id, channel, recipient, status, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (alert_id, channel, recipient, status, error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging notification: {str(e)}")
    
    def _update_statistics(self, alert: Dict):
        """Update alert statistics."""
        try:
            self.alert_stats['total_alerts'] += 1
            self.alert_stats['alerts_by_priority'][alert['priority']] += 1
            self.alert_stats['alerts_by_type'][alert['type']] += 1
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id]['status'] = AlertStatus.ACKNOWLEDGED.value
                self.active_alerts[alert_id]['acknowledged_by'] = user
                self.active_alerts[alert_id]['acknowledged_at'] = datetime.now()
                
                # Update database
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE alerts SET status = ?, acknowledged_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (AlertStatus.ACKNOWLEDGED.value, alert_id))
                conn.commit()
                conn.close()
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
            return False
    
    def resolve_alert(self, alert_id: str, user: str = "system", resolution_note: str = "") -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id]['status'] = AlertStatus.RESOLVED.value
                self.active_alerts[alert_id]['resolved_by'] = user
                self.active_alerts[alert_id]['resolved_at'] = datetime.now()
                self.active_alerts[alert_id]['resolution_note'] = resolution_note
                
                # Update database
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE alerts SET status = ?, resolved_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (AlertStatus.RESOLVED.value, alert_id))
                conn.commit()
                conn.close()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            return False
    
    def mark_false_positive(self, alert_id: str, user: str = "system") -> bool:
        """Mark an alert as false positive."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alerts SET false_positive = 1 WHERE id = ?
            ''', (alert_id,))
            conn.commit()
            conn.close()
            
            self.alert_stats['false_positives'] += 1
            
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} marked as false positive by {user}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking false positive: {str(e)}")
            return False
    
    def get_active_alerts(self, priority_filter: Optional[AlertPriority] = None) -> List[Dict]:
        """Get active alerts."""
        try:
            alerts = list(self.active_alerts.values())
            
            if priority_filter:
                alerts = [a for a in alerts if a['priority'] >= priority_filter.value]
            
            # Sort by priority and creation time
            alerts.sort(key=lambda x: (-x['priority'], x['created_at']))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {str(e)}")
            return []
    
    def get_alert_history(self, limit: int = 100, alert_type: Optional[str] = None) -> List[Dict]:
        """Get alert history from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM alerts"
            params = []
            
            if alert_type:
                query += " WHERE type = ?"
                params.append(alert_type)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [description[0] for description in cursor.description]
            alerts = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alert history: {str(e)}")
            return []
    
    def get_alert_statistics(self) -> Dict:
        """Get comprehensive alert statistics."""
        try:
            stats = self.alert_stats.copy()
            
            # Add database stats
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Total alerts in database
            cursor.execute("SELECT COUNT(*) FROM alerts")
            stats['total_alerts_db'] = cursor.fetchone()[0]
            
            # False positive rate
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE false_positive = 1")
            false_positives = cursor.fetchone()[0]
            if stats['total_alerts_db'] > 0:
                stats['false_positive_rate'] = false_positives / stats['total_alerts_db']
            
            # Average response time
            cursor.execute('''
                SELECT AVG(julianday(acknowledged_at) - julianday(created_at)) * 24 * 60
                FROM alerts WHERE acknowledged_at IS NOT NULL
            ''')
            avg_response = cursor.fetchone()[0]
            stats['avg_response_time_minutes'] = avg_response or 0
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return self.alert_stats.copy()
    
    def cleanup(self):
        """Cleanup resources and stop processing."""
        try:
            self.stop_processing()
            logger.info("Advanced Alert System cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Example usage of the Advanced Alert System."""
    # Initialize system
    alert_system = AdvancedAlertSystem()
    alert_system.start_processing()
    
    try:
        # Example: Create different types of alerts
        
        # High priority intrusion alert
        alert_id1 = alert_system.create_alert(
            alert_type="restricted_area_violation",
            title="Unauthorized Access Detected",
            description="Person detected in restricted area",
            priority=AlertPriority.HIGH,
            source_location="Camera_01_Main_Entrance",
            confidence=0.95
        )
        
        # Critical unattended object alert
        alert_id2 = alert_system.create_alert(
            alert_type="unattended_object",
            title="Suspicious Package Detected",
            description="Unattended bag detected for over 5 minutes",
            priority=AlertPriority.CRITICAL,
            source_location="Camera_03_Lobby",
            confidence=0.87,
            metadata={"object_type": "bag", "duration": 320}
        )
        
        # Wait for processing
        time.sleep(2)
        
        # Check active alerts
        active_alerts = alert_system.get_active_alerts()
        print(f"Active alerts: {len(active_alerts)}")
        
        # Acknowledge an alert
        alert_system.acknowledge_alert(alert_id1, "security_officer_1")
        
        # Get statistics
        stats = alert_system.get_alert_statistics()
        print(f"Alert statistics: {stats}")
        
        # Wait a bit more
        time.sleep(3)
        
    finally:
        alert_system.cleanup()


if __name__ == "__main__":
    main()

"""
Configuration settings for the Smart Surveillance System.
"""

# Video source settings
VIDEO_SOURCE = 0  # 0 for webcam, or provide path/URL for CCTV feed
FRAME_WIDTH = 640  # Width to resize frames to
FRAME_HEIGHT = 480  # Height to resize frames to
FPS_TARGET = 30  # Target frames per second

# Object detection settings
DETECTION_CONFIDENCE = 0.5  # Minimum confidence threshold for detections
DETECTION_CLASSES = [  # Classes to detect (COCO dataset)
    'person',
    'backpack',
    'umbrella',
    'handbag',
    'suitcase',
    'bottle',
    'laptop',
    'cell phone',
]

# Anomaly detection settings
RESTRICTED_AREAS = [  # Example: [[x1, y1, x2, y2], ...]
    [100, 100, 300, 300],  # Top-left to bottom-right coordinates
]
UNATTENDED_OBJECT_TIMEOUT = 30  # Seconds before an object is considered unattended

# Alert system settings
TWILIO_ACCOUNT_SID = 'your_account_sid'  # Replace with your Twilio account SID
TWILIO_AUTH_TOKEN = 'your_auth_token'  # Replace with your Twilio auth token
TWILIO_FROM_NUMBER = 'your_twilio_number'  # Replace with your Twilio phone number
TWILIO_TO_NUMBER = 'your_phone_number'  # Replace with your phone number
ALERT_COOLDOWN = 60  # Seconds between alerts to prevent flooding

# Dashboard settings
FLASK_SECRET_KEY = 'your_secret_key'  # Replace with a random secret key
FLASK_HOST = '0.0.0.0'  # Host to run Flask server on
FLASK_PORT = 5000  # Port to run Flask server on
FLASK_DEBUG = True  # Enable debug mode for development
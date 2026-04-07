"""
Advanced Features Module

This module contains cutting-edge surveillance features including:
- Facial recognition and person re-identification
- Behavior analysis and activity recognition
- Multi-camera synchronization
- Real-time analytics and insights
- Advanced alert mechanisms
"""

from .facial_recognition import FacialRecognitionSystem
from .behavior_analysis import BehaviorAnalyzer
from .person_reid import PersonReID
from .multi_camera_sync import MultiCameraManager
from .real_time_analytics import AnalyticsEngine
from .advanced_alerts import AdvancedAlertSystem

__all__ = [
    'FacialRecognitionSystem',
    'BehaviorAnalyzer', 
    'PersonReID',
    'MultiCameraManager',
    'AnalyticsEngine',
    'AdvancedAlertSystem'
]

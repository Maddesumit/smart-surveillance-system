"""
Advanced Features Module

This module contains cutting-edge surveillance features including:
- Facial recognition and person re-identification
- Behavior analysis and activity recognition
- Multi-camera synchronization
- Real-time analytics and insights
- Advanced alert mechanisms
- Weapon detection
- Violence/fight detection
- PPE compliance detection
- Fall detection
- Crowd density estimation with heatmaps
- Loitering detection
- Abandoned object detection
"""

from .facial_recognition import FacialRecognitionSystem
from .behavior_analysis import BehaviorAnalyzer
from .person_reid import PersonReID
from .multi_camera_sync import MultiCameraManager
from .real_time_analytics import AnalyticsEngine
from .advanced_alerts import AdvancedAlertSystem
from .weapon_detection import WeaponDetector
from .violence_detection import ViolenceDetector
from .ppe_detection import PPEDetector
from .fall_detection import FallDetector
from .crowd_density import CrowdDensityEstimator
from .loitering_detection import LoiteringDetector
from .abandoned_object import AbandonedObjectDetector

__all__ = [
    'FacialRecognitionSystem',
    'BehaviorAnalyzer', 
    'PersonReID',
    'MultiCameraManager',
    'AnalyticsEngine',
    'AdvancedAlertSystem',
    'WeaponDetector',
    'ViolenceDetector',
    'PPEDetector',
    'FallDetector',
    'CrowdDensityEstimator',
    'LoiteringDetector',
    'AbandonedObjectDetector'
]

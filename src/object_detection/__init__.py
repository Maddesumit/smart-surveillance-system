"""
Object Detection Package

This package provides functionality for detecting and tracking objects in video frames
using deep learning models like YOLOv5.
"""

from .detector import ObjectDetector
from .tracker import ObjectTracker

__all__ = ['ObjectDetector', 'ObjectTracker']
"""
Video Processing Package

This package provides functionality for video capture and processing,
including camera integration, frame preprocessing, and motion detection.
"""

from .video_stream import VideoStream
from .processing_pipeline import ProcessingPipeline

__all__ = ['VideoStream', 'ProcessingPipeline']
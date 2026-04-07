#!/usr/bin/env python3
"""
Data Collection Module

This module provides functionality for collecting and managing
training data for surveillance model training.
"""

import os
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Data collection system for surveillance training data.
    
    This class handles:
    - Automatic frame collection from video streams
    - Data organization and management
    - Quality filtering and deduplication
    - Metadata tracking
    """
    
    def __init__(self, 
                 collection_dir: str = "collected_data",
                 db_path: str = "data_collection.db"):
        """
        Initialize the data collector.
        
        Args:
            collection_dir: Directory to store collected data
            db_path: Path to SQLite database for metadata
        """
        self.collection_dir = Path(collection_dir)
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.collection_dir / "images"
        self.annotations_dir = self.collection_dir / "annotations"
        self.metadata_dir = self.collection_dir / "metadata"
        
        for dir_path in [self.images_dir, self.annotations_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
        
        logger.info(f"DataCollector initialized with collection dir: {collection_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collected_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                timestamp DATETIME,
                source TEXT,
                width INTEGER,
                height INTEGER,
                quality_score REAL,
                object_count INTEGER,
                has_annotations BOOLEAN DEFAULT FALSE,
                is_validated BOOLEAN DEFAULT FALSE,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                start_time DATETIME,
                end_time DATETIME,
                total_frames INTEGER,
                source_info TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_collection_session(self, 
                                session_name: str,
                                video_source: int = 0,
                                collection_interval: float = 1.0,
                                duration_minutes: int = 10,
                                quality_threshold: float = 0.5) -> int:
        """
        Start an automated data collection session.
        
        Args:
            session_name: Name for this collection session
            video_source: Video source (0 for webcam, or path/URL)
            collection_interval: Interval between frame collections (seconds)
            duration_minutes: Duration of collection session
            quality_threshold: Minimum quality score for frame acceptance
            
        Returns:
            Number of frames collected
        """
        logger.info(f"Starting collection session: {session_name}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Record session start
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = datetime.now()
        cursor.execute('''
            INSERT INTO collection_sessions (session_name, start_time, source_info)
            VALUES (?, ?, ?)
        ''', (session_name, start_time, str(video_source)))
        
        session_id = cursor.lastrowid
        conn.commit()
        
        frames_collected = 0
        start_timestamp = datetime.now().timestamp()
        end_timestamp = start_timestamp + (duration_minutes * 60)
        
        try:
            while datetime.now().timestamp() < end_timestamp:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from video source")
                    continue
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(frame)
                
                if quality_score >= quality_threshold:
                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{session_name}_{timestamp}.jpg"
                    file_path = self.images_dir / filename
                    
                    cv2.imwrite(str(file_path), frame)
                    
                    # Store metadata
                    self._store_frame_metadata(
                        file_path=str(file_path),
                        frame=frame,
                        source=str(video_source),
                        quality_score=quality_score,
                        session_id=session_id
                    )
                    
                    frames_collected += 1
                    
                    if frames_collected % 10 == 0:
                        logger.info(f"Collected {frames_collected} frames...")
                
                # Wait for next collection
                cv2.waitKey(int(collection_interval * 1000))
        
        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Update session end time
            end_time = datetime.now()
            cursor.execute('''
                UPDATE collection_sessions 
                SET end_time = ?, total_frames = ?
                WHERE id = ?
            ''', (end_time, frames_collected, session_id))
            
            conn.commit()
            conn.close()
        
        logger.info(f"Collection session completed. Collected {frames_collected} frames")
        return frames_collected
    
    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """
        Calculate quality score for a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score between 0 and 1
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Calculate brightness distribution
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
        
        # Calculate contrast (standard deviation)
        contrast_score = min(np.std(gray) / 127.5, 1.0)
        
        # Combine scores
        quality_score = (sharpness_score * 0.5 + 
                        brightness_score * 0.3 + 
                        contrast_score * 0.2)
        
        return quality_score
    
    def _store_frame_metadata(self, 
                             file_path: str,
                             frame: np.ndarray,
                             source: str,
                             quality_score: float,
                             session_id: int):
        """Store frame metadata in database."""
        # Calculate file hash for deduplication
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        h, w = frame.shape[:2]
        timestamp = datetime.now()
        
        metadata = {
            'session_id': session_id,
            'collection_method': 'automated',
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO collected_frames 
                (file_path, file_hash, timestamp, source, width, height, 
                 quality_score, object_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, file_hash, timestamp, source, w, h, 
                  quality_score, 0, json.dumps(metadata)))
            
            conn.commit()
        except sqlite3.IntegrityError:
            # Duplicate frame (same file path)
            logger.debug(f"Duplicate frame skipped: {file_path}")
        
        conn.close()
    
    def collect_frames_from_video(self, 
                                 video_path: str,
                                 frame_interval: int = 30,
                                 max_frames: int = 1000) -> List[str]:
        """
        Extract frames from a video file for training data.
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frame paths
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        video_name = Path(video_path).stem
        extracted_frames = []
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"Extracting frames from: {video_path}")
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Calculate quality score
                quality_score = self._calculate_quality_score(frame)
                
                if quality_score > 0.3:  # Basic quality threshold
                    filename = f"{video_name}_frame_{extracted_count:06d}.jpg"
                    file_path = self.images_dir / filename
                    
                    cv2.imwrite(str(file_path), frame)
                    extracted_frames.append(str(file_path))
                    
                    # Store metadata
                    self._store_frame_metadata(
                        file_path=str(file_path),
                        frame=frame,
                        source=video_path,
                        quality_score=quality_score,
                        session_id=0  # No session for manual extraction
                    )
                    
                    extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from video")
        return extracted_frames
    
    def get_collected_data_summary(self) -> Dict:
        """
        Get summary of collected data.
        
        Returns:
            Summary statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total frames
        cursor.execute("SELECT COUNT(*) FROM collected_frames")
        total_frames = cursor.fetchone()[0]
        
        # Get annotated frames
        cursor.execute("SELECT COUNT(*) FROM collected_frames WHERE has_annotations = TRUE")
        annotated_frames = cursor.fetchone()[0]
        
        # Get average quality score
        cursor.execute("SELECT AVG(quality_score) FROM collected_frames")
        avg_quality = cursor.fetchone()[0] or 0
        
        # Get sessions
        cursor.execute("SELECT COUNT(*) FROM collection_sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Get frames by source
        cursor.execute("""
            SELECT source, COUNT(*) 
            FROM collected_frames 
            GROUP BY source
        """)
        frames_by_source = dict(cursor.fetchall())
        
        conn.close()
        
        summary = {
            'total_frames': total_frames,
            'annotated_frames': annotated_frames,
            'annotation_percentage': (annotated_frames / total_frames * 100) if total_frames > 0 else 0,
            'average_quality_score': round(avg_quality, 3),
            'total_sessions': total_sessions,
            'frames_by_source': frames_by_source
        }
        
        return summary
    
    def export_training_dataset(self, 
                               output_dir: str,
                               train_split: float = 0.8,
                               min_quality: float = 0.5,
                               annotated_only: bool = True) -> Dict[str, List[str]]:
        """
        Export collected data as a training dataset.
        
        Args:
            output_dir: Output directory for dataset
            train_split: Percentage for training set
            min_quality: Minimum quality score threshold
            annotated_only: Only include annotated frames
            
        Returns:
            Dictionary with train/val image and label paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset structure
        train_img_dir = output_dir / 'images' / 'train'
        val_img_dir = output_dir / 'images' / 'val'
        train_lbl_dir = output_dir / 'labels' / 'train'
        val_lbl_dir = output_dir / 'labels' / 'val'
        
        for dir_path in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get frames from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT file_path FROM collected_frames WHERE quality_score >= ?"
        params = [min_quality]
        
        if annotated_only:
            query += " AND has_annotations = TRUE"
        
        cursor.execute(query, params)
        frame_paths = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        if not frame_paths:
            logger.warning("No frames found matching criteria")
            return {}
        
        # Shuffle and split
        np.random.shuffle(frame_paths)
        split_idx = int(len(frame_paths) * train_split)
        
        train_frames = frame_paths[:split_idx]
        val_frames = frame_paths[split_idx:]
        
        # Copy files
        dataset_info = {
            'train_images': [],
            'train_labels': [],
            'val_images': [],
            'val_labels': []
        }
        
        for frames, img_dir, lbl_dir, split_name in [
            (train_frames, train_img_dir, train_lbl_dir, 'train'),
            (val_frames, val_img_dir, val_lbl_dir, 'val')
        ]:
            for frame_path in frames:
                src_img = Path(frame_path)
                src_lbl = self.annotations_dir / f"{src_img.stem}.txt"
                
                if annotated_only and not src_lbl.exists():
                    continue
                
                # Copy image
                dst_img = img_dir / src_img.name
                os.link(src_img, dst_img)  # Hard link to save space
                dataset_info[f'{split_name}_images'].append(str(dst_img))
                
                # Copy annotation if exists
                if src_lbl.exists():
                    dst_lbl = lbl_dir / src_lbl.name
                    os.link(src_lbl, dst_lbl)
                    dataset_info[f'{split_name}_labels'].append(str(dst_lbl))
        
        logger.info(f"Exported dataset to {output_dir}")
        logger.info(f"Train: {len(dataset_info['train_images'])} images")
        logger.info(f"Val: {len(dataset_info['val_images'])} images")
        
        return dataset_info


def main():
    """Example usage of the DataCollector."""
    collector = DataCollector()
    
    # Example: Start a collection session
    # collector.start_collection_session(
    #     session_name="test_session",
    #     duration_minutes=1,  # Short test session
    #     collection_interval=2.0
    # )
    
    # Get summary
    summary = collector.get_collected_data_summary()
    logger.info(f"Data collection summary: {summary}")


if __name__ == "__main__":
    main()

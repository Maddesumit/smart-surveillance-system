#!/usr/bin/env python3
"""
Crowd Density Estimation & Heatmap Module

Provides real-time crowd density estimation with visual heatmaps
showing concentration levels across the surveillance area.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque
import sqlite3

logger = logging.getLogger(__name__)


class CrowdDensityEstimator:
    """
    Crowd density estimation system with heatmap visualization.
    
    Features:
    - Real-time person density calculation per zone
    - Cumulative heatmap showing crowd patterns over time
    - Overcrowding alerts
    - Zone-based density monitoring
    - Historical density analytics
    """
    
    def __init__(self,
                 database_path: str = "crowd_density.db",
                 grid_size: Tuple[int, int] = (16, 12),
                 overcrowding_threshold: int = 10,
                 heatmap_decay: float = 0.95,
                 alert_cooldown: float = 60.0):
        """
        Initialize crowd density estimator.
        
        Args:
            database_path: Path to SQLite database
            grid_size: Grid dimensions for density calculation (cols, rows)
            overcrowding_threshold: Max persons in a zone before alert
            heatmap_decay: Decay factor for heatmap (0-1, higher = longer persistence)
            alert_cooldown: Seconds between overcrowding alerts
        """
        self.database_path = database_path
        self.grid_size = grid_size
        self.overcrowding_threshold = overcrowding_threshold
        self.heatmap_decay = heatmap_decay
        self.alert_cooldown = alert_cooldown
        
        # Heatmap accumulator
        self.heatmap = np.zeros((grid_size[1], grid_size[0]), dtype=np.float32)
        
        # Historical data
        self.density_history = deque(maxlen=300)  # 5 minutes at 1fps
        self.last_alert_time = None
        
        # Zone definitions (can be customized)
        self.zones = {}
        
        # Stats
        self.stats = {
            'max_density_ever': 0,
            'current_person_count': 0,
            'overcrowding_alerts': 0,
            'avg_density': 0.0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Crowd Density Estimator initialized (grid: {grid_size})")
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS density_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                person_count INTEGER,
                max_zone_density INTEGER,
                avg_density REAL,
                overcrowded_zones TEXT,
                heatmap_data BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS overcrowding_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                zone_id TEXT,
                person_count INTEGER,
                threshold INTEGER,
                duration_seconds REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def update(self,
               frame: np.ndarray,
               person_detections: List[Dict]) -> Dict:
        """
        Update crowd density estimation with new frame data.
        
        Args:
            frame: Input video frame
            person_detections: List of person detections
            
        Returns:
            Density analysis results
        """
        frame_h, frame_w = frame.shape[:2]
        cell_w = frame_w / self.grid_size[0]
        cell_h = frame_h / self.grid_size[1]
        
        # Current frame density grid
        current_grid = np.zeros((self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        
        # Count persons in each grid cell
        person_count = 0
        person_positions = []
        
        for person in person_detections:
            if person.get('class_name') != 'person':
                continue
            
            person_count += 1
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            # Person centroid
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            person_positions.append((cx, cy))
            
            # Map to grid cell
            grid_x = int(min(cx / cell_w, self.grid_size[0] - 1))
            grid_y = int(min(cy / cell_h, self.grid_size[1] - 1))
            
            # Add to current grid (with gaussian spread)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = grid_y + dy, grid_x + dx
                    if 0 <= ny < self.grid_size[1] and 0 <= nx < self.grid_size[0]:
                        distance = abs(dx) + abs(dy)
                        weight = 1.0 if distance == 0 else 0.3
                        current_grid[ny, nx] += weight
        
        # Update cumulative heatmap with decay
        self.heatmap = self.heatmap * self.heatmap_decay + current_grid
        
        # Find overcrowded zones
        max_zone_density = int(np.max(current_grid))
        overcrowded_zones = []
        
        if max_zone_density >= self.overcrowding_threshold:
            # Find which zones are overcrowded
            overcrowded_cells = np.argwhere(current_grid >= self.overcrowding_threshold)
            for cell in overcrowded_cells:
                overcrowded_zones.append({
                    'grid_y': int(cell[0]),
                    'grid_x': int(cell[1]),
                    'count': int(current_grid[cell[0], cell[1]])
                })
        
        # Update stats
        self.stats['current_person_count'] = person_count
        self.stats['max_density_ever'] = max(self.stats['max_density_ever'], max_zone_density)
        self.stats['avg_density'] = float(np.mean(current_grid))
        
        # Store in history
        self.density_history.append({
            'timestamp': datetime.now().isoformat(),
            'person_count': person_count,
            'max_density': max_zone_density
        })
        
        # Generate alerts
        alerts = []
        if overcrowded_zones:
            alerts = self._generate_overcrowding_alerts(overcrowded_zones, person_count)
        
        return {
            'person_count': person_count,
            'max_zone_density': max_zone_density,
            'avg_density': float(np.mean(current_grid)),
            'overcrowded_zones': overcrowded_zones,
            'alerts': alerts,
            'grid': current_grid,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_overcrowding_alerts(self, overcrowded_zones: List[Dict], person_count: int) -> List[Dict]:
        """Generate overcrowding alerts with cooldown."""
        current_time = datetime.now()
        
        if (self.last_alert_time and 
            (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown):
            return []
        
        self.last_alert_time = current_time
        self.stats['overcrowding_alerts'] += 1
        
        alerts = [{
            'type': 'overcrowding',
            'message': f"Overcrowding detected! {person_count} persons, max zone density: {overcrowded_zones[0]['count']}",
            'priority': 'high' if person_count > self.overcrowding_threshold * 2 else 'medium',
            'timestamp': current_time.isoformat(),
            'person_count': person_count,
            'zones': overcrowded_zones
        }]
        
        return alerts
    
    def generate_heatmap_overlay(self,
                                  frame: np.ndarray,
                                  opacity: float = 0.4) -> np.ndarray:
        """
        Generate heatmap overlay on the frame.
        
        Args:
            frame: Input video frame
            opacity: Heatmap overlay opacity (0-1)
            
        Returns:
            Frame with heatmap overlay
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Normalize heatmap to 0-255
        if np.max(self.heatmap) > 0:
            normalized = (self.heatmap / np.max(self.heatmap) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.heatmap, dtype=np.uint8)
        
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(normalized, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Create mask (only show where there's density)
        mask = heatmap_resized > 10  # Threshold to avoid showing very low values
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        
        # Blend with original frame
        output = frame.copy()
        output[mask_3ch] = cv2.addWeighted(
            frame, 1 - opacity, heatmap_colored, opacity, 0
        )[mask_3ch]
        
        # Add density info text
        cv2.putText(output, f"Persons: {self.stats['current_person_count']}", 
                   (10, frame_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Max Density: {self.stats['max_density_ever']}", 
                   (10, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output
    
    def get_density_history(self, minutes: int = 5) -> List[Dict]:
        """Get density history for the last N minutes."""
        return list(self.density_history)
    
    def reset_heatmap(self):
        """Reset the cumulative heatmap."""
        self.heatmap = np.zeros((self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        logger.info("Heatmap reset")
    
    def get_stats(self) -> Dict:
        """Get crowd density statistics."""
        return self.stats.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Crowd Density Estimator cleanup complete")

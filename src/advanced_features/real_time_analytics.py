#!/usr/bin/env python3
"""
Real-Time Analytics Engine

This module provides comprehensive real-time analytics and insights
for surveillance systems including performance monitoring, trend analysis,
and predictive analytics.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import sqlite3
import json
import threading
import time
from collections import defaultdict, deque
import statistics

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Plotting libraries not available. Install with: pip install matplotlib seaborn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Real-time analytics engine for surveillance systems.
    
    Features:
    - Real-time performance monitoring
    - Activity pattern analysis
    - Crowd dynamics tracking
    - Anomaly trend detection
    - Predictive analytics
    - Custom dashboard metrics
    - Alert correlation analysis
    - Resource utilization tracking
    """
    
    def __init__(self,
                 database_path: str = "analytics.db",
                 analysis_window: int = 3600,  # seconds (1 hour)
                 update_interval: int = 60):   # seconds (1 minute)
        """
        Initialize the analytics engine.
        
        Args:
            database_path: Path to SQLite database
            analysis_window: Time window for analysis (seconds)
            update_interval: Update interval for real-time metrics
        """
        self.database_path = database_path
        self.analysis_window = analysis_window
        self.update_interval = update_interval
        
        # Real-time data storage
        self.metrics_buffer = defaultdict(deque)  # metric_name -> values
        self.event_buffer = deque()  # Recent events
        self.performance_metrics = {}  # Current performance state
        
        # Analytics data
        self.activity_patterns = {}  # Time-based activity patterns
        self.crowd_analytics = {}   # Crowd behavior analytics
        self.alert_correlation = {} # Alert pattern correlation
        self.trend_analysis = {}    # Trend analysis results
        
        # System monitoring
        self.system_health = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_traffic': 0.0,
            'camera_status': {},
            'processing_latency': 0.0
        }
        
        # Threading
        self.is_running = False
        self.analytics_thread = None
        self.analytics_lock = threading.Lock()
        
        # Initialize components
        self._init_database()
        
        logger.info("Analytics Engine initialized")
    
    def _init_database(self):
        """Initialize SQLite database for analytics data."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Real-time metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    metric_unit TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    tags TEXT
                )
            ''')
            
            # Activity patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    time_period TEXT,
                    pattern_data TEXT,
                    confidence REAL,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Analytics insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT,
                    title TEXT,
                    description TEXT,
                    severity TEXT,
                    data TEXT,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Performance benchmarks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_name TEXT,
                    target_value REAL,
                    current_value REAL,
                    threshold_min REAL,
                    threshold_max REAL,
                    unit TEXT,
                    category TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Analytics database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing analytics database: {str(e)}")
    
    def start_analytics(self):
        """Start real-time analytics processing."""
        try:
            if self.is_running:
                logger.warning("Analytics already running")
                return
            
            self.is_running = True
            
            # Start analytics thread
            self.analytics_thread = threading.Thread(
                target=self._analytics_loop,
                daemon=True
            )
            self.analytics_thread.start()
            
            logger.info("Real-time analytics started")
            
        except Exception as e:
            logger.error(f"Error starting analytics: {str(e)}")
    
    def stop_analytics(self):
        """Stop real-time analytics processing."""
        try:
            self.is_running = False
            
            if self.analytics_thread and self.analytics_thread.is_alive():
                self.analytics_thread.join(timeout=5.0)
            
            logger.info("Real-time analytics stopped")
            
        except Exception as e:
            logger.error(f"Error stopping analytics: {str(e)}")
    
    def _analytics_loop(self):
        """Main analytics processing loop."""
        try:
            while self.is_running:
                start_time = time.time()
                
                # Update real-time metrics
                self._update_realtime_metrics()
                
                # Analyze activity patterns
                self._analyze_activity_patterns()
                
                # Perform trend analysis
                self._analyze_trends()
                
                # Generate insights
                self._generate_insights()
                
                # Update system health
                self._update_system_health()
                
                # Control update frequency
                processing_time = time.time() - start_time
                sleep_time = max(0, self.update_interval - processing_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in analytics loop: {str(e)}")
    
    def record_metric(self, 
                     metric_name: str, 
                     value: float, 
                     unit: str = "", 
                     source: str = "system",
                     tags: Dict = None):
        """
        Record a real-time metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            source: Source of the metric
            tags: Additional tags as dictionary
        """
        try:
            with self.analytics_lock:
                # Add to buffer
                timestamp = datetime.now()
                metric_data = {
                    'value': value,
                    'timestamp': timestamp,
                    'unit': unit,
                    'source': source,
                    'tags': tags or {}
                }
                
                self.metrics_buffer[metric_name].append(metric_data)
                
                # Limit buffer size
                if len(self.metrics_buffer[metric_name]) > 1000:
                    self.metrics_buffer[metric_name].popleft()
                
                # Store in database
                self._store_metric(metric_name, value, unit, source, tags)
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {str(e)}")
    
    def record_event(self, 
                    event_type: str, 
                    event_data: Dict, 
                    severity: str = "info",
                    source: str = "system"):
        """
        Record a system event for analysis.
        
        Args:
            event_type: Type of event
            event_data: Event data dictionary
            severity: Event severity level
            source: Event source
        """
        try:
            with self.analytics_lock:
                event = {
                    'type': event_type,
                    'data': event_data,
                    'severity': severity,
                    'source': source,
                    'timestamp': datetime.now()
                }
                
                self.event_buffer.append(event)
                
                # Limit buffer size
                if len(self.event_buffer) > 5000:
                    self.event_buffer.popleft()
            
        except Exception as e:
            logger.error(f"Error recording event {event_type}: {str(e)}")
    
    def _update_realtime_metrics(self):
        """Update real-time performance metrics."""
        try:
            current_time = datetime.now()
            
            # Calculate metrics from recent data
            for metric_name, metric_data in self.metrics_buffer.items():
                if not metric_data:
                    continue
                
                # Filter recent data (last hour)
                recent_data = [
                    item for item in metric_data
                    if (current_time - item['timestamp']).total_seconds() < self.analysis_window
                ]
                
                if recent_data:
                    values = [item['value'] for item in recent_data]
                    
                    # Calculate statistics
                    self.performance_metrics[metric_name] = {
                        'current': values[-1] if values else 0,
                        'average': statistics.mean(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                        'trend': self._calculate_trend(values),
                        'last_updated': current_time
                    }
            
        except Exception as e:
            logger.error(f"Error updating realtime metrics: {str(e)}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        try:
            if len(values) < 2:
                return "stable"
            
            # Simple trend calculation using linear regression slope
            x = list(range(len(values)))
            n = len(values)
            
            slope = (n * sum(x[i] * values[i] for i in range(n)) - sum(x) * sum(values)) / \
                   (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
            
            threshold = 0.1 * (max(values) - min(values)) / len(values)
            
            if slope > threshold:
                return "increasing"
            elif slope < -threshold:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            return "unknown"
    
    def _analyze_activity_patterns(self):
        """Analyze activity patterns over time."""
        try:
            current_time = datetime.now()
            
            # Analyze hourly patterns
            hourly_activity = defaultdict(list)
            daily_activity = defaultdict(list)
            
            # Process recent events
            cutoff_time = current_time - timedelta(seconds=self.analysis_window)
            recent_events = [
                event for event in self.event_buffer
                if event['timestamp'] > cutoff_time
            ]
            
            for event in recent_events:
                hour = event['timestamp'].hour
                day = event['timestamp'].strftime('%A')
                
                hourly_activity[hour].append(event)
                daily_activity[day].append(event)
            
            # Calculate patterns
            self.activity_patterns = {
                'hourly': {
                    hour: {
                        'count': len(events),
                        'event_types': list(set(e['type'] for e in events)),
                        'peak_activity': len(events) > statistics.mean([len(v) for v in hourly_activity.values()]) if hourly_activity.values() else False
                    }
                    for hour, events in hourly_activity.items()
                },
                'daily': {
                    day: {
                        'count': len(events),
                        'event_types': list(set(e['type'] for e in events)),
                        'activity_level': self._classify_activity_level(len(events))
                    }
                    for day, events in daily_activity.items()
                },
                'last_updated': current_time
            }
            
        except Exception as e:
            logger.error(f"Error analyzing activity patterns: {str(e)}")
    
    def _classify_activity_level(self, event_count: int) -> str:
        """Classify activity level based on event count."""
        if event_count == 0:
            return "inactive"
        elif event_count < 10:
            return "low"
        elif event_count < 50:
            return "moderate"
        elif event_count < 100:
            return "high"
        else:
            return "very_high"
    
    def _analyze_trends(self):
        """Perform trend analysis on metrics and events."""
        try:
            current_time = datetime.now()
            
            # Analyze metric trends
            metric_trends = {}
            for metric_name, metric_stats in self.performance_metrics.items():
                if metric_name in self.metrics_buffer:
                    metric_data = self.metrics_buffer[metric_name]
                    values = [item['value'] for item in metric_data]
                    
                    if len(values) >= 10:  # Need minimum data points
                        trend_info = {
                            'direction': metric_stats.get('trend', 'stable'),
                            'volatility': metric_stats.get('std_dev', 0) / metric_stats.get('average', 1),
                            'recent_change': self._calculate_recent_change(values),
                            'prediction': self._predict_next_value(values)
                        }
                        metric_trends[metric_name] = trend_info
            
            # Analyze event trends
            event_trends = self._analyze_event_trends()
            
            self.trend_analysis = {
                'metrics': metric_trends,
                'events': event_trends,
                'last_updated': current_time
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
    
    def _calculate_recent_change(self, values: List[float]) -> float:
        """Calculate recent change percentage."""
        try:
            if len(values) < 2:
                return 0.0
            
            recent_avg = statistics.mean(values[-5:])  # Last 5 values
            older_avg = statistics.mean(values[-10:-5]) if len(values) >= 10 else values[0]
            
            if older_avg == 0:
                return 0.0
            
            return ((recent_avg - older_avg) / older_avg) * 100
            
        except Exception as e:
            logger.error(f"Error calculating recent change: {str(e)}")
            return 0.0
    
    def _predict_next_value(self, values: List[float]) -> float:
        """Simple prediction of next value using linear trend."""
        try:
            if len(values) < 3:
                return values[-1] if values else 0.0
            
            # Simple linear extrapolation
            recent_values = values[-5:]  # Use last 5 values
            x = list(range(len(recent_values)))
            n = len(recent_values)
            
            # Linear regression
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(recent_values)
            
            slope = sum((x[i] - x_mean) * (recent_values[i] - y_mean) for i in range(n)) / \
                   sum((x[i] - x_mean)**2 for i in range(n))
            
            intercept = y_mean - slope * x_mean
            
            # Predict next value
            next_x = len(recent_values)
            prediction = slope * next_x + intercept
            
            return max(0, prediction)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error predicting next value: {str(e)}")
            return values[-1] if values else 0.0
    
    def _analyze_event_trends(self) -> Dict:
        """Analyze trends in event occurrences."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(seconds=self.analysis_window)
            
            # Group events by type and time
            event_timeline = defaultdict(list)
            
            for event in self.event_buffer:
                if event['timestamp'] > cutoff_time:
                    event_type = event['type']
                    event_timeline[event_type].append(event['timestamp'])
            
            # Analyze each event type
            event_trends = {}
            for event_type, timestamps in event_timeline.items():
                if len(timestamps) >= 3:
                    # Calculate frequency trend
                    time_intervals = []
                    sorted_times = sorted(timestamps)
                    
                    for i in range(1, len(sorted_times)):
                        interval = (sorted_times[i] - sorted_times[i-1]).total_seconds()
                        time_intervals.append(interval)
                    
                    avg_interval = statistics.mean(time_intervals) if time_intervals else 0
                    frequency = 3600 / avg_interval if avg_interval > 0 else 0  # Events per hour
                    
                    event_trends[event_type] = {
                        'frequency': frequency,
                        'total_count': len(timestamps),
                        'avg_interval': avg_interval,
                        'trend': self._calculate_frequency_trend(time_intervals)
                    }
            
            return event_trends
            
        except Exception as e:
            logger.error(f"Error analyzing event trends: {str(e)}")
            return {}
    
    def _calculate_frequency_trend(self, intervals: List[float]) -> str:
        """Calculate trend in event frequency."""
        try:
            if len(intervals) < 3:
                return "stable"
            
            # Compare recent intervals with older ones
            recent_avg = statistics.mean(intervals[-3:])
            older_avg = statistics.mean(intervals[:-3])
            
            if recent_avg < older_avg * 0.8:  # 20% decrease in interval = increase in frequency
                return "increasing"
            elif recent_avg > older_avg * 1.2:  # 20% increase in interval = decrease in frequency
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating frequency trend: {str(e)}")
            return "unknown"
    
    def _generate_insights(self):
        """Generate analytical insights from data."""
        try:
            insights = []
            
            # Performance insights
            for metric_name, metric_stats in self.performance_metrics.items():
                # High variability insight
                if metric_stats.get('std_dev', 0) > metric_stats.get('average', 0) * 0.5:
                    insights.append({
                        'type': 'performance',
                        'severity': 'warning',
                        'title': f'High Variability in {metric_name}',
                        'description': f'{metric_name} shows high variability (std dev: {metric_stats["std_dev"]:.2f})',
                        'recommendation': f'Investigate causes of {metric_name} fluctuations'
                    })
                
                # Trend insights
                if metric_stats.get('trend') == 'increasing' and 'error' in metric_name.lower():
                    insights.append({
                        'type': 'performance',
                        'severity': 'critical',
                        'title': f'Increasing {metric_name}',
                        'description': f'{metric_name} is showing an increasing trend',
                        'recommendation': 'Immediate investigation required'
                    })
            
            # Activity pattern insights
            if self.activity_patterns:
                hourly_data = self.activity_patterns.get('hourly', {})
                peak_hours = [
                    hour for hour, data in hourly_data.items()
                    if data.get('peak_activity', False)
                ]
                
                if peak_hours:
                    insights.append({
                        'type': 'activity',
                        'severity': 'info',
                        'title': 'Peak Activity Hours Identified',
                        'description': f'Peak activity detected during hours: {peak_hours}',
                        'recommendation': 'Consider resource allocation during peak hours'
                    })
            
            # Store insights
            for insight in insights:
                self._store_insight(insight)
                
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
    
    def _update_system_health(self):
        """Update system health metrics."""
        try:
            # Basic system monitoring (simplified)
            self.system_health.update({
                'last_updated': datetime.now(),
                'analytics_engine_status': 'healthy' if self.is_running else 'stopped',
                'total_metrics': len(self.metrics_buffer),
                'total_events': len(self.event_buffer),
                'active_patterns': len(self.activity_patterns),
                'memory_usage_mb': len(self.metrics_buffer) * 0.1  # Rough estimate
            })
            
        except Exception as e:
            logger.error(f"Error updating system health: {str(e)}")
    
    def _store_metric(self, metric_name: str, value: float, unit: str, source: str, tags: Dict):
        """Store metric in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_metrics 
                (metric_name, metric_value, metric_unit, source, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', (metric_name, value, unit, source, json.dumps(tags or {})))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing metric: {str(e)}")
    
    def _store_insight(self, insight: Dict):
        """Store insight in database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analytics_insights 
                (insight_type, title, description, severity, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                insight['type'],
                insight['title'],
                insight['description'],
                insight['severity'],
                json.dumps(insight)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing insight: {str(e)}")
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data."""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'activity_patterns': self.activity_patterns,
                'trend_analysis': self.trend_analysis,
                'system_health': self.system_health,
                'recent_insights': self._get_recent_insights(limit=10)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {}
    
    def _get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """Get recent insights from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT insight_type, title, description, severity, data, created_time
                FROM analytics_insights
                WHERE status = 'active'
                ORDER BY created_time DESC
                LIMIT ?
            ''', (limit,))
            
            insights = []
            for row in cursor.fetchall():
                insight = {
                    'type': row[0],
                    'title': row[1],
                    'description': row[2],
                    'severity': row[3],
                    'data': json.loads(row[4]) if row[4] else {},
                    'created_time': row[5]
                }
                insights.append(insight)
            
            conn.close()
            return insights
            
        except Exception as e:
            logger.error(f"Error getting recent insights: {str(e)}")
            return []
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict]:
        """Get historical data for a specific metric."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute('''
                SELECT metric_value, timestamp, source, tags
                FROM realtime_metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (metric_name, cutoff_time.isoformat()))
            
            history = []
            for row in cursor.fetchall():
                entry = {
                    'value': row[0],
                    'timestamp': row[1],
                    'source': row[2],
                    'tags': json.loads(row[3]) if row[3] else {}
                }
                history.append(entry)
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"Error getting metric history: {str(e)}")
            return []
    
    def generate_report(self, report_type: str = "summary", time_range: int = 24) -> Dict:
        """
        Generate analytical report.
        
        Args:
            report_type: Type of report ('summary', 'performance', 'activity')
            time_range: Time range in hours
            
        Returns:
            Report data dictionary
        """
        try:
            current_time = datetime.now()
            start_time = current_time - timedelta(hours=time_range)
            
            report = {
                'report_type': report_type,
                'generated_time': current_time.isoformat(),
                'time_range_hours': time_range,
                'start_time': start_time.isoformat(),
                'end_time': current_time.isoformat()
            }
            
            if report_type == "summary":
                report.update({
                    'total_metrics': len(self.performance_metrics),
                    'total_events': len(self.event_buffer),
                    'system_health': self.system_health,
                    'key_insights': self._get_recent_insights(limit=5),
                    'top_metrics': self._get_top_metrics_summary()
                })
            
            elif report_type == "performance":
                report.update({
                    'performance_metrics': self.performance_metrics,
                    'trend_analysis': self.trend_analysis,
                    'performance_insights': [
                        insight for insight in self._get_recent_insights(limit=20)
                        if insight['type'] == 'performance'
                    ]
                })
            
            elif report_type == "activity":
                report.update({
                    'activity_patterns': self.activity_patterns,
                    'event_trends': self.trend_analysis.get('events', {}),
                    'activity_insights': [
                        insight for insight in self._get_recent_insights(limit=20)
                        if insight['type'] == 'activity'
                    ]
                })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {'error': str(e)}
    
    def _get_top_metrics_summary(self) -> List[Dict]:
        """Get summary of top metrics."""
        try:
            top_metrics = []
            
            for metric_name, metric_stats in self.performance_metrics.items():
                summary = {
                    'name': metric_name,
                    'current_value': metric_stats.get('current', 0),
                    'average': metric_stats.get('average', 0),
                    'trend': metric_stats.get('trend', 'stable'),
                    'unit': 'units'  # Would need to be tracked separately
                }
                top_metrics.append(summary)
            
            # Sort by relevance (example: by standard deviation)
            top_metrics.sort(key=lambda x: abs(x['current_value'] - x['average']), reverse=True)
            
            return top_metrics[:10]  # Top 10 metrics
            
        except Exception as e:
            logger.error(f"Error getting top metrics summary: {str(e)}")
            return []
    
    def export_data(self, format_type: str = "json", time_range: int = 24) -> str:
        """
        Export analytics data in specified format.
        
        Args:
            format_type: Export format ('json', 'csv')
            time_range: Time range in hours
            
        Returns:
            Exported data as string
        """
        try:
            data = self.generate_report("summary", time_range)
            
            if format_type == "json":
                return json.dumps(data, indent=2, default=str)
            
            elif format_type == "csv":
                # Convert to CSV format (simplified)
                csv_data = "metric_name,current_value,average,trend\n"
                
                for metric in data.get('top_metrics', []):
                    csv_data += f"{metric['name']},{metric['current_value']},{metric['average']},{metric['trend']}\n"
                
                return csv_data
            
            else:
                return json.dumps(data, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return ""
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.stop_analytics()
            logger.info("Analytics engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Example usage of the Analytics Engine."""
    # Initialize analytics engine
    analytics = AnalyticsEngine()
    
    # Start analytics
    analytics.start_analytics()
    
    try:
        # Simulate recording metrics
        for i in range(100):
            # Record some sample metrics
            analytics.record_metric("detection_count", np.random.randint(0, 50), "detections")
            analytics.record_metric("processing_fps", np.random.uniform(20, 30), "fps")
            analytics.record_metric("memory_usage", np.random.uniform(40, 80), "percentage")
            
            # Record some events
            if i % 10 == 0:
                analytics.record_event("person_detected", {"confidence": 0.8}, "info")
            
            if i % 25 == 0:
                analytics.record_event("anomaly_detected", {"type": "loitering"}, "warning")
            
            time.sleep(1)
            
            # Print dashboard data every 20 iterations
            if i % 20 == 0:
                dashboard_data = analytics.get_dashboard_data()
                print(f"Iteration {i}: {len(dashboard_data.get('performance_metrics', {}))} metrics tracked")
    
    except KeyboardInterrupt:
        print("Stopping analytics...")
    
    finally:
        analytics.cleanup()


if __name__ == "__main__":
    main()

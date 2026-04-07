#!/usr/bin/env python3
"""
Test script for Behavior Analysis functionality
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from advanced_features.behavior_analysis import BehaviorAnalyzer
    print("✅ BehaviorAnalyzer imported successfully")
    
    # Initialize behavior analyzer
    analyzer = BehaviorAnalyzer()
    print("✅ BehaviorAnalyzer initialized successfully")
    
    # Test rule management
    rules = analyzer.get_behavior_rules()
    print(f"✅ Got behavior rules: {len(rules)} rules loaded")
    print(f"   - Enabled: {rules['enabled']}")
    print(f"   - Rules: {list(rules['rules'].keys())}")
    print(f"   - Patterns: {list(rules['patterns'].keys())}")
    
    # Test rule updates
    test_rules = {
        'enabled': True,
        'rules': {
            'loitering_threshold': 600,  # 10 minutes
            'running_speed_threshold': 3.0
        }
    }
    
    success = analyzer.update_behavior_rules(test_rules)
    print(f"✅ Rule update {'successful' if success else 'failed'}")
    
    # Test activity patterns
    patterns = analyzer.get_activity_patterns('24h')
    print(f"✅ Activity patterns retrieved: {patterns.get('total_events', 0)} total events")
    
    # Test recent events
    events = analyzer.get_recent_behavior_events(limit=5)
    print(f"✅ Recent events retrieved: {len(events)} events")
    
    # Test detailed stats
    stats = analyzer.get_detailed_stats('24h')
    print(f"✅ Detailed stats retrieved:")
    print(f"   - Enabled: {stats['enabled']}")
    print(f"   - Total events: {stats['total_events']}")
    print(f"   - Suspicious events: {stats['suspicious_events']}")
    print(f"   - Unique persons: {stats['unique_persons']}")
    
    # Test behavior stats
    behavior_stats = analyzer.get_behavior_stats()
    print(f"✅ Behavior stats: {behavior_stats}")
    
    # Cleanup
    analyzer.cleanup()
    print("✅ Behavior analysis test completed successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed")
    
except Exception as e:
    print(f"❌ Error testing behavior analysis: {e}")
    import traceback
    traceback.print_exc()

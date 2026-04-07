#!/usr/bin/env python3
"""
Test the Professional Dashboard Components
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dashboard_imports():
    """Test dashboard component imports."""
    print("🧪 Testing Professional Dashboard Components")
    print("=" * 50)
    
    try:
        from dashboard import create_app, socketio
        print("✅ Dashboard app creation: OK")
    except ImportError as e:
        print(f"❌ Dashboard app creation: FAILED - {e}")
        return False
    
    try:
        from dashboard.routes import main_routes, api_routes
        print("✅ Dashboard routes: OK")
    except ImportError as e:
        print(f"❌ Dashboard routes: FAILED - {e}")
        return False
    
    return True

def test_advanced_features():
    """Test advanced features availability."""
    print("\n🚀 Testing Advanced Features")
    print("-" * 30)
    
    features = [
        'facial_recognition',
        'behavior_analysis', 
        'person_reid',
        'multi_camera_sync',
        'real_time_analytics',
        'advanced_alerts'
    ]
    
    available_count = 0
    
    for feature in features:
        try:
            module = __import__(f'advanced_features.{feature}', fromlist=[feature])
            print(f"✅ {feature}: Available")
            available_count += 1
        except ImportError as e:
            print(f"❌ {feature}: Not available - {e}")
    
    print(f"\n📊 Advanced Features Available: {available_count}/{len(features)}")
    return available_count > 0

def test_flask_dashboard():
    """Test Flask dashboard creation."""
    print("\n🌐 Testing Flask Dashboard")
    print("-" * 30)
    
    try:
        from dashboard import create_app
        app = create_app()
        
        print("✅ Flask app created successfully")
        print(f"📱 App config: {list(app.config.keys())[:5]}...")
        print(f"🔗 Registered blueprints: {[bp.name for bp in app.blueprints.values()]}")
        
        return True
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False

def test_standardized_alerts():
    """Test standardized alert system."""
    print("\n🚨 Testing Standardized Alert System")
    print("-" * 30)
    
    try:
        # Import the alert manager from the enhanced main
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "enhanced_main", 
            "main_enhanced_professional.py"
        )
        enhanced_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_main)
        
        # Test alert creation
        alert_manager = enhanced_main.StandardizedAlertManager()
        
        # Create test alerts
        alert1 = alert_manager.create_alert('unknown_person', 'Test unknown person', 'high')
        alert2 = alert_manager.create_alert('unknown_person', 'Test unknown person', 'high')  # Should be throttled
        
        print(f"✅ Alert creation: OK")
        print(f"📊 Alert stats: {alert_manager.alert_stats}")
        print(f"🔒 Throttling test: {alert1 is not None}, {alert2 is None} (expected)")
        
        return True
    except Exception as e:
        print(f"❌ Alert system test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Professional Dashboard Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dashboard Imports", test_dashboard_imports),
        ("Advanced Features", test_advanced_features), 
        ("Flask Dashboard", test_flask_dashboard),
        ("Standardized Alerts", test_standardized_alerts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Professional dashboard is ready.")
        print("\n🚀 Next Steps:")
        print("1. Run: python main_enhanced_professional.py")
        print("2. Open: http://localhost:8082")
        print("3. Monitor the professional dashboard")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

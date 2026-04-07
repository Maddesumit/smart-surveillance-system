#!/usr/bin/env python3
"""
Quick test of the enhanced surveillance system
"""

import os
import sys
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Test the enhanced main system
def test_enhanced_system():
    """Test the enhanced surveillance system initialization."""
    print("🧠 Testing Enhanced Surveillance System...")
    
    try:
        # Import the enhanced system
        from main_enhanced import EnhancedSurveillanceSystem
        
        print("✅ Enhanced system imported successfully")
        
        # Create system instance
        system = EnhancedSurveillanceSystem()
        print("✅ System instance created")
        
        # Test core component initialization
        print("🔧 Testing core components initialization...")
        system.initialize_core_components()
        print("✅ Core components initialized")
        
        # Test advanced features initialization
        print("🚀 Testing advanced features initialization...")
        system.initialize_advanced_features()
        print("✅ Advanced features initialized")
        
        # Check which advanced features are available
        features_status = {
            'Facial Recognition': system.face_recognition is not None,
            'Behavior Analysis': system.behavior_analyzer is not None,
            'Person Re-ID': system.person_reid is not None,
            'Multi-Camera Sync': system.camera_manager is not None,
            'Real-time Analytics': system.analytics_engine is not None,
            'Advanced Alerts': system.advanced_alerts is not None
        }
        
        print("\n📊 Advanced Features Status:")
        for feature, status in features_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {feature}: {'Available' if status else 'Not Available'}")
        
        # Cleanup
        system._cleanup()
        print("\n✅ System test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {str(e)}")
        return False

def main():
    """Run the test."""
    success = test_enhanced_system()
    
    if success:
        print("\n🎉 Enhanced Surveillance System is ready!")
        print("\n🚀 To run the full system:")
        print("   python main_enhanced.py")
        print("\n📊 To view advanced features status:")
        print("   python test_advanced_features.py")
    else:
        print("\n⚠️ Please check the error messages above and fix any issues.")

if __name__ == "__main__":
    main()

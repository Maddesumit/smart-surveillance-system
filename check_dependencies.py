#!/usr/bin/env python3
"""
Dependency Check Script

Quick script to verify all required dependencies are properly installed.
"""

import sys

def check_package(package_name, import_name=None, version_attr=None):
    """Check if a package is installed and optionally get version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        
        # Try to get version
        version = "unknown"
        if version_attr:
            version = getattr(module, version_attr, "unknown")
        elif hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        
        print(f"✓ {package_name:<20} - Version: {version}")
        return True
        
    except ImportError as e:
        print(f"✗ {package_name:<20} - Not installed ({str(e)})")
        return False

def main():
    """Check all required dependencies."""
    print("Checking Smart Surveillance System Dependencies...")
    print("=" * 55)
    
    packages_to_check = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('flask', 'flask'),
        ('scipy', 'scipy'),
        ('pillow', 'PIL'),
        ('pyyaml', 'yaml'),
        ('tqdm', 'tqdm'),
        ('python-dotenv', 'dotenv')
    ]
    
    all_installed = True
    missing_packages = []
    
    for package_info in packages_to_check:
        if len(package_info) == 2:
            package_name, import_name = package_info
        else:
            package_name = import_name = package_info[0]
        
        if not check_package(package_name, import_name):
            all_installed = False
            missing_packages.append(package_name)
    
    print("\n" + "=" * 55)
    
    if all_installed:
        print("🎉 All dependencies are installed correctly!")
        print("\nYou can now run:")
        print("  python setup_training.py")
        print("  python train_model.py --help")
        print("  python test_accuracy.py --help")
    else:
        print(f"❌ Missing {len(missing_packages)} dependencies:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
    
    print("=" * 55)
    
    # Test OpenCV specifically since it's often problematic
    print("\nTesting OpenCV functionality...")
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test basic functionality
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV basic operations working")
        
        # Test video capture availability
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera access available")
            cap.release()
        else:
            print("⚠️  Camera not available (this is OK if no camera connected)")
            
    except Exception as e:
        print(f"✗ OpenCV test failed: {str(e)}")
    
    # Test YOLO model loading
    print("\nTesting YOLO model loading...")
    try:
        from ultralytics import YOLO
        # Don't actually load the model, just test import
        print("✓ Ultralytics YOLO available")
    except Exception as e:
        print(f"✗ YOLO test failed: {str(e)}")
    
    print("\nDependency check complete!")

if __name__ == "__main__":
    main()

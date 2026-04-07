#!/usr/bin/env python3
"""
OpenCV Fix Script

This script helps diagnose and fix common OpenCV installation issues.
"""

import sys
import subprocess
import os

def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_opencv_installation():
    """Check current OpenCV installation status."""
    print("🔍 Checking OpenCV installation...")
    
    # Check if opencv is importable
    try:
        import cv2
        version = cv2.__version__
        print(f"✓ OpenCV is installed: version {version}")
        return True, version
    except ImportError as e:
        print(f"✗ OpenCV import failed: {str(e)}")
        return False, None

def check_pip_packages():
    """Check what OpenCV-related packages are installed via pip."""
    print("\n🔍 Checking pip packages...")
    
    success, stdout, stderr = run_command("pip list | grep -i opencv")
    if success and stdout.strip():
        print("📦 Found OpenCV packages:")
        print(stdout)
        return True
    else:
        print("📦 No OpenCV packages found in pip")
        return False

def fix_opencv_installation():
    """Attempt to fix OpenCV installation."""
    print("\n🔧 Attempting to fix OpenCV installation...")
    
    # Step 1: Uninstall all opencv packages
    print("Step 1: Removing existing OpenCV packages...")
    opencv_packages = [
        "opencv-python",
        "opencv-contrib-python", 
        "opencv-python-headless",
        "opencv-contrib-python-headless"
    ]
    
    for pkg in opencv_packages:
        print(f"  Uninstalling {pkg}...")
        run_command(f"pip uninstall -y {pkg}")
    
    # Step 2: Clear pip cache
    print("Step 2: Clearing pip cache...")
    run_command("pip cache purge")
    
    # Step 3: Install opencv-python
    print("Step 3: Installing opencv-python...")
    success, stdout, stderr = run_command("pip install opencv-python>=4.5.0")
    
    if success:
        print("✓ opencv-python installed successfully")
    else:
        print(f"✗ Failed to install opencv-python: {stderr}")
        
        # Try headless version
        print("Trying headless version...")
        success, stdout, stderr = run_command("pip install opencv-python-headless>=4.5.0")
        
        if success:
            print("✓ opencv-python-headless installed successfully")
        else:
            print(f"✗ Failed to install opencv-python-headless: {stderr}")
            return False
    
    # Step 4: Test installation
    print("Step 4: Testing installation...")
    success, version = check_opencv_installation()
    
    if success:
        print(f"🎉 OpenCV installation successful! Version: {version}")
        return True
    else:
        print("❌ OpenCV installation still failing")
        return False

def test_opencv_functionality():
    """Test basic OpenCV functionality."""
    print("\n🧪 Testing OpenCV functionality...")
    
    try:
        import cv2
        import numpy as np
        
        # Test 1: Create image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("✓ Image creation works")
        
        # Test 2: Color conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✓ Color conversion works")
        
        # Test 3: Basic operations
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print("✓ Image filtering works")
        
        # Test 4: Feature detection
        detector = cv2.ORB_create()
        kp, des = detector.detectAndCompute(gray, None)
        print("✓ Feature detection works")
        
        # Test 5: Video capture (optional - camera might not be available)
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✓ Camera capture works")
                else:
                    print("⚠️  Camera available but can't read frames")
                cap.release()
            else:
                print("⚠️  No camera available (this is normal)")
        except:
            print("⚠️  Camera test skipped")
        
        print("🎉 All OpenCV functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV functionality test failed: {str(e)}")
        return False

def main():
    """Main function to diagnose and fix OpenCV issues."""
    print("🚀 OpenCV Diagnostic and Fix Tool")
    print("=" * 40)
    
    # Check current status
    opencv_working, version = check_opencv_installation()
    
    if opencv_working:
        print(f"\n✅ OpenCV is working correctly (version {version})")
        
        # Run functionality tests
        if test_opencv_functionality():
            print("\n🎊 OpenCV is fully functional!")
            print("You can now run the training scripts:")
            print("  python setup_training.py")
            print("  python train_model.py")
        else:
            print("\n⚠️  OpenCV imports but some functionality is broken")
            print("Consider reinstalling OpenCV")
        
        return
    
    # Check what's installed
    has_packages = check_pip_packages()
    
    # Attempt to fix
    print(f"\n🔧 OpenCV needs to be fixed...")
    
    if fix_opencv_installation():
        print("\n✅ OpenCV has been fixed!")
        
        # Test functionality
        if test_opencv_functionality():
            print("\n🎊 OpenCV is fully functional!")
        else:
            print("\n⚠️  OpenCV was installed but some functionality may not work")
            print("This might be due to system-specific issues")
            print("Try running: pip install opencv-contrib-python")
    
    else:
        print("\n❌ Unable to fix OpenCV automatically")
        print("\nManual steps to try:")
        print("1. Check Python version compatibility")
        print("2. Update pip: python -m pip install --upgrade pip")
        print("3. Try system package manager:")
        print("   - Ubuntu/Debian: sudo apt-get install python3-opencv")
        print("   - macOS: brew install opencv")
        print("   - Windows: Use Anaconda")
        print("4. Create fresh virtual environment")

if __name__ == "__main__":
    main()

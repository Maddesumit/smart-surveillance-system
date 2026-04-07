#!/usr/bin/env python3
"""
Comprehensive Facial Recognition Test Script

This script provides a complete test suite for the facial recognition system,
including setup, testing, and validation.
"""

import cv2
import sys
import os
import numpy as np
from datetime import datetime
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required imports are available."""
    print("🔍 Testing Imports...")
    
    try:
        import face_recognition
        print("✅ face_recognition library available")
    except ImportError:
        print("❌ face_recognition library not found")
        print("Install with: pip install face_recognition")
        return False
    
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        print("✅ Facial recognition module available")
    except ImportError as e:
        print(f"❌ Cannot import facial recognition module: {e}")
        return False
    
    return True

def setup_demo_faces():
    """Set up demo face images for testing."""
    print("\n📁 Setting up demo face structure...")
    
    # Create known_faces directory structure
    known_faces_dir = "known_faces"
    demo_persons = ["demo_person_1", "demo_person_2", "test_person"]
    
    for person in demo_persons:
        person_dir = os.path.join(known_faces_dir, person)
        os.makedirs(person_dir, exist_ok=True)
        print(f"📂 Created directory: {person_dir}")
    
    print("✅ Demo directory structure created")
    print(f"📝 Add face images to: {known_faces_dir}/[person_name]/")
    return known_faces_dir

def capture_test_face():
    """Capture a face from webcam for testing."""
    print("\n📷 Capture Test Face")
    print("-" * 40)
    
    # Check camera availability
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        print("💡 Try using external USB camera if built-in camera fails")
        return None
    
    name = input("Enter name for test person (or press Enter for 'test_person'): ").strip()
    if not name:
        name = "test_person"
    
    # Create directory
    person_dir = os.path.join("known_faces", name.lower().replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)
    
    print(f"\n📷 Camera ready for {name}")
    print("📋 Instructions:")
    print("  - Look directly at camera")
    print("  - Ensure good lighting")
    print("  - Press SPACE to capture")
    print("  - Press 'q' to quit")
    
    captured_path = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Add instructions on frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw face detection rectangle (optional)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(display_frame, "Face detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Capture Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                captured_path = os.path.join(person_dir, f"{name.lower().replace(' ', '_')}_{timestamp}.jpg")
                cv2.imwrite(captured_path, frame)
                print(f"✅ Image captured: {captured_path}")
                break
            elif key == ord('q'):
                print("❌ Capture cancelled")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return captured_path

def test_face_detection(image_path=None):
    """Test face detection on an image."""
    print("\n🔍 Testing Face Detection...")
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        # Initialize system
        face_system = FacialRecognitionSystem()
        
        if image_path and os.path.exists(image_path):
            # Test with provided image
            print(f"📸 Testing with image: {image_path}")
            image = cv2.imread(image_path)
        else:
            # Test with webcam
            print("📸 Testing with webcam (press SPACE to test, 'q' to quit)")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Cannot access camera")
                return False
            
            image = None
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    cv2.putText(frame, "Press SPACE to test detection, 'q' to quit", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow('Face Detection Test', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        image = frame.copy()
                        break
                    elif key == ord('q'):
                        print("❌ Test cancelled")
                        return False
            finally:
                cap.release()
                cv2.destroyAllWindows()
        
        if image is None:
            print("❌ No image to test")
            return False
        
        # Perform face detection
        print("🔍 Detecting faces...")
        detections = face_system.detect_faces(image)
        
        print(f"✅ Found {len(detections)} face(s)")
        
        if detections:
            for i, detection in enumerate(detections):
                print(f"  Face {i+1}:")
                print(f"    Name: {detection['name']}")
                print(f"    Confidence: {detection['confidence']:.3f}")
                print(f"    Known: {detection['is_known']}")
                print(f"    Bbox: {detection['bbox']}")
        
        # Draw and display results
        result_frame = face_system.draw_face_detections(image, detections)
        cv2.imshow('Detection Results', result_frame)
        print("👀 Displaying results (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return len(detections) > 0
    
    except Exception as e:
        print(f"❌ Error during face detection: {e}")
        return False

def test_known_person_addition():
    """Test adding a known person to the database."""
    print("\n➕ Testing Known Person Addition...")
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        # Initialize system
        face_system = FacialRecognitionSystem()
        
        # Check for existing images
        known_faces_dir = "known_faces"
        if os.path.exists(known_faces_dir):
            for person_dir in os.listdir(known_faces_dir):
                person_path = os.path.join(known_faces_dir, person_dir)
                if os.path.isdir(person_path):
                    for img_file in os.listdir(person_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(person_path, img_file)
                            print(f"📸 Found image: {img_path}")
                            
                            # Add person
                            success = face_system.add_known_person(person_dir, img_path)
                            if success:
                                print(f"✅ Added {person_dir} to database")
                            else:
                                print(f"❌ Failed to add {person_dir}")
        
        # Check known persons
        known_persons = face_system.get_known_persons()
        print(f"👥 Total known persons: {len(known_persons)}")
        for person in known_persons:
            print(f"  - {person}")
        
        return len(known_persons) > 0
    
    except Exception as e:
        print(f"❌ Error adding known persons: {e}")
        return False

def test_recognition_performance():
    """Test recognition performance and statistics."""
    print("\n📊 Testing Recognition Performance...")
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        face_system = FacialRecognitionSystem()
        stats = face_system.get_recognition_stats()
        
        print("📈 Current Statistics:")
        print(f"  Total faces detected: {stats['total_faces_detected']}")
        print(f"  Known faces identified: {stats['known_faces_identified']}")
        print(f"  Unknown faces detected: {stats['unknown_faces_detected']}")
        print(f"  Average processing time: {stats['processing_time_avg']:.4f}s")
        
        return True
    
    except Exception as e:
        print(f"❌ Error checking performance: {e}")
        return False

def test_database_functionality():
    """Test database operations."""
    print("\n💾 Testing Database Functionality...")
    
    try:
        import sqlite3
        
        db_path = "face_database.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📊 Database tables: {[table[0] for table in tables]}")
            
            # Check known faces
            cursor.execute("SELECT COUNT(*) FROM known_faces")
            known_count = cursor.fetchone()[0]
            print(f"👥 Known faces in database: {known_count}")
            
            # Check detections
            cursor.execute("SELECT COUNT(*) FROM face_detections")
            detection_count = cursor.fetchone()[0]
            print(f"🔍 Total detections recorded: {detection_count}")
            
            conn.close()
            return True
        else:
            print("📝 No database found (will be created on first use)")
            return True
    
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 Comprehensive Facial Recognition Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Database Test", test_database_functionality),
        ("Known Person Addition", test_known_person_addition),
        ("Performance Test", test_recognition_performance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Summary")
    print("-" * 30)
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Facial recognition is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

def interactive_menu():
    """Interactive menu for testing."""
    while True:
        print("\n🎯 Facial Recognition Test Menu")
        print("-" * 40)
        print("1. Run comprehensive test")
        print("2. Setup demo face structure")
        print("3. Capture test face from webcam")
        print("4. Test face detection")
        print("5. Test known person addition")
        print("6. Test performance statistics")
        print("7. Test database")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            run_comprehensive_test()
        elif choice == "2":
            setup_demo_faces()
        elif choice == "3":
            captured_path = capture_test_face()
            if captured_path:
                print(f"✅ Face captured: {captured_path}")
        elif choice == "4":
            test_face_detection()
        elif choice == "5":
            test_known_person_addition()
        elif choice == "6":
            test_recognition_performance()
        elif choice == "7":
            test_database_functionality()
        elif choice == "8":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please select 1-8.")

def main():
    """Main function."""
    print("🎯 Facial Recognition Test Script")
    print("=================================")
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "test":
            run_comprehensive_test()
        elif arg == "setup":
            setup_demo_faces()
        elif arg == "capture":
            capture_test_face()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python test_facial_recognition.py [test|setup|capture]")
    else:
        interactive_menu()

if __name__ == "__main__":
    main()

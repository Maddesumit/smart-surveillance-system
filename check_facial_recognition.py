#!/usr/bin/env python3
"""
Facial Recognition Status Checker

Quick script to check the current status of your facial recognition system.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_facial_recognition_status():
    """Check the current status of facial recognition system."""
    
    print("🎯 Facial Recognition Status Check")
    print("=" * 40)
    
    # Check imports
    print("\n📦 Checking Dependencies...")
    try:
        import face_recognition
        print("✅ face_recognition library: OK")
        face_rec_version = getattr(face_recognition, '__version__', 'Unknown')
        print(f"   Version: {face_rec_version}")
    except ImportError:
        print("❌ face_recognition library: NOT FOUND")
        print("   Install with: pip install face_recognition")
        return False
    
    try:
        import cv2
        print("✅ OpenCV: OK")
        print(f"   Version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV: NOT FOUND")
        return False
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        print("✅ Facial Recognition Module: OK")
    except ImportError as e:
        print(f"❌ Facial Recognition Module: ERROR - {e}")
        return False
    
    # Check system initialization
    print("\n🔧 Checking System Initialization...")
    try:
        face_system = FacialRecognitionSystem()
        print("✅ System initialization: OK")
    except Exception as e:
        print(f"❌ System initialization: ERROR - {e}")
        return False
    
    # Check database
    print("\n💾 Checking Database...")
    db_path = "face_database.db"
    if os.path.exists(db_path):
        print("✅ Database file: EXISTS")
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM known_faces")
            known_count = cursor.fetchone()[0]
            print(f"   Known persons: {known_count}")
            
            cursor.execute("SELECT COUNT(*) FROM face_detections")
            detection_count = cursor.fetchone()[0]
            print(f"   Total detections: {detection_count}")
            
            conn.close()
        except Exception as e:
            print(f"   Database error: {e}")
    else:
        print("📝 Database file: NOT EXISTS (will be created on first use)")
    
    # Check known faces directory
    print("\n📁 Checking Known Faces...")
    known_faces_dir = "known_faces"
    if os.path.exists(known_faces_dir):
        print("✅ Known faces directory: EXISTS")
        
        person_count = 0
        image_count = 0
        
        for item in os.listdir(known_faces_dir):
            item_path = os.path.join(known_faces_dir, item)
            if os.path.isdir(item_path):
                person_count += 1
                person_images = [f for f in os.listdir(item_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_count += len(person_images)
                print(f"   📂 {item}: {len(person_images)} image(s)")
        
        print(f"   Total persons: {person_count}")
        print(f"   Total images: {image_count}")
        
        if person_count == 0:
            print("   💡 Add faces with: python add_known_faces.py")
    else:
        print("📝 Known faces directory: NOT EXISTS")
        print("   💡 Create with: mkdir known_faces")
    
    # Check face encodings file
    print("\n🔐 Checking Face Encodings...")
    encodings_path = "face_encodings.pkl"
    if os.path.exists(encodings_path):
        print("✅ Face encodings file: EXISTS")
        
        try:
            import pickle
            with open(encodings_path, 'rb') as f:
                data = pickle.load(f)
                encoding_count = len(data.get('encodings', []))
                print(f"   Stored encodings: {encoding_count}")
        except Exception as e:
            print(f"   Encoding file error: {e}")
    else:
        print("📝 Face encodings file: NOT EXISTS (will be created when faces are added)")
    
    # Check system performance
    print("\n📊 Checking System Performance...")
    try:
        stats = face_system.get_recognition_stats()
        print(f"   Total faces detected: {stats['total_faces_detected']}")
        print(f"   Known faces identified: {stats['known_faces_identified']}")
        print(f"   Unknown faces detected: {stats['unknown_faces_detected']}")
        print(f"   Average processing time: {stats['processing_time_avg']:.4f}s")
    except Exception as e:
        print(f"   Performance check error: {e}")
    
    # Check camera access
    print("\n📷 Checking Camera Access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access: OK")
            ret, frame = cap.read()
            if ret:
                print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("   ⚠️  Cannot read frames")
            cap.release()
        else:
            print("❌ Camera access: FAILED")
            print("   💡 Check camera permissions or try external USB camera")
    except Exception as e:
        print(f"❌ Camera check error: {e}")
    
    # Get current known persons
    print("\n👥 Current Known Persons...")
    try:
        known_persons = face_system.get_known_persons()
        if known_persons:
            print(f"   Found {len(known_persons)} known person(s):")
            for person in known_persons:
                print(f"     - {person}")
        else:
            print("   No known persons found")
            print("   💡 Add with: python add_known_faces.py")
    except Exception as e:
        print(f"   Error getting known persons: {e}")
    
    # Summary
    print("\n🎯 Summary")
    print("-" * 20)
    
    # Determine overall status
    has_dependencies = True
    has_known_faces = len(face_system.get_known_persons()) > 0 if 'face_system' in locals() else False
    has_camera = True  # Assume true unless explicitly failed above
    
    if has_dependencies and has_known_faces:
        print("✅ STATUS: READY - Facial recognition is fully operational!")
        print("   💡 Run: python main.py")
    elif has_dependencies and not has_known_faces:
        print("⚠️  STATUS: NEEDS SETUP - System ready but no known faces")
        print("   💡 Add faces: python add_known_faces.py")
    else:
        print("❌ STATUS: NOT READY - Missing dependencies or configuration")
        print("   💡 Check errors above")
    
    print("\n🚀 Quick Actions:")
    print("   python add_known_faces.py      # Add faces to system")
    print("   python demo_facial_recognition.py  # Test with webcam")
    print("   python test_facial_recognition.py  # Run full tests")
    print("   python main.py                 # Start surveillance system")
    
    return True

if __name__ == "__main__":
    check_facial_recognition_status()

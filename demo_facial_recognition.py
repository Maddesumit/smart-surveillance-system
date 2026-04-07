#!/usr/bin/env python3
"""
Quick Facial Recognition Demo

A simple script to quickly test facial recognition capabilities.
"""

import cv2
import sys
import os
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_demo():
    """Quick demonstration of facial recognition."""
    print("🎯 Quick Facial Recognition Demo")
    print("================================")
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        # Initialize facial recognition
        print("🔧 Initializing facial recognition system...")
        face_system = FacialRecognitionSystem()
        
        # Check for known faces
        known_persons = face_system.get_known_persons()
        print(f"👥 Known persons in database: {len(known_persons)}")
        
        if known_persons:
            print("📋 Known persons:")
            for person in known_persons:
                print(f"  - {person}")
        else:
            print("📝 No known persons found. You can add them using:")
            print("   python setup_facial_recognition.py")
        
        # Start webcam
        print("\n📷 Starting webcam demo...")
        print("📋 Controls:")
        print("  - 's' to save detection screenshot")
        print("  - 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access webcam")
            return
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces
                detections = face_system.detect_faces(frame)
                
                # Draw detections
                result_frame = face_system.draw_face_detections(frame, detections)
                
                # Add status text
                status_text = f"Faces: {len(detections)} | Known: {len([d for d in detections if d['is_known']])}"
                cv2.putText(result_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(result_frame, "Press 's' to screenshot, 'q' to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display
                cv2.imshow('Facial Recognition Demo', result_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"demo_screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, result_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
                
                # Print detection info (optional, for debugging)
                if detections:
                    for detection in detections:
                        name = detection['name']
                        confidence = detection['confidence']
                        is_known = detection['is_known']
                        if is_known:
                            print(f"👤 Recognized: {name} (confidence: {confidence:.3f})")
                        else:
                            print(f"❓ Unknown person detected")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Show statistics
        stats = face_system.get_recognition_stats()
        print("\n📊 Session Statistics:")
        print(f"  Total faces detected: {stats['total_faces_detected']}")
        print(f"  Known faces identified: {stats['known_faces_identified']}")
        print(f"  Unknown faces detected: {stats['unknown_faces_detected']}")
        print(f"  Screenshots saved: {screenshot_count}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure face_recognition is installed: pip install face_recognition")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_with_image():
    """Test facial recognition with a static image."""
    print("🖼️  Static Image Test")
    print("===================")
    
    # Ask for image path
    image_path = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if not image_path or not os.path.exists(image_path):
        print("❌ Invalid or missing image path")
        return
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        # Initialize system
        face_system = FacialRecognitionSystem()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Cannot load image")
            return
        
        print(f"📸 Testing with: {image_path}")
        
        # Detect faces
        detections = face_system.detect_faces(image)
        
        print(f"🔍 Found {len(detections)} face(s)")
        
        # Show results
        if detections:
            for i, detection in enumerate(detections):
                print(f"  Face {i+1}: {detection['name']} (confidence: {detection['confidence']:.3f})")
        
        # Draw and display
        result_frame = face_system.draw_face_detections(image, detections)
        cv2.imshow('Static Image Test', result_frame)
        print("👀 Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🚀 Choose Demo Mode:")
    print("1. Live webcam demo")
    print("2. Static image test")
    print("3. Both")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        quick_demo()
    elif choice == "2":
        test_with_image()
    elif choice == "3":
        test_with_image()
        input("\nPress Enter to start webcam demo...")
        quick_demo()
    else:
        print("❌ Invalid choice. Running webcam demo...")
        quick_demo()

if __name__ == "__main__":
    main()

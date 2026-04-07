#!/usr/bin/env python3
"""
Facial Recognition Setup and Test Script

This script helps you set up and test the facial recognition system.
"""

import cv2
import sys
import os
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_features.facial_recognition import FacialRecognitionSystem

def capture_and_add_person():
    """Capture a photo from webcam and add person to database."""
    print("🎯 Capture and Add Known Person")
    print("-" * 40)
    
    name = input("Enter person's name: ").strip()
    if not name:
        print("❌ Name cannot be empty")
        return
    
    # Create known_faces directory if it doesn't exist
    known_faces_dir = "known_faces"
    person_dir = os.path.join(known_faces_dir, name.lower().replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print(f"\n📷 Camera started for {name}")
    print("Position your face in the frame and press SPACE to capture")
    print("Press 'q' to quit without capturing")
    
    captured = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Capture Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(person_dir, f"{name.lower().replace(' ', '_')}_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"✅ Image captured: {image_path}")
                
                # Add to facial recognition system
                face_system = FacialRecognitionSystem()
                success = face_system.add_known_person(name, image_path)
                
                if success:
                    print(f"✅ {name} added to facial recognition database!")
                    captured = True
                else:
                    print(f"❌ Failed to add {name} to database")
                
                break
            elif key == ord('q'):
                print("❌ Capture cancelled")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return captured

def test_facial_recognition():
    """Test facial recognition with live video."""
    print("\n🔍 Testing Facial Recognition")
    print("-" * 40)
    
    face_system = FacialRecognitionSystem()
    
    # Get current known persons
    known_persons = face_system.get_known_persons()
    print(f"Known persons in database: {len(known_persons)}")
    for person in known_persons:
        print(f"  - {person}")
    
    if not known_persons:
        print("⚠️ No known persons in database. Add some first!")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n📹 Starting facial recognition test...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance
            if frame_count % 3 == 0:
                # Detect faces
                detections = face_system.detect_faces(frame)
                
                # Print detection results
                if detections:
                    print(f"\nFrame {frame_count}: {len(detections)} faces detected")
                    for i, detection in enumerate(detections):
                        name = detection['name']
                        confidence = detection['confidence']
                        is_known = detection['is_known']
                        bbox = detection['bbox']
                        print(f"  Face {i+1}: {name} (Known: {is_known}, Confidence: {confidence:.3f})")
                
                # Draw results on frame
                result_frame = face_system.draw_face_detections(frame, detections)
            else:
                result_frame = frame
            
            # Add info overlay
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, "Press 'q' to quit, 's' for screenshot", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Facial Recognition Test', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"face_recognition_test_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, result_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_system.cleanup()

def view_statistics():
    """View facial recognition statistics."""
    print("\n📊 Facial Recognition Statistics")
    print("-" * 40)
    
    face_system = FacialRecognitionSystem()
    
    # Get statistics
    stats = face_system.get_recognition_stats()
    print(f"Total faces detected: {stats['total_faces_detected']}")
    print(f"Known faces identified: {stats['known_faces_identified']}")
    print(f"Unknown faces detected: {stats['unknown_faces_detected']}")
    print(f"Average processing time: {stats['processing_time_avg']:.3f}s")
    
    # Get known persons
    known_persons = face_system.get_known_persons()
    print(f"\nKnown persons ({len(known_persons)}):")
    for person in known_persons:
        print(f"  - {person}")

def remove_person():
    """Remove a person from the database."""
    print("\n🗑️ Remove Person from Database")
    print("-" * 40)
    
    face_system = FacialRecognitionSystem()
    known_persons = face_system.get_known_persons()
    
    if not known_persons:
        print("❌ No known persons in database")
        return
    
    print("Known persons:")
    for i, person in enumerate(known_persons, 1):
        print(f"  {i}. {person}")
    
    try:
        choice = input(f"\nEnter number (1-{len(known_persons)}) or name to remove: ").strip()
        
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(known_persons):
                person_to_remove = known_persons[index]
            else:
                print("❌ Invalid choice")
                return
        else:
            person_to_remove = choice
        
        success = face_system.remove_known_person(person_to_remove)
        if success:
            print(f"✅ {person_to_remove} removed from database")
        else:
            print(f"❌ Failed to remove {person_to_remove}")
    
    except (ValueError, KeyboardInterrupt):
        print("❌ Operation cancelled")

def main_menu():
    """Display main menu and handle user choices."""
    while True:
        print("\n" + "="*50)
        print("🎯 FACIAL RECOGNITION SETUP & TEST")
        print("="*50)
        print("1. Capture and add new person")
        print("2. Test facial recognition")
        print("3. View statistics")
        print("4. Remove person from database")
        print("5. Exit")
        print("-" * 50)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            capture_and_add_person()
        elif choice == '2':
            test_facial_recognition()
        elif choice == '3':
            view_statistics()
        elif choice == '4':
            remove_person()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🎯 Facial Recognition Setup and Test")
    print("Make sure your camera is connected and working.")
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your camera and try again.")

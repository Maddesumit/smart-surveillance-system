#!/usr/bin/env python3
"""
Simple Face Addition Script

A straightforward script to add known faces to the surveillance system.
"""

import cv2
import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def add_face_from_webcam():
    """Capture and add a face from webcam."""
    print("📷 Add Face from Webcam")
    print("======================")
    
    # Get person's name
    name = input("Enter person's name: ").strip()
    if not name:
        print("❌ Name is required")
        return False
    
    # Create directory for this person
    person_dir = os.path.join("known_faces", name.lower().replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)
    
    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return False
    
    print(f"\n📸 Camera ready for {name}")
    print("Instructions:")
    print("- Look directly at the camera")
    print("- Ensure good lighting")
    print("- Press SPACE to capture")
    print("- Press 'q' to cancel")
    
    captured = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create display frame
            display_frame = frame.copy()
            
            # Add text overlay
            cv2.putText(display_frame, f"Adding: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE to capture, 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Optional: Show face detection
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(display_frame, "Face detected", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except:
                pass
            
            cv2.imshow('Add Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name.lower().replace(' ', '_')}_{timestamp}.jpg"
                image_path = os.path.join(person_dir, filename)
                
                cv2.imwrite(image_path, frame)
                print(f"✅ Image saved: {image_path}")
                
                # Add to facial recognition system
                try:
                    from advanced_features.facial_recognition import FacialRecognitionSystem
                    face_system = FacialRecognitionSystem()
                    success = face_system.add_known_person(name, image_path)
                    
                    if success:
                        print(f"✅ {name} added to facial recognition database!")
                        captured = True
                    else:
                        print(f"❌ Failed to add {name} to database")
                        print("   - Make sure your face is clearly visible")
                        print("   - Try again with better lighting")
                
                except Exception as e:
                    print(f"❌ Error adding to database: {e}")
                
                break
                
            elif key == ord('q'):  # Quit
                print("❌ Cancelled")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return captured

def add_face_from_file():
    """Add a face from an existing image file."""
    print("📁 Add Face from File")
    print("====================")
    
    name = input("Enter person's name: ").strip()
    if not name:
        print("❌ Name is required")
        return False
    
    image_path = input("Enter path to image file: ").strip()
    if not image_path or not os.path.exists(image_path):
        print("❌ Invalid image path")
        return False
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        face_system = FacialRecognitionSystem()
        success = face_system.add_known_person(name, image_path)
        
        if success:
            print(f"✅ {name} added to facial recognition database!")
            return True
        else:
            print(f"❌ Failed to add {name}")
            print("   - Check image quality")
            print("   - Ensure face is clearly visible")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def list_known_faces():
    """List all known faces in the database."""
    print("👥 Known Faces")
    print("=============")
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        face_system = FacialRecognitionSystem()
        known_persons = face_system.get_known_persons()
        
        if known_persons:
            print(f"Found {len(known_persons)} known person(s):")
            for i, person in enumerate(known_persons, 1):
                print(f"  {i}. {person}")
        else:
            print("No known persons in database")
        
        return known_persons
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def remove_known_face():
    """Remove a known face from the database."""
    print("🗑️  Remove Known Face")
    print("====================")
    
    # List current faces
    known_persons = list_known_faces()
    if not known_persons:
        return
    
    print("\nEnter the name to remove:")
    name = input("Name: ").strip()
    
    if name not in known_persons:
        print(f"❌ '{name}' not found in database")
        return
    
    confirm = input(f"Are you sure you want to remove '{name}'? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled")
        return
    
    try:
        from advanced_features.facial_recognition import FacialRecognitionSystem
        
        face_system = FacialRecognitionSystem()
        success = face_system.remove_known_person(name)
        
        if success:
            print(f"✅ {name} removed from database")
        else:
            print(f"❌ Failed to remove {name}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main menu."""
    while True:
        print("\n🎯 Facial Recognition Setup")
        print("============================")
        print("1. Add face from webcam")
        print("2. Add face from image file")
        print("3. List known faces")
        print("4. Remove known face")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            add_face_from_webcam()
        elif choice == "2":
            add_face_from_file()
        elif choice == "3":
            list_known_faces()
        elif choice == "4":
            remove_known_face()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()

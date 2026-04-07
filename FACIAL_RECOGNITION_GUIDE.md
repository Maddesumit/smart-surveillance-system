# 🎯 Facial Recognition Guide - Smart Surveillance System

## Overview
Your Smart Surveillance System includes a powerful facial recognition system that can:
- Detect faces in real-time video
- Recognize known persons from a database
- Alert when unknown faces are detected
- Track face detection statistics
- Manage a database of known faces

## 🚀 Quick Start

### 1. **Run the System with Facial Recognition**
```bash
# Your main.py now includes facial recognition
python main.py
```

### 2. **Run the Enhanced System (Recommended)**
```bash
# Full featured system with visual display
python main_enhanced.py
```

## 📋 Step-by-Step Facial Recognition Setup

### **Step 1: Add Known Persons to the Database**

Create a script to add known persons:

```python
#!/usr/bin/env python3
"""
Add Known Persons to Facial Recognition Database
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_features.facial_recognition import FacialRecognitionSystem

def add_known_person():
    # Initialize facial recognition system
    face_system = FacialRecognitionSystem()
    
    # Add a known person (you need to provide actual image paths)
    success = face_system.add_known_person(
        name="John Doe", 
        image_path="path/to/john_doe.jpg"
    )
    
    if success:
        print("✅ Person added successfully!")
    else:
        print("❌ Failed to add person")
    
    # Get statistics
    stats = face_system.get_recognition_stats()
    print(f"Recognition stats: {stats}")
    
    # Get list of known persons
    known_persons = face_system.get_known_persons()
    print(f"Known persons: {known_persons}")

if __name__ == "__main__":
    add_known_person()
```

### **Step 2: Organize Known Face Images**

Create a directory structure for known faces:

```
smart_surveillance/
├── known_faces/
│   ├── john_doe/
│   │   ├── john_01.jpg
│   │   ├── john_02.jpg
│   │   └── john_03.jpg
│   ├── jane_smith/
│   │   ├── jane_01.jpg
│   │   └── jane_02.jpg
│   └── admin/
│       └── admin_01.jpg
```

### **Step 3: Test Facial Recognition**

Create a test script:

```python
#!/usr/bin/env python3
"""
Test Facial Recognition System
"""
import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_features.facial_recognition import FacialRecognitionSystem

def test_facial_recognition():
    # Initialize system
    face_system = FacialRecognitionSystem()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            detections = face_system.detect_faces(frame)
            
            # Draw face detections
            result_frame = face_system.draw_face_detections(frame, detections)
            
            # Display results
            cv2.imshow('Facial Recognition Test', result_frame)
            
            # Print detection info
            if detections:
                for detection in detections:
                    name = detection['name']
                    confidence = detection['confidence']
                    is_known = detection['is_known']
                    print(f"Face: {name}, Confidence: {confidence:.3f}, Known: {is_known}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('face_detection_screenshot.jpg', result_frame)
                print("Screenshot saved!")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_system.cleanup()

if __name__ == "__main__":
    test_facial_recognition()
```

## 🔧 Configuration Options

### **Facial Recognition Settings**

You can customize the facial recognition system:

```python
# Initialize with custom settings
face_system = FacialRecognitionSystem(
    database_path="custom_face_database.db",
    encodings_path="custom_face_encodings.pkl",
    known_faces_dir="custom_known_faces",
    tolerance=0.6,  # Lower = stricter matching (0.4-0.8)
    model="hog"     # "hog" for CPU, "cnn" for GPU
)
```

### **Performance Settings**

```python
# For better performance
model="hog"        # Faster, CPU-based
tolerance=0.6      # Balanced accuracy/speed

# For better accuracy
model="cnn"        # More accurate, requires GPU
tolerance=0.4      # Stricter matching
```

## 📊 Facial Recognition Features

### **1. Real-time Face Detection**
- Detects multiple faces in each frame
- Works with various lighting conditions
- Handles different face angles and sizes

### **2. Face Recognition**
- Matches detected faces against known database
- Provides confidence scores for matches
- Handles partial face occlusions

### **3. Database Management**
- SQLite database for face metadata
- Pickle files for face encodings
- Automatic backup and recovery

### **4. Statistics and Reporting**
```python
# Get recognition statistics
stats = face_system.get_recognition_stats()
print(f"Total faces detected: {stats['total_faces_detected']}")
print(f"Known faces identified: {stats['known_faces_identified']}")
print(f"Unknown faces detected: {stats['unknown_faces_detected']}")
```

## 🚨 Alert Integration

The facial recognition system integrates with your alert system:

### **Unknown Face Alerts**
- Automatically triggered when unknown faces are detected
- Includes face image and location information
- Configurable alert thresholds

### **VIP Detection Alerts**
- Special alerts for important known persons
- Customizable notification rules
- Priority-based alert routing

## 🎯 Advanced Usage

### **1. Batch Add Known Persons**

```python
def batch_add_known_faces():
    face_system = FacialRecognitionSystem()
    
    # Define persons to add
    persons = [
        ("John Doe", "images/john_doe.jpg"),
        ("Jane Smith", "images/jane_smith.jpg"),
        ("Admin User", "images/admin.jpg")
    ]
    
    for name, image_path in persons:
        success = face_system.add_known_person(name, image_path)
        print(f"Added {name}: {'✅' if success else '❌'}")
```

### **2. Search for Specific Persons**

```python
def search_person_history():
    face_system = FacialRecognitionSystem()
    
    # Search for when a specific person was last seen
    # (This would require additional database queries)
    known_persons = face_system.get_known_persons()
    print(f"Known persons in database: {known_persons}")
```

### **3. Face Quality Assessment**

The system automatically assesses face quality:
- Checks image sharpness
- Evaluates lighting conditions
- Filters low-quality detections

## 🛠 Troubleshooting

### **Common Issues:**

1. **"No faces detected"**
   - Ensure good lighting
   - Check camera quality
   - Verify face size in frame

2. **"Poor recognition accuracy"**
   - Lower tolerance value (0.4-0.5)
   - Add more training images per person
   - Ensure high-quality reference images

3. **"Slow performance"**
   - Use "hog" model for CPU
   - Reduce frame processing rate
   - Consider hardware upgrades

### **Performance Optimization:**

```python
# Optimize for speed
face_system = FacialRecognitionSystem(
    model="hog",           # CPU-optimized
    tolerance=0.6          # Balanced accuracy
)

# Process every 3rd frame for better performance
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if frame_count % 3 == 0:  # Process every 3rd frame
        detections = face_system.detect_faces(frame)
```

## 📈 Integration with Main System

Your main.py now includes facial recognition and will:

1. **Automatically detect faces** in the video stream
2. **Log recognition results** to the system logs
3. **Generate alerts** for unknown faces
4. **Track statistics** for known/unknown faces
5. **Integrate with the web dashboard**

## 🔍 Monitoring and Logs

Check the logs for facial recognition activity:

```bash
# View recent logs
tail -f logs/surveillance_*.log | grep -i face

# Search for specific face detections
grep "known faces" logs/surveillance_*.log
grep "unknown faces" logs/surveillance_*.log
```

## 🎉 You're Ready!

Your facial recognition system is now integrated and ready to use! The system will:

✅ **Automatically detect and recognize faces**  
✅ **Alert on unknown face detection**  
✅ **Log all recognition activities**  
✅ **Maintain a database of known persons**  
✅ **Provide real-time statistics**  

Start with the basic setup and gradually add more known persons to improve the system's effectiveness!

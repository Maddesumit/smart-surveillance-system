# 🚀 Quick Facial Recognition Setup - 5 Minutes!

## Step 1: Add Your Face to the System
```bash
# Run this script to add faces via webcam
python add_known_faces.py
```

**OR use the enhanced setup script:**
```bash
python setup_facial_recognition.py
```

## Step 2: Test the System
```bash
# Quick demo with webcam
python demo_facial_recognition.py

# Full test suite
python test_facial_recognition.py
```

## Step 3: Run Your Surveillance System
```bash
# Basic system with facial recognition
python main.py

# Enhanced system with visual display
python main_enhanced.py
```

## 🎯 What Each Script Does

### `add_known_faces.py` 
Simple interface to:
- Add faces from webcam
- Add faces from image files
- List/remove known faces

### `demo_facial_recognition.py`
Quick test:
- Live webcam demonstration
- Shows recognition in real-time
- Save screenshots with 's' key

### `test_facial_recognition.py`
Comprehensive testing:
- Test all components
- Check database
- Performance statistics
- Interactive menu

### `setup_facial_recognition.py`
Full setup wizard:
- Guided face capture
- System testing
- Configuration help

## 📋 Quick Tips

1. **Good lighting** is crucial for face recognition
2. **Multiple angles** per person improve accuracy
3. **Clear, front-facing photos** work best
4. **2-3 images per person** recommended

## 🔧 Troubleshooting

**Camera not working?**
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

**Face not detected?**
- Ensure good lighting
- Face directly towards camera
- Remove glasses/hat if needed
- Try different distance from camera

**Poor recognition?**
- Add more photos of the same person
- Use different lighting conditions
- Adjust tolerance in settings

---
**Your facial recognition system is ready! Start with `add_known_faces.py` to add some faces.** 🎉

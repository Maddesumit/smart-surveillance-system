# 🎯 Model Training and Accuracy Improvement Guide

This guide explains how to train custom YOLO models to improve detection accuracy for your specific surveillance scenarios.

## 🚀 Quick Start

### 1. **Install Additional Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Verify Installation**
```bash
# Check if all dependencies are correctly installed
python check_dependencies.py

# If OpenCV issues, run the fix script
python fix_opencv.py
```

### 3. **Train a Custom Model (Automated)**
```bash
# Train using webcam data collection (10 minutes)
python train_model.py --project-name my_surveillance --duration 10

# Train using existing video
python train_model.py --project-name my_surveillance --collection-method video --video-source path/to/video.mp4

# Train using existing dataset
python train_model.py --project-name my_surveillance --data-dir path/to/dataset
```

### 4. **Test Model Accuracy**
```bash
# Compare models using webcam
python test_accuracy.py --test-method webcam --duration 30

# Test with custom model
python test_accuracy.py --custom-model models/custom/my_model.pt
```

## 📁 Training Pipeline Components

### **CustomModelTrainer** (`src/model_training/trainer.py`)
- **Purpose**: Train custom YOLO models optimized for surveillance
- **Features**:
  - Surveillance-specific data augmentation
  - Automatic model optimization
  - Best model selection and saving
  - Performance tracking

### **DataCollector** (`src/model_training/data_collector.py`)
- **Purpose**: Collect and manage training data
- **Features**:
  - Automated frame collection from webcam/video
  - Quality filtering and deduplication
  - Metadata tracking with SQLite database
  - Auto-annotation using pre-trained models

### **ModelEvaluator** (`src/model_training/evaluator.py`)
- **Purpose**: Comprehensive model evaluation and comparison
- **Features**:
  - Detailed accuracy metrics (mAP, precision, recall, F1)
  - Performance visualization
  - Model comparison reports
  - Per-class analysis

### **EnhancedObjectDetector** (`src/object_detection/enhanced_detector.py`)
- **Purpose**: Improved detector with surveillance optimizations
- **Features**:
  - Surveillance-specific preprocessing
  - Smart filtering for better accuracy
  - Custom model support
  - Performance statistics

## 🛠️ Training Methods

### **Method 1: Automatic Data Collection + Training**
```python
from model_training.trainer import CustomModelTrainer
from model_training.data_collector import DataCollector

# Collect training data
collector = DataCollector()
frames_collected = collector.start_collection_session(
    session_name="my_training_data",
    duration_minutes=15,
    collection_interval=2.0
)

# Auto-annotate collected data
trainer = CustomModelTrainer()
annotation_paths = trainer.auto_annotate_with_pretrained(
    image_paths=collected_images,
    confidence_threshold=0.6
)

# Train custom model
dataset_config = trainer.prepare_dataset(...)
model_path = trainer.train_model(dataset_config)
```

### **Method 2: Using Existing Dataset**
```python
# Prepare your dataset in YOLO format:
# dataset/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── labels/
#       ├── train/
#       └── val/

trainer = CustomModelTrainer()
dataset_config = trainer.prepare_dataset(
    train_images=train_image_paths,
    train_labels=train_label_paths,
    val_images=val_image_paths,
    val_labels=val_label_paths
)

model_path = trainer.train_model(dataset_config, epochs=100)
```

### **Method 3: Video-based Training**
```python
collector = DataCollector()

# Extract frames from surveillance videos
frames = collector.collect_frames_from_video(
    video_path="surveillance_footage.mp4",
    frame_interval=30,  # Every 30th frame
    max_frames=1000
)

# Auto-annotate and train
# ... (same as Method 1)
```

## ⚙️ Configuration Options

### **Training Configuration** (`config/model_config.py`)
```python
TRAINING_CONFIG = {
    'base_model': 'yolov8s.pt',  # Base model size
    'epochs': 100,               # Training epochs
    'batch_size': 16,           # Batch size
    'img_size': 640,            # Input image size
    'learning_rate': 0.01,      # Learning rate
    'confidence_threshold': 0.25, # Detection threshold
    'surveillance_mode': True,   # Enable surveillance optimizations
}
```

### **Data Collection Configuration**
```python
DATA_COLLECTION_CONFIG = {
    'collection_interval': 2.0,     # Seconds between frames
    'quality_threshold': 0.5,       # Minimum quality score
    'auto_annotation_confidence': 0.6, # Auto-annotation threshold
    'validation_split': 0.2,        # Train/val split ratio
}
```

## 📊 Model Evaluation

### **Automatic Evaluation**
```python
from model_training.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(
    model_path="models/custom/my_model.pt",
    test_dataset="datasets/my_data/data.yaml"
)

print(f"mAP@0.5: {metrics['overall_metrics']['mAP50']:.3f}")
print(f"Precision: {metrics['overall_metrics']['precision']:.3f}")
print(f"Recall: {metrics['overall_metrics']['recall']:.3f}")
```

### **Model Comparison**
```python
# Compare multiple models
model_results = [
    evaluator.evaluate_model("yolov8s.pt", test_dataset),
    evaluator.evaluate_model("models/custom/my_model.pt", test_dataset)
]

comparison = evaluator.compare_models(
    model_results, 
    comparison_name="pretrained_vs_custom"
)
```

## 🎯 Surveillance-Specific Optimizations

### **Enhanced Detection Features**
- **Surveillance Classes Priority**: Focuses on security-relevant objects
- **Spatial Analysis**: Tracks object zones and movement patterns
- **Quality Filtering**: Removes low-quality detections automatically
- **Edge Detection Handling**: Better handling of partial objects at frame edges

### **Custom Preprocessing**
- **Contrast Enhancement**: CLAHE for better small object detection
- **Noise Reduction**: Bilateral filtering to reduce false positives
- **Multi-scale Detection**: Optimized for various object sizes

### **Performance Optimizations**
- **Test-time Augmentation**: Improves accuracy at inference
- **Class-aware NMS**: Better handling of overlapping objects
- **Confidence Boosting**: Prioritizes surveillance-relevant objects

## 📈 Improving Model Accuracy

### **1. Data Quality**
- **Diverse Scenarios**: Collect data from various lighting conditions, angles, and times
- **High-Quality Frames**: Use quality threshold filtering (>0.5)
- **Sufficient Data**: Aim for 500+ annotated images per class

### **2. Training Parameters**
```python
# For better accuracy, try these settings:
config = {
    'epochs': 150,              # More training
    'batch_size': 8,           # Smaller batches for stability
    'learning_rate': 0.005,    # Lower learning rate
    'patience': 75,            # More patience for convergence
    'augmentation': True,      # Enable data augmentation
}
```

### **3. Model Selection**
- **YOLOv8n**: Fastest, lowest accuracy
- **YOLOv8s**: Good balance of speed/accuracy
- **YOLOv8m**: Higher accuracy, slower
- **YOLOv8l/x**: Best accuracy, much slower

### **4. Surveillance-Specific Tips**
- **Focus on Important Classes**: Train primarily on security-relevant objects
- **Use High-Resolution Input**: 640x640 or higher for small object detection
- **Multiple Angles**: Include various camera angles and perspectives
- **Time-based Data**: Collect data during different times of day

## 🔧 Troubleshooting

### **Installation Issues**

**1. OpenCV Import Error (opencv-python installed but not detected)**
```bash
# Issue: Package installed but import fails
# Solution: Check the actual installation
python -c "import cv2; print(cv2.__version__)"

# If that fails, reinstall opencv:
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python>=4.5.0

# For some systems, you might need:
pip install opencv-python-headless>=4.5.0
```

**2. Dependencies Check Script**
```bash
# Run dependency checker for detailed diagnosis
python check_dependencies.py

# This will show exactly which packages are missing and test functionality
```

**3. Virtual Environment Issues**
```bash
# Create fresh virtual environment
python -m venv surveillance_env
source surveillance_env/bin/activate  # On Windows: surveillance_env\Scripts\activate
pip install -r requirements.txt
```

### **Common Training Issues & Solutions**

**1. Low mAP Score (<0.3)**
```bash
# Solutions:
- Increase training epochs (100-200)
- Collect more diverse training data
- Check annotation quality
- Use a larger model (yolov8m instead of yolov8s)
```

**2. Model Not Detecting Small Objects**
```bash
# Solutions:
- Increase input image size (832x832)
- Collect more data with small objects
- Use multi-scale training
- Lower confidence threshold
```

**3. Too Many False Positives**
```bash
# Solutions:
- Increase confidence threshold
- Improve training data quality
- Use surveillance-specific filtering
- Add more negative examples
```

**4. Training Takes Too Long**
```bash
# Solutions:
- Use smaller model (yolov8n)
- Reduce batch size
- Use mixed precision training
- Enable GPU acceleration
```

## 📋 Command Line Examples

### **Complete Training Pipeline**
```bash
# Full automated training (webcam data)
python train_model.py \
    --project-name surveillance_v1 \
    --base-model yolov8s.pt \
    --collection-method webcam \
    --duration 15 \
    --epochs 100

# Training with existing video
python train_model.py \
    --project-name cctv_training \
    --collection-method video \
    --video-source footage/cctv_day1.mp4 \
    --epochs 150

# Training with prepared dataset
python train_model.py \
    --project-name custom_dataset \
    --data-dir datasets/surveillance_data \
    --epochs 200 \
    --base-model yolov8m.pt
```

### **Model Testing**
```bash
# Quick webcam test
python test_accuracy.py --test-method webcam --duration 30

# Comprehensive comparison
python test_accuracy.py \
    --test-method images \
    --custom-model models/custom/surveillance_v1.pt

# Video analysis
python test_accuracy.py \
    --test-method video \
    --video-path test_videos/surveillance_test.mp4
```

## 📊 Expected Results

### **Performance Improvements**
After custom training, you can expect:

- **mAP Improvement**: 15-30% increase for surveillance-specific objects
- **False Positive Reduction**: 40-60% fewer irrelevant detections
- **Small Object Detection**: 20-40% better detection of bags, phones, etc.
- **Confidence Scores**: More reliable confidence values

### **Training Time Estimates**
- **YOLOv8n**: 30-60 minutes (100 epochs, 500 images)
- **YOLOv8s**: 1-2 hours (100 epochs, 500 images)
- **YOLOv8m**: 2-4 hours (100 epochs, 500 images)

*Times vary based on hardware (GPU recommended)*

## 🚀 Next Steps

1. **Start with Quick Training**: Use webcam collection for initial testing
2. **Evaluate Results**: Compare custom model vs. pre-trained
3. **Iterative Improvement**: Collect more data for classes with low accuracy
4. **Deploy Best Model**: Update your surveillance system to use the custom model
5. **Monitor Performance**: Continuously evaluate and retrain as needed

---

For more detailed information, check the individual module documentation in the `src/model_training/` directory.

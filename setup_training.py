#!/usr/bin/env python3
"""
Setup Script for Smart Surveillance System Model Training

This script helps set up the environment and provides guided setup
for model training and accuracy improvement.
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'models/custom',
        'models/pretrained',
        'models/backup',
        'datasets',
        'test_images',
        'collected_data/images',
        'collected_data/annotations',
        'collected_data/metadata',
        'evaluation_results/plots',
        'evaluation_results/reports',
        'evaluation_results/comparisons',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if required dependencies are installed."""
    # Map package names to their import names
    package_imports = {
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'torch': 'torch',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'flask': 'flask',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
        'seaborn': 'seaborn',
        'pillow': 'PIL',
        'pyyaml': 'yaml'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def download_sample_model():
    """Download a sample YOLO model for testing."""
    try:
        from ultralytics import YOLO
        print("Downloading YOLOv8s model...")
        model = YOLO('yolov8s.pt')
        print("✓ YOLOv8s model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading model: {str(e)}")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    config_content = '''# Sample Training Configuration
# Copy this file and modify for your specific needs

PROJECT_NAME = "my_surveillance_model"
BASE_MODEL = "yolov8s.pt"
TRAINING_EPOCHS = 50
BATCH_SIZE = 16
CONFIDENCE_THRESHOLD = 0.25

# Data Collection Settings
COLLECTION_DURATION_MINUTES = 10
COLLECTION_INTERVAL_SECONDS = 2.0
QUALITY_THRESHOLD = 0.5

# Surveillance Classes to Focus On
PRIORITY_CLASSES = [
    "person",
    "backpack", 
    "handbag",
    "suitcase",
    "bottle",
    "laptop",
    "cell phone"
]
'''
    
    config_path = Path('my_training_config.py')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"✓ Created sample config: {config_path}")
    else:
        print(f"✓ Config file already exists: {config_path}")

def print_quick_start_guide():
    """Print quick start instructions."""
    print("\n" + "="*60)
    print("🎯 SMART SURVEILLANCE - MODEL TRAINING SETUP COMPLETE!")
    print("="*60)
    print("\n📚 QUICK START GUIDE:")
    print("\n1. TRAIN A CUSTOM MODEL:")
    print("   # Using webcam (10 minutes of data collection)")
    print("   python train_model.py --project-name my_model --duration 10")
    print("")
    print("   # Using existing video")
    print("   python train_model.py --collection-method video --video-source path/to/video.mp4")
    print("")
    print("   # Using existing dataset")
    print("   python train_model.py --data-dir path/to/dataset")
    
    print("\n2. TEST MODEL ACCURACY:")
    print("   # Compare models using webcam")
    print("   python test_accuracy.py --test-method webcam --duration 30")
    print("")
    print("   # Test specific model")
    print("   python test_accuracy.py --custom-model models/custom/my_model.pt")
    
    print("\n3. RUN SURVEILLANCE SYSTEM:")
    print("   # Start with enhanced detection")
    print("   python main.py")
    
    print("\n📖 DOCUMENTATION:")
    print("   - Training Guide: TRAINING_GUIDE.md")
    print("   - Main README: README.md")
    print("   - Configuration: config/model_config.py")
    
    print("\n🔧 ADVANCED OPTIONS:")
    print("   - Collect data only: See DataCollector class")
    print("   - Evaluate existing models: See ModelEvaluator class")
    print("   - Custom training configs: Modify config/model_config.py")
    
    print("\n💡 TIPS FOR BETTER ACCURACY:")
    print("   - Collect diverse training data (different lighting, angles)")
    print("   - Use at least 500+ images per object class")
    print("   - Train for 100+ epochs for best results")
    print("   - Use larger models (yolov8m, yolov8l) for higher accuracy")
    
    print("\n" + "="*60)
    print("Ready to improve your surveillance system accuracy! 🚀")
    print("="*60)

def main():
    """Main setup function."""
    print("🔧 Setting up Smart Surveillance Model Training Environment...")
    print("")
    
    # Create directories
    print("1. Creating project directories...")
    create_directories()
    print("")
    
    # Check dependencies
    print("2. Checking dependencies...")
    deps_ok = check_dependencies()
    print("")
    
    if not deps_ok:
        print("❌ Please install missing dependencies first:")
        print("   pip install -r requirements.txt")
        return
    
    # Download sample model
    print("3. Downloading sample YOLO model...")
    download_sample_model()
    print("")
    
    # Create sample config
    print("4. Creating sample configuration...")
    create_sample_config()
    print("")
    
    # Print guide
    print_quick_start_guide()

if __name__ == "__main__":
    main()

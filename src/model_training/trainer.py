#!/usr/bin/env python3
"""
Custom Model Trainer

This module provides functionality for training custom YOLO models
optimized for surveillance scenarios with improved accuracy.
"""

import os
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomModelTrainer:
    """
    Custom trainer for YOLO models optimized for surveillance scenarios.
    
    This class handles:
    - Custom dataset preparation
    - Model training with surveillance-specific configurations
    - Model evaluation and validation
    - Best model selection and saving
    """
    
    def __init__(self, 
                 base_model: str = 'yolov8s.pt',
                 project_name: str = 'surveillance_training',
                 data_dir: str = 'datasets/surveillance'):
        """
        Initialize the custom model trainer.
        
        Args:
            base_model: Base YOLO model to start training from
            project_name: Name for the training project
            data_dir: Directory containing training data
        """
        self.base_model = base_model
        self.project_name = project_name
        self.data_dir = Path(data_dir)
        self.model = None
        self.training_config = None
        
        # Create necessary directories
        self.project_dir = Path(f'runs/train/{project_name}')
        self.model_dir = Path('models/custom')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Surveillance-specific classes (can be customized)
        self.surveillance_classes = [
            'person', 'backpack', 'handbag', 'suitcase', 
            'bottle', 'laptop', 'cell phone', 'bag',
            'suspicious_object', 'weapon', 'vehicle',
            'bicycle', 'motorbike', 'car', 'truck'
        ]
        
        logger.info(f"CustomModelTrainer initialized with base model: {base_model}")
    
    def prepare_dataset(self, 
                       train_images: List[str],
                       train_labels: List[str],
                       val_images: List[str] = None,
                       val_labels: List[str] = None,
                       val_split: float = 0.2) -> str:
        """
        Prepare dataset for training in YOLO format.
        
        Args:
            train_images: List of training image paths
            train_labels: List of training label paths (YOLO format)
            val_images: List of validation image paths
            val_labels: List of validation label paths
            val_split: Validation split ratio if val_images not provided
            
        Returns:
            Path to dataset configuration file
        """
        # Create dataset directory structure
        dataset_dir = self.data_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        train_img_dir = dataset_dir / 'images' / 'train'
        train_lbl_dir = dataset_dir / 'labels' / 'train'
        val_img_dir = dataset_dir / 'images' / 'val'
        val_lbl_dir = dataset_dir / 'labels' / 'val'
        
        for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Handle validation split
        if val_images is None:
            # Split training data into train/val
            split_idx = int(len(train_images) * (1 - val_split))
            val_images = train_images[split_idx:]
            val_labels = train_labels[split_idx:]
            train_images = train_images[:split_idx]
            train_labels = train_labels[:split_idx]
        
        # Copy images and labels
        self._copy_dataset_files(train_images, train_labels, train_img_dir, train_lbl_dir)
        self._copy_dataset_files(val_images, val_labels, val_img_dir, val_lbl_dir)
        
        # Create dataset configuration file
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.surveillance_classes)}
        }
        
        config_path = dataset_dir / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Dataset prepared with {len(train_images)} training and {len(val_images)} validation images")
        return str(config_path)
    
    def _copy_dataset_files(self, images: List[str], labels: List[str], 
                           img_dir: Path, lbl_dir: Path):
        """Copy dataset files to the appropriate directories."""
        for img_path, lbl_path in zip(images, labels):
            # Copy image
            img_dest = img_dir / Path(img_path).name
            shutil.copy2(img_path, img_dest)
            
            # Copy label
            lbl_dest = lbl_dir / Path(lbl_path).name
            shutil.copy2(lbl_path, lbl_dest)
    
    def create_training_config(self, 
                              epochs: int = 100,
                              batch_size: int = 16,
                              img_size: int = 640,
                              learning_rate: float = 0.01,
                              augmentation: bool = True) -> Dict:
        """
        Create training configuration optimized for surveillance.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            learning_rate: Initial learning rate
            augmentation: Enable data augmentation
            
        Returns:
            Training configuration dictionary
        """
        config = {
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': learning_rate,
            'lrf': 0.01,  # Final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,  # Box loss gain
            'cls': 0.5,   # Class loss gain
            'dfl': 1.5,   # DFL loss gain
            'pose': 12.0, # Pose loss gain
            'kobj': 1.0,  # Keypoint object loss gain
            'label_smoothing': 0.0,
            'nbs': 64,    # Nominal batch size
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': -1,
            'cache': False,
            'device': '',
            'workers': 8,
            'project': str(self.project_dir.parent),
            'name': self.project_name,
            'exist_ok': False,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
        }
        
        # Surveillance-specific augmentations
        if augmentation:
            config.update({
                'hsv_h': 0.015,    # Hue augmentation
                'hsv_s': 0.7,      # Saturation augmentation
                'hsv_v': 0.4,      # Value augmentation
                'degrees': 0.0,    # Rotation (disabled for surveillance)
                'translate': 0.1,  # Translation
                'scale': 0.5,      # Scale variation
                'shear': 0.0,      # Shear (disabled)
                'perspective': 0.0, # Perspective (disabled)
                'flipud': 0.0,     # Vertical flip (disabled)
                'fliplr': 0.5,     # Horizontal flip
                'mosaic': 1.0,     # Mosaic augmentation
                'mixup': 0.0,      # Mixup augmentation
                'copy_paste': 0.0  # Copy-paste augmentation
            })
        
        self.training_config = config
        return config
    
    def train_model(self, 
                   dataset_config: str,
                   custom_config: Dict = None) -> str:
        """
        Train the custom YOLO model.
        
        Args:
            dataset_config: Path to dataset configuration file
            custom_config: Custom training configuration
            
        Returns:
            Path to the best trained model
        """
        try:
            # Load base model
            self.model = YOLO(self.base_model)
            
            # Use provided config or create default
            if custom_config is None:
                custom_config = self.create_training_config()
            
            logger.info(f"Starting training with {custom_config['epochs']} epochs...")
            logger.info(f"Dataset: {dataset_config}")
            logger.info(f"Base model: {self.base_model}")
            
            # Start training
            results = self.model.train(
                data=dataset_config,
                **custom_config
            )
            
            # Get best model path
            best_model_path = self.project_dir / 'weights' / 'best.pt'
            
            if best_model_path.exists():
                # Copy best model to models directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                custom_model_path = self.model_dir / f'surveillance_model_{timestamp}.pt'
                shutil.copy2(best_model_path, custom_model_path)
                
                logger.info(f"Training completed successfully!")
                logger.info(f"Best model saved to: {custom_model_path}")
                
                return str(custom_model_path)
            else:
                raise FileNotFoundError("Best model not found after training")
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate_model(self, model_path: str, test_data: str) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            model_path: Path to the trained model
            test_data: Path to test dataset configuration
            
        Returns:
            Evaluation metrics
        """
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(data=test_data)
            
            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
            }
            
            logger.info("Model Evaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def create_sample_annotations(self, 
                                 image_paths: List[str], 
                                 output_dir: str = "annotations") -> List[str]:
        """
        Create sample annotation files for labeling.
        
        Args:
            image_paths: List of image paths to create annotations for
            output_dir: Directory to save annotation files
            
        Returns:
            List of created annotation file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        annotation_paths = []
        
        for img_path in image_paths:
            img_name = Path(img_path).stem
            annotation_path = output_dir / f"{img_name}.txt"
            
            # Create empty annotation file
            with open(annotation_path, 'w') as f:
                f.write("# YOLO format annotations\n")
                f.write("# Format: class_id center_x center_y width height\n")
                f.write("# All coordinates are normalized (0-1)\n")
                f.write("# Example: 0 0.5 0.5 0.2 0.3\n")
            
            annotation_paths.append(str(annotation_path))
        
        logger.info(f"Created {len(annotation_paths)} sample annotation files in {output_dir}")
        return annotation_paths
    
    def auto_annotate_with_pretrained(self, 
                                     image_paths: List[str],
                                     confidence_threshold: float = 0.5) -> List[str]:
        """
        Auto-annotate images using a pre-trained model as a starting point.
        
        Args:
            image_paths: List of image paths to annotate
            confidence_threshold: Minimum confidence for auto-annotations
            
        Returns:
            List of generated annotation file paths
        """
        try:
            # Load pre-trained model for initial annotations
            pretrained_model = YOLO('yolov8s.pt')
            
            annotation_paths = []
            
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # Run detection
                results = pretrained_model(img, conf=confidence_threshold)
                
                # Create annotation file
                img_name = Path(img_path).stem
                annotation_path = f"annotations/{img_name}.txt"
                
                os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
                
                with open(annotation_path, 'w') as f:
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for i in range(len(boxes)):
                                # Get box coordinates (xyxy format)
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                
                                # Convert to YOLO format (normalized xywh)
                                center_x = (x1 + x2) / 2 / w
                                center_y = (y1 + y2) / 2 / h
                                width = (x2 - x1) / w
                                height = (y2 - y1) / h
                                
                                class_id = int(boxes.cls[i].cpu().numpy())
                                confidence = float(boxes.conf[i].cpu().numpy())
                                
                                # Write annotation
                                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                annotation_paths.append(annotation_path)
            
            logger.info(f"Auto-annotated {len(annotation_paths)} images")
            return annotation_paths
            
        except Exception as e:
            logger.error(f"Error during auto-annotation: {str(e)}")
            raise


def main():
    """Example usage of the CustomModelTrainer."""
    trainer = CustomModelTrainer()
    
    # Example: Create sample annotations
    sample_images = ["sample1.jpg", "sample2.jpg"]  # Replace with actual image paths
    trainer.create_sample_annotations(sample_images)
    
    logger.info("Custom model trainer example completed!")


if __name__ == "__main__":
    main()

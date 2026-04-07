"""
Model Configuration Settings

This file contains configuration settings for model training and detection.
"""

# Model Training Configuration
TRAINING_CONFIG = {
    # Base model settings
    'base_model': 'yolov8s.pt',  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'learning_rate': 0.01,
    'patience': 50,  # Early stopping patience
    
    # Data augmentation
    'augmentation': True,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'mosaic': 1.0,
    
    # Surveillance-specific settings
    'surveillance_mode': True,
    'enhanced_detection': True,
}

# Data Collection Configuration
DATA_COLLECTION_CONFIG = {
    'collection_interval': 2.0,  # Seconds between frame collection
    'quality_threshold': 0.5,    # Minimum quality score for frames
    'max_frames_per_session': 1000,
    'auto_annotation_confidence': 0.6,
    'validation_split': 0.2,
}

# Model Paths
MODEL_PATHS = {
    'pretrained_dir': 'models/pretrained',
    'custom_dir': 'models/custom',
    'backup_dir': 'models/backup',
    'datasets_dir': 'datasets',
    'results_dir': 'evaluation_results',
}

# Surveillance Classes (high priority for detection)
SURVEILLANCE_CLASSES = [
    'person',
    'backpack', 
    'handbag', 
    'suitcase',
    'bottle',
    'laptop',
    'cell phone',
    'bag',
    'bicycle',
    'motorbike',
    'car',
    'truck',
    'knife',
    'scissors',
    'sports ball'
]

# Detection Optimization Settings
DETECTION_CONFIG = {
    'use_enhanced_detector': True,
    'surveillance_mode': True,
    'max_detections': 300,
    'agnostic_nms': False,
    'test_time_augmentation': True,
    
    # Filtering settings
    'min_object_size': 0.001,  # Minimum object area ratio
    'max_object_size': 0.8,    # Maximum object area ratio
    'edge_detection_threshold': 0.6,  # Higher threshold for edge detections
    
    # Performance settings
    'half_precision': True,  # Use FP16 for faster inference
    'device': 'auto',  # 'auto', 'cpu', 'cuda', 'mps'
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'confidence_thresholds': [0.1, 0.25, 0.5, 0.75],
    'iou_thresholds': [0.3, 0.45, 0.5, 0.75],
    'save_plots': True,
    'save_confusion_matrix': True,
    'per_class_analysis': True,
}

# Hardware Optimization
HARDWARE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_fraction': 0.8,
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'save_logs': True,
    'log_dir': 'logs',
    'tensorboard_logging': True,
    'wandb_logging': False,  # Set to True if using Weights & Biases
    'wandb_project': 'surveillance-training',
}

# Custom Model Selection Strategy
MODEL_SELECTION = {
    'auto_select_best': True,  # Automatically use the best available model
    'preferred_metric': 'mAP50',  # 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1'
    'fallback_to_pretrained': True,
    'model_age_threshold_days': 30,  # Consider models older than this as outdated
}

# Advanced Training Options
ADVANCED_TRAINING = {
    'freeze_backbone': False,
    'freeze_epochs': 0,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'label_smoothing': 0.0,
    'box_gain': 0.05,
    'cls_gain': 0.5,
    'dfl_gain': 1.5,
    'dropout': 0.0,
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'cos_lr': False,
    'lr_factor': 0.01,
    'scheduler': 'linear',  # 'linear', 'cosine', 'polynomial'
}

# Data Pipeline Configuration
DATA_PIPELINE = {
    'cache_images': False,  # Cache images in memory (requires more RAM)
    'cache_ram': False,     # Cache in RAM vs disk
    'workers': 8,           # Number of dataloader workers
    'rect_training': False, # Rectangular training
    'single_cls': False,    # Train as single-class dataset
    'image_weights': False, # Use weighted image selection
    'multi_scale': True,    # Multi-scale training
    'overlap_mask': True,   # Overlap masks
    'mask_ratio': 4,        # Mask downsample ratio
}

def get_config(config_name: str) -> dict:
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'training': TRAINING_CONFIG,
        'data_collection': DATA_COLLECTION_CONFIG,
        'detection': DETECTION_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'logging': LOGGING_CONFIG,
        'model_selection': MODEL_SELECTION,
        'advanced_training': ADVANCED_TRAINING,
        'data_pipeline': DATA_PIPELINE,
    }
    
    return configs.get(config_name, {})

def get_model_path(model_type: str = 'auto') -> str:
    """
    Get the path to the model to use.
    
    Args:
        model_type: 'auto', 'custom', 'pretrained', or specific path
        
    Returns:
        Path to model file
    """
    import os
    from pathlib import Path
    
    if model_type == 'auto':
        # Auto-select best available model
        custom_dir = Path(MODEL_PATHS['custom_dir'])
        
        if custom_dir.exists():
            custom_models = list(custom_dir.glob('*.pt'))
            if custom_models:
                # Get the most recent custom model
                latest_model = max(custom_models, key=os.path.getctime)
                return str(latest_model)
        
        # Fall back to pretrained model
        return TRAINING_CONFIG['base_model']
    
    elif model_type == 'custom':
        custom_dir = Path(MODEL_PATHS['custom_dir'])
        custom_models = list(custom_dir.glob('*.pt'))
        if custom_models:
            return str(max(custom_models, key=os.path.getctime))
        else:
            raise FileNotFoundError("No custom models found")
    
    elif model_type == 'pretrained':
        return TRAINING_CONFIG['base_model']
    
    else:
        # Assume it's a specific path
        return model_type

def update_config(config_name: str, updates: dict):
    """
    Update configuration values.
    
    Args:
        config_name: Name of the configuration to update
        updates: Dictionary of updates to apply
    """
    configs = {
        'training': TRAINING_CONFIG,
        'data_collection': DATA_COLLECTION_CONFIG,
        'detection': DETECTION_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'logging': LOGGING_CONFIG,
        'model_selection': MODEL_SELECTION,
        'advanced_training': ADVANCED_TRAINING,
        'data_pipeline': DATA_PIPELINE,
    }
    
    if config_name in configs:
        configs[config_name].update(updates)
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

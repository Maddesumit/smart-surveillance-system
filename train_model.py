#!/usr/bin/env python3
"""
Model Training Pipeline

This script provides a complete pipeline for training custom YOLO models
optimized for surveillance scenarios with improved accuracy.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_training.trainer import CustomModelTrainer
from model_training.data_collector import DataCollector
from model_training.evaluator import ModelEvaluator

# Configure logging
def setup_logging(log_level='INFO'):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('training_pipeline')

class TrainingPipeline:
    """
    Complete training pipeline for surveillance-optimized YOLO models.
    """
    
    def __init__(self, 
                 project_name: str = 'surveillance_training',
                 base_model: str = 'yolov8s.pt'):
        """
        Initialize the training pipeline.
        
        Args:
            project_name: Name for the training project
            base_model: Base YOLO model to start from
        """
        self.project_name = project_name
        self.base_model = base_model
        self.logger = logging.getLogger('training_pipeline')
        
        # Initialize components
        self.trainer = CustomModelTrainer(
            base_model=base_model,
            project_name=project_name
        )
        self.collector = DataCollector()
        self.evaluator = ModelEvaluator()
        
        self.logger.info(f"Training pipeline initialized: {project_name}")
    
    def collect_training_data(self, 
                             collection_method: str = 'webcam',
                             duration_minutes: int = 10,
                             video_source: str = None) -> int:
        """
        Collect training data using specified method.
        
        Args:
            collection_method: 'webcam', 'video', or 'manual'
            duration_minutes: Duration for automated collection
            video_source: Path to video file or camera index
            
        Returns:
            Number of frames collected
        """
        self.logger.info(f"Starting data collection: {collection_method}")
        
        if collection_method == 'webcam':
            frames_collected = self.collector.start_collection_session(
                session_name=f"{self.project_name}_webcam_data",
                video_source=0,
                duration_minutes=duration_minutes,
                collection_interval=2.0,  # Collect every 2 seconds
                quality_threshold=0.5
            )
        
        elif collection_method == 'video' and video_source:
            frames_list = self.collector.collect_frames_from_video(
                video_path=video_source,
                frame_interval=30,  # Every 30th frame
                max_frames=1000
            )
            frames_collected = len(frames_list)
        
        else:
            self.logger.warning("Manual collection selected - user must provide data")
            return 0
        
        self.logger.info(f"Data collection completed: {frames_collected} frames")
        return frames_collected
    
    def auto_annotate_data(self, confidence_threshold: float = 0.6) -> int:
        """
        Auto-annotate collected data using pre-trained model.
        
        Args:
            confidence_threshold: Minimum confidence for auto-annotations
            
        Returns:
            Number of annotated images
        """
        self.logger.info("Starting auto-annotation process...")
        
        # Get collected images
        images_dir = Path(self.collector.collection_dir) / 'images'
        image_files = list(images_dir.glob('*.jpg'))
        
        if not image_files:
            self.logger.warning("No images found for annotation")
            return 0
        
        # Auto-annotate
        annotation_paths = self.trainer.auto_annotate_with_pretrained(
            image_paths=[str(img) for img in image_files],
            confidence_threshold=confidence_threshold
        )
        
        self.logger.info(f"Auto-annotation completed: {len(annotation_paths)} files")
        return len(annotation_paths)
    
    def prepare_dataset(self, 
                       data_dir: str = None,
                       val_split: float = 0.2) -> str:
        """
        Prepare dataset for training.
        
        Args:
            data_dir: Directory containing training data (if None, use collected data)
            val_split: Validation split ratio
            
        Returns:
            Path to dataset configuration file
        """
        self.logger.info("Preparing dataset for training...")
        
        if data_dir is None:
            # Use collected data
            dataset_info = self.collector.export_training_dataset(
                output_dir=f"datasets/{self.project_name}",
                train_split=1.0 - val_split,
                min_quality=0.4,
                annotated_only=True
            )
            
            if not dataset_info:
                raise ValueError("No annotated data available for training")
            
            # Create dataset config
            config_path = self.trainer.prepare_dataset(
                train_images=dataset_info['train_images'],
                train_labels=dataset_info['train_labels'],
                val_images=dataset_info['val_images'],
                val_labels=dataset_info['val_labels']
            )
        else:
            # Use provided data directory
            data_path = Path(data_dir)
            
            # Scan for images and labels
            train_images = list((data_path / 'images' / 'train').glob('*.jpg'))
            train_labels = list((data_path / 'labels' / 'train').glob('*.txt'))
            val_images = list((data_path / 'images' / 'val').glob('*.jpg'))
            val_labels = list((data_path / 'labels' / 'val').glob('*.txt'))
            
            config_path = self.trainer.prepare_dataset(
                train_images=[str(img) for img in train_images],
                train_labels=[str(lbl) for lbl in train_labels],
                val_images=[str(img) for img in val_images],
                val_labels=[str(lbl) for lbl in val_labels]
            )
        
        self.logger.info(f"Dataset prepared: {config_path}")
        return config_path
    
    def train_model(self, 
                   dataset_config: str,
                   epochs: int = 100,
                   batch_size: int = 16,
                   img_size: int = 640,
                   learning_rate: float = 0.01) -> str:
        """
        Train the custom model.
        
        Args:
            dataset_config: Path to dataset configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            learning_rate: Initial learning rate
            
        Returns:
            Path to trained model
        """
        self.logger.info("Starting model training...")
        
        # Create training configuration
        config = self.trainer.create_training_config(
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            learning_rate=learning_rate,
            augmentation=True
        )
        
        # Train model
        model_path = self.trainer.train_model(dataset_config, config)
        
        self.logger.info(f"Training completed: {model_path}")
        return model_path
    
    def evaluate_model(self, 
                      model_path: str, 
                      test_dataset: str = None) -> dict:
        """
        Evaluate the trained model.
        
        Args:
            model_path: Path to trained model
            test_dataset: Path to test dataset (if None, use validation set)
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating trained model...")
        
        if test_dataset is None:
            # Use the training dataset config for validation
            test_dataset = self.trainer.data_dir / 'data.yaml'
        
        metrics = self.evaluator.evaluate_model(
            model_path=model_path,
            test_dataset=str(test_dataset),
            model_name=f"{self.project_name}_custom"
        )
        
        self.logger.info("Model evaluation completed")
        return metrics
    
    def run_full_pipeline(self, 
                         collection_method: str = 'webcam',
                         duration_minutes: int = 10,
                         epochs: int = 50,
                         video_source: str = None) -> dict:
        """
        Run the complete training pipeline.
        
        Args:
            collection_method: Data collection method
            duration_minutes: Duration for data collection
            epochs: Number of training epochs
            video_source: Video source for data collection
            
        Returns:
            Results summary
        """
        results = {
            'project_name': self.project_name,
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # Step 1: Collect data
            self.logger.info("STEP 1: Collecting training data")
            frames_collected = self.collect_training_data(
                collection_method=collection_method,
                duration_minutes=duration_minutes,
                video_source=video_source
            )
            results['frames_collected'] = frames_collected
            results['steps_completed'].append('data_collection')
            
            if frames_collected == 0:
                raise ValueError("No training data collected")
            
            # Step 2: Auto-annotate
            self.logger.info("STEP 2: Auto-annotating data")
            annotated_count = self.auto_annotate_data()
            results['annotated_frames'] = annotated_count
            results['steps_completed'].append('auto_annotation')
            
            if annotated_count < 10:
                raise ValueError("Insufficient annotated data for training")
            
            # Step 3: Prepare dataset
            self.logger.info("STEP 3: Preparing dataset")
            dataset_config = self.prepare_dataset()
            results['dataset_config'] = dataset_config
            results['steps_completed'].append('dataset_preparation')
            
            # Step 4: Train model
            self.logger.info("STEP 4: Training model")
            model_path = self.train_model(
                dataset_config=dataset_config,
                epochs=epochs
            )
            results['model_path'] = model_path
            results['steps_completed'].append('model_training')
            
            # Step 5: Evaluate model
            self.logger.info("STEP 5: Evaluating model")
            metrics = self.evaluate_model(model_path)
            results['evaluation_metrics'] = metrics
            results['steps_completed'].append('model_evaluation')
            
            results['end_time'] = datetime.now().isoformat()
            results['success'] = True
            
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Trained model saved to: {model_path}")
            
            # Print summary
            self._print_summary(results)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results['errors'].append(str(e))
            results['success'] = False
            results['end_time'] = datetime.now().isoformat()
        
        return results
    
    def _print_summary(self, results: dict):
        """Print pipeline results summary."""
        print("\n" + "="*60)
        print("TRAINING PIPELINE SUMMARY")
        print("="*60)
        print(f"Project: {results['project_name']}")
        print(f"Success: {results.get('success', False)}")
        print(f"Frames Collected: {results.get('frames_collected', 0)}")
        print(f"Annotated Frames: {results.get('annotated_frames', 0)}")
        
        if 'evaluation_metrics' in results:
            metrics = results['evaluation_metrics']['overall_metrics']
            print(f"\nModel Performance:")
            print(f"  mAP@0.5: {metrics.get('mAP50', 0):.3f}")
            print(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.3f}")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall: {metrics.get('recall', 0):.3f}")
            print(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
        
        if 'model_path' in results:
            print(f"\nTrained Model: {results['model_path']}")
        
        print("="*60)


def main():
    """Main function for running the training pipeline."""
    parser = argparse.ArgumentParser(description='Train custom YOLO model for surveillance')
    
    parser.add_argument('--project-name', type=str, default='surveillance_custom',
                       help='Name for the training project')
    parser.add_argument('--base-model', type=str, default='yolov8s.pt',
                       help='Base YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)')
    parser.add_argument('--collection-method', type=str, default='webcam',
                       choices=['webcam', 'video', 'manual'],
                       help='Data collection method')
    parser.add_argument('--duration', type=int, default=10,
                       help='Data collection duration in minutes')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--video-source', type=str, default=None,
                       help='Path to video file (for video collection method)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory with existing training data')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Initialize pipeline
        pipeline = TrainingPipeline(
            project_name=args.project_name,
            base_model=args.base_model
        )
        
        if args.data_dir:
            # Use existing data
            logger.info(f"Using existing data from: {args.data_dir}")
            dataset_config = pipeline.prepare_dataset(data_dir=args.data_dir)
            model_path = pipeline.train_model(dataset_config, epochs=args.epochs)
            metrics = pipeline.evaluate_model(model_path)
            
            print(f"\nTraining completed! Model saved to: {model_path}")
            
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline(
                collection_method=args.collection_method,
                duration_minutes=args.duration,
                epochs=args.epochs,
                video_source=args.video_source
            )
            
            if not results.get('success', False):
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

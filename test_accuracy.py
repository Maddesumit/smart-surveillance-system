#!/usr/bin/env python3
"""
Model Accuracy Testing Script

This script helps test and compare the accuracy of different models
for surveillance object detection.
"""

import os
import sys
import cv2
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from object_detection.detector import ObjectDetector
from object_detection.enhanced_detector import EnhancedObjectDetector
from model_training.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('accuracy_test')

class AccuracyTester:
    """
    Test and compare model accuracy for surveillance scenarios.
    """
    
    def __init__(self):
        self.results = {}
        self.test_images = []
        self.ground_truth = []
    
    def compare_models(self, 
                      models_config: dict,
                      test_method: str = 'webcam',
                      duration_seconds: int = 30):
        """
        Compare accuracy of different models.
        
        Args:
            models_config: Dictionary of model configurations
            test_method: 'webcam', 'images', or 'video'
            duration_seconds: Duration for webcam test
        """
        logger.info(f"Starting model comparison: {test_method}")
        
        results = {}
        
        for model_name, config in models_config.items():
            logger.info(f"Testing model: {model_name}")
            
            try:
                # Initialize detector
                if config['type'] == 'enhanced':
                    detector = EnhancedObjectDetector(
                        model_path=config['model_path'],
                        confidence_threshold=config.get('confidence', 0.25),
                        surveillance_mode=config.get('surveillance_mode', True)
                    )
                else:
                    detector = ObjectDetector(
                        model_size=config.get('model_size', 's'),
                        confidence_threshold=config.get('confidence', 0.25),
                        use_enhanced=False
                    )
                
                # Run test
                if test_method == 'webcam':
                    result = self._test_webcam(detector, model_name, duration_seconds)
                elif test_method == 'images':
                    result = self._test_images(detector, model_name)
                elif test_method == 'video':
                    result = self._test_video(detector, model_name, config.get('video_path'))
                
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Generate comparison report
        self._generate_comparison_report(results)
        return results
    
    def _test_webcam(self, detector, model_name: str, duration: int) -> dict:
        """Test model with webcam feed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        start_time = time.time()
        frames_processed = 0
        total_detections = 0
        total_confidence = 0.0
        processing_times = []
        
        logger.info(f"Testing {model_name} with webcam for {duration} seconds...")
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Time the detection
                detect_start = time.time()
                detections = detector.detect(frame)
                detect_time = time.time() - detect_start
                
                # Collect statistics
                frames_processed += 1
                total_detections += len(detections)
                processing_times.append(detect_time * 1000)  # Convert to ms
                
                if detections:
                    total_confidence += sum(d['confidence'] for d in detections)
                
                # Display frame with detections
                result_frame = detector.draw_detections(frame, detections)
                cv2.putText(result_frame, f"Model: {model_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(f'Testing {model_name}', result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Calculate metrics
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_detections_per_frame = total_detections / frames_processed if frames_processed > 0 else 0
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        fps = frames_processed / duration if duration > 0 else 0
        
        return {
            'frames_processed': frames_processed,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': avg_processing_time,
            'fps': fps,
            'duration_seconds': duration
        }
    
    def _test_images(self, detector, model_name: str) -> dict:
        """Test model with static images."""
        # Look for test images
        test_dir = Path('test_images')
        if not test_dir.exists():
            logger.warning("No test_images directory found")
            return {'error': 'No test images directory'}
        
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        if not image_files:
            logger.warning("No test images found")
            return {'error': 'No test images found'}
        
        total_detections = 0
        total_confidence = 0.0
        processing_times = []
        images_processed = 0
        
        for img_path in image_files[:10]:  # Test first 10 images
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                
                detect_start = time.time()
                detections = detector.detect(frame)
                detect_time = time.time() - detect_start
                
                images_processed += 1
                total_detections += len(detections)
                processing_times.append(detect_time * 1000)
                
                if detections:
                    total_confidence += sum(d['confidence'] for d in detections)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_detections_per_image = total_detections / images_processed if images_processed > 0 else 0
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        
        return {
            'images_processed': images_processed,
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections_per_image,
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': avg_processing_time
        }
    
    def _test_video(self, detector, model_name: str, video_path: str) -> dict:
        """Test model with video file."""
        if not video_path or not os.path.exists(video_path):
            return {'error': 'Video file not found'}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video file'}
        
        frames_processed = 0
        total_detections = 0
        total_confidence = 0.0
        processing_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detect_start = time.time()
                detections = detector.detect(frame)
                detect_time = time.time() - detect_start
                
                frames_processed += 1
                total_detections += len(detections)
                processing_times.append(detect_time * 1000)
                
                if detections:
                    total_confidence += sum(d['confidence'] for d in detections)
                
                # Process every 10th frame to speed up testing
                for _ in range(9):
                    cap.read()
        
        finally:
            cap.release()
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_detections_per_frame = total_detections / frames_processed if frames_processed > 0 else 0
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        
        return {
            'frames_processed': frames_processed,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'avg_confidence': avg_confidence,
            'avg_processing_time_ms': avg_processing_time
        }
    
    def _generate_comparison_report(self, results: dict):
        """Generate a comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"model_comparison_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("Model Accuracy Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Model':<20} {'Avg Detect':<12} {'Avg Conf':<10} {'Proc Time':<12} {'FPS':<8}\n")
            f.write("-" * 62 + "\n")
            
            for model_name, result in results.items():
                if 'error' in result:
                    f.write(f"{model_name:<20} ERROR: {result['error']}\n")
                    continue
                
                avg_detect = result.get('avg_detections_per_frame', 
                                      result.get('avg_detections_per_image', 0))
                avg_conf = result.get('avg_confidence', 0)
                proc_time = result.get('avg_processing_time_ms', 0)
                fps = result.get('fps', 0)
                
                f.write(f"{model_name:<20} {avg_detect:<12.2f} {avg_conf:<10.3f} "
                       f"{proc_time:<12.1f} {fps:<8.1f}\n")
            
            f.write("\nDetailed Results:\n")
            f.write("-" * 17 + "\n")
            for model_name, result in results.items():
                f.write(f"\n{model_name}:\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
        
        logger.info(f"Comparison report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"{model_name}: ERROR - {result['error']}")
                continue
            
            avg_detect = result.get('avg_detections_per_frame', 
                                  result.get('avg_detections_per_image', 0))
            avg_conf = result.get('avg_confidence', 0)
            proc_time = result.get('avg_processing_time_ms', 0)
            
            print(f"\n{model_name}:")
            print(f"  Average Detections: {avg_detect:.2f}")
            print(f"  Average Confidence: {avg_conf:.3f}")
            print(f"  Processing Time: {proc_time:.1f}ms")
            if 'fps' in result:
                print(f"  FPS: {result['fps']:.1f}")
        
        print("="*60)


def main():
    """Main function for model accuracy testing."""
    parser = argparse.ArgumentParser(description='Test and compare model accuracy')
    
    parser.add_argument('--test-method', type=str, default='webcam',
                       choices=['webcam', 'images', 'video'],
                       help='Testing method')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration for webcam test (seconds)')
    parser.add_argument('--video-path', type=str, default=None,
                       help='Path to video file for testing')
    parser.add_argument('--custom-model', type=str, default=None,
                       help='Path to custom model for testing')
    
    args = parser.parse_args()
    
    # Define models to test
    models_config = {
        'YOLOv8s_Basic': {
            'type': 'basic',
            'model_size': 's',
            'confidence': 0.25
        },
        'YOLOv8s_Enhanced': {
            'type': 'enhanced',
            'model_path': 'yolov8s.pt',
            'confidence': 0.25,
            'surveillance_mode': True
        }
    }
    
    # Add custom model if provided
    if args.custom_model and os.path.exists(args.custom_model):
        models_config['Custom_Model'] = {
            'type': 'enhanced',
            'model_path': args.custom_model,
            'confidence': 0.25,
            'surveillance_mode': True
        }
    
    # Add any available custom models
    custom_models_dir = Path('models/custom')
    if custom_models_dir.exists():
        custom_models = list(custom_models_dir.glob('*.pt'))
        for i, model_path in enumerate(custom_models[:2]):  # Test up to 2 custom models
            models_config[f'Custom_Model_{i+1}'] = {
                'type': 'enhanced',
                'model_path': str(model_path),
                'confidence': 0.25,
                'surveillance_mode': True
            }
    
    # Add video path to all configs if provided
    if args.video_path:
        for config in models_config.values():
            config['video_path'] = args.video_path
    
    # Run comparison
    tester = AccuracyTester()
    results = tester.compare_models(
        models_config=models_config,
        test_method=args.test_method,
        duration_seconds=args.duration
    )
    
    logger.info("Model accuracy testing completed!")


if __name__ == "__main__":
    main()

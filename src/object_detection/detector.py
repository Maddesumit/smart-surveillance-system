import cv2
import numpy as np
import logging
import torch
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import enhanced detector
try:
    from .enhanced_detector import EnhancedObjectDetector
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logger.warning("Enhanced detector not available, using basic detector")

class ObjectDetector:
    def __init__(self, model_size='s', confidence_threshold=0.4, use_enhanced=True,
                 imgsz=960, device='auto', allowed_classes=None):
        """
        Initialize the ObjectDetector.

        Args:
            model_size (str): Size of the model to use (e.g., 's', 'm', 'l' for YOLO).
                                Corresponds to YOLOv8 model sizes (n, s, m, l, x).
                                Larger models (m/l) are more accurate but slower.
            confidence_threshold (float): Minimum confidence score for a detection.
            use_enhanced (bool): Use enhanced detector if available.
            imgsz (int): Inference resolution. Higher values (960, 1280) improve
                         detection of small/distant objects at the cost of speed.
            device (str): 'auto' (CUDA > MPS > CPU), or force 'cuda'/'mps'/'cpu'.
            allowed_classes (list[str]|None): Restrict detection to these class
                         names. None = all 80 COCO classes.
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        self.imgsz = imgsz
        self.device = device
        self.allowed_classes = allowed_classes
        
        if self.use_enhanced:
            # Use enhanced detector
            logger.info("Using Enhanced Object Detector")
            model_path = f'yolov8{model_size}.pt'
            
            # Check for custom models
            custom_models_dir = Path('models/custom')
            if custom_models_dir.exists():
                custom_models = list(custom_models_dir.glob('*.pt'))
                if custom_models:
                    # Use the most recent custom model
                    latest_model = max(custom_models, key=os.path.getctime)
                    logger.info(f"Found custom model: {latest_model}")
                    model_path = str(latest_model)
            
            self.detector = EnhancedObjectDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                surveillance_mode=True,
                imgsz=imgsz,
                device=device,
                allowed_classes=allowed_classes
            )
        else:
            # Use basic detector
            logger.info("Using Basic Object Detector")
            self._init_basic_detector()
    
    def _init_basic_detector(self):
        """Initialize basic YOLO detector."""
        try:
            from ultralytics import YOLO
            model_name = f'yolov8{self.model_size}.pt' 
            self.model = YOLO(model_name)
            # Resolve and apply device (CUDA > MPS > CPU)
            self.resolved_device = self._resolve_basic_device(self.device)
            try:
                self.model.to(self.resolved_device)
            except Exception as e:
                logger.warning(f"Could not move basic model to {self.resolved_device}: {e}")
                self.resolved_device = 'cpu'
            logger.info(f"{model_name} loaded on device: {self.resolved_device}")
            # Resolve allowed class names -> ids for the basic path
            self.basic_class_filter = None
            if self.allowed_classes and hasattr(self.model, 'names'):
                name_to_id = {name: idx for idx, name in self.model.names.items()}
                ids = [name_to_id[n] for n in self.allowed_classes if n in name_to_id]
                self.basic_class_filter = sorted(ids) if ids else None
        except Exception as e:
            logger.error(f"Error loading YOLO model ({model_name}): {e}")
            self.model = None 
            raise

    @staticmethod
    def _resolve_basic_device(device):
        """Resolve best device for the basic detector path."""
        if device and device != 'auto':
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
        return 'cpu'

    def detect(self, frame):
        """
        Detect objects in a given frame.

        Args:
            frame (numpy.ndarray): The input frame (image) for object detection.

        Returns:
            list: A list of detections. Each detection should be a dictionary with keys
                  like 'bbox' (bounding box [x1, y1, x2, y2]), 'class_id', 'class_name',
                  and 'confidence'.
        """
        if self.use_enhanced:
            return self.detector.detect(frame)
        else:
            return self._basic_detect(frame)
    
    def _basic_detect(self, frame):
        """Basic detection using original implementation."""
        if self.model is None:
            logger.warning("Object detection model not loaded or failed to load. Returning empty detections.")
            return []

        results = self.model(frame, conf=self.confidence_threshold,
                              imgsz=self.imgsz,
                              device=getattr(self, 'resolved_device', None),
                              classes=getattr(self, 'basic_class_filter', None),
                              verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            names = result.names
            for i in range(len(boxes)):
                if boxes.conf[i] >= self.confidence_threshold:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    detections.append({
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'class_name': names[int(boxes.cls[i].cpu().numpy())],
                        'confidence': float(boxes.conf[i].cpu().numpy())
                    })
        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels for detections on the frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            detections (list): A list of detections from the detect() method.

        Returns:
            numpy.ndarray: The frame with detections drawn.
        """
        if self.use_enhanced:
            return self.detector.draw_detections(frame, detections, show_metadata=True)
        else:
            return self._basic_draw_detections(frame, detections)
    
    def _basic_draw_detections(self, frame, detections):
        """Basic drawing using original implementation."""
        result_frame = frame.copy()
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Put label above the bounding box
            cv2.putText(result_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        return result_frame
    
    def load_custom_model(self, model_path: str) -> bool:
        """
        Load a custom trained model.
        
        Args:
            model_path: Path to custom model
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_enhanced:
            return self.detector.load_custom_model(model_path)
        else:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                logger.info(f"Custom model loaded: {model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load custom model: {str(e)}")
                return False
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        if self.use_enhanced:
            return {
                'type': 'enhanced',
                'model_path': self.detector.model_path,
                'surveillance_mode': self.detector.surveillance_mode,
                'stats': self.detector.get_detection_stats()
            }
        else:
            return {
                'type': 'basic',
                'model_size': self.model_size,
                'confidence_threshold': self.confidence_threshold
            }

if __name__ == '__main__':
    logger.info("Attempting to initialize ObjectDetector for webcam test...")
    try:
        detector = ObjectDetector(model_size='s', confidence_threshold=0.25) # You can adjust model_size and threshold
        logger.info("ObjectDetector initialized.")

        cap = cv2.VideoCapture(0) # 0 is usually the default webcam
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            exit()
        
        logger.info("Webcam opened. Starting detection loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                break

            # Perform detection
            detections = detector.detect(frame)
            # logger.info(f"Detected {len(detections)} objects.") # Optional: uncomment for verbose logging

            # Draw detections on the frame
            frame_with_detections = detector.draw_detections(frame, detections)

            # Display the resulting frame
            cv2.imshow('Webcam Object Detection Test', frame_with_detections)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                logger.info("Quit key pressed. Exiting loop.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam test finished.")

    except Exception as e:
        logger.error(f"Error during detector webcam test: {e}", exc_info=True) # exc_info=True gives more details

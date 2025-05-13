import cv2
import numpy as np
import logging
import torch # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_size='s', confidence_threshold=0.25):
        """
        Initialize the ObjectDetector.

        Args:
            model_size (str): Size of the model to use (e.g., 's', 'm', 'l' for YOLO).
                                Corresponds to YOLOv8 model sizes (n, s, m, l, x).
            confidence_threshold (float): Minimum confidence score for a detection.
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        # Load YOLO model
        try:
            from ultralytics import YOLO
            # Corrected model name: e.g., 'yolov8s.pt', 'yolov8m.pt'
            # The ultralytics library will download these if not found locally.
            model_name = f'yolov8{model_size}.pt' 
            self.model = YOLO(model_name)
            
            # Setting confidence threshold on the model directly might not be available
            # or might be done differently in YOLOv8. It's often applied during inference or post-processing.
            # For now, we'll rely on filtering by confidence_threshold in the detect method.
            # If self.model.conf exists and is settable, this would be: self.model.conf = confidence_threshold
            logger.info(f"{model_name} model loaded successfully (or will be downloaded).")
        except Exception as e:
            logger.error(f"Error loading YOLO model ({model_name}): {e}")
            self.model = None 
            raise
        # logger.info(f"ObjectDetector initialized with model_size='{model_size}' and confidence_threshold={confidence_threshold}")
        # Placeholder: Replace with actual model loading
        # self.model = None # Remove this line as model is loaded above

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
        if self.model is None:
            logger.warning("Object detection model not loaded or failed to load. Returning empty detections.")
            return []

        results = self.model(frame)
        # logger.debug(f"Raw model results: {results}") # Uncomment for very verbose output
        
        detections = []
        for result in results:
            boxes = result.boxes
            names = result.names
            # logger.debug(f"Processing result with {len(boxes)} potential boxes.")
            for i in range(len(boxes)):
                # logger.debug(f"Box {i} conf: {boxes.conf[i]}, threshold: {self.confidence_threshold}")
                if boxes.conf[i] >= self.confidence_threshold:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    detections.append({
                        'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'class_name': names[int(boxes.cls[i].cpu().numpy())],
                        'confidence': float(boxes.conf[i].cpu().numpy())
                    })
        # logger.info(f"Returning {len(detections)} detections after filtering.")
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

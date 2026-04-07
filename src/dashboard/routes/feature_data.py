"""
Feature Details Data Module

Contains comprehensive information about each AI feature for the feature detail pages.
"""

FEATURES_DATA = {
    'object-detection': {
        'title': 'Object Detection',
        'slug': 'object-detection',
        'icon': 'fas fa-cube',
        'gradient': 'linear-gradient(135deg, #667eea, #764ba2)',
        'description': 'Real-time detection and classification of 80+ object classes using state-of-the-art YOLOv8 deep learning model.',
        'tags': ['YOLOv8', 'Deep Learning', 'Real-time', 'COCO Dataset', 'PyTorch'],
        'overview': '''Object detection is the foundational capability of our surveillance system. Using YOLOv8 (You Only Look Once version 8), we can detect and classify objects in video frames at speeds exceeding 30 FPS on consumer hardware. The model is pre-trained on the COCO dataset containing 80 object categories including people, vehicles, animals, and everyday objects. Unlike traditional two-stage detectors, YOLO performs detection in a single forward pass through the neural network, making it ideal for real-time applications.''',
        'metrics': [
            {'value': '80+', 'label': 'Object Classes'},
            {'value': '30+', 'label': 'FPS'},
            {'value': '95%', 'label': 'mAP Score'},
            {'value': '<50ms', 'label': 'Latency'}
        ],
        'pipeline': [
            {'name': 'Frame Input', 'icon': 'fas fa-image'},
            {'name': 'Preprocessing', 'icon': 'fas fa-compress'},
            {'name': 'CNN Backbone', 'icon': 'fas fa-brain'},
            {'name': 'Feature Pyramid', 'icon': 'fas fa-layer-group'},
            {'name': 'Detection Head', 'icon': 'fas fa-crosshairs'},
            {'name': 'NMS', 'icon': 'fas fa-filter'},
            {'name': 'Bounding Boxes', 'icon': 'fas fa-vector-square'}
        ],
        'algorithm_steps': [
            {
                'title': 'Image Preprocessing',
                'description': 'The input frame is resized to 640x640 pixels and normalized. Letterboxing is applied to maintain aspect ratio while fitting the model input dimensions.',
                'details': ['Resize with letterboxing to 640x640', 'Normalize pixel values to [0,1]', 'Convert BGR to RGB color space', 'Create batch tensor for GPU processing']
            },
            {
                'title': 'Feature Extraction (CSPDarknet Backbone)',
                'description': 'The backbone network extracts hierarchical features using Cross-Stage Partial connections. This architecture reduces computation while maintaining representational power.',
                'details': ['CSPDarknet53 backbone architecture', 'Cross-Stage Partial connections reduce parameters by 20%', 'SiLU activation functions for smooth gradients', 'Outputs feature maps at 3 scales (P3, P4, P5)']
            },
            {
                'title': 'Feature Pyramid Network (PANet)',
                'description': 'Path Aggregation Network combines features from different scales, enabling detection of objects of varying sizes from small (8px) to large (640px).',
                'details': ['Bottom-up and top-down feature aggregation', 'Detects small objects using high-resolution features', 'Detects large objects using semantic-rich features', 'Produces 3 detection heads for multi-scale detection']
            },
            {
                'title': 'Detection Head & Predictions',
                'description': 'Anchor-free detection heads predict bounding boxes, class probabilities, and objectness scores for each grid cell. YOLOv8 uses decoupled heads for classification and localization.',
                'details': ['Predicts (x, y, w, h) bounding box coordinates', 'Outputs 80 class probability scores', 'Objectness score indicates detection confidence', 'Multiple predictions per grid cell']
            },
            {
                'title': 'Non-Maximum Suppression (NMS)',
                'description': 'Overlapping detections are filtered using NMS to keep only the most confident predictions. IoU threshold of 0.45 removes duplicate boxes.',
                'details': ['Sort predictions by confidence score', 'Calculate IoU between overlapping boxes', 'Suppress boxes with IoU > 0.45', 'Return final filtered detections']
            }
        ],
        'algorithm_details': [
            'Model: YOLOv8n/s/m/l/x (scalable complexity)',
            'Backbone: CSPDarknet with C2f modules',
            'Neck: PANet for multi-scale feature fusion',
            'Head: Decoupled anchor-free detection',
            'Loss: BCE + DFL + CIoU combined loss',
            'Training: COCO 2017 dataset (330K images)'
        ],
        'config_details': [
            'Input Size: 640x640 pixels',
            'Confidence Threshold: 0.5 (configurable)',
            'NMS IoU Threshold: 0.45',
            'Max Detections: 300 per frame',
            'Device: CUDA (GPU) or CPU fallback',
            'Batch Size: 1 for real-time streaming'
        ],
        'code_example': '''# YOLOv8 Object Detection
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # nano model for speed

# Run inference on frame
results = model(frame, conf=0.5, iou=0.45)

# Process detections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # bounding box
    conf = box.conf[0]            # confidence
    cls = int(box.cls[0])         # class ID
    label = model.names[cls]      # class name''',
        'tech_stack': [
            {'icon': '📦', 'name': 'Ultralytics YOLOv8', 'description': 'State-of-the-art object detection framework'},
            {'icon': '🔥', 'name': 'PyTorch', 'description': 'Deep learning backend with CUDA support'},
            {'icon': '👁️', 'name': 'OpenCV', 'description': 'Image preprocessing and visualization'},
            {'icon': '🎮', 'name': 'CUDA', 'description': 'GPU acceleration for real-time inference'}
        ],
        'use_cases': [
            {'icon': 'fas fa-users', 'title': 'Crowd Monitoring', 'description': 'Count and track people in public spaces for capacity management.'},
            {'icon': 'fas fa-car', 'title': 'Vehicle Detection', 'description': 'Detect cars, trucks, bikes in parking lots and traffic monitoring.'},
            {'icon': 'fas fa-paw', 'title': 'Animal Detection', 'description': 'Identify animals for wildlife monitoring or pet detection.'}
        ],
        'related_features': [
            {'slug': 'object-tracking', 'title': 'Object Tracking', 'icon': 'fas fa-route'},
            {'slug': 'facial-recognition', 'title': 'Facial Recognition', 'icon': 'fas fa-user-check'},
            {'slug': 'behavior-analysis', 'title': 'Behavior Analysis', 'icon': 'fas fa-brain'}
        ]
    },
    
    'object-tracking': {
        'title': 'Object Tracking',
        'slug': 'object-tracking',
        'icon': 'fas fa-route',
        'gradient': 'linear-gradient(135deg, #f093fb, #f5576c)',
        'description': 'Advanced multi-object tracking using Deep SORT algorithm with Kalman filtering and deep appearance features.',
        'tags': ['Deep SORT', 'Kalman Filter', 'ReID', 'Multi-Object Tracking', 'Real-time'],
        'overview': '''Object tracking maintains consistent identities for detected objects across video frames. Our system uses Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric), which combines motion prediction using Kalman filters with deep learning-based appearance features. This enables tracking even through brief occlusions, camera motion, and crowded scenes. Each tracked object receives a unique ID that persists throughout its visibility in the video.''',
        'metrics': [
            {'value': '95%', 'label': 'MOTA Score'},
            {'value': '100+', 'label': 'Simultaneous Tracks'},
            {'value': '30+', 'label': 'FPS'},
            {'value': '<20ms', 'label': 'Per Frame'}
        ],
        'pipeline': [
            {'name': 'Detection', 'icon': 'fas fa-cube'},
            {'name': 'Feature Extract', 'icon': 'fas fa-fingerprint'},
            {'name': 'Kalman Predict', 'icon': 'fas fa-chart-line'},
            {'name': 'Hungarian Match', 'icon': 'fas fa-link'},
            {'name': 'Track Update', 'icon': 'fas fa-sync'}
        ],
        'algorithm_steps': [
            {
                'title': 'Detection Input',
                'description': 'Object detections from YOLO (bounding boxes, class labels, confidence scores) are passed to the tracker each frame.',
                'details': ['Receive detections from YOLOv8', 'Filter by confidence threshold (>0.5)', 'Extract bounding box coordinates', 'Associate class labels with detections']
            },
            {
                'title': 'Appearance Feature Extraction',
                'description': 'A deep CNN extracts 128-dimensional appearance features from each detection. These features capture visual characteristics like color, texture, and shape.',
                'details': ['ResNet-based feature extractor', 'Produces 128D L2-normalized vector', 'Trained on pedestrian re-identification datasets', 'Features are robust to pose and lighting changes']
            },
            {
                'title': 'Kalman Filter Prediction',
                'description': 'For each existing track, the Kalman filter predicts the expected position in the current frame based on motion history.',
                'details': ['State: [x, y, aspect_ratio, height, velocities]', 'Constant velocity motion model', 'Predicts bounding box for next frame', 'Accounts for uncertainty in motion']
            },
            {
                'title': 'Data Association (Hungarian Algorithm)',
                'description': 'Detections are matched to predicted track positions using a cost matrix that combines motion distance (Mahalanobis) and appearance similarity (cosine distance).',
                'details': ['Mahalanobis distance for motion cost', 'Cosine distance for appearance cost', 'Hungarian algorithm finds optimal assignment', 'Gating rejects unlikely associations']
            },
            {
                'title': 'Track Management',
                'description': 'Tracks are updated with matched detections, new tracks are created for unmatched detections, and stale tracks are deleted after a timeout.',
                'details': ['Update matched tracks with Kalman filter', 'Initialize new tracks for unmatched detections', 'Tentative tracks need 3 consecutive matches', 'Delete tracks after 30 frames without match']
            }
        ],
        'algorithm_details': [
            'Algorithm: Deep SORT (Simple Online Realtime Tracking)',
            'Motion Model: Kalman Filter with constant velocity',
            'Appearance: 128D CNN features (ResNet backbone)',
            'Association: Hungarian algorithm with cascade matching',
            'Distance: Weighted Mahalanobis + Cosine similarity',
            'Training: Market-1501, MARS pedestrian datasets'
        ],
        'config_details': [
            'Max Age: 30 frames (track deletion threshold)',
            'N_init: 3 (confirmations needed for new track)',
            'Max IoU Distance: 0.7',
            'Max Cosine Distance: 0.2',
            'NN Budget: 100 (appearance feature memory)',
            'Lambda: 0.98 (appearance vs motion weight)'
        ],
        'code_example': '''# Deep SORT Object Tracking
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize tracker
tracker = DeepSort(
    max_age=30,           # Frames before track deletion
    n_init=3,             # Detections to confirm track
    max_cosine_distance=0.2,
    nn_budget=100
)

# Process each frame
detections = detector.detect(frame)  # From YOLO

# Format: [[x1,y1,x2,y2,conf,class], ...]
bboxes = [[*d['bbox'], d['confidence'], d['class_id']] for d in detections]

# Update tracker
tracks = tracker.update_tracks(bboxes, frame=frame)

# Get confirmed tracks
for track in tracks:
    if not track.is_confirmed():
        continue
    track_id = track.track_id
    bbox = track.to_ltrb()  # [left, top, right, bottom]
    print(f"Track {track_id}: {bbox}")''',
        'tech_stack': [
            {'icon': '🔄', 'name': 'Deep SORT', 'description': 'State-of-the-art multi-object tracker'},
            {'icon': '📊', 'name': 'Kalman Filter', 'description': 'Motion prediction and smoothing'},
            {'icon': '🧠', 'name': 'ResNet Features', 'description': 'Deep appearance embeddings'},
            {'icon': '🔗', 'name': 'Hungarian Algorithm', 'description': 'Optimal detection-track matching'}
        ],
        'use_cases': [
            {'icon': 'fas fa-users', 'title': 'People Counting', 'description': 'Count unique individuals entering/exiting areas.'},
            {'icon': 'fas fa-car', 'title': 'Traffic Analysis', 'description': 'Track vehicle movements and traffic patterns.'},
            {'icon': 'fas fa-stopwatch', 'title': 'Dwell Time', 'description': 'Measure how long objects remain in specific zones.'}
        ],
        'related_features': [
            {'slug': 'object-detection', 'title': 'Object Detection', 'icon': 'fas fa-cube'},
            {'slug': 'person-reid', 'title': 'Person Re-ID', 'icon': 'fas fa-user-friends'},
            {'slug': 'behavior-analysis', 'title': 'Behavior Analysis', 'icon': 'fas fa-brain'}
        ]
    },
    
    'facial-recognition': {
        'title': 'Facial Recognition',
        'slug': 'facial-recognition',
        'icon': 'fas fa-user-check',
        'gradient': 'linear-gradient(135deg, #4facfe, #00f2fe)',
        'description': 'Deep learning-based face detection and recognition using FaceNet embeddings for accurate person identification.',
        'tags': ['FaceNet', 'Deep Learning', '128D Embeddings', 'Siamese Networks', 'Triplet Loss'],
        'overview': '''Our facial recognition system uses a two-stage approach: first detecting faces using HOG (Histogram of Oriented Gradients) or CNN detectors, then extracting 128-dimensional face embeddings using a FaceNet-inspired architecture. These embeddings are compared against a database of known faces using cosine similarity. The system supports dynamic enrollment of new faces and maintains recognition accuracy even with varying lighting, angles, and partial occlusions.''',
        'metrics': [
            {'value': '99.6%', 'label': 'Accuracy (LFW)'},
            {'value': '128D', 'label': 'Embedding Size'},
            {'value': '0.6', 'label': 'Tolerance'},
            {'value': '<100ms', 'label': 'Per Face'}
        ],
        'pipeline': [
            {'name': 'Frame Input', 'icon': 'fas fa-image'},
            {'name': 'Face Detection', 'icon': 'fas fa-search'},
            {'name': 'Alignment', 'icon': 'fas fa-arrows-alt'},
            {'name': 'Embedding', 'icon': 'fas fa-fingerprint'},
            {'name': 'Comparison', 'icon': 'fas fa-balance-scale'},
            {'name': 'Identity', 'icon': 'fas fa-user-check'}
        ],
        'algorithm_steps': [
            {
                'title': 'Face Detection (HOG/CNN)',
                'description': 'Faces are detected using either HOG (fast, CPU-based) or CNN (accurate, GPU-based) detector. The detector returns bounding boxes for all faces in the frame.',
                'details': ['HOG: Histogram of Oriented Gradients + SVM classifier', 'CNN: Deep learning detector with higher accuracy', 'Returns face locations as (top, right, bottom, left)', 'Handles multiple faces simultaneously']
            },
            {
                'title': 'Face Alignment & Normalization',
                'description': 'Detected faces are aligned to a canonical pose using facial landmarks (eyes, nose, mouth). This normalization improves recognition accuracy across different head poses.',
                'details': ['68-point facial landmark detection', 'Affine transformation for alignment', 'Normalize to 160x160 pixel face crop', 'Apply histogram equalization for lighting']
            },
            {
                'title': 'Face Embedding Extraction',
                'description': 'A deep neural network (based on InceptionResNetV1) maps the aligned face to a 128-dimensional vector. This embedding captures the unique facial features in a compact representation.',
                'details': ['InceptionResNetV1 architecture', 'Trained with triplet loss function', 'Produces 128-dimensional L2-normalized vector', 'Embeddings are invariant to pose, lighting, expression']
            },
            {
                'title': 'Database Comparison',
                'description': 'The extracted embedding is compared against all known face embeddings using Euclidean distance. Faces with distance below the tolerance threshold (0.6) are considered matches.',
                'details': ['Calculate Euclidean distance to all known faces', 'Tolerance threshold: 0.6 (configurable)', 'Lower distance = higher similarity', 'Return closest match or "Unknown"']
            },
            {
                'title': 'Identity Assignment & Tracking',
                'description': 'Matched faces are labeled with the person\'s name. Unknown faces are tracked separately and can be enrolled into the database for future recognition.',
                'details': ['Assign name from database match', 'Track unknown faces for potential enrollment', 'Update last-seen timestamp for known persons', 'Generate alerts for unknown individuals']
            }
        ],
        'algorithm_details': [
            'Detection: dlib HOG/CNN face detector',
            'Landmarks: 68-point facial landmark model',
            'Embedding: 128-dimensional FaceNet-style vectors',
            'Distance Metric: Euclidean (L2) distance',
            'Training: Triplet loss on VGGFace2 dataset',
            'Accuracy: 99.6% on LFW benchmark'
        ],
        'config_details': [
            'Tolerance: 0.6 (lower = stricter matching)',
            'Model: "hog" (fast) or "cnn" (accurate)',
            'Jitters: 1 (re-samples for better embeddings)',
            'Database: SQLite for face metadata',
            'Storage: Pickle file for face encodings',
            'Known Faces Directory: known_faces/'
        ],
        'code_example': '''# Facial Recognition with face_recognition library
import face_recognition
import numpy as np

# Load and encode known face
known_image = face_recognition.load_image_file("person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Detect faces in frame
face_locations = face_recognition.face_locations(frame, model="hog")
face_encodings = face_recognition.face_encodings(frame, face_locations)

# Compare with known faces
for encoding in face_encodings:
    distances = face_recognition.face_distance([known_encoding], encoding)
    if distances[0] < 0.6:  # Match found
        print("Known person detected!")''',
        'tech_stack': [
            {'icon': '👤', 'name': 'face_recognition', 'description': 'High-level facial recognition library'},
            {'icon': '🔍', 'name': 'dlib', 'description': 'Face detection and landmark prediction'},
            {'icon': '🧠', 'name': 'FaceNet Architecture', 'description': '128D embedding neural network'},
            {'icon': '🗄️', 'name': 'SQLite', 'description': 'Face metadata database storage'}
        ],
        'use_cases': [
            {'icon': 'fas fa-door-open', 'title': 'Access Control', 'description': 'Identify authorized personnel for secure entry.'},
            {'icon': 'fas fa-user-secret', 'title': 'Intruder Detection', 'description': 'Alert on unknown faces in restricted areas.'},
            {'icon': 'fas fa-clock', 'title': 'Attendance Tracking', 'description': 'Automatic attendance logging for employees/students.'}
        ],
        'related_features': [
            {'slug': 'person-reid', 'title': 'Person Re-ID', 'icon': 'fas fa-user-friends'},
            {'slug': 'behavior-analysis', 'title': 'Behavior Analysis', 'icon': 'fas fa-brain'},
            {'slug': 'object-detection', 'title': 'Object Detection', 'icon': 'fas fa-cube'}
        ]
    },
    
    'violence-detection': {
        'title': 'Violence Detection',
        'slug': 'violence-detection',
        'icon': 'fas fa-fist-raised',
        'gradient': 'linear-gradient(135deg, #dc3545, #ff6b6b)',
        'description': 'CNN + LSTM based model for detecting violent activities like fighting and assault in real-time video streams.',
        'tags': ['CNN', 'LSTM', 'Action Recognition', 'Temporal Analysis', 'Safety'],
        'overview': '''Violence detection uses a hybrid CNN-LSTM architecture that analyzes both spatial features (what objects/poses are in the frame) and temporal features (how things change over time). The CNN extracts visual features from individual frames, while the LSTM processes sequences of these features to understand motion patterns characteristic of violent behavior. This enables detection of fighting, punching, kicking, and other aggressive actions with high accuracy.''',
        'metrics': [
            {'value': '92%', 'label': 'Accuracy'},
            {'value': '16', 'label': 'Frame Sequence'},
            {'value': '<200ms', 'label': 'Detection Time'},
            {'value': '95%', 'label': 'Recall'}
        ],
        'pipeline': [
            {'name': 'Frame Sequence', 'icon': 'fas fa-film'},
            {'name': 'CNN Features', 'icon': 'fas fa-brain'},
            {'name': 'LSTM Analysis', 'icon': 'fas fa-stream'},
            {'name': 'Classification', 'icon': 'fas fa-tags'},
            {'name': 'Alert', 'icon': 'fas fa-bell'}
        ],
        'algorithm_steps': [
            {
                'title': 'Frame Sequence Collection',
                'description': 'A sliding window of 16 consecutive frames (about 0.5 seconds at 30 FPS) is maintained. This temporal context is essential for understanding actions.',
                'details': ['Buffer of 16 consecutive frames', 'Sliding window with 50% overlap', 'Frames resized to 224x224 pixels', 'Normalized using ImageNet statistics']
            },
            {
                'title': 'Spatial Feature Extraction (CNN)',
                'description': 'Each frame is passed through a pre-trained CNN (ResNet or MobileNet) to extract visual features. These features encode objects, poses, and scene context.',
                'details': ['ResNet50 or MobileNetV2 backbone', 'Extract 2048D feature vector per frame', 'Transfer learning from ImageNet', 'Optional: Pose estimation features']
            },
            {
                'title': 'Temporal Modeling (LSTM)',
                'description': 'The sequence of CNN features is processed by an LSTM network that learns temporal dependencies. This captures motion patterns like punching, falling, or running.',
                'details': ['2-layer LSTM with 512 hidden units', 'Processes sequence of 16 frame features', 'Learns patterns like "wind-up, strike, fall"', 'Bidirectional LSTM for better context']
            },
            {
                'title': 'Violence Classification',
                'description': 'The LSTM output is passed through a fully connected layer with sigmoid activation to classify the sequence as violent or non-violent.',
                'details': ['Binary classification: Violence vs Normal', 'Confidence threshold: 0.7', 'Outputs probability score 0-1', 'Can be extended to multi-class (fight, fall, accident)']
            },
            {
                'title': 'Alert Generation',
                'description': 'If violence is detected above the confidence threshold, an immediate high-priority alert is generated with a snapshot and timestamp.',
                'details': ['Generate HIGH priority alert', 'Capture snapshot of incident', 'Log to database with timestamp', 'Optional: Audio alarm, SMS notification']
            }
        ],
        'algorithm_details': [
            'Architecture: CNN (spatial) + LSTM (temporal)',
            'CNN Backbone: ResNet50 or MobileNetV2',
            'LSTM: 2 layers, 512 hidden units, bidirectional',
            'Training Dataset: RWF-2000, Hockey Fight, Movies',
            'Loss Function: Binary Cross-Entropy',
            'Data Augmentation: Random crop, flip, temporal jitter'
        ],
        'config_details': [
            'Sequence Length: 16 frames',
            'Frame Rate: 30 FPS (0.5 sec analysis window)',
            'Confidence Threshold: 0.7',
            'Cooldown: 30 seconds between alerts',
            'Alert Priority: HIGH (immediate notification)',
            'GPU Recommended: For real-time processing'
        ],
        'code_example': '''# Violence Detection with CNN-LSTM
import torch
import torch.nn as nn
from torchvision.models import resnet50

class ViolenceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN backbone
        resnet = resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(2048, 512, 2, bidirectional=True, batch_first=True)
        
        # Classification head
        self.fc = nn.Linear(1024, 1)
    
    def forward(self, x):  # x: (batch, 16, 3, 224, 224)
        features = [self.cnn(x[:, i]) for i in range(16)]
        features = torch.stack(features, dim=1).squeeze()
        _, (h_n, _) = self.lstm(features)
        out = self.fc(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return torch.sigmoid(out)''',
        'tech_stack': [
            {'icon': '🔥', 'name': 'PyTorch', 'description': 'Deep learning framework'},
            {'icon': '📊', 'name': 'ResNet/MobileNet', 'description': 'Pre-trained CNN backbones'},
            {'icon': '🔄', 'name': 'LSTM', 'description': 'Long Short-Term Memory for sequences'},
            {'icon': '🎥', 'name': 'OpenCV', 'description': 'Video frame processing'}
        ],
        'use_cases': [
            {'icon': 'fas fa-school', 'title': 'School Safety', 'description': 'Detect fights in school premises for immediate response.'},
            {'icon': 'fas fa-train', 'title': 'Public Transport', 'description': 'Monitor metro stations and buses for violent incidents.'},
            {'icon': 'fas fa-hospital', 'title': 'Healthcare', 'description': 'Detect aggressive behavior in hospitals and care facilities.'}
        ],
        'related_features': [
            {'slug': 'weapon-detection', 'title': 'Weapon Detection', 'icon': 'fas fa-gun'},
            {'slug': 'behavior-analysis', 'title': 'Behavior Analysis', 'icon': 'fas fa-brain'},
            {'slug': 'anomaly-detection', 'title': 'Anomaly Detection', 'icon': 'fas fa-exclamation-triangle'}
        ]
    },
    
    'weapon-detection': {
        'title': 'Weapon Detection',
        'slug': 'weapon-detection',
        'icon': 'fas fa-gun',
        'gradient': 'linear-gradient(135deg, #343a40, #6c757d)',
        'description': 'Specialized YOLO model fine-tuned for detecting firearms, knives, and other weapons in real-time.',
        'tags': ['YOLOv8', 'Custom Training', 'Firearms', 'Security', 'Critical Alerts'],
        'overview': '''Weapon detection uses a custom-trained YOLOv8 model specifically fine-tuned on weapon datasets. The model can detect various weapon types including handguns, rifles, knives, and blunt weapons. Due to the critical nature of weapon detection, the system is tuned for high recall (detecting all weapons) even at the cost of some false positives. Detection triggers immediate high-priority alerts with visual and audio notifications.''',
        'metrics': [
            {'value': '94%', 'label': 'mAP'},
            {'value': '5', 'label': 'Weapon Classes'},
            {'value': 'HIGH', 'label': 'Alert Priority'},
            {'value': '<100ms', 'label': 'Latency'}
        ],
        'pipeline': [
            {'name': 'Frame', 'icon': 'fas fa-image'},
            {'name': 'YOLO Inference', 'icon': 'fas fa-brain'},
            {'name': 'Weapon Detection', 'icon': 'fas fa-crosshairs'},
            {'name': 'Verification', 'icon': 'fas fa-check-double'},
            {'name': 'CRITICAL Alert', 'icon': 'fas fa-exclamation-circle'}
        ],
        'algorithm_steps': [
            {
                'title': 'Model Architecture',
                'description': 'YOLOv8 model fine-tuned on weapon-specific datasets. The model is optimized for detecting partially visible or concealed weapons.',
                'details': ['YOLOv8s or YOLOv8m for balance of speed/accuracy', 'Custom head for weapon classes', 'Trained on 10,000+ weapon images', 'Augmentation: occlusion, varying lighting']
            },
            {
                'title': 'Multi-Frame Verification',
                'description': 'To reduce false positives, weapons must be detected in multiple consecutive frames before triggering an alert.',
                'details': ['Require detection in 3 of 5 frames', 'Prevents alerts from single-frame artifacts', 'Tracks weapon position across frames', 'Confidence aggregation across detections']
            },
            {
                'title': 'Alert Escalation',
                'description': 'Weapon detection triggers immediate CRITICAL priority alert with snapshot, location, and timestamp. Notifications sent via all channels.',
                'details': ['Priority: CRITICAL (highest)', 'Multi-channel notification (SMS, email, push)', 'Capture high-resolution snapshot', 'Log to database with video clip']
            }
        ],
        'algorithm_details': [
            'Model: YOLOv8 fine-tuned on weapon datasets',
            'Classes: Handgun, Rifle, Knife, Blunt Weapon, Other',
            'Training Data: COCO + Open Images + Custom',
            'Confidence Threshold: 0.4 (lower for high recall)',
            'Multi-frame verification: 3/5 frames',
            'Alert Priority: CRITICAL'
        ],
        'config_details': [
            'Model Size: YOLOv8s (faster) or YOLOv8m (accurate)',
            'Confidence: 0.4 (tuned for recall)',
            'Verification Frames: 3 of 5',
            'Alert Cooldown: 10 seconds',
            'Snapshot Quality: Original resolution',
            'Audio Alarm: Enabled by default'
        ],
        'code_example': '''# Weapon Detection
from ultralytics import YOLO

# Load custom weapon detection model
weapon_model = YOLO('weapon_detection.pt')  # Custom trained

# Run inference
results = weapon_model(frame, conf=0.4)

# Check for weapons
for box in results[0].boxes:
    cls = int(box.cls[0])
    if cls in [0, 1, 2]:  # handgun, rifle, knife
        # CRITICAL ALERT
        alert_system.trigger_critical(
            type="WEAPON_DETECTED",
            weapon=weapon_model.names[cls],
            bbox=box.xyxy[0].tolist(),
            confidence=float(box.conf[0])
        )''',
        'tech_stack': [
            {'icon': '📦', 'name': 'YOLOv8 Custom', 'description': 'Fine-tuned weapon detection model'},
            {'icon': '📊', 'name': 'Custom Dataset', 'description': '10,000+ weapon images'},
            {'icon': '🚨', 'name': 'Alert System', 'description': 'Multi-channel critical alerts'},
            {'icon': '🎮', 'name': 'CUDA', 'description': 'GPU acceleration for real-time'}
        ],
        'use_cases': [
            {'icon': 'fas fa-building', 'title': 'Building Security', 'description': 'Detect weapons at entry points and lobbies.'},
            {'icon': 'fas fa-school', 'title': 'School Safety', 'description': 'Early detection of weapons on school premises.'},
            {'icon': 'fas fa-landmark', 'title': 'Public Venues', 'description': 'Monitor stadiums, malls, and public gatherings.'}
        ],
        'related_features': [
            {'slug': 'violence-detection', 'title': 'Violence Detection', 'icon': 'fas fa-fist-raised'},
            {'slug': 'object-detection', 'title': 'Object Detection', 'icon': 'fas fa-cube'},
            {'slug': 'anomaly-detection', 'title': 'Anomaly Detection', 'icon': 'fas fa-exclamation-triangle'}
        ]
    },
    
    'fire-detection': {
        'title': 'Fire & Smoke Detection',
        'slug': 'fire-detection',
        'icon': 'fas fa-fire',
        'gradient': 'linear-gradient(135deg, #fd7e14, #ffc107)',
        'description': 'Multi-modal detection combining color analysis, motion patterns, and CNN classification for early fire and smoke detection.',
        'tags': ['Multi-Modal', 'CNN', 'Color Analysis', 'Motion Detection', 'Safety'],
        'overview': '''Fire and smoke detection employs a multi-modal approach combining traditional computer vision with deep learning. Color analysis detects fire-like hues (red, orange, yellow) in HSV color space. Motion analysis identifies the characteristic flickering of flames. A CNN classifier provides the final verification by analyzing the texture and shape of potential fire/smoke regions. This multi-stage approach minimizes false positives while ensuring early detection.''',
        'metrics': [
            {'value': '96%', 'label': 'Detection Rate'},
            {'value': '<3s', 'label': 'Response Time'},
            {'value': '2%', 'label': 'False Positive'},
            {'value': '24/7', 'label': 'Monitoring'}
        ],
        'pipeline': [
            {'name': 'Frame', 'icon': 'fas fa-image'},
            {'name': 'Color Filter', 'icon': 'fas fa-palette'},
            {'name': 'Motion Analysis', 'icon': 'fas fa-wave-square'},
            {'name': 'CNN Classify', 'icon': 'fas fa-brain'},
            {'name': 'FIRE Alert', 'icon': 'fas fa-fire-alt'}
        ],
        'algorithm_steps': [
            {
                'title': 'Color-Based Detection',
                'description': 'Fire regions are identified by their characteristic colors in HSV color space. Fire typically shows high saturation orange/red, while smoke appears as low saturation gray.',
                'details': ['Convert frame to HSV color space', 'Fire mask: H(0-30), S(100-255), V(200-255)', 'Smoke mask: Low saturation, medium value', 'Morphological operations to clean mask']
            },
            {
                'title': 'Motion & Flicker Analysis',
                'description': 'Fire exhibits characteristic flickering motion. Frame differencing and optical flow detect the oscillating patterns unique to flames.',
                'details': ['Frame differencing for motion areas', 'Analyze motion frequency (flicker rate)', 'Fire flickers at 10-15 Hz', 'Combine with color mask for candidates']
            },
            {
                'title': 'CNN Classification',
                'description': 'Candidate regions are verified by a CNN trained specifically on fire/smoke images. The CNN analyzes texture, shape, and context.',
                'details': ['MobileNetV2 backbone trained on fire dataset', 'Binary classification: Fire/Smoke vs Normal', 'Confidence threshold: 0.8', 'Processes only candidate regions (efficient)']
            },
            {
                'title': 'Alert & Response',
                'description': 'Confirmed fire/smoke triggers CRITICAL alert with location, time, and snapshots. Can integrate with building fire systems.',
                'details': ['CRITICAL priority alert', 'Multiple snapshot captures', 'Integration with fire alarm systems', 'SMS/Email to safety personnel']
            }
        ],
        'algorithm_details': [
            'Color Model: HSV color space filtering',
            'Motion: Frame differencing + frequency analysis',
            'CNN: MobileNetV2 fine-tuned on fire dataset',
            'Multi-modal fusion: Weighted voting',
            'Dataset: FIRESENSE, Smoke Detection datasets',
            'Verification: 2-stage (color + CNN)'
        ],
        'config_details': [
            'Color Threshold: Adaptive HSV ranges',
            'Flicker Detection: 10-15 Hz analysis',
            'CNN Confidence: 0.8 threshold',
            'Alert Cooldown: 60 seconds',
            'Region Minimum Size: 500 pixels',
            'Night Mode: Enhanced sensitivity'
        ],
        'code_example': '''# Fire Detection with Multi-Modal Approach
import cv2
import numpy as np

def detect_fire(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire color mask (orange/red)
    lower_fire = np.array([0, 100, 200])
    upper_fire = np.array([30, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Find contours
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            # Extract region for CNN verification
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            
            # CNN classification
            if cnn_classify_fire(roi) > 0.8:
                return True, (x, y, w, h)
    
    return False, None''',
        'tech_stack': [
            {'icon': '🎨', 'name': 'HSV Color Analysis', 'description': 'Color-based fire region detection'},
            {'icon': '📊', 'name': 'Motion Analysis', 'description': 'Flicker and movement patterns'},
            {'icon': '🧠', 'name': 'MobileNetV2', 'description': 'Efficient CNN for verification'},
            {'icon': '🚨', 'name': 'Alert Integration', 'description': 'Building fire system compatible'}
        ],
        'use_cases': [
            {'icon': 'fas fa-industry', 'title': 'Industrial Safety', 'description': 'Early detection in factories and warehouses.'},
            {'icon': 'fas fa-home', 'title': 'Residential', 'description': 'Smart home fire monitoring system.'},
            {'icon': 'fas fa-tree', 'title': 'Forest Monitoring', 'description': 'Detect wildfires in early stages.'}
        ],
        'related_features': [
            {'slug': 'anomaly-detection', 'title': 'Anomaly Detection', 'icon': 'fas fa-exclamation-triangle'},
            {'slug': 'object-detection', 'title': 'Object Detection', 'icon': 'fas fa-cube'},
            {'slug': 'violence-detection', 'title': 'Violence Detection', 'icon': 'fas fa-fist-raised'}
        ]
    },
    
    'person-reid': {
        'title': 'Person Re-Identification',
        'slug': 'person-reid',
        'icon': 'fas fa-user-friends',
        'gradient': 'linear-gradient(135deg, #fa709a, #fee140)',
        'description': 'Track individuals across multiple cameras using appearance-based deep learning features.',
        'tags': ['Deep Learning', 'ReID', 'Multi-Camera', 'Appearance Features', 'Tracking'],
        'overview': '''Person Re-Identification (ReID) enables tracking the same person across different cameras where traditional face recognition may fail (back view, distance, occlusion). The system extracts appearance features including clothing color, body shape, and walking gait using a deep neural network. These features create a unique "signature" for each person that persists across camera views, enabling continuous tracking throughout a facility.''',
        'metrics': [
            {'value': '85%', 'label': 'Rank-1 Accuracy'},
            {'value': '256D', 'label': 'Feature Vector'},
            {'value': 'Multi', 'label': 'Camera Support'},
            {'value': '<150ms', 'label': 'Per Person'}
        ],
        'pipeline': [
            {'name': 'Detect Person', 'icon': 'fas fa-user'},
            {'name': 'Crop & Align', 'icon': 'fas fa-crop'},
            {'name': 'Feature Extract', 'icon': 'fas fa-fingerprint'},
            {'name': 'Gallery Match', 'icon': 'fas fa-search'},
            {'name': 'Track Assign', 'icon': 'fas fa-link'}
        ],
        'algorithm_steps': [
            {
                'title': 'Person Detection & Cropping',
                'description': 'Persons are detected using YOLO and cropped with padding. The crop is resized to a standard size (256x128) for feature extraction.',
                'details': ['YOLO person detection (class 0)', 'Add 10% padding to bounding box', 'Resize to 256x128 pixels', 'Normalize using ImageNet statistics']
            },
            {
                'title': 'Appearance Feature Extraction',
                'description': 'A ReID-specific CNN (OSNet, ResNet-IBN, etc.) extracts a 256-dimensional appearance vector encoding clothing, body shape, and accessories.',
                'details': ['OSNet (Omni-Scale Network) backbone', 'Multi-scale feature aggregation', 'Output: 256D L2-normalized vector', 'Trained on Market-1501, DukeMTMC datasets']
            },
            {
                'title': 'Gallery Comparison',
                'description': 'The extracted feature is compared against a gallery of known person features using cosine similarity. The closest match above threshold is assigned.',
                'details': ['Cosine similarity for comparison', 'Threshold: 0.5 for same-person match', 'Gallery updated with new detections', 'Temporal consistency across frames']
            },
            {
                'title': 'Cross-Camera Tracking',
                'description': 'When a person disappears from one camera and appears in another, ReID links these detections to maintain a consistent identity across the facility.',
                'details': ['Match features across camera views', 'Handle appearance variations (lighting)', 'Maintain global person IDs', 'Track movement patterns']
            }
        ],
        'algorithm_details': [
            'Model: OSNet (Omni-Scale Network)',
            'Feature Dimension: 256D L2-normalized',
            'Training: Market-1501, DukeMTMC-reID',
            'Loss: Triplet Loss + Cross-Entropy',
            'Distance Metric: Cosine Similarity',
            'Gallery Management: FIFO with max 1000'
        ],
        'config_details': [
            'Input Size: 256x128 pixels',
            'Similarity Threshold: 0.5',
            'Gallery Size: 1000 entries max',
            'Feature Update: Exponential moving average',
            'Cross-Camera Delay: 0-30 seconds',
            'GPU: Recommended for real-time'
        ],
        'code_example': '''# Person Re-Identification
import torch
from torchreid.models import build_model
from torchreid import transforms

# Load pre-trained ReID model
model = build_model('osnet_x1_0', 256, pretrained=True)
model.eval()

# Extract features for a person crop
transform = transforms.build_transforms(256, 128)
person_crop = transform(person_image).unsqueeze(0)

with torch.no_grad():
    features = model(person_crop)  # 256D vector
    features = F.normalize(features, dim=1)

# Compare with gallery
similarities = torch.mm(features, gallery_features.T)
best_match = similarities.argmax()
if similarities[0, best_match] > 0.5:
    print(f"Matched to person {gallery_ids[best_match]}")''',
        'tech_stack': [
            {'icon': '🧠', 'name': 'OSNet', 'description': 'Omni-Scale ReID Network'},
            {'icon': '📊', 'name': 'TorchReID', 'description': 'Person ReID library'},
            {'icon': '🔗', 'name': 'Multi-Camera', 'description': 'Cross-camera tracking'},
            {'icon': '📦', 'name': 'Feature Gallery', 'description': 'Dynamic person database'}
        ],
        'use_cases': [
            {'icon': 'fas fa-building', 'title': 'Building Security', 'description': 'Track visitors across multiple floors and cameras.'},
            {'icon': 'fas fa-shopping-cart', 'title': 'Retail Analytics', 'description': 'Understand customer journeys through stores.'},
            {'icon': 'fas fa-search', 'title': 'Missing Persons', 'description': 'Search for specific individuals across camera network.'}
        ],
        'related_features': [
            {'slug': 'facial-recognition', 'title': 'Facial Recognition', 'icon': 'fas fa-user-check'},
            {'slug': 'object-tracking', 'title': 'Object Tracking', 'icon': 'fas fa-route'},
            {'slug': 'behavior-analysis', 'title': 'Behavior Analysis', 'icon': 'fas fa-brain'}
        ]
    },
    
    'license-plate': {
        'title': 'License Plate Recognition',
        'slug': 'license-plate',
        'icon': 'fas fa-car',
        'gradient': 'linear-gradient(135deg, #20c997, #198754)',
        'description': 'Automatic Number Plate Recognition (ANPR) using YOLO detection and PaddleOCR for accurate text extraction.',
        'tags': ['ANPR', 'OCR', 'PaddleOCR', 'Vehicle Tracking', 'Database Lookup'],
        'overview': '''License Plate Recognition (LPR/ANPR) combines object detection with Optical Character Recognition. First, a YOLO model detects license plates in the frame. The plate region is then processed by PaddleOCR, a state-of-the-art OCR system that can read text even from tilted, blurry, or partially obscured plates. Recognized plates can be matched against a database for vehicle identification and access control.''',
        'metrics': [
            {'value': '98%', 'label': 'Detection Rate'},
            {'value': '95%', 'label': 'OCR Accuracy'},
            {'value': 'Multi', 'label': 'Plate Formats'},
            {'value': '<200ms', 'label': 'Per Plate'}
        ],
        'pipeline': [
            {'name': 'Frame', 'icon': 'fas fa-image'},
            {'name': 'Plate Detect', 'icon': 'fas fa-search'},
            {'name': 'Crop & Enhance', 'icon': 'fas fa-crop'},
            {'name': 'OCR Read', 'icon': 'fas fa-font'},
            {'name': 'DB Lookup', 'icon': 'fas fa-database'}
        ],
        'algorithm_steps': [
            {
                'title': 'License Plate Detection',
                'description': 'A YOLO model trained on license plate datasets detects plate regions. It handles various plate sizes, colors, and orientations.',
                'details': ['YOLOv8 fine-tuned on plate dataset', 'Handles multiple plate formats', 'Detects plates at various angles', 'Works with different lighting conditions']
            },
            {
                'title': 'Image Enhancement',
                'description': 'The cropped plate region is enhanced using image processing techniques to improve OCR accuracy.',
                'details': ['Deskew tilted plates', 'Contrast enhancement (CLAHE)', 'Noise reduction (bilateral filter)', 'Resize to optimal OCR input size']
            },
            {
                'title': 'Text Recognition (OCR)',
                'description': 'PaddleOCR performs text detection and recognition on the enhanced plate image. It can handle various fonts and character types.',
                'details': ['PaddleOCR with PP-OCRv3 model', 'Character-level segmentation', 'Support for letters and numbers', 'Language-agnostic recognition']
            },
            {
                'title': 'Database Matching',
                'description': 'Recognized plate numbers are matched against a vehicle database for identification, access control, or alerting.',
                'details': ['Match against known vehicles DB', 'Auto-log entry/exit times', 'Trigger alerts for flagged plates', 'Integration with parking systems']
            }
        ],
        'algorithm_details': [
            'Detection: YOLOv8 on license plate dataset',
            'OCR: PaddleOCR PP-OCRv3',
            'Preprocessing: CLAHE, deskew, denoise',
            'Character Set: A-Z, 0-9, regional chars',
            'Database: SQLite vehicle registry',
            'Fuzzy Matching: Edit distance for OCR errors'
        ],
        'config_details': [
            'Plate Detection Conf: 0.6',
            'OCR Confidence: 0.8',
            'Supported Formats: Indian, US, EU plates',
            'Max Parallel Plates: 10',
            'Database Update: Real-time',
            'Log Retention: 30 days'
        ],
        'code_example': '''# License Plate Recognition
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize
plate_detector = YOLO('license_plate.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Detect plates
results = plate_detector(frame, conf=0.6)

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_crop = frame[y1:y2, x1:x2]
    
    # OCR
    ocr_result = ocr.ocr(plate_crop, cls=True)
    if ocr_result and ocr_result[0]:
        plate_text = ''.join([line[1][0] for line in ocr_result[0]])
        print(f"Plate: {plate_text}")
        
        # Database lookup
        vehicle = db.lookup_plate(plate_text)
        if vehicle:
            print(f"Owner: {vehicle.owner}")''',
        'tech_stack': [
            {'icon': '📦', 'name': 'YOLOv8', 'description': 'License plate detection'},
            {'icon': '📝', 'name': 'PaddleOCR', 'description': 'State-of-the-art OCR'},
            {'icon': '🗄️', 'name': 'SQLite', 'description': 'Vehicle database'},
            {'icon': '🔧', 'name': 'OpenCV', 'description': 'Image preprocessing'}
        ],
        'use_cases': [
            {'icon': 'fas fa-parking', 'title': 'Parking Systems', 'description': 'Automatic entry/exit logging for parking lots.'},
            {'icon': 'fas fa-road', 'title': 'Traffic Management', 'description': 'Monitor traffic flow and violations.'},
            {'icon': 'fas fa-shield-alt', 'title': 'Security', 'description': 'Alert on stolen or flagged vehicles.'}
        ],
        'related_features': [
            {'slug': 'object-detection', 'title': 'Object Detection', 'icon': 'fas fa-cube'},
            {'slug': 'object-tracking', 'title': 'Object Tracking', 'icon': 'fas fa-route'},
            {'slug': 'anomaly-detection', 'title': 'Anomaly Detection', 'icon': 'fas fa-exclamation-triangle'}
        ]
    }
}

def get_feature(slug):
    """Get feature data by slug."""
    return FEATURES_DATA.get(slug)

def get_all_features():
    """Get all features."""
    return FEATURES_DATA

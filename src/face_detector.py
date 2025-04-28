"""
Face Detector Module
Provides detection of faces in images using various methods (opencv, dlib, or torch).
"""

import os
import cv2
import numpy as np

# Make torch optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. GPU acceleration disabled.")
    TORCH_AVAILABLE = False

# Import dlib for facial landmark detection
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("dlib not available. Using OpenCV for detection.")
    DLIB_AVAILABLE = False

class FaceDetector:
    """Face detection using various methods with GPU support where available"""
    
    def __init__(self, method='dlib', use_gpu=True, confidence_threshold=0.5):
        """
        Initialize the face detector
        
        Args:
            method (str): Detection method ('opencv', 'dlib', or 'torch')
            use_gpu (bool): Whether to use GPU acceleration (if available)
            confidence_threshold (float): Confidence threshold for detections (0.0-1.0)
        """
        self.method = method
        self.use_gpu = use_gpu and (TORCH_AVAILABLE or method != 'torch')
        self.confidence_threshold = confidence_threshold
        
        # Absolute path to models directory
        models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        
        # Initialize the face detector based on the method
        if method == 'opencv':
            # Use OpenCV DNN face detector
            model_file = os.path.join(models_dir, 'opencv_face_detector.caffemodel')
            config_file = os.path.join(models_dir, 'opencv_face_detector.prototxt')
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
                if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                print(f"OpenCV face detector model not found at {model_file}")
                print("Using OpenCV's built-in face detector instead")
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        elif method == 'dlib' and DLIB_AVAILABLE:
            # Use dlib's face detector
            print("Using dlib face detector")
            self.detector = dlib.get_frontal_face_detector()
            
            # Load landmark predictor if available
            landmark_model = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
            if os.path.exists(landmark_model):
                self.landmark_predictor = dlib.shape_predictor(landmark_model)
                print(f"Loaded landmark model from: {landmark_model}")
            else:
                print(f"Dlib face landmark model not found at {landmark_model}")
                print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("Extract and place it in the 'models' directory")
                self.landmark_predictor = None
        
        elif method == 'torch' and TORCH_AVAILABLE:
            # Use PyTorch model (RetinaFace) for face detection
            try:
                from facenet_pytorch import MTCNN
                self.detector = MTCNN(
                    keep_all=True, 
                    device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu',
                    thresholds=[0.6, 0.7, 0.9]  # Adjust for precision vs. recall
                )
            except ImportError:
                print("facenet_pytorch not available. Falling back to OpenCV.")
                self.method = 'opencv'
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        else:
            # Fallback to OpenCV's built-in face detector
            print(f"Method {method} not available. Using OpenCV's built-in face detector.")
            self.method = 'opencv'
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect(self, image):
        """
        Detect faces in the image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of face bounding boxes as (x, y, w, h)
        """
        if self.method == 'opencv' and isinstance(self.detector, cv2.dnn.Net):
            return self._detect_opencv_dnn(image)
        elif self.method == 'opencv':
            return self._detect_opencv_cascade(image)
        elif self.method == 'dlib' and DLIB_AVAILABLE:
            return self._detect_dlib(image)
        elif self.method == 'torch' and TORCH_AVAILABLE:
            return self._detect_torch(image)
        else:
            return self._detect_opencv_cascade(image)
    
    def _detect_opencv_dnn(self, image):
        """Detect faces using OpenCV DNN"""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Ensure bounding box is within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def _detect_opencv_cascade(self, image):
        """Detect faces using OpenCV Cascade Classifier"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def _detect_dlib(self, image):
        """Detect faces using dlib"""
        # Convert to RGB for dlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        dlib_faces = self.detector(rgb_image)
        
        # Convert to OpenCV format (x, y, w, h)
        faces = []
        for face in dlib_faces:
            x = face.left()
            y = face.top()
            w = face.width()
            h = face.height()
            faces.append((x, y, w, h))
        
        return faces
    
    def _detect_torch(self, image):
        """Detect faces using PyTorch model"""
        if not TORCH_AVAILABLE:
            return self._detect_opencv_cascade(image)
        
        # Detect faces
        boxes, _ = self.detector.detect(image)
        
        # If no faces detected, return empty list
        if boxes is None:
            return []
        
        # Convert to OpenCV format (x, y, w, h)
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def get_landmarks(self, image, face):
        """
        Get facial landmarks for a detected face
        
        Args:
            image (numpy.ndarray): Input image
            face (tuple): Face bounding box as (x, y, w, h)
            
        Returns:
            list: List of landmark points as (x, y) tuples
        """
        if self.method == 'dlib' and hasattr(self, 'landmark_predictor') and self.landmark_predictor:
            # Convert to dlib rectangle
            x, y, w, h = face
            dlib_rect = dlib.rectangle(x, y, x+w, y+h)
            
            # Get landmarks
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = self.landmark_predictor(rgb_image, dlib_rect)
            
            # Convert to list of (x, y) tuples
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]
            return landmarks
        else:
            # No landmarks available
            return []
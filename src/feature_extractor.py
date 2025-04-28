"""
Feature Extractor Module
Extracts facial features from detected faces for health analysis with GPU acceleration.
"""

import cv2
import numpy as np
import os
import time
import dlib

# Make torch optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU acceleration for feature extraction disabled.")

class FeatureExtractor:
    """A class to extract facial features from detected faces with GPU acceleration"""
    
    def __init__(self, use_gpu=True):
        """
        Initialize the feature extractor with necessary models and GPU support
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and TORCH_AVAILABLE  # Only use GPU if torch is available
        
        # Check GPU availability for torch operations
        self.has_cuda = False
        if TORCH_AVAILABLE and self.use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.has_cuda = torch.cuda.is_available()
            
            if self.use_gpu and not self.has_cuda:
                print("GPU requested but CUDA is not available. Using CPU for feature extraction.")
                self.use_gpu = False
        
        # Initialize dlib's face landmark predictor
        # Use absolute path based on the file's location
        models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        predictor_path = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
        
        if os.path.exists(predictor_path):
            self.landmark_predictor = dlib.shape_predictor(predictor_path)
            self.has_landmark_detector = True
            print(f"Loaded landmark model from: {predictor_path}")
        else:
            print(f"Dlib face landmark model not found at {predictor_path}")
            print("Please ensure it is in the models directory")
            self.has_landmark_detector = False
        
        # Define regions of interest for health analysis (using dlib's 68 point model indices)
        self.regions = {
            'eyes': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
            'eyebrows': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],
            'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
            'jaw': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        }
        
        # Colors for visualization
        self.colors = {
            'eyes': (255, 0, 0),      # Blue
            'eyebrows': (0, 255, 0),  # Green
            'nose': (0, 0, 255),      # Red
            'mouth': (255, 255, 0),   # Cyan
            'jaw': (255, 0, 255),     # Magenta
        }
        
        # Performance metrics
        self.processing_time = 0
        self.fps = 0
    
    def extract_features_from_frame(self, frame, face_bbox):
        """
        Extract facial features from a video frame for real-time analysis
        
        Args:
            frame: Video frame as numpy array
            face_bbox: Face bounding box [x, y, width, height]
            
        Returns:
            dict: Extracted facial features
        """
        start_time = time.time()
        
        # Initialize features dictionary
        features = {
            'bbox': face_bbox,
            'landmarks': [],
            'metrics': {},
            'symmetry': {},
            'skin': {},
            'facial_ratios': {}
        }
        
        # If we don't have the landmark detector, just return basic features
        if not self.has_landmark_detector:
            features['skin'] = self._analyze_skin(frame, face_bbox)
            
            # Calculate processing time and FPS
            end_time = time.time()
            self.processing_time = end_time - start_time
            self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
            
            return features
        
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract face ROI
        x, y, w, h = face_bbox
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get facial landmarks
        try:
            shape = self.landmark_predictor(gray, dlib_rect)
            landmarks = []
            for i in range(68):  # dlib has 68 landmarks
                landmarks.append((shape.part(i).x, shape.part(i).y))
            
            features['landmarks'] = landmarks
            
            # Calculate facial metrics - use GPU if available
            if self.has_cuda:
                features['metrics'] = self._calculate_metrics_gpu(landmarks)
            else:
                features['metrics'] = self._calculate_metrics(landmarks)
            
            # Calculate facial symmetry
            features['symmetry'] = self._calculate_symmetry(landmarks)
            
            # Extract skin features
            features['skin'] = self._analyze_skin(frame, face_bbox)
            
            # Calculate facial ratios (golden ratio analysis)
            features['facial_ratios'] = self._calculate_facial_ratios(landmarks)
        except Exception as e:
            print(f"Error extracting facial features: {e}")
        
        # Calculate processing time and FPS
        end_time = time.time()
        self.processing_time = end_time - start_time
        self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
        
        return features
    
    def extract_features(self, image_path, face_bbox):
        """
        Extract facial features from an image file
        
        Args:
            image_path (str): Path to the image file
            face_bbox (list): Face bounding box [x, y, width, height]
            
        Returns:
            dict: Dictionary of extracted facial features
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Use the frame processing function for consistency
        return self.extract_features_from_frame(image, face_bbox)
    
    def _calculate_metrics_gpu(self, landmarks):
        """Calculate facial metrics using GPU acceleration"""
        metrics = {}
        
        if len(landmarks) < 10 or not TORCH_AVAILABLE:
            # Fall back to CPU implementation if torch is not available
            return self._calculate_metrics(landmarks)
        
        # Convert landmarks to PyTorch tensors
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32, device=self.device)
        
        # Eye measurements
        # Left eye (landmarks 36-41)
        left_eye_width = torch.norm(
            landmarks_tensor[36] - landmarks_tensor[39]
        )
        
        # Right eye (landmarks 42-47)
        right_eye_width = torch.norm(
            landmarks_tensor[42] - landmarks_tensor[45]
        )
        
        metrics['left_eye_width'] = left_eye_width.item()
        metrics['right_eye_width'] = right_eye_width.item()
        metrics['eye_width_ratio'] = (left_eye_width / right_eye_width).item() if right_eye_width > 0 else 0
        
        # Face width and height
        # Face width: distance between temples (landmarks 0 and 16)
        face_width = torch.norm(
            landmarks_tensor[16] - landmarks_tensor[0]
        )
        
        # Face height: distance from chin to forehead (landmarks 8 and 27)
        face_height = torch.norm(
            landmarks_tensor[8] - landmarks_tensor[27]
        )
        
        metrics['face_width'] = face_width.item()
        metrics['face_height'] = face_height.item()
        metrics['face_width_height_ratio'] = (face_width / face_height).item() if face_height > 0 else 0
        
        return metrics
    
    def _calculate_metrics(self, landmarks):
        """Calculate key facial metrics from landmarks (CPU version)"""
        metrics = {}
        
        # Organize landmarks by facial regions
        regions_landmarks = {}
        for region, indices in self.regions.items():
            regions_landmarks[region] = [landmarks[i] for i in indices if i < len(landmarks)]
        
        # Eye measurements
        if 'eyes' in regions_landmarks and len(regions_landmarks['eyes']) >= 6:
            # Left eye width (landmarks 36 and 39)
            left_eye_width = np.linalg.norm(
                np.array(landmarks[36]) - np.array(landmarks[39])
            )
            
            # Right eye width (landmarks 42 and 45)
            right_eye_width = np.linalg.norm(
                np.array(landmarks[42]) - np.array(landmarks[45])
            )
            
            metrics['left_eye_width'] = float(left_eye_width)
            metrics['right_eye_width'] = float(right_eye_width)
            metrics['eye_width_ratio'] = float(left_eye_width / right_eye_width if right_eye_width > 0 else 0)
        
        # Face width and height
        if len(landmarks) >= 17:
            # Face width: distance between temples (landmarks 0 and 16)
            face_width = np.linalg.norm(
                np.array(landmarks[16]) - np.array(landmarks[0])
            )
            
            # Face height: distance from chin to forehead (landmarks 8 and 27)
            face_height = np.linalg.norm(
                np.array(landmarks[8]) - np.array(landmarks[27])
            )
            
            metrics['face_width'] = float(face_width)
            metrics['face_height'] = float(face_height)
            metrics['face_width_height_ratio'] = float(face_width / face_height if face_height > 0 else 0)
        
        return metrics
    
    def _calculate_symmetry(self, landmarks):
        """Calculate facial symmetry metrics"""
        symmetry = {
            'eyes_level': 1.0,
            'overall_symmetry': 1.0
        }
        
        if len(landmarks) >= 68:  # Full set of dlib landmarks
            # Get vertical positions of the eyes
            left_eye_y = landmarks[37][1]  # Left eye upper point
            right_eye_y = landmarks[44][1]  # Right eye upper point
            
            # Calculate eye level difference (normalized by face height)
            face_height = landmarks[8][1] - landmarks[27][1]  # Chin to nose bridge
            if face_height > 0:
                eye_level_diff = abs(left_eye_y - right_eye_y) / face_height
                symmetry['eyes_level'] = 1.0 - min(1.0, eye_level_diff * 10)
            
            # Calculate overall symmetry by comparing left and right sides
            # Define the midpoint of the face (vertical line through nose)
            nose_tip = landmarks[30]  # Nose tip
            
            # Define pairs of landmarks to compare (left and right side of face)
            # Reduced number of pairs for real-time performance
            landmark_pairs = [
                (36, 45),    # Eyes outer corners
                (48, 54),    # Mouth corners
                (21, 22),    # Eyebrows
                (31, 35)     # Nose
            ]
            
            # Calculate asymmetry score
            asymmetry_score = 0
            for left_idx, right_idx in landmark_pairs:
                left_point = landmarks[left_idx]
                right_point = landmarks[right_idx]
                
                # Reflect right point across the vertical line through nose_tip
                reflected_right = (2 * nose_tip[0] - right_point[0], right_point[1])
                
                # Distance between left point and reflected right point
                distance = np.linalg.norm(np.array(left_point) - np.array(reflected_right))
                
                # Normalize by face width
                face_width = np.linalg.norm(np.array(landmarks[16]) - np.array(landmarks[0]))
                if face_width > 0:
                    asymmetry_score += distance / face_width
            
            # Average and convert to symmetry value (1.0 = perfect symmetry)
            if landmark_pairs:
                asymmetry_score /= len(landmark_pairs)
                symmetry['overall_symmetry'] = max(0.0, 1.0 - min(1.0, asymmetry_score * 3))
        
        return symmetry
    
    def _analyze_skin(self, image, face_bbox):
        """Analyze skin features in the face region - optimized for real-time"""
        x, y, w, h = face_bbox
        
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]
        
        # Skip processing for very small regions to prevent errors
        if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            return {'skin_tone': {'hue': 0, 'saturation': 0, 'value': 0}, 'texture': 0}
        
        # Convert to different color spaces for analysis
        face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Extract skin tone (average hue and saturation in HSV)
        h_channel, s_channel, v_channel = cv2.split(face_hsv)
        
        # For real-time performance, sample rather than process entire image
        sample_step = max(1, min(face_roi.shape[0], face_roi.shape[1]) // 30)
        h_sampled = h_channel[::sample_step, ::sample_step]
        s_sampled = s_channel[::sample_step, ::sample_step]
        v_sampled = v_channel[::sample_step, ::sample_step]
        
        avg_hue = np.mean(h_sampled)
        avg_sat = np.mean(s_sampled)
        avg_val = np.mean(v_sampled)
        
        # Detect skin texture features - downsample for speed
        face_gray_small = cv2.resize(face_gray, (0, 0), fx=0.5, fy=0.5)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(face_gray_small, (5, 5), 0)
        
        # Use Laplacian for edge detection (texture)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Basic skin conditions check
        skin_data = {
            'skin_tone': {
                'hue': float(avg_hue),
                'saturation': float(avg_sat),
                'value': float(avg_val)
            },
            'texture': float(texture_variance),
        }
        
        return skin_data
    
    def _calculate_facial_ratios(self, landmarks):
        """Calculate golden ratio and other important facial ratios - optimized for real-time"""
        ratios = {}
        
        if len(landmarks) >= 68:  # Full set of dlib landmarks
            # Calculate key facial distances
            
            # Vertical thirds (forehead, nose, lower face)
            # Note: dlib doesn't detect hairline, use eyebrow top as approximation
            eyebrow_to_nose = np.linalg.norm(np.array(landmarks[21]) - np.array(landmarks[27]))
            nose_to_mouth = np.linalg.norm(np.array(landmarks[27]) - np.array(landmarks[51]))
            mouth_to_chin = np.linalg.norm(np.array(landmarks[51]) - np.array(landmarks[8]))
            
            # The golden ratio is approximately 1.618
            golden_ratio = 1.618
            
            # Compare to golden ratio
            if nose_to_mouth > 0:
                top_ratio = eyebrow_to_nose / nose_to_mouth
                ratios['top_third_ratio'] = float(top_ratio)
                ratios['top_golden_ratio_diff'] = float(abs(top_ratio - golden_ratio))
            
            if mouth_to_chin > 0:
                middle_ratio = nose_to_mouth / mouth_to_chin
                ratios['middle_third_ratio'] = float(middle_ratio)
                ratios['middle_golden_ratio_diff'] = float(abs(middle_ratio - golden_ratio))
            
            # Eye spacing ratios
            inner_eye_distance = np.linalg.norm(np.array(landmarks[39]) - np.array(landmarks[42]))
            eye_width_left = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[39]))
            eye_width_right = np.linalg.norm(np.array(landmarks[42]) - np.array(landmarks[45]))
            
            if (eye_width_left + eye_width_right) > 0:
                eye_spacing_ratio = inner_eye_distance / ((eye_width_left + eye_width_right) / 2)
                ratios['eye_spacing_ratio'] = float(eye_spacing_ratio)
        
        return ratios
        
    def get_processing_stats(self):
        """Return processing statistics for display"""
        return {
            'processing_time_ms': self.processing_time * 1000,
            'fps': self.fps,
            'gpu_enabled': self.use_gpu and self.has_cuda
        }
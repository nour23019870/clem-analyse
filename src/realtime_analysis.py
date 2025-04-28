#!/usr/bin/env python
"""
Real-Time Facial Analysis System with GPU Acceleration
This script provides real-time facial analysis using webcam or video input with GPU acceleration.
"""

import os
import cv2
import time
import argparse
import threading
import numpy as np
from datetime import datetime

# Try importing PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch available but no GPU detected. Using CPU.")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU acceleration disabled.")

from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from health_analyzer import HealthAnalyzer
from data_storage import DataStorage

class RealtimeFacialAnalyzer:
    """Real-time facial analysis system with GPU acceleration"""
    
    def __init__(self, detection_method='dlib', output_dir=None, 
                 save_format='json', use_gpu=True, camera_id=0, 
                 save_interval=10, display_landmarks=True):
        """
        Initialize the real-time facial analyzer
        
        Args:
            detection_method (str): Face detection method ('opencv', 'dlib')
            output_dir (str): Directory to save analysis results
            save_format (str): Format to save results ('json', 'csv', 'xlsx')
            use_gpu (bool): Whether to use GPU acceleration
            camera_id (int): Camera ID for webcam (usually 0 for built-in)
            save_interval (int): Interval in seconds between data saves
            display_landmarks (bool): Whether to display facial landmarks on video
        """
        # Use absolute path for output directory if one wasn't provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
            
        self.output_dir = output_dir
        self.save_format = save_format
        self.use_gpu = use_gpu
        self.camera_id = camera_id
        self.save_interval = save_interval
        self.display_landmarks = display_landmarks
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components with GPU support
        print(f"Initializing with GPU acceleration: {use_gpu}")
        self.detector = FaceDetector(method=detection_method, use_gpu=use_gpu)
        self.feature_extractor = FeatureExtractor(use_gpu=use_gpu)
        self.health_analyzer = HealthAnalyzer()
        self.storage = DataStorage()
        
        # For video processing
        self.video_capture = None
        self.frame_count = 0
        self.skip_frames = 1  # Process every Nth frame (adjust for performance)
        self.processing_fps = 0
        self.display_fps = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # Update FPS display every 1 second
        
        # For multithreaded processing
        self.processing_thread = None
        self.running = False
        self.current_frame = None
        self.processed_frame = None
        self.current_faces = []
        self.lock = threading.Lock()
        
        # For tracking a single primary face (the user)
        self.primary_face = None  # Will store the primary face for analysis
        self.primary_face_features = None
        
        # For health tracking and analysis
        self.health_history = []
        self.accumulated_data = []
        self.last_health_update = time.time()
        self.health_trends = {
            'facial_symmetry': [],
            'eye_fatigue': [],
            'skin_texture': [],
            'eyes_level_symmetry': []
        }
        
        # Health status indicators
        self.overall_health_score = 0
        self.health_status = "Analyzing..."
        self.health_recommendations = []
    
    def start(self):
        """Start real-time facial analysis"""
        # Start storage thread for real-time saving
        self.storage.start_real_time_saving(self.output_dir, self.save_format, self.save_interval)
        
        # Start video capture
        self.video_capture = cv2.VideoCapture(self.camera_id)
        
        # Check if camera opened successfully
        if not self.video_capture.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Configure camera for higher resolution if possible
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        print("Real-time facial analysis started")
        
        # Display loop (main thread)
        self._display_loop()
        
        return True
    
    def stop(self):
        """Stop real-time facial analysis"""
        self.running = False
        
        # Stop background threads
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.storage.stop_real_time_saving()
        
        # Release video capture
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("Real-time facial analysis stopped")
    
    def _processing_worker(self):
        """Background worker thread for face detection and analysis"""
        frame_times = []
        max_times = 30  # For rolling average
        face_detection_count = 0
        
        while self.running:
            # Get the current frame with thread safety
            with self.lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                
                frame = self.current_frame.copy()
            
            start_time = time.time()
            
            # Detect faces
            faces = self.detector.detect(frame)
            
            # Find primary face (largest in the frame, assumed to be the user)
            primary_face = None
            primary_face_size = 0
            
            if faces:
                # Find largest face (assumed to be the user/closest to camera)
                for face_bbox in faces:
                    x, y, w, h = face_bbox
                    face_size = w * h
                    if face_size > primary_face_size:
                        primary_face = face_bbox
                        primary_face_size = face_size
                
                # Only process the primary face
                if primary_face:
                    face_detection_count += 1
                    
                    # Extract features for the primary face
                    features = self.feature_extractor.extract_features_from_frame(frame, primary_face)
                    
                    # Analyze health indicators
                    health_data = self.health_analyzer.analyze(features)
                    
                    # Update the primary face attributes
                    self.primary_face = primary_face
                    self.primary_face_features = features
                    
                    # Update health tracking data
                    self._update_health_tracking(health_data)
                    
                    # Prepare data for storage (only store one record per save interval)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result = {
                        'timestamp': timestamp,
                        'frame_id': self.frame_count,
                        'face_id': 0,  # Always 0 for primary face
                        'features': features,
                        'health_analysis': health_data,
                        'health_status': self.health_status,
                        'health_score': self.overall_health_score,
                        'recommendations': self.health_recommendations
                    }
                    
                    # Store only one record per save interval
                    self.accumulated_data.append(result)
                    
                    # Queue the latest data for display
                    display_data = (primary_face, features, health_data)
            
            # Process the frame for display
            display_frame = frame.copy()
            
            # Add faces to current faces list with thread safety
            with self.lock:
                if faces:
                    self.current_faces = [primary_face] if primary_face else []
                else:
                    self.current_faces = []
            
            # Draw only the primary face
            if primary_face:
                x, y, w, h = primary_face
                # Use blue for primary face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw landmarks if available and requested
                if self.display_landmarks and 'landmarks' in features and features['landmarks']:
                    # Draw facial landmarks
                    for point in features['landmarks']:
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                    
                    # Add key measurements and health indicators as text
                    self._draw_health_indicators(display_frame, primary_face, health_data)
                    
                    # Add overall health status on top of the frame
                    cv2.putText(display_frame, f"Health Status: {self.health_status}", 
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display health score
                    cv2.putText(display_frame, f"Health Score: {self.overall_health_score:.1f}/10", 
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display top recommendation if available
                    if self.health_recommendations:
                        cv2.putText(display_frame, f"Recommendation: {self.health_recommendations[0]}", 
                                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate FPS
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update rolling FPS average
            frame_times.append(processing_time)
            if len(frame_times) > max_times:
                frame_times.pop(0)
            
            avg_time = sum(frame_times) / len(frame_times)
            self.processing_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Update the processed frame for display
            with self.lock:
                self.processed_frame = display_frame
            
            # Check if it's time to save data
            current_time = time.time()
            if (current_time - self.last_health_update >= self.save_interval and 
                len(self.accumulated_data) > 0):
                # Queue only the most recent data point for saving (prevents multiple faces being saved)
                self.storage.queue_data_for_saving(self.accumulated_data[-1])
                self.last_health_update = current_time
                self.accumulated_data = []  # Clear accumulated data after saving
                
    def _draw_health_indicators(self, frame, face_bbox, health_data):
        """Draw health indicators on the frame for the primary face"""
        x, y, w, h = face_bbox
        y_offset = y - 10
        
        # Display facial symmetry
        if 'facial_symmetry' in health_data:
            value = health_data['facial_symmetry']
            color = self._get_indicator_color(value, 0.7, 0.9)
            sym_text = f"Symmetry: {value:.2f}"
            cv2.putText(frame, sym_text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 20
        
        # Display eye fatigue if available
        if 'eye_fatigue' in health_data:
            fatigue = health_data['eye_fatigue']
            # Map text values to numeric for color
            fatigue_map = {"Minimal": 0.1, "Mild": 0.3, "Moderate": 0.6, "Severe": 0.9}
            fatigue_val = fatigue_map.get(fatigue, 0.5)
            color = self._get_indicator_color(1.0-fatigue_val, 0.3, 0.7)  # Invert since lower fatigue is better
            fatigue_text = f"Eye Fatigue: {fatigue}"
            cv2.putText(frame, fatigue_text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 20
            
        # Display facial fullness if available
        if 'facial_fullness' in health_data:
            value = health_data['facial_fullness']
            # For fullness, middle values are better (not too high or low)
            color = self._get_indicator_color(1.0 - abs(value-0.5)*2, 0.3, 0.7)
            fullness_text = f"Facial Fullness: {value:.2f}"
            cv2.putText(frame, fullness_text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 20
        
        # Display skin-related notes if available
        if 'skin_tone_note' in health_data:
            skin_text = f"Skin: {health_data['skin_tone_note']}"
            cv2.putText(frame, skin_text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset -= 20
        
        # Display eye bag evaluation if available
        if 'eye_bags_evaluation' in health_data:
            eyebags = health_data['eye_bags_evaluation']
            # Map text values to numeric for color
            eyebag_map = {"Minimal": 0.1, "Minor": 0.3, "Moderate": 0.6, "Severe": 0.9}
            eyebag_val = eyebag_map.get(eyebags, 0.5)
            color = self._get_indicator_color(1.0-eyebag_val, 0.3, 0.7)  # Invert since fewer eye bags is better
            eyebags_text = f"Eye Bags: {eyebags}"
            cv2.putText(frame, eyebags_text, (x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 20
        
        # Add any additional important health indicators
        for key in ['symmetry_evaluation', 'fullness_evaluation']:
            if key in health_data:
                note_text = f"{key.replace('_', ' ').title()}: {health_data[key]}"
                cv2.putText(frame, note_text, (x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset -= 20
    
    def _get_indicator_color(self, value, yellow_threshold=0.4, green_threshold=0.7):
        """Get color for health indicator based on value (0-1 scale)"""
        if value < yellow_threshold:
            # Red to yellow gradient
            red = 255
            green = int(255 * (value / yellow_threshold))
            blue = 0
        elif value < green_threshold:
            # Yellow to green gradient
            factor = (value - yellow_threshold) / (green_threshold - yellow_threshold)
            red = int(255 * (1 - factor))
            green = 255
            blue = 0
        else:
            # Green with increasing blue for excellent values
            factor = (value - green_threshold) / (1.0 - green_threshold)
            red = 0
            green = 255
            blue = int(255 * factor)
        
        return (blue, green, red)  # BGR format for OpenCV
        
    def _update_health_tracking(self, health_data):
        """Update health tracking and generate health status"""
        # Add new data to history
        self.health_history.append(health_data)
        
        # Keep only recent history (last 100 frames)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        # Update trend tracking for key metrics
        for key in self.health_trends.keys():
            if key in health_data:
                # Convert string values to numeric if needed
                if key == 'eye_fatigue':
                    fatigue_map = {"Minimal": 0.1, "Mild": 0.3, "Moderate": 0.6, "Severe": 0.9}
                    value = fatigue_map.get(health_data[key], 0.5)
                else:
                    value = health_data[key]
                
                if isinstance(value, (int, float)):
                    self.health_trends[key].append(value)
                    if len(self.health_trends[key]) > 30:  # Keep last 30 values
                        self.health_trends[key] = self.health_trends[key][-30:]
        
        # Calculate overall health score (0-10 scale)
        self._calculate_health_score()
        
        # Generate health status and recommendations
        self._generate_health_status()
        
    def _calculate_health_score(self):
        """Calculate overall health score from tracked metrics"""
        if not self.health_history:
            return
        
        # Get most recent health data
        health = self.health_history[-1]
        score = 0
        components = 0
        
        # Facial symmetry (0-1 scale, higher is better)
        if 'facial_symmetry' in health:
            symmetry = health['facial_symmetry']
            # Scale to 0-10: good symmetry is >0.8, excellent >0.9
            symmetry_score = min(10, symmetry * 11.1)  # 0.9 symmetry -> score of 10
            score += symmetry_score
            components += 1
        
        # Eye level symmetry (0-1 scale, higher is better)
        if 'eyes_level_symmetry' in health:
            eye_sym = health['eyes_level_symmetry']
            eye_sym_score = min(10, eye_sym * 12.5)  # 0.8 eye level -> score of 10
            score += eye_sym_score
            components += 1
        
        # Eye fatigue (text, convert to 0-1 scale where lower is better)
        if 'eye_fatigue' in health:
            fatigue = health['eye_fatigue']
            fatigue_map = {"Minimal": 0.1, "Mild": 0.3, "Moderate": 0.6, "Severe": 0.9}
            fatigue_val = fatigue_map.get(fatigue, 0.5)
            # Convert to score (10 = no fatigue, 0 = severe fatigue)
            fatigue_score = (1 - fatigue_val) * 10
            score += fatigue_score
            components += 1
        
        # Skin texture (higher values may indicate roughness/problems)
        # Need to normalize based on observed range
        if 'skin_texture' in health:
            texture = health['skin_texture']
            # Normalize: assume normal range is 5-40, with lower being better
            norm_texture = max(0, min(1, (40 - texture) / 35))
            texture_score = norm_texture * 10
            score += texture_score
            components += 1
        
        # Golden ratio harmony if available
        if 'golden_ratio_harmony' in health:
            harmony = health['golden_ratio_harmony']
            # Higher is better, scale to 0-10
            harmony_score = harmony * 10
            score += harmony_score
            components += 1
        
        # Calculate average score if we have components
        self.overall_health_score = round(score / max(1, components), 1)
    
    def _generate_health_status(self):
        """Generate health status message and recommendations based on metrics"""
        if not self.health_history:
            self.health_status = "Analyzing..."
            self.health_recommendations = ["Please hold still for analysis"]
            return
        
        # Reset recommendations
        self.health_recommendations = []
        
        # Get most recent health data
        health = self.health_history[-1]
        
        # Determine overall health status based on score
        if self.overall_health_score >= 8.5:
            self.health_status = "Excellent"
        elif self.overall_health_score >= 7.0:
            self.health_status = "Good"
        elif self.overall_health_score >= 5.5:
            self.health_status = "Fair"
        elif self.overall_health_score >= 4.0:
            self.health_status = "Concerning"
        else:
            self.health_status = "Poor"
        
        # Generate specific recommendations based on metrics
        
        # Check for eye fatigue
        if 'eye_fatigue' in health:
            fatigue = health['eye_fatigue']
            if fatigue in ["Moderate", "Severe"]:
                self.health_recommendations.append("Take a break from screen time")
                self.health_recommendations.append("Apply the 20-20-20 rule (look 20ft away for 20s every 20min)")
        
        # Check for symmetry issues
        if 'facial_symmetry' in health and health['facial_symmetry'] < 0.7:
            self.health_recommendations.append("Check for sleeping position issues")
            self.health_recommendations.append("Consider facial exercises to improve muscle tone")
        
        # Check skin issues
        if 'skin_texture' in health and health['skin_texture'] > 30:
            self.health_recommendations.append("Consider hydration and skincare routine")
        
        if 'skin_tone_note' in health and "yellowish" in health['skin_tone_note'].lower():
            self.health_recommendations.append("Consider checking liver health & hydration")
        
        # Eye bags
        if 'eye_bags_evaluation' in health and health['eye_bags_evaluation'] in ["Moderate", "Severe"]:
            self.health_recommendations.append("Improve sleep quality and duration")
            self.health_recommendations.append("Consider reducing salt intake")
        
        # If high facial fullness (potential fluid retention)
        if 'facial_fullness' in health and health['facial_fullness'] > 0.9:
            self.health_recommendations.append("Monitor for fluid retention/edema")
        
        # Add generic recommendations if no specific ones
        if not self.health_recommendations:
            if self.overall_health_score < 6:
                self.health_recommendations.append("Consider consulting a healthcare professional")
            else:
                self.health_recommendations.append("Maintain healthy habits and adequate rest")
    
    def _display_loop(self):
        """Main display loop for showing processed frames"""
        display_times = []
        max_times = 30  # For rolling average
        
        while self.running:
            # Read frame from camera
            ret, frame = self.video_capture.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Update the current frame for processing with thread safety
            with self.lock:
                self.current_frame = frame
            
            # Skip display update if processed frame is not ready
            with self.lock:
                if self.processed_frame is None:
                    continue
                
                display_frame = self.processed_frame.copy()
                faces = self.current_faces.copy()
            
            # Frame counter
            self.frame_count += 1
            
            # Process only every Nth frame for performance
            if self.frame_count % self.skip_frames != 0:
                continue
            
            # Add performance stats to display
            start_display = time.time()
            
            # Add performance statistics
            cv2.putText(display_frame, f"Processing FPS: {self.processing_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Display FPS: {self.display_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add GPU/CPU info
            gpu_text = "GPU" if self.use_gpu and TORCH_AVAILABLE else "CPU"
            cv2.putText(display_frame, f"Using: {gpu_text}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display detected faces count
            cv2.putText(display_frame, f"Primary face detected: {'Yes' if faces else 'No'}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the resulting frame
            cv2.imshow('Real-time Facial Analysis', display_frame)
            
            # End display timing
            end_display = time.time()
            display_time = end_display - start_display
            
            # Update display FPS
            display_times.append(display_time)
            if len(display_times) > max_times:
                display_times.pop(0)
            
            avg_display_time = sum(display_times) / len(display_times) if display_times else 0
            self.display_fps = 1.0 / avg_display_time if avg_display_time > 0 else 0
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-time Facial Analysis with GPU Acceleration')
    parser.add_argument('--method', '-m', type=str, default='dlib',
                      choices=['opencv', 'dlib'],
                      help='Face detection method')
    # Use absolute path for output directory
    default_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    parser.add_argument('--output', '-o', type=str, default=default_output_dir,
                      help='Directory to save analysis results')
    parser.add_argument('--format', '-f', type=str, default='json',
                      choices=['json', 'csv', 'xlsx'],
                      help='Output format for storage')
    parser.add_argument('--camera', '-c', type=int, default=0,
                      help='Camera ID (usually 0 for built-in webcam)')
    parser.add_argument('--interval', '-i', type=int, default=10,
                      help='Save interval in seconds')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage instead of GPU')
    parser.add_argument('--skip-frames', '-s', type=int, default=1,
                      help='Process every Nth frame (1 = process all frames)')
    parser.add_argument('--no-landmarks', action='store_true',
                      help='Do not display facial landmarks')
    
    return parser.parse_args()

def download_resources():
    """Download required model files if they don't exist"""
    # Check for dlib face landmark predictor using absolute path
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    predictor_path = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
    
    if not os.path.exists(predictor_path):
        os.makedirs(models_dir, exist_ok=True)
        print("\nDlib face landmark model not found")
        print("Please download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(f"Extract and place it in: {os.path.abspath(models_dir)}\n")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Check for required resources
    download_resources()
    
    analyzer = RealtimeFacialAnalyzer(
        detection_method=args.method,
        output_dir=args.output,
        save_format=args.format,
        use_gpu=not args.cpu,
        camera_id=args.camera,
        save_interval=args.interval,
        display_landmarks=not args.no_landmarks
    )
    
    analyzer.skip_frames = args.skip_frames
    
    try:
        analyzer.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        analyzer.stop()
    
    print("Facial analysis complete.")

if __name__ == "__main__":
    main()
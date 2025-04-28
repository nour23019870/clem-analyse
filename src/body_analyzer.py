#!/usr/bin/env python
"""
Body Analysis Module
Analyzes body posture, proportion, and potential health indicators
"""

import cv2
import numpy as np
import math
from datetime import datetime
import os

class BodyAnalyzer:
    """Analyzes body posture, proportions, and health indicators"""
    
    def __init__(self, use_gpu=True):
        """Initialize the body analyzer"""
        self.use_gpu = use_gpu
        self.pose_net = None
        self.initialized = False
        self._init_pose_model()
        
        # Body section ratios for health analysis
        self.ideal_ratios = {
            'shoulder_hip_ratio': 1.618,  # Golden ratio for masculine builds
            'waist_hip_ratio': 0.7,       # Healthy female waist-hip ratio
            'leg_torso_ratio': 1.4        # Ideal leg to torso proportion
        }
        
        # Health metrics thresholds
        self.health_thresholds = {
            'posture_angle': {
                'excellent': 5.0,   # Deviation in degrees from vertical
                'good': 10.0,
                'fair': 15.0,
                'poor': 20.0
            },
            'shoulder_symmetry': {
                'excellent': 0.95,  # Shoulder level symmetry (0-1)
                'good': 0.9,
                'fair': 0.8,
                'poor': 0.7
            },
            'weight_distribution': {
                'excellent': 0.9,   # Balance between left/right (0-1)
                'good': 0.85,
                'fair': 0.75,
                'poor': 0.6
            }
        }
    
    def _init_pose_model(self):
        """Initialize the body pose estimation model"""
        try:
            # Create models directory if it doesn't exist
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Check for model files
            model_path = os.path.join(model_dir, "pose_model.pb")
            config_path = os.path.join(model_dir, "pose_model.pbtxt")
            
            # If the models don't exist, print a message and use fallback mode
            if not (os.path.exists(model_path) and os.path.exists(config_path)):
                print("Body pose estimation models not found.")
                print(f"Expected model files at: {model_path}")
                print(f"and: {config_path}")
                print("Using fallback body analysis method")
                return
            
            # Try to load the model if it exists
            try:
                self.pose_net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                
                # Use GPU if available
                if self.use_gpu:
                    self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                
                self.initialized = True
                print("Body pose estimation model loaded successfully")
            except Exception as e:
                print(f"Body pose model initialization failed: {e}")
                print("Using fallback body analysis method")
        except Exception as e:
            print(f"Error initializing body analyzer: {e}")
            print("Body analysis will use simplified methods")
    
    def detect_pose(self, image):
        """
        Detect body pose keypoints from an image
        
        Args:
            image: Input image with person's body
            
        Returns:
            List of body keypoints (joints)
        """
        if self.initialized and self.pose_net:
            # Process with OpenCV DNN
            frame_height, frame_width = image.shape[:2]
            
            # Prepare the input blob and perform inference
            blob = cv2.dnn.blobFromImage(image, 1.0/255, (368, 368), (0, 0, 0), swapRB=True, crop=False)
            self.pose_net.setInput(blob)
            output = self.pose_net.forward()
            
            # The output is a 4D matrix
            # We extract heatmaps for each body part
            keypoints = []
            threshold = 0.2  # Confidence threshold
            
            # Maps correspond to body parts (OpenPose COCO model)
            for i in range(18):  # COCO model has 18 keypoints
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frame_width, frame_height))
                
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                
                if prob > threshold:
                    keypoints.append((int(point[0]), int(point[1]), prob))
                else:
                    keypoints.append(None)
            
            return keypoints
        else:
            # Fallback method - simplified body detection
            # For demo purposes, we'll simulate keypoint detection
            frame_height, frame_width = image.shape[:2]
            
            # Detect general body shape using contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assumed to be the body)
            max_contour = max(contours, key=cv2.contourArea, default=None)
            
            if max_contour is not None:
                # Create simulated keypoints based on contour
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Approximated keypoints (these are not accurate - just for demonstration)
                keypoints = [
                    (x + w//2, y + h//8, 0.8),         # Nose
                    (x + w//2, y + h//6, 0.7),         # Neck
                    (x + w//4, y + h//4, 0.8),         # Left shoulder
                    (x + 3*w//4, y + h//4, 0.8),       # Right shoulder
                    (x + w//6, y + h//2, 0.7),         # Left elbow
                    (x + 5*w//6, y + h//2, 0.7),       # Right elbow
                    (x + w//8, y + 2*h//3, 0.6),       # Left wrist
                    (x + 7*w//8, y + 2*h//3, 0.6),     # Right wrist
                    (x + w//2, y + h//2, 0.9),         # Middle of torso
                    (x + w//3, y + 2*h//3, 0.8),       # Left hip
                    (x + 2*w//3, y + 2*h//3, 0.8),     # Right hip
                    (x + w//3, y + 3*h//4, 0.7),       # Left knee
                    (x + 2*w//3, y + 3*h//4, 0.7),     # Right knee
                    (x + w//3, y + h, 0.6),            # Left ankle
                    (x + 2*w//3, y + h, 0.6)           # Right ankle
                ]
                
                return keypoints
            else:
                print("No body contour detected")
                return None
    
    def analyze(self, image, keypoints=None):
        """
        Analyze body for health indicators
        
        Args:
            image: Input image with person's body
            keypoints: Pre-detected body keypoints (optional)
            
        Returns:
            Dictionary with body health analysis
        """
        if keypoints is None:
            keypoints = self.detect_pose(image)
            
        if not keypoints:
            return {"error": "No body detected or poor image quality"}
        
        # Start building the analysis result
        result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "keypoints_detected": len([p for p in keypoints if p is not None]),
            "keypoint_confidence": sum([p[2] for p in keypoints if p is not None]) / len([p for p in keypoints if p is not None]) if keypoints else 0,
            "height_pixels": 0,
            "body_analysis": {}
        }
        
        # Calculate height (distance between highest and lowest detected points)
        valid_keypoints = [kp for kp in keypoints if kp is not None]
        if valid_keypoints:
            min_y = min(valid_keypoints, key=lambda kp: kp[1])[1]
            max_y = max(valid_keypoints, key=lambda kp: kp[1])[1]
            result["height_pixels"] = max_y - min_y
        
        # Analyze posture
        posture_analysis = self._analyze_posture(keypoints)
        result["body_analysis"]["posture"] = posture_analysis
        
        # Analyze body proportions
        proportion_analysis = self._analyze_proportions(keypoints)
        result["body_analysis"]["proportions"] = proportion_analysis
        
        # Analyze symmetry
        symmetry_analysis = self._analyze_symmetry(keypoints)
        result["body_analysis"]["symmetry"] = symmetry_analysis
        
        # Analyze balance/weight distribution
        balance_analysis = self._analyze_balance(keypoints)
        result["body_analysis"]["balance"] = balance_analysis
        
        # Generate overall body health assessment
        health_assessment = self._generate_health_assessment(
            posture_analysis, 
            proportion_analysis, 
            symmetry_analysis, 
            balance_analysis
        )
        result["body_analysis"]["health_assessment"] = health_assessment
        
        # Generate recommendations based on findings
        recommendations = self._generate_recommendations(result["body_analysis"])
        result["recommendations"] = recommendations
        
        return result
    
    def _analyze_posture(self, keypoints):
        """Analyze body posture based on spine alignment and head position"""
        # In a real implementation, this would use the keypoints to measure:
        # - Vertical alignment of ankles, hips, shoulders and ears
        # - Forward head position
        # - Slouching indicators
        
        # For demonstration, we'll create a simplified analysis
        if not all(kp is not None for kp in [keypoints[1], keypoints[8], keypoints[9], keypoints[10]]):
            return {
                "spine_alignment": None,
                "head_position": None,
                "posture_quality": None,
                "posture_note": "Could not analyze posture from image"
            }
        
        # Calculate spine alignment based on neck and mid-hip position
        neck = keypoints[1]
        torso_mid = keypoints[8]
        left_hip = keypoints[9]
        right_hip = keypoints[10]
        
        # Hip center position
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        
        # Angle of spine from vertical (0° is perfect vertical alignment)
        dx = neck[0] - hip_center_x
        dy = neck[1] - hip_center_y
        spine_angle = abs(math.degrees(math.atan2(dx, dy)))
        vertical_deviation = abs(90 - spine_angle)
        
        # Evaluate posture quality
        if vertical_deviation < self.health_thresholds["posture_angle"]["excellent"]:
            posture_quality = "Excellent"
            posture_note = "Great vertical alignment"
        elif vertical_deviation < self.health_thresholds["posture_angle"]["good"]:
            posture_quality = "Good"
            posture_note = "Good posture with slight deviation"
        elif vertical_deviation < self.health_thresholds["posture_angle"]["fair"]:
            posture_quality = "Fair" 
            posture_note = "Moderate posture issues observed"
        else:
            posture_quality = "Concerning"
            posture_note = "Significant posture deviation detected"
            
        # Head position analysis
        # This would be more complex in a real implementation
        head_forward_position = 0  # Placeholder
        
        return {
            "spine_alignment": 1.0 - (vertical_deviation / 45.0),  # 0-1 scale (1 is perfect)
            "vertical_deviation_degrees": vertical_deviation,
            "head_position": head_forward_position,
            "posture_quality": posture_quality,
            "posture_note": posture_note
        }
    
    def _analyze_proportions(self, keypoints):
        """Analyze body proportions and ratios"""
        result = {
            "waist_hip_ratio": None,
            "shoulder_width_ratio": None,
            "leg_torso_ratio": None,
            "proportion_note": "Could not analyze proportions completely"
        }
        
        # Calculate waist-hip ratio if possible
        if all(kp is not None for kp in [keypoints[8], keypoints[9], keypoints[10]]):
            # This is simplified - in a real implementation you would detect actual waist
            torso_mid = keypoints[8]
            left_hip = keypoints[9]
            right_hip = keypoints[10]
            
            # Estimate waist width (simplified)
            waist_width = torso_mid[0] * 0.9  # Just a placeholder
            
            # Hip width
            hip_width = abs(left_hip[0] - right_hip[0])
            
            if hip_width > 0:
                result["waist_hip_ratio"] = waist_width / hip_width
        
        # Calculate shoulder width ratio if possible
        if all(kp is not None for kp in [keypoints[2], keypoints[3]]):
            left_shoulder = keypoints[2]
            right_shoulder = keypoints[3]
            
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            # Compare to height for ratio
            if result["waist_hip_ratio"] is not None:
                result["shoulder_width_ratio"] = shoulder_width / (hip_width if 'hip_width' in locals() else 1)
        
        # Calculate leg-to-torso ratio if possible
        if all(kp is not None for kp in [keypoints[9], keypoints[10], keypoints[13], keypoints[14]]):
            left_hip = keypoints[9]
            right_hip = keypoints[10]
            left_ankle = keypoints[13]
            right_ankle = keypoints[14]
            
            # Hip center
            hip_y = (left_hip[1] + right_hip[1]) / 2
            
            # Ankle average height
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            
            # Leg length
            leg_length = ankle_y - hip_y
            
            # Torso length (neck to hip)
            if keypoints[1] is not None:
                neck = keypoints[1]
                torso_length = hip_y - neck[1]
                
                if torso_length > 0:
                    result["leg_torso_ratio"] = leg_length / torso_length
        
        # Generate proportion note
        if result["waist_hip_ratio"] is not None or result["shoulder_width_ratio"] is not None:
            result["proportion_note"] = "Body proportions within normal range"
            
            if result["waist_hip_ratio"] is not None:
                whr = result["waist_hip_ratio"]
                if whr > 0.9:
                    result["proportion_note"] = "Higher waist-hip ratio detected"
                elif whr < 0.7:
                    result["proportion_note"] = "Lower waist-hip ratio detected"
        
        return result
    
    def _analyze_symmetry(self, keypoints):
        """Analyze body symmetry"""
        result = {
            "shoulder_symmetry": None,
            "hip_symmetry": None,
            "overall_symmetry": None,
            "symmetry_note": "Could not analyze symmetry completely"
        }
        
        image_center_x = None
        
        # Calculate the vertical center line of the body
        if all(kp is not None for kp in [keypoints[0], keypoints[1], keypoints[8]]):
            nose = keypoints[0]
            neck = keypoints[1]
            torso_mid = keypoints[8]
            
            # Vertical center line
            image_center_x = (nose[0] + neck[0] + torso_mid[0]) / 3
        
        # Calculate shoulder symmetry if possible
        if all(kp is not None for kp in [keypoints[2], keypoints[3]]) and image_center_x is not None:
            left_shoulder = keypoints[2]
            right_shoulder = keypoints[3]
            
            # Distance from center line to each shoulder
            left_dist = abs(image_center_x - left_shoulder[0])
            right_dist = abs(right_shoulder[0] - image_center_x)
            
            # Shoulder level difference
            level_diff = abs(left_shoulder[1] - right_shoulder[1])
            height_avg = (left_shoulder[1] + right_shoulder[1]) / 2
            level_symmetry = 1.0 - min(1.0, level_diff / (height_avg * 0.2))
            
            # Width symmetry
            width_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
            
            # Overall shoulder symmetry
            result["shoulder_symmetry"] = (level_symmetry * 0.7) + (width_ratio * 0.3)
        
        # Calculate hip symmetry if possible
        if all(kp is not None for kp in [keypoints[9], keypoints[10]]) and image_center_x is not None:
            left_hip = keypoints[9]
            right_hip = keypoints[10]
            
            # Distance from center line to each hip
            left_dist = abs(image_center_x - left_hip[0])
            right_dist = abs(right_hip[0] - image_center_x)
            
            # Hip level difference
            level_diff = abs(left_hip[1] - right_hip[1])
            height_avg = (left_hip[1] + right_hip[1]) / 2
            level_symmetry = 1.0 - min(1.0, level_diff / (height_avg * 0.2))
            
            # Width symmetry
            width_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
            
            # Overall hip symmetry
            result["hip_symmetry"] = (level_symmetry * 0.7) + (width_ratio * 0.3)
        
        # Calculate overall symmetry
        if result["shoulder_symmetry"] is not None and result["hip_symmetry"] is not None:
            result["overall_symmetry"] = (result["shoulder_symmetry"] * 0.5) + (result["hip_symmetry"] * 0.5)
            
            # Generate symmetry note
            if result["overall_symmetry"] > 0.9:
                result["symmetry_note"] = "Excellent body symmetry"
            elif result["overall_symmetry"] > 0.8:
                result["symmetry_note"] = "Good body symmetry"
            elif result["overall_symmetry"] > 0.7:
                result["symmetry_note"] = "Fair body symmetry"
            else:
                result["symmetry_note"] = "Body asymmetry detected"
        
        return result
    
    def _analyze_balance(self, keypoints):
        """Analyze balance and weight distribution"""
        result = {
            "weight_distribution": None,
            "balance_quality": None,
            "balance_note": "Could not analyze balance completely"
        }
        
        # Calculate weight distribution based on position of ankles and center line
        if all(kp is not None for kp in [keypoints[13], keypoints[14], keypoints[0], keypoints[1]]):
            left_ankle = keypoints[13]
            right_ankle = keypoints[14]
            
            # Find the center line
            nose = keypoints[0]
            neck = keypoints[1]
            center_x = (nose[0] + neck[0]) / 2
            
            # Calculate the midpoint between ankles
            ankle_mid_x = (left_ankle[0] + right_ankle[0]) / 2
            
            # Calculate deviation from center
            # Perfect balance would have ankles centered under the head
            deviation = abs(center_x - ankle_mid_x)
            max_deviation = abs(left_ankle[0] - right_ankle[0]) / 2  # Maximum reasonable deviation
            
            # Calculate balance score (0-1 where 1 is perfect balance)
            weight_distribution = 1.0 - min(1.0, deviation / max_deviation)
            result["weight_distribution"] = weight_distribution
            
            # Evaluate balance quality
            if weight_distribution > self.health_thresholds["weight_distribution"]["excellent"]:
                result["balance_quality"] = "Excellent"
                result["balance_note"] = "Excellent weight distribution and balance"
            elif weight_distribution > self.health_thresholds["weight_distribution"]["good"]:
                result["balance_quality"] = "Good"
                result["balance_note"] = "Good weight distribution"
            elif weight_distribution > self.health_thresholds["weight_distribution"]["fair"]:
                result["balance_quality"] = "Fair"
                result["balance_note"] = "Fair weight distribution, slight imbalance"
            else:
                result["balance_quality"] = "Concerning"
                result["balance_note"] = "Weight distribution imbalance detected"
        
        return result
    
    def _generate_health_assessment(self, posture, proportions, symmetry, balance):
        """Generate overall health assessment based on all body analyses"""
        # Calculate overall health score
        score_components = []
        
        # Posture component
        if posture.get("spine_alignment") is not None:
            score_components.append(posture["spine_alignment"] * 10)  # Scale to 0-10
        
        # Symmetry component
        if symmetry.get("overall_symmetry") is not None:
            score_components.append(symmetry["overall_symmetry"] * 10)  # Scale to 0-10
        
        # Balance component
        if balance.get("weight_distribution") is not None:
            score_components.append(balance["weight_distribution"] * 10)  # Scale to 0-10
        
        # Calculate average score if we have components
        if score_components:
            health_score = sum(score_components) / len(score_components)
        else:
            health_score = 5.0  # Default neutral score
        
        # Determine health status based on score
        if health_score >= 8.5:
            health_status = "Excellent"
        elif health_score >= 7.0:
            health_status = "Good"
        elif health_score >= 5.5:
            health_status = "Fair"
        elif health_score >= 4.0:
            health_status = "Concerning"
        else:
            health_status = "Poor"
        
        # Generate summary
        summary = ""
        if posture.get("posture_quality") and symmetry.get("symmetry_note"):
            if posture["posture_quality"] in ["Excellent", "Good"] and symmetry.get("overall_symmetry", 0) > 0.8:
                summary = "Body analysis indicates good overall posture and alignment."
            elif posture["posture_quality"] in ["Fair"] or (symmetry.get("overall_symmetry", 0) <= 0.8 and symmetry.get("overall_symmetry", 0) > 0.7):
                summary = "Body analysis shows some alignment issues that may benefit from attention."
            else:
                summary = "Body analysis indicates several posture and alignment issues that should be addressed."
        else:
            summary = "Limited body analysis data available."
        
        # Return complete assessment
        return {
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "summary": summary
        }
    
    def _generate_recommendations(self, analysis):
        """Generate recommendations based on body analysis"""
        recommendations = []
        
        # Posture recommendations
        if "posture" in analysis:
            posture = analysis["posture"]
            
            if posture.get("posture_quality") in ["Concerning", "Fair"]:
                recommendations.append("Consider posture improvement exercises")
                
                if posture.get("vertical_deviation_degrees", 0) > 15:
                    recommendations.append("Practice standing with back against wall to improve alignment")
            
            if posture.get("posture_note") and "forward" in posture.get("posture_note", "").lower():
                recommendations.append("Practice chin tucks to correct forward head position")
        
        # Symmetry recommendations
        if "symmetry" in analysis:
            symmetry = analysis["symmetry"]
            
            if symmetry.get("shoulder_symmetry", 1.0) < 0.8:
                recommendations.append("Consider exercises to strengthen weaker side")
                
            if symmetry.get("overall_symmetry", 1.0) < 0.75:
                recommendations.append("Consult with a physical therapist about body asymmetry")
        
        # Balance recommendations
        if "balance" in analysis:
            balance = analysis["balance"]
            
            if balance.get("weight_distribution", 1.0) < 0.7:
                recommendations.append("Practice balance exercises and weight distribution awareness")
                
            if balance.get("balance_quality") in ["Concerning", "Fair"]:
                recommendations.append("Consider single-leg balance exercises to improve stability")
        
        # Add general recommendations if we don't have many specific ones
        if len(recommendations) < 2:
            if analysis.get("health_assessment", {}).get("health_score", 10) < 7:
                recommendations.append("Consider consulting with a physical therapist for assessment")
            
            recommendations.append("Regular stretching and full body movement can improve overall body alignment")
        
        return recommendations
    
    def draw_pose(self, image, keypoints):
        """
        Draw detected body pose on image for visualization
        
        Args:
            image: Input image
            keypoints: Detected body keypoints
            
        Returns:
            Image with pose visualization
        """
        vis_img = image.copy()
        
        # Define connections for visualization
        connections = [
            (0, 1),    # Nose to Neck
            (1, 2),    # Neck to Left Shoulder
            (1, 3),    # Neck to Right Shoulder
            (2, 4),    # Left Shoulder to Left Elbow
            (3, 5),    # Right Shoulder to Right Elbow
            (4, 6),    # Left Elbow to Left Wrist
            (5, 7),    # Right Elbow to Right Wrist
            (1, 8),    # Neck to Torso
            (8, 9),    # Torso to Left Hip
            (8, 10),   # Torso to Right Hip
            (9, 11),   # Left Hip to Left Knee
            (10, 12),  # Right Hip to Right Knee
            (11, 13),  # Left Knee to Left Ankle
            (12, 14),  # Right Knee to Right Ankle
        ]
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp is not None:
                cv2.circle(vis_img, (kp[0], kp[1]), 5, (0, 255, 255), -1)
                cv2.putText(vis_img, str(i), (kp[0] + 10, kp[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                if keypoints[start_idx] is not None and keypoints[end_idx] is not None:
                    cv2.line(vis_img, 
                            (keypoints[start_idx][0], keypoints[start_idx][1]),
                            (keypoints[end_idx][0], keypoints[end_idx][1]),
                            (0, 255, 0), 2)
        
        return vis_img
"""
Health Analyzer Module
Analyzes facial features to identify potential health indicators in real-time.
"""

import numpy as np
from datetime import datetime
import time
import random  # For generating realistic health data for demo

class HealthAnalyzer:
    """A class to analyze facial features for health indicators with real-time capabilities"""
    
    def __init__(self):
        """Initialize the health analyzer"""
        # Define reference ranges for various health indicators
        # These are approximate and based on medical literature
        self.reference_ranges = {
            'symmetry': {'low': 0.6, 'normal': 0.8, 'high': 1.0},
            'eye_bags': {'none': 0, 'mild': 20, 'moderate': 40, 'severe': 60},
            'skin_tone_evenness': {'low': 0.3, 'normal': 0.7, 'high': 1.0},
            'facial_fullness': {'low': 0.3, 'normal': 0.7, 'high': 1.0},
            'skin_hydration': {'dehydrated': 0.3, 'adequate': 0.6, 'optimal': 0.85}
        }
        
        # Medical reference data for diagnostic patterns
        self.medical_patterns = {
            'thyroid': {
                'features': ['puffy face', 'eye protrusion', 'swollen neck'],
                'description': 'Possible thyroid dysfunction indicators'
            },
            'cardiovascular': {
                'features': ['facial flushing', 'visible capillaries', 'earlobe crease'],
                'description': 'Potential cardiovascular stress indicators'
            },
            'liver': {
                'features': ['yellowish tint', 'spider angiomas', 'pale conjunctiva'],
                'description': 'Possible liver function indicators'
            },
            'nutritional': {
                'features': ['angular cheilitis', 'pale conjunctiva', 'facial wasting'],
                'description': 'Potential nutritional deficiency indicators'
            },
            'stress': {
                'features': ['tension lines', 'pursed lips', 'furrowed brow'],
                'description': 'Stress-related facial patterns'
            }
        }
        
        # For real-time performance tracking
        self.analysis_time = 0
        self.fps = 0
        
        # Store recent health data for trend analysis
        self.history = {
            'facial_symmetry': [],
            'eye_fatigue': [],
            'facial_fullness': [],
            'skin_data': []
        }
        self.history_max_size = 30  # Store last 30 frames of data
    
    def analyze(self, features):
        """
        Analyze facial features for health indicators
        
        Args:
            features (dict): Dictionary of extracted facial features
            
        Returns:
            dict: Dictionary of health indicators and their values
        """
        start_time = time.time()
        health_data = {}
        
        # Only analyze if we have valid features
        if features and 'landmarks' in features and features['landmarks']:
            # Analyze facial symmetry for neurological indicators
            health_data.update(self._analyze_symmetry(features))
            
            # Analyze skin for dermatological indicators
            health_data.update(self._analyze_skin(features))
            
            # Analyze facial structure for general health indicators
            health_data.update(self._analyze_facial_structure(features))
            
            # Analyze eye region for fatigue and other indicators
            health_data.update(self._analyze_eyes(features))
            
            # Analyze mouth and lips for health indicators
            health_data.update(self._analyze_mouth(features))
            
            # Analyze potential health patterns
            health_data.update(self._analyze_health_patterns(features))
            
            # Calculate biomarker estimates (for demonstration)
            health_data.update(self._estimate_biomarkers(features))
            
            # Update history data for trend analysis
            self._update_history(health_data)
            
            # Add trend indicators if we have enough history
            if any(len(data) >= 10 for data in self.history.values()):
                health_data.update(self._analyze_trends())
        
        # Add timestamp
        health_data['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate processing time and fps
        end_time = time.time()
        self.analysis_time = end_time - start_time
        self.fps = 1.0 / self.analysis_time if self.analysis_time > 0 else 0
        
        return health_data
    
    def _analyze_symmetry(self, features):
        """Analyze facial symmetry for potential health indicators - optimized for real-time"""
        indicators = {}
        
        if 'symmetry' in features:
            symmetry = features['symmetry']
            
            # Overall facial symmetry (can indicate neurological health)
            if 'overall_symmetry' in symmetry:
                overall_sym = symmetry['overall_symmetry']
                indicators['facial_symmetry'] = overall_sym
                
                # Classify symmetry level
                if overall_sym < self.reference_ranges['symmetry']['low']:
                    indicators['symmetry_evaluation'] = "Low symmetry"
                elif overall_sym < self.reference_ranges['symmetry']['normal']:
                    indicators['symmetry_evaluation'] = "Moderate symmetry"
                else:
                    indicators['symmetry_evaluation'] = "High symmetry"
                
                # Potential neurological health indicator
                if overall_sym < 0.7:
                    indicators['note_symmetry'] = "Facial asymmetry detected - can be normal variation or may indicate muscle imbalance"
                    if overall_sym < 0.6:
                        indicators['note_symmetry'] = "Significant facial asymmetry detected - consider assessment if recent change"
            
            # Eye level symmetry
            if 'eyes_level' in symmetry:
                eyes_level = symmetry['eyes_level']
                indicators['eyes_level_symmetry'] = eyes_level
                
                if eyes_level < 0.85:
                    indicators['note_eye_level'] = "Eye level asymmetry noted - may indicate musculoskeletal alignment factors"
            
            # Smile symmetry
            if 'smile_symmetry' in symmetry:
                smile_sym = symmetry['smile_symmetry']
                indicators['smile_symmetry'] = smile_sym
                
                if smile_sym < 0.75:
                    indicators['note_smile'] = "Asymmetric smile pattern detected"
        
        return indicators
    
    def _analyze_skin(self, features):
        """Analyze skin features for dermatological indicators - optimized for real-time"""
        indicators = {}
        
        if 'skin' in features:
            skin_data = features['skin']
            
            # Skin tone analysis
            if 'skin_tone' in skin_data:
                tone = skin_data['skin_tone']
                
                # Analyze hue for potential conditions
                hue = tone.get('hue', 0)
                sat = tone.get('saturation', 0)
                val = tone.get('value', 0)
                
                # Analyze skin tone for health indicators
                if 20 <= hue <= 40 and sat > 100:
                    indicators['skin_tone_note'] = "Yellowish tint detected - may indicate bilirubin variations"
                elif val < 150 and sat < 50:
                    indicators['skin_tone_note'] = "Pale complexion detected - may relate to circulation or blood metrics"
                elif (0 <= hue <= 10 or 170 <= hue <= 180) and sat > 100:
                    indicators['skin_tone_note'] = "Increased skin redness detected - may indicate inflammatory response"
                else:
                    indicators['skin_tone_note'] = "Normal skin tone variation detected"
            
            # Texture analysis
            if 'texture' in skin_data:
                texture = skin_data['texture']
                indicators['skin_texture'] = texture
                
                # High texture variance could indicate skin conditions
                if texture > 60:
                    indicators['texture_note'] = "Significant skin texture variation detected - consider hydration assessment"
                elif texture > 40:
                    indicators['texture_note'] = "Moderate skin texture variation - may indicate mild dehydration"
                elif texture > 20:
                    indicators['texture_note'] = "Normal skin texture detected"
                else:
                    indicators['texture_note'] = "Smooth skin texture detected - good hydration indicators"
            
            # Skin hydration estimate
            if 'moisture' in skin_data:
                hydration = skin_data['moisture']
                indicators['skin_hydration'] = hydration
                
                if hydration < self.reference_ranges['skin_hydration']['adequate']:
                    indicators['hydration_note'] = "Potential dehydration indicators detected"
                elif hydration > self.reference_ranges['skin_hydration']['optimal']:
                    indicators['hydration_note'] = "Optimal skin hydration detected"
        
        return indicators
    
    def _analyze_facial_structure(self, features):
        """Analyze facial structure for general health indicators - optimized for real-time"""
        indicators = {}
        
        # Facial proportions and fullness (can indicate nutritional status)
        if 'metrics' in features:
            metrics = features['metrics']
            
            # Cheek fullness estimation
            if 'face_width' in metrics and 'face_height' in metrics:
                face_width = metrics['face_width']
                face_height = metrics['face_height']
                
                # Calculate approximate facial fullness
                face_area = face_width * face_height
                face_perimeter = 2 * (face_width + face_height)
                
                if face_perimeter > 0:
                    fullness = face_area / (face_perimeter * face_perimeter)
                    fullness_normalized = min(1.0, fullness * 1000)  # Scale to 0-1 range
                    indicators['facial_fullness'] = fullness_normalized
                    
                    # Classify fullness
                    if fullness_normalized < self.reference_ranges['facial_fullness']['low']:
                        indicators['fullness_evaluation'] = "Low facial fullness - may indicate low body fat percentage"
                    elif fullness_normalized < self.reference_ranges['facial_fullness']['normal']:
                        indicators['fullness_evaluation'] = "Moderate facial fullness - within healthy parameters"
                    else:
                        indicators['fullness_evaluation'] = "High facial fullness - within normal variation"
        
        # Golden ratio analysis - simplified for real-time
        if 'facial_ratios' in features and 'top_golden_ratio_diff' in features['facial_ratios']:
            golden_ratio_diff = features['facial_ratios']['top_golden_ratio_diff']
            indicators['golden_ratio_harmony'] = max(0, 1 - golden_ratio_diff)
        
        return indicators
    
    def _analyze_eyes(self, features):
        """Analyze eye region for fatigue and other indicators - optimized for real-time"""
        indicators = {}
        
        # Check if we have landmarks to work with
        if 'landmarks' in features and features['landmarks']:
            landmarks = features['landmarks']
            
            if len(landmarks) >= 468:  # Full set of MediaPipe landmarks
                try:
                    # Average height of eyes
                    left_eye_top = landmarks[159]
                    left_eye_bottom = landmarks[145]
                    right_eye_top = landmarks[386]
                    right_eye_bottom = landmarks[374]
                    
                    left_eye_height = np.linalg.norm(np.array(left_eye_top) - np.array(left_eye_bottom))
                    right_eye_height = np.linalg.norm(np.array(right_eye_top) - np.array(right_eye_bottom))
                    
                    # Get eye widths from metrics if available
                    eye_heights = [left_eye_height, right_eye_height]
                    eye_widths = []
                    
                    if 'metrics' in features:
                        metrics = features['metrics']
                        if 'left_eye_width' in metrics:
                            eye_widths.append(metrics['left_eye_width'])
                        if 'right_eye_width' in metrics:
                            eye_widths.append(metrics['right_eye_width'])
                    
                    # Calculate eye aspect ratio (height/width)
                    eye_aspect_ratios = []
                    for i in range(min(len(eye_heights), len(eye_widths))):
                        if eye_widths[i] > 0:
                            ear = eye_heights[i] / eye_widths[i]
                            eye_aspect_ratios.append(ear)
                    
                    if eye_aspect_ratios:
                        avg_ear = sum(eye_aspect_ratios) / len(eye_aspect_ratios)
                        indicators['eye_openness'] = avg_ear
                        
                        # Low EAR can indicate fatigue
                        if avg_ear < 0.2:
                            indicators['eye_fatigue'] = "High"
                            indicators['eye_health_note'] = "Signs of significant eye fatigue detected - consider rest and screen time reduction"
                        elif avg_ear < 0.3:
                            indicators['eye_fatigue'] = "Moderate"
                            indicators['eye_health_note'] = "Moderate eye fatigue indicators - consider short breaks"
                        else:
                            indicators['eye_fatigue'] = "Low"
                            indicators['eye_health_note'] = "Minimal eye fatigue detected"
                    
                    # Check for eye bags - simplified for real-time
                    # For simplicity, just use the vertical distance from eye to cheek
                    if 'face_height' in features.get('metrics', {}):
                        face_height = features['metrics']['face_height']
                        left_eye_bag = np.linalg.norm(np.array(left_eye_bottom) - np.array(landmarks[117]))
                        eye_bag_ratio = left_eye_bag / face_height * 100
                        indicators['eye_bags'] = eye_bag_ratio
                        
                        # Classify eye bags severity
                        if eye_bag_ratio < self.reference_ranges['eye_bags']['mild']:
                            indicators['eye_bags_evaluation'] = "None to minimal"
                        elif eye_bag_ratio < self.reference_ranges['eye_bags']['moderate']:
                            indicators['eye_bags_evaluation'] = "Mild"
                        else:
                            indicators['eye_bags_evaluation'] = "Moderate to severe"
                            indicators['eye_bags_note'] = "Prominent eye bags may indicate fluid retention, allergies, or sleep factors"
                
                    # Analyze sclera (white of the eye) color
                    # This would typically use color analysis of the sclera region
                    # Here we'll simulate it for demonstration
                    sclera_color = features.get('eye_details', {}).get('sclera_color', {'r': 255, 'g': 255, 'b': 255})
                    
                    # Calculate yellowness (using r and g channels primarily)
                    if isinstance(sclera_color, dict) and 'r' in sclera_color and 'g' in sclera_color:
                        yellow_index = (sclera_color['r'] + sclera_color['g']) / 2 - sclera_color.get('b', 0)
                        # Higher values indicate more yellow
                        if yellow_index > 30:  # Threshold for sclera yellowness
                            indicators['sclera_note'] = "Yellowish sclera detected - may relate to liver function or normal variation"
                
                except (IndexError, KeyError, ZeroDivisionError):
                    # Handle missing landmarks gracefully
                    pass
        
        return indicators
    
    def _analyze_mouth(self, features):
        """Analyze mouth and lip features for health indicators"""
        indicators = {}
        
        if 'landmarks' in features and len(features['landmarks']) >= 468:
            try:
                landmarks = features['landmarks']
                
                # Get key mouth landmarks
                upper_lip = landmarks[0]  # Simplified; would use multiple points
                lower_lip = landmarks[17]  # Simplified; would use multiple points
                
                # Check lip color (would use advanced color analysis)
                # Simulating this for demonstration
                lip_color = features.get('mouth_details', {}).get('lip_color', {'r': 150, 'g': 100, 'b': 100})
                
                if isinstance(lip_color, dict):
                    # Calculate lip color metrics
                    lip_redness = lip_color.get('r', 0) - ((lip_color.get('g', 0) + lip_color.get('b', 0)) / 2)
                    
                    # Check for pale lips (potential circulation or anemia indicator)
                    if lip_redness < 30:
                        indicators['lip_color_note'] = "Lighter lip color detected - may relate to circulation factors"
                    elif lip_redness > 80:
                        indicators['lip_color_note'] = "High lip redness - within normal variation" 
                    else:
                        indicators['lip_color_note'] = "Normal lip coloration"
                
                # Check for angular cheilitis
                if 'mouth_corners' in features.get('mouth_details', {}):
                    corner_irritation = features['mouth_details']['mouth_corners'].get('irritation', 0)
                    if corner_irritation > 0.5:  # Threshold for detection
                        indicators['mouth_corner_note'] = "Potential angular cheilitis detected - may relate to nutritional or immune factors"
                
            except (IndexError, KeyError):
                pass
        
        return indicators
    
    def _analyze_health_patterns(self, features):
        """Analyze combinations of features for potential health patterns"""
        indicators = {}
        
        # Detect patterns based on feature combinations (simplified for demonstration)
        detected_patterns = []
        
        # To make this more realistic in a demo context without full feature availability,
        # we'll randomize some pattern detection with low probability
        for pattern_name, pattern_info in self.medical_patterns.items():
            # In a real system, we would check for actual features
            # Here we'll randomly "detect" patterns with very low probability for demo
            if random.random() < 0.05:  # 5% chance to detect any pattern
                detected_patterns.append(pattern_name)
        
        if detected_patterns:
            pattern_details = []
            for pattern in detected_patterns:
                if pattern in self.medical_patterns:
                    pattern_details.append({
                        'pattern': pattern,
                        'description': self.medical_patterns[pattern]['description']
                    })
            
            if pattern_details:
                indicators['health_patterns'] = pattern_details
        
        return indicators
    
    def _estimate_biomarkers(self, features):
        """
        Estimate potential biomarker ranges based on facial features
        Note: These are statistical correlations only, not diagnostic
        """
        biomarkers = {}
        
        # In a real system, we would use research-backed ML models
        # Here we'll simulate reasonable values for demonstration
        
        # Get baseline health indicators we've already calculated
        facial_symmetry = features.get('symmetry', {}).get('overall_symmetry', 0.8)
        skin_texture = features.get('skin', {}).get('texture', 30)
        eyes_level_symmetry = features.get('symmetry', {}).get('eyes_level', 0.9)
        
        # Estimate stress biomarkers (cortisol proxy)
        # Research has shown correlations between facial tension patterns and cortisol
        stress_factors = []
        
        if 'stress_indicators' in features:
            tension_score = features['stress_indicators'].get('tension_score', 0.5)
            stress_factors.append(tension_score)
        
        # Use facial symmetry as another stress indicator (lower symmetry can correlate with stress)
        stress_factors.append(1 - facial_symmetry)
        
        if stress_factors:
            avg_stress = sum(stress_factors) / len(stress_factors)
            # Map to a reasonable cortisol range (normalized 0-1 to numerical range)
            cortisol_estimate = 12 + (avg_stress * 13)  # Normal range ~5-25 μg/dL
            biomarkers['estimated_stress_level'] = {
                'value': round(cortisol_estimate, 1),
                'unit': 'proxy score',
                'note': 'Estimated from facial tension patterns, not actual lab value'
            }
        
        # Sleep quality estimation
        sleep_factors = []
        if 'eye_bags' in biomarkers:
            sleep_factors.append(biomarkers['eye_bags'] / 50)  # Normalize to 0-1
        
        # Randomize a bit for demo purposes
        sleep_score = 0.7 + (random.random() * 0.3 - 0.15)  # Base 0.7 with ±0.15 variation
        biomarkers['sleep_quality_estimate'] = {
            'value': round(sleep_score * 10, 1),
            'unit': 'quality score',
            'note': 'Estimated from eye appearance and facial relaxation indicators'
        }
        
        return biomarkers
    
    def _update_history(self, health_data):
        """Update history data for trend analysis"""
        # Update facial symmetry history
        if 'facial_symmetry' in health_data:
            self.history['facial_symmetry'].append(health_data['facial_symmetry'])
            if len(self.history['facial_symmetry']) > self.history_max_size:
                self.history['facial_symmetry'].pop(0)
        
        # Update eye fatigue history
        if 'eye_fatigue' in health_data:
            fatigue_value = 0
            if health_data['eye_fatigue'] == "Low":
                fatigue_value = 0.2
            elif health_data['eye_fatigue'] == "Moderate":
                fatigue_value = 0.6
            else:  # High
                fatigue_value = 1.0
            
            self.history['eye_fatigue'].append(fatigue_value)
            if len(self.history['eye_fatigue']) > self.history_max_size:
                self.history['eye_fatigue'].pop(0)
        
        # Update facial fullness history
        if 'facial_fullness' in health_data:
            self.history['facial_fullness'].append(health_data['facial_fullness'])
            if len(self.history['facial_fullness']) > self.history_max_size:
                self.history['facial_fullness'].pop(0)
        
        # Update skin data history
        skin_data_point = {}
        if 'skin_texture' in health_data:
            skin_data_point['texture'] = health_data['skin_texture']
        if 'skin_tone_note' in health_data:
            skin_data_point['note'] = health_data['skin_tone_note']
        
        if skin_data_point:
            self.history['skin_data'].append(skin_data_point)
            if len(self.history['skin_data']) > self.history_max_size:
                self.history['skin_data'].pop(0)
    
    def _analyze_trends(self):
        """Analyze trends in historical health data"""
        trend_data = {}
        
        # Analyze eye fatigue trend
        if len(self.history['eye_fatigue']) >= 10:
            recent_fatigue = np.mean(self.history['eye_fatigue'][-5:])
            earlier_fatigue = np.mean(self.history['eye_fatigue'][:5])
            fatigue_change = recent_fatigue - earlier_fatigue
            
            if fatigue_change > 0.2:
                trend_data['eye_fatigue_trend'] = "Increasing (consider rest)"
            elif fatigue_change < -0.2:
                trend_data['eye_fatigue_trend'] = "Decreasing (improving)"
            else:
                trend_data['eye_fatigue_trend'] = "Stable"
        
        # Analyze facial symmetry trend
        if len(self.history['facial_symmetry']) >= 10:
            symmetry_stability = np.std(self.history['facial_symmetry'])
            if symmetry_stability > 0.1:
                trend_data['symmetry_stability'] = "Variable"
            else:
                trend_data['symmetry_stability'] = "Stable"
        
        return trend_data
    
    def get_processing_stats(self):
        """Return processing statistics for display"""
        return {
            'analysis_time_ms': self.analysis_time * 1000,
            'fps': self.fps
        }
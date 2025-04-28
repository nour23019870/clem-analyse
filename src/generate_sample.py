#!/usr/bin/env python
"""
Sample Data Generator
Creates sample facial analysis data for testing the viewer
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the path to import data_storage
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_storage import DataStorage

def generate_sample_data():
    """Generate sample facial analysis data"""
    
    # Create sample data structure
    sample_data = [
        {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'frame_id': 1,
            'face_id': 0,
            'features': {
                'metrics': {
                    'face_width': 240.5,
                    'face_height': 320.8,
                    'face_width_height_ratio': 0.75,
                    'left_eye_width': 45.2,
                    'right_eye_width': 44.8,
                    'eye_width_ratio': 0.99
                },
                'symmetry': {
                    'eyes_level': 0.98,
                    'nose_deviation': 0.02,
                    'mouth_symmetry': 0.97,
                    'overall_symmetry': 0.96
                },
                'facial_ratios': {
                    'eye_spacing_ratio': 1.62,
                    'top_third_ratio': 0.33,
                    'middle_third_ratio': 0.34
                }
            },
            'health_analysis': {
                'facial_symmetry': 0.96,
                'symmetry_evaluation': 'Excellent facial symmetry',
                'note_symmetry': 'Very balanced features',
                'facial_fullness': 0.65,
                'fullness_evaluation': 'Normal fullness',
                'eye_openness': 0.88,
                'eye_fatigue': 'Minimal',
                'eye_fatigue_trend': 'No significant fatigue detected',
                'eye_bags': 0.15,
                'eye_bags_evaluation': 'Minor eye bags present',
                'eyes_level_symmetry': 0.98,
                'note_eye_level': 'Eyes are well-aligned',
                'skin_texture': 0.92,
                'texture_note': 'Skin appears healthy',
                'skin_tone_note': 'Even skin tone',
                'golden_ratio_harmony': 0.94
            }
        },
        {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'frame_id': 2,
            'face_id': 0,
            'features': {
                'metrics': {
                    'face_width': 238.7,
                    'face_height': 318.5,
                    'face_width_height_ratio': 0.75,
                    'left_eye_width': 44.9,
                    'right_eye_width': 43.2,
                    'eye_width_ratio': 0.96
                },
                'symmetry': {
                    'eyes_level': 0.97,
                    'nose_deviation': 0.03,
                    'mouth_symmetry': 0.95,
                    'overall_symmetry': 0.94
                },
                'facial_ratios': {
                    'eye_spacing_ratio': 1.59,
                    'top_third_ratio': 0.32,
                    'middle_third_ratio': 0.35
                }
            },
            'health_analysis': {
                'facial_symmetry': 0.94,
                'symmetry_evaluation': 'Good facial symmetry',
                'note_symmetry': 'Slight asymmetry in eye region',
                'facial_fullness': 0.60,
                'fullness_evaluation': 'Normal fullness',
                'eye_openness': 0.82,
                'eye_fatigue': 'Mild',
                'eye_fatigue_trend': 'Slight increase in fatigue indicators',
                'eye_bags': 0.25,
                'eye_bags_evaluation': 'Moderate eye bags',
                'eyes_level_symmetry': 0.97,
                'note_eye_level': 'Eyes are well-aligned with slight tilt',
                'skin_texture': 0.88,
                'texture_note': 'Skin appears mostly healthy',
                'skin_tone_note': 'Slightly uneven tone in cheek area',
                'golden_ratio_harmony': 0.91
            }
        }
    ]
    
    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataStorage instance
    storage = DataStorage()
    
    # Save the sample data in different formats
    formats = ['json', 'csv', 'xlsx']
    saved_files = []
    
    for fmt in formats:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'facial_analysis_{timestamp}')
        saved_file = storage.save(sample_data, output_path, format=fmt)
        saved_files.append(saved_file)
        print(f"Generated sample {fmt.upper()} file: {saved_file}")
    
    return saved_files

if __name__ == "__main__":
    print("Generating sample facial analysis data...")
    generate_sample_data()
    print("\nSample data generation complete!")
    print("You can now run the viewer: python src/view_results.py")
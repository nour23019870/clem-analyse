#!/usr/bin/env python
"""
Main entry point for health analysis application
Provides options for facial analysis, body analysis, or complete health analysis
"""

import os
import sys
import argparse
import time
from datetime import datetime

from realtime_analysis import RealtimeFacialAnalyzer
from complete_health_analyzer import CompleteHealthAnalyzer

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Health Analysis System with facial and body analysis'
    )
    
    # Analysis mode selection
    parser.add_argument('--mode', '-m', type=str, choices=['face', 'complete'], 
                      default='complete',
                      help='Analysis mode: face for facial-only, complete for full health analysis')
    
    # Common parameters
    parser.add_argument('--output', '-o', type=str, 
                      default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'),
                      help='Directory to save analysis results')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'csv', 'xlsx'],
                      default='json', help='Output format for storage')
    parser.add_argument('--camera', '-c', type=int, default=0,
                      help='Camera ID (usually 0 for built-in webcam)')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage instead of GPU')
    
    # Face-specific parameters
    parser.add_argument('--method', type=str, default='dlib',
                      choices=['opencv', 'dlib'],
                      help='Face detection method (facial analysis only)')
    parser.add_argument('--interval', '-i', type=int, default=10,
                      help='Save interval in seconds (facial analysis only)')
    parser.add_argument('--no-landmarks', action='store_true',
                      help='Do not display facial landmarks (facial analysis only)')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Show welcome message
    print("\n" + "=" * 80)
    print("Health Analysis System".center(80))
    print("=" * 80 + "\n")
    
    use_gpu = not args.cpu
    
    # Run the selected analysis mode
    if args.mode == 'face':
        print("Starting Facial Analysis Mode")
        analyzer = RealtimeFacialAnalyzer(
            detection_method=args.method,
            output_dir=args.output,
            save_format=args.format,
            use_gpu=use_gpu,
            camera_id=args.camera,
            save_interval=args.interval,
            display_landmarks=not args.no_landmarks
        )
        
        try:
            analyzer.start()
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            analyzer.stop()
            
        print("Facial analysis complete.")
        
    else:  # Complete health analysis
        print("Starting Complete Health Analysis Mode")
        analyzer = CompleteHealthAnalyzer(
            output_dir=args.output,
            save_format=args.format,
            use_gpu=use_gpu,
            camera_id=args.camera
        )
        
        try:
            analyzer.start()
        except KeyboardInterrupt:
            print("\nStopped by user")
            
        print("Complete health analysis finished.")
    
    # Display view instructions
    print("\nTo view your results, run:")
    print("python src/view_results.py")

if __name__ == "__main__":
    main()
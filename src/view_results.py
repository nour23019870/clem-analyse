#!/usr/bin/env python
"""
Facial Analysis Results Viewer
A simple script to view the results of facial analysis stored in output files.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add the current directory to the path to import data_storage
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_storage import DataStorage

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_value(value):
    """Format a value for display"""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def print_header(text, width=80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)

def print_section(title, data, indent=2):
    """Print a section of data"""
    if not data:
        return
    
    print("\n" + " " * indent + title + ":")
    indent_str = " " * (indent + 2)
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            for sub_key, sub_value in value.items():
                print(f"{indent_str}  {sub_key}: {format_value(sub_value)}")
        else:
            print(f"{indent_str}{key}: {format_value(value)}")

def view_results(storage, results_file):
    """View a single result file in detail"""
    try:
        # Get file extension
        _, ext = os.path.splitext(results_file)
        
        # Load results
        results = storage.load(results_file)
        
        if not results:
            print("No results found in the file.")
            return
        
        clear_screen()
        print_header(f"Facial Analysis Results - {os.path.basename(results_file)}")
        
        # Display file info
        print(f"\nFile: {os.path.basename(results_file)}")
        print(f"Format: {ext[1:]}")
        print(f"Size: {os.path.getsize(results_file) / 1024:.1f} KB")
        print(f"Last Modified: {datetime.fromtimestamp(os.path.getmtime(results_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Records: {len(results)}")
        
        # Let user select a face to view
        print("\nAvailable records:")
        for i, result in enumerate(results):
            timestamp = result.get('timestamp', 'N/A')
            face_id = result.get('face_id', i)
            print(f"{i+1}. Face {face_id} - {timestamp}")
        
        while True:
            choice = input("\nEnter record number to view details (or 0 to go back): ")
            if choice == '0':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    view_face_details(results[idx])
                else:
                    print("Invalid record number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    except Exception as e:
        print(f"Error viewing results: {e}")

def view_face_details(result):
    """Display detailed information about a face"""
    clear_screen()
    
    # Display basic info
    print_header(f"Face Analysis Details")
    print(f"\nTimestamp: {result.get('timestamp', 'N/A')}")
    print(f"Frame ID: {result.get('frame_id', 'N/A')}")
    print(f"Face ID: {result.get('face_id', 'N/A')}")
    
    # Display health analysis
    if 'health_analysis' in result:
        print_section("Health Analysis", result['health_analysis'])
    
    # Display facial features
    if 'features' in result:
        features = result['features']
        
        # Display metrics
        if 'metrics' in features:
            print_section("Facial Metrics", features['metrics'])
        
        # Display symmetry
        if 'symmetry' in features:
            print_section("Facial Symmetry", features['symmetry'])
        
        # Display ratios
        if 'facial_ratios' in features:
            print_section("Facial Ratios", features['facial_ratios'])
    
    input("\nPress Enter to continue...")

def generate_report(storage, results_file):
    """Generate a human-readable report from results"""
    try:
        # Load results
        results = storage.load(results_file)
        
        if not results:
            print("No results found to generate report.")
            return
        
        # Create output file name
        base_name = os.path.splitext(os.path.basename(results_file))[0]
        report_path = os.path.join(os.path.dirname(results_file), f"{base_name}_report.md")
        
        # Generate report
        storage.generate_health_report(results, report_path)
        
        print(f"\nReport generated: {report_path}")
        
        # Ask if user wants to open the report
        if os.name == 'nt':  # Windows
            if input("Open report now? (y/n): ").lower() == 'y':
                os.system(f'start "" "{report_path}"')
    
    except Exception as e:
        print(f"Error generating report: {e}")

def view_complete_results(storage, results_file):
    """View a complete health analysis result file in detail"""
    try:
        # Get file extension
        _, ext = os.path.splitext(results_file)
        
        # Load results
        results = storage.load(results_file)
        
        if not results:
            print("No results found in the file.")
            return
        
        clear_screen()
        print_header(f"Complete Health Analysis Results - {os.path.basename(results_file)}")
        
        # Display file info
        print(f"\nFile: {os.path.basename(results_file)}")
        print(f"Format: {ext[1:]}")
        print(f"Size: {os.path.getsize(results_file) / 1024:.1f} KB")
        print(f"Last Modified: {datetime.fromtimestamp(os.path.getmtime(results_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Records: {len(results)}")
        
        # Let user select a record to view
        print("\nComplete health analysis records:")
        for i, result in enumerate(results):
            timestamp = result.get('timestamp', 'N/A')
            overall_status = result.get('overall_health_status', 'N/A')
            overall_score = result.get('overall_health_score', 'N/A')
            print(f"{i+1}. Analysis from {timestamp} - Status: {overall_status}, Score: {overall_score}/10")
        
        while True:
            choice = input("\nEnter record number to view details (or 0 to go back): ")
            if choice == '0':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    view_complete_health_details(results[idx])
                else:
                    print("Invalid record number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    except Exception as e:
        print(f"Error viewing results: {e}")

def view_complete_health_details(result):
    """Display detailed information about complete health analysis"""
    clear_screen()
    
    # Display basic info
    print_header(f"Complete Health Analysis Details")
    print(f"\nTimestamp: {result.get('timestamp', 'N/A')}")
    
    # Display overall health status
    if 'overall_health_status' in result and 'overall_health_score' in result:
        status = result['overall_health_status']
        score = result['overall_health_score']
        print(f"\n--- OVERALL HEALTH ASSESSMENT ---")
        print(f"Status: {status}")
        print(f"Score: {score}/10")
    
    # Display facial analysis
    if 'facial_analysis' in result:
        facial = result['facial_analysis']
        print(f"\n--- FACIAL ANALYSIS SUMMARY ---")
        
        if 'health_status' in facial and 'health_score' in facial:
            print(f"Facial Health Status: {facial['health_status']}")
            print(f"Facial Health Score: {facial['health_score']}/10")
        
        if 'health_analysis' in facial:
            print("\nKey Facial Health Indicators:")
            health = facial['health_analysis']
            
            # Show key indicators
            priority_indicators = ['facial_symmetry', 'symmetry_evaluation', 'eyes_level_symmetry', 
                                  'eye_fatigue', 'skin_texture', 'skin_tone_note']
            
            for indicator in priority_indicators:
                if indicator in health:
                    value = health[indicator]
                    key_name = indicator.replace('_', ' ').title()
                    print(f"  - {key_name}: {format_value(value)}")
    
    # Display body analysis
    if 'body_analysis' in result:
        body = result['body_analysis']
        print(f"\n--- BODY ANALYSIS SUMMARY ---")
        
        # Extract body health assessment if available
        if 'body_analysis' in body and 'health_assessment' in body['body_analysis']:
            assessment = body['body_analysis']['health_assessment']
            if 'health_status' in assessment and 'health_score' in assessment:
                print(f"Body Health Status: {assessment['health_status']}")
                print(f"Body Health Score: {assessment['health_score']}/10")
            
            if 'summary' in assessment:
                print(f"Summary: {assessment['summary']}")
        
        # Show key body metrics
        print("\nKey Body Health Indicators:")
        
        # Posture
        if 'body_analysis' in body and 'posture' in body['body_analysis']:
            posture = body['body_analysis']['posture']
            if 'posture_quality' in posture:
                print(f"  - Posture Quality: {posture['posture_quality']}")
            if 'spine_alignment' in posture:
                print(f"  - Spine Alignment: {format_value(posture['spine_alignment'])} (1.0 is perfect)")
        
        # Symmetry
        if 'body_analysis' in body and 'symmetry' in body['body_analysis']:
            symmetry = body['body_analysis']['symmetry']
            if 'overall_symmetry' in symmetry:
                print(f"  - Body Symmetry: {format_value(symmetry['overall_symmetry'])} (1.0 is perfect)")
            if 'symmetry_note' in symmetry:
                print(f"  - {symmetry['symmetry_note']}")
        
        # Balance
        if 'body_analysis' in body and 'balance' in body['body_analysis']:
            balance = body['body_analysis']['balance']
            if 'balance_quality' in balance:
                print(f"  - Balance Quality: {balance['balance_quality']}")
            if 'weight_distribution' in balance:
                print(f"  - Weight Distribution: {format_value(balance['weight_distribution'])} (1.0 is perfect)")
    
    # Display combined recommendations
    if 'recommendations' in result:
        print("\n--- HEALTH RECOMMENDATIONS ---")
        for i, rec in enumerate(result['recommendations']):
            print(f"  {i+1}. {rec}")
    
    input("\nPress Enter to continue...")

def main():
    """Main function to run the results viewer"""
    # Default output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(os.path.dirname(script_dir), 'output')
    
    # Allow specifying a different output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else default_output_dir
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        print("Please run facial analysis first to generate results.")
        return
    
    # Create DataStorage instance
    storage = DataStorage()
    
    while True:
        clear_screen()
        print_header("Health Analysis Results Viewer")
        
        # Find all result files
        result_files = []
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if os.path.isfile(filepath) and (
                'facial_analysis' in filename or 'complete_health_analysis' in filename):
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.json', '.csv', '.xlsx']:
                    result_files.append(filepath)
        
        # Sort by modification time (newest first)
        result_files.sort(key=os.path.getmtime, reverse=True)
        
        if not result_files:
            print("\nNo result files found in the output directory.")
            print(f"Check: {output_dir}")
            print("\nPlease run facial analysis first to generate results.")
            return
        
        # Display menu
        print(f"\nFound {len(result_files)} result files in: {output_dir}\n")
        
        for i, filepath in enumerate(result_files):
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath) / 1024  # Size in KB
            modified = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Add an icon to differentiate between facial and complete analysis
            file_type = "👤" if 'facial_analysis' in filename else "👤👫"
            
            print(f"{i+1}. {file_type} {filename} ({filesize:.1f} KB, {modified})")
        
        print("\nOptions:")
        print("  v - View a result file")
        print("  r - Generate a report from a result file")
        print("  q - Quit")
        
        choice = input("\nEnter your choice: ").lower()
        
        if choice == 'q':
            break
        elif choice == 'v':
            file_num = input("Enter the number of the file to view: ")
            try:
                idx = int(file_num) - 1
                if 0 <= idx < len(result_files):
                    if 'complete_health_analysis' in result_files[idx]:
                        view_complete_results(storage, result_files[idx])
                    else:
                        view_results(storage, result_files[idx])
                else:
                    print("Invalid file number.")
                    input("Press Enter to continue...")
            except ValueError:
                print("Please enter a valid number.")
                input("Press Enter to continue...")
        elif choice == 'r':
            file_num = input("Enter the number of the file to generate a report from: ")
            try:
                idx = int(file_num) - 1
                if 0 <= idx < len(result_files):
                    generate_report(storage, result_files[idx])
                    input("Press Enter to continue...")
                else:
                    print("Invalid file number.")
                    input("Press Enter to continue...")
            except ValueError:
                print("Please enter a valid number.")
                input("Press Enter to continue...")
        else:
            print("Invalid choice.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
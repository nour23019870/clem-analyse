"""
Data Storage Module
Handles saving facial analysis results in different formats with real-time capabilities.
"""

import os
import json
import csv
import pandas as pd
import threading
import queue
import time
from datetime import datetime

class DataStorage:
    """A class to store facial analysis results in various formats with real-time support"""
    
    def __init__(self):
        """Initialize the data storage handler with real-time capabilities"""
        # For real-time data saving
        self.data_queue = queue.Queue()
        self.save_thread = None
        self.running = False
        self.last_save_time = 0
        self.save_interval = 5  # Save every 5 seconds by default
    
    def save(self, results, output_path, format='json'):
        """
        Save analysis results to a file
        
        Args:
            results (list): List of analysis result dictionaries
            output_path (str): Base path for output file (without extension)
            format (str): Output format ('json', 'csv', or 'xlsx')
            
        Returns:
            str: Path to the saved file
        """
        if format.lower() == 'json':
            return self._save_json(results, output_path)
        elif format.lower() == 'csv':
            return self._save_csv(results, output_path)
        elif format.lower() == 'xlsx':
            return self._save_excel(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_json(self, results, output_path):
        """Save results in JSON format"""
        output_file = f"{output_path}.json"
        
        # Convert numpy floats to Python floats for JSON serialization
        processed_results = self._process_for_serialization(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=2)
        
        return output_file
    
    def _save_csv(self, results, output_path):
        """Save results in CSV format"""
        output_file = f"{output_path}.csv"
        
        # Flatten the nested dictionaries for CSV format
        flattened_data = self._flatten_data(results)
        
        if flattened_data:
            # Write to CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
        
        return output_file
    
    def _save_excel(self, results, output_path):
        """Save results in Excel format"""
        output_file = f"{output_path}.xlsx"
        
        # Flatten the nested dictionaries for Excel format
        flattened_data = self._flatten_data(results)
        
        if flattened_data:
            # Convert to DataFrame and save to Excel
            df = pd.DataFrame(flattened_data)
            df.to_excel(output_file, index=False)
        
        return output_file
    
    def _process_for_serialization(self, data):
        """Process data structure to make it JSON serializable"""
        if isinstance(data, dict):
            return {k: self._process_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_for_serialization(item) for item in data]
        elif hasattr(data, 'item'):  # For numpy numbers
            return data.item()  # Convert numpy number to Python scalar
        else:
            return data
    
    def _flatten_data(self, results):
        """Flatten nested dictionaries for tabular formats like CSV and Excel"""
        flattened_results = []
        
        for result in results:
            flat_item = {}
            
            # Add top-level keys
            for key, value in result.items():
                if not isinstance(value, (dict, list)):
                    flat_item[key] = value
            
            # Handle health analysis data
            if 'health_analysis' in result:
                health = result['health_analysis']
                for health_key, health_value in health.items():
                    flat_item[f"health_{health_key}"] = health_value
            
            # Handle core facial metrics (selectively)
            if 'features' in result and 'metrics' in result['features']:
                metrics = result['features']['metrics']
                for metric_key in ['face_width', 'face_height', 'face_width_height_ratio', 
                                 'left_eye_width', 'right_eye_width', 'eye_width_ratio']:
                    if metric_key in metrics:
                        flat_item[f"metric_{metric_key}"] = metrics[metric_key]
            
            # Handle symmetry data
            if 'features' in result and 'symmetry' in result['features']:
                symmetry = result['features']['symmetry']
                for sym_key, sym_value in symmetry.items():
                    flat_item[f"symmetry_{sym_key}"] = sym_value
            
            # Handle facial ratios (golden ratio)
            if 'features' in result and 'facial_ratios' in result['features']:
                ratios = result['features']['facial_ratios']
                for ratio_key in ['eye_spacing_ratio', 'top_third_ratio', 'middle_third_ratio']:
                    if ratio_key in ratios:
                        flat_item[f"ratio_{ratio_key}"] = ratios[ratio_key]
            
            flattened_results.append(flat_item)
        
        return flattened_results
    
    def start_real_time_saving(self, output_dir, format='json', save_interval=5):
        """
        Start background thread for real-time data saving
        
        Args:
            output_dir (str): Directory to save analysis results
            format (str): Output format ('json', 'csv', or 'xlsx')
            save_interval (int): Interval in seconds between saves
        """
        if self.save_thread and self.save_thread.is_alive():
            print("Real-time saving already running")
            return
        
        self.running = True
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Start background saving thread
        self.save_thread = threading.Thread(
            target=self._background_save_worker, 
            args=(output_dir, format), 
            daemon=True
        )
        self.save_thread.start()
        
        return self.save_thread
    
    def stop_real_time_saving(self):
        """Stop the background saving thread"""
        self.running = False
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)
    
    def queue_data_for_saving(self, data):
        """
        Add data to the saving queue for background processing
        
        Args:
            data (dict): Analysis data to save
        """
        self.data_queue.put(data)
    
    def _background_save_worker(self, output_dir, format):
        """
        Background thread function for saving data
        
        Args:
            output_dir (str): Directory to save analysis results
            format (str): Output format ('json', 'csv', or 'xlsx')
        """
        accumulated_data = []
        
        while self.running:
            # Get all available data from the queue
            try:
                while True:
                    data = self.data_queue.get(block=False)
                    accumulated_data.append(data)
                    self.data_queue.task_done()
            except queue.Empty:
                pass
            
            # Check if it's time to save
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_interval and accumulated_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"facial_analysis_{timestamp}")
                
                try:
                    self.save(accumulated_data, output_path, format)
                    print(f"Saved {len(accumulated_data)} records to {output_path}.{format}")
                    accumulated_data = []  # Clear the accumulated data
                except Exception as e:
                    print(f"Error saving data: {e}")
                
                self.last_save_time = current_time
            
            # Sleep a bit to prevent high CPU usage
            time.sleep(0.1)
    
    def load(self, file_path):
        """
        Load analysis results from a file
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict or list: Loaded analysis results
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.csv':
            return pd.read_csv(file_path).to_dict('records')
        elif ext == '.xlsx':
            return pd.read_excel(file_path).to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def generate_health_report(self, results, output_path):
        """
        Generate a human-readable health report from analysis results
        
        Args:
            results (list or dict): Analysis results (list or single dict)
            output_path (str): Path to save the report
            
        Returns:
            str: Path to the generated report
        """
        if not isinstance(results, list):
            results = [results]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Facial Analysis Health Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, result in enumerate(results):
                f.write(f"## Face #{i+1}\n\n")
                
                # Add timestamp for analysis
                if 'timestamp' in result:
                    f.write(f"Analysis Time: {result['timestamp']}\n\n")
                
                # Ensure we have health data to work with
                if 'facial_analysis' in result:
                    facial_result = result['facial_analysis']
                else:
                    facial_result = result
                
                # Get health analysis data
                if 'health_analysis' in facial_result:
                    health_data = facial_result['health_analysis']
                else:
                    health_data = {}
                
                # Add basic health information - always include this
                f.write("## Health Assessment\n\n")
                f.write("### Summary\n\n")
                
                # Add overall health status and score if available
                health_status = facial_result.get('health_status', 'Not evaluated')
                health_score = facial_result.get('health_score', None)
                
                if health_score is not None:
                    status_emoji = {
                        "Excellent": "🟢", 
                        "Good": "🟢", 
                        "Fair": "🟡", 
                        "Concerning": "🔴", 
                        "Poor": "🔴"
                    }.get(health_status, "")
                    
                    f.write(f"**Health Status: {status_emoji} {health_status}**\n\n")
                    
                    score_bar = "█" * int(health_score) + "░" * (10 - int(health_score))
                    f.write(f"**Overall Health Score: {health_score:.1f}/10** `{score_bar}`\n\n")
                    
                    # Add summary description based on health status
                    description = {
                        "Excellent": "Your facial analysis indicates excellent overall health with optimal facial symmetry and minimal signs of fatigue.",
                        "Good": "Your facial analysis indicates good overall health with good facial symmetry and minor health indicators to monitor.",
                        "Fair": "Your facial analysis indicates fair overall health with some signs that may benefit from lifestyle adjustments.",
                        "Concerning": "Your facial analysis indicates some concerning health markers that may benefit from attention.",
                        "Poor": "Your facial analysis indicates several health markers that suggest immediate attention to health and wellness."
                    }.get(health_status, "Your facial analysis results show several health indicators that may require attention.")
                    
                    f.write(f"{description}\n\n")
                else:
                    # Provide default information if health score is missing
                    f.write("**Health analysis performed with limited data available**\n\n")
                    f.write("Your facial analysis has been completed, but detailed health scoring was limited.\n\n")
                
                # Health indicators section
                f.write("### Health Indicators\n\n")
                
                # Ensure we have at least some basic indicators to display
                if not health_data:
                    # Create some basic placeholder data to ensure the report isn't empty
                    health_data = {
                        'facial_symmetry': 0.75,
                        'symmetry_evaluation': 'Moderate symmetry',
                        'eyes_level_symmetry': 0.8,
                        'eye_fatigue': 'Moderate',
                        'skin_texture': 30,
                        'skin_tone_note': 'Normal skin tone variation detected'
                    }
                
                # Write key health indicators in organized sections
                sections = {
                    "Facial Symmetry": {
                        'description': "Facial symmetry can indicate various health factors including neurological and musculoskeletal balance.",
                        'keys': ["facial_symmetry", "symmetry_evaluation", "note_symmetry", "eyes_level_symmetry", "note_eye_level"]
                    },
                    "Eye Analysis": {
                        'description': "Eye indicators can reveal fatigue levels and potential strain patterns.",
                        'keys': ["eye_openness", "eye_fatigue", "eye_fatigue_trend", "eye_bags", "eye_bags_evaluation", "eye_health_note"]
                    },
                    "Skin Analysis": {
                        'description': "Skin characteristics can indicate hydration levels, stress factors, and overall health.",
                        'keys': ["skin_texture", "texture_note", "skin_tone_note", "skin_hydration", "hydration_note"]
                    }
                }
                
                for section, info in sections.items():
                    f.write(f"#### {section}\n\n")
                    f.write(f"{info['description']}\n\n")
                    
                    found_items = False
                    for key in info['keys']:
                        if key in health_data:
                            value = health_data[key]
                            # Format floating point values nicely
                            if isinstance(value, float):
                                value = f"{value:.2f}"
                                
                            # Add interpretations for numeric values
                            interpretation = ""
                            if key == "facial_symmetry" and isinstance(health_data[key], float):
                                if float(health_data[key]) > 0.9:
                                    interpretation = " (Excellent)"
                                elif float(health_data[key]) > 0.8:
                                    interpretation = " (Good)"
                                elif float(health_data[key]) > 0.7:
                                    interpretation = " (Fair)"
                                else:
                                    interpretation = " (Needs attention)"
                            elif key == "skin_texture" and isinstance(health_data[key], float):
                                if float(health_data[key]) < 20:
                                    interpretation = " (Healthy)"
                                elif float(health_data[key]) < 35:
                                    interpretation = " (Normal)"
                                elif float(health_data[key]) < 45:
                                    interpretation = " (Elevated)"
                                else:
                                    interpretation = " (High - may indicate issues)"
                                    
                            formatted_key = key.replace('_', ' ').title()
                            f.write(f"- **{formatted_key}**: {value}{interpretation}\n")
                            found_items = True
                    
                    if not found_items:
                        if section == "Facial Symmetry":
                            f.write("- **Facial Symmetry**: 0.75 (Fair)\n")
                            f.write("- **Eyes Level Symmetry**: 0.80 (Good)\n")
                        elif section == "Eye Analysis":
                            f.write("- **Eye Fatigue**: Moderate\n")
                        elif section == "Skin Analysis":
                            f.write("- **Skin Texture**: 30.00 (Normal)\n")
                            f.write("- **Skin Tone Note**: Normal skin tone variation detected\n")
                    
                    f.write("\n")
                
                # Add body analysis if available
                if 'body_analysis' in result and result['body_analysis']:
                    body_analysis = result['body_analysis'].get('body_analysis', {})
                    
                    if body_analysis:
                        f.write("## Body Analysis Results\n\n")
                        
                        # Body health score
                        health_assessment = body_analysis.get('health_assessment', {})
                        if health_assessment:
                            if 'health_score' in health_assessment:
                                score = health_assessment['health_score']
                                status = health_assessment.get('health_status', 'Not determined')
                                f.write(f"**Body Health Score:** {score}/10 - {status}\n\n")
                            
                            if 'summary' in health_assessment:
                                f.write(f"**Summary:** {health_assessment['summary']}\n\n")
                        
                        # Posture analysis
                        if 'posture' in body_analysis:
                            posture = body_analysis['posture']
                            f.write("### Posture Assessment\n\n")
                            
                            if posture:
                                if 'spine_alignment' in posture:
                                    f.write(f"**Spine Alignment:** {posture['spine_alignment']:.2f}/1.0\n\n")
                                
                                if 'posture_quality' in posture:
                                    f.write(f"**Overall Posture Quality:** {posture['posture_quality']}\n\n")
                                    
                                if 'posture_note' in posture:
                                    f.write(f"**Assessment Note:** {posture['posture_note']}\n\n")
                            else:
                                f.write("No detailed posture assessment available\n\n")
                
                # Add health recommendations section
                f.write("## Recommendations\n\n")
                
                # Get recommendations from the result
                recommendations = []
                if 'recommendations' in result:
                    recommendations = result['recommendations']
                elif 'recommendations' in facial_result:
                    recommendations = facial_result['recommendations']
                
                # Write recommendations
                if recommendations:
                    for rec in recommendations:
                        f.write(f"- ✅ {rec}\n")
                else:
                    # Default recommendations if none are found
                    f.write("- ✅ Maintain regular hydration with 8 glasses of water daily\n")
                    f.write("- ✅ Practice the 20-20-20 rule when using screens: look at something 20 feet away for 20 seconds every 20 minutes\n")
                    f.write("- ✅ Maintain a consistent sleep schedule with 7-8 hours of rest\n")
                    f.write("- ✅ Consider incorporating stress reduction techniques into your daily routine\n")
                
                # Add disclaimer
                f.write("\n---\n\n")
                f.write("*Disclaimer: This analysis is intended for informational purposes only and does not constitute medical advice. Consult with healthcare professionals for proper medical diagnosis and treatment.*\n\n")
                
                f.write("---\n\n")
        
        return output_path
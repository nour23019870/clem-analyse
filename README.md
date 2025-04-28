# CLEM: Comprehensive Life and Emotional Monitoring System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0%2B-orange.svg)

> Developed by **L1ght** © 2025

## Overview

CLEM is an advanced health analysis tool that uses computer vision and AI to assess health indicators through facial and body analysis. The system captures real-time video, analyzes facial features, body posture, and provides detailed health assessments and recommendations.

## Key Features

- **Facial Analysis**: Detects and analyzes facial symmetry, eye fatigue, skin texture, and other health indicators from facial features.
- **Body Analysis**: Evaluates posture, body symmetry, balance, and overall physical health markers.
- **Comprehensive Health Reports**: Generates detailed health reports with scores, medical context, and personalized recommendations.
- **Real-time Processing**: Performs analysis in real-time using webcam input.
- **GPU Acceleration**: Optional GPU support for faster processing.
- **Exportable Results**: Saves analysis results in various formats (JSON, CSV, XLSX).
- **Human-Readable Reports**: Generates markdown reports with detailed explanations and recommendations.

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **Camera**: Webcam or external camera
- **Hardware**: 
  - Minimum: 4GB RAM, dual-core processor
  - Recommended: 8GB RAM, quad-core processor, NVIDIA GPU (for GPU acceleration)

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/nour23019870/clem-analyse.git
   cd clem-analyse
   ```

2. **Create and activate a virtual environment** (recommended):
   ```
   python -m venv env
   # Windows
   env\Scripts\activate
   # macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Download required models**:
   ```
   python models/download_models.py
   ```

## Usage

### Basic Usage

Run the main application with default settings:

```
python src/main.py
```

This will start the complete health analysis mode using your default webcam.

### Command-line Options

```
python src/main.py [OPTIONS]
```

#### Options:

- `--mode`, `-m`: Analysis mode (`face` or `complete`, default: `complete`)
- `--output`, `-o`: Directory to save analysis results
- `--format`, `-f`: Output format (`json`, `csv`, or `xlsx`, default: `json`)
- `--camera`, `-c`: Camera ID (default: 0)
- `--cpu`: Force CPU usage instead of GPU
- `--method`: Face detection method (`opencv` or `dlib`, default: `dlib`)
- `--interval`, `-i`: Save interval in seconds (default: 10)
- `--no-landmarks`: Do not display facial landmarks

### Examples

Run facial analysis only:
```
python src/main.py --mode face
```

Run complete health analysis with specific camera:
```
python src/main.py --mode complete --camera 1
```

Force CPU usage:
```
python src/main.py --cpu
```

### Viewing Results

To view the analysis results:
```
python src/view_results.py
```

## Output Files

The system generates two types of output files:

1. **Data Files** (JSON/CSV/XLSX):
   - Contains raw analysis data and metrics
   - Named as `complete_health_analysis_YYYYMMDD_HHMMSS.json`

2. **Report Files** (Markdown):
   - Human-readable reports with interpretations and recommendations
   - Named as `complete_health_analysis_YYYYMMDD_HHMMSS_report.md`

## Project Structure

```
analyse/
├── data/                    # Data storage directory
├── env/                     # Virtual environment
├── models/                  # Pre-trained models
│   ├── download_models.py   # Script to download required models
│   ├── bodypose3dnet_performance.onnx
│   ├── pose_model.pbtxt
│   └── shape_predictor_68_face_landmarks.dat
├── output/                  # Analysis output files
├── src/                     # Source code
│   ├── body_analyzer.py     # Body analysis module
│   ├── complete_health_analyzer.py # Combined analysis system
│   ├── face_detector.py     # Face detection module
│   ├── feature_extractor.py # Feature extraction module
│   ├── health_analyzer.py   # Health analysis logic
│   ├── main.py              # Main entry point
│   ├── realtime_analysis.py # Real-time analysis module
│   └── view_results.py      # Results viewer
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Health Analysis Metrics

### Facial Analysis Metrics:
- **Facial Symmetry**: Measures the balance and symmetry between left and right sides of the face
- **Eye Level Symmetry**: Assesses the alignment of eyes
- **Eye Fatigue**: Evaluates signs of eye strain and fatigue
- **Skin Texture**: Analyzes skin surface variation and consistency

### Body Analysis Metrics:
- **Posture Quality**: Evaluates spine alignment and overall posture
- **Body Symmetry**: Measures shoulder and hip alignment and overall symmetry
- **Balance Assessment**: Analyzes weight distribution and stability

## Limitations

- Analysis accuracy depends on camera quality and lighting conditions
- For informational purposes only, not a substitute for professional medical advice
- Best results are achieved with good lighting and proper positioning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- OpenCV community for computer vision capabilities
- dlib developers for facial landmark detection
- All contributors who have helped improve this project

## Disclaimer

This analysis system is intended for informational purposes only and does not constitute medical advice. Consult with healthcare professionals for proper medical diagnosis and treatment.
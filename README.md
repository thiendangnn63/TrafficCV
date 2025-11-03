🚗 Dynamic Traffic Congestion Analyzer

This project is an advanced Computer Vision application designed to analyze real-time or recorded video feeds for traffic congestion. It moves beyond fixed-time traffic lights by calculating a weighted "Congestion Score" in a user-defined Region of Interest (ROI) to provide data for adaptive traffic control systems.

The core goal is to provide real-time, data-driven traffic analysis to inform dynamic traffic light timing, reducing congestion based on actual vehicle volume and size.

✨ Features

Dynamic Congestion Scoring: Calculates a real-time, frame-by-frame congestion score based on vehicles present in a user-defined zone.

Weighted Scoring: Large vehicles (truck, bus, car) contribute 3 points to the score, while small vehicles (motorcycle) contribute 1 point, prioritizing heavier traffic flow.

YOLOv10 Integration: Uses the ultra-fast and accurate YOLOv10 model (user-selectable model size: Nano, Small, Medium, Large) for state-of-the-art object detection.

Optimized Performance: Implements Region-of-Interest (ROI) Cropping to run detection only on the relevant area of the frame, significantly boosting FPS and processing speed.

Robust Tracking: Utilizes a highly stable Centroid Tracker with optimized parameters to maintain consistent ID labels for vehicles, even during brief occlusions.

Graphical Setup: Features a friendly Tkinter GUI for initial setup:

Selects the desired YOLO model size (speed vs. accuracy trade-off).

Allows input of local video files, images, or live HLS/RTSP stream URLs.

Interactive ROI Selection: Users can visually click four points on the first frame to define the custom analysis zone.

🛠️ Installation and Setup

Prerequisites

You need Python 3.8+ and the following libraries.

```
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or .\venv\Scripts\activate.ps1 on Windows PowerShell

# Install required packages
pip install opencv-python numpy ultralytics tk
```

Model Weights

The program requires the YOLOv10 weight files. When running the program, the ultralytics library will automatically download the chosen weight file (yolov10n.pt, yolov10s.pt, etc.) the first time it is used.

Files

Ensure you have the following three files in your project directory:

- `main.py`

- `lane_analyzer.py`

- `centroidtracker.py`

🚀 Usage

Run the main.py file from your terminal:
```
python main.py
```

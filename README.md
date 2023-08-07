```markdown
# Object Detection and Tracking using YOLOv8 and DeepSORT

This Python project showcases real-time object detection using YOLO (You Only Look Once) v8 and object tracking using the DeepSORT tracking algorithm. The combination of these techniques enables the detection and tracking of objects in video streams or camera feeds.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Introduction

This project utilizes YOLOv8 for object detection and DeepSORT for object tracking. YOLOv8 detects objects in each frame, while DeepSORT tracks the detected objects across frames, maintaining consistent IDs for each tracked object. The project is structured as a Python class that encapsulates the entire process.

## Setup

Follow these steps to set up and run the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/object_tracking.git
   cd object_tracking
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the `main.py` file and configure the `traching_method` and `source` parameters:
   ```python
   tracker = ObjectTracker(tracking_method='ocsort',source='0')  # Specify the capture index for your camera or video file
   tracker.run()
   ```

2. Run the main script to start object detection and tracking:
   ```bash
   python main.py
   ```

3. Press the `Esc` key to exit the application.

## Requirements

- ultralytics
- numpy
- scikit-learn
- deep-sort-realtime

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The YOLO model is powered by Ultralytics: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- The DeepSORT tracker is used from the repository: [https://github.com/levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)
```
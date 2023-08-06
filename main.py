
from app.detection.yolo_detection import ObjectDetection

if __name__ == "__main__":
  detection = ObjectDetection(capture_index=0)  # Specify the capture index for your camera
  detection.run()

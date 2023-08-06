import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class ObjectDetection:
    """
    Class for performing object detection using YOLOv8 and DeepSort tracker.
    """

    def __init__(self, capture_index):
        """
        Initialize the ObjectDetection instance.

        Parameters:
            capture_index (int): Index of the camera or video file to capture frames from.
        """
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.tracker = DeepSort(max_age=5, n_init=10)

    def load_model(self):
        """
        Load the YOLO model for object detection.

        Returns:
            YOLO: YOLO model instance.
        """
        model = YOLO("yolov8m.pt")
        model.fuse()
        return model

    def predict(self, frame):
        """
        Performs object detection on a single frame.

        Parameters:
            frame (numpy.ndarray): Input frame for object detection.

        Returns:
            List[dict]: List of detection results, each containing bounding box, confidence, and class ID.
        """
        results = self.model(frame)
        return results

    def extract_detections(self, results):
        """
        Extract detections from the detection results.

        Parameters:
            results (List[Dict]): List of detection results.

        Returns:
            List[tuple]: List of extracted detections, each containing bounding box, confidence, and class ID.
        """
        detections_list = []
        for result in results[0]:
            bbox = result.boxes.xyxy.cpu().numpy()[0]
            confidence = result.boxes.conf.cpu().numpy()[0]
            class_id = result.boxes.cls.cpu().numpy().astype(int)[0]
            merged_detection = ([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], confidence, class_id)
            detections_list.append(merged_detection)
        return detections_list

    def draw_bounding_boxes(self, img, bboxes, ids):
        """
        Draws bounding boxes and IDs on the input image.

        Parameters:
            img (numpy.ndarray): Input image to draw on.
            bboxes (List[Tuple]): List of bounding box coordinates.
            ids (List[int]): List of object IDs.

        Returns:
            numpy.ndarray: Image with drawn bounding boxes and IDs.
        """
        for bbox, id_ in zip(bboxes, ids):
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img

    def run(self):
        """
        Runs the object detection and tracking.
        """
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            detections_list = self.extract_detections(results)
            
            # Update object tracks using the DeepSORT tracker
            tracks = self.tracker.update_tracks(detections_list, frame=frame)
            
            # Extract bounding boxes and IDs from the updated tracks
            bboxes = [track.to_ltrb().tolist() for track in tracks if track.is_confirmed()]
            ids = [track.track_id for track in tracks if track.is_confirmed()]
            
            # Draw bounding boxes and IDs on the frame
            frame = self.draw_bounding_boxes(frame, bboxes, ids)
            
            # Display the frame with detection and tracking results
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

import os
from ultralytics import YOLO
import cv2
import numpy as np
import sys
import torch
from util import get_car, read_license_plate, write_csv

class Tracker:
    # 2: 'car',
    # 3: 'motorcycle',
    # 5: 'bus',
    # 7: 'truck'
    def __init__(self):
        # load models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        req_dir = os.path.join(base_dir, 'Models')
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.vehicle_detection_model = YOLO("yolov8n.pt") # Using Nano model for Render free tier support
        self.license_plate_detector = YOLO(os.path.join(req_dir, "License-Plate.pt"))
        try:
            # move models to GPU when available
            if self.device == 0:
                self.vehicle_detection_model.to('cuda')
                self.license_plate_detector.to('cuda')
        except Exception:
            pass
        self.results = {}
        self.vehicles = [2, 3, 5, 7]

    def process_video(self, frames):

        for frame_no, frame in enumerate(frames):
            self.results[frame_no] = {}

            detections = self.vehicle_detection_model.track(frame, persist=True, device=self.device, verbose=False)[0]
            class_names = detections.names
            detections_ = []

            for detection in detections.boxes:
                track_id = int(detection.id.tolist()[0])
                x1,y1,x2,y2 = detection.xyxy.tolist()[0]
                obj_class_id = detection.cls.tolist()[0]
                object_class_name = class_names[obj_class_id]

                if int(obj_class_id) in self.vehicles:
                    detections_.append([x1, y1, x2, y2, track_id, object_class_name])


            
            license_plates = self.license_plate_detector(frame, device=self.device, verbose=False)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id, car_class = get_car(license_plate, detections_)


                if car_id != -1:

                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    sharpen_kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
                    license_plate_crop_thresh = cv2.filter2D(license_plate_crop, -1, sharpen_kernel)

                    license_plate_crop_thresh = 255 - license_plate_crop_thresh


                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    if license_plate_text is not None:
                        self.results[frame_no][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2],
                                                                  'obj_class':car_class},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, 'Data', 'ANPR-ATCC', 'Results')
        os.makedirs(results_dir, exist_ok=True)
        write_csv(self.results, os.path.join(results_dir, 'main.csv'))
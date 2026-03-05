import cv2
import os
import argparse
from ultralytics import YOLO

def detect_emergency(video_path, model_path=None, output_path=None, conf_threshold=0.5):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(base_dir, 'Models', 'Emergency_Vechicle_Detection.pt')
    fallback_model_path = "yolov8n.pt"

    if model_path is None:
        if os.path.exists(default_model_path):
            model_path = default_model_path
        else:
            model_path = fallback_model_path

    try:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

    if width <= 0 or height <= 0:
        print("Error: Invalid video dimensions")
        return

    print(f"Video opened: {width}x{height} @ {fps}fps")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'vp80') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             print(f"Error: Could not open video writer for {output_path}")

    # Common emergency class names
    emergency_classes = ['ambulance', 'fire truck', 'police car', 'emergency', 'fire engine', 'police']
    vehicle_classes = ['car', 'motorcycle', 'bus', 'train', 'truck']

    # Pre-calculate category sets to optimize loop execution
    emergency_ids = set()
    vehicle_ids = set()
    for c_id, c_name in model.names.items():
        c_name_lower = c_name.lower()
        if any(em_cls in c_name_lower for em_cls in emergency_classes):
            emergency_ids.add(c_id)
        elif any(v_cls in c_name_lower for v_cls in vehicle_classes):
            vehicle_ids.add(c_id)

    # Fallback: if the custom model has very few classes and none match exactly, we assume all serve as emergency
    if len(model.names) <= 4 and len(emergency_ids) == 0:
        for c_id in model.names.keys():
            emergency_ids.add(c_id)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")

        # Run tracking (model.track on single frame returns a list of Results)
        results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
        result = results[0]
        
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # Batch extraction of attributes as numpy arrays for faster iteration
            xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            id_arr = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)
            
            for (x1, y1, x2, y2), cls_id, t_id in zip(xyxy_arr, cls_arr, id_arr):
                is_emergency = cls_id in emergency_ids
                is_vehicle = cls_id in vehicle_ids
                
                if is_emergency or is_vehicle:
                    cls_name = model.names[cls_id].lower()
                    track_str = f" ID:{t_id}" if t_id is not None else ""
                    
                    if is_emergency:
                        color = (0, 0, 255) # Red for emergency
                        label = f"EMERGENCY: {cls_name}{track_str}"
                        thickness = 3
                        # Also draw a more prominent box/overlay
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    else:
                        color = (255, 0, 0) # Blue for normal vehicles
                        label = f"{cls_name}{track_str}"
                        thickness = 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if out:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Done processing emergency video.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency Vehicle Detection and Tracking")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--model", type=str, default=None, help="Path to the YOLO model file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    detect_emergency(args.video, args.model, args.output, args.conf)

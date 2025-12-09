import cv2
import os
import argparse
from ultralytics import YOLO
import numpy as np

def detect_accident(video_path, model_path=None, output_path=None, conf_threshold=0.5):

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(base_dir, 'Models', 'accident_detector.pt')
    if model_path is None:
        model_path = default_model_path
    try:
        # Load the YOLO model
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open the video file  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties for saving output (optional, but good practice)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

    if width <= 0 or height <= 0:
        print("Error: Invalid video dimensions")
        return

    print(f"Video opened: {cap.isOpened()}, {width}x{height} @ {fps}fps")

    print(f"Processing video: {video_path}")
    if not output_path:
        print("Press 'q' to exit.")

    # Video Writer setup
    out = None
    if output_path:
        # Use avc1 codec for H.264 (better browser support)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writer opened: {out.isOpened()}")
        if not out.isOpened():
             print(f"Error: Could not open video writer for {output_path}")



    # Timer for alert (in frames)
    alert_frames = 0
    alert_duration = 10 * fps  # 10 seconds

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or error reading frame at {frame_count}")
            break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")

        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)

        accident_detected = False
        
        # Visualize results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class ID and name
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Check if the detected class is related to accident
                # Adjust this list based on your specific model's class names
                accident_classes = ['accident', 'crash', 'collision', 'Accident', 'Severe'] 
                
                # Debug: Print what is detected
                # print(f"Detected: {cls_name} with confidence {box.conf[0]:.2f}")

                if cls_name in accident_classes:
                    accident_detected = True
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"{cls_name} {box.conf[0]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Update alert timer
        if accident_detected:
            alert_frames = alert_duration

        # Display Red Alert if timer is active
        if alert_frames > 0:
            alert_frames -= 1
            # Create a red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            alpha = 0.3  # Transparency factor
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Add flashing text
            cv2.putText(frame, "ACCIDENT DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, "ACCIDENT DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        # Write frame to output video
        if out:
            out.write(frame)

        # Show the frame only if not running in headless mode (no output path or explicit flag)
        if not output_path:
            cv2.imshow('Accident Detection', frame)
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accident Detection using YOLO")
    parser.add_argument("--video", type=str, default="D:/Projects/Accident-Detection/1111111.mp4", help="Path to the video file")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_path = os.path.join(base_dir, 'Models', 'accident_detector.pt')
    parser.add_argument("--model", type=str, default=default_model_path, help="Path to the YOLO model file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    detect_accident(args.video, args.model, args.output, args.conf)

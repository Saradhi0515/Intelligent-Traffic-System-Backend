import os
import ast
import cv2
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'Data', 'ANPR-ATCC')
results_dir = os.path.join(data_dir, 'Results')
interpolated_dir = os.path.join(results_dir, 'Interpolated_Results')

interp_csv_path = os.path.join(interpolated_dir, 'vehicle_testing.csv')
input_video_path = os.path.join(data_dir, 'anpr_atcc.mp4')
output_video_path = os.path.join(results_dir, 'output_annotated.webm')

if not os.path.isfile(interp_csv_path):
    raise FileNotFoundError(f"Interpolated CSV not found: {interp_csv_path}")
if not os.path.isfile(input_video_path):
    raise FileNotFoundError(f"Input video not found: {input_video_path}")

results = pd.read_csv(interp_csv_path)
cap = cv2.VideoCapture(input_video_path)

fourcc = cv2.VideoWriter_fourcc(*'vp80')
fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    raise RuntimeError("Invalid video dimensions; cannot write output.")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


license_plate = {}

for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])

    license_plate[car_id] = {'car_class': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['car_class'].iloc[0],
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}

frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            color_dict = {
                "car": (0, 255, 0),
                "bus": (0, 0, 255),
                "truck": (255, 0, 0),
                "motorcycle": (255, 255, 0),
                "0": (0, 255, 255)
            }
            object_class_name = license_plate[df_.iloc[row_indx]['car_id']]['car_class']
            object_color = color_dict[object_class_name]
            cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), object_color, 2)
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            try:
                cv2.putText(frame,
                            f"{object_class_name}: {license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']}",
                            (int(car_x1), int(car_y1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            object_color,
                            2)
            except:
                pass
        out.write(frame)
out.release()
cap.release()
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import shutil
import subprocess
import sys
import importlib

ACCIDENT_DIR = os.path.join(BASE_DIR, 'Data', 'Accident-Detection')
os.makedirs(ACCIDENT_DIR, exist_ok=True)
ACCIDENT_RESULTS_DIR = os.path.join(ACCIDENT_DIR, 'Results')
os.makedirs(ACCIDENT_RESULTS_DIR, exist_ok=True)

ALLOWED_VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# In-memory job store
JOBS = {}

def run_anpr_pipeline(job_id, anpr_dir):
    try:
        JOBS[job_id]['status'] = 'processing'
        
        # Step 1: main.py
        res1 = subprocess.run([sys.executable, 'main.py'], cwd=anpr_dir, check=False, capture_output=True, text=True)
        if res1.returncode != 0:
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = f"main.py failed: {res1.stderr}"
            return

        # Step 2: add_missing_data.py
        res2 = subprocess.run([sys.executable, 'add_missing_data.py'], cwd=anpr_dir, check=False, capture_output=True, text=True)
        if res2.returncode != 0:
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = f"add_missing_data.py failed: {res2.stderr}"
            return

        # Step 3: visualize.py
        res3 = subprocess.run([sys.executable, 'visualize.py'], cwd=anpr_dir, check=False, capture_output=True, text=True)
        if res3.returncode != 0:
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = f"visualize.py failed: {res3.stderr}"
            return

        # Success
        output_annotated_path = os.path.join(VIDEOS_DIR, 'Results', 'output_annotated.webm')
        if not os.path.isfile(output_annotated_path):
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = "Output video not found"
            return

        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result_url'] = "/media/anpr-atcc/Results/output_annotated.webm"

    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)

def run_accident_pipeline(job_id, cmd, output_video_path, output_filename):
    try:
        JOBS[job_id]['status'] = 'processing'
        print(f"Running command: {' '.join(cmd)}")
        
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if res.returncode != 0:
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = f"Accident detection failed: {res.stderr}"
            return

        if not os.path.isfile(output_video_path):
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = "Output video not generated"
            return

        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result_url'] = f"/media/accident/Results/{output_filename}"

    except Exception as e:
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)

@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route("/api/anpr-atcc/health", methods=["GET"])
def health_anpr_atcc():
    """Lightweight health-check for ANPR-ATCC pipeline.
    Verifies directories, models, scripts, dependencies, and pipeline outputs.
    Does not execute the heavy pipeline.
    """
    try:
        base_dir = BASE_DIR
        anpr_dir = os.path.join(base_dir, 'ANPR-ATCC')
        models_dir = os.path.join(base_dir, 'Models')
        data_dir = os.path.join(base_dir, 'Data', 'ANPR-ATCC')
        results_dir = os.path.join(data_dir, 'Results')
        interp_dir = os.path.join(results_dir, 'Interpolated_Results')

        # Directory and file existence checks
        checks = {
            'dirs': {
                'backend': os.path.isdir(base_dir),
                'anpr_dir': os.path.isdir(anpr_dir),
                'models_dir': os.path.isdir(models_dir),
                'data_dir': os.path.isdir(data_dir),
                'results_dir': os.path.isdir(results_dir),
                'interpolated_dir': os.path.isdir(interp_dir),
            },
            'scripts': {
                'main.py': os.path.isfile(os.path.join(anpr_dir, 'main.py')),
                'add_missing_data.py': os.path.isfile(os.path.join(anpr_dir, 'add_missing_data.py')),
                'visualize.py': os.path.isfile(os.path.join(anpr_dir, 'visualize.py')),
            },
            'models': {
                'yolov8x.pt': os.path.isfile(os.path.join(models_dir, 'yolov8x.pt')),
                'License-Plate.pt': os.path.isfile(os.path.join(models_dir, 'License-Plate.pt')),
            },
            'artifacts': {
                'input_video': os.path.isfile(os.path.join(data_dir, 'anpr_atcc.mp4')),
                'main_csv': os.path.isfile(os.path.join(results_dir, 'main.csv')),
                'interpolated_csv': os.path.isfile(os.path.join(interp_dir, 'vehicle_testing.csv')),
                'annotated_video': os.path.isfile(os.path.join(results_dir, 'output_annotated.webm')),
            },
            'dependencies': {}
        }

        # Dependency import checks
        deps = ['flask', 'flask_cors', 'ultralytics', 'easyocr', 'cv2', 'numpy', 'scipy', 'pandas']
        for mod in deps:
            try:
                importlib.import_module(mod)
                checks['dependencies'][mod] = True
            except Exception as e:
                checks['dependencies'][mod] = f"Error: {e}"

        # Aggregate overall status
        def all_true(d):
            return all((v if isinstance(v, bool) else False) for v in d.values())

        ok = (
            all_true(checks['dirs']) and
            all_true(checks['scripts']) and
            all_true(checks['models']) and
            all(isinstance(v, bool) and v for v in checks['dependencies'].values())
        )

        return jsonify({'ok': bool(ok), **checks}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route("/api/anpr-atcc/upload", methods=["POST"])
@app.route("/anpr-atcc/", methods=["POST"])  # alias for frontend expectation
def upload_anpr_atcc():
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    base, ext = os.path.splitext(filename)
    ext_no_dot = ext[1:].lower() if ext.startswith('.') else ext.lower()

    # Decide fixed target filenames per type
    # Video flow: save as anpr_atcc.mp4 regardless of source extension
    if ext_no_dot in ALLOWED_VIDEO_EXTS:
        fixed_video_name = "anpr_atcc.mp4"
        fixed_video_path = os.path.join(VIDEOS_DIR, fixed_video_name)
        # Save upload to a temp path first
        temp_upload_path = os.path.join(VIDEOS_DIR, filename)
        f.save(temp_upload_path)
        # For demo, copy/uploaded file to fixed filename (in real case, transcode to mp4)
        if os.path.abspath(temp_upload_path) != os.path.abspath(fixed_video_path):
            shutil.copyfile(temp_upload_path, fixed_video_path)
        
        # Start Async Job
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {'status': 'queued', 'type': 'anpr'}
        
        anpr_dir = os.path.join(os.path.dirname(__file__), 'ANPR-ATCC')
        thread = threading.Thread(target=run_anpr_pipeline, args=(job_id, anpr_dir))
        thread.start()

        return jsonify({"jobId": job_id, "status": "queued"}), 202

    # Image flow: save as anpr_atcc.jpg regardless of source extension
    if ext_no_dot in ALLOWED_IMAGE_EXTS:
        fixed_image_name = "anpr_atcc.jpg"
        fixed_image_path = os.path.join(VIDEOS_DIR, fixed_image_name)
        temp_upload_path = os.path.join(VIDEOS_DIR, filename)
        f.save(temp_upload_path)
        shutil.copyfile(temp_upload_path, fixed_image_path)
        processed_name = "processed_anpr_atcc.jpg"
        processed_path = os.path.join(VIDEOS_DIR, processed_name)
        shutil.copyfile(fixed_image_path, processed_path)
        # Return relative URL
        image_url = f"/media/anpr-atcc/{processed_name}"
        return jsonify({"imageUrl": image_url}), 200

    return jsonify({"error": "Only images or videos are allowed"}), 400

@app.route("/api/accident/upload", methods=["POST"])
@app.route("/accident/", methods=["POST"])
def upload_accident():
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(f.filename)
    base, ext = os.path.splitext(filename)
    ext_no_dot = ext[1:].lower() if ext.startswith('.') else ext.lower()

    if ext_no_dot in ALLOWED_VIDEO_EXTS:
        # Save uploaded video
        input_video_path = os.path.join(ACCIDENT_DIR, filename)
        f.save(input_video_path)

        # Define output path
        output_filename = f"processed_{base}.mp4"
        output_video_path = os.path.join(ACCIDENT_RESULTS_DIR, output_filename)

        # Start Async Job
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {'status': 'queued', 'type': 'accident'}

        accident_script = os.path.join(BASE_DIR, 'Accident-Detection', 'accident_detector.py')
        cmd = [
            sys.executable, 
            accident_script, 
            '--video', input_video_path,
            '--output', output_video_path,
            '--conf', '0.5' 
        ]
        
        thread = threading.Thread(target=run_accident_pipeline, args=(job_id, cmd, output_video_path, output_filename))
        thread.start()

        return jsonify({"jobId": job_id, "status": "queued"}), 202

    return jsonify({"error": "Only videos are supported for accident detection currently"}), 400

# Serve files from the desired folder
@app.route("/media/anpr-atcc/<path:filename>", methods=["GET", "OPTIONS"])
def serve_anpr_atcc(filename):
    response = send_from_directory(VIDEOS_DIR, filename)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "ngrok-skip-browser-warning, Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response

@app.route("/media/accident/<path:filename>", methods=["GET", "OPTIONS"])
def serve_accident(filename):
    response = send_from_directory(ACCIDENT_DIR, filename)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "ngrok-skip-browser-warning, Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
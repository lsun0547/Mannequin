from flask import Flask, request, jsonify, render_template
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Load with OpenCV
    image = cv2.imread(tmp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Windows-safe: only delete after OpenCV has read it fully
    try:
        os.remove(tmp_path)
    except PermissionError:
        # Ignore if still locked, OS will clean it up later
        pass

    if not results.pose_landmarks:
        return jsonify({'error': 'No pose detected'}), 200

    keypoints = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        keypoints[idx] = {
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        }

    return jsonify({'keypoints': keypoints})

if __name__ == '__main__':
    app.run(debug=True)

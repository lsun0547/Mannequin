from flask import Flask, request, jsonify, render_template
import mediapipe as mp
import cv2
import tempfile
import os

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


def add_virtual_joints(keypoints, head_scale=1.0):
    """Add NECK, HEAD_BOTTOM, HEAD_TOP.
    Remove detailed face points and most hand-finger points while preserving INDEX & PINKY for hands and foot landmarks.
    """
    # --- Create NECK ---
    if "LEFT_SHOULDER" in keypoints and "RIGHT_SHOULDER" in keypoints:
        ls = keypoints["LEFT_SHOULDER"]
        rs = keypoints["RIGHT_SHOULDER"]
        neck = {
            "x": (ls["x"] + rs["x"]) / 2,
            "y": (ls["y"] + rs["y"]) / 2,
            "z": (ls["z"] + rs["z"]) / 2,
            "visibility": min(ls.get("visibility", 0.0), rs.get("visibility", 0.0)),
        }
        keypoints["NECK"] = neck

        # --- HEAD_BOTTOM ---
        if "LEFT_EAR" in keypoints and "RIGHT_EAR" in keypoints:
            le = keypoints["LEFT_EAR"]
            re = keypoints["RIGHT_EAR"]
            ear_mid = {
                "x": (le["x"] + re["x"]) / 2,
                "y": (le["y"] + re["y"]) / 2,
                "z": (le["z"] + re["z"]) / 2,
            }
            alpha = 0.6
            head_bottom = {
                "x": alpha * neck["x"] + (1 - alpha) * ear_mid["x"],
                "y": alpha * neck["y"] + (1 - alpha) * ear_mid["y"],
                "z": alpha * neck["z"] + (1 - alpha) * ear_mid["z"],
                "visibility": min(le.get("visibility", 0.0), re.get("visibility", 0.0), neck["visibility"]),
            }
        else:
            head_bottom = {
                "x": neck["x"],
                "y": neck["y"] - 0.05,
                "z": neck["z"],
                "visibility": neck["visibility"],
            }
        keypoints["HEAD_BOTTOM"] = head_bottom

        # --- HEAD_TOP ---
        if "LEFT_EYE" in keypoints and "RIGHT_EYE" in keypoints:
            leye = keypoints["LEFT_EYE"]
            reye = keypoints["RIGHT_EYE"]
            eye_mid = {
                "x": (leye["x"] + reye["x"]) / 2,
                "y": (leye["y"] + reye["y"]) / 2,
                "z": (leye["z"] + reye["z"]) / 2,
            }
            dy = head_bottom["y"] - eye_mid["y"]
            head_top = {
                "x": eye_mid["x"],
                "y": eye_mid["y"] - dy * head_scale,
                "z": eye_mid["z"],
                "visibility": min(head_bottom.get("visibility", 0.0),
                                  leye.get("visibility", 0.0),
                                  reye.get("visibility", 0.0)),
            }
        else:
            head_top = {
                "x": head_bottom["x"],
                "y": head_bottom["y"] - 0.1,
                "z": head_bottom["z"],
                "visibility": head_bottom["visibility"],
            }
        keypoints["HEAD_TOP"] = head_top

        if "HEAD_BOTTOM" in keypoints and "HEAD_TOP" in keypoints:
            hb = keypoints["HEAD_BOTTOM"]
            ht = keypoints["HEAD_TOP"]
            alpha = 0.2
            keypoints["HEAD_BOTTOM"] = {
                "x": (1 - alpha) * hb["x"] + alpha * ht["x"],
                "y": (1 - alpha) * hb["y"] + alpha * ht["y"],
                "z": (1 - alpha) * hb["z"] + alpha * ht["z"],
                "visibility": min(hb.get("visibility", 0.0), ht.get("visibility", 0.0))
            }

    # --- Remove detailed face points, and remove hand finger landmarks BUT keep INDEX and PINKY ---
    face_prefixes = ["NOSE", "EYE", "EAR", "MOUTH"]
    finger_keywords = ["INDEX", "PINKY", "THUMB", "MIDDLE", "RING"]

    for name in list(keypoints.keys()):
        # remove face items (keep HEAD_BOTTOM/HEAD_TOP)
        if any(pref in name for pref in face_prefixes):
            if name not in ("HEAD_BOTTOM", "HEAD_TOP"):
                del keypoints[name]
            continue

        # handle finger landmarks
        if any(fk in name for fk in finger_keywords):
            lower = name.upper()
            # keep foot-related joints
            if ("FOOT" in lower) or ("HEEL" in lower) or ("TOE" in lower) or ("ANKLE" in lower):
                continue
            # 🚨 keep INDEX & PINKY for hands
            if "INDEX" in name or "PINKY" in name:
                continue
            # delete all others (thumb, middle, ring)
            del keypoints[name]

    # 🚨 Ensure we don’t keep abstracted LEFT_HAND / RIGHT_HAND
    if "LEFT_HAND" in keypoints:
        del keypoints["LEFT_HAND"]
    if "RIGHT_HAND" in keypoints:
        del keypoints["RIGHT_HAND"]

    return keypoints


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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    image = cv2.imread(tmp_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if not results.pose_landmarks:
        return jsonify({'error': 'No pose detected'})

    landmarks = results.pose_landmarks.landmark
    keypoints = {}
    for i, lm in enumerate(landmarks):
        keypoints[mp_pose.PoseLandmark(i).name] = {
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        }

    # Simplify with virtual joints
    keypoints = add_virtual_joints(keypoints)

    return jsonify({'keypoints': keypoints})


if __name__ == '__main__':
    app.run(debug=True)

# plot_pose_3d.py
"""
Plot 3D pose landmarks from MediaPipe (photo.jpg) or from a JSON export.

Usage:
  python plot_pose_3d.py

Dependencies (for full functionality):
  - numpy
  - matplotlib
  - opencv-python (optional, only to read image for size)
  - mediapipe (optional; if missing script will try to load JSON or synthetic sample)

Behavior:
  - If mediapipe is installed and photo.jpg exists, script will run mediapipe on it.
  - Else it will look for photo_keypoints_named.json (format produced by the MediaPipe named export).
  - Else it will use a synthetic demo pose.
"""

import os
import json
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed to enable 3D projection

# Attempt to import optional libs
_have_mediapipe = False
_have_cv2 = False
try:
    import mediapipe as mp
    _have_mediapipe = True
except Exception:
    _have_mediapipe = False

try:
    import cv2
    _have_cv2 = True
except Exception:
    _have_cv2 = False

# File names
Z_SCALE = -1000

IMAGE_PATH = "NotBlake.JPG"
JSON_PATH = "photo_keypoints_named.json"
OUT_SNAPSHOT = "photo_pose_3d.png"

# A small set of connections for MediaPipe POSE (use mp_pose.POSE_CONNECTIONS if mediapipe present)
DEFAULT_CONNECTIONS = [
    # Torso
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),

    # Neck + Head
    ("LEFT_SHOULDER", "NECK"),
    ("RIGHT_SHOULDER", "NECK"),
    ("NECK", "HEAD_BOTTOM"),
    ("HEAD_BOTTOM", "HEAD_TOP"),  # 👈 New head line

    # Arms
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),

    # Legs
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),

    # Feet
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    ("LEFT_FOOT_INDEX", "LEFT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("RIGHT_FOOT_INDEX", "RIGHT_ANKLE"),
]

# If mediapipe available, build index->name and connections from it
if _have_mediapipe:
    mp_pose = mp.solutions.pose
    LANDMARK_IDX_TO_NAME = {lm.value: lm.name for lm in mp_pose.PoseLandmark}
    # Build name->idx mapping too
    LANDMARK_NAME_TO_IDX = {v: k for k, v in LANDMARK_IDX_TO_NAME.items()}
    # Use official connections where possible
    try:
        # mp_pose.POSE_CONNECTIONS contains pairs of index ints
        POSE_CONNECTIONS = []
        for a, b in mp_pose.POSE_CONNECTIONS:
            name_a = LANDMARK_IDX_TO_NAME.get(a, None)
            name_b = LANDMARK_IDX_TO_NAME.get(b, None)
            if name_a and name_b:
                POSE_CONNECTIONS.append((name_a, name_b))
        CONNECTIONS = POSE_CONNECTIONS
    except Exception:
        CONNECTIONS = DEFAULT_CONNECTIONS
else:
    # Fallback mapping (subset used in the visualization)
    LANDMARK_IDX_TO_NAME = {
        0: "NOSE", 11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER",
        23: "LEFT_HIP", 24: "RIGHT_HIP",
        13: "LEFT_ELBOW", 14: "RIGHT_ELBOW",
        15: "LEFT_WRIST", 16: "RIGHT_WRIST",
        25: "LEFT_KNEE", 26: "RIGHT_KNEE",
        27: "LEFT_ANKLE", 28: "RIGHT_ANKLE",
    }
    LANDMARK_NAME_TO_IDX = {v: k for k, v in LANDMARK_IDX_TO_NAME.items()}
    CONNECTIONS = DEFAULT_CONNECTIONS

def load_from_mediapipe(image_path):
    """Run MediaPipe on the image and return a dict name -> {x_px,y_px,z,visibility}"""
    assert _have_mediapipe and _have_cv2, "mediapipe+cv2 required for this function"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open {image_path}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        res = pose.process(img_rgb)
    if not res.pose_landmarks:
        raise RuntimeError("No pose detected by MediaPipe.")
    lm_list = res.pose_landmarks.landmark
    out = {}
    for idx, lm in enumerate(lm_list):
        name = LANDMARK_IDX_TO_NAME.get(idx, f"LANDMARK_{idx}")
        x = float(lm.x) * w
        y = float(lm.y) * h
        z = float(lm.z)  # MediaPipe z is in roughly normalized units relative to hips/size
        v = float(lm.visibility)
        out[name] = {"x": x, "y": y, "z": z, "visibility": v}
    return out, (w, h)

def load_from_json(json_path):
    """Load previously exported JSON mapping name->values"""
    with open(json_path, "r") as f:
        data = json.load(f)
    # Expecting format: {"image":..., "width":w, "height":h, "people":[{name:{x,y,z,visibility}}]}
    if "people" in data and len(data["people"]) > 0:
        person = data["people"][0]
        # convert keys to floats and ensure necessary fields exist
        out = {}
        for name, entry in person.items():
            out[name] = {
                "x": float(entry.get("x", 0.0)),
                "y": float(entry.get("y", 0.0)),
                "z": float(entry.get("z", 0.0)),
                "visibility": float(entry.get("visibility", 0.0)),
            }
        w = int(data.get("width", 640))
        h = int(data.get("height", 480))
        return out, (w, h)
    else:
        raise RuntimeError("JSON does not contain people/keypoints.")

def synthetic_pose():
    """Return a tiny demonstrative pose (centered)"""
    # Simple stickman with a few keypoints
    out = {
        "NOSE": {"x": 320, "y": 80, "z": -0.2, "visibility": 1.0},
        "LEFT_SHOULDER": {"x": 250, "y": 140, "z": -0.1, "visibility": 1.0},
        "RIGHT_SHOULDER": {"x": 390, "y": 140, "z": -0.1, "visibility": 1.0},
        "LEFT_ELBOW": {"x": 220, "y": 220, "z": 0.0, "visibility": 1.0},
        "RIGHT_ELBOW": {"x": 420, "y": 220, "z": 0.0, "visibility": 1.0},
        "LEFT_WRIST": {"x": 200, "y": 320, "z": 0.1, "visibility": 1.0},
        "RIGHT_WRIST": {"x": 440, "y": 320, "z": 0.1, "visibility": 1.0},
        "LEFT_HIP": {"x": 280, "y": 300, "z": 0.05, "visibility": 1.0},
        "RIGHT_HIP": {"x": 360, "y": 300, "z": 0.05, "visibility": 1.0},
        "LEFT_KNEE": {"x": 280, "y": 420, "z": 0.15, "visibility": 1.0},
        "RIGHT_KNEE": {"x": 360, "y": 420, "z": 0.15, "visibility": 1.0},
        "LEFT_ANKLE": {"x": 280, "y": 540, "z": 0.25, "visibility": 1.0},
        "RIGHT_ANKLE": {"x": 360, "y": 540, "z": 0.25, "visibility": 1.0},
    }
    return out, (640, 640)

def add_virtual_joints(keypoints, head_scale=1.0):
    """Add NECK, HEAD_BOTTOM, and HEAD_TOP joints using face points, then remove detailed face landmarks."""

    # --- Create NECK ---
    if "LEFT_SHOULDER" in keypoints and "RIGHT_SHOULDER" in keypoints:
        ls = keypoints["LEFT_SHOULDER"]
        rs = keypoints["RIGHT_SHOULDER"]
        neck = {
            "x": (ls["x"] + rs["x"]) / 2,
            "y": (ls["y"] + rs["y"]) / 2,
            "z": (ls["z"] + rs["z"]) / 2,
            "visibility": min(ls["visibility"], rs["visibility"]),
        }
        keypoints["NECK"] = neck

        # --- HEAD_BOTTOM: blend between ears and neck ---
        if "LEFT_EAR" in keypoints and "RIGHT_EAR" in keypoints:
            le = keypoints["LEFT_EAR"]
            re = keypoints["RIGHT_EAR"]
            ear_mid = {
                "x": (le["x"] + re["x"]) / 2,
                "y": (le["y"] + re["y"]) / 2,
                "z": (le["z"] + re["z"]) / 2,
            }
            # α = 0.6 pulls it 60% toward NECK, 40% toward ears
            alpha = 0.6
            head_bottom = {
                "x": alpha * neck["x"] + (1 - alpha) * ear_mid["x"],
                "y": alpha * neck["y"] + (1 - alpha) * ear_mid["y"],
                "z": alpha * neck["z"] + (1 - alpha) * ear_mid["z"],
                "visibility": min(le["visibility"], re["visibility"], neck["visibility"]),
            }
        else:
            # fallback
            head_bottom = {
                "x": neck["x"],
                "y": neck["y"] - 20,
                "z": neck["z"],
                "visibility": neck["visibility"],
            }
        keypoints["HEAD_BOTTOM"] = head_bottom

        # --- HEAD_TOP: based on eye midpoint, extended upward ---
        if "LEFT_EYE" in keypoints and "RIGHT_EYE" in keypoints:
            leye = keypoints["LEFT_EYE"]
            reye = keypoints["RIGHT_EYE"]
            eye_mid = {
                "x": (leye["x"] + reye["x"]) / 2,
                "y": (leye["y"] + reye["y"]) / 2,
                "z": (leye["z"] + reye["z"]) / 2,
            }
            dy = head_bottom["y"] - eye_mid["y"]  # vertical span from eyes to ear level
            head_top = {
                "x": eye_mid["x"],
                "y": eye_mid["y"] - dy * head_scale,  # extend upward
                "z": eye_mid["z"],
                "visibility": min(head_bottom["visibility"], leye["visibility"], reye["visibility"]),
            }
        else:
            # fallback: just push upward from head_bottom
            head_top = {
                "x": head_bottom["x"],
                "y": head_bottom["y"] - 40,
                "z": head_bottom["z"],
                "visibility": head_bottom["visibility"],
            }
        keypoints["HEAD_TOP"] = head_top

    # --- Remove ALL other face points (nose, eyes, ears, mouth, etc.) ---
    face_prefixes = ["NOSE", "EYE", "EAR", "MOUTH"]
    for n in list(keypoints.keys()):
        if any(pref in n for pref in face_prefixes):
            if n not in ["HEAD_BOTTOM", "HEAD_TOP"]:  # keep the abstracted head joints
                del keypoints[n]

    return keypoints

def build_xyz_arrays(keypoints):
    """Return numpy arrays X, Y, Z and list of names ordered for plotting."""
    names = list(keypoints.keys())
    X = np.array([keypoints[n]["x"] for n in names], dtype=float)
    Y = np.array([keypoints[n]["y"] for n in names], dtype=float)
    Z = np.array([keypoints[n]["z"] for n in names], dtype=float) * Z_SCALE
    V = np.array([keypoints[n].get("visibility", 0.0) for n in names], dtype=float)
    return X, Y, Z, V, names

def plot_3d(keypoints, img_size=None, connections=None, save_snapshot=True):
    """Make an interactive 3D plot of the pose."""
    X, Y, Z, V, names = build_xyz_arrays(keypoints)
    # Flip Y so that origin is bottom-left when plotting (image coords: y down)
    Y_plot = -Y

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=-60)  # initial view

    # Scatter points
    ax.scatter(X, Y_plot, Z, s=40)

    # Label points
    for xi, yi, zi, nm, vi in zip(X, Y_plot, Z, names, V):
        if vi >= 0.15:
            ax.text(xi + 5.0, yi + 5.0, zi, nm, fontsize=8)

    # 🔧 FIX: Draw skeleton connections using scaled Z and flipped Y
    if connections:
        for a_name, b_name in connections:
            if a_name in keypoints and b_name in keypoints:
                xa, ya, za = (
                    keypoints[a_name]["x"],
                    -keypoints[a_name]["y"],
                    keypoints[a_name]["z"] * Z_SCALE,   # scale Z same as points
                )
                xb, yb, zb = (
                    keypoints[b_name]["x"],
                    -keypoints[b_name]["y"],
                    keypoints[b_name]["z"] * Z_SCALE,   # scale Z same as points
                )
                ax.plot([xa, xb], [ya, yb], [za, zb], c="blue")

    # Axis labels and equal aspect
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px, inverted)")
    ax.set_zlabel("Z (relative depth)")
    ax.set_title("3D Pose Visualization (rotate with mouse)")

    try:
        max_range = np.array([X.max()-X.min(), Y_plot.max()-Y_plot.min(), Z.max()-Z.min()]).max()
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y_plot.max()+Y_plot.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    except Exception:
        pass

    plt.tight_layout()

    if save_snapshot:
        plt.savefig(OUT_SNAPSHOT, dpi=150)
        print(f"Saved snapshot to {OUT_SNAPSHOT}")

    plt.show()

def main():
    kp = None
    size = None
    # 1) try mediapipe
    if _have_mediapipe and _have_cv2 and os.path.exists(IMAGE_PATH):
        try:
            print("Running MediaPipe on image...")
            kp, size = load_from_mediapipe(IMAGE_PATH)
            print(f"Loaded {len(kp)} landmarks from MediaPipe (image size {size}).")
        except Exception as e:
            print("MediaPipe run failed:", e)
            kp = None

    # 2) fallback to JSON if exists
    if kp is None and os.path.exists(JSON_PATH):
        try:
            print("Loading keypoints from JSON...")
            kp, size = load_from_json(JSON_PATH)
            print(f"Loaded {len(kp)} landmarks from JSON (image size {size}).")
        except Exception as e:
            print("Failed to load JSON:", e)
            kp = None

    # 3) fallback synthetic
    if kp is None:
        print("Using synthetic demo pose (no mediapipe/photo/json available).")
        kp, size = synthetic_pose()

    # Choose which connections to show: if we have official connections, use them
    connections = DEFAULT_CONNECTIONS

    # Add virtual NECK + HEAD, remove face points
    kp = add_virtual_joints(kp, head_scale=1.2)

    # Plot
    plot_3d(kp, img_size=size, connections=connections)

if __name__ == "__main__":
    main()

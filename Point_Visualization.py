# visualize_mediapipe_pose_named.py
import cv2
import mediapipe as mp
import json
import os

IMG_PATH = "NotBlake.JPG"
OUT_IMAGE = "NotBlake_annotated2.jpg"
OUT_JSON = "photo_keypoints_named.json"
VISIBILITY_THRESHOLD = 0.25  # only label points with visibility above this

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Build an index->name mapping from MediaPipe's PoseLandmark enum
LANDMARK_IDX_TO_NAME = {lm.value: lm.name for lm in mp_pose.PoseLandmark}

def lm_to_pixel(lm, img_w, img_h):
    # Convert normalized coords to pixel coords; clamp inside image
    x = int(round(lm.x * img_w))
    y = int(round(lm.y * img_h))
    # ensure coords are within image bounds
    x = max(0, min(img_w - 1, x))
    y = max(0, min(img_h - 1, y))
    return x, y, float(lm.z), float(lm.visibility)

# Load image
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"Could not open {IMG_PATH} — put the image in this folder or change IMG_PATH.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

annotated = img.copy()

with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
    results = pose.process(img_rgb)

export = {"image": IMG_PATH, "width": w, "height": h, "people": []}

if not results.pose_landmarks:
    print("No pose detected.")
else:
    # MediaPipe Pose returns a single pose (best single-person). We'll treat it as person 0.
    person_kps = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        x_px, y_px, z, v = lm_to_pixel(lm, w, h)
        name = LANDMARK_IDX_TO_NAME.get(idx, f"LANDMARK_{idx}")
        person_kps[name] = {"x": x_px, "y": y_px, "z": z, "visibility": v}

        # Draw marker for visible landmarks
        if v > 0.05:
            cv2.circle(annotated, (x_px, y_px), 4, (0, 140, 255), -1)

        # Draw text label only if landmark is confidently visible
        if v >= VISIBILITY_THRESHOLD:
            # choose text position offset so text doesn't overlap point
            text_pos = (x_px + 6, y_px - 6)
            # draw text with thin black outline for readability
            cv2.putText(annotated, name, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
            cv2.putText(annotated, name, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Draw pose skeleton connections (nice clean lines)
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2),
    )

    export["people"].append(person_kps)

# Save JSON and annotated image
with open(OUT_JSON, "w") as f:
    json.dump(export, f, indent=2)

cv2.imwrite(OUT_IMAGE, annotated)
print(f"Annotated image saved to {OUT_IMAGE}")
print(f"Named keypoints saved to {OUT_JSON}")

# Display (press any key to close)
cv2.imshow("Pose (named landmarks)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

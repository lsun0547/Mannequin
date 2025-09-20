# visualize_mediapipe_pose.py
import cv2
import mediapipe as mp
import json
import os

IMG_PATH = "NotBlake.JPG"
OUT_IMAGE = "NotBlake_annotated.jpg"
OUT_JSON = "photo_keypoints.json"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Helper: convert normalized landmark to pixel coords
def lm_to_pixel(lm, img_w, img_h):
    return int(lm.x * img_w), int(lm.y * img_h), lm.z, lm.visibility

# Run MediaPipe Pose (static image mode)
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"Could not open {IMG_PATH} — put the image in this folder or change IMG_PATH.")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
    results = pose.process(img_rgb)

# Prepare storage for export
export = {"image": IMG_PATH, "width": w, "height": h, "people": []}

annotated = img.copy()

if not results.pose_landmarks:
    print("No pose detected.")
else:
    # MediaPipe Pose returns a single pose (best single-person). We'll treat it as person 0.
    landmarks = results.pose_landmarks.landmark
    person_kps = {}
    for i, lm in enumerate(landmarks):
        x_px, y_px, z, v = lm_to_pixel(lm, w, h)
        person_kps[i] = {"x": x_px, "y": y_px, "z": float(z), "visibility": float(v)}

    export["people"].append(person_kps)

    # --- Drawing: use MediaPipe's drawing utils for clean skeleton + add labels/circles ---
    # Default drawing (nice lines)
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2),
    )

    # Add numeric labels next to keypoints (index + small circle)
    for idx, lm in enumerate(landmarks):
        x_px, y_px, z, v = lm_to_pixel(lm, w, h)
        # label only if reasonably visible
        if v > 0.25:
            cv2.circle(annotated, (x_px, y_px), 4, (255, 0, 0), -1)  # solid dot
            cv2.putText(
                annotated,
                str(idx),
                (x_px + 6, y_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

# Save JSON
with open(OUT_JSON, "w") as f:
    json.dump(export, f, indent=2)

# Show and save annotated image
cv2.imwrite(OUT_IMAGE, annotated)
print(f"Annotated image saved to {OUT_IMAGE}")
print(f"Keypoints saved to {OUT_JSON}")

# Display (press any key to close)
cv2.imshow("Pose (annotated)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

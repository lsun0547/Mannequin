# visualize_mediapipe_pose_pnp.py
import cv2
import mediapipe as mp
import json
import numpy as np
import math

IMG_PATH = "NotBlake.JPG"
OUT_IMAGE = "NotBlake_annotated_pnp.jpg"
OUT_JSON = "photo_keypoints_pnp.json"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Helpers -----------------------------------------------------------------
def lm_to_pixel(lm, img_w, img_h):
    return float(lm.x * img_w), float(lm.y * img_h), float(lm.z), float(lm.visibility)

def solve_quadratic(a, b, c):
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    sqrt_d = math.sqrt(max(0.0, disc))
    return [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)]

def find_point_on_ray_with_distance(P_cam, q_cam, L):
    """
    Solve for s > 0 such that ||P_cam - s*q_cam|| = L.
    q_cam = K_inv @ [u,v,1] (a 3-vector), not necessarily normalized.
    Returns 3D point s*q_cam (camera coords) or None.
    """
    a = float(np.dot(q_cam, q_cam))
    b = -2.0 * float(np.dot(P_cam, q_cam))
    c = float(np.dot(P_cam, P_cam) - L * L)
    sols = solve_quadratic(a, b, c)
    sols = [s for s in sols if s > 0]
    if not sols:
        return None
    s = min(sols)  # choose the closer solution (smaller s)
    return (s * q_cam).reshape(3)

# --- Canonical 3D model (meters, rough human proportions) ---------------------
# We only need a small torso/head model for solvePnP + bone lengths for limbs.
# Indices match MediaPipe: 0 nose, 11 L_shoulder, 12 R_shoulder, 23 L_hip, 24 R_hip, ...
CANONICAL = {
    0:  np.array([0.0,  0.25, 0.0]),   # nose (y up)
    11: np.array([-0.20, 0.0,  0.0]),  # left_shoulder
    12: np.array([ 0.20, 0.0,  0.0]),  # right_shoulder
    23: np.array([-0.15,-0.45, 0.0]),  # left_hip
    24: np.array([ 0.15,-0.45, 0.0]),  # right_hip
}

# Bone length (meters) for chain reconstruction (child_index -> length to parent)
BONE_LENGTHS = {
    13: 0.28,  # left_elbow  (parent 11)
    15: 0.25,  # left_wrist  (parent 13)
    14: 0.28,  # right_elbow (parent 12)
    16: 0.25,  # right_wrist (parent 14)
    25: 0.45,  # left_knee   (parent 23)
    27: 0.45,  # left_ankle  (parent 25)
    26: 0.45,  # right_knee  (parent 24)
    28: 0.45,  # right_ankle (parent 26)
}

PARENT = {
    13: 11, 15: 13,
    14: 12, 16: 14,
    25: 23, 27: 25,
    26: 24, 28: 26,
}

# --- Load image and MediaPipe Pose ------------------------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit(f"Could not open {IMG_PATH}.")
h, w = img.shape[:2]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(img_rgb)

export = {"image": IMG_PATH, "width": w, "height": h, "people": []}
annotated = img.copy()

if not results.pose_landmarks:
    print("No pose detected.")
else:
    lm_list = results.pose_landmarks.landmark
    # pixel coords and visibility lookup
    pix = {}
    vis = {}
    for i, lm in enumerate(lm_list):
        u, v, z_raw, vis_i = lm_to_pixel(lm, w, h)
        pix[i] = (u, v)
        vis[i] = vis_i

    # Choose points for solvePnP (only those we have and visible)
    object_pts = []
    image_pts = []
    pnp_indices = []
    for idx in [0, 11, 12, 23, 24]:
        if idx in CANONICAL and vis.get(idx, 0.0) > 0.25:
            object_pts.append(CANONICAL[idx])
            image_pts.append(pix[idx])
            pnp_indices.append(idx)

    person_cam_positions = {}  # index -> 3D camera coords (x,y,z)

    # Build camera intrinsics (approximate)
    focal = max(w, h)  # coarse focal length in pixels
    camera_matrix = np.array([[focal, 0, w / 2.0],
                              [0, focal, h / 2.0],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    if len(object_pts) >= 4:
        objp = np.array(object_pts, dtype=np.float64)
        imgp = np.array(image_pts, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            print("solvePnP failed; falling back to simple z scaling.")
        else:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)

            # transform canonical model points used in PnP into camera coords
            for idx, X in zip(pnp_indices, objp):
                X_cam = (R @ X) + t
                person_cam_positions[idx] = X_cam

            # compute shoulder width in camera coords (for normalization)
            if 11 in person_cam_positions and 12 in person_cam_positions:
                shoulder_width_cam = float(np.linalg.norm(person_cam_positions[11] - person_cam_positions[12]))
                if shoulder_width_cam <= 1e-6:
                    shoulder_width_cam = 1.0
            else:
                shoulder_width_cam = 1.0

            # Precompute inverse camera matrix for backprojection
            K_inv = np.linalg.inv(camera_matrix)

            # Reconstruct limb joints by chaining from parents outward
            # Order the reconstruction so parents are known first
            reconstruction_order = [13, 14, 15, 16, 25, 26, 27, 28]
            for child in reconstruction_order:
                parent = PARENT.get(child)
                if parent is None:
                    continue
                if parent not in person_cam_positions:
                    # if parent wasn't produced by PnP (e.g. hip/shoulder was missing), skip
                    continue
                # need pixel location for child and visibility
                if vis.get(child, 0.0) < 0.05:
                    continue

                u, v = pix[child]
                # backproject: q_cam = K_inv * [u, v, 1]^T
                q = K_inv @ np.array([u, v, 1.0])
                # parent camera position
                P_cam = person_cam_positions[parent]
                L = BONE_LENGTHS.get(child, None)
                if L is None:
                    continue

                found = find_point_on_ray_with_distance(P_cam, q, L)
                if found is None:
                    # fallback: try using slightly larger L or use MediaPipe z scaled
                    # Use MediaPipe z scaled by image/shoulder width in pixels (a graceful fallback)
                    lm = lm_list[child]
                    z_fallback = lm.z * float(w)
                    # We don't have a direct mapping to camera Z; approximate by mapping z_fallback -> z_cam
                    # place at parent z minus small amount (heuristic)
                    approx_cam = P_cam.copy()
                    approx_cam[2] = approx_cam[2] - (0.1 * L)  # heuristic
                    person_cam_positions[child] = approx_cam
                else:
                    person_cam_positions[child] = np.array(found)

            # For any remaining landmarks that are in CANONICAL but not processed, compute camera pos via R,t:
            for idx in CANONICAL:
                if idx not in person_cam_positions:
                    X_cam = (R @ CANONICAL[idx]) + t
                    person_cam_positions[idx] = X_cam

            # As a last step, for other landmarks not reconstructed, we can attempt simple ray-depth using parent's average bone length
            for idx in range(len(lm_list)):
                if idx in person_cam_positions:
                    continue
                # attempt to get parent and reconstruct; otherwise fallback to MediaPipe z (scaled)
                if idx in PARENT and PARENT[idx] in person_cam_positions and vis.get(idx, 0.0) > 0.05:
                    parent = PARENT[idx]
                    u, v = pix[idx]
                    q = K_inv @ np.array([u, v, 1.0])
                    P_cam = person_cam_positions[parent]
                    L = BONE_LENGTHS.get(idx, 0.3)  # default
                    found = find_point_on_ray_with_distance(P_cam, q, L)
                    if found is not None:
                        person_cam_positions[idx] = np.array(found)
                        continue
                # fallback: estimate z by mapping MediaPipe raw z to camera units relative to shoulder width in pixels
                lm = lm_list[idx]
                z_approx = lm.z * float(w)  # original trick: lm.z * image_width
                # we don't have direct conversion to camera coords; put it near torso depth:
                torso_z = np.mean([person_cam_positions[i][2] for i in (11,12) if i in person_cam_positions]) if (11 in person_cam_positions or 12 in person_cam_positions) else 2.0
                # heuristic: if lm.z is negative (closer), subtract, else add
                person_cam_positions[idx] = np.array([lm.x * w, lm.y * h, torso_z + z_approx * 0.001])

    else:
        print("Not enough reliable PnP points; falling back to simple shoulder-normalized z.")
        # fallback: previous approach: use lm.z * w / shoulder_pixel_width
        # compute shoulder pixel width:
        if vis.get(11,0) > 0.05 and vis.get(12,0) > 0.05:
            ls = np.array(pix[11])
            rs = np.array(pix[12])
            shoulder_pix = np.linalg.norm(ls - rs)
        else:
            shoulder_pix = max(w, h) * 0.25
        for i, lm in enumerate(lm_list):
            u, v, z_raw, vv = lm_to_pixel(lm, w, h)
            z_norm = (z_raw * w) / max(shoulder_pix, 1.0)
            # put camera approximate position: x,y from pixel but z as z_norm
            person_cam_positions[i] = np.array([u, v, z_norm])

    # Build export dict: include pixel coords, camera z (meters-ish), normalized z (by shoulder width in camera units)
    # compute shoulder width in camera units if available
    if 11 in person_cam_positions and 12 in person_cam_positions:
        sw_cam = float(np.linalg.norm(person_cam_positions[11] - person_cam_positions[12]))
        if sw_cam < 1e-6:
            sw_cam = 1.0
    else:
        sw_cam = 1.0

    person_kps = {}
    for i, lm in enumerate(lm_list):
        u, v = pix[i]
        cam_pt = person_cam_positions.get(i)
        if cam_pt is not None:
            z_cam = float(cam_pt[2])
            z_norm = float(z_cam / sw_cam)
        else:
            z_cam = None
            z_norm = None
        person_kps[i] = {
            "x": float(u),
            "y": float(v),
            "z_cam": z_cam,        # camera-space Z (positive forward)
            "z_norm": z_norm,     # normalized by shoulder width in camera units
            "visibility": float(vis.get(i, 0.0))
        }

    export["people"].append(person_kps)

    # Draw skeleton and annotate z_norm on the image for debugging
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2),
    )
    for idx, lm in enumerate(lm_list):
        u, v = int(pix[idx][0]), int(pix[idx][1])
        info = person_kps[idx]
        if info["z_norm"] is not None:
            txt = f"{idx}:{info['z_norm']:.2f}"
        else:
            txt = str(idx)
        if info["visibility"] > 0.25:
            cv2.putText(annotated, txt, (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

# Save results
with open(OUT_JSON, "w") as f:
    json.dump(export, f, indent=2)

cv2.imwrite(OUT_IMAGE, annotated)
print(f"Annotated image saved to {OUT_IMAGE}")
print(f"Keypoints saved to {OUT_JSON}")

cv2.imshow("Pose (annotated)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

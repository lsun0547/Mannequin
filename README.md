# Flask Pose Mannequin

This project uses **Flask**, **MediaPipe**, and **Three.js** to generate a 3D mannequin from an uploaded photo.

## Goals
- This project was created or the **Hello World** Hackathon at Purdue University.
- The goal of this project is to allow users to create a 3D model based off an inputted 2D image.
  - This is intented to offer artists an extra tool for perspective drawing and could further be utilized for basic 3d animation.
- Developing this project further would include adding functionality with **Blender**, allowing users to upload a photo and get a model rigged to that pose.




## Features
- Upload a photo -> mannequin is generated automatically
- Pose extracted with **MediaPipe**
- 3D mannequin rendered with **Three.js**
  - Green spheres = joints
  - Red polygons = limbs, torso, hands, feet, head
- **OrbitControls** enabled (rotate, zoom, pan)
<img width="1220" height="887" alt="image" src="https://github.com/user-attachments/assets/5ca0d3fc-6025-4712-98eb-912dda010e36" />

<img width="1220" height="843" alt="image" src="https://github.com/user-attachments/assets/b39d44b0-af69-4004-a570-f46c10d19cd8" />


## Abstractions

To simplify the skeleton, some joints are combined into higher-level shapes:

| Body Part | MediaPipe Joints Used | Mannequin Shape                           |
|-----------|-----------------------|-------------------------------------------|
| **Feet**  | Ankle, Heel, Toe (Index) | Red rectangular prism                     |
| **Hands** | Wrist, Index, Pinky | Red rectangular prism                     |
| **Head**  | Neck, Head Bottom, Head Top | Green sphere (neck) + red cylinder (head) |
| **Torso** | Left Shoulder, Right Shoulder, Left Hip, Right Hip | Red convex mesh                           |
| **Limbs** | Shoulder -> Elbow, Elbow -> Wrist, Hip -> Knee, Knee -> Ankle | Red cylinders                             |
| **Joints** | All visible (except abstracted ones) | Green spheres                             |

**Hidden joints:**  
- `INDEX`, `PINKY` (hand fingers)  
- `HEAD_TOP`, `HEAD_BOTTOM` (kept internally, not rendered as spheres)  
- `FOOT_INDEX` (only used to shape feet)  


## Requirements
- Python 3.9 to 3.11
- Flask
- MediaPipe
- OpenCV

Install dependencies:
```bash
pip install flask mediapipe opencv-python

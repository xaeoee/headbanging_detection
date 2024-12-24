import cv2
import numpy as np

def head_mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not results.multi_face_landmarks:  # Check if no face landmarks are detected
            return frame, None, None, None, None
    for face_landmarks in results.multi_face_landmarks:
        # Extract landmarks as a NumPy array
        landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark])
        left_eye_coords = landmarks[133]
        right_eye_coords = landmarks[362]
        left_lip_coords = landmarks[61]
        right_lip_coords = landmarks[291]

    return frame, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords

def head_width_height_scailing(height, width, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords):
    left_eye_coords = (int(left_eye_coords[0] * width), int(left_eye_coords[1] * height), int(left_eye_coords[2]))
    right_eye_coords = (int(right_eye_coords[0] * width), int(right_eye_coords[1] * height), int(right_eye_coords[2]))
    left_lip_coords = (int(left_lip_coords[0] * width), int(left_lip_coords[1] * height), int(left_lip_coords[2]))
    right_lip_coords = (int(right_lip_coords[0] * width), int(right_lip_coords[1] * height), int(right_lip_coords[2]))

    return left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords

def head_coords_from_2d_to_3d(left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords):
    left_eye_coords = (left_eye_coords[0], left_eye_coords[1])
    right_eye_coords = (right_eye_coords[0], right_eye_coords[1])
    left_lip_coords = (left_lip_coords[0], left_lip_coords[1])
    right_lip_coords = (right_lip_coords[0], right_lip_coords[1])

    return left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords

def head_plot_circle(image, center_forehead_coords, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords):
    cv2.circle(image, center_forehead_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, left_eye_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, right_eye_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, left_lip_coords, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, right_lip_coords, radius=3, color=(0, 255, 0), thickness=-1)

def draw_vector(image, start_point, end_point, color=(0, 255, 0), thickness=2):
    cv2.line(image, start_point, end_point, color, thickness)
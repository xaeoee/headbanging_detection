import matplotlib
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from functions import *

RED = "\033[31m"
RESET = "\033[0m"
GREEN = "\033[32m"

# Import MediaPipe FaceMesh module
mp_face_mesh = mp.solutions.face_mesh

# Import MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

def headbanging_detection(angle_threshold, time_threshold, headbanging_threshold, verbose, debug, flip, video_path):
    # Normal variables
    # Initialize the previous normal vector to None. Used to calculate with the current normal vector
    previous_normal_vector = None

    # Used to track time
    last_direction_changed_time = time.time()

    # Used to count direction changes
    direction_change_count = 0

    # Used to store movement direction status
    last_movement_direction = None

    # Create a video capture instance
    cap = cv2.VideoCapture(video_path)

    current_frame_number = 0

    # Initialize FaceMesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,  # Process video streams instead of static images
        max_num_faces=1,  # Maximum number of faces to detect
        refine_landmarks=True,  # Enable refined landmark detection
        min_detection_confidence=0.5,  # Minimum confidence for face detection
        min_tracking_confidence=0.5  # Minimum confidence for face tracking
    ) as face_mesh:
        # Run while the video stream is open
        while cap.isOpened():
            # total frame number
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Read a frame
            ret, frame = cap.read()
            # If reading the frame fails
            if not ret:
                print("Error: Frame could not be read.")
                break
            # Flip the frame horizontally
            if flip:
                frame = cv2.flip(frame, 1)

            current_frame_number += 1
            
            # Save the height and width for scaling MediaPipe's normalized coordinates (0~1) to actual screen size when drawing lines
            height, width, _ = frame.shape

            # Call the head_mediapipe_detection function to get the processed result and coordinates
            image, left_eye_coords_ndarry, right_eye_coords_ndarry, left_lip_coords_ndarry, right_lip_coords_ndarry, center_forehead_coords_ndarray = head_mediapipe_detection(frame, face_mesh)

            # If the face is not detected and the return values of the head_mediapipe_detection function are None, move to the next frame
            if left_eye_coords_ndarry is None or right_eye_coords_ndarry is None or left_lip_coords_ndarry is None or right_lip_coords_ndarry is None or center_forehead_coords_ndarray is None:
                if debug:
                    print("Result is None")
                continue
            
            left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords = head_width_height_scailing(height, width, left_eye_coords_ndarry, right_eye_coords_ndarry, left_lip_coords_ndarry, right_lip_coords_ndarry, center_forehead_coords_ndarray)

            left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords = head_coords_from_3d_to_2d(left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords ,center_forehead_coords)

            forehead_to_left_lip_vector = left_lip_coords_ndarry -  center_forehead_coords_ndarray

            forehead_to_right_lip_vector = right_lip_coords_ndarry - center_forehead_coords_ndarray

            normal_vector = np.cross(forehead_to_left_lip_vector, forehead_to_right_lip_vector)

            if np.linalg.norm(normal_vector) != 0:
                # Normalize the original normal_vector by its norm
                current_normal_vector = normal_vector / np.linalg.norm(normal_vector)

                draw_normal_vector(image, center_forehead_coords, current_normal_vector)

                if previous_normal_vector is not None:
                    angle_difference = head_angle_calculation(previous_normal_vector, current_normal_vector)
                    
                    # If the angle difference between the previous and current vectors exceeds the set threshold, execute the following code (indicates a significant change in angle)
                    if angle_difference > angle_threshold:
                        if verbose:
                            print(f"{RED}Angle changed more than threshold{RESET}")

                        # Save the current time since the angle has changed
                        current_time = time.time()

                        # If the time difference between now and the last direction change exceeds the time threshold (default: 1 second), reset the direction change count. 
                        # (This means that within 1 second, n direction changes must occur for it to be recognized as clapping. If it's more than 1 second, reset the count.)
                        if current_time - last_direction_changed_time > time_threshold:
                            direction_change_count = 0

                        current_movement_direction = head_movement_direction(previous_normal_vector, current_normal_vector)
                        if verbose:
                            print(current_movement_direction)

                        if last_movement_direction != current_movement_direction:
                            direction_change_count += 1

                            last_movement_direction = current_movement_direction

                            last_direction_changed_time = current_time

                        if direction_change_count >= headbanging_threshold:
                            print(f"{GREEN}HEADBANGING DETECTED!{RESET} at {current_frame_number / fps:.2f}")
                            direction_change_count = 0

                    # Update the current vector to be the previous vector
                    previous_normal_vector = current_normal_vector

                else:
                    previous_normal_vector = current_normal_vector
            else:
                if debug:
                    print("Normalized vector's value is 0")

            # Current coordinates are in the form of (integer, integer) tuples
            head_plot_circle(image, center_forehead_coords, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords)

            draw_vector(image, center_forehead_coords, left_lip_coords)
            draw_vector(image, center_forehead_coords, right_lip_coords)
            # Display the processed result frame
            cv2.imshow("Head Detection", image)

            # Exit when the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and window resources
    cap.release()
    cv2.destroyAllWindows()

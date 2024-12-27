import matplotlib
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from functions import *


mp_face_mesh = mp.solutions.face_mesh  # MediaPipe FaceMesh 모듈 가져오기
mp_drawing = mp.solutions.drawing_utils  # MediaPipe 도구 유틸리티 가져오기

def headbanging_detection():
    # args variable from the user
    angle_threshold = 3 # can be vary
    time_threshold = 1
    

    # normal variable
    normal_vector_prev = None
    angle_difference = 0
    last_direction_changed_time = time.time()
    direction_change_count = 0

    # 웹캠에서 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(0)

    # FaceMesh 초기화
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,  # 정적 이미지가 아닌 비디오 스트림에서 처리
        max_num_faces=1,  # 최대 탐지할 얼굴 수
        refine_landmarks=True,  # 정밀한 랜드마크 탐지 활성화
        min_detection_confidence=0.5,  # 얼굴 탐지 최소 신뢰도
        min_tracking_confidence=0.5  # 얼굴 추적 최소 신뢰도
    ) as face_mesh:

        while cap.isOpened():  # 비디오 스트림이 열려 있는 동안 실행
            # 프레임 읽기
            ret, frame = cap.read()
            # 프레임 읽기에 실패한 경우
            if not ret:
                print("Error: Frame could not be read.")
                break
            
            # 0~1 사이의 값으로 정규화된 mediapipe의 좌표값을 선을 그릴때는 실제 화면 사이즈로 키워줘야 할때 쓰려고 저장하는 height, width.
            height, width, _ = frame.shape

            # head_mediapipe_detection 함수 호출하여 처리 결과 및 좌표 추출
            image, left_eye_coords_ndarry, right_eye_coords_ndarry, left_lip_coords_ndarry, right_lip_coords_ndarry, center_forehead_coords_ndarray = head_mediapipe_detection(frame, face_mesh)

            # 얼굴을 감지하지 못하여 head_mediapipe_detection 함수의 리턴값이 None인 경우 다음 프레임으로 진행
            if left_eye_coords_ndarry is None or right_eye_coords_ndarry is None or left_lip_coords_ndarry is None or right_lip_coords_ndarry is None or center_forehead_coords_ndarray is None:
                print("Result is None")
                continue
            
            left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords = head_width_height_scailing(height, width, left_eye_coords_ndarry, right_eye_coords_ndarry, left_lip_coords_ndarry, right_lip_coords_ndarry, center_forehead_coords_ndarray)

            left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords, center_forehead_coords = head_coords_from_3d_to_2d(left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords ,center_forehead_coords)

            forehead_to_left_lip_vector = left_lip_coords_ndarry -  center_forehead_coords_ndarray

            forehead_to_right_lip_vector = right_lip_coords_ndarry - center_forehead_coords_ndarray

            normal_vector = np.cross(forehead_to_left_lip_vector, forehead_to_right_lip_vector)

            if np.linalg.norm(normal_vector) != 0:
                # the original normal_vector is divided by it's norm.
                normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

                # x_cord.append(normal_vector_normalized[0])
                # y_cord.append(normal_vector_normalized[1])
                # z_cord.append(normal_vector_normalized[2])

                draw_normal_vector(image, center_forehead_coords, normal_vector_normalized)

                if normal_vector_prev is not None:
                    angle_difference = head_angle_calculation(normal_vector_prev, normal_vector_normalized)
                    print(f"Angle difference: {angle_difference} degrees")
                    
                    # 이전 벡터와, 현재 벡터의 각도 차이가 우리가 세팅해놓은 임계값 이상이 될경우 아래 코드를 실행한다. (즉 각도가 크게 변화되었음을 나타낸다.)
                    if angle_difference > angle_threshold:
                        print("Angle changed more than threshold")

                        # 각도가 변했으니 우선 현재 시간을 저장
                        current_time = time.time()
                        
                        # 각도에 변화가 생긴 지금 현재 시간과 이전에 방향이 바뀐 시간의 차이가 시간 임계값(기본 1초) 보다 크다면 방향 바뀐 횟수를 저장 하는 변수를 0으로 만든다 (이것은 1초안에 n번 방향 전환이 일어나야 박수로 인식할건데, 1초보다 크니 누적 횟수를 없애는거다), 
                        if current_time - last_direction_changed_time > time_threshold:
                            direction_change_count = 0

                        



                    # 현재 벡터를 이전 벡터로 업데이트
                    normal_vector_prev = normal_vector_normalized

                else:
                    print("First frame")
                    normal_vector_prev = normal_vector_normalized
            else:
                print("Normalised vectors' value is 0")

            # 좌표값은 현재 (정수, 정수) 형태의 튜플 타입이다.
            head_plot_circle(image, center_forehead_coords, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords)

            draw_vector(image, center_forehead_coords, left_lip_coords)
            draw_vector(image, center_forehead_coords, right_lip_coords)
            # 처리된 결과 프레임 출력
            cv2.imshow("Head Detection", image)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 비디오 캡처 및 윈도우 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
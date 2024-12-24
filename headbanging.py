import matplotlib
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import *


mp_face_mesh = mp.solutions.face_mesh  # MediaPipe FaceMesh 모듈 가져오기
mp_drawing = mp.solutions.drawing_utils  # MediaPipe 도구 유틸리티 가져오기

def headbanging_detection():
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
            ret, frame = cap.read()  # 프레임 읽기

            if not ret:  # 프레임 읽기에 실패한 경우
                print("Error: Frame could not be read.")  # 오류 메시지 출력
                break

            height, width, _ = frame.shape

            # head_mediapipe_detection 함수 호출하여 처리 결과 및 좌표 추출
            image, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords = head_mediapipe_detection(frame, face_mesh)

            
            # 얼굴을 감지하지 못하여 head_mediapipe_detection 함수의 리턴값이 None인 경우 다음 프레임으로 진행
            if left_eye_coords is None or right_eye_coords is None or left_lip_coords is None or right_lip_coords is None:
                print("Result is None")  # 결과 없음 메시지 출력
                continue
            
            left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords = head_width_height_scailing(height, width, left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords)

            left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords = head_coords_from_2d_to_3d(left_eye_coords, right_eye_coords, left_lip_coords, right_lip_coords)

            center_forehead_coords = ((left_eye_coords[0] + right_eye_coords[0])//2, int(left_eye_coords[1] + right_eye_coords[1])//2 )

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


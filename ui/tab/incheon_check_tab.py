# app/pages/track_tab.py

import streamlit as st
import cv2
import torch
import tempfile
import os
import numpy as np
import time
from external.adaptors.detector import Detector
from tracker.boost_track import BoostTrack
from PIL import Image

FPS = 30
UPDATE_INTERVAL = 3 * FPS
MIN_DISTANCE = 5.0

def extract_frame_and_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None, None, None

    height, width = frame.shape[:2]
    # ROI는 전체 프레임 크기로 자동 설정 (직사각형)
    roi_polygon = np.array([
        (0, 0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1)
    ], np.int32)

    return frame, width, height, roi_polygon


def is_inside_roi(point, roi_polygon):
    return cv2.pointPolygonTest(roi_polygon, (int(point[0]), int(point[1])), False) >= 0

def calculate_roi_mid_length(roi_polygon):
    left_mid = ((roi_polygon[0][0] + roi_polygon[1][0]) // 2,
                (roi_polygon[0][1] + roi_polygon[1][1]) // 2)
    right_mid = ((roi_polygon[2][0] + roi_polygon[3][0]) // 2,
                 (roi_polygon[2][1] + roi_polygon[3][1]) // 2)
    roi_length = np.sqrt((right_mid[0]-left_mid[0])**2 + (right_mid[1]-left_mid[1])**2)
    return roi_length, left_mid, right_mid

def check_tab():
    uploaded_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "avi", "mov", "MOV"])
    if uploaded_file is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    os.unlink(video_path)

    if not ret:
        st.error("영상을 읽을 수 없습니다.")
        return

    height, width = frame.shape[:2]
    full_frame_roi = np.array([
        (0, 0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1)
    ], np.int32)

    # ROI 표시 (선택 사항)
    cv2.polylines(frame, [full_frame_roi], isClosed=True, color=(255, 0, 0), thickness=2)

    # 모델 초기화
    detector = Detector("yolox", "external/weights/bytetrack_x_mot20.tar", "custom")
    tracker = BoostTrack()

    # 추론용 프레임 변환
    img_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # # YOLOX 입력 크기 맞게 조정 (예: 640x384 or 640x640 등) --> 이거 문제네
    desired_size = (640, 384)  # 또는 (640, 640)
    img_numpy_resized = cv2.resize(img_numpy, desired_size)
    img = torch.from_numpy(img_numpy_resized).permute(2, 0, 1).float().cuda() / 255.0
    img = img.unsqueeze(0) if img.ndim == 3 else img

    # img = torch.from_numpy(img_numpy).permute(2, 0, 1).float().cuda() / 255.0
    # if img.ndim == 3:
    #     img = img.unsqueeze(0)

    # 탐지 및 추적
    with torch.no_grad():
        outputs = detector.detect(img)
    if outputs is None:
        st.warning("탐지된 객체가 없습니다.")
        st.image(img_numpy, caption="탐지 결과 없음", channels="RGB")
        return

    targets = tracker.update(outputs, img, img_numpy, "custom_video")
    if isinstance(targets, np.ndarray):
        targets = [targets]

    for target in targets:
        if isinstance(target, np.ndarray) and target.ndim == 2:
            for obj in target:
                x1, y1, x2, y2, track_id = map(int, obj[:5])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # 바운딩박스 및 ID 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 출력 (1프레임만)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="첫 프레임 추론 결과", channels="RGB", use_container_width=True)
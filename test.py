from boostracker.boostrackerpp import BoostTrackWrapper

def main():
        
    tracker = BoostTrackWrapper()

    # 이미지 파일 처리 (바운딩 박스 정보만 반환)
    boxes = tracker.predict("assets/sample1.png", return_type="bbox")

    # 이미지 파일 처리 (바운딩 박스가 그려진 이미지 반환)
    bbox_img = tracker.predict("assets/sample1.png", return_type="vis_bbox")

    # 이미지 파일 처리 (중심점이 그려진 이미지 반환)
    center_img = tracker.predict("assets/sample1.png", return_type="vis_center")

    # 이미지 파일 처리 (모든 결과 반환)
    all_results = tracker.predict("assets/sample1.png", return_type="all")

    # 비디오 파일 처리 (실시간 화면 표시하며 처리)
    video_result = tracker.predict(
        "assets/sample1.mp4", 
        return_type="all", 
        show_realtime=True, 
    )

    # 바운딩 박스만 실시간으로 보기
    bbox_result = tracker.predict(
        "assets/sample1.mp4", 
        return_type="vis_bbox", 
        show_realtime=True
    )

    # 중심점만 실시간으로 보기
    center_result = tracker.predict(
        "assets/sample1.mp4", 
        return_type="vis_center", 
        show_realtime=True
    )

    # 바운딩 박스 정보만 추출하면서 실시간으로 진행 상황 보기
    tracking_data = tracker.predict(
        "assets/sample1.mp4", 
        return_type="bbox", 
        show_realtime=True
    )

    # 처리 결과 출력
    print(f"처리된 총 프레임 수: {video_result['processed_frames']}")
    if 'bbox_video_path' in video_result:
        print(f"바운딩 박스 비디오 경로: {video_result['bbox_video_path']}")
    if 'center_video_path' in video_result:
        print(f"중심점 비디오 경로: {video_result['center_video_path']}")

if __name__ == "__main__":
    main()
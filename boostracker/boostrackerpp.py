import cv2
import torch
import numpy as np
import os
import tempfile
from external.adaptors.detector import Detector
from tracker.boost_track import BoostTrack
from typing import List, Dict, Tuple, Union, Optional

class BoostTrackWrapper:
    """
    BoostTrack 모델을 간편하게 사용할 수 있는 래퍼 클래스
    YOLOv8 스타일의 사용자 친화적 인터페이스 제공
    """
    
    def __init__(self, model_type="yolox", model_path="external/weights/bytetrack_x_mot20.tar", det_type="custom"):
        """
        BoostTrack 래퍼 클래스 초기화
        
        Args:
            model_type (str): 사용할 모델 타입 (기본값: "yolox")
            model_path (str): 모델 가중치 경로 (기본값: "external/weights/bytetrack_x_mot20.tar")
            det_type (str): 탐지 타입 (기본값: "custom")
        """
        self.detector = Detector(model_type, model_path, det_type)
        self.tracker = BoostTrack()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.object_positions = {}  # {track_id: {"start": (x, y), "last": (x, y)}}
        self.object_speeds = {}     # {track_id: speed (px/s)}
        self.object_last_frame = {} # {track_id: 마지막 프레임 번호}
        self.frame_count = 0

        # ECC 관련 에러 방지를 위한 이미지 크기 저장
        self.standard_size = None
    
    def predict(self, source: Union[str, np.ndarray], return_type: str = "bbox", show_realtime: bool = False, window_name: str = "BoostTrack Tracking", display_size: Tuple[int, int] = None) -> Union[List, np.ndarray]:
        """
        이미지나 비디오 소스로부터 객체 탐지 및 추적 수행
        
        Args:
            source (str or np.ndarray): 이미지 경로, 비디오 경로 또는 이미지 배열
            return_type (str): 반환할 결과 타입 ("bbox", "vis_bbox", "vis_center", "all")
            show_realtime (bool): 실시간으로 처리 과정을 화면에 표시할지 여부
            window_name (str): 실시간 표시 창의 이름
            display_size (Tuple[int, int]): 표시 창의 크기 (width, height), None이면 원본 크기
            
        Returns:
            결과 타입에 따라 다양한 형태로 반환:
            - "bbox": 바운딩 박스 좌표(x1, y1, x2, y2)와 ID 리스트
            - "vis_bbox": 바운딩 박스가 그려진 시각화 이미지
            - "vis_center": 중심점이 그려진 시각화 이미지
            - "all": 모든 결과를 포함하는 딕셔너리
        """
        if isinstance(source, str):
            if os.path.isfile(source):
                # 파일 확장자 확인
                ext = os.path.splitext(source)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    return self._predict_image(source, return_type)
                elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    return self._predict_video(source, return_type, show_realtime, window_name, display_size)
                else:
                    raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")
            else:
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {source}")
        
        elif isinstance(source, np.ndarray):
            return self._process_frame(source, return_type)
        
        else:
            raise TypeError("source는 파일 경로(str) 또는 이미지 배열(np.ndarray)이어야 합니다.")
    
    def _predict_image(self, image_path: str, return_type: str) -> Union[List, np.ndarray, Dict]:
        """
        이미지 파일에서 객체 탐지 및 추적 수행
        
        Args:
            image_path (str): 이미지 파일 경로
            return_type (str): 반환할 결과 타입
            
        Returns:
            요청된 return_type에 따른 결과
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        return self._process_frame(image, return_type)
    
    def _predict_video(self, video_path: str, return_type: str, show_realtime: bool = False, window_name: str = "BoostTrack Tracking", display_size: Tuple[int, int] = None) -> Dict:
        """
        비디오 파일에서 객체 탐지 및 추적 수행
        
        Args:
            video_path (str): 비디오 파일 경로
            return_type (str): 반환할 결과 타입
            show_realtime (bool): 실시간으로 처리 과정을 화면에 표시할지 여부
            window_name (str): 실시간 표시 창의 이름
            display_size (Tuple[int, int]): 표시 창의 크기 (width, height), None이면 원본 크기
            
        Returns:
            처리 결과를 포함하는 딕셔너리
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
            
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.standard_size = (frame_width, frame_height)
            
            if show_realtime:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                if display_size is not None:
                    cv2.resizeWindow(window_name, display_size[0], display_size[1])
                else:
                    cv2.resizeWindow(window_name, frame_width, frame_height)
            
            output_dir = os.path.join("results", "output")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(video_path)
            
            bbox_output_path = None
            center_output_path = None
            
            if return_type == "vis_bbox" or return_type == "all":
                bbox_output_path = os.path.join(output_dir, f"bbox_{base_name}")
                bbox_writer = cv2.VideoWriter(
                    bbox_output_path, 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    frame_rate, 
                    (frame_width, frame_height)
                )
            
            if return_type == "vis_center" or return_type == "all":
                center_output_path = os.path.join(output_dir, f"center_{base_name}")
                center_writer = cv2.VideoWriter(
                    center_output_path, 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    frame_rate, 
                    (frame_width, frame_height)
                )
            
            all_bboxes = []
            self.frame_count = 0
            self.tracker = BoostTrack()
            prev_time = cv2.getTickCount()
            fps = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.frame_count += 1
                print(f"Processing frame {self.frame_count}/{total_frames}")
                
                try:
                    start_time = cv2.getTickCount()
                    
                    result = self._process_frame(frame, return_type)
                    
                    current_time = cv2.getTickCount()
                    time_diff = (current_time - prev_time) / cv2.getTickFrequency()
                    if time_diff > 0:
                        fps = 1.0 / time_diff
                    prev_time = current_time
                    
                    if return_type == "bbox":
                        all_bboxes.append(result)
                        if show_realtime:
                            display_frame = frame.copy()
                            for box_info in result:
                                x1, y1, x2, y2 = box_info["bbox"]
                                track_id = box_info["id"]
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            cv2.putText(display_frame, f"FPS: {fps:.2f} | Frame: {self.frame_count}/{total_frames}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            cv2.imshow(window_name, display_frame)
                            
                            key = cv2.waitKey(1)
                            if key == 27:  # ESC 키
                                print("사용자에 의해 처리가 중단되었습니다.")
                                break
                    
                    elif return_type == "vis_bbox":
                        bbox_writer.write(result)
                        
                        if show_realtime:
                            cv2.putText(result, f"FPS: {fps:.2f} | Frame: {self.frame_count}/{total_frames}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            cv2.imshow(window_name, result)
                            key = cv2.waitKey(1)
                            if key == 27:  # ESC 키
                                print("사용자에 의해 처리가 중단되었습니다.")
                                break
                    
                    elif return_type == "vis_center":
                        center_writer.write(result)
                        
                        if show_realtime:
                            cv2.putText(result, f"FPS: {fps:.2f} | Frame: {self.frame_count}/{total_frames}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            cv2.imshow(window_name, result)
                            key = cv2.waitKey(1)
                            if key == 27:  # ESC 키
                                print("사용자에 의해 처리가 중단되었습니다.")
                                break
                    
                    elif return_type == "all":
                        all_bboxes.append(result["bboxes"])
                        bbox_writer.write(result["vis_bbox"])
                        center_writer.write(result["vis_center"])
                        
                        if show_realtime:
                            bbox_display = result["vis_bbox"].copy()
                            cv2.putText(bbox_display, f"FPS: {fps:.2f} | Frame: {self.frame_count}/{total_frames}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            center_display = result["vis_center"].copy()
                            cv2.putText(center_display, f"FPS: {fps:.2f} | Frame: {self.frame_count}/{total_frames}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            combined_display = np.hstack((bbox_display, center_display))
                            
                            if combined_display.shape[1] > 1920:
                                scale = 1920 / combined_display.shape[1]
                                new_width = int(combined_display.shape[1] * scale)
                                new_height = int(combined_display.shape[0] * scale)
                                combined_display = cv2.resize(combined_display, (new_width, new_height))
                            
                            cv2.imshow(window_name, combined_display)
                            key = cv2.waitKey(1)
                            if key == 27:  # ESC 키
                                print("사용자에 의해 처리가 중단되었습니다.")
                                break
                
                except Exception as e:
                    print(f"프레임 {self.frame_count} 처리 중 오류 발생: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 자원 해제
            cap.release()
            if return_type == "vis_bbox" or return_type == "all":
                bbox_writer.release()
            if return_type == "vis_center" or return_type == "all":
                center_writer.release()
            
            # 창 닫기
            if show_realtime:
                cv2.destroyWindow(window_name)
            
            # 결과 반환
            result_dict = {
                "total_frames": total_frames,
                "frame_rate": frame_rate,
                "processed_frames": self.frame_count
            }
            
            if return_type == "bbox":
                result_dict["bboxes"] = all_bboxes
            
            elif return_type == "vis_bbox":
                result_dict["video_path"] = bbox_output_path
            
            elif return_type == "vis_center":
                result_dict["video_path"] = center_output_path
            
            elif return_type == "all":
                result_dict["bboxes"] = all_bboxes
                result_dict["bbox_video_path"] = bbox_output_path
                result_dict["center_video_path"] = center_output_path
            
            print("비디오 처리 완료, 결과 반환:")
            print(result_dict)
            return result_dict
            
        except Exception as e:
            print(f"비디오 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 발생 시에도 기본 결과 제공
            return {
                "error": str(e),
                "total_frames": 0,
                "processed_frames": self.frame_count
            }   

    def _process_frame(self, frame: np.ndarray, return_type: str) -> Union[List, np.ndarray, Dict]:
        """
        단일 프레임에서 객체 탐지 및 추적 수행
        
        Args:
            frame (np.ndarray): 처리할 이미지 프레임
            return_type (str): 반환할 결과 타입
            
        Returns:
            요청된 return_type에 따른 결과
        """
        original_size = frame.shape[:2]  # 원본 이미지 크기 저장
        # YOLOX는 특정 입력 크기를 요구할 수 있으므로, 크기 조정
        h, w = original_size
        new_h, new_w = h, w
        
        # 32의 배수가 되도록 조정
        if h % 32 != 0:
            new_h = (h // 32 + 1) * 32
        if w % 32 != 0:
            new_w = (w // 32 + 1) * 32
            
        if new_h != h or new_w != w:
            resized_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 114  # 회색 배경
            resized_img[:h, :w] = frame  # 원본 이미지 배치
            frame = resized_img
        
        img_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img = torch.from_numpy(img_numpy).permute(2, 0, 1).float()
        if self.device.type == "cuda":
            img = img.cuda()
        img /= 255.0
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.detector.detect(img)
        
        if outputs is None:
            # 탐지된 객체가 없을 경우
            if return_type == "bbox":
                return []
            elif return_type == "vis_bbox":
                return frame[:h, :w].copy()  
            elif return_type == "vis_center":
                center_vis = np.ones((h, w, 3), dtype=np.uint8) * 255  
                return center_vis
            elif return_type == "all":
                return {
                    "bboxes": [],
                    "vis_bbox": frame[:h, :w].copy(),  
                    "vis_center": np.ones((h, w, 3), dtype=np.uint8) * 255  
                }
        
        targets = self.tracker.update(outputs, img, img_numpy, "custom_video")
        if isinstance(targets, np.ndarray):
            targets = [targets]
        
        bbox_results = []
        
        bbox_vis = frame[:h, :w].copy()  
        center_vis = np.ones((h, w, 3), dtype=np.uint8) * 255  
        
        for target in targets:
            if isinstance(target, np.ndarray) and target.ndim == 2:
                for obj in target:
                    x1, y1, x2, y2, track_id = map(int, obj[:5])
                    
                    # 크기 조정된 경우 좌표 조정
                    if new_h != h or new_w != w:
                        # 원본 크기를 벗어나는 좌표 처리
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(0, min(x2, w-1))
                        y2 = max(0, min(y2, h-1))
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    bbox_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "id": track_id,
                        "center": center
                    })
                    
                    cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bbox_vis, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.circle(center_vis, center, 5, (0, 0, 255), -1)  
                    cv2.putText(center_vis, f"{track_id}", (center[0] + 10, center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if return_type == "bbox":
            return bbox_results
        
        elif return_type == "vis_bbox":
            return bbox_vis
        
        elif return_type == "vis_center":
            return center_vis
        
        elif return_type == "all":
            return {
                "bboxes": bbox_results,
                "vis_bbox": bbox_vis,
                "vis_center": center_vis
            }
        
        else:
            raise ValueError(f"지원하지 않는 return_type입니다: {return_type}")


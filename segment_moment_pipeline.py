"""
Segment-based Moment Retrieval Pipeline
Timestamp 기반으로 비디오 구간을 잘라서 frame selection 후 moment retrieval 수행
"""

import os
import subprocess
import tempfile
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import sys

sys.path.append('/home/intern/a2a_project/retrieval/vtl_retrieval_agent2/lighthouse')
from lighthouse.models import CGDETRPredictor
from lighthouse.common.utils.span_utils import span_cxw_to_xx


class SegmentMomentRetrievalPipeline:
    """
    Timestamp 기반 구간별 Moment Retrieval 파이프라인
    """
    
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        slowfast_path: str = "/home/intern/a2a_project/retrieval/lighthouse/SLOWFAST_8x8_R50.pkl",
        pann_path: Optional[str] = None,
        feature_name: str = "clip_slowfast",
        max_moments: int = 10,
        confidence_threshold: float = 0.0,
    ):
        """
        Args:
            ckpt_path: CG-DETR 체크포인트 파일 경로
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
            slowfast_path: SlowFast 모델 가중치 파일 경로
            pann_path: PANN 모델 가중치 파일 경로 (선택사항)
            feature_name: 사용할 특징 추출기 이름
            max_moments: 반환할 최대 moment 개수
            confidence_threshold: confidence 임계값
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.max_moments = max_moments
        self.confidence_threshold = confidence_threshold
        
        # 모델 초기화
        self.model = CGDETRPredictor(
            ckpt_path=ckpt_path,
            device=self.device,
            feature_name=feature_name,
            slowfast_path=slowfast_path,
            pann_path=pann_path
        )
        
        # 모델 정보 저장
        self.clip_len = self.model._clip_len
        self.feature_name = feature_name
        
    def _extract_video_segment(self, video_path: str, start_time: float, end_time: float) -> str:
        """
        비디오에서 특정 구간을 추출하여 임시 파일로 저장합니다.
        
        Args:
            video_path: 원본 비디오 파일 경로
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            
        Returns:
            추출된 비디오 구간의 임시 파일 경로
        """
        if start_time >= end_time:
            raise ValueError(f"Invalid time range: start_time ({start_time}) >= end_time ({end_time})")
        
        print(f"[Segment Pipeline] Extracting segment: {video_path} [{start_time:.2f}s - {end_time:.2f}s]")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # ffmpeg를 사용하여 구간 추출
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                temp_path,
                '-y'  # 덮어쓰기 허용
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            
            return temp_path
            
        except Exception as e:
            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

    def encode_video_segment(self, video_path: str, start_time: float, end_time: float) -> Dict[str, Optional[torch.Tensor]]:
        """
        비디오의 특정 구간을 인코딩하여 특징을 추출합니다.
        
        Args:
            video_path: 비디오 파일 경로
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            
        Returns:
            인코딩된 비디오 구간 특징 딕셔너리
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        temp_segment_path = None
        try:
            # 1. 구간 추출
            temp_segment_path = self._extract_video_segment(video_path, start_time, end_time)
            print(f"[Segment Pipeline] Segment extracted: {temp_segment_path}")
            
            # 2. 추출된 구간을 인코딩
            video_features = self.model.encode_video(temp_segment_path)
            print(f"[Segment Pipeline] Segment encoding completed. Features shape: {video_features['video_feats'].shape}")
            
            return video_features
            
        finally:
            # 임시 파일 정리
            if temp_segment_path and os.path.exists(temp_segment_path):
                os.unlink(temp_segment_path)

    def predict_moments_in_segment(
        self, 
        query: str, 
        video_path: str,
        start_time: float,
        end_time: float
    ) -> Optional[Dict[str, Union[List[float], List[List[float]]]]]:
        """
        비디오의 특정 구간에서 쿼리에 대한 moment를 예측합니다.
        
        Args:
            query: 검색 쿼리 텍스트
            video_path: 비디오 파일 경로
            start_time: 구간 시작 시간 (초)
            end_time: 구간 종료 시간 (초)
            
        Returns:
            예측 결과 딕셔너리 (원본 비디오 기준의 절대 시간으로 조정됨)
        """
        print(f"[Segment Pipeline] Predicting moments in segment: '{query}' [{start_time:.2f}s - {end_time:.2f}s]")
        
        # 구간 인코딩
        segment_features = self.encode_video_segment(video_path, start_time, end_time)
        
        # 구간에서 moment 예측
        prediction = self.model.predict(query, segment_features)
        
        if prediction is None:
            print("[Segment Pipeline] No moments found in the segment.")
            return None
        
        # 시간을 원본 비디오 기준으로 조정
        adjusted_moments = []
        for moment in prediction['pred_relevant_windows']:
            if len(moment) >= 3:
                # 구간 내 상대 시간을 원본 비디오의 절대 시간으로 변환
                segment_start, segment_end, confidence = moment[0], moment[1], moment[2]
                absolute_start = start_time + segment_start
                absolute_end = start_time + segment_end
                
                # 원본 구간을 벗어나지 않도록 클리핑
                absolute_start = max(absolute_start, start_time)
                absolute_end = min(absolute_end, end_time)
                
                if absolute_start < absolute_end:
                    adjusted_moments.append([absolute_start, absolute_end, confidence])
        
        if not adjusted_moments:
            print("[Segment Pipeline] No valid moments found after time adjustment.")
            return None
        
        adjusted_prediction = {
            'pred_relevant_windows': adjusted_moments,
            'pred_saliency_scores': prediction['pred_saliency_scores'][:len(adjusted_moments)]
        }
        
        print(f"[Segment Pipeline] Found {len(adjusted_moments)} moments in segment (adjusted to absolute time)")
        return adjusted_prediction

    def process_video_segment_query(
        self, 
        video_path: str, 
        query: str,
        start_time: float,
        end_time: float
    ) -> Dict[str, Union[List[List[float]], List[float], Dict[str, float]]]:
        """
        비디오의 특정 구간에서 쿼리를 처리하여 moment retrieval을 수행합니다.
        
        Args:
            video_path: 비디오 파일 경로
            query: 검색 쿼리 텍스트
            start_time: 구간 시작 시간 (초)
            end_time: 구간 종료 시간 (초)
            
        Returns:
            결과 딕셔너리:
            - candidates: moment 후보들 (절대 시간 기준)
            - scores: 각 moment의 saliency scores
            - metadata: 비디오 및 구간 정보
        """
        try:
            # 1. 구간에서 Moment 예측
            prediction = self.predict_moments_in_segment(query, video_path, start_time, end_time)
            
            if prediction is None:
                return {
                    "candidates": [],
                    "scores": [],
                    "metadata": {
                        "video_path": video_path,
                        "query": query,
                        "segment_start": start_time,
                        "segment_end": end_time,
                        "segment_duration": end_time - start_time,
                        "status": "no_moments_found"
                    }
                }
            
            # 2. 결과 필터링
            candidates, scores = self._filter_moments(prediction)
            
            # 3. 메타데이터 생성
            metadata = {
                "video_path": video_path,
                "query": query,
                "segment_start": float(start_time),
                "segment_end": float(end_time),
                "segment_duration": float(end_time - start_time),
                "clip_length": float(self.clip_len),
                "feature_name": self.feature_name,
                "device": self.device,
                "status": "success"
            }
            
            return {
                "candidates": candidates,
                "scores": scores,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "candidates": [],
                "scores": [],
                "metadata": {
                    "video_path": video_path,
                    "query": query,
                    "segment_start": start_time,
                    "segment_end": end_time,
                    "segment_duration": end_time - start_time,
                    "status": "error",
                    "error_message": str(e)
                }
            }
    
    def _filter_moments(
        self, 
        prediction: Dict[str, Union[List[float], List[List[float]]]]
    ) -> Tuple[List[List[float]], List[float]]:
        """
        예측된 moments를 필터링합니다.
        
        Args:
            prediction: 모델 예측 결과
            
        Returns:
            필터링된 moments와 scores
        """
        moments = prediction['pred_relevant_windows']
        scores = prediction['pred_saliency_scores']
        
        # threshold 없이 모든 moment 사용 (confidence_threshold가 0.0이므로)
        filtered_moments = moments.copy()
        filtered_scores = scores.copy()
        
        # max_moments 개수로 제한
        if len(filtered_moments) > self.max_moments:
            filtered_moments = filtered_moments[:self.max_moments]
            filtered_scores = filtered_scores[:self.max_moments]
        
        return filtered_moments, filtered_scores

    # 전체 비디오 처리 메서드들 (호환성 유지)
    def encode_video(self, video_path: str) -> Dict[str, Optional[torch.Tensor]]:
        """전체 비디오를 인코딩합니다."""
        return self.model.encode_video(video_path)
    
    def predict_moments(
        self, 
        query: str, 
        video_features: Dict[str, Optional[torch.Tensor]]
    ) -> Optional[Dict[str, Union[List[float], List[List[float]]]]]:
        """전체 비디오에서 moment를 예측합니다."""
        return self.model.predict(query, video_features)

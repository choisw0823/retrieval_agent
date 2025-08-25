"""
MR (Moment Retrieval) Backend
비디오에서 moment retrieval을 수행하는 백엔드 클래스들
"""

import os
import sys
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# SegmentMomentRetrievalPipeline import
from segment_moment_pipeline import SegmentMomentRetrievalPipeline

# Type definitions
Span = Dict[str, Any]  # {t0,t1,score?,conf?,saliency?,tag?,inst_id?,meta?}


class MRBackend(ABC):
    """MR 백엔드의 추상 클래스"""
    
    @abstractmethod
    def fetch(self, query_text: str, window: Optional[List[float]], topk: int, hint: Dict[str, Any]) -> List[Span]:
        """
        쿼리에 대한 moment retrieval 수행
        
        Args:
            query_text: 검색 쿼리
            window: 시간 윈도우 제한 [start_time, end_time] (초 단위)
            topk: 반환할 최대 개수
            hint: 추가 힌트 정보
            
        Returns:
            검색된 moment들의 리스트
        """
        pass


class MomentRetrievalMR(MRBackend):
    """SegmentMomentRetrievalPipeline을 사용하는 MR 백엔드"""
    
    def __init__(self, 
                 video_path: str,
                 ckpt_path: str = "/home/intern/a2a_project/retrieval/lighthouse/weights/clip_slowfast_cg_detr_qvhighlight.ckpt",
                 device: str = "cuda",
                 slowfast_path: str = "/home/intern/a2a_project/retrieval/lighthouse/SLOWFAST_8x8_R50.pkl"):
        """
        Args:
            video_path: 처리할 비디오 파일 경로
            ckpt_path: CG-DETR 체크포인트 경로
            device: 사용할 디바이스
            slowfast_path: SlowFast 모델 경로
        """
        self.video_path = video_path
        
        # SegmentMomentRetrievalPipeline 초기화
        self.pipeline = SegmentMomentRetrievalPipeline(
            ckpt_path=ckpt_path,
            device=device,
            slowfast_path=slowfast_path,
            max_moments=50,  # 충분한 개수로 설정
            confidence_threshold=0.0,
        )

        self.video_features = None  # 변수만 초기화
        
        # 비디오를 미리 인코딩해서 캐시 (전체 영상 처리용)
        print(f"  [MR] Pre-encoding video: {video_path}")
        self.video_features = self.pipeline.encode_video(video_path)
        print(f"  [MR] Video encoding completed")
        
    def fetch(self, query_text: str, window: Optional[List[float]], topk: int, hint: Dict[str, Any]) -> List[Span]:
        """
        쿼리에 대한 moment retrieval 수행
        
        Args:
            query_text: 검색 쿼리
            window: 시간 윈도우 제한 [start_time, end_time] (초 단위)
            topk: 반환할 최대 개수
            hint: 추가 힌트 정보
            
        Returns:
            검색된 moment들의 리스트
        """
        print(f"  [MR FETCH] query='{query_text}' window={window} topk={topk}")
        
        try:
            # window가 지정된 경우 구간별 처리, 아니면 전체 영상 처리
            if window is not None:
                start_time, end_time = window[0], window[1]
                print(f"  [MR] Using segment-based processing for window [{start_time:.2f}s - {end_time:.2f}s]")
                prediction = self.pipeline.predict_moments_in_segment(
                    query_text, self.video_path, start_time, end_time
                )
            else:
                print(f"  [MR] Using full video processing")
                prediction = self.pipeline.predict_moments(query_text, self.video_features)
            
            if prediction is None:
                print(f"  [MR] No moments found for query: '{query_text}'")
                return []
            
            # 결과를 Span 형태로 변환
            moments = prediction['pred_relevant_windows']
            scores = prediction['pred_saliency_scores']
            
            spans = []
            for i, (moment, saliency_score) in enumerate(zip(moments, scores)):
                if len(moment) >= 3:
                    start_time, end_time, confidence = moment[0], moment[1], moment[2]
                    
                    # 윈도우 필터링 (지정된 경우)
                    if window is not None:
                        win_start, win_end = window[0], window[1]
                        # moment가 윈도우와 겹치는지 확인
                        if end_time < win_start or start_time > win_end:
                            continue
                        # 윈도우에 맞게 클리핑
                        start_time = max(start_time, win_start)
                        end_time = min(end_time, win_end)
                        if start_time >= end_time:
                            continue
                    if float(confidence) < 0.6:
                        continue
                    
                    span = {
                        "t0": float(start_time),
                        "t1": float(end_time),
                        "score": float(confidence),  # confidence score 사용
                        "conf": float(confidence),  # confidence도 저장
                        "tag": query_text,  # 쿼리를 태그로 사용
                        "inst_id": i,
                        "meta": {
                            "query": query_text,
                            "saliency_score": float(saliency_score),
                            "confidence": float(confidence)
                        }
                    }
                    spans.append(span)
            
            # 스코어 기준으로 정렬 (높은 순)
            spans.sort(key=lambda x: x["score"], reverse=True)
            
            # topk 개수만 반환
            result = spans[:topk]
            print(f"  [MR] Found {len(result)} moments (filtered from {len(spans)} candidates)")

            print(f"  - Result: {result}")
            
            return result
            
        except Exception as e:
            print(f"  [MR ERROR] Failed to fetch moments for '{query_text}': {e}")
            return []


if __name__ == "__main__":
    """
    MR Backend 테스트 코드
    사용법: python mr_backend.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="MR Backend 테스트")
    parser.add_argument("--video", type=str, 
                       default="/hub_data2/intern/jinwoo/FlashVTG/qvhighlight/videos/RoripwjYFp8_60.0_210.0.mp4",
                       help="비디오 파일 경로")
    parser.add_argument("--query", type=str, 
                       default="Fruit scene before man is talking ",
                       help="검색 쿼리")
    parser.add_argument("--window", type=str, default=None,
                       help="시간 윈도우 (예: '10,50' = 10초~50초)")
    parser.add_argument("--topk", type=int, default=5,
                       help="반환할 최대 개수")
    
    args = parser.parse_args()
    
    # 윈도우 파싱
    window = None
    if args.window:
        try:
            start, end = map(float, args.window.split(','))
            window = [start, end]
            print(f"[TEST] 시간 윈도우: [{start:.1f}s - {end:.1f}s]")
        except:
            print(f"[ERROR] 잘못된 윈도우 형식: {args.window} (예: '10,50')")
            exit(1)
    
    print(f"[TEST] 비디오: {args.video}")
    print(f"[TEST] 쿼리: '{args.query}'")
    print(f"[TEST] Top-K: {args.topk}")
    print("=" * 60)
    
    try:
        # MR 백엔드 초기화
        print("[1/2] MR 백엔드 초기화 중...")
        mr_backend = MomentRetrievalMR(
            video_path=args.video,
            device="cuda"
        )
        
        # 쿼리 실행
        print(f"[2/2] 쿼리 실행: '{args.query}'")
        results = mr_backend.fetch(
            query_text=args.query,
            window=window,
            topk=args.topk,
            hint={}
        )
        
        # 결과 출력
        print("=" * 60)
        print(f"[RESULTS] {len(results)}개 결과 발견:")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. 시간: {result['t0']:.2f}s - {result['t1']:.2f}s")
                print(f"     점수: {result['score']:.4f} (confidence)")
                print(f"     태그: {result['tag']}")
                if 'meta' in result:
                    print(f"     Saliency: {result['meta'].get('saliency_score', 'N/A'):.4f}")
                print()
        else:
            print("  결과가 없습니다.")
            
        print("=" * 60)
        print("[완료] 테스트 종료")
        
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

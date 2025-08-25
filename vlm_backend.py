"""
VLM (Vision-Language Model) Backend
비디오 구간에 대해 VLM 질문을 처리하는 백엔드 클래스들
"""

import os
import tempfile
import subprocess
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

# Type definitions  
Span = Dict[str, Any]  # {t0,t1,score?,conf?,saliency?,tag?,inst_id?,meta?}


def _geo_mean(vals: List[float]) -> float:
    """기하평균 계산"""
    import math
    vals = [max(1e-6, float(v)) for v in vals if v is not None]
    if not vals: return 0.0
    return float(math.exp(sum(math.log(v) for v in vals)/len(vals)))


class VLMBackend:
    """VLM 질문을 처리하는 백엔드"""
    
    def __init__(self, video_path: str):
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"VLMBackend requires a valid video path, but got: {video_path}")
        self.video_path = video_path
        
        # Replicate 클라이언트 초기화
        import replicate
        self.replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        print(f"  [VLM] Initialized with Replicate API")

    def _extract_video_segment(self, t0: float, t1: float) -> str:
        """
        비디오에서 특정 구간을 추출하여 임시 파일로 저장합니다.
        
        Args:
            t0: 시작 시간 (초)
            t1: 종료 시간 (초)
            
        Returns:
            추출된 비디오 구간의 임시 파일 경로
        """
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # ffmpeg로 구간 추출
            cmd = [
                'ffmpeg', '-i', self.video_path,
                '-ss', str(t0),
                '-t', str(t1 - t0),
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

    def _ask_vlm_question(self, video_segment_path: str, question: str) -> tuple[str, float]:
        """
        VLM에 비디오 구간과 질문을 전달하여 답변을 받습니다.
        
        Args:
            video_segment_path: 비디오 구간 파일 경로
            question: 질문
            
        Returns:
            (VLM 답변, 신뢰도 점수)
        """
        try:
            with open(video_segment_path, "rb") as input_video:
                output = self.replicate_client.run(
                    "chenxwh/cogvlm2-video:9da7e9a554d36bb7b5fec36b43b00e4616dc1e819bc963ded8e053d8d8196cb5",
                    input={
                        "prompt": question,
                        "input_video": input_video
                    }
                )
            
            # output 처리
            if isinstance(output, list) and len(output) > 0:
                answer = output[0]
            else:
                answer = str(output)
            
            # 답변에서 신뢰도 추정 (간단한 휴리스틱)
            confidence = 0.8  # 기본 신뢰도
            if "yes" in answer.lower() or "true" in answer.lower():
                confidence = 0.9
            elif "no" in answer.lower() or "false" in answer.lower():
                confidence = 0.7
            elif "maybe" in answer.lower() or "possibly" in answer.lower():
                confidence = 0.5
                
            return answer, confidence
            
        except Exception as e:
            print(f"  [VLM ERROR] Failed to get VLM response: {e}")
            return f"VLM query failed: {str(e)}", 0.1

    def query(self, t0: float, t1: float, question: str) -> tuple[bool, float]:
        """
        주어진 시간 구간과 질문으로 VLM에 질의합니다.
        
        Args:
            t0: 시작 시간 (초)
            t1: 종료 시간 (초)
            question: VLM에 할 질문 (예: "Is a person opening a door?")
            
        Returns:
            (질문이 참인가?, VLM의 신뢰도 점수)
        """
        print(f"  [VLM Query] Video: '{os.path.basename(self.video_path)}', "
              f"Time: [{t0:.2f}s, {t1:.2f}s], "
              f"Question: \"{question}\"")
        
        temp_segment_path = None
        try:
            # 1. 비디오 구간 추출
            temp_segment_path = self._extract_video_segment(t0, t1)
            print(f"  [VLM] Extracted segment: {temp_segment_path}")
            
            # 2. VLM에 질문
            answer, confidence = self._ask_vlm_question(temp_segment_path, question)
            print(f"  [VLM Response] Answer: '{answer}' (confidence: {confidence:.3f})")
            
            # 3. 답변을 boolean으로 변환
            is_positive = self._parse_answer_to_boolean(answer)
            
            return (is_positive, confidence)
            
        except Exception as e:
            print(f"  [VLM ERROR] Query failed: {e}")
            return (False, 0.1)
        
        finally:
            # 임시 파일 정리
            if temp_segment_path and os.path.exists(temp_segment_path):
                os.unlink(temp_segment_path)

    def _parse_answer_to_boolean(self, answer: str) -> bool:
        """
        VLM 답변을 boolean으로 파싱합니다.
        
        Args:
            answer: VLM 답변 텍스트
            
        Returns:
            답변이 긍정적인지 여부
        """
        answer_lower = answer.lower()
        
        # 명확한 긍정 답변
        positive_indicators = ["yes", "true", "correct", "right", "indeed", "definitely", "certainly"]
        if any(indicator in answer_lower for indicator in positive_indicators):
            return True
        
        # 명확한 부정 답변
        negative_indicators = ["no", "false", "incorrect", "wrong", "not", "cannot", "unable"]
        if any(indicator in answer_lower for indicator in negative_indicators):
            return False
        
        # 애매한 경우 - 답변 길이와 내용으로 추정
        if len(answer) > 50:  # 긴 답변은 보통 긍정적
            return True
        
        # 기본값은 False (보수적 접근)
        return False


def generate_vlm_question(condition: Dict[str, Any], t0: float, t1: float) -> str:
    """Join 조건으로부터 VLM에 할 질문을 생성합니다."""
    op = condition.get("op")
    time_str = f"In the time interval [{t0:.2f}s, {t1:.2f}s]"

    if op == "ACTION":
        verb = condition.get("verb", "interacting with")
        # In a real system, you'd get the object labels from aliases
        actor_label = "the first object"
        object_label = "the second object"
        return f"{time_str}, is {actor_label} {verb} {object_label}?"
    
    if op == "RELATION":
        rel_type = condition.get("type", "related to").replace("_", " ")
        left_label = "the first object"
        right_label = "the second object"
        return f"{time_str}, is {left_label} {rel_type} {right_label}?"
        
    return f"{time_str}, do the two events have the specified relationship?"


def join_with_vlm_verification(
    A: List[Span], B: List[Span], join_node: Any, vlm_backend: VLMBackend
) -> List[Span]:
    """시간적으로 겹치는 후보군에 대해 VLM으로 관계/행동을 검증하는 Join"""
    A = sorted(A, key=lambda s: (s["t0"], s["t1"]))
    B = sorted(B, key=lambda s: (s["t0"], s["t1"]))
    i = j = 0
    out: List[Span] = []

    while i < len(A) and j < len(B):
        a, b = A[i], B[j]
        inter_t0 = max(a["t0"], b["t0"])
        inter_t1 = min(a["t1"], b["t1"])

        if inter_t1 > inter_t0: # 1. 시간적으로 겹치는 후보 찾기
            # 2. VLM에 할 질문 생성
            question = generate_vlm_question(join_node.condition, inter_t0, inter_t1)
            
            # 3. VLM에 질문하여 관계/행동 검증
            is_true, vlm_score = vlm_backend.query(inter_t0, inter_t1, question)
            
            if is_true:
                # 4. 검증 성공 시, 최종 점수 계산 및 결과 추가
                # 부모의 confidence와 VLM의 confidence를 모두 결합
                final_score = _geo_mean([a.get("conf"), b.get("conf"), vlm_score])
                out.append({
                    "t0": inter_t0, "t1": inter_t1,
                    "score": final_score,
                    "conf": final_score, # VLM 검증 후 confidence는 score와 동일
                    "meta": {"vlm_question": question, "vlm_score": vlm_score}
                })

        if a["t1"] <= b["t1"]:
            i += 1
        else:
            j += 1
    return out

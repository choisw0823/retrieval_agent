# inspector_join.py
# ------------------------------------------------------------
# JOIN 연산 전용 검사기 (AND JOIN - DURING, ACTION/RELATION JOIN)
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import statistics

Span = Dict[str, Any]

EPS_NEAR = 0.5           # 초단위 근접(threshold)
OVERLONG_FACTOR = 2.5    # 중앙길이 × 계수

# ---------- JOIN 이슈 유형별 권장 조치 목록 ----------
ISSUE_ACTIONS = {
    "EmptyLeft": [
        "Review action history to identify potential time ranges where left input events might occur. Divide the video into shorter segments and examine each systematically.",
    ],
    "EmptyRight": [
        "Review action history to identify potential time ranges where right input events might occur. Divide the video into shorter segments and examine each systematically.",
    ],
    "NoIntersection": [
        """Left and right spans do not intersect temporally. Check if events are truly simultaneous or if one contains the other.
        For DURING joins: Look for containment relationships where one event happens within another.
        For ACTION/RELATION joins: Both conditions must be satisfied simultaneously (AND operation).""",
        "Expand temporal windows of both inputs to capture broader context around potential intersection points."
    ],
    "OverlongIntersection": [
        "Split overly long intersection spans that incorrectly merge multiple distinct events into smaller, precise segments.",
    ],
    "WeakOverlap": [
        "For minimal overlaps: Verify if the overlap represents a genuine intersection or coincidental timing.",
        "Consider expanding the temporal context around weak overlap regions to capture the full event scope.",
    ]
}

# ---------- 유틸 ----------

def _conf(s: Span) -> float:
    """Span의 confidence 값을 추출합니다."""
    return float(s.get("conf", s.get("score", 0.0)))

def _len(s: Span) -> float:
    """Span의 길이를 계산합니다."""
    return float(s["t1"]) - float(s["t0"])

def _median_length(spans: List[Span]) -> float:
    """Span들의 중앙값 길이를 계산합니다."""
    if not spans:
        return 0.0
    lengths = [_len(s) for s in spans]
    return statistics.median(lengths)

def _check_during_intersection(left: Span, right: Span) -> bool:
    """DURING 관계 확인 - 한 span이 다른 span에 포함되는지"""
    l_start, l_end = left["t0"], left["t1"]
    r_start, r_end = right["t0"], right["t1"]
    
    # left가 right에 포함되거나 right가 left에 포함되면 true
    return (r_start <= l_start and l_end <= r_end) or (l_start <= r_start and r_end <= l_end)

def _check_temporal_overlap(left: Span, right: Span) -> bool:
    """시간적 겹침 확인"""
    return left["t0"] < right["t1"] and right["t0"] < left["t1"]

def _compute_intersection(left: Span, right: Span) -> Dict[str, float]:
    """두 span의 교집합 계산"""
    t0 = max(left["t0"], right["t0"])
    t1 = min(left["t1"], right["t1"])
    if t0 < t1:
        return {"t0": t0, "t1": t1, "length": t1 - t0}
    return {"t0": 0, "t1": 0, "length": 0}

# ---------- 검사 함수들 ----------

def _check_empty_inputs(left_spans: List[Span], right_spans: List[Span]) -> List[Dict[str, Any]]:
    """빈 입력 검사"""
    issues = []
    
    if not left_spans:
        issues.append({
            "type": "EmptyLeft",
            "description": "Left input has no spans",
            "priority": 1,
            "actions": ISSUE_ACTIONS["EmptyLeft"]
        })
    
    if not right_spans:
        issues.append({
            "type": "EmptyRight", 
            "description": "Right input has no spans",
            "priority": 1,
            "actions": ISSUE_ACTIONS["EmptyRight"]
        })
    
    return issues

def _check_intersections(left_spans: List[Span], right_spans: List[Span], 
                        join_type: str = "DURING") -> List[Dict[str, Any]]:
    """교집합 관련 이슈 검사"""
    issues = []
    intersections = []
    
    for i, left in enumerate(left_spans):
        for j, right in enumerate(right_spans):
            if join_type == "DURING":
                has_intersection = _check_during_intersection(left, right)
            else:  # ACTION, RELATION
                has_intersection = _check_temporal_overlap(left, right)
            
            if has_intersection:
                intersection = _compute_intersection(left, right)
                if intersection["length"] > 0:
                    intersections.append({
                        "left_idx": i, "right_idx": j,
                        "left_span": left, "right_span": right,
                        "intersection": intersection
                    })
    
    if not intersections:
        issues.append({
            "type": "NoIntersection",
            "description": f"No valid {join_type} intersections found between left and right spans",
            "priority": 2,
            "actions": ISSUE_ACTIONS["NoIntersection"],
            "left_count": len(left_spans),
            "right_count": len(right_spans)
        })
        return issues
    
    # 교집합 길이 분석
    intersection_lengths = [item["intersection"]["length"] for item in intersections]
    median_length = statistics.median(intersection_lengths)
    
    # 너무 긴 교집합 검사
    for item in intersections:
        intersection_len = item["intersection"]["length"]
        if intersection_len > median_length * OVERLONG_FACTOR:
            issues.append({
                "type": "OverlongIntersection",
                "description": f"Intersection span too long: {intersection_len:.2f}s (median: {median_length:.2f}s)",
                "priority": 3,
                "actions": ISSUE_ACTIONS["OverlongIntersection"],
                "left_idx": item["left_idx"],
                "right_idx": item["right_idx"],
                "intersection": item["intersection"]
            })
    
    # 약한 겹침 검사
    weak_threshold = median_length * 0.3
    for item in intersections:
        intersection_len = item["intersection"]["length"]
        if intersection_len < weak_threshold and intersection_len > 0:
            issues.append({
                "type": "WeakOverlap",
                "description": f"Weak intersection: {intersection_len:.2f}s (threshold: {weak_threshold:.2f}s)",
                "priority": 4,
                "actions": ISSUE_ACTIONS["WeakOverlap"],
                "left_idx": item["left_idx"],
                "right_idx": item["right_idx"],
                "intersection": item["intersection"]
            })
    
    return issues

def _compute_join_result(left_spans: List[Span], right_spans: List[Span], 
                        join_type: str = "DURING") -> List[Dict[str, Any]]:
    """JOIN 결과 계산"""
    result_spans = []
    
    for left in left_spans:
        for right in right_spans:
            if join_type == "DURING":
                if _check_during_intersection(left, right):
                    # DURING: 교집합 영역
                    intersection = _compute_intersection(left, right)
                    if intersection["length"] > 0:
                        result_spans.append({
                            "t0": intersection["t0"],
                            "t1": intersection["t1"],
                            "conf": min(_conf(left), _conf(right)),
                            "source": "join_during"
                        })
            elif join_type in ["ACTION", "RELATION"]:
                if _check_temporal_overlap(left, right):
                    # ACTION과 RELATION 모두 교집합 영역 계산 (AND 연산)
                    intersection = _compute_intersection(left, right)
                    if intersection["length"] > 0:
                        source = "join_action" if join_type == "ACTION" else "join_relation"
                        result_spans.append({
                            "t0": intersection["t0"],
                            "t1": intersection["t1"],
                            "conf": min(_conf(left), _conf(right)),
                            "source": source
                        })
    
    # 중복 제거
    seen = set()
    unique_spans = []
    for s in result_spans:
        key = (round(s["t0"], 5), round(s["t1"], 5))
        if key not in seen:
            seen.add(key)
            unique_spans.append(s)
    
    return unique_spans

# ---------- 메인 검사 함수 ----------

def inspect_join(left_spans: List[Span], right_spans: List[Span], 
                join_type: str = "DURING", **kwargs) -> Dict[str, Any]:
    """
    JOIN 연산 검사 함수
    
    Args:
        left_spans: 왼쪽 입력 spans
        right_spans: 오른쪽 입력 spans  
        join_type: JOIN 타입 ("DURING", "ACTION", "RELATION")
        **kwargs: 추가 매개변수
        
    Returns:
        검사 결과 딕셔너리
    """
    issues = []
    
    # 1. 빈 입력 검사
    issues.extend(_check_empty_inputs(left_spans, right_spans))
    
    # 빈 입력이 있으면 추가 검사 중단
    if any(issue["type"] in ["EmptyLeft", "EmptyRight"] for issue in issues):
        return {
            "issues": issues,
            "result_spans": [],
            "summary": f"JOIN inspection failed: empty inputs (left: {len(left_spans)}, right: {len(right_spans)})"
        }
    
    # 2. 교집합 관련 검사
    issues.extend(_check_intersections(left_spans, right_spans, join_type))
    
    # 3. JOIN 결과 계산
    result_spans = _compute_join_result(left_spans, right_spans, join_type)
    
    # 4. 결과 요약
    summary = f"JOIN {join_type} inspection: {len(left_spans)} left × {len(right_spans)} right → {len(result_spans)} results, {len(issues)} issues"
    
    return {
        "issues": issues,
        "result_spans": result_spans,
        "left_count": len(left_spans),
        "right_count": len(right_spans),
        "result_count": len(result_spans),
        "join_type": join_type,
        "summary": summary
    }

# ---------- 편의 함수들 ----------

def inspect_during_join(left_spans: List[Span], right_spans: List[Span], **kwargs) -> Dict[str, Any]:
    """DURING JOIN 전용 검사"""
    return inspect_join(left_spans, right_spans, "DURING", **kwargs)

def inspect_action_join(left_spans: List[Span], right_spans: List[Span], **kwargs) -> Dict[str, Any]:
    """ACTION JOIN 전용 검사"""
    return inspect_join(left_spans, right_spans, "ACTION", **kwargs)

def inspect_relation_join(left_spans: List[Span], right_spans: List[Span], **kwargs) -> Dict[str, Any]:
    """RELATION JOIN 전용 검사"""
    return inspect_join(left_spans, right_spans, "RELATION", **kwargs)


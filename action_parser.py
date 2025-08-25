#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
action_parser.py - 통합 액션 처리 모듈

모든 액션들을 처리:
- split: 특정 candidate 구간에 대해서 timestamp 리스트를 받아서 구간을 분할
- concat: 여러 candidate들을 받아서 하나의 구간으로 처리하고, 기존 구간들 삭제
- probe: 새로운 후보들을 찾거나 더 나은 후보들을 찾기
- verify: VLM을 사용해서 candidate pairs를 검증
- remove: VLM을 사용해서 조건에 맞지 않는 span들 제거
- stop: LLM이 제안한 해결책을 적용
"""

from typing import Dict, List, Any, Tuple, Optional
import copy

Span = Dict[str, Any]

def _conf(s: Span) -> float:
    """Span의 confidence 값을 추출합니다."""
    return float(s.get("conf", s.get("score", 0.0)))

def _summ(s: Span) -> Dict[str, Any]: 
    """Span을 요약 형태로 변환"""
    return {"t0": s["t0"], "t1": s["t1"], "conf": float(s.get("conf", s.get("score", 1.0)))}

def _dedup_spans(spans: List[Span]) -> List[Span]:
    """중복 span 제거"""
    seen = set()
    out = []
    for s in spans:
        k = (round(float(s["t0"]), 5), round(float(s["t1"]), 5), s.get("tag", ""))
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

def _pairs_to_AB(pairs: List[Dict[str, Any]]) -> Tuple[List[Span], List[Span]]:
    """Pairs를 A, B 리스트로 변환"""
    def _f(s): 
        return {"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(s.get("conf", 1.0))}
    A = [_f(p.get("a", p.get("A"))) for p in pairs if p.get("a") or p.get("A")]
    B = [_f(p.get("b", p.get("B"))) for p in pairs if p.get("b") or p.get("B")]
    return A, B

def _fetch(mr_backend, query: str, window: Optional[List[float]], topk: int = 5) -> List[Span]:
    """MR 백엔드에서 스팬을 가져옵니다."""
    return mr_backend.fetch(query, window, topk, {}) or []

def _split_span_in_list(span_list: List[Span], target_span: Dict[str, Any], timestamps: List[float], list_name: str) -> Tuple[List[Span], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    스팬 리스트에서 특정 스팬을 찾아서 분할합니다.
    
    Returns:
        (new_span_list, added_spans, removed_spans)
    """
    new_list = span_list.copy()
    added_spans = []
    removed_spans = []
    
    for j, span in enumerate(new_list):
        if (abs(float(span["t0"]) - float(target_span["t0"])) < 0.1 and 
            abs(float(span["t1"]) - float(target_span["t1"])) < 0.1):
            # 기존 span 제거
            removed_span = new_list.pop(j)
            removed_spans.append({"list": list_name, "index": j, "span": removed_span})
            
            # timestamp로 새로운 span들 생성
            sorted_timestamps = sorted(timestamps)
            original_conf = float(span.get("conf", span.get("score", 1.0)))
            
            for k in range(len(sorted_timestamps) - 1):
                new_span = {
                    "t0": float(sorted_timestamps[k]),
                    "t1": float(sorted_timestamps[k + 1]),
                    "conf": original_conf,
                    "tag": span.get("tag", "") + f"_split_{k}"
                }
                new_list.append(new_span)
                added_spans.append({"span": new_span, "source": f"split_{list_name}"})
            break
            
    return new_list, added_spans, removed_spans

def _split_action_new_format(action: Dict[str, Any], A: List[Span], B: List[Span]) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    새로운 형식의 split 액션을 처리합니다.
    A와 B에 대해 서로 다른 split 방식을 개별적으로 지정할 수 있습니다.
    """
    split_specs = action.get("split_specs", [])
    
    if not split_specs:
        return A, B, {"error": "No split_specs provided", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    new_A, new_B = A.copy(), B.copy()
    
    added_spans = []
    removed_spans = []
    
    for spec in split_specs:
        target = spec.get("target", "").upper()
        span_to_split = spec.get("span", {})
        timestamps = spec.get("timestamps", [])
        
        if not timestamps or not span_to_split:
            continue
            
        if target == "A":
            new_A, added_a, removed_a = _split_span_in_list(new_A, span_to_split, timestamps, "A")
            added_spans.extend(added_a)
            removed_spans.extend(removed_a)
        elif target == "B":
            new_B, added_b, removed_b = _split_span_in_list(new_B, span_to_split, timestamps, "B")
            added_spans.extend(added_b)
            removed_spans.extend(removed_b)
    
    # 중복 제거
    new_A = _dedup_spans(new_A)
    new_B = _dedup_spans(new_B)
    
    action_changes = {
        "action_type": "split",
        "format": "new",
        "specs_processed": len(split_specs),
        "added_spans": added_spans,
        "removed_spans": removed_spans
    }
    
    return new_A, new_B, action_changes

def split_action(action: Dict[str, Any], A: List[Span], B: List[Span]) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    Split 액션 처리: A와 B 리스트의 특정 spans를 각각 다른 방식으로 분할
    
    Args:
        action: 다음 두 가지 형식 지원:
            1. 기존 형식: {"action": "split", "pairs": [...], "timestamps": [[t1, t2, ...], ...]}
            2. 새로운 형식: {"action": "split", "split_specs": [
                {"target": "A", "span": {...}, "timestamps": [t1, t2, ...]},
                {"target": "B", "span": {...}, "timestamps": [t1, t2, ...]},
                ...
            ]}
        A, B: 현재 span 리스트들
        
    Returns:
        (new_A, new_B, action_changes)
    """
    # 새로운 형식 지원
    if "split_specs" in action:
        return _split_action_new_format(action, A, B)
    
    # 기존 형식 호환성 유지
    pairs = action.get("pairs", [])
    timestamps_list = action.get("timestamps", [])
    
    if not pairs:
        return A, B, {"error": "No pairs provided for split action", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    new_A, new_B = A.copy(), B.copy()
    
    added_spans = []
    removed_spans = []
    
    for i, pair in enumerate(pairs):
        # 각 pair에 대한 timestamp 리스트 가져오기
        timestamps = timestamps_list[i] if i < len(timestamps_list) else []
        if not timestamps:
            continue
            
        # a와 b span 정보 추출
        a_span = pair.get("a", {})
        b_span = pair.get("b", {})
        
        # A 리스트에서 해당 span 찾기 및 분할
        if a_span:
            new_A, added_a, removed_a = _split_span_in_list(new_A, a_span, timestamps, "A")
            added_spans.extend(added_a)
            removed_spans.extend(removed_a)
        
        # B 리스트에서 해당 span 찾기 및 분할
        if b_span:
            new_B, added_b, removed_b = _split_span_in_list(new_B, b_span, timestamps, "B")
            added_spans.extend(added_b)
            removed_spans.extend(removed_b)
    
    # 중복 제거
    new_A = _dedup_spans(new_A)
    new_B = _dedup_spans(new_B)
    
    action_changes = {
        "action_type": "split",
        "pairs_processed": len(pairs),
        "timestamps_used": len(timestamps_list),
        "added_spans": added_spans,
        "removed_spans": removed_spans
    }
    
    return new_A, new_B, action_changes

def _concat_spans_in_list(span_list: List[Span], spans_to_concat: List[Dict[str, Any]], list_name: str) -> Tuple[List[Span], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    스팬 리스트에서 특정 스팬들을 찾아서 합칩니다.
    
    Returns:
        (new_span_list, added_spans, removed_spans)
    """
    new_list = span_list.copy()
    added_spans = []
    removed_spans = []
    
    matching_spans = []
    indices_to_remove = []
    
    # 합칠 스팬들을 찾기
    for span_to_find in spans_to_concat:
        for j, span in enumerate(new_list):
            if (abs(float(span["t0"]) - float(span_to_find["t0"])) < 0.1 and 
                abs(float(span["t1"]) - float(span_to_find["t1"])) < 0.1):
                matching_spans.append(span)
                indices_to_remove.append(j)
                removed_spans.append({"list": list_name, "index": j, "span": span})
                break
    
    if not matching_spans:
        return new_list, added_spans, removed_spans
    
    # 최소 t0와 최대 t1 찾기
    min_t0 = min(float(span["t0"]) for span in matching_spans)
    max_t1 = max(float(span["t1"]) for span in matching_spans)
    
    # 평균 confidence 계산
    avg_conf = sum(float(span.get("conf", span.get("score", 1.0))) for span in matching_spans) / len(matching_spans)
    
    # 새로운 합쳐진 span 생성
    concatenated_span = {
        "t0": min_t0,
        "t1": max_t1,
        "conf": avg_conf,
        "tag": "concatenated"
    }
    
    # 역순으로 제거 (인덱스 변경 방지)
    for idx in sorted(indices_to_remove, reverse=True):
        if idx < len(new_list):
            new_list.pop(idx)
    
    # 새로운 span 추가
    new_list.append(concatenated_span)
    added_spans.append({"span": concatenated_span, "source": f"concat_{list_name}"})
    
    return new_list, added_spans, removed_spans

def _concat_action_new_format(action: Dict[str, Any], A: List[Span], B: List[Span]) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    새로운 형식의 concat 액션을 처리합니다.
    A와 B에 대해 서로 다른 concat 방식을 개별적으로 지정할 수 있습니다.
    """
    concat_specs = action.get("concat_specs", [])
    
    if not concat_specs:
        return A, B, {"error": "No concat_specs provided", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    new_A, new_B = A.copy(), B.copy()
    
    added_spans = []
    removed_spans = []
    
    for spec in concat_specs:
        target = spec.get("target", "").upper()
        spans_to_concat = spec.get("spans", [])
        
        if not spans_to_concat:
            continue
            
        if target == "A":
            new_A, added_a, removed_a = _concat_spans_in_list(new_A, spans_to_concat, "A")
            added_spans.extend(added_a)
            removed_spans.extend(removed_a)
        elif target == "B":
            new_B, added_b, removed_b = _concat_spans_in_list(new_B, spans_to_concat, "B")
            added_spans.extend(added_b)
            removed_spans.extend(removed_b)
    
    # 중복 제거
    new_A = _dedup_spans(new_A)
    new_B = _dedup_spans(new_B)
    
    action_changes = {
        "action_type": "concat",
        "format": "new",
        "specs_processed": len(concat_specs),
        "added_spans": added_spans,
        "removed_spans": removed_spans
    }
    
    return new_A, new_B, action_changes

def concat_action(action: Dict[str, Any], A: List[Span], B: List[Span]) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    Concat 액션 처리: 여러 candidate들을 하나의 구간으로 합치고 기존 구간들 삭제
    
    Args:
        action: 다음 두 가지 형식 지원:
            1. 기존 형식: {"action": "concat", "pairs": [...], "target_list": "A" | "B"}
            2. 새로운 형식: {"action": "concat", "concat_specs": [
                {"target": "A", "spans": [{"t0": ..., "t1": ...}, ...]},
                {"target": "B", "spans": [{"t0": ..., "t1": ...}, ...]},
                ...
            ]}
        A, B: 현재 span 리스트들
        
    Returns:
        (new_A, new_B, action_changes)
    """
    # 새로운 형식 지원
    if "concat_specs" in action:
        return _concat_action_new_format(action, A, B)
    
    # 기존 형식 호환성 유지
    pairs = action.get("pairs", [])
    target_list = action.get("target_list", "A")  # 기본적으로 A 리스트에서 concat
    
    if not pairs:
        return A, B, {"error": "No pairs provided for concat action", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    new_A, new_B = A.copy(), B.copy()
    
    added_spans = []
    removed_spans = []
    
    # concat할 span들 수집
    spans_to_concat = []
    
    if target_list == "A":
        target_spans = new_A
        other_spans = new_B
        target_name = "A"
    else:
        target_spans = new_B
        other_spans = new_A
        target_name = "B"
    
    # pairs에서 해당하는 span들 찾기
    for pair in pairs:
        if target_list == "A":
            span_data = pair.get("a", {})
        else:
            span_data = pair.get("b", {})
            
        if not span_data:
            continue
            
        # 해당 span을 target_spans에서 찾기
        for j, span in enumerate(target_spans):
            if (abs(float(span["t0"]) - float(span_data["t0"])) < 0.1 and 
                abs(float(span["t1"]) - float(span_data["t1"])) < 0.1):
                spans_to_concat.append(span)
                break
    
    if not spans_to_concat:
        return A, B, {"error": "No matching spans found for concat", "added_spans": [], "removed_spans": []}
    
    # 최소 t0와 최대 t1 찾기
    min_t0 = min(float(span["t0"]) for span in spans_to_concat)
    max_t1 = max(float(span["t1"]) for span in spans_to_concat)
    
    # 평균 confidence 계산
    avg_conf = sum(float(span.get("conf", span.get("score", 1.0))) for span in spans_to_concat) / len(spans_to_concat)
    
    # 새로운 합쳐진 span 생성
    concatenated_span = {
        "t0": min_t0,
        "t1": max_t1,
        "conf": avg_conf,
        "tag": "concatenated"
    }
    
    # 기존 span들 제거
    indices_to_remove = []
    for span_to_remove in spans_to_concat:
        for j, span in enumerate(target_spans):
            if (abs(float(span["t0"]) - float(span_to_remove["t0"])) < 0.1 and 
                abs(float(span["t1"]) - float(span_to_remove["t1"])) < 0.1):
                indices_to_remove.append(j)
                removed_spans.append({"list": target_name, "index": j, "span": span})
                break
    
    # 역순으로 제거 (인덱스 변경 방지)
    for idx in sorted(indices_to_remove, reverse=True):
        if idx < len(target_spans):
            target_spans.pop(idx)
    
    # 새로운 span 추가
    target_spans.append(concatenated_span)
    added_spans.append({"span": concatenated_span, "source": f"concat_{target_name}"})
    
    # 결과 업데이트
    if target_list == "A":
        new_A = _dedup_spans(target_spans)
        new_B = other_spans
    else:
        new_A = other_spans
        new_B = _dedup_spans(target_spans)
    
    action_changes = {
        "action_type": "concat",
        "target_list": target_list,
        "pairs_processed": len(pairs),
        "spans_concatenated": len(spans_to_concat),
        "added_spans": added_spans,
        "removed_spans": removed_spans,
        "concatenated_span": concatenated_span
    }
    
    return new_A, new_B, action_changes

def remove_action(action: Dict[str, Any], A: List[Span], B: List[Span], vlm_backend=None) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    Remove 액션 처리: VLM을 사용해서 조건에 맞지 않는 span들 제거
    기존 코드를 여기로 이동
    """
    pairs = action.get("pairs", [])
    prompt = action.get("prompt", "")
    
    if not pairs or not prompt or not vlm_backend:
        return A, B, {"error": "Invalid remove action or VLM not available", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    
    print(f"[REMOVE] Checking {len(pairs)} pairs with prompt: '{prompt}'")
    verified_pairs = []
    verification_results = []
    
    for pi, pair in enumerate(pairs):
        a_span, b_span = pair.get("a", {}), pair.get("b", {})
        t0 = min(float(a_span.get("t0", 0)), float(b_span.get("t0", 0)))
        t1 = max(float(a_span.get("t1", 0)), float(b_span.get("t1", 0)))
        
        try:
            answer, confidence = vlm_backend.query(t0, t1, prompt)
            answer_text = str(answer).lower()
            print(f"[REMOVE] Pair {pi+1}: t0={t0:.1f}s-{t1:.1f}s, VLM answer: '{answer}' (conf: {confidence:.3f})")
            
            if "yes" in answer_text or answer is True:
                verified_pairs.append(pair)
                print(f"[REMOVE] ✅ Pair {pi+1} KEPT (VLM: yes)")
                verification_results.append({"pair_id": pi, "kept": True, "answer": answer, "confidence": confidence})
            else:
                print(f"[REMOVE] ❌ Pair {pi+1} REMOVED (VLM: no)")
                verification_results.append({"pair_id": pi, "kept": False, "answer": answer, "confidence": confidence})
        except Exception as e:
            print(f"[REMOVE] Error verifying pair {pi+1}: {e}. Keeping pair as fallback.")
            verified_pairs.append(pair)
    
    if verified_pairs:
        print(f"[REMOVE] {len(verified_pairs)}/{len(pairs)} pairs passed verification")
        # pairs_to_AB 함수 구현
        def _f(s): 
            return {"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(s.get("conf", 1.0))}
        new_A = [_f(p.get("a", p.get("A"))) for p in verified_pairs if p.get("a") or p.get("A")]
        new_B = [_f(p.get("b", p.get("B"))) for p in verified_pairs if p.get("b") or p.get("B")]
        new_A, new_B = _dedup_spans(new_A), _dedup_spans(new_B)
        
        # 변경사항 계산
        removed_from_A = [s for s in A_before if s not in new_A]
        removed_from_B = [s for s in B_before if s not in new_B]
        added_to_A = [s for s in new_A if s not in A_before]
        added_to_B = [s for s in new_B if s not in B_before]
        
        action_changes = {
            "action_type": "remove",
            "prompt": prompt,
            "pairs_checked": len(pairs),
            "pairs_kept": len(verified_pairs),
            "verification_results": verification_results,
            "added_spans": [{"span": s, "source": "remove_A"} for s in added_to_A] + [{"span": s, "source": "remove_B"} for s in added_to_B],
            "removed_spans": [{"list": "A", "span": s, "reason": "remove_filter"} for s in removed_from_A] + [{"list": "B", "span": s, "reason": "remove_filter"} for s in removed_from_B]
        }
    else:
        print(f"[REMOVE] No pairs passed verification. Keeping original spans.")
        new_A, new_B = A, B
        action_changes = {
            "action_type": "remove",
            "prompt": prompt,
            "pairs_checked": len(pairs),
            "pairs_kept": 0,
            "verification_results": verification_results,
            "added_spans": [],
            "removed_spans": []
        }
    
    return new_A, new_B, action_changes

def probe_action(action: Dict[str, Any], A: List[Span], B: List[Span], 
                current_issue: Dict[str, Any], mr_backend) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    Probe 액션 처리: 새로운 후보들을 찾거나 더 나은 후보들을 찾습니다.
    
    Args:
        action: probe 액션 딕셔너리
        A, B: 현재 span 리스트들
        current_issue: 현재 해결해야 할 이슈
        mr_backend: moment retrieval 백엔드
        
    Returns:
        (new_A, new_B, action_changes)
    """
    target_list = action.get("target_list")
    q = action.get("query")
    windows = action.get("windows")
    
    if target_list not in ["A", "B"] or not q or not windows:
        return A, B, {"error": "Invalid probe action", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업 (변경사항 추적용)
    A_before, B_before = A.copy(), B.copy()
    
    added: List[Span] = []
    for w in windows:
        new_spans = _fetch(mr_backend, q, w, 5)
        added.extend(new_spans)
        print(f"[PROBE] Added {len(new_spans)} spans from {q} in window {w}")
    
    issue_type = current_issue.get("type", "")
    ia = current_issue.get("ia")
    ib = current_issue.get("ib")
    
    # Overlap 이슈인 경우 기존 span 삭제
    removed_spans = []
    print(f"[PROBE] Issue type: {issue_type}, ia={ia}, ib={ib}")
    print(f"[PROBE] Before removal - A: {len(A)} spans, B: {len(B)} spans")
    
    if "Overlap" in issue_type:
        if ia is not None and 0 <= ia < len(A):
            removed_span = A[ia].copy()
            removed_spans.append(("A", ia, removed_span))
            A = [s for i, s in enumerate(A) if i != ia]
            print(f"[PROBE] Removed A[{ia}]: {removed_span}")
        if ib is not None and 0 <= ib < len(B):
            removed_span = B[ib].copy()
            removed_spans.append(("B", ib, removed_span))
            B = [s for i, s in enumerate(B) if i != ib]
            print(f"[PROBE] Removed B[{ib}]: {removed_span}")
        print(f"[PROBE] After removal - A: {len(A)} spans, B: {len(B)} spans")
    
    # 새로운 span들 추가
    if target_list == "A":
        A = _dedup_spans(A + added)
        added_after_dedup = [s for s in A if s not in A_before]
    else:
        B = _dedup_spans(B + added)
        added_after_dedup = [s for s in B if s not in B_before]
    
    # 변경사항 기록
    action_changes = {
        "action_type": "probe",
        "target_list": target_list,
        "query": q,
        "windows": windows,
        "added_spans": [{"span": s, "source": "probe"} for s in added_after_dedup],
        "removed_spans": [{"list": lst, "index": idx, "span": span} for lst, idx, span in removed_spans]
    }
    
    print(f"[PROBE] Final result - A: {len(A)} spans, B: {len(B)} spans")
    print(f"[PROBE] Final A: {[{'t0': s['t0'], 't1': s['t1'], 'conf': _conf(s)} for s in A]}")
    print(f"[PROBE] Final B: {[{'t0': s['t0'], 't1': s['t1'], 'conf': _conf(s)} for s in B]}")
    
    return A, B, action_changes

def verify_action(action: Dict[str, Any], A: List[Span], B: List[Span], 
                 vlm_backend) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    Verify 액션 처리: VLM을 사용해서 candidate pairs를 검증합니다.
    
    Args:
        action: verify 액션 딕셔너리
        A, B: 현재 span 리스트들
        vlm_backend: VLM 백엔드
        
    Returns:
        (new_A, new_B, action_changes)
    """
    pairs = action.get("pairs", [])
    prompt = action.get("prompt", "")
    
    if not pairs or not prompt or not vlm_backend:
        return A, B, {"error": "Invalid verify action or VLM not available", "added_spans": [], "removed_spans": []}
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    
    print(f"[VERIFY] Checking {len(pairs)} pairs with prompt: '{prompt}'")
    verified_pairs = []
    verification_results = []
    
    for pi, pair in enumerate(pairs):
        a_span, b_span = pair.get("a", {}), pair.get("b", {})
        t0 = min(float(a_span.get("t0", 0)), float(b_span.get("t0", 0)))
        t1 = max(float(a_span.get("t1", 0)), float(b_span.get("t1", 0)))
        
        try:
            answer, confidence = vlm_backend.query(t0, t1, prompt)
            answer_text = str(answer).lower()
            print(f"[VERIFY] Pair {pi+1}: t0={t0:.1f}s-{t1:.1f}s, VLM answer: '{answer}' (conf: {confidence:.3f})")
            
            if "yes" in answer_text or answer is True:
                verified_pairs.append(pair)
                print(f"[VERIFY] ✅ Pair {pi+1} KEPT (VLM: yes)")
                verification_results.append({"pair_id": pi, "kept": True, "answer": answer, "confidence": confidence})
            else:
                print(f"[VERIFY] ❌ Pair {pi+1} REMOVED (VLM: no)")
                verification_results.append({"pair_id": pi, "kept": False, "answer": answer, "confidence": confidence})
        except Exception as e:
            print(f"[VERIFY] Error verifying pair {pi+1}: {e}. Keeping pair as fallback.")
            verified_pairs.append(pair)
    
    if verified_pairs:
        print(f"[VERIFY] {len(verified_pairs)}/{len(pairs)} pairs passed verification")
        A, B = _pairs_to_AB(verified_pairs)
        A, B = _dedup_spans(A), _dedup_spans(B)
        
        # 변경사항 계산
        removed_from_A = [s for s in A_before if s not in A]
        removed_from_B = [s for s in B_before if s not in B]
        added_to_A = [s for s in A if s not in A_before]
        added_to_B = [s for s in B if s not in B_before]
        
        action_changes = {
            "action_type": "verify",
            "prompt": prompt,
            "pairs_checked": len(pairs),
            "pairs_kept": len(verified_pairs),
            "verification_results": verification_results,
            "added_spans": [{"span": s, "source": "verify_A"} for s in added_to_A] + [{"span": s, "source": "verify_B"} for s in added_to_B],
            "removed_spans": [{"list": "A", "span": s, "reason": "verify_filter"} for s in removed_from_A] + [{"list": "B", "span": s, "reason": "verify_filter"} for s in removed_from_B]
        }
    else:
        print(f"[VERIFY] No pairs passed verification. Keeping original spans.")
        action_changes = {
            "action_type": "verify",
            "prompt": prompt,
            "pairs_checked": len(pairs),
            "pairs_kept": 0,
            "verification_results": verification_results,
            "added_spans": [],
            "removed_spans": []
        }
    
    return A, B, action_changes

def stop_action(action: Dict[str, Any], A: List[Span], B: List[Span]) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    Stop 액션 처리: LLM이 제안한 해결책을 적용합니다.
    
    Args:
        action: stop 액션 딕셔너리
        A, B: 현재 span 리스트들
        
    Returns:
        (new_A, new_B, action_changes)
    """
    proposed_pairs = action.get("pairs", [])
    
    if not proposed_pairs:
        return A, B, {
            "action_type": "stop",
            "proposed_pairs": [],
            "added_spans": [],
            "removed_spans": []
        }
    
    # 현재 상태 백업
    A_before, B_before = A.copy(), B.copy()
    
    print(f"[STOP] LLM proposed a solution. Applying and re-evaluating.")
    A_new, B_new = _pairs_to_AB(proposed_pairs)
    A, B = _dedup_spans(A_new), _dedup_spans(B_new)
    
    # 변경사항 계산
    removed_from_A = [s for s in A_before if s not in A]
    removed_from_B = [s for s in B_before if s not in B]
    added_to_A = [s for s in A if s not in A_before]
    added_to_B = [s for s in B if s not in B_before]
    
    action_changes = {
        "action_type": "stop",
        "proposed_pairs": proposed_pairs,
        "added_spans": [{"span": s, "source": "stop_solution_A"} for s in added_to_A] + [{"span": s, "source": "stop_solution_B"} for s in added_to_B],
        "removed_spans": [{"list": "A", "span": s, "reason": "stop_solution_replace"} for s in removed_from_A] + [{"list": "B", "span": s, "reason": "stop_solution_replace"} for s in removed_from_B]
    }
    
    return A, B, action_changes

def process_action(action: Dict[str, Any], A: List[Span], B: List[Span], **kwargs) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    액션을 처리하는 메인 함수 (모든 액션 타입 지원)
    
    Args:
        action: 처리할 액션 딕셔너리
        A, B: 현재 span 리스트들
        **kwargs: 추가 인자들 (vlm_backend, current_issue, mr_backend 등)
        
    Returns:
        (new_A, new_B, action_changes)
    """
    action_type = action.get("action", "")
    
    # 새로운 액션들
    if action_type == "split":
        return split_action(action, A, B)
    elif action_type == "concat":
        return concat_action(action, A, B)
    elif action_type == "remove":
        vlm_backend = kwargs.get("vlm_backend")
        return remove_action(action, A, B, vlm_backend)
    
    # 기존 액션들
    elif action_type == "probe":
        current_issue = kwargs.get("current_issue", {})
        mr_backend = kwargs.get("mr_backend")
        return probe_action(action, A, B, current_issue, mr_backend)
    elif action_type == "verify":
        vlm_backend = kwargs.get("vlm_backend")
        return verify_action(action, A, B, vlm_backend)
    elif action_type == "stop" and action.get("status") == "SUCCESS":
        return stop_action(action, A, B)
    
    # 알 수 없는 액션
    else:
        return A, B, {
            "error": f"Unknown action type: {action_type}",
            "added_spans": [],
            "removed_spans": []
        }

# 하위 호환성을 위한 별명
def process_legacy_action(action: Dict[str, Any], A: List[Span], B: List[Span], 
                         current_issue: Dict[str, Any], mr_backend, vlm_backend) -> Tuple[List[Span], List[Span], Dict[str, Any]]:
    """
    기존 액션들을 처리하는 함수 (하위 호환성 유지)
    """
    return process_action(action, A, B, 
                         current_issue=current_issue, 
                         mr_backend=mr_backend, 
                         vlm_backend=vlm_backend)

# inspector.py
# ------------------------------------------------------------
# A(ANCHOR) BEFORE/AFTER B 전용 “간단 검사기”
# ... (이전 주석은 동일) ...
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import statistics
import json

Span = Dict[str, Any]

EPS_NEAR = 0.5           # 초단위 근접(threshold)
OVERLONG_FACTOR = 2.5    # 중앙길이 × 계수

# ---------- 이슈 유형별 권장 조치 목록 ----------
ISSUE_ACTIONS = {
    "EmptyA": [
        "Review action history to identify potential time ranges where A events might occur. Divide the video into shorter segments and examine each systematically.",
    ],
    "EmptyB": [
        "Review action history to identify potential time ranges where B events might occur. Divide the video into shorter segments and examine each systematically.",
    ],
    "NoCandidates": [
        """Short A or B events may be embedded within longer segments from other queries. Expand existing candidate segments slightly and re-search both queries.
        Example: Expand candidate [10, 20] to [9, 21] and probe both query_a and query_b. Adjust expansion margins based on video duration and adjacent segments.""",
        "Check areas before and after each existing candidate to find corresponding A or B events."
    ],
    "OverlongSpan": [
        "Split overly long spans that incorrectly merge multiple distinct events into smaller, precise segments.",
    ],
    "OverlapWithAnchor": [
        "For containment cases (A contains B or vice versa): Probe the longer segment's beginning and end regions separately.",
        "For partial overlaps: Probe the combined region covering both segments, searching each query independently.",
        """For persistent overlaps after re-probing: Events may be truly simultaneous. Split into three segments: non-overlapping A, overlap region, and non-overlapping B.
        Example: A=[10,20], B=[15,25] → Split into [10,15], [15,20], [20,25].""",
        "For very short segments: Use verification instead of further splitting."
    ],
    "OverlapWithB": [
        "For containment cases (A contains B or vice versa): Probe the longer segment's beginning and end regions separately.",
        "For partial overlaps: Probe the combined region covering both segments, searching each query independently.", 
        """For persistent overlaps after re-probing: Events may be truly simultaneous. Split into three segments: non-overlapping A, overlap region, and non-overlapping B.
        Example: A=[10,20], B=[15,25] → Split into [10,15], [15,20], [20,25].""",
        "For very short segments: Use verification instead of further splitting."
    ]
}


# ---------- 유틸 ----------

def _conf(s: Span) -> float:
    return float(s.get("conf", s.get("score", 0.0)))

def _dur(s: Span) -> float:
    return max(0.0, float(s["t1"]) - float(s["t0"]))

def _overlap_len(a: Span, b: Span) -> float:
    lo = max(a["t0"], b["t0"]); hi = min(a["t1"], b["t1"])
    return max(0.0, hi - lo)

def _median_len(spans: List[Span]) -> float:
    if not spans: return 0.0
    return statistics.median([_dur(s) for s in spans])

def _summ(s: Span) -> Dict[str, Any]:
    return {
        "t0": round(float(s["t0"]), 3),
        "t1": round(float(s["t1"]), 3),
        "dur": round(_dur(s), 3),
        "conf": round(_conf(s), 4),
        "tag": s.get("tag", "")
    }

# ---------- 이슈 생성 헬퍼 ----------
def _create_issue(issue_type: str, **kwargs) -> Dict[str, Any]:
    """Creates an issue dictionary and automatically adds actionable advice."""
    issue = {"type": issue_type}
    issue.update(kwargs)
    issue["actions"] = ISSUE_ACTIONS.get(issue_type, ["No specific action defined for this issue."])
    return issue


# ---------- 요약 텍스트 생성 (타임스탬프 및 조치 포함하도록 수정) ----------
def _generate_summary_text(
    issues: List[Dict[str, Any]],
    A: List[Span],
    B: List[Span],
    candidates: List[Dict[str, Any]],
    emit: str
) -> str:
    """Generates a human-readable English summary with timestamps and actions in issue messages."""
    if not issues and not candidates:
        return "Inspection complete. No valid candidates and no issues were found."
    if not issues:
        return f"Inspection complete. Found {len(candidates)} valid candidates with no issues."

    summary_lines = [
        f"Inspection complete. Found {len(candidates)} valid candidates with {len(issues)} issues detected:"
    ]

    # 1. 각 이슈에 대한 상세 설명 (타임스탬프와 조치 포함)
    for issue in issues:
        issue_type = issue["type"]
        line = f"\n- [{issue_type}] "
        
        if issue_type == "EmptyA":
            line += "The anchor list 'A' is empty. Cannot perform inspection."
        elif issue_type == "EmptyB":
            line += "The target list 'B' is empty. No candidates to evaluate."
        elif issue_type == "NoCandidates":
            total_pairs = issue.get('total_pairs', 0)
            line += (f"No valid candidates found that satisfy the {issue.get('operator', 'temporal')} constraint. "
                     f"Out of {len(A)} anchor spans and {len(B)} target spans, "
                     f"{total_pairs} potential pairs were examined but none met the temporal requirements.")
        
        elif issue_type == "OverlongSpan":
            side = issue['side']
            idx = issue['idx']
            span = A[idx] if side == 'A' else B[idx]
            tag = span.get('tag', f'#{idx}')
            ts_str = f"(t: {span['t0']:.2f} to {span['t1']:.2f})"
            line += (f"Span {side}['{tag}'] {ts_str} has a duration of {issue['dur']:.2f}s, "
                     f"which is considered overlong (threshold based on median of {issue['median']:.2f}s).")
        
        elif issue_type in {"OverlapWithAnchor", "OverlapWithB"}:
            ia, ib = issue['ia'], issue['ib']
            span_a, span_b = A[ia], B[ib]
            tag_a = span_a.get('tag', f'#{ia}')
            tag_b = span_b.get('tag', f'#{ib}')
            ts_a_str = f"(t: {span_a['t0']:.2f} to {span_a['t1']:.2f})"
            ts_b_str = f"(t: {span_b['t0']:.2f} to {span_b['t1']:.2f})"
            
            line += (f"Anchor A['{tag_a}'] {ts_a_str} and span B['{tag_b}'] {ts_b_str} overlap by "
                         f"{issue['overlap']:.2f}s, violating the order constraint.")

        issue["text"] = line.strip()
        summary_lines.append(line.strip())

        # 권장 조치 추가
        if issue.get("actions"):
            summary_lines.append("  Recommended Actions:")
            for action in issue["actions"]:
                summary_lines.append(f"    - {action}")
        
    # 2. 후보 목록에 포함되었지만 이슈가 있는 스팬에 대한 경고 (타임스탬프 포함)
    candidate_b_indices = set()
    if emit in {'right', 'pair'} and candidates:
        candidate_b_indices = {c['b_idx'] for c in candidates}

    problematic_b_spans = {}  # key: b_idx, value: list of issue types
    for issue in issues:
        idx = -1
        if issue.get('side') == 'B':
            idx = issue['idx']
        elif 'ib' in issue:
            idx = issue['ib']
        if idx != -1:
            problematic_b_spans.setdefault(idx, []).append(issue['type'])

    conflicting_indices = candidate_b_indices.intersection(problematic_b_spans.keys())

    if conflicting_indices:
        summary_lines.append("\n--- Warnings for Spans in the Candidate List ---")
        summary_lines.append("The following spans are valid candidates but also have associated issues:")
        for b_idx in sorted(list(conflicting_indices)):
            span = B[b_idx]
            tag = span.get('tag', f'#{b_idx}')
            ts_str = f"(t: {span['t0']:.2f} to {span['t1']:.2f})"
            issue_types = ", ".join(sorted(list(set(problematic_b_spans[b_idx]))))
            summary_lines.append(f"- Candidate B['{tag}'] {ts_str} has the following warnings: {issue_types}.")
        
    return "\n".join(summary_lines)


# ---------- 핵심 검사기 ----------

def inspect_order(
    A: List[Span],
    B: List[Span],
    *,
    operator: str = "BEFORE",
    emit: str = "right"
) -> Dict[str, Any]:
    operator = operator.upper()
    assert operator in {"BEFORE", "AFTER"}
    assert emit in {"left", "right", "pair", "union"}

    issues: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []

    if not A:
        issues.append(_create_issue("EmptyA"))
    if not B:
        issues.append(_create_issue("EmptyB"))
    if not A or not B:
        summary_text = _generate_summary_text(issues, A, B, [], emit)
        return {
            "operator": operator, "anchor": "A", "emit": emit, "issues": issues,
            "candidates": [], "pairs": [], "summary_text": summary_text
        }

    medA = _median_len(A) or 0.0
    medB = _median_len(B) or 0.0
    thrA = medA * OVERLONG_FACTOR
    thrB = medB * OVERLONG_FACTOR
    for i, a in enumerate(A):
        dur_a = _dur(a)
        if medA > 0.0 and dur_a > thrA and dur_a > 15:
            issues.append(_create_issue("OverlongSpan", side="A", idx=i, dur=round(dur_a,3), median=round(medA,3)))
    for j, b in enumerate(B):
        dur_b = _dur(b)
        if medB > 0.0 and dur_b > thrB and dur_b > 15:
            issues.append(_create_issue("OverlongSpan", side="B", idx=j, dur=round(dur_b,3), median=round(medB,3)))

    A_sorted = sorted(list(enumerate(A)), key=lambda t: (t[1]["t0"], t[1]["t1"]))
    B_sorted = sorted(list(enumerate(B)), key=lambda t: (t[1]["t0"], t[1]["t1"]))

    for ia, a in A_sorted:
        a0, a1 = float(a["t0"]), float(a["t1"])
        for ib, b in B_sorted:
            b0, b1 = float(b["t0"]), float(b["t1"])
            overlap = _overlap_len(a, b)
            
            if operator == "BEFORE":
                if b1 <= a0 - EPS_NEAR: continue
                if b0 < a1 and overlap > 0:
                    issues.append(_create_issue("OverlapWithAnchor", ia=ia, ib=ib, overlap=round(overlap,3))); continue
                if b0 > a1 + EPS_NEAR:
                    pairs.append({"a_idx": ia, "b_idx": ib, "a": _summ(a), "b": _summ(b), "score": round(_conf(b), 6)})
            else:  # AFTER
                if overlap > 0:
                    issues.append(_create_issue("OverlapWithB", ia=ia, ib=ib, overlap=round(overlap,3))); continue
                if a0 > b1 + EPS_NEAR:
                    pairs.append({"a_idx": ia, "b_idx": ib, "a": _summ(a), "b": _summ(b), "score": round(_conf(b), 6)})
    
    candidates: List[Dict[str, Any]] = []
    if emit == "pair":
        candidates = sorted(pairs, key=lambda x: x["score"], reverse=True)

    elif emit == "right":
        best_by_b: Dict[int, Dict[str, Any]] = {}
        for pr in pairs:
            ib = pr["b_idx"]
            if (ib not in best_by_b) or (pr["score"] > best_by_b[ib]["score"]):
                best_by_b[ib] = pr
        candidates = [
            {"b_idx": ib, "b": v["b"], "score": v["score"], "from_anchor": v["a_idx"]}
            for ib, v in best_by_b.items()
        ]
        candidates.sort(key=lambda x: x["score"], reverse=True)

    elif emit == "union":
        candidates = []
        # A쪽 후보
        seen_a = set()
        for pr in pairs:
            ia = pr["a_idx"]
            if ia in seen_a: 
                continue
            seen_a.add(ia)
            candidates.append({
                "a_idx": ia,
                "a": _summ(A[ia]),
                "score": pr["score"]
            })
        # B쪽 후보
        seen_b = set()
        for pr in pairs:
            ib = pr["b_idx"]
            if ib in seen_b: 
                continue
            seen_b.add(ib)
            candidates.append({
                "b_idx": ib,
                "b": _summ(B[ib]),
                "score": pr["score"]
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)

    else:  # emit == "left"
        best_by_a: Dict[int, float] = {}
        for pr in pairs:
            ia = pr["a_idx"]
            best_by_a[ia] = max(best_by_a.get(ia, 0.0), pr["score"])
        for ia, a in A_sorted:
            if ia in best_by_a:
                candidates.append({"a_idx": ia, "a": _summ(a), "score": round(best_by_a[ia], 6)})
        candidates.sort(key=lambda x: x["score"], reverse=True)

    # 후보가 없을 때 이슈 추가
    if not candidates and A and B:
        total_potential_pairs = len(A) * len(B)
        issues.append(_create_issue(
            "NoCandidates", 
            operator=operator,
            total_pairs=total_potential_pairs,
            a_count=len(A),
            b_count=len(B),
            pairs_examined=len(pairs)
        ))


    summary_text = _generate_summary_text(issues, A, B, candidates, emit)
    
    return {
        "operator": operator, "anchor": "A", "emit": emit, "issues": issues,
        "candidates": candidates, "pairs": pairs, "summary_text": summary_text
    }


# ------------------ 간단 테스트 ------------------
if __name__ == "__main__":
    print("=== Test Case 1: Normal case with some issues ===")
    A = [
        {"t0": 10.0, "t1": 20.0, "conf": 0.9, "tag": "A0"},
        {"t0": 40.0, "t1": 60.0, "conf": 0.8, "tag": "A1"},
    ]
    B = [
        {"t0":  5.0, "t1":  9.2, "conf": 0.6, "tag": "B0_before_A0_near"},
        {"t0": 19.2, "t1": 25.0, "conf": 0.7, "tag": "B1_meet/overlap"}, # OverlapWithAnchor 발생
        {"t0": 22.0, "t1": 30.0, "conf": 0.95, "tag": "B2_valid_after_A0"},
        {"t0": 65.0, "t1":120.0, "conf": 0.4, "tag": "B3_overlong"},      # OverlongSpan 발생
    ]

    out = inspect_order(A, B, operator="AFTER", emit="right")
    print("Summary:", out["summary_text"])
    print("Candidates found:", len(out["candidates"]))
    
    print("\n=== Test Case 2: No candidates case ===")
    # A가 B보다 모두 늦게 시작하는 경우 (BEFORE 연산자에서 후보 없음)
    A_late = [
        {"t0": 50.0, "t1": 60.0, "conf": 0.9, "tag": "A0_late"},
        {"t0": 70.0, "t1": 80.0, "conf": 0.8, "tag": "A1_late"},
    ]
    B_early = [
        {"t0": 10.0, "t1": 20.0, "conf": 0.6, "tag": "B0_early"},
        {"t0": 25.0, "t1": 35.0, "conf": 0.7, "tag": "B1_early"},
    ]

    out_no_candidates = inspect_order(A_late, B_early, operator="BEFORE", emit="right")
    print("Summary:", out_no_candidates["summary_text"])
    print("Candidates found:", len(out_no_candidates["candidates"]))
    print("Issues:", [issue["type"] for issue in out_no_candidates["issues"]])

    print("\n--- Full JSON Output (Issues only) ---")
    print(json.dumps(out_no_candidates["issues"], indent=2))
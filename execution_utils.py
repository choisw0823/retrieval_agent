#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
execution_utils.py - 실행 관련 유틸리티 함수들

시각화, span 처리, 프롬프트 생성 등의 유틸리티 함수들을 제공합니다.
"""

from typing import Dict, List, Any, Optional
import json
import re
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Span = Dict[str, Any]

def _conf(s: Span) -> float:
    """Span의 confidence 값을 추출합니다."""
    return float(s.get("conf", s.get("score", 0.0)))

def _fetch(mr_backend, query: str, window: Optional[List[float]], topk: int = 5) -> List[Span]:
    """MR 백엔드에서 스팬을 가져옵니다."""
    return mr_backend.fetch(query, window, topk, {}) or []

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

def _summarize_spans(spans: List[Span], k: int = 100) -> Dict[str, Any]:
    """Span 리스트를 요약합니다."""
    if not spans:
        return {"count": 0, "samples": []}
    top = sorted(spans, key=_conf, reverse=True)[:k]
    return {
        "count": len(spans), 
        "samples": [{"t0": s["t0"], "t1": s["t1"], "conf": _conf(s), "tag": s.get("tag", "")} for s in top]
    }

def _summ(s: Span) -> Dict[str, Any]:
    """Span을 요약 형태로 변환"""
    return {"t0": s["t0"], "t1": s["t1"], "conf": _conf(s)}

def _emit_spans_from_pairs(pairs: List[Dict[str, Any]], emit: str) -> List[Span]:
    """Pairs에서 span들을 추출합니다."""
    out = []
    if emit == "pair":
        for pr in pairs:
            a, b = pr["a"], pr["b"]
            t0, t1 = min(float(a["t0"]), float(b["t0"])), max(float(a["t1"]), float(b["t1"]))
            out.append({"t0": t0, "t1": t1, "conf": float(pr.get("score", b.get("conf", 1.0)))})
    elif emit == "union":
        seen = set()
        for pr in pairs:
            for key in ("a", "b"):
                s = pr[key]
                k = (round(float(s["t0"]), 3), round(float(s["t1"]), 3), key)
                if k in seen:
                    continue
                seen.add(k)
                out.append({"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(s.get("conf", 1.0))})
    else:
        target_key = "b" if emit == "right" else "a"
        seen = set()
        for pr in pairs:
            s = pr[target_key]
            k = (round(float(s["t0"]), 3), round(float(s["t1"]), 3))
            if k in seen:
                continue
            seen.add(k)
            out.append({"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(pr.get("score", s.get("conf", 1.0)))})
    return out

def _compute_pairs_basic(A: List[Span], B: List[Span], operator: str) -> List[Dict[str, Any]]:
    """기본적인 temporal relation pairs를 계산합니다."""
    pairs = []
    A_s = sorted(A, key=lambda s: s["t0"])
    B_s = sorted(B, key=lambda s: s["t0"])
    
    for a in A_s:
        for b in B_s:
            valid = False
            EPS_NEAR = 1.0
            if operator == "BEFORE" and b["t0"] > a["t1"] + EPS_NEAR:
                valid = True
            elif operator == "AFTER" and a["t0"] > b["t1"] + EPS_NEAR:
                valid = True
            if valid:
                pairs.append({"a": _summ(a), "b": _summ(b), "score": _conf(b)})
    
    return sorted(pairs, key=lambda x: x["score"], reverse=True)

def _ffprobe_duration(path: str) -> Optional[float]:
    """FFprobe를 사용해서 영상 길이를 가져옵니다."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path
        ], stderr=subprocess.STDOUT).decode("utf-8", "ignore")
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None

def visualize_spans(A: List[Span], B: List[Span], a_query: str, b_query: str, 
                   title: str, video_end: Optional[float] = None, save_path: Optional[str] = None):
    """Span들을 시각화합니다."""
    print(f"[VIZ] Visualizing {len(A)} A spans and {len(B)} B spans for: {title}")
    print(f"[VIZ] A spans: {[{'t0': s['t0'], 't1': s['t1'], 'conf': _conf(s)} for s in A]}")
    print(f"[VIZ] B spans: {[{'t0': s['t0'], 't1': s['t1'], 'conf': _conf(s)} for s in B]}")
    
    if not A and not B:
        print(f"[VIZ] No spans to visualize for: {title}")
        return
    
    plt.figure(figsize=(14, 6))
    y_positions = {"A": 1, "B": 0}
    y_labels = [f"{b_query[:30]}...", f"{a_query[:30]}..."]
    colors = {"A": "#90EE90", "B": "#FFB6C1"}
    
    for span in A:
        t0, t1, conf = float(span["t0"]), float(span["t1"]), _conf(span)
        rect = patches.Rectangle((t0, y_positions["A"] - 0.3), t1 - t0, 0.6,
                               linewidth=1, edgecolor='darkgreen', facecolor=colors["A"], alpha=0.7)
        plt.gca().add_patch(rect)
        plt.text(t0 + (t1 - t0) / 2, y_positions["A"], f'{conf:.4f}',
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    for span in B:
        t0, t1, conf = float(span["t0"]), float(span["t1"]), _conf(span)
        rect = patches.Rectangle((t0, y_positions["B"] - 0.3), t1 - t0, 0.6,
                               linewidth=1, edgecolor='darkred', facecolor=colors["B"], alpha=0.7)
        plt.gca().add_patch(rect)
        plt.text(t0 + (t1 - t0) / 2, y_positions["B"], f'{conf:.4f}',
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1], y_labels)
    
    all_spans = A + B
    if all_spans:
        min_t = min(float(s["t0"]) for s in all_spans)
        max_t = max(float(s["t1"]) for s in all_spans)
        margin = (max_t - min_t) * 0.1
        plt.xlim(max(0, min_t - margin), max_t + margin)
    
    if video_end:
        plt.axvline(x=video_end, color='red', linestyle='--', alpha=0.5, label=f'Video End ({video_end:.1f}s)')
        plt.legend()
    
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    legend_elements = [
        patches.Patch(color=colors["A"], label=f'A: {a_query[:20]}...'),
        patches.Patch(color=colors["B"], label=f'B: {b_query[:20]}...')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved visualization: {save_path}")
    else:
        plt.show()
    plt.close()

def _get_priority_issue(issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """우선순위에 따라 이슈를 선택합니다."""
    if not issues:
        return None
    priority_order = ["EmptyA", "EmptyB", "NoCandidates", "OverlapWithAnchor", "OverlapWithB", "OverlongSpan"]
    for p_type in priority_order:
        for issue in issues:
            if issue.get("type") == p_type:
                return issue
    return issues[0]

# 프롬프트 관련 상수
COT_PROFESSIONAL_SYSTEM_PROMPT = """## Core Mission
You are an expert AI agent acting as a Next-Step Planner for resolving complex **temporal queries** in video analysis. Your mission is to meticulously analyze the current problem state and history, reason through a logical solution, and then formulate a precise, machine-executable JSON command to advance the problem-solving process.

## Output Format (Chain-of-Thought)
You MUST structure your response in two distinct parts:

1.  **Analysis**:
    In this section, you will think step-by-step. First, summarize the `CURRENT ISSUE` in your own words. Second, review the `HISTORY` to understand what has already been tried. Third, evaluate the `CURRENT RETRIEVAL RESULTS`. Finally, explain the rationale behind the action you are about to propose and why it is the most logical next step.

2.  **Proposed Action**:
    In this section, provide ONLY the final, executable JSON list. The JSON must be enclosed in a markdown code block (```json).

## Action Schemas
The JSON in your "Proposed Action" section must strictly adhere to one of the following schemas:

1.  **`probe`**: To find new or better candidates.
    ```json
    {"action": "probe", "target_list": "A" | "B", "query": "<search_query_text>", "windows": [[<t0_float>, <t1_float>], ...], "topk": <integer>}
    ```

2.  **`remove`**: To filter candidate pairs using a VLM based on a yes/no question. If the query is not in the video span, span will be removed.
    ```json
    {"action": "remove", "pairs": [{"a":{...}, "b":{...}}, ...], "prompt": "Is <description_of_event> present in this video segment? Answer only yes or no."}
    ```

3.  **`split`**: Split video segments into multiple parts using timestamps. Two formats supported:
    
    **Format 1 (Basic)**: Same timestamps for both A and B spans in each pair
    ```json
    {"action": "split", "pairs": [{"a":{...}, "b":{...}}, ...], "timestamps": [[t1, t2, t3, ...], [t4, t5, t6, ...], ...]}
    ```
    
    **Format 2 (Advanced)**: Individual control for A and B spans with different timestamps
    ```json
    {"action": "split", "split_specs": [
        {"target": "A", "span": {"t0": 10.0, "t1": 20.0, "conf": 0.8}, "timestamps": [10.0, 13.0, 16.0, 20.0]},
        {"target": "B", "span": {"t0": 25.0, "t1": 35.0, "conf": 0.9}, "timestamps": [25.0, 30.0, 35.0]},
        {"target": "A", "span": {"t0": 40.0, "t1": 50.0, "conf": 0.7}, "timestamps": [40.0, 42.0, 45.0, 48.0, 50.0]}
    ]}
    ```
    **Usage**: Use Format 2 when you need different split points for A vs B spans, or when splitting multiple spans independently.

4.  **`concat`**: Merge multiple video segments into one and remove originals. Two formats supported:
    
    **Format 1 (Target-based)**: Specify which list (A or B) to merge spans in
    ```json
    {"action": "concat", "pairs": [{"a":{...}, "b":{...}}, ...], "target_list": "A"}
    ```
    
    **Format 2 (Advanced)**: Individual control for multiple concatenations
    ```json
    {"action": "concat", "concat_specs": [
        {"target": "A", "spans": [{"t0": 10.0, "t1": 15.0}, {"t0": 16.0, "t1": 20.0}]},
        {"target": "B", "spans": [{"t0": 30.0, "t1": 35.0}, {"t0": 37.0, "t1": 42.0}]}
    ]}
    ```
    **Usage**: Use Format 2 for complex merging scenarios where you need precise control over which spans to merge in each list.

## Guiding Principles
- **Logic over Instinct**: Your analysis must clearly justify your proposed action.
- **Query Strategy**: When using `probe`, prefer the original query. Simplify it to core keywords only if previous attempts with the original query have failed to produce good results.
- **Format Compliance**: Adhere strictly to the two-part "Analysis" and "Proposed Action" format.
- **Performance of probe model** : Probe model is not perfect. It may not find all candidates. So you can use remove action to filter out candidates or re-probe short area with both queries.
"""

def _create_focused_prompt(issue: Dict[str, Any], video_end: Optional[float],
                          history: List[Dict[str, Any]], summaries: Dict[str, Any],
                          a_query: str, b_query: str) -> str:
    """
    LLM에 전달할 시스템 프롬프트와 사용자 프롬프트를 생성합니다.
    Chain-of-Thought 구조를 사용합니다.
    """
    # --- 1. 이슈 정보 추출 및 포맷팅 ---
    issue_type = issue.get("type", "Unknown")
    # inspector.py v2에서 'actions'는 리스트이므로 문자열로 변환
    recommended_actions_list = issue.get("actions", ["No specific action recommended."])
    recommended_action_str = "\n".join(f"- {act}" for act in recommended_actions_list)
    
    if 'text' in issue:
        issue_details_str = issue['text']
    else:
        issue_details_str = json.dumps({k: v for k, v in issue.items() if k not in ['type', 'actions']})

    # --- 2. 동적 컨텍스트 정보 포맷팅 ---
    context_str = f"## CONTEXT\n- Video Duration: {video_end:.2f} seconds. All timestamps must be within [0.0, {video_end:.2f}].\n- Original Query for 'A': \"{a_query}\"\n- Original Query for 'B': \"{b_query}\"" if video_end is not None else ""

    history_str = "## HISTORY\n"
    if not history:
        history_str += "- No history yet. This is the first iteration."
    else:
        for i, entry in enumerate(history):
            iter_num = entry.get('iter', i + 1)
            issue_faced = entry.get('focused_issue', {}).get('type', 'N/A')
            actions_taken = entry.get('llm_actions', [])
            action_summary = ", ".join(a.get('action', 'unknown') for a in actions_taken)
            
            # 변경사항 요약 정보 추가
            changes = entry.get('changes', [])
            total_added = sum(len(change.get('added_spans', [])) for change in changes)
            total_removed = sum(len(change.get('removed_spans', [])) for change in changes)
            
            final_A = entry.get('final_A_count', 'N/A')
            final_B = entry.get('final_B_count', 'N/A')
            
            change_info = f" (Added: {total_added}, Removed: {total_removed}, Final: A={final_A}, B={final_B})" if changes else ""
            
            history_str += f"- Iteration {iter_num}: Faced issue '{issue_faced}', Took action(s) '{action_summary}'{change_info}.\n"

    summaries_str = f"## CURRENT RETRIEVAL RESULTS\n{json.dumps(summaries, indent=2)}"

    # --- 3. 사용자 프롬프트 최종 조합 ---
    user_prompt = f"""
Analyze the following state of a temporal query task and generate a response in the required Chain-of-Thought format.

{context_str}

{history_str}

{summaries_str}

## CURRENT ISSUE TO SOLVE
- Type: "{issue_type}"
- Details: {issue_details_str}
- Possible Actions:
{recommended_action_str}
"""
    
    # --- 4. 시스템 프롬프트와 조합하여 반환 ---
    return COT_PROFESSIONAL_SYSTEM_PROMPT + user_prompt

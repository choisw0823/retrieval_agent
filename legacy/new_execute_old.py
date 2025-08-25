#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
new_execute.py (v3.0 - Refactored Core Logic Only)

요약
- `plan_select.py`로부터 실행 계획(Plan)을 전달받아 순서대로 실행.
- Probe는 MR(Moment Retrieval)을 통해 실행.
- BEFORE/AFTER 연산은 "inspector.py" + LLM 채팅 루프를 사용해 해결.
- 모듈화된 구조로 핵심 로직만 포함

사용
  - (plan_select.py를 통해 Plan을 먼저 생성한 후)
  - from new_execute import PlanExecutor
  - PlanExecutor(video_path,...).run(ranked_plans)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import os
import traceback

from dotenv import load_dotenv
load_dotenv()

# --- 프로젝트 모듈 임포트
from planner import PlanNode, Probe, Sequence
from mr_backend import MomentRetrievalMR
from inspector.inspector_order import inspect_order
from plan_selector import PlanSelector

# 새로 분리된 모듈들
from action_parser import process_action
from probe_actions import process_legacy_action
from history_manager import (
    print_history_summary, create_iteration_history, 
    finalize_iteration_history, add_action_changes
)
from execution_utils import (
    _conf, _dedup_spans, _fetch, _get_priority_issue, _create_focused_prompt,
    visualize_spans, _emit_spans_from_pairs, _compute_pairs_basic, _ffprobe_duration
)
from gemini_chat import GeminiChatJSON

try:
    from vlm_backend import VLMBackend
except Exception:
    VLMBackend = None
# ---------------- Types ----------------
Span = Dict[str, Any]
InspectorFn = Callable[[List[Span], List[Span], str, str], Dict[str, Any]]

# ---------- inspector parsing & helpers ----------
def _get_priority_issue(issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not issues: return None
    priority_order = ["EmptyA", "EmptyB", "NoCandidates", "OverlapWithAnchor", "OverlapWithB", "OverlongSpan"]
    for p_type in priority_order:
        for issue in issues:
            if issue.get("type") == p_type: return issue
    return issues[0]


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

3.  **`split`**: Split a video segment into multiple segments using timestamp list.
    ```json
    {"action": "split", "pairs": [{"a":{...}, "b":{...}}, ...], "timestamps": [[t1, t2, t3, ...], [t4, t5, t6, ...], ...]}
    ```

4.  **`concat`**: Concatenate multiple video segments into one and remove original segments.
    ```json
    {"action": "concat", "pairs": [{"a":{...}, "b":{...}}, ...], "target_list": "A" | "B"}
    ```

## Guiding Principles
- **Logic over Instinct**: Your analysis must clearly justify your proposed action.
- **Query Strategy**: When using `probe`, prefer the original query. Simplify it to core keywords only if previous attempts with the original query have failed to produce good results.
- **Format Compliance**: Adhere strictly to the two-part "Analysis" and "Proposed Action" format.
- **Performance of probe model** : Probe model is not perfect. It may not find all candidates. So you can use remove action to filter out candidates or re-probe short area with both queries.
"""


def _create_focused_prompt(
    issue: Dict[str, Any],
    video_end: Optional[float],
    history: List[Dict[str, Any]],
    summaries: Dict[str, Any],
    a_query: str,
    #a_query_list: List[str],
    b_query: str,
    #b_query_list: List[str]
) -> str:
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
        issue_details_str = json.dumps({k:v for k,v in issue.items() if k not in ['type', 'actions']})

    # --- 2. 동적 컨텍스트 정보 포맷팅 ---
    context_str = f"## CONTEXT\n- Video Duration: {video_end:.2f} seconds. All timestamps must be within [0.0, {video_end:.2f}].\n- Original Query for 'A': \"{a_query}\"\n- Original Query for 'B': \"{b_query}\"" if video_end is not None else ""
    #context_str += f"\n- Current candidate for {a_query}: {a_query_list}\n- Current candidate for {b_query}: {b_query_list}"

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
- Possible Actions:\n{recommended_action_str}
"""
    
    # --- 4. 시스템 프롬프트와 조합하여 반환 ---
    return COT_PROFESSIONAL_SYSTEM_PROMPT+user_prompt



def _pairs_to_AB(pairs: List[Dict[str, Any]]) -> Tuple[List[Span], List[Span]]:
    def _f(s): return {"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(s.get("conf", 1.0))}
    A = [_f(p.get("a", p.get("A"))) for p in pairs if p.get("a") or p.get("A")]
    B = [_f(p.get("b", p.get("B"))) for p in pairs if p.get("b") or p.get("B")]
    return A, B

def _emit_spans_from_pairs(pairs: List[Dict[str,Any]], emit: str) -> List[Span]:
    out=[]
    if emit == "pair":
        for pr in pairs:
            a, b = pr["a"], pr["b"]; t0, t1 = min(float(a["t0"]), float(b["t0"])), max(float(a["t1"]), float(b["t1"])); out.append({"t0": t0, "t1": t1, "conf": float(pr.get("score", b.get("conf", 1.0)))})
    elif emit == "union":
        seen=set()
        for pr in pairs:
            for key in ("a","b"):
                s = pr[key]; k=(round(float(s["t0"]),3), round(float(s["t1"]),3), key)
                if k in seen: continue
                seen.add(k); out.append({"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(s.get("conf", 1.0))})
    else:
        target_key = "b" if emit == "right" else "a"; seen=set()
        for pr in pairs:
            s = pr[target_key]; k=(round(float(s["t0"]),3), round(float(s["t1"]),3))
            if k in seen: continue
            seen.add(k); out.append({"t0": float(s["t0"]), "t1": float(s["t1"]), "conf": float(pr.get("score", s.get("conf",1.0)))})
    return out

class GeminiChatJSON:
    def __init__(self, model: str = "gemini-2.5-pro"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        self.model = model; self.client = genai.Client(api_key=api_key); self.config = genai.types.GenerateContentConfig(response_mime_type="application/json"); self.history: List[Any] = []

    def ask(self, system_prompt: str) -> List[Dict[str, Any]]:
        contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=system_prompt)])]
        try:
            resp = self.client.models.generate_content(model=self.model, contents=contents, config=self.config)
            text = (resp.text or "").strip()
            self.history.append({'role': 'model', 'parts': [text]})
            if text.startswith("```"): 
                text = text.strip("`")
                text = text[4:].strip() if text.startswith("json") else text
            m = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if not m: raise ValueError(f"LLM returned non-JSON: {text[:200]}")
            data = json.loads(m.group(0))
            
            if isinstance(data, dict): return [data]
            if isinstance(data, list) and all(isinstance(x, dict) for x in data): return data
            raise ValueError("LLM returned JSON that is neither dict nor list of dicts.")
        except Exception as e:
            print(f"Error during Gemini API call: {e}"); return [{"action": "stop", "status": "FAIL", "reason": f"LLM API Error: {e}", "pairs": []}]

# ---------------- Small utilities ----------------
def _conf(s: Span) -> float: return float(s.get("conf", s.get("score", 0.0)))

def _dedup_spans(spans: List[Span]) -> List[Span]:
    seen=set(); out=[]
    for s in spans:
        k=(round(float(s["t0"]),5), round(float(s["t1"]),5), s.get("tag",""))
        if k not in seen: seen.add(k); out.append(s)
    return out

def _summarize_spans(spans: List[Span], k: int = 100) -> Dict[str, Any]:
    if not spans: return {"count": 0, "samples": []}
    top = sorted(spans, key=_conf, reverse=True)[:k]
    return {"count": len(spans), "samples": [{"t0": s["t0"], "t1": s["t1"], "conf": _conf(s), "tag": s.get("tag","")} for s in top]}

def _fetch(mr: MomentRetrievalMR, query: str, window: Optional[List[float]], topk: int=5) -> List[Span]:
    return mr.fetch(query, window, topk, {}) or []

def _summ(s: Span) -> Dict[str, Any]: return {"t0": s["t0"], "t1": s["t1"], "conf": _conf(s)}

def _compute_pairs_basic(A: List[Span], B: List[Span], operator: str) -> List[Dict[str, Any]]:
    pairs=[]; A_s=sorted(A,key=lambda s:s["t0"]); B_s=sorted(B,key=lambda s:s["t0"])
    for a in A_s:
        for b in B_s:
            valid=False; EPS_NEAR = 1.0
            if operator=="BEFORE" and b["t0"] > a["t1"] + EPS_NEAR: valid=True
            elif operator=="AFTER" and a["t0"] > b["t1"] + EPS_NEAR: valid=True
            if valid: pairs.append({"a":_summ(a),"b":_summ(b),"score":_conf(b)})
    return sorted(pairs, key=lambda x:x["score"], reverse=True)

def _ffprobe_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","json",path], stderr=subprocess.STDOUT).decode("utf-8","ignore")
        return float(json.loads(out)["format"]["duration"])
    except Exception: return None

def visualize_spans(A: List[Span], B: List[Span], a_query: str, b_query: str, title: str, video_end: Optional[float] = None, save_path: Optional[str] = None):
    if not A and not B: print(f"[VIZ] No spans to visualize for: {title}"); return
    plt.figure(figsize=(14, 6)); y_positions = {"A": 1, "B": 0}; y_labels = [f"{b_query[:30]}...", f"{a_query[:30]}..."]; colors = {"A": "#90EE90", "B": "#FFB6C1"}
    for span in A: t0, t1, conf = float(span["t0"]), float(span["t1"]), _conf(span); rect = patches.Rectangle((t0, y_positions["A"] - 0.3), t1 - t0, 0.6, linewidth=1, edgecolor='darkgreen', facecolor=colors["A"], alpha=0.7); plt.gca().add_patch(rect); plt.text(t0 + (t1-t0)/2, y_positions["A"], f'{conf:.4f}', ha='center', va='center', fontsize=8, fontweight='bold')
    for span in B: t0, t1, conf = float(span["t0"]), float(span["t1"]), _conf(span); rect = patches.Rectangle((t0, y_positions["B"] - 0.3), t1 - t0, 0.6, linewidth=1, edgecolor='darkred', facecolor=colors["B"], alpha=0.7); plt.gca().add_patch(rect); plt.text(t0 + (t1-t0)/2, y_positions["B"], f'{conf:.4f}', ha='center', va='center', fontsize=8, fontweight='bold')
    plt.ylim(-0.5, 1.5); plt.yticks([0, 1], y_labels)
    all_spans = A + B
    if all_spans: min_t, max_t = min(float(s["t0"]) for s in all_spans), max(float(s["t1"]) for s in all_spans); margin = (max_t - min_t) * 0.1; plt.xlim(max(0, min_t - margin), max_t + margin)
    if video_end: plt.axvline(x=video_end, color='red', linestyle='--', alpha=0.5, label=f'Video End ({video_end:.1f}s)'); plt.legend()
    plt.xlabel('Time (s)'); plt.title(title); plt.grid(True, alpha=0.3)
    legend_elements = [patches.Patch(color=colors["A"], label=f'A: {a_query[:20]}...'), patches.Patch(color=colors["B"], label=f'B: {b_query[:20]}...')]
    plt.legend(handles=legend_elements, loc='upper right'); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"[VIZ] Saved visualization: {save_path}")
    else: plt.show()
    plt.close()

# ---------------- BEFORE/AFTER relation chat runner ----------------
@dataclass
class ExecContext:
    mr: MomentRetrievalMR; vlm: Optional[VLMBackend]; model: str; video_end: Optional[float]


class BARelationChatRunner:
    def __init__(self, ctx: ExecContext, inspector_fn: InspectorFn): 
        self.ctx = ctx
        self.inspector_fn = inspector_fn

    def run(self, operator: str, emit: str, a_query: str, b_query: str, max_iters: int = 8) -> Dict[str, Any]:
        operator, emit = operator.upper(), emit.lower(); 
        video_window = [0, self.ctx.video_end] if self.ctx.video_end else None
        
        A = _dedup_spans(_fetch(self.ctx.mr, a_query, video_window, 5)); 
        B = _dedup_spans(_fetch(self.ctx.mr, b_query, video_window, 5))
        history: List[Dict[str,Any]] = []; 
        chat = GeminiChatJSON(model=self.ctx.model)
        
        visualize_spans(A, B, a_query, b_query, f"Initial MR Results: {operator} relation", self.ctx.video_end, f"viz_initial_{operator.lower()}.png")
        
        for it in range(1, max_iters + 1):
            insp_result = self.inspector_fn(A, B, operator=operator, emit=emit)
            issues = insp_result.get("issues", [])
            summaries = {"A": _summarize_spans(A), "B": _summarize_spans(B)}

            print(f"[ITER {it}] Inspector found {len(issues)} issues.")
            if not issues:
                final_pairs = insp_result.get("pairs", [])
                emitted = _emit_spans_from_pairs(final_pairs, emit)
                visualize_spans(A, B, a_query, b_query, f"FINAL SUCCESS: {operator} relation (Iter {it})", self.ctx.video_end, f"viz_final_success_{operator.lower()}.png")
                return {"success": bool(emitted), "status": "SUCCESS", "reason": "All issues resolved", "pairs": final_pairs, "emitted_spans": emitted, "history": history}
            
            current_issue = _get_priority_issue(issues)
            print(f"[ITER {it}] Focusing on issue: {current_issue}")
            focused_prompt = _create_focused_prompt(current_issue, self.ctx.video_end, history, summaries, a_query, b_query)
            #payload = {"iteration": it, "operator": operator, "emit": emit, "queries": {"A": a_query, "B": b_query}, "current_issue_to_solve": current_issue, "summaries": {"A": _summarize_spans(A), "B": _summarize_spans(B)}, "video_end": self.ctx.video_end}
            actions = chat.ask(focused_prompt)
            print(f"[ITER {it}] LLM Actions ({len(actions)}): {actions}")
            
            # 히스토리 항목 초기화 (변경사항 추적용)
            iter_history = {
                "iter": it, 
                "focused_issue": current_issue, 
                "llm_actions": actions,
                "changes": []  # 각 액션에서 일어난 변경사항들
            }
            
            for ai, action in enumerate(actions, 1):
                print(f"[ITER {it}] [ACTION {ai}/{len(actions)}] {action}"); act = action.get("action")
                action_changes = {"action_id": ai, "action_type": act, "added_spans": [], "removed_spans": []}
                
                if act == "probe":
                    target_list, q, windows = action.get("target_list"), action.get("query"), action.get("windows")
                    if target_list not in ["A", "B"] or not q or not windows: 
                        print(f"[WARN] Invalid 'probe' action from LLM. Skipping."); continue
                    
                    # 현재 상태 백업 (변경사항 추적용)
                    A_before, B_before = A.copy(), B.copy()
                    
                    added: List[Span] = []
                    for w in windows:
                        new_spans = _fetch(self.ctx.mr, q, w, 5)
                        added.extend(new_spans)
                        print(f"[ACTION] Added {len(new_spans)} spans from {q} in window {w}")
                    
                    issue_type, ia, ib = current_issue.get("type", ""), current_issue.get("ia"), current_issue.get("ib")
                    
                    # Overlap 이슈인 경우 기존 span 삭제
                    removed_spans = []
                    if "Overlap" in issue_type:
                        if ia is not None and 0 <= ia < len(A): 
                            removed_spans.append(("A", ia, A[ia].copy()))
                            A = [s for i, s in enumerate(A) if i != ia]
                        if ib is not None and 0 <= ib < len(B): 
                            removed_spans.append(("B", ib, B[ib].copy()))
                            B = [s for i, s in enumerate(B) if i != ib]
                        print(f"[ACTION] Dropped conflicting spans A[{ia}] and B[{ib}] before injecting new probe results.")
                    
                    # 새로운 span들 추가
                    if target_list == "A": 
                        A = _dedup_spans(A + added)
                        added_after_dedup = [s for s in A if s not in A_before]
                    else: 
                        B = _dedup_spans(B + added)
                        added_after_dedup = [s for s in B if s not in B_before]
                    
                    # 변경사항 기록
                    action_changes.update({
                        "target_list": target_list,
                        "query": q,
                        "windows": windows,
                        "added_spans": [{"span": s, "source": "probe"} for s in added_after_dedup],
                        "removed_spans": [{"list": lst, "index": idx, "span": span} for lst, idx, span in removed_spans]
                    })

                elif act in ["split", "concat", "remove"]:
                    # 새로운 액션들을 action_parser 모듈로 처리
                    print(f"[{act.upper()}] Processing action: {action}")
                    try:
                        A, B, action_changes = process_action(action, A, B, vlm_backend=self.ctx.vlm)
                        print(f"[{act.upper()}] Action completed: {action_changes}")
                    except Exception as e:
                        print(f"[{act.upper()}] Error processing action: {e}")
                        action_changes.update({
                            "error": str(e),
                            "added_spans": [],
                            "removed_spans": []
                        })
                
                elif act == "verify":
                    pairs, prompt = action.get("pairs", []), action.get("prompt", "");
                    if not pairs or not prompt or not self.ctx.vlm: print(f"[WARN] Invalid 'verify' action or VLM not available. Skipping."); continue
                    
                    # 현재 상태 백업
                    A_before, B_before = A.copy(), B.copy()
                    
                    print(f"[VERIFY] Checking {len(pairs)} pairs with prompt: '{prompt}'"); verified_pairs = []
                    verification_results = []
                    
                    for pi, pair in enumerate(pairs):
                        a_span, b_span = pair.get("a", {}), pair.get("b", {}); t0, t1 = min(float(a_span.get("t0", 0)), float(b_span.get("t0", 0))), max(float(a_span.get("t1", 0)), float(b_span.get("t1", 0)))
                        try:
                            answer, confidence = self.ctx.vlm.query(t0, t1, prompt); answer_text = str(answer).lower()
                            print(f"[VERIFY] Pair {pi+1}: t0={t0:.1f}s-{t1:.1f}s, VLM answer: '{answer}' (conf: {confidence:.3f})")
                            if "yes" in answer_text or answer is True: 
                                verified_pairs.append(pair); print(f"[VERIFY] ✅ Pair {pi+1} KEPT (VLM: yes)")
                                verification_results.append({"pair_id": pi, "kept": True, "answer": answer, "confidence": confidence})
                            else: 
                                print(f"[VERIFY] ❌ Pair {pi+1} REMOVED (VLM: no)")
                                verification_results.append({"pair_id": pi, "kept": False, "answer": answer, "confidence": confidence})
                        except Exception as e: print(f"[VERIFY] Error verifying pair {pi+1}: {e}. Keeping pair as fallback."); verified_pairs.append(pair)
                    
                    if verified_pairs: 
                        print(f"[VERIFY] {len(verified_pairs)}/{len(pairs)} pairs passed verification")
                        A, B = _pairs_to_AB(verified_pairs); A, B = _dedup_spans(A), _dedup_spans(B)
                        
                        # 변경사항 계산
                        removed_from_A = [s for s in A_before if s not in A]
                        removed_from_B = [s for s in B_before if s not in B]
                        added_to_A = [s for s in A if s not in A_before]
                        added_to_B = [s for s in B if s not in B_before]
                        
                        action_changes.update({
                            "prompt": prompt,
                            "pairs_checked": len(pairs),
                            "pairs_kept": len(verified_pairs),
                            "verification_results": verification_results,
                            "added_spans": [{"span": s, "source": "verify_A"} for s in added_to_A] + [{"span": s, "source": "verify_B"} for s in added_to_B],
                            "removed_spans": [{"list": "A", "span": s, "reason": "verify_filter"} for s in removed_from_A] + [{"list": "B", "span": s, "reason": "verify_filter"} for s in removed_from_B]
                        })
                    else: 
                        print(f"[VERIFY] No pairs passed verification. Keeping original spans.")
                        action_changes.update({
                            "prompt": prompt,
                            "pairs_checked": len(pairs),
                            "pairs_kept": 0,
                            "verification_results": verification_results,
                            "added_spans": [],
                            "removed_spans": []
                        })
                
                elif act == "stop" and action.get("status") == "SUCCESS":
                    proposed_pairs = action.get("pairs", [])
                    if proposed_pairs: 
                        # 현재 상태 백업
                        A_before, B_before = A.copy(), B.copy()
                        
                        print(f"[ITER {it}] LLM proposed a solution. Applying and re-evaluating.")
                        A_new, B_new = _pairs_to_AB(proposed_pairs)
                        A, B = _dedup_spans(A_new), _dedup_spans(B_new)
                        
                        # 변경사항 계산
                        removed_from_A = [s for s in A_before if s not in A]
                        removed_from_B = [s for s in B_before if s not in B]
                        added_to_A = [s for s in A if s not in A_before]
                        added_to_B = [s for s in B if s not in B_before]
                        
                        action_changes.update({
                            "proposed_pairs": proposed_pairs,
                            "added_spans": [{"span": s, "source": "stop_solution_A"} for s in added_to_A] + [{"span": s, "source": "stop_solution_B"} for s in added_to_B],
                            "removed_spans": [{"list": "A", "span": s, "reason": "stop_solution_replace"} for s in removed_from_A] + [{"list": "B", "span": s, "reason": "stop_solution_replace"} for s in removed_from_B]
                        })
                        
                        chat = GeminiChatJSON(model=self.ctx.model)
                    else:
                        action_changes.update({
                            "proposed_pairs": [],
                            "added_spans": [],
                            "removed_spans": []
                        })
                    break
                
                else: 
                    print(f"[WARN] Unknown or failed action: {action}")
                    action_changes.update({
                        "error": f"Unknown or failed action: {action}",
                        "added_spans": [],
                        "removed_spans": []
                    })
                    continue
                
                # 각 액션의 변경사항을 기록
                iter_history["changes"].append(action_changes)
            
            # 이번 반복의 최종 상태 추가
            iter_history.update({
                "final_A_count": len(A),
                "final_B_count": len(B),
                "final_A_spans": [{"t0": s.get("t0"), "t1": s.get("t1"), "score": s.get("score")} for s in A],
                "final_B_spans": [{"t0": s.get("t0"), "t1": s.get("t1"), "score": s.get("score")} for s in B]
            })
            
            # 히스토리에 이번 반복 기록 추가
            history.append(iter_history)
            
            visualize_spans(A, B, a_query, b_query, f"After Iteration {it}: {operator} relation", self.ctx.video_end, f"viz_iter{it:02d}_{operator.lower()}.png")
        
        final_pairs = _compute_pairs_basic(A, B, operator)
       
        if final_pairs:
            emitted = _emit_spans_from_pairs(final_pairs, emit)
            return {"success": bool(emitted), "status":"PARTIAL_SUCCESS", "reason":"Timeout, basic pairs returned", "pairs": final_pairs, "emitted_spans": emitted, "history": history}
        
        return {"success": False, "status":"TIMEOUT", "reason":"Max iterations reached", "pairs": [], "emitted_spans": [], "history": history}

# ---------------- Plan Executor ----------------
class PlanExecutor:
    def __init__(self, video_path: str, model: str = "gemini-2.5-pro", device: str = "cuda"):
        self.video_path = video_path; self.model = model
        self.mr = MomentRetrievalMR(video_path=video_path, device=device)
        try: self.vlm = VLMBackend(video_path=video_path); print("[VLM] enabled")
        except Exception as e: self.vlm = None; print(f"[VLM] disabled: {e}")
        self.video_end = _ffprobe_duration(video_path)

    def _eval_probe(self, node: Probe) -> Tuple[str, List[Span], str]:
        key, q = node.target_alias or "probe", node.query_text
        spans = _fetch(self.mr, q, node.temporal_window, int(node.hint.get("mr_topk", 5)))
        return key, spans, q

    def _eval_sequence_ba(self, seq: Sequence, l: Probe, r: Probe) -> Tuple[str, List[Span]]:
        _, _, lq = self._eval_probe(l); _, _, rq = self._eval_probe(r)
        cond = seq.condition or {}; op = cond.get("op","BEFORE").upper()
        if op == ">>": op = "BEFORE"
        emit = getattr(seq, "emit", "right").lower()
        ctx = ExecContext(mr=self.mr, vlm=self.vlm, model=self.model, video_end=self.video_end)
        runner = BARelationChatRunner(ctx, inspect_order)
        res = runner.run(operator=op, emit=emit, a_query=lq, b_query=rq)
        return f"seq_{op.lower()}_{emit}", (res["emitted_spans"] if res.get("success") else [])

    def _eval_node(self, node: PlanNode) -> Tuple[str, List[Span]]:
        if isinstance(node, Probe): return self._eval_probe(node)[0:2]
        if isinstance(node, Sequence) and len(node.steps)==2 and all(isinstance(s,Probe) for s in node.steps):
            return self._eval_sequence_ba(node, node.steps[0], node.steps[1])
        return "unsupported_node", []

    def run(self, ranked_plans: List[Tuple[int, PlanNode]]) -> Dict[str, Any]:
        """Executes a list of ranked plans."""
        exec_report=[]
        for pidx, plan in ranked_plans:
            print(f"[TRYING] Plan {pidx}: {plan}")
            try:
                key, spans = self._eval_node(plan)
                exec_report.append({"plan_index": pidx, "key": key, "spans_found": len(spans)})
                if spans:
                    spans_sorted = sorted(spans, key=_conf, reverse=True)
                    return {"success": True, "plan_index": pidx, "plan": str(plan), "answer_key": key, "spans": spans_sorted, "exec_report": exec_report}
            except Exception as e:
                traceback.print_exc()
                exec_report.append({"plan_index": pidx, "error": str(e)})
        return {"success": False, "reason": "No plan produced spans", "exec_report": exec_report}

# ---------------- History Utilities ----------------

def print_history_summary(history: List[Dict[str, Any]]) -> None:
    """
    히스토리의 변경사항을 요약해서 출력합니다.
    """
    print("\n" + "="*60)
    print("EXECUTION HISTORY SUMMARY")
    print("="*60)
    
    for iter_data in history:
        iter_num = iter_data.get("iter", 0)
        focused_issue = iter_data.get("focused_issue", {})
        changes = iter_data.get("changes", [])
        
        print(f"\n--- Iteration {iter_num} ---")
        print(f"Focused Issue: {focused_issue.get('type', 'Unknown')} (A[{focused_issue.get('ia')}], B[{focused_issue.get('ib')}])")
        
        if not changes:
            print("  No changes recorded")
            continue
            
        for i, change in enumerate(changes, 1):
            action_type = change.get("action_type", "unknown")
            added = change.get("added_spans", [])
            removed = change.get("removed_spans", [])
            
            print(f"  Action {i}: {action_type}")
            
            if action_type == "probe":
                target = change.get("target_list", "?")
                query = change.get("query", "?")
                print(f"    Query: '{query}' → {target}")
                
            elif action_type == "verify":
                checked = change.get("pairs_checked", 0)
                kept = change.get("pairs_kept", 0)
                print(f"    Verified: {kept}/{checked} pairs kept")
                
            elif action_type == "stop":
                pairs = len(change.get("proposed_pairs", []))
                print(f"    Proposed: {pairs} solution pairs")
            
            # 변경 요약
            if added or removed:
                print(f"    Changes: +{len(added)} spans, -{len(removed)} spans")
                
                if added:
                    print(f"      Added to A: {len([s for s in added if 'A' in s.get('source', '')])}")
                    print(f"      Added to B: {len([s for s in added if 'B' in s.get('source', '')])}")
                    
                if removed:
                    removed_A = [s for s in removed if s.get('list') == 'A']
                    removed_B = [s for s in removed if s.get('list') == 'B']
                    if removed_A:
                        print(f"      Removed from A: {len(removed_A)}")
                    if removed_B:
                        print(f"      Removed from B: {len(removed_B)}")
        
        # 최종 상태
        final_A = iter_data.get("final_A_count", 0)
        final_B = iter_data.get("final_B_count", 0)
        print(f"  Final state: A={final_A} spans, B={final_B} spans")

def print_detailed_history(history: List[Dict[str, Any]]) -> None:
    """
    히스토리의 상세한 변경사항을 출력합니다.
    """
    print("\n" + "="*80)
    print("DETAILED EXECUTION HISTORY")
    print("="*80)
    
    for iter_data in history:
        iter_num = iter_data.get("iter", 0)
        print(f"\n{'='*20} Iteration {iter_num} {'='*20}")
        
        # 포커스된 이슈
        focused_issue = iter_data.get("focused_issue", {})
        print(f"Focused Issue: {focused_issue}")
        
        # 각 액션의 상세 변경사항
        changes = iter_data.get("changes", [])
        for i, change in enumerate(changes, 1):
            print(f"\n--- Action {i}: {change.get('action_type', 'unknown')} ---")
            
            # 추가된 span 상세 정보
            added = change.get("added_spans", [])
            if added:
                print(f"Added {len(added)} spans:")
                for j, span_info in enumerate(added[:3]):  # 최대 3개만 출력
                    span = span_info.get("span", {})
                    source = span_info.get("source", "unknown")
                    print(f"  {j+1}. [{span.get('t0', 0):.1f}s-{span.get('t1', 0):.1f}s] score={span.get('score', 0):.3f} (from {source})")
                if len(added) > 3:
                    print(f"  ... and {len(added)-3} more")
            
            # 제거된 span 상세 정보
            removed = change.get("removed_spans", [])
            if removed:
                print(f"Removed {len(removed)} spans:")
                for j, span_info in enumerate(removed[:3]):  # 최대 3개만 출력
                    span = span_info.get("span", {})
                    reason = span_info.get("reason", "unknown")
                    list_name = span_info.get("list", "?")
                    print(f"  {j+1}. {list_name}[{span_info.get('index', '?')}]: [{span.get('t0', 0):.1f}s-{span.get('t1', 0):.1f}s] (reason: {reason})")
                if len(removed) > 3:
                    print(f"  ... and {len(removed)-3} more")
        
        # 최종 span 정보
        final_A_spans = iter_data.get("final_A_spans", [])
        final_B_spans = iter_data.get("final_B_spans", [])
        print(f"\nFinal state: A={len(final_A_spans)} spans, B={len(final_B_spans)} spans")

# ---------------- CLI Demo ----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plan runner with inspector.py + Gemini loop (BEFORE/AFTER)")
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--query", required=True, type=str)
    ap.add_argument("--model", default="gemini-2.5-pro", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--top_plans", default=1, type=int)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}"); raise SystemExit(1)

    # 1. Select Plan(s)
    selector = PlanSelector(model=args.model)
    plans_to_execute = selector.select_plans(args.query, top_k=args.top_plans)

    # 2. Execute Plan(s)
    executor = PlanExecutor(video_path=args.video, model=args.model, device=args.device)
    result = executor.run(plans_to_execute)

    print("\n" + "="*50)
    if result.get("success"):
        print(f"\n[RESULT] SUCCESS with plan {result['plan_index']}, found {len(result['spans'])} spans.")
        print(f"  Plan: {result.get('plan')}")
        for i, s in enumerate(result["spans"][:10], 1):
            print(f"  {i:02d}. t0={s['t0']:.2f}s  t1={s['t1']:.2f}s  conf={s['conf']:.4f}")
        
        # 실행 히스토리 출력
        exec_report = result.get("exec_report", [])
        for report in exec_report:
            if "history" in report and report["history"]:
                print_history_summary(report["history"])
                break
    else:
        print("\n[RESULT] FAIL")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 실패한 경우에도 히스토리가 있으면 출력
        exec_report = result.get("exec_report", [])
        for report in exec_report:
            if "history" in report and report["history"]:
                print_history_summary(report["history"])
                break
    print("="*50)
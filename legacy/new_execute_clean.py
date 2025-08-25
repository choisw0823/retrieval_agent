#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
new_execute_clean.py (v3.0 - Refactored Core Logic Only)

요약
- `plan_select.py`로부터 실행 계획(Plan)을 전달받아 순서대로 실행.
- Probe는 MR(Moment Retrieval)을 통해 실행.
- BEFORE/AFTER 연산은 "inspector.py" + LLM 채팅 루프를 사용해 해결.
- 모듈화된 구조로 핵심 로직만 포함

사용
  - (plan_select.py를 통해 Plan을 먼저 생성한 후)
  - from new_execute_clean import PlanExecutor
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
    visualize_spans, _emit_spans_from_pairs, _compute_pairs_basic, _ffprobe_duration,
    _summarize_spans
)
from gemini_chat import GeminiChatJSON

try:
    from vlm_backend import VLMBackend
except Exception:
    VLMBackend = None

# ---------------- Types ----------------
Span = Dict[str, Any]
InspectorFn = Callable[[List[Span], List[Span], str, str], Dict[str, Any]]

# ---------------- Core Classes ----------------
@dataclass
class ExecContext:
    """실행 컨텍스트를 담는 데이터 클래스"""
    mr: MomentRetrievalMR
    vlm: Optional[VLMBackend]
    model: str
    video_end: Optional[float]

class BARelationChatRunner:
    """BEFORE/AFTER relation을 위한 채팅 기반 실행 런너"""
    
    def __init__(self, ctx: ExecContext, inspector_fn: InspectorFn): 
        self.ctx = ctx
        self.inspector_fn = inspector_fn

    def run(self, operator: str, emit: str, a_query: str, b_query: str, max_iters: int = 8) -> Dict[str, Any]:
        """메인 실행 로직"""
        operator, emit = operator.upper(), emit.lower()
        video_window = [0, self.ctx.video_end] if self.ctx.video_end else None
        
        # 초기 MR 검색
        A = _dedup_spans(_fetch(self.ctx.mr, a_query, video_window, 5))
        B = _dedup_spans(_fetch(self.ctx.mr, b_query, video_window, 5))
        history: List[Dict[str, Any]] = []
        chat = GeminiChatJSON(model=self.ctx.model)
        
        # 초기 시각화
        visualize_spans(A, B, a_query, b_query, f"Initial MR Results: {operator} relation", 
                       self.ctx.video_end, f"viz_initial_{operator.lower()}.png")
        
        # 반복적 문제 해결
        for it in range(1, max_iters + 1):
            insp_result = self.inspector_fn(A, B, operator=operator, emit=emit)
            issues = insp_result.get("issues", [])
            summaries = {"A": _summarize_spans(A), "B": _summarize_spans(B)}

            print(f"[ITER {it}] Inspector found {len(issues)} issues.")
            
            # 성공 조건 확인
            if not issues:
                final_pairs = insp_result.get("pairs", [])
                emitted = _emit_spans_from_pairs(final_pairs, emit)
                visualize_spans(A, B, a_query, b_query, f"FINAL SUCCESS: {operator} relation (Iter {it})", 
                              self.ctx.video_end, f"viz_final_success_{operator.lower()}.png")
                return {
                    "success": bool(emitted), 
                    "status": "SUCCESS", 
                    "reason": "All issues resolved", 
                    "pairs": final_pairs, 
                    "emitted_spans": emitted, 
                    "history": history
                }
            
            # 이슈 처리
            current_issue = _get_priority_issue(issues)
            print(f"[ITER {it}] Focusing on issue: {current_issue}")
            
            # LLM에게 액션 요청
            focused_prompt = _create_focused_prompt(current_issue, self.ctx.video_end, history, summaries, a_query, b_query)
            actions = chat.ask(focused_prompt)
            print(f"[ITER {it}] LLM Actions ({len(actions)}): {actions}")
            
            # 히스토리 초기화
            iter_history = create_iteration_history(it, current_issue, actions)
            
            # 각 액션 실행
            for ai, action in enumerate(actions, 1):
                print(f"[ITER {it}] [ACTION {ai}/{len(actions)}] {action}")
                act = action.get("action")
                
                try:
                    # 액션 타입에 따른 처리
                    if act in ["split", "concat", "remove"]:
                        # 새로운 액션들
                        A, B, action_changes = process_action(action, A, B, vlm_backend=self.ctx.vlm)
                    elif act in ["probe", "verify", "stop"]:
                        # 기존 액션들
                        A, B, action_changes = process_legacy_action(action, A, B, current_issue, self.ctx.mr, self.ctx.vlm)
                    else:
                        print(f"[WARN] Unknown action: {act}")
                        action_changes = {
                            "action_id": ai,
                            "action_type": act,
                            "error": f"Unknown action: {act}",
                            "added_spans": [],
                            "removed_spans": []
                        }
                    
                    # 액션 ID 추가
                    action_changes["action_id"] = ai
                    if "action_type" not in action_changes:
                        action_changes["action_type"] = act
                    
                    # 히스토리에 추가
                    add_action_changes(iter_history, action_changes)
                    
                    # stop 액션이면 채팅 초기화
                    if act == "stop" and action.get("status") == "SUCCESS":
                        chat = GeminiChatJSON(model=self.ctx.model)
                        break
                        
                except Exception as e:
                    print(f"[ERROR] Action {act} failed: {e}")
                    action_changes = {
                        "action_id": ai,
                        "action_type": act,
                        "error": str(e),
                        "added_spans": [],
                        "removed_spans": []
                    }
                    add_action_changes(iter_history, action_changes)
            
            # 반복 히스토리 완료
            finalize_iteration_history(iter_history, A, B)
            history.append(iter_history)
            
            # 반복 시각화
            visualize_spans(A, B, a_query, b_query, f"After Iteration {it}: {operator} relation", 
                          self.ctx.video_end, f"viz_iter{it:02d}_{operator.lower()}.png")
        
        # 최대 반복 도달 시 기본 결과 반환
        final_pairs = _compute_pairs_basic(A, B, operator)
        
        if final_pairs:
            emitted = _emit_spans_from_pairs(final_pairs, emit)
            return {
                "success": bool(emitted), 
                "status": "PARTIAL_SUCCESS", 
                "reason": "Timeout, basic pairs returned", 
                "pairs": final_pairs, 
                "emitted_spans": emitted, 
                "history": history
            }
        
        return {
            "success": False, 
            "status": "TIMEOUT", 
            "reason": "Max iterations reached", 
            "pairs": [], 
            "emitted_spans": [], 
            "history": history
        }

# ---------------- Plan Executor ----------------
class PlanExecutor:
    """Plan을 실행하는 메인 클래스"""
    
    def __init__(self, video_path: str, model: str = "gemini-2.5-pro", device: str = "cuda"):
        self.video_path = video_path
        self.model = model
        self.mr = MomentRetrievalMR(video_path=video_path, device=device)
        
        try:
            self.vlm = VLMBackend(video_path=video_path)
            print("[VLM] enabled")
        except Exception as e:
            self.vlm = None
            print(f"[VLM] disabled: {e}")
        
        self.video_end = _ffprobe_duration(video_path)

    def _eval_probe(self, node: Probe) -> Tuple[str, List[Span], str]:
        """Probe 노드 평가"""
        key = node.target_alias or "probe"
        q = node.query_text
        spans = _fetch(self.mr, q, node.temporal_window, int(node.hint.get("mr_topk", 5)))
        return key, spans, q

    def _eval_sequence_ba(self, seq: Sequence, l: Probe, r: Probe) -> Tuple[str, List[Span]]:
        """BEFORE/AFTER 시퀀스 평가"""
        _, _, lq = self._eval_probe(l)
        _, _, rq = self._eval_probe(r)
        
        cond = seq.condition or {}
        op = cond.get("op", "BEFORE").upper()
        if op == ">>":
            op = "BEFORE"
        
        emit = getattr(seq, "emit", "right").lower()
        ctx = ExecContext(mr=self.mr, vlm=self.vlm, model=self.model, video_end=self.video_end)
        runner = BARelationChatRunner(ctx, inspect_order)
        res = runner.run(operator=op, emit=emit, a_query=lq, b_query=rq)
        
        return f"seq_{op.lower()}_{emit}", (res["emitted_spans"] if res.get("success") else [])

    def _eval_node(self, node: PlanNode) -> Tuple[str, List[Span]]:
        """노드 평가"""
        if isinstance(node, Probe):
            return self._eval_probe(node)[0:2]
        
        if isinstance(node, Sequence) and len(node.steps) == 2 and all(isinstance(s, Probe) for s in node.steps):
            return self._eval_sequence_ba(node, node.steps[0], node.steps[1])
        
        return "unsupported_node", []

    def run(self, ranked_plans: List[Tuple[int, PlanNode]]) -> Dict[str, Any]:
        """ranked plans 리스트를 실행"""
        exec_report = []
        
        for pidx, plan in ranked_plans:
            print(f"[TRYING] Plan {pidx}: {plan}")
            
            try:
                key, spans = self._eval_node(plan)
                exec_report.append({"plan_index": pidx, "key": key, "spans_found": len(spans)})
                
                if spans:
                    spans_sorted = sorted(spans, key=_conf, reverse=True)
                    return {
                        "success": True,
                        "plan_index": pidx,
                        "plan": str(plan),
                        "answer_key": key,
                        "spans": spans_sorted,
                        "exec_report": exec_report
                    }
                    
            except Exception as e:
                traceback.print_exc()
                exec_report.append({"plan_index": pidx, "error": str(e)})
        
        return {
            "success": False,
            "reason": "No plan produced spans",
            "exec_report": exec_report
        }

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
        print(f"[ERROR] Video not found: {args.video}")
        raise SystemExit(1)

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
        print(f"Reason: {result.get('reason', 'Unknown')}")
        
        # 실패한 경우에도 히스토리가 있으면 출력
        exec_report = result.get("exec_report", [])
        for report in exec_report:
            if "history" in report and report["history"]:
                print_history_summary(report["history"])
                break
    print("="*50)

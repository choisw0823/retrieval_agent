#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
new_execute_clean.py (v3.2 - Unified ChatRunner)

요약
- `plan_select.py`로부터 실행 계획(Plan)을 전달받아 순서대로 실행.
- Probe는 MR(Moment Retrieval)을 통해 실행.
- Join, BEFORE/AFTER 연산 모두 inspector와 LLM을 사용하는 단일 ChatRunner로 처리.
- 중복 로직을 제거하고 구조를 일반화.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import os
import traceback
import json

from dotenv import load_dotenv
load_dotenv()

# --- 프로젝트 모듈 임포트
from planner import PlanNode, Probe, Sequence, Join, Choice, Filter
from mr_backend import MomentRetrievalMR
from inspector.inspector_order import inspect_order
from inspector.inspector_join import inspect_join
from plan_selector import PlanSelector

# 새로 분리된 모듈들
from action_parser import process_action
from history_manager import (
    print_history_summary, create_iteration_history,
    finalize_iteration_history, add_action_changes
)
from execution_utils import (
    _conf, _dedup_spans, _fetch, _get_priority_issue, _create_focused_prompt,
    visualize_spans, _emit_spans_from_pairs, _compute_pairs_basic, _ffprobe_duration,
    _summarize_spans, _create_focused_prompt
)
from gemini_chat import GeminiChatJSON

try:
    from vlm_backend import VLMBackend
except Exception:
    VLMBackend = None

# ---------------- Types ----------------
Span = Dict[str, Any]
InspectorFn = Callable[..., Dict[str, Any]]
PromptCreatorFn = Callable[..., str]

# ---------------- Core Classes ----------------
@dataclass
class ExecContext:
    """실행 컨텍스트를 담는 데이터 클래스"""
    mr: MomentRetrievalMR
    vlm: Optional[VLMBackend]
    model: str
    video_end: Optional[float]

class ChatRunner:
    """Inspector와 LLM을 사용하여 문제를 반복적으로 해결하는 범용 실행 런너"""

    def __init__(self, ctx: ExecContext, inspector_fn: InspectorFn):
        self.ctx = ctx
        self.inspector_fn = inspector_fn

    def run(self, left_spans: List[Span], right_spans: List[Span], left_query: str, right_query: str,
            prompt_creator: PromptCreatorFn, max_iters: int = 8, **context_kwargs) -> Dict[str, Any]:
        """메인 실행 로직"""
        A = left_spans.copy()
        B = right_spans.copy()
        history: List[Dict[str, Any]] = []
        chat = GeminiChatJSON(model=self.ctx.model)

        # 반복적 문제 해결
        for it in range(1, max_iters + 1):
            insp_result = self.inspector_fn(A, B, **context_kwargs)
            issues = insp_result.get("issues", [])
            
            print(f"[ITER {it}] Inspector found {len(issues)} issues.")
            
            # 성공 조건 확인
            if not issues:
                return { "success": True, "status": "SUCCESS", "reason": "All issues resolved",
                         "final_left": A, "final_right": B, "history": history, "inspection_result": insp_result }

            # 이슈 처리
            current_issue = _get_priority_issue(issues)
            print(f"[ITER {it}] Focusing on issue: {current_issue.get('type')}: {current_issue.get('description')}")

            # LLM에게 액션 요청
            summaries = {"A": _summarize_spans(A), "B": _summarize_spans(B)}

            prompt = prompt_creator(current_issue, self.ctx.video_end, history, summaries, left_query, right_query, **context_kwargs)
            actions = chat.ask(prompt)
            print(f"[ITER {it}] LLM Actions ({len(actions)}): {actions}")

            iter_history = create_iteration_history(it, current_issue, actions)

            # 각 액션 실행
            for ai, action in enumerate(actions, 1):
                act = action.get("action")
                try:
                    A, B, action_changes = process_action(action, A, B, vlm_backend=self.ctx.vlm,
                                                          current_issue=current_issue, mr_backend=self.ctx.mr)
                    action_changes["action_id"] = ai
                    add_action_changes(iter_history, action_changes)
                    if act == "stop" and action.get("status") == "SUCCESS":
                        chat = GeminiChatJSON(model=self.ctx.model)
                        break
                except Exception as e:
                    print(f"[ERROR] Action {act} failed: {e}")
                    action_changes = {"action_id": ai, "action_type": act, "error": str(e)}
                    add_action_changes(iter_history, action_changes)
            
            finalize_iteration_history(iter_history, A, B)
            history.append(iter_history)

            if any(a.get("action") == "stop" for a in actions):
                break

        # 최종 검사 후 반환
        final_insp_result = self.inspector_fn(A, B, **context_kwargs)
        final_issues = final_insp_result.get("issues", [])
        if not final_issues:
            return { "success": True, "status": "SUCCESS", "reason": "Issues resolved within iterations.",
                     "final_left": A, "final_right": B, "history": history, "inspection_result": final_insp_result }
        else:
            return { "success": False, "status": "TIMEOUT", "reason": "Max iterations reached or stopped.",
                     "final_left": A, "final_right": B, "history": history, "inspection_result": final_insp_result }

# ---------------- Plan Executor ----------------
class PlanExecutor:
    """Plan을 실행하는 메인 클래스"""
    
    def __init__(self, video_path: str, model: str = "gemini-1.5-pro-latest", device: str = "cuda"):
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
        self.ctx = ExecContext(mr=self.mr, vlm=self.vlm, model=self.model, video_end=self.video_end)

    def _eval_probe(self, node: Probe) -> Tuple[str, List[Span], str]:
        key = node.target_alias or "probe"
        q = node.query_text
        spans = _fetch(self.mr, q, node.temporal_window, int(node.hint.get("mr_topk", 5)))
        return key, spans, q

    def _eval_sequence_ba(self, seq: Sequence, l: Probe | Join | list[Span], r: Probe | Join | list[Span]) -> Tuple[str, List[Span]]:
        if isinstance(l, Probe):
            _, lspans, lq = self._eval_probe(l)
        elif isinstance(l, Join):
            _, lspans = self._eval_node(l)
            lq = ''
        elif isinstance(l, list):
            lspans = l
            lq = ''

        if isinstance(r, Probe):
            _, rspans, rq = self._eval_probe(r)   
        elif isinstance(r, Join):
            _, rspans = self._eval_node(r)
            rq = ''
        elif isinstance(r, list):
            rspans = r
            rq = ''

        op = (seq.condition or {}).get("op", "BEFORE").upper()
        emit = getattr(seq, "emit", "right").lower()

        # 초기 데이터 Fetch
        A = _dedup_spans(lspans)
        B = _dedup_spans(rspans)

        runner = ChatRunner(self.ctx, inspect_order)
        context = {"operator": op, "emit": emit, "a_query": lq, "b_query": rq}
        res = runner.run(A, B, lq, rq, prompt_creator=_create_focused_prompt, **context)

        if res.get("success"):
            final_pairs = res["inspection_result"].get("pairs", [])
            return f"seq_{op.lower()}_{emit}", _emit_spans_from_pairs(final_pairs, emit)
        return f"seq_{op.lower()}_{emit}", []

    def _eval_join(self, join: Join) -> Tuple[str, List[Span]]:
        print(f"[JOIN] Evaluating join with {len(join.inputs)} inputs")
        input_results = [self._eval_node(node) for node in join.inputs]
        result_spans = []
        if join.condition.get("op", "").lower() == 'action':
            for inp in join.inputs:
                if join.condition.get("args", {}).get("actor") == inp.target_alias:
                    actor = inp.query_text
                if join.condition.get("args", {}).get("object") == inp.target_alias:
                    object = inp.query_text
            core_query = ("" if actor is None else actor) + join.condition.get('verb', "") +  ("" if object is None else object)
            question = "Is there scene that" + core_query + "in video? Give me the answer yes or no."

            for i, span in enumerate(input_results):
                answer = self.ctx.vlm.query(span.get('t0'), span.get('t1'), question)
                if 'yes' in answer[0].lower():
                    result_spans.append(span)
            return f"join_action", result_spans, core_query

        elif  join.condition.get("op", "").lower() == 'relation':
            for inp in join.inputs:
                if join.condition.get("args", "").get("left") == inp.target_alias:
                    left = inp.query_text
                if join.condition.get("args", "").get("right") == inp.target_alias:
                    right = inp.query_text
            core_query = ("" if left is None else left) + join.condition.get('type', "") +  ("" if right is None else right)
            question = "Is there scene that" + core_query + "in video? Give me the answer yes or no."

            for i, span in enumerate(input_results):
                answer = self.ctx.vlm.query(span.get('t0'), span.get('t1'), question)
                if 'yes' in answer[0].lower():
                    result_spans.append(span)
            return f"join_relation", result_spans, core_query

        else:
            result_key, result_spans, result_query = input_results[0]

            for i in range(1, len(input_results)):
                right_key, right_spans, right_query = input_results[i]
                print(f"[JOIN] Processing step: {len(result_spans)} spans × {len(right_spans)} spans")
                
                time_mode = (join.policy or {}).get("time_mode", "DURING")
                
                runner = ChatRunner(self.ctx, inspect_join)
                context = {"time_mode": time_mode, "join_node": join}
                res = runner.run(result_spans, right_spans, result_query, right_query, prompt_creator=_create_focused_prompt, **context)

                if res.get("success"):
                    result_spans = res["inspection_result"].get("result_spans", [])
                    result_key = f"{result_key}_{right_key}"
                    print(f"[JOIN] Step successful: {len(result_spans)} spans")
                else:
                    print(f"[JOIN] Step failed, returning empty result")
                    return f"join_failed_{result_key}", [], f"({result_query} {join.condition.get('op', '')} {right_query})"

            return f"join_{time_mode.lower()}_{result_key}", result_spans, f"({result_query} {join.condition.get('op', '')} {right_query})"

    def _eval_node(self, node: PlanNode) -> Tuple[str, List[Span]]:
        node_type = type(node).__name__
        print(f"[EVAL] Evaluating {node_type} node")
        
        if isinstance(node, Probe):
            return self._eval_probe(node)
        elif isinstance(node, Sequence):
            return self._eval_sequence(node)
        elif isinstance(node, Join):
            return self._eval_join(node)
        elif hasattr(node, 'inputs') and isinstance(node.inputs, list):
            return self._eval_generic_node(node)
        else:
            print(f"[EVAL] Unsupported node type: {node_type}")
            return "unsupported_node", []

    def _eval_sequence(self, sequence: Sequence) -> Tuple[str, List[Span]]:
        """2단계 Sequence를 평가하여 파이프라인 처리"""
        if len(sequence.steps) != 2:
            print(f"[SEQUENCE] Expected 2 steps, got {len(sequence.steps)}")
            return "sequence_invalid", []
        
        # 1. BEFORE/AFTER 특수 케이스 (2개 Probe)
        if all(isinstance(s, Probe) for s in sequence.steps):
            return self._eval_sequence_ba(sequence, sequence.steps[0], sequence.steps[1])
        
        # 첫 번째 단계 실행
        step1_key, step1_spans = self._eval_node(sequence.steps[0])
        
        if not step1_spans:
            print(f"[SEQUENCE] Step 1 produced no spans, returning empty result")
            return f"seq_{step1_key}_empty", []
        
        step2_key, step2_spans = self._eval_node(sequence.steps[1])
        
        if not step2_spans:
            print(f"[SEQUENCE] Step 2 produced no spans, returning step 1 result")
            return f"seq_{step1_key}_only", step1_spans
        
        # 두 단계 결과를 결합 (AND 로직: 시간적 겹침)
        combined_spans = []
        for span1 in step1_spans:
            for span2 in step2_spans:
                # 시간적 겹침이 있으면 결합
                if (span1['t1'] > span2['t0'] and span1['t0'] < span2['t1']):
                    combined_span = {
                        't0': max(span1['t0'], span2['t0']),
                        't1': min(span1['t1'], span2['t1']),
                        'score': min(span1.get('score', 0), span2.get('score', 0))
                    }
                    combined_spans.append(combined_span)
        
        print(f"[SEQUENCE] Combined result: {len(combined_spans)} overlapping spans")
        return f"seq_{step1_key}_{step2_key}", combined_spans

    def _eval_generic_node(self, node: PlanNode) -> Tuple[str, List[Span]]:
        all_spans = []
        input_keys = []
        for i, input_node in enumerate(node.inputs):
            key, spans = self._eval_node(input_node)
            all_spans.extend(spans)
            input_keys.append(key)
        result_spans = _dedup_spans(all_spans)
        result_key = f"{type(node).__name__.lower()}_{'_'.join(input_keys)}"
        return result_key, result_spans

    def run(self, ranked_plans: List[Tuple[int, PlanNode]]) -> Dict[str, Any]:
        exec_report = []
        for pidx, plan in ranked_plans:
            print(f"\n[TRYING] Plan {pidx}: {plan}")
            try:
                key, spans = self._eval_node(plan)
                exec_report.append({"plan_index": pidx, "key": key, "spans_found": len(spans)})
                if spans:
                    spans_sorted = sorted(spans, key=_conf, reverse=True)
                    return {"success": True, "plan_index": pidx, "plan": str(plan),
                            "answer_key": key, "spans": spans_sorted, "exec_report": exec_report}
            except Exception as e:
                print(f"[FATAL ERROR] in Plan {pidx}: {e}")
                traceback.print_exc()
                exec_report.append({"plan_index": pidx, "error": str(e)})
        return {"success": False, "reason": "No plan produced spans", "exec_report": exec_report}

# ---------------- CLI Demo ----------------
if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Plan runner with unified ChatRunner (BEFORE/AFTER + JOIN)")
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
history_manager.py - 실행 히스토리 관리 모듈

실행 과정에서 발생하는 변경사항들을 추적하고 요약하는 기능을 제공합니다.
"""

from typing import Dict, List, Any

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

def create_iteration_history(iter_num: int, focused_issue: Dict[str, Any], 
                           llm_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    새로운 반복의 히스토리 항목을 생성합니다.
    """
    return {
        "iter": iter_num,
        "focused_issue": focused_issue,
        "llm_actions": llm_actions,
        "changes": []  # 각 액션에서 일어난 변경사항들
    }

def finalize_iteration_history(iter_history: Dict[str, Any], A: List[Any], B: List[Any]) -> Dict[str, Any]:
    """
    반복 히스토리에 최종 상태를 추가합니다.
    """
    iter_history.update({
        "final_A_count": len(A),
        "final_B_count": len(B),
        "final_A_spans": [{"t0": s.get("t0"), "t1": s.get("t1"), "score": s.get("score")} for s in A],
        "final_B_spans": [{"t0": s.get("t0"), "t1": s.get("t1"), "score": s.get("score")} for s in B]
    })
    return iter_history

def add_action_changes(iter_history: Dict[str, Any], action_changes: Dict[str, Any]) -> None:
    """
    반복 히스토리에 액션 변경사항을 추가합니다.
    """
    iter_history["changes"].append(action_changes)

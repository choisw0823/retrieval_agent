#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plan_select.py

요약:
- 자연어(NL) 쿼리를 VTL/AST로 컴파일합니다.
- AST를 기반으로 여러 실행 계획(Plan)을 생성합니다.
- LLM(Gemini)을 이용해 생성된 계획들의 순위를 매기고 최적의 계획을 선택합니다.
- 선택된 PlanNode 객체 리스트를 반환합니다.

필수 의존:
  - pip install google-genai python-dotenv
  - 프로젝트 모듈:
      compiler.NL2VTLCompiler
      planner.PlanGenerator
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os, re, json
from dotenv import load_dotenv

from google import genai

# --- 프로젝트 모듈 임포트
from compiler import NL2VTLCompiler
from planner import PlanGenerator, PlanNode, Probe, Join, Sequence, Choice, Filter

load_dotenv()

# ---------------- Plan Ranking JSON client ----------------
class GeminiRankingJSON:
    def __init__(self, model: str = "gemini-2.5-pro"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.config = genai.types.GenerateContentConfig(response_mime_type="application/json")

    def generate(self, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        contents = [
            genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part.from_text(text=prompt),
                    genai.types.Part.from_text(text=json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
            ),
        ]
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=self.config)
        txt = resp.text.strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            if txt.startswith("json"): txt = txt[4:]
        m = re.search(r'\{.*\}\s*$', txt, re.DOTALL)
        if m: txt = m.group(0)
        return json.loads(txt)


# ---------------- LLM prompts ----------------
RANK_AND_REFINE_PROMPT = """
You are a video query planning specialist. Rank candidate plans for the user's query, optionally refining only the Probe query_texts. Do NOT invent new nodes or drop semantics.

Return ONLY JSON of a dictionary with a single key "ordered" which is a list of objects.
Each object must have:
- "plan_index": integer (0-based index of the original plan)
- "score": float (0.0-1.0, higher is better)
- "reason": string (brief explanation)
- "refined_plan": optional, only if you want to refine Probe query_texts

Rules:
1. The video is cooking video. So scene does not change(in kitchen) and only cooking related action happens.
2. You can ignore  words in query such as person, kitchen because all video scenes have person and kitchen. This is unnecessary. So you can delete such node
3. Focus on ingredient and action.
4. Do not make query complex. Make simple query.
5. Remove node like a person, in kitchen(relation)



Example:
{
  "ordered": [
    {"plan_index": 0, "score": 0.9, "reason": "Best temporal logic"},
    {"plan_index": 1, "score": 0.7, "reason": "Good but less precise"}
  ]
}
"""

def _dict_to_plan(d: Dict[str,Any]) -> PlanNode:
    nt = d.get("node_type")
    if nt == "Probe": return Probe(**{k:v for k,v in d.items() if k!='node_type'})
    if nt == "Join": return Join(inputs=[_dict_to_plan(x) for x in d.get("inputs",[])], **{k:v for k,v in d.items() if k not in ['node_type','inputs']})
    if nt == "Sequence": return Sequence(steps=[_dict_to_plan(x) for x in d.get("steps",[])], **{k:v for k,v in d.items() if k not in ['node_type','steps']})
    if nt == "Choice": return Choice(options=[_dict_to_plan(x) for x in d.get("options",[])], **{k:v for k,v in d.items() if k not in ['node_type','options']})
    if nt == "Filter": return Filter(input=_dict_to_plan(d["input"]), **{k:v for k,v in d.items() if k not in ['node_type','input']})
    return Probe(target_alias="q", query_text="unknown")


# ---------------- Plan Selector ----------------
class PlanSelector:
    def __init__(self, model: str = "gemini-2.5-pro"):
        self.model = model
        self.compiler = NL2VTLCompiler(model=self.model, expand_macro=True)
        self.ranker = GeminiRankingJSON(model=self.model)

    def _rank_plans(self, plans: List[PlanNode], query: str, top_k: int) -> List[Tuple[int, PlanNode]]:
        if not plans: return []
        #print(f"[DEBUG] Plans to rank: {plans}")
        try:
            plans_json = [p.to_dict() for p in plans]
            payload = {"QUERY": query, "CANDIDATE_PLANS": plans_json}
            rank_resp = self.ranker.generate(RANK_AND_REFINE_PROMPT, payload)
            #print(f"[DEBUG] Ranking response: {rank_resp}")
            
            ordered = rank_resp.get("ordered", [])
            if not ordered:
                print("[WARN] No 'ordered' key in ranking response. Using original order.")
                return [(i, plans[i]) for i in range(min(top_k, len(plans)))]
            
            ordered = sorted(ordered, key=lambda x: (-float(x.get("score", 0.0))))
            result = []
            
            for i, item in enumerate(ordered[:top_k]):
                pidx = int(item.get("plan_index", i))
                if not (0 <= pidx < len(plans)):
                    pidx = i % len(plans)
                
                refined = item.get("refined_plan")
                plan_obj = _dict_to_plan(refined) if isinstance(refined, dict) and refined.get("node_type") else plans[pidx]
                result.append((pidx, plan_obj))
            
            return result
        except Exception as e:
            print(f"[WARN] Plan ranking failed ({e}). Using original order.")
            return [(i, plans[i]) for i in range(min(top_k, len(plans)))]

    def select_plans(self, query: str, top_k: int = 1) -> List[Tuple[int, PlanNode]]:
        """Compiles, generates, and ranks plans for a given query."""
        comp = self.compiler.compile(query)
        ast = comp.ast
        #print(f"[DEBUG] AST formula: {ast}")
        if not isinstance(ast, dict):
            print("[WARN] No valid AST generated. Falling back to a simple Probe plan.")
            plans = [Probe(target_alias="holistic_action", query_text=query)]
        else:
            plans = PlanGenerator(ast).generate_plans()
            if not plans:
                print("[WARN] No plans generated from AST. Falling back to a simple Probe plan.")
                plans = [Probe(target_alias="holistic_action", query_text=query)]
        
        print(f"[RANKING] Found {len(plans)} plans. Ranking top {top_k}...")
        ranked_plans = self._rank_plans(plans, query, top_k)
        print(ranked_plans)
        return ranked_plans

# ---------------- CLI Demo ----------------
if __name__ == "__main__":
    query_text = "Find where the person gets the cauliflower and a knife from the kitchen before cutting the cauliflower."
    print(f"--- Testing PlanSelector with query: '{query_text}' ---")
    
    selector = PlanSelector()
    selected_plans = selector.select_plans(query_text, top_k=2)
    
    print("\n--- Selected Plans ---")
    if selected_plans:
        for p_idx, plan in selected_plans:
            print(f"Original Index: {p_idx}, Plan: {plan}")
    else:
        print("No plans were selected.")
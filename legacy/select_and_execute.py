#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_and_execute.py
- NL → VTL/AST(compiler) → PlanGenerator → Gemini 2.5 Plan Ranking/Refinement → Gemini Controller Loop → MR/VLM 실행

변경점(요청 반영):
  * 컨트롤러는 PLAN만 보고 동작 (별도 hints/expected_ops 없음)
  * 툴 파라미터(policy/eps/within/topk)는 LLM이 실행 중 동적으로 결정
  * 필요 시 sub-task로 추가 PROBE/검증 등을 자유롭게 호출 가능
  * 최대 스텝 제한 제거 (STOP을 LLM이 명시적으로 반환할 때까지 반복)

필수:
  pip install google-genai pydantic torch python-dotenv
  export GEMINI_API_KEY=...   # 또는 GOOGLE_API_KEY
옵션(VLM 사용 시):
  export REPLICATE_API_TOKEN=...

의존:
  - compiler.py (NL2VTLCompiler)
  - planner.py  (PlanNode/Probe/Join/Sequence/Choice/Filter, PlanGenerator)
  - mr_backend.py (MomentRetrievalMR)
  - vlm_backend.py (VLMBackend, join_with_vlm_verification)
"""

from __future__ import annotations
import os, json, argparse, traceback, re, itertools, logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

# ---------------- Project-local deps ----------------
from dotenv import load_dotenv
load_dotenv()

# ---------------- Logging setup ----------------
def setup_logging(log_file: str = None):
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"select_and_execute_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return log_file

def log_info(message: str):
    logging.info(message)

def log_error(message: str):
    logging.error(message)

def log_warning(message: str):
    logging.warning(message)

from compiler import NL2VTLCompiler
from planner import PlanGenerator, PlanNode, Probe, Join, Sequence, Choice, Filter
from mr_backend import MomentRetrievalMR, MRBackend
from vlm_backend import VLMBackend, join_with_vlm_verification

# ---------------- Gemini JSON client ----------------
def _get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def _parse_json_safely(txt: str) -> Dict[str, Any]:
    t = txt.strip()
    # ```json ...``` 형태 방어
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    # JSON 덩어리만 추출(혹시 앞뒤에 설명 붙어오면)
    m = re.search(r'\{.*\}\s*$', t, re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)

class GeminiJSON:
    def __init__(self, model: str = "gemini-2.5-pro"):
        from google import genai
        from google.genai import types
        api_key = _get_api_key()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model
        self.json_cfg = types.GenerateContentConfig(response_mime_type="application/json")

    def generate(self, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        full_prompt = prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)
        
        # Log LLM input state only
        log_info(f"LLM INPUT - State:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[self.types.Content(role="user",
                                         parts=[self.types.Part(text=full_prompt)])],
            config=self.json_cfg
        )
        
        result = _parse_json_safely(resp.text)
        
        # Log LLM output result only
        log_info(f"LLM OUTPUT - Result:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return result

# ---------------- Prompts ----------------

RANK_AND_REFINE_PROMPT = """
You are a video query planning specialist for temporal complex moment retrieval task. Rank candidate plans for the user's query, optionally refining only the Probe query_texts (fusion allowed) and tuning node policies minimally. Do NOT invent new nodes or drop required semantics.

Rules:
- If a Join merges simple tokens that actually describe a single action (e.g., "person opening a door"), fuse into one Probe with a more descriptive query_text.
- Preserve Join if the query implies co-occurrence (e.g., "person with a car").
- If the query implies ordering (after/then), prefer Sequence.
- Choose time_mode carefully (OVERLAPS, DURING, STARTS, FINISHES, EQUALS). For "after", prefer Sequence with condition {op:"BEFORE", within:[gap_min, gap_max] or null}.
- Tune policies gently (min_iou, min_dur, stitch_gap, eps, alpha).

Return ONLY JSON:
{
  "ordered": [
    {
      "plan_index": <int index into the provided plans array>,
      "score": <float 0..1>,
      "rationale": "<short reason>",
      "refined_plan": <optional PlanNode JSON>
    }
  ]
}
"""

CONTROLLER_PROMPT = """
You are the Planner-Executor Controller for video moment retrieval. Work ONLY from the PLAN and observations (BUFFERS). Propose exactly ONE next step as a tool call, or STOP. Return ONLY JSON.

Key principles:
- Respect PLAN structure and semantics. But you may add sub-steps (extra PROBE, alternate phrasings, windowed re-probes, verification) to maximize accuracy.
- Infer tool parameters yourself (policy/eps/within/topk). Do not rely on defaults; set explicit values as needed.
- Be robust to containment issues (e.g., B spans engulf A). You may probe either side first and switch order if results look wrong.
- Prefer a single fused PROBE for atomic actions. Use JOIN_* for co-occurrence. Use SEQUENCE_BEFORE for ordering. Use VLM_VERIFY when PLAN Join.condition op is ACTION/RELATION (fallback to JOIN_OVERLAPS if no VLM).
- Create intermediate buffers with unique short keys when helpful. Reuse buffers instead of re-running identical calls.
- Stop when a buffer clearly satisfies the PLAN semantics with good confidence, or when further steps won’t improve results.

PLAN → tool mapping rules:
- Probe → PROBE(key=normalized target_alias, query_text=probe.query_text, window=probe.temporal_window or null, topk from probe.hint.mr_topk if present or set an explicit value)
- Join.policy.time_mode:
  - OVERLAPS → JOIN_OVERLAPS(policy with min_iou/min_dur/stitch_gap/alpha/score_agg)
  - DURING/STARTS/FINISHES/EQUALS → JOIN_DURING/JOIN_STARTS/JOIN_FINISHES/JOIN_EQUALS (use eps if given)
  - ACTION/RELATION → VLM_VERIFY(verb or relation extracted from Join.condition). If VLM unavailable, fallback to JOIN_OVERLAPS(policy).
- Sequence.condition {op: BEFORE/MEETS/>>} → SEQUENCE_BEFORE(left=first step buffer, right_query from second step Probe or a refined phrase, within if provided). If second step is not a single Probe, assemble it via sub-steps (PROBE + JOIN) inside the derived window.

TOOLS:
- PROBE: run MR spans
  args: { "key": "<buf key>", "query_text": "<str>", "window": [t0,t1] or null, "topk": <int>, "hint": { ... } }
- JOIN_OVERLAPS:
  args: { "key": "<out>", "left": "<buf>", "right": "<buf>", "policy": { "min_iou": <f>, "min_dur": <f>, "stitch_gap": <f>, "alpha": <f>, "score_agg": "noisy_or|max|mean" } }
- JOIN_DURING | JOIN_STARTS | JOIN_FINISHES | JOIN_EQUALS:
  args: { "key": "<out>", "left": "<buf>", "right": "<buf>", "eps": <f> }
- SEQUENCE_BEFORE:
  args: { "key": "<out>", "left": "<buf A>", "right_query": "<query for B>", "within": [gap_min, gap_max] or null, "topk": <int> }
- VLM_VERIFY:
  args: { "key": "<out>", "left": "<buf>", "right": "<buf>", "relation": "<text>" or null, "verb": "<text>" or null, "policy": { ... } }
- UNION_AGG:
  args: { "key": "<out>", "inputs": ["k1","k2",...], "policy": { "overlap_iou": <f>, "stitch_gap": <f> } }
- FILTER_PASS:
  args: { "key": "<out>", "input": "<buf>" }

Return exactly one of:
- {"action_type":"RUN","operation":"<TOOL>","args":{...},"reason":"..."}
- {"action_type":"STOP","status":"SUCCESS|FAIL","answer_key":"<key|null>","reason":"..."}
"""

# ---------------- Span utilities & joins ----------------
Span = Dict[str, Any]
EPS_TIME = 0.5

def _geo_mean(vals: List[float]) -> float:
    import math
    vs = [max(1e-6, float(v)) for v in vals if v is not None]
    if not vs: return 0.0
    return float(math.exp(sum(math.log(v) for v in vs)/len(vs)))

def _iou_time(a: Span, b: Span) -> float:
    t0 = max(a["t0"], b["t0"]); t1 = min(a["t1"], b["t1"])
    if t1 <= t0: return 0.0
    inter = t1 - t0
    union = max(a["t1"], b["t1"]) - min(a["t0"], b["t0"])
    return inter/union if union>0 else 0.0

def _merge_overlaps(spans: List[Span], iou_thr: float, score_agg: str = "noisy_or") -> List[Span]:
    if not spans: return []
    spans = sorted(spans, key=lambda s: (s["t0"], s["t1"]))
    out: List[Span] = []; cur = spans[0]
    for s in spans[1:]:
        if _iou_time(cur, s) >= iou_thr:
            cur_t0 = min(cur["t0"], s["t0"]); cur_t1 = max(cur["t1"], s["t1"])
            a = cur.get("conf", cur.get("score", 1.0)); b = s.get("conf", s.get("score", 1.0))
            if score_agg == "max": sc = max(a, b)
            elif score_agg == "mean": sc = (a+b)/2
            else: sc = 1 - (1-a)*(1-b)
            cur = {**cur, "t0": cur_t0, "t1": cur_t1, "score": sc}
        else:
            out.append(cur); cur = s
    out.append(cur)
    return out

def _stitch_by_gap(spans: List[Span], max_gap: float) -> List[Span]:
    if not spans or max_gap <= 0: return spans
    spans = sorted(spans, key=lambda s: (s["t0"], s["t1"]))
    out: List[Span] = []; cur = spans[0]
    for s in spans[1:]:
        gap = s["t0"] - cur["t1"]
        if 0 <= gap <= max_gap:
            sc = (cur.get("conf", cur.get("score", 1.0)) + s.get("conf", s.get("score", 1.0)))/2
            cur = {**cur, "t1": s["t1"], "score": sc}
        else:
            out.append(cur); cur = s
    out.append(cur)
    return out

def intersection_join(A: List[Span], B: List[Span], policy: Dict[str, Any], alpha_default: float = 0.5) -> List[Span]:
    A = sorted(A, key=lambda s: (s["t0"], s["t1"]))
    B = sorted(B, key=lambda s: (s["t0"], s["t1"]))
    i = j = 0; out: List[Span] = []
    min_dur = float(policy.get("min_dur", 0.0))
    min_iou = float(policy.get("min_iou", 0.0))
    alpha = float(policy.get("alpha", alpha_default))
    score_agg = policy.get("score_agg", "noisy_or")
    while i < len(A) and j < len(B):
        a, b = A[i], B[j]
        t0 = max(a["t0"], b["t0"]); t1 = min(a["t1"], b["t1"])
        if t1 > t0:
            dur = t1 - t0
            union = max(a["t1"], b["t1"]) - min(a["t0"], b["t0"])
            iou = dur/union if union>0 else 0.0
            if dur >= min_dur and iou >= min_iou:
                base = _geo_mean([a.get("conf", a.get("score",1.0)), b.get("conf", b.get("score",1.0))])
                quality = alpha*iou + (1-alpha)
                if score_agg == "max": base = max(a.get("conf",1.0), b.get("conf",1.0))
                elif score_agg == "mean": base = (a.get("conf",1.0)+b.get("conf",1.0))/2
                out.append({"t0": t0, "t1": t1, "score": base*quality, "tags": {"left": a.get("tag"), "right": b.get("tag")}})
        if a["t1"] <= b["t1"]: i += 1
        else: j += 1
    g = float(policy.get("stitch_gap", 0.0))
    return _stitch_by_gap(out, g)

def union_aggregate(branches: List[List[Span]], overlap_iou: float, stitch_gap: float, max_out: int) -> List[Span]:
    pool: List[Span] = []
    for b in branches: pool.extend(b)
    merged = _merge_overlaps(pool, overlap_iou)
    return _stitch_by_gap(merged, stitch_gap)[:max_out]

def join_during(A: List[Span], B: List[Span], eps: float = EPS_TIME) -> List[Span]:
    A = sorted(A, key=lambda s: (s["t0"], s["t1"]))
    B = sorted(B, key=lambda s: (s["t0"], s["t1"]))
    i = j = 0; out: List[Span] = []
    while i < len(A) and j < len(B):
        a, b = A[i], B[j]
        if a["t0"] >= b["t0"] - eps and a["t1"] <= b["t1"] + eps:
            out.append({**a}); i += 1
        else:
            if a["t1"] < b["t0"]: i += 1
            elif b["t1"] < a["t0"]: j += 1
            else:
                if a["t1"] <= b["t1"]: i += 1
                else: j += 1
    return out

def join_starts(A: List[Span], B: List[Span], eps: float = EPS_TIME) -> List[Span]:
    A = sorted(A, key=lambda s: (s["t0"], s["t1"]))
    B = sorted(B, key=lambda s: (s["t0"], s["t1"]))
    i = j = 0; out: List[Span] = []
    while i < len(A) and j < len(B):
        a, b = A[i], B[j]
        if abs(a["t0"] - b["t0"]) <= eps:
            t0, t1 = max(a["t0"], b["t0"]), min(a["t1"], b["t1"])
            if t1 > t0:
                out.append({"t0": t0, "t1": t1, "score": _geo_mean([a.get("conf", a.get("score",1.0)), b.get("conf", b.get("score",1.0))])})
            if a["t1"] <= b["t1"]: i += 1
            else: j += 1
        elif a["t0"] < b["t0"] - eps: i += 1
        else: j += 1
    return out

def join_finishes(A: List[Span], B: List[Span], eps: float = EPS_TIME) -> List[Span]:
    A = sorted(A, key=lambda s: (s["t1"], s["t0"]))
    B = sorted(B, key=lambda s: (s["t1"], s["t0"]))
    i = j = 0; out: List[Span] = []
    while i < len(A) and j < len(B):
        a, b = A[i], B[j]
        if abs(a["t1"] - b["t1"]) <= eps:
            t0, t1 = max(a["t0"], b["t0"]), min(a["t1"], b["t1"])
            if t1 > t0:
                out.append({"t0": t0, "t1": t1, "score": _geo_mean([a.get("conf", a.get("score",1.0)), b.get("conf", b.get("score",1.0))])})
            if a["t0"] <= b["t0"]: i += 1
            else: j += 1
        elif a["t1"] < b["t1"] - eps: i += 1
        else: j += 1
    return out

def join_equals(A: List[Span], B: List[Span], eps: float = EPS_TIME) -> List[Span]:
    out: List[Span] = []
    for a in A:
        for b in B:
            if abs(a["t0"] - b["t0"]) <= eps and abs(a["t1"] - b["t1"]) <= eps:
                out.append({**a})
    return out

def derive_window_from_left(left: Span, within: Optional[List[float]], video_end: Optional[float]) -> Optional[List[float]]:
    start = left["t1"]
    if within and len(within) == 2:
        return [start + float(within[0]), start + float(within[1])]
    horizon = video_end if (video_end and video_end > start) else start + 60.0
    return [start, horizon]

# ---------------- TOOLS spec & buffer summary ----------------
SPAN_SCHEMA = {
    "t0": "float>=0", "t1": "float>t0",
    "score?": "float in [0,1]", "conf?": "float in [0,1]", "tag?": "str"
}

TOOLS_SPEC = {
    "PROBE": {
        "desc": "Run Moment Retrieval on a query, optionally within a time window.",
        "args": {
            "key":        {"type":"str",  "required":True,  "pattern": r"^[a-z0-9_]{1,32}$", "desc":"output buffer"},
            "query_text": {"type":"str",  "required":True,  "max_len":256, "desc":"natural-language or fused phrase"},
            "window":     {"type":"[float,float]|null", "required":False, "units":"sec",
                           "desc":"[t0,t1], inclusive start; ignored if null"},
            "topk":       {"type":"int",  "required":False, "default":5, "range":[1,512]},
            "hint":       {"type":"obj",  "required":False, "desc":"backend-specific hints (e.g., {\"mr_topk\":64})"}
        },
        "returns": {"buffer":"key", "spans":"List[Span] per SPAN_SCHEMA"},
        "pre":  ["video loaded"],
        "post": ["buffers[key] overwritten with at most topk spans"],
        "failures": ["EMPTY_QUERY","ARG_RANGE","WINDOW_INVALID","BACKEND_ERROR"],
        "example": {"key":"door", "query_text":"person opening a door", "window":None, "topk":64}
    },
    "JOIN_OVERLAPS": {
        "desc": "Temporal intersection join with IoU/min_dur and optional stitching.",
        "args": {
            "key":    {"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
            "left":   {"type":"str","required":True},
            "right":  {"type":"str","required":True},
            "policy": {"type":"obj","required":True,"props":{
                "min_iou":{"type":"float","default":0.0,"range":[0.0,1.0]},
                "min_dur":{"type":"float","default":0.0,"units":"sec","range":[0.0, 1e9]},
                "stitch_gap":{"type":"float","default":0.0,"units":"sec","range":[0.0, 10.0]},
                "alpha":{"type":"float","default":0.5,"range":[0.0,1.0]},
                "score_agg":{"type":"enum","values":["noisy_or","max","mean"],"default":"noisy_or"}
            }}
        },
        "returns":{"buffer":"key","spans":"intersection spans"},
        "pre":["buffers[left], buffers[right] may be empty"],
        "post":["buffers[key] size ≤ max_spans_per_node"],
        "failures":["KEY_NOT_FOUND","ARG_RANGE"]
    },
    "JOIN_DURING": {
        "desc":"Keep A spans that lie DURING B spans (with eps).",
        "args":{
            "key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
            "left":{"type":"str","required":True},
            "right":{"type":"str","required":True},
            "eps":{"type":"float","default":0.5,"units":"sec","range":[0.0, 5.0]}
        },
        "returns":{"buffer":"key"}, "failures":["KEY_NOT_FOUND","ARG_RANGE"]
    },
    "JOIN_STARTS": {
        "desc":"Keep overlaps where start times match within eps.",
        "args":{"key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
                "left":{"type":"str","required":True},"right":{"type":"str","required":True},
                "eps":{"type":"float","default":0.5,"units":"sec","range":[0.0,5.0]}},
        "returns":{"buffer":"key"}, "failures":["KEY_NOT_FOUND","ARG_RANGE"]
    },
    "JOIN_FINISHES": {
        "desc":"Keep overlaps where end times match within eps.",
        "args":{"key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
                "left":{"type":"str","required":True},"right":{"type":"str","required":True},
                "eps":{"type":"float","default":0.5,"units":"sec","range":[0.0,5.0]}},
        "returns":{"buffer":"key"}, "failures":["KEY_NOT_FOUND","ARG_RANGE"]
    },
    "JOIN_EQUALS": {
        "desc":"Keep spans equal in [t0,t1] within eps.",
        "args":{"key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
                "left":{"type":"str","required":True},"right":{"type":"str","required":True},
                "eps":{"type":"float","default":0.5,"units":"sec","range":[0.0,5.0]}},
        "returns":{"buffer":"key"}, "failures":["KEY_NOT_FOUND","ARG_RANGE"]
    },
    "SEQUENCE_BEFORE": {
        "desc":"For each span in A, search B after A (optionally within [gap_min,gap_max]).",
        "args":{
            "key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
            "left":{"type":"str","required":True},
            "right_query":{"type":"str","required":True,"max_len":256},
            "within":{"type":"[float,float]|null","required":False,"units":"sec",
                      "desc":"allowed gap after A.t1 to search B; null = [0, horizon]"},
            "topk":{"type":"int","required":False,"default":64,"range":[1,512]}
        },
        "returns":{"buffer":"key"},
        "notes":["Scores penalized by gap via 1/(1+gap). Window derived per left span."],
        "failures":["KEY_NOT_FOUND","ARG_RANGE","BACKEND_ERROR"]
    },
    "VLM_VERIFY": {
        "desc":"Verify relation/action between A and B using VLM; if VLM absent, fallback to JOIN_OVERLAPS policy.",
        "args":{
            "key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
            "left":{"type":"str","required":True},
            "right":{"type":"str","required":True},
            "relation":{"type":"str|null","required":False},
            "verb":{"type":"str|null","required":False},
            "policy":{"type":"obj","required":False,"desc":"used in overlap fallback and scoring"}
        },
        "returns":{"buffer":"key"},
        "failures":["KEY_NOT_FOUND","VLM_UNAVAILABLE","BACKEND_ERROR"]
    },
    "UNION_AGG": {
        "desc":"Merge multiple buffers then overlap-merge and stitch.",
        "args":{
            "key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
            "inputs":{"type":"[str,...]","required":True,"min_len":1},
            "policy":{"type":"obj","required":True,"props":{
                "overlap_iou":{"type":"float","default":0.3,"range":[0.0,1.0]},
                "stitch_gap":{"type":"float","default":0.2,"units":"sec","range":[0.0,10.0]}
            }}
        },
        "returns":{"buffer":"key"},
        "failures":["KEY_NOT_FOUND","ARG_RANGE"]
    },
    "FILTER_PASS": {
        "desc":"Copy input buffer to key (no filtering).",
        "args":{"key":{"type":"str","required":True,"pattern": r"^[a-z0-9_]{1,32}$"},
                "input":{"type":"str","required":True}},
        "returns":{"buffer":"key"},
        "failures":["KEY_NOT_FOUND"]
    }
}


def summarize_spans(spans: List[Span], sample_k: int = 5) -> Dict[str, Any]:
    if not spans:
        return {"count": 0, "top1_score": 0.0, "mean_score": 0.0, "coverage": 0.0, "samples": []}
    s_sorted = sorted(spans, key=lambda s: s.get("score", s.get("conf", 0.0)), reverse=True)
    scores = [s.get("score", s.get("conf", 0.0)) for s in spans]
    mean_s = sum(scores)/len(scores) if scores else 0.0
    duration = sum(max(0.0, s["t1"]-s["t0"]) for s in spans)
    return {
        "count": len(spans),
        "top1_score": s_sorted[0].get("score", s_sorted[0].get("conf", 0.0)),
        "mean_score": mean_s,
        "coverage": duration,
        "samples": [{"t0": s["t0"], "t1": s["t1"], "score": s.get("score", s.get("conf", 0.0))} for s in s_sorted[:sample_k]]
    }

# ---------------- OP implementations ----------------
@dataclass
class ExecConfig:
    topk_default: int = 64
    union_overlap_iou: float = 0.3
    stitch_gap: float = 0.2
    max_spans_per_node: int = 128

def op_PROBE(buffers: Dict[str, List[Span]], args: Dict, mr: MRBackend, cfg: ExecConfig):
    key = args["key"]; q = args["query_text"]; win = args.get("window")
    topk = int(args.get("topk", cfg.topk_default)); hint = args.get("hint", {}) or {}
    spans = mr.fetch(q, win, topk, hint)[:topk]
    buffers[key] = spans

def op_JOIN_OVERLAPS(buffers: Dict[str, List[Span]], args: Dict, cfg: ExecConfig):
    key=args["key"]; left=args["left"]; right=args["right"]; policy=args.get("policy",{}) or {}
    A=buffers.get(left,[]); B=buffers.get(right,[])
    buffers[key]=intersection_join(A,B,policy)[:cfg.max_spans_per_node]

def _edge_join(buffers: Dict[str, List[Span]], args: Dict, mode: str, cfg: ExecConfig):
    key=args["key"]; left=args["left"]; right=args["right"]; eps=float(args.get("eps", EPS_TIME))
    A=buffers.get(left,[]); B=buffers.get(right,[])
    if mode=="DURING": spans=join_during(A,B,eps=eps)
    elif mode=="STARTS": spans=join_starts(A,B,eps=eps)
    elif mode=="FINISHES": spans=join_finishes(A,B,eps=eps)
    elif mode=="EQUALS": spans=join_equals(A,B,eps=eps)
    else: spans=[]
    buffers[key]=spans[:cfg.max_spans_per_node]

def op_JOIN_DURING(buffers,args,cfg):   _edge_join(buffers,args,"DURING",cfg)

def op_JOIN_STARTS(buffers,args,cfg):   _edge_join(buffers,args,"STARTS",cfg)

def op_JOIN_FINISHES(buffers,args,cfg): _edge_join(buffers,args,"FINISHES",cfg)

def op_JOIN_EQUALS(buffers,args,cfg):   _edge_join(buffers,args,"EQUALS",cfg)

def op_SEQUENCE_BEFORE(buffers: Dict[str, List[Span]], args: Dict, mr: MRBackend, video_end: Optional[float], cfg: ExecConfig):
    key=args["key"]; left=args["left"]; right_q=args["right_query"]; within=args.get("within")
    topk=int(args.get("topk", cfg.topk_default))
    out: List[Span]=[]
    for a in buffers.get(left, []):
        win=derive_window_from_left(a, within, video_end)
        if win is None: continue
        step_spans=mr.fetch(right_q, win, topk, {})
        for b in step_spans:
            gap=max(0.0, b["t0"]-a["t1"])
            sc=_geo_mean([a.get("conf", a.get("score",1.0)), b.get("conf", b.get("score",1.0))])*(1.0/(1.0+gap))
            out.append({"t0":min(a["t0"],b["t0"]), "t1":max(a["t1"],b["t1"]), "score": sc, "conf": sc})
    buffers[key]=out[:cfg.max_spans_per_node]

def op_VLM_VERIFY(buffers: Dict[str, List[Span]], args: Dict, vlm: Optional[VLMBackend], cfg: ExecConfig):
    key=args["key"]; left=args["left"]; right=args["right"]
    relation=args.get("relation"); verb=args.get("verb"); policy=args.get("policy",{}) or {}
    A=buffers.get(left,[]); B=buffers.get(right,[])
    if not vlm:
        buffers[key]=intersection_join(A,B,policy)[:cfg.max_spans_per_node]; return
    # vlm_backend의 API 특성상 Join 노드의 condition을 사용하므로 dummy join 생성
    cond: Dict[str,Any] = {"op":"RELATION","type":"related_to"}
    if verb: cond={"op":"ACTION","verb":verb}
    elif relation: cond={"op":"RELATION","type":relation}
    dummy_join = Join(inputs=[], condition=cond)
    spans = join_with_vlm_verification(A,B,dummy_join,vlm)
    buffers[key]=spans[:cfg.max_spans_per_node]

def op_UNION_AGG(buffers: Dict[str, List[Span]], args: Dict, cfg: ExecConfig):
    key=args["key"]; inputs=args.get("inputs",[]); pol=args.get("policy",{}) or {}
    ov=float(pol.get("overlap_iou", 0.3)); sg=float(pol.get("stitch_gap", 0.2))
    branches=[buffers.get(k,[]) for k in inputs]
    buffers[key]=union_aggregate(branches, ov, sg, cfg.max_spans_per_node)

def op_FILTER_PASS(buffers: Dict[str, List[Span]], args: Dict, cfg: ExecConfig):
    buffers[args["key"]] = list(buffers.get(args["input"], []))

OP_LIBRARY = {
    "PROBE": lambda b,a,ctx: op_PROBE(b,a,ctx["mr"],ctx["cfg"]),
    "JOIN_OVERLAPS": lambda b,a,ctx: op_JOIN_OVERLAPS(b,a,ctx["cfg"]),
    "JOIN_DURING": lambda b,a,ctx: op_JOIN_DURING(b,a,ctx["cfg"]),
    "JOIN_STARTS": lambda b,a,ctx: op_JOIN_STARTS(b,a,ctx["cfg"]),
    "JOIN_FINISHES": lambda b,a,ctx: op_JOIN_FINISHES(b,a,ctx["cfg"]),
    "JOIN_EQUALS": lambda b,a,ctx: op_JOIN_EQUALS(b,a,ctx["cfg"]),
    "SEQUENCE_BEFORE": lambda b,a,ctx: op_SEQUENCE_BEFORE(b,a,ctx["mr"],ctx.get("video_end"),ctx["cfg"]),
    "VLM_VERIFY": lambda b,a,ctx: op_VLM_VERIFY(b,a,ctx.get("vlm"),ctx["cfg"]),
    "UNION_AGG": lambda b,a,ctx: op_UNION_AGG(b,a,ctx["cfg"]),
    "FILTER_PASS": lambda b,a,ctx: op_FILTER_PASS(b,a,ctx["cfg"]),
}

# ---------------- Controller Loop ----------------
class ControllerExecutor:
    def __init__(self, mr: MRBackend, vlm: Optional[VLMBackend], model: str = "gemini-2.5-pro", cfg: Optional[ExecConfig]=None):
        self.mr = mr
        self.vlm = vlm
        self.cfg = cfg or ExecConfig()
        self.g = GeminiJSON(model=model)

    def _buffers_summary(self, buffers: Dict[str, List[Span]]) -> Dict[str, Any]:
        return {k: summarize_spans(v) for k,v in buffers.items()}

    def run(self, nl_query: str, plan: PlanNode, video_duration: Optional[float] = None) -> Dict[str, Any]:
        buffers: Dict[str, List[Span]] = {}
        history: List[Dict[str, Any]] = []

        for step in itertools.count(1):  # no max step; stop when LLM says STOP
            state = {
                "QUERY": nl_query,
                "PLAN": plan.to_dict(),
                "BUFFERS": self._buffers_summary(buffers),
                "CAPABILITIES": {"has_vlm": self.vlm is not None, "video_duration": video_duration},
                "TOOLS_SPEC": TOOLS_SPEC
            }
            
            try:
                action = self.g.generate(CONTROLLER_PROMPT, state)
            except Exception as e:
                print(f"[CONTROLLER STEP {step}] LLM Error: {e}")
                history.append({"step": step, "error": f"LLM error: {e}"})
                break

            history.append({"step": step, "action": action})

            at = action.get("action_type")
            if at == "STOP":
                key = action.get("answer_key")
                return {
                    "success": (action.get("status") == "SUCCESS" and key in buffers and len(buffers[key])>0),
                    "answer_key": key,
                    "spans": buffers.get(key, []),
                    "history": history,
                    "buffers": buffers
                }

            if at != "RUN":
                history[-1]["note"] = f"Non-RUN action '{at}' ignored."
                continue

            op = action.get("operation"); args = action.get("args", {}) or {}
            if op not in OP_LIBRARY:
                history[-1]["error"] = f"Unknown operation: {op}"
                continue

            try:
                OP_LIBRARY[op](buffers, args, {"mr": self.mr, "vlm": self.vlm, "cfg": self.cfg, "video_end": video_duration})
            except Exception as e:
                history[-1]["error"] = f"Operator '{op}' failed: {e}"

        return {"success": False, "answer_key": None, "spans": [], "history": history, "buffers": buffers}

# ---------------- Helpers ----------------
def dict_to_plan(d: Dict[str,Any]) -> PlanNode:
    nt = d.get("node_type")
    if nt == "Probe":
        return Probe(target_alias=d["target_alias"], query_text=d["query_text"],
                     temporal_window=d.get("temporal_window"), hint=d.get("hint",{}))
    if nt == "Join":
        inputs=[dict_to_plan(x) for x in d.get("inputs",[])]
        return Join(inputs=inputs, condition=d.get("condition",{}), policy=d.get("policy",{}))
    if nt == "Sequence":
        steps=[dict_to_plan(x) for x in d.get("steps",[])]
        return Sequence(steps=steps, condition=d.get("condition",{}))
    if nt == "Choice":
        options=[dict_to_plan(x) for x in d.get("options",[])]
        return Choice(options=options, union_policy=d.get("union_policy",{}))
    if nt == "Filter":
        return Filter(input=dict_to_plan(d["input"]), condition=d.get("condition",{}))
    # fallback
    return Probe(target_alias="q", query_text="unknown")

def _ffprobe_duration(video_path: str) -> Optional[float]:
    try:
        import subprocess, json as _json
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","json",video_path],
            stderr=subprocess.STDOUT
        ).decode("utf-8","ignore")
        d = _json.loads(out)
        return float(d["format"]["duration"])
    except Exception:
        return None

# ---------------- Main Orchestration ----------------
def main():
    ap = argparse.ArgumentParser(description="NL→VTL/AST + Plan Ranking + Controller Execution (Gemini 2.5)")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--top_plans", type=int, default=3)
    ap.add_argument("--show_vtl", action="store_true")
    ap.add_argument("--log_file", type=str, help="Log file path (default: auto-generated)")
    args = ap.parse_args()

    # Setup logging
    log_file = setup_logging(args.log_file)
    print(f"Logging initialized. Log file: {log_file}")

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return

    # 1) NL → VTL/AST
    print(f"\n[1/5] Compiling NL → VTL/AST ...")
    compiler = NL2VTLCompiler(model=args.model, expand_macro=True)
    comp = compiler.compile(args.query)
    if comp.errors:
        print("[WARN] Compiler reported errors:")
        for e in comp.errors: print("  -", e)
    if args.show_vtl:
        print("\n[VTL]")
        print(comp.vtl)
    ast = comp.ast
    if not isinstance(ast, dict) or "formula" not in ast:
        print("[ERROR] Compilation produced no valid AST.")
        return

    # 2) Generate candidate plans
    print(f"\n[2/5] Generating candidate plans ...")
    plans = PlanGenerator(ast).generate_plans()
    if not plans:
        print("[WARN] No plans from planner. Falling back to single Probe.")
        plans = [Probe(target_alias="q", query_text=args.query, hint={"mr_topk": 5})]
    print(f"Plans: {plans}")
    
    # 3) Rank plans with Gemini 2.5 (and optional refinement)
    print(f"\n[3/5] Ranking plans with Gemini 2.5 ...")
    ranker = GeminiJSON(model=args.model)
    plans_json = [p.to_dict() for p in plans]
    payload = {"QUERY": args.query, "CANDIDATE_PLANS": plans_json}
    
    try:
        rank_resp = ranker.generate(RANK_AND_REFINE_PROMPT, payload)
        ordered = sorted(rank_resp.get("ordered", []), key=lambda x: (-float(x.get("score", 0.0))))
    except Exception as e:
        print(f"[WARN] Plan ranking failed ({e}). Using original order.")
        ordered = [{"plan_index": i, "score": 0.5, "rationale": "fallback"} for i in range(len(plans))]

    print("  Top indices:", [it["plan_index"] for it in ordered[:args.top_plans]])
    print("  Top plans:", ordered[:args.top_plans])

    # 4) Init MR/VLM
    print(f"\n[4/5] Initializing MR/VLM ...")
    mr = MomentRetrievalMR(video_path=args.video, device=args.device)
    try:
        vlm = VLMBackend(video_path=args.video)
        print("  [VLM] enabled")
    except Exception as e:
        vlm = None
        print(f"  [VLM] disabled ({e})")
    video_dur = _ffprobe_duration(args.video)
    if video_dur: print(f"  [VIDEO] duration ≈ {video_dur:.2f}s")

    # 5) Iterative execution per ranked plan
    print(f"\n[5/5] Executing ranked plans with Controller ...")
    exec_cfg = ExecConfig()
    controller = ControllerExecutor(mr, vlm, model=args.model, cfg=exec_cfg)

    final = None
    tried = 0
    for item in ordered:
        if tried >= args.top_plans: break
        tried += 1
        pidx = int(item["plan_index"])
        refined = item.get("refined_plan")
        plan_obj = dict_to_plan(refined) if isinstance(refined, dict) else plans[pidx]
        print(f"\n[TRY] plan_index={pidx}, score={float(item.get('score',0.0)):.2f}, rationale={item.get('rationale','')}")
        res = controller.run(args.query, plan_obj, video_duration=video_dur)
        if res.get("success") and res.get("spans"):
            final = (pidx, res)
            break

    if not final:
        print("\n[RESULT] No successful plan execution.")
        return

    pidx, res = final
    spans = sorted(res["spans"], key=lambda s: s.get("score", s.get("conf", 0.0)), reverse=True)
    print(f"\n[RESULT] SUCCESS with plan {pidx}, spans={len(spans)}")
    for i, s in enumerate(spans[:10], 1):
        print(f"  {i:02d}. t0={s.get('t0',0):.2f}s  t1={s.get('t1',0):.2f}s  score={s.get('score', s.get('conf',0.0)):.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()

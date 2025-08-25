# planner.py
# VTL Planner: span-standardized planning for ACTION/RELATION + Allen temporal relations
# Supports: AND/OR/NOT, n-ary AND/OR, within[a,b] pushdown hints, macro >>, MEETS gap, AFTER order fix
# Produces candidate Plan Trees (dedup-only). Now encodes operator semantics into Join.policy["time_mode"].
# UPDATED: Sequence.emit support + AST Operator.emit/anchor ingestion

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json

# ==============================================================================
# 0) Utilities
# ==============================================================================

_VALID_EMIT = {"left", "right", "union", "pair"}

def _normalize_emit(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    v = str(v).strip().lower()
    return v if v in _VALID_EMIT else None

def _swap_emit_lr(v: Optional[str]) -> Optional[str]:
    if v == "left":
        return "right"
    if v == "right":
        return "left"
    return v  # union/pair/None 그대로


# ==============================================================================
# 1) Plan Tree Nodes
# ==============================================================================

@dataclass
class PlanNode:
    node_type: str = field(init=False)

    def __post_init__(self):
        self.node_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        def _encode(x):
            if hasattr(x, 'to_dict'):
                return x.to_dict()
            if isinstance(x, list):
                return [_encode(i) for i in x]
            return x
        d = asdict(self)
        d['node_type'] = self.node_type
        for k, v in list(d.items()):
            d[k] = _encode(v)
        return d

    def canonical_key(self) -> str:
        # Stable key for deduplication
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


@dataclass
class Probe(PlanNode):
    target_alias: str
    query_text: str
    temporal_window: Optional[List[float]] = None  # [t_start, t_end], hint for executor
    hint: Dict[str, Any] = field(default_factory=dict)  # e.g., {"mr_topk": 5}


@dataclass
class Join(PlanNode):
    inputs: List[PlanNode]
    condition: Dict[str, Any]
    policy: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "intersection",     # "intersection" | "coexist"
        "time_mode": "OVERLAPS",
        "relation_type": None,
    })


@dataclass
class Sequence(PlanNode):
    steps: List[PlanNode]
    condition: Dict[str, Any]  # {op: BEFORE/AFTER/MEETS/>>, within:[a,b], gap:0, ...}
    emit: str = "union"

    def __post_init__(self):
        super().__post_init__()
        self.emit = _normalize_emit(self.emit) or "union"


@dataclass
class Choice(PlanNode):  # OR
    options: List[PlanNode]
    union_policy: Dict[str, Any] = field(default_factory=lambda: {
        "merge_overlaps": True, "stitch_gap": 0.0, "topk_per_branch": None,
        "topk_global": None, "score_agg": "noisy_or", "overlap_iou": 0.3,
    })


@dataclass
class Filter(PlanNode):  # NOT / post filter
    input: PlanNode
    condition: Dict[str, Any]  # {op: "NOT"} or any predicate


@dataclass
class KOfN(PlanNode):  # m-of-n (OR generalization)
    children: List[PlanNode]
    m: int


# ==============================================================================
# 2) Plan Generator
# ==============================================================================

class PlanGenerator:
    def __init__(self, ast: Dict[str, Any], beam_k: int = 16):
        self.ast = ast
        self.beam_k = max(1, beam_k)
        self.alias_to_label: Dict[str, str] = {
            decl.get('name', ''): (decl.get('object', {}) or {}).get('label', 'unknown')
            for decl in ast.get('declarations', [])
        }

    def generate_plans(self) -> List[PlanNode]:
        if 'formula' not in self.ast:
            return []
        
        #print(f"[DEBUG] AST formula: {self.ast['formula']}")
        
        plans = self._generate_from_node(self.ast['formula'])
        print(f"[DEBUG] Generated {len(plans)} plans")
        return self._dedup(plans)

    def _generate_from_node(self, node: Dict[str, Any]) -> List[PlanNode]:
        if not node or not isinstance(node, dict) or 'type' not in node:
            return []

        node_type = node.get('type')

        if node_type == 'Operator':
            return self._from_operator(node)
        elif node_type == 'Action':
            return self._from_action(node)
        elif node_type == 'Relation':
            return self._from_relation(node)
        # --- 추가된 부분 ---
        elif node_type == 'Reference':
            return self._from_reference(node)
        # --- 여기까지 ---
        else:
            return []

    def _from_operator(self, node: Dict[str, Any]) -> List[PlanNode]:
        op = node.get('op', '').upper()
        operands = node.get('operands', [])
        within = node.get('within')
        emit_in = _normalize_emit(node.get('emit') or node.get('anchor'))
        emit_final = emit_in or "union"

        if op == 'NOT' and len(operands) == 1:
            sub_plans = self._generate_from_node(operands[0])
            return [Filter(input=p, condition={'op': 'NOT'}) for p in sub_plans]

        if op == 'OR':
            option_plans: List[PlanNode] = []
            for ch in operands:
                option_plans.extend(self._generate_from_node(ch))
            return [Choice(options=self._dedup(option_plans))]

        seq_ops = {"BEFORE", "AFTER", "MEETS", ">>"}
        if op in seq_ops:
            if len(operands) != 2: return []
            left_plans = self._generate_from_node(operands[0])
            right_plans = self._generate_from_node(operands[1])
            
            # If either side fails to produce a plan, the sequence is impossible
            if not left_plans or not right_plans:
                return []

            combined: List[PlanNode] = []
            for l in left_plans:
                for r in right_plans:
                    cond = {'op': 'BEFORE'}
                    if within is not None: cond['within'] = within
                    if op == 'MEETS': cond['gap'] = 0
                    if op == '>>': cond['macro'] = 'END(A) BEFORE START(B)'
                    
                    if op == 'AFTER':
                        steps = [r, l]
                        e = _swap_emit_lr(emit_final)
                        cond['source_op'] = 'AFTER'
                        combined.append(Sequence(steps=steps, condition=cond, emit=e or "union"))
                    else:
                        steps = [l, r]
                        combined.append(Sequence(steps=steps, condition=cond, emit=emit_final))
            return self._dedup(combined)

        allen = {"OVERLAPS", "DURING", "STARTS", "FINISHES", "EQUALS"}
        if op in allen:
            if len(operands) != 2: return []
            left_plans = self._generate_from_node(operands[0])
            right_plans = self._generate_from_node(operands[1])
            if not left_plans or not right_plans: return []
            out: List[PlanNode] = []
            for l in left_plans:
                for r in right_plans:
                    pol = self._default_join_policy(mode='intersection')
                    pol['time_mode'] = op
                    out.append(Join(inputs=[l, r], condition={'op':op, 'within':within}, policy=pol))
            return self._dedup(out)

        if op == 'AND' and len(operands) >= 2:
            acc = self._generate_from_node(operands[0])
            for ch in operands[1:]:
                if not acc: break
                rhs = self._generate_from_node(ch)
                if not rhs:
                    acc = []
                    break
                new_acc: List[PlanNode] = []
                for a in acc:
                    for b in rhs:
                        pol = self._default_join_policy(mode='intersection')
                        pol['time_mode'] = 'OVERLAPS'
                        new_acc.append(Join(inputs=[a, b], condition={'op': 'AND'}, policy=pol))
                acc = self._dedup(new_acc)
            return acc

        return []

    # --- 추가된 메소드 ---
    def _from_reference(self, node: Dict[str, Any]) -> List[PlanNode]:
        """Create a Probe plan for a Reference node."""
        ref_name = node.get('name')
        if not ref_name:
            return []
        
        label = self.alias_to_label.get(ref_name, ref_name) # Fallback to name if not in decls
        query = f"a {label}"
        
        # A Reference is a direct instruction to find something, so it becomes a Probe.
        return [Probe(target_alias=ref_name, query_text=query, hint={"mr_topk": 5})]
    # --- 여기까지 ---

    def _from_action(self, node: Dict[str, Any]) -> List[PlanNode]:
        verb = node.get('verb', 'interacting')
        args = node.get('args', {}) or {}
        actor = args.get('actor') or {}; obj = args.get('object') or None
        actor_name = actor.get('name') if isinstance(actor, dict) else None
        object_name = obj.get('name') if isinstance(obj, dict) else None
        plans: List[PlanNode] = []

        holistic_query = self._create_holistic_query(node)
        plans.append(Probe(target_alias=f"holistic_action", query_text=holistic_query, hint={"mr_topk": 5}))

        child_probes: List[Probe] = []
        if actor_name:
            label = self.alias_to_label.get(actor_name, 'someone')
            child_probes.append(Probe(target_alias=actor_name, query_text=f"a {label}", hint={"mr_topk": 5}))
        if object_name:
            label = self.alias_to_label.get(object_name, 'something')
            child_probes.append(Probe(target_alias=object_name, query_text=f"a {label}", hint={"mr_topk": 5}))
        if child_probes:
            pol = self._default_join_policy(mode='coexist')
            pol['time_mode'] = 'ACTION'; pol['verb'] = verb
            plans.append(Join(inputs=child_probes,
                               condition={"op": "ACTION", "verb": verb,
                                          "args": {"actor": actor_name, "object": object_name}},
                               policy=pol))
        return self._dedup(plans)

    def _from_relation(self, node: Dict[str, Any]) -> List[PlanNode]:
        left = (node.get('left') or {}).get('name')
        right = (node.get('right') or {}).get('name')
        rel_type = node.get('label', 'related_to')
        plans: List[PlanNode] = []

        plans.append(Probe(target_alias=f"holistic_relation",
                           query_text=self._create_holistic_query(node),
                           hint={"mr_topk": 5}))

        child_probes: List[Probe] = []
        if left:
            label = self.alias_to_label.get(left, 'something')
            child_probes.append(Probe(target_alias=left, query_text=f"a {label}", hint={"mr_topk": 5}))
        if right:
            label = self.alias_to_label.get(right, 'something')
            child_probes.append(Probe(target_alias=right, query_text=f"a {label}", hint={"mr_topk": 5}))
        if child_probes:
            pol = self._default_join_policy(mode='intersection')
            pol['time_mode'] = 'RELATION'; pol['relation_type'] = rel_type
            plans.append(Join(inputs=child_probes,
                               condition={"op": "RELATION", "type": rel_type,
                                          "args": {"left": left, "right": right}},
                               policy=pol))
        return self._dedup(plans)

    def _create_holistic_query(self, node: Dict[str, Any]) -> str:
        ntype = node.get('type')
        if ntype == 'Action':
            args = node.get('args', {}) or {}
            actor = args.get('actor') or {}; object_ = args.get('object') or None
            actor_name = actor.get('name') if isinstance(actor, dict) else None
            object_name = object_.get('name') if isinstance(object_, dict) else None
            verb = node.get('verb', 'interacting')
            actor_label = self.alias_to_label.get(actor_name, 'someone')
            if object_name:
                object_label = self.alias_to_label.get(object_name, 'something')
                return f"a {actor_label} that is {verb} a {object_label}"
            return f"a {actor_label} that is {verb}"
        if ntype == 'Relation':
            left_name = (node.get('left') or {}).get('name')
            right_name = (node.get('right') or {}).get('name')
            rel_type = (node.get('label') or 'related to').replace('_', ' ').lower()
            left_label = self.alias_to_label.get(left_name, 'something')
            right_label = self.alias_to_label.get(right_name, 'something')
            return f"a {left_label} that is {rel_type} a {right_label}"
        return "complex event"

    def _default_join_policy(self, mode: str = 'intersection') -> Dict[str, Any]:
        return {
            "mode": mode, "time_mode": "OVERLAPS",  "relation_type": None,
        }

    def _dedup(self, plans: List[PlanNode]) -> List[PlanNode]:
        seen = set(); out: List[PlanNode] = []
        for p in plans:
            key = p.canonical_key()
            if key not in seen:
                seen.add(key); out.append(p)
        return out
# main_pipeline.py
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import traceback
from itertools import permutations # 순열 생성을 위해 import

# --- 1. 컴파일러 임포트 ---
try:
    from compiler import NL2VTLCompiler, NL2VTLResult
except ImportError:
    print("Error: 'compiler.py' not found. Please ensure it is in the same directory.")
    exit(1)

# ==============================================================================
# 2. Plan Tree 데이터 구조 정의 (변경 없음)
# ==============================================================================

@dataclass
class PlanNode:
    node_type: str = field(init=False)

    def __post_init__(self):
        self.node_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'to_dict'):
                d[k] = v.to_dict()
            elif isinstance(v, list):
                d[k] = [item.to_dict() if hasattr(item, 'to_dict') else item for item in v]
            else:
                d[k] = v
        return d

@dataclass
class Probe(PlanNode):
    target_alias: str
    query_text: str
    temporal_window: Optional[List[str]] = None

@dataclass
class Sequence(PlanNode):
    steps: List[PlanNode]

@dataclass
class Join(PlanNode):
    inputs: List[PlanNode]
    condition: Dict[str, Any]

# Fallback은 더 이상 사용하지 않음

# ==============================================================================
# 3. Plan Generator 구현 (모든 조합 생성)
# ==============================================================================

class PlanGenerator:
    """VTL AST를 분석하여 가능한 모든 실행 계획 조합을 생성합니다."""
    
    def __init__(self, ast: Dict[str, Any]):
        self.ast = ast
        self.alias_to_label: Dict[str, str] = {
            decl.get('name', ''): decl.get('object', {}).get('label', 'unknown')
            for decl in ast.get('declarations', [])
        }
        print(f"\n[DEBUG] PlanGenerator initialized with alias map: {self.alias_to_label}")

    def generate_plans(self) -> List[PlanNode]:
        """최상위 formula 노드로부터 Plan Tree 생성을 시작합니다."""
        if 'formula' not in self.ast:
            return []
        return self._generate_from_node(self.ast['formula'])

    def _generate_from_node(self, node: Dict[str, Any]) -> List[PlanNode]:
        """AST 노드를 재귀적으로 분석하여 가능한 모든 Plan Tree 노드의 리스트를 반환합니다."""
        if not node or not isinstance(node, dict) or 'type' not in node:
            return []
            
        node_type = node.get("type")
        print(f"[DEBUG] Processing AST node of type: {node_type}")

        if node_type == "Operator":
            op = node.get("op", "")
            is_sequential = op in ["BEFORE", "AFTER", "MEETS", ">>"]
            
            # 자식 피연산자들에 대한 모든 가능한 plan들을 재귀적으로 가져옴
            operand_plans = [self._generate_from_node(op_node) for op_node in node.get("operands", [])]
            
            if len(operand_plans) != 2:
                 return [] # 현재는 2개의 피연산자만 가정

            left_plans, right_plans = operand_plans[0], operand_plans[1]
            
            combined_plans = []
            # 모든 자식 plan 조합에 대해 새로운 plan 노드를 생성
            for l_plan in left_plans:
                for r_plan in right_plans:
                    if is_sequential:
                        combined_plans.append(Sequence(steps=[l_plan, r_plan]))
                    else:
                        combined_plans.append(Join(inputs=[l_plan, r_plan], condition=node))
            return combined_plans

        elif node_type in ["Action", "Relation"]:
            all_possible_plans = []

            # 전략 1: Holistic Plan (통합 검색)
            holistic_query = self._create_holistic_query(node)
            holistic_plan = Probe(target_alias=f"holistic_{node_type.lower()}", query_text=holistic_query)
            all_possible_plans.append(holistic_plan)

            # 분해 계획을 위한 기본 Probe 생성
            child_probes = []
            refs = self._extract_refs(node)
            for ref_name in refs:
                label = self.alias_to_label.get(ref_name, "unknown")
                child_probes.append(Probe(target_alias=ref_name, query_text=f"a {label}"))
            
            # 참조된 객체가 2개 이상일 때만 분해 계획이 의미가 있음
            if len(child_probes) > 1:
                # 전략 2: Decomposed Join Plan (병렬 분해 검색)
                decomposed_join_plan = Join(inputs=child_probes, condition=node)
                all_possible_plans.append(decomposed_join_plan)

                # 전략 3: Decomposed Sequence Plans (순차 분해 검색의 모든 순열)
                for p in permutations(child_probes):
                    decomposed_sequence_plan = Sequence(steps=list(p))
                    all_possible_plans.append(decomposed_sequence_plan)

            return all_possible_plans

        else:
            print(f"[DEBUG] Warning: Unhandled node type '{node_type}' encountered.")
            return []

    def _create_holistic_query(self, node: Dict[str, Any]) -> str:
        # 이 함수는 변경 없음
        node_type = node.get("type")
        if node_type == "Action":
            args = node.get('args', {})
            actor_node = args.get('actor')
            actor_name = actor_node.get('name') if isinstance(actor_node, dict) else None
            object_node = args.get('object')
            object_name = object_node.get('name') if isinstance(object_node, dict) else None
            verb = node.get('verb', 'interacting')
            actor_label = self.alias_to_label.get(actor_name, 'someone')
            if object_name:
                object_label = self.alias_to_label.get(object_name, 'something')
                return f"a {actor_label} that is {verb} a {object_label}"
            return f"a {actor_label} that is {verb}"
        elif node_type == "Relation":
            left_name = node.get('left', {}).get('name')
            right_name = node.get('right', {}).get('name')
            rel_type = node.get('label', 'related to').replace('_', ' ').lower()
            left_label = self.alias_to_label.get(left_name, 'something')
            right_label = self.alias_to_label.get(right_name, 'something')
            return f"a {left_label} that is {rel_type} a {right_label}"
        return "complex event"
        
    def _extract_refs(self, node: Dict[str, Any]) -> List[str]:
        # 이 함수는 변경 없음
        refs = set()
        if node.get("type") == "Action":
            args = node.get("args", {})
            actor_node = args.get("actor")
            if isinstance(actor_node, dict) and actor_node.get("name"):
                refs.add(actor_node["name"])
            object_node = args.get("object")
            if isinstance(object_node, dict) and object_node.get("name"):
                refs.add(object_node["name"])
        elif node.get("type") == "Relation":
            if node.get("left", {}).get("name"):
                refs.add(node["left"]["name"])
            if node.get("right", {}).get("name"):
                refs.add(node["right"]["name"])
        return list(refs)

# ==============================================================================
# 4. 전체 파이프라인 실행
# ==============================================================================

def run_pipeline(nl_query: str):
    print("="*60)
    print(f"Input Natural Language Query:\n  '{nl_query}'")
    print("="*60)
    try:
        compiler = NL2VTLCompiler()
        compile_result = compiler.compile(nl_query)
        if compile_result.errors:
            print("\n[!] Compilation failed with errors:")
            for error in compile_result.errors:
                print(f"  - {error}")
            return
        print("\n[1] VTL Compilation Result:")
        print(f"  VTL: {compile_result.vtl}")
        print("\n[DEBUG] Full AST received from compiler:")
        print(json.dumps(compile_result.ast, indent=2))
        ast = compile_result.ast
        generator = PlanGenerator(ast)
        possible_plans = generator.generate_plans()
        print("\n" + "="*60)
        print(f"[2] Generated {len(possible_plans)} Possible Plan Tree(s) from AST")
        print("="*60)
        if not possible_plans:
            print("\nNo executable plans could be generated.")
            return
        for i, plan in enumerate(possible_plans):
            print(f"\n--- Plan Tree Option #{i+1} ---")
            print(json.dumps(plan.to_dict(), indent=2))
    except Exception as e:
        print(f"\n[!!!] An error occurred during the pipeline: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # run_pipeline("A car that is next to a building drives away.")
    
    # 예제 2: Action AND Action
    run_pipeline("A person enters a room and closes the door.")
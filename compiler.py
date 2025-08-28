# compiler.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import json, re
from google import genai
from google.genai import types
import sys 
from dotenv import load_dotenv 
import os 

load_dotenv()

with open('./prompts/nl_to_vtl.md') as f:
    SYSTEM_PROMPT = f.read()

@dataclass
class NL2VTLResult:
    vtl: str
    ast: Dict[str, Any]
    notes: List[str]
    errors: List[str]

class NL2VTLCompiler:
    """
    NL → VTL front-end:
      - Wraps the system prompt and user payload.
      - Calls a Large Language Model (LLM) to perform the compilation.
      - Performs detailed validation on the returned VTL string and AST.
      - (Optional) Expands macros in the VTL string for downstream compatibility.
    """
    def __init__(self, llm_client: Optional[genai.Client] = None, model: str = 'gemini-2.5-pro', *, expand_macro: bool = False):
        if not llm_client:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = llm_client
        

        self.model = model
        self.expand_macro = expand_macro

    def compile(self, nl_query: str, hints: Optional[Dict[str, Any]] = None) -> NL2VTLResult:
        user_payload = {
            "task": "compile_nl_to_vtl",
            "nl_query": nl_query,
            "hints": hints or {}
        }
        
        resp_text = self._call_llm(SYSTEM_PROMPT, json.dumps(user_payload))
        
        try:
            # 수정된 부분: LLM 응답에 다른 텍스트가 포함되어도 JSON 부분만 안정적으로 추출합니다.
            json_match = re.search(r'\{.*\}', resp_text, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON object found in the response", resp_text, 0)
            clean_resp_text = json_match.group(0)
            obj = json.loads(clean_resp_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM did not return valid JSON: {e}\nRaw response: {resp_text}")

        vtl = obj.get("vtl", "").strip()
        ast = obj.get("ast", {})
        notes = obj.get("notes", []) or []
        errors = obj.get("errors", []) or []

        errors.extend(self._validate_vtl(vtl))
        errors.extend(self._validate_ast(ast))

        if self.expand_macro and not errors:
            vtl = self._expand_macro_in_vtl(vtl)

        return NL2VTLResult(vtl=vtl, ast=ast, notes=notes, errors=errors)

    def _call_llm(self, system_prompt: str, user_json: str) -> str:
        """
        사용자님께서 제공해주신 코드 구조를 그대로 사용하여 LLM을 호출합니다.
        """
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=system_prompt),
                    types.Part.from_text(text=user_json),
                ],
            ),
        ]

        # GenerateContentConfig 설정
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text


    def _validate_vtl(self, vtl: str) -> List[str]:
        errs: List[str] = []
        if not vtl:
            errs.append("VTL Error: Missing 'vtl' string in the response.")
            return errs
        if not vtl.endswith(";"):
            errs.append("VTL Error: VTL string must end with a ';'.")
        return errs

    # ==============================================================================
    # 핵심 수정 영역 (Core Modification Area)
    # ==============================================================================

    def _validate_ast(self, ast: Dict[str, Any]) -> List[str]:
        """
        시스템 프롬프트에 정의된 엄격한 스키마에 맞춰 AST를 상세하고 재귀적으로 검증합니다.
        """
        errs: List[str] = []
        if not isinstance(ast, dict):
            return ["AST Error: Root must be a JSON object."]
        
        if "formula" not in ast or "declarations" not in ast:
            errs.append("AST Error: Root must include 'formula' and 'declarations' fields.")
            return errs

        # 1. 모든 선언된 별칭(alias)을 수집하고 각 선언의 유효성을 검사합니다.
        declared_aliases: Set[str] = set()
        declarations = ast.get("declarations", [])
        if not isinstance(declarations, list):
            errs.append("AST Error: 'declarations' field must be a list.")
            return errs

        for i, declaration in enumerate(declarations):
            if not isinstance(declaration, dict) or declaration.get("type") != "Declaration":
                errs.append(f"AST Error: Declaration #{i} is malformed or missing type 'Declaration'.")
                continue
            
            alias_name = declaration.get("name")
            if not alias_name or not isinstance(alias_name, str) or not alias_name.startswith('$'):
                errs.append(f"AST Error: Declaration #{i} is missing a valid 'name' (must be a string starting with $).")
            else:
                declared_aliases.add(alias_name)
            
            obj_node = declaration.get("object")
            if not isinstance(obj_node, dict) or obj_node.get("type") != "Object":
                errs.append(f"AST Error: Alias `{alias_name}` must bind to a node with type 'Object'.")

        # 2. 수집된 별칭 정보를 가지고 formula 전체를 재귀적으로 검증합니다.
        if isinstance(ast.get("formula"), dict):
            self._walk_ast(ast.get("formula"), declared_aliases, errs)
        else:
            errs.append("AST Error: 'formula' field must be a JSON object.")
        
        return errs

    def _walk_ast(self, node: Any, declared_aliases: Set[str], errs: List[str]):
        """AST를 재귀적으로 순회하며 각 노드의 스키마와 의미론적 유효성을 검증합니다."""
        if not isinstance(node, dict) or "type" not in node:
            errs.append(f"AST Error: Malformed node found without a 'type' field: {str(node)[:70]}")
            return

        node_type = node.get("type")

        if node_type == "Reference":
            ref_name = node.get("name")
            if not ref_name or not isinstance(ref_name, str):
                errs.append("AST Error: Reference node is missing a 'name'.")
            elif ref_name not in declared_aliases:
                errs.append(f"AST Error: Undeclared alias `{ref_name}` was referenced.")

        elif node_type == "Action":
            if "verb" not in node or "args" not in node or not isinstance(node.get("args"), dict):
                errs.append("AST Error: Action node must have 'verb' and 'args' fields.")
                return 
            
            actor = node.get("args", {}).get("actor")
            if not actor:
                errs.append("AST Error: Action node must have an 'actor' argument.")
            else:
                self._walk_ast(actor, declared_aliases, errs)

            obj = node.get("args", {}).get("object")
            if obj: # object는 선택 사항이므로 있을 경우에만 검증
                self._walk_ast(obj, declared_aliases, errs)

        elif node_type == "Relation":
            if "left" not in node or "right" not in node or "label" not in node:
                errs.append("AST Error: Relation node is malformed. Must have 'left', 'right', and 'label'.")
                return
            self._walk_ast(node.get("left", {}), declared_aliases, errs)
            self._walk_ast(node.get("right", {}), declared_aliases, errs)

        # 추가된 부분: Operator 노드 검증
        elif node_type == "Operator":
            if "op" not in node or "operands" not in node:
                errs.append("AST Error: Operator node must have 'op' and 'operands' fields.")
                return
            
            operands = node.get("operands")
            if not isinstance(operands, list) or len(operands) == 0:
                errs.append("AST Error: Operator 'operands' must be a non-empty list.")
                return
            
            for operand in operands:
                self._walk_ast(operand, declared_aliases, errs)
        
        else:
            # Declaration, Object 등 formula에 나타나면 안 되는 타입 포함
            errs.append(f"AST Error: Unknown or misplaced node type '{node_type}' found in formula.")

    def _expand_macro_in_vtl(self, vtl: str) -> str:
        """
        'A >> B' 매크로를 확장합니다. 새로운 VTL 문법을 처리하도록 정규식이 개선되었습니다.
        """
        # 정규식 패턴: 별칭($word) 또는 함수 형태의 구문(PREDICATE(...))을 인식
        term_pattern = r'\$\w+|[A-Z_]+\((?:[^()]|\([^)]*\))*\)'

        pat = re.compile(
            rf'(?P<lhs>{term_pattern})\s*>>\s*(?P<rhs>{term_pattern})\s*(?P<within>within\s*\[[^\]]+\])?'
        )

        def repl(m):
            lhs = m.group('lhs')
            rhs = m.group('rhs')
            win = m.group('within') or ""
            return f"END({lhs}) BEFORE START({rhs}) {win}".strip()
        
        return pat.sub(repl, vtl)

# ==============================================================================

if __name__ == "__main__":
    try:
        compiler = NL2VTLCompiler(expand_macro=True)
        nl_query = "Man and woman have a conversation in a train station"
        
        print("="*50)
        print(f"Compiling Query: \"{nl_query}\"")
        print("="*50)

        result = compiler.compile(nl_query)
        
        if result.errors:
            print("\n[!] Compilation finished with errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.notes:
            print("\n[*] Compiler Notes:")
            for note in result.notes:
                print(f"  - {note}")

        print("\n[>] VTL String (Macro Expanded):")
        print(result.vtl)
        
        print("\n[>] AST (JSON):")
        print(json.dumps(result.ast, indent=2))
        print("="*50)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\n[!!!] A critical error occurred: {e}", file=sys.stderr)
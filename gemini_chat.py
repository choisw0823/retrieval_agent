#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_chat.py - Gemini API 채팅 인터페이스

Gemini API를 통한 JSON 응답 처리를 담당합니다.
"""

import os
import re
import json
from typing import List, Dict, Any

try:
    from google import genai
except ImportError:
    genai = None

class GeminiChatJSON:
    """Gemini API를 사용해서 JSON 형태의 응답을 받는 채팅 클래스"""
    
    def __init__(self, model: str = "gemini-2.5-pro"):
        if genai is None:
            raise ImportError("google.genai is not available")
            
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.config = genai.types.GenerateContentConfig(response_mime_type="application/json")
        self.history: List[Any] = []

    def ask(self, system_prompt: str) -> List[Dict[str, Any]]:
        """
        시스템 프롬프트를 보내고 JSON 응답을 받습니다.
        
        Args:
            system_prompt: 전송할 프롬프트
            
        Returns:
            파싱된 JSON 응답 (리스트 형태)
        """
        contents = [genai.types.Content(role="user", parts=[genai.types.Part.from_text(text=system_prompt)])]
        
        try:
            resp = self.client.models.generate_content(
                model=self.model, 
                contents=contents, 
                config=self.config
            )
            text = (resp.text or "").strip()
            self.history.append({'role': 'model', 'parts': [text]})
            
            # JSON 추출 및 파싱
            if text.startswith("```"):
                text = text.strip("`")
                text = text[4:].strip() if text.startswith("json") else text
            
            m = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
            if not m:
                raise ValueError(f"LLM returned non-JSON: {text[:200]}")
            
            data = json.loads(m.group(0))
            
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                return data
            
            raise ValueError("LLM returned JSON that is neither dict nor list of dicts.")
            
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return [{
                "action": "stop", 
                "status": "FAIL", 
                "reason": f"LLM API Error: {e}", 
                "pairs": []
            }]
    
    def clear_history(self):
        """채팅 히스토리를 초기화합니다."""
        self.history.clear()

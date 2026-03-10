"""LLM决策后端

基于大语言模型（vLLM/OpenAI API）的决策后端实现。
通过HTTP调用LLM API生成对话响应。
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from ..base import DecisionBackend
from ..prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class LLMBackend(DecisionBackend):
    """LLM决策后端

    调用vLLM/OpenAI等大语言模型API生成对话响应。
    支持通过人格向量条件化LLM的行为。
    """

    def __init__(
        self,
        llm_endpoint: str,
        model_name: str,
        temperature: float = 0.8,
        max_tokens: int = 256,
        base_system_prompt: str = "",
    ):
        """初始化LLM后端

        Args:
            llm_endpoint: LLM API端点URL
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大生成token数
            base_system_prompt: 基础系统提示
        """
        self.endpoint = llm_endpoint
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = PromptBuilder(base_system_prompt)
        self._client = httpx.Client(timeout=60.0)

    def generate_response(
        self,
        user_input: str,
        personality_vector: np.ndarray,
        conversation_history: List[Dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """使用LLM生成响应"""
        # 构建完整prompt
        prompt_text = self.prompt_builder.build_prompt_text(
            personality_vector, conversation_history, user_input
        )

        # 调用LLM API
        try:
            response = self._call_llm_api(prompt_text)
            return response
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return f"[系统错误] 无法生成响应: {e}"

    def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API

        兼容vLLM的 /v1/gentext 端点和 /generate 端点。

        Args:
            prompt: 完整prompt文本

        Returns:
            LLM生成的文本
        """
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = self._client.post(
            self.endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()

        # 兼容不同API格式
        if "text" in result:
            return result["text"].strip()
        elif "generated_text" in result:
            return result["generated_text"].strip()
        elif "choices" in result:
            return result["choices"][0]["text"].strip()
        else:
            logger.warning("Unexpected API response format: %s", result.keys())
            return str(result)

    def update_model(self, training_data: List[Dict]) -> bool:
        """LLM通常不支持在线更新"""
        logger.debug("LLM backend does not support online model updates")
        return False

    def evaluate_action(
        self, action: str, state: Dict, personality_vector: np.ndarray
    ) -> float:
        """LLM后端默认返回最大效用"""
        return 1.0

    def get_backend_info(self) -> Dict[str, Any]:
        """返回后端信息"""
        return {
            "type": "llm",
            "model": self.model_name,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def close(self) -> None:
        """关闭HTTP客户端"""
        self._client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

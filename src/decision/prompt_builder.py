"""Prompt构建器

负责将人格向量转换为自然语言描述，并构建LLM所需的完整提示。
"""

import logging
from typing import Dict, List

import numpy as np

from ..personality.trait_vector import DIMENSION_LABELS_CN, TraitVector

logger = logging.getLogger(__name__)


class PromptBuilder:
    """人格驱动的Prompt构建器

    将人格向量转换为自然语言描述，并组装系统提示和对话历史。
    """

    def __init__(self, base_system_prompt: str = ""):
        """初始化构建器

        Args:
            base_system_prompt: 基础系统提示（人格描述会追加到此之后）
        """
        self.base_system_prompt = base_system_prompt or (
            "你是Chrysalis，一个拥有独特人格特质的AI助手。"
            "你的人格特征会随着与用户的交互逐渐演化。"
        )

    def vector_to_text(self, personality_vector: np.ndarray, threshold: float = 0.3) -> str:
        """将人格向量转换为自然语言描述

        Args:
            personality_vector: 人格向量
            threshold: 显著性阈值

        Returns:
            人格特征的自然语言描述
        """
        tv = TraitVector(
            dimensions=len(personality_vector),
            initial_values=personality_vector,
        )
        return tv.to_description(threshold)

    def build_system_prompt(self, personality_vector: np.ndarray) -> str:
        """构建包含人格描述的系统提示

        Args:
            personality_vector: 当前人格向量

        Returns:
            完整系统提示字符串
        """
        personality_desc = self.vector_to_text(personality_vector)

        system_prompt = (
            f"{self.base_system_prompt}\n\n"
            f"你当前的人格特征：{personality_desc}。\n"
            f"请在对话中自然地体现这些特征，不要直接提及你的人格设定。"
        )

        return system_prompt

    def build_messages(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        user_input: str,
        max_history: int = 10,
    ) -> List[Dict[str, str]]:
        """构建LLM的消息列表

        Args:
            system_prompt: 系统提示
            conversation_history: 对话历史记录
            user_input: 当前用户输入
            max_history: 最大历史轮数

        Returns:
            消息列表，格式为 [{"role": "...", "content": "..."}]
        """
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话
        recent_history = conversation_history[-max_history:]
        for record in recent_history:
            messages.append(
                {"role": "user", "content": record.get("user_input", "")}
            )
            messages.append(
                {"role": "assistant", "content": record.get("agent_response", "")}
            )

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def build_prompt_text(
        self,
        personality_vector: np.ndarray,
        conversation_history: List[Dict],
        user_input: str,
    ) -> str:
        """构建纯文本格式的prompt（用于非chat模式的LLM）

        Args:
            personality_vector: 人格向量
            conversation_history: 对话历史
            user_input: 用户输入

        Returns:
            完整prompt文本
        """
        system_prompt = self.build_system_prompt(personality_vector)

        parts = [f"[系统指令]\n{system_prompt}\n"]

        # 添加历史对话
        for record in conversation_history[-5:]:
            parts.append(f"[用户]: {record.get('user_input', '')}")
            parts.append(f"[助手]: {record.get('agent_response', '')}")

        parts.append(f"[用户]: {user_input}")
        parts.append("[助手]: ")

        return "\n".join(parts)

"""决策后端抽象接口

定义统一的决策推理接口，支持LLM、自训练模型、传统AI算法的灵活切换。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class DecisionBackend(ABC):
    """决策后端抽象基类

    支持LLM/自训练模型/传统AI的无缝切换。
    所有具体后端实现必须遵循此接口。
    """

    @abstractmethod
    def generate_response(
        self,
        user_input: str,
        personality_vector: np.ndarray,
        conversation_history: List[Dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成对话响应

        Args:
            user_input: 用户输入文本
            personality_vector: 当前人格向量
            conversation_history: 对话历史记录
            context: 额外上下文信息

        Returns:
            生成的响应文本
        """
        pass

    @abstractmethod
    def update_model(self, training_data: List[Dict]) -> bool:
        """更新决策模型

        适用于可训练后端（如自训练模型）。

        Args:
            training_data: 训练数据（交互历史+反馈）

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def evaluate_action(
        self, action: str, state: Dict, personality_vector: np.ndarray
    ) -> float:
        """评估动作的效用值

        适用于Utility AI后端。

        Args:
            action: 候选动作
            state: 当前状态
            personality_vector: 人格向量

        Returns:
            效用值 [0, 1]
        """
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """返回后端类型和配置信息"""
        pass

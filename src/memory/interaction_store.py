"""交互记录存储管理器

提供交互历史的统一管理接口，通过依赖注入存储后端实现解耦。
"""

import logging
from typing import Any, Dict, List, Optional

from .base import MemoryBackend

logger = logging.getLogger(__name__)


class InteractionStore:
    """交互记录存储管理器

    依赖注入存储后端，提供统一的交互记录管理接口。
    业务代码只需与InteractionStore交互，无需关心底层存储细节。
    """

    def __init__(self, backend: MemoryBackend):
        """初始化存储管理器

        Args:
            backend: 存储后端实例
        """
        self.backend = backend

    def add_interaction(
        self,
        user_input: str,
        agent_response: str,
        feedback: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """添加交互记录

        Args:
            user_input: 用户输入
            agent_response: Agent响应
            feedback: 反馈评分 [0, 1]
            metadata: 额外元数据
            embedding: 文本向量嵌入

        Returns:
            记录ID
        """
        return self.backend.add_interaction(
            user_input, agent_response, feedback, metadata, embedding
        )

    def update_feedback(self, interaction_id: str, feedback: float) -> bool:
        """更新交互记录的反馈评分

        Args:
            interaction_id: 记录ID
            feedback: 反馈评分 [0, 1]

        Returns:
            是否更新成功
        """
        return self.backend.update_feedback(interaction_id, feedback)

    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近N条交互记录"""
        return self.backend.get_recent(n)

    def get_by_id(self, interaction_id: str) -> Optional[Dict]:
        """根据ID获取交互记录"""
        return self.backend.get_by_id(interaction_id)

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """语义搜索相似交互"""
        return self.backend.search_similar(query_embedding, top_k)

    def get_all_feedbacks(self) -> List[Dict]:
        """获取所有包含反馈的交互记录"""
        return self.backend.get_all_feedbacks()

    def count(self) -> int:
        """获取交互记录总数"""
        return self.backend.count()

    def save(self) -> None:
        """持久化存储"""
        self.backend.save()

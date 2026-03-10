"""存储后端抽象接口

定义统一的CRUD和语义搜索接口，支持无缝切换不同存储实现。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryBackend(ABC):
    """存储后端抽象基类

    支持从Markdown文件到向量数据库的无缝切换。
    所有具体后端实现必须遵循此接口。
    """

    @abstractmethod
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
            user_input: 用户输入文本
            agent_response: Agent响应文本
            feedback: 用户反馈评分 [0, 1]
            metadata: 额外元数据
            embedding: 文本向量嵌入（预留向量数据库使用）

        Returns:
            交互记录的唯一ID
        """
        pass

    @abstractmethod
    def update_feedback(self, interaction_id: str, feedback: float) -> bool:
        """更新指定交互记录的反馈

        Args:
            interaction_id: 交互记录ID
            feedback: 反馈评分 [0, 1]

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近N条交互

        Args:
            n: 返回数量

        Returns:
            交互记录列表，按时间倒序
        """
        pass

    @abstractmethod
    def get_by_id(self, interaction_id: str) -> Optional[Dict]:
        """根据ID获取交互记录

        Args:
            interaction_id: 交互记录ID

        Returns:
            交互记录字典，不存在时返回None
        """
        pass

    @abstractmethod
    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """语义相似度搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回最相似的K条记录

        Returns:
            交互记录列表，按相似度排序
        """
        pass

    @abstractmethod
    def get_all_feedbacks(self) -> List[Dict]:
        """获取所有包含反馈的交互记录

        Returns:
            包含feedback字段的交互记录列表
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """获取交互记录总数"""
        pass

    @abstractmethod
    def save(self) -> None:
        """持久化数据到存储"""
        pass

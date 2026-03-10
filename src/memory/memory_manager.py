"""记忆管理器

提供存储后端的工厂方法，根据配置创建合适的后端实例。
"""

import logging
from typing import Optional

from .backends.markdown_backend import MarkdownFileBackend
from .base import MemoryBackend
from .interaction_store import InteractionStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 存储后端工厂"""

    @staticmethod
    def create_backend(backend_type: str, **kwargs) -> MemoryBackend:
        """根据配置创建存储后端

        Args:
            backend_type: 后端类型 ("markdown", "chromadb", "milvus", "pinecone")
            **kwargs: 后端初始化参数

        Returns:
            存储后端实例
        """
        if backend_type == "markdown":
            storage_path = kwargs.get(
                "storage_path", "data/interactions/history.md"
            )
            return MarkdownFileBackend(storage_path)
        else:
            raise ValueError(
                f"Unsupported memory backend: {backend_type}. "
                f"Available: markdown"
            )

    @staticmethod
    def create_store(backend_type: str, **kwargs) -> InteractionStore:
        """便捷方法：创建存储后端并封装为InteractionStore

        Args:
            backend_type: 后端类型
            **kwargs: 后端初始化参数

        Returns:
            InteractionStore实例
        """
        backend = MemoryManager.create_backend(backend_type, **kwargs)
        return InteractionStore(backend)

"""Memory Manager

Provides factory methods for storage backends, creating appropriate backend instances based on configuration.
"""

import logging
from typing import Optional

from .backends.markdown_backend import MarkdownFileBackend
from .base import MemoryBackend
from .interaction_store import InteractionStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory Manager - Storage Backend Factory"""

    @staticmethod
    def create_backend(backend_type: str, **kwargs) -> MemoryBackend:
        """Create storage backend based on configuration

        Args:
            backend_type: Backend type ("markdown", "chromadb", "milvus", "pinecone")
            **kwargs: Backend initialization parameters

        Returns:
            Storage backend instance
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
        """Convenience method: Create storage backend and wrap as InteractionStore

        Args:
            backend_type: Backend type
            **kwargs: Backend initialization parameters

        Returns:
            InteractionStore instance
        """
        backend = MemoryManager.create_backend(backend_type, **kwargs)
        return InteractionStore(backend)

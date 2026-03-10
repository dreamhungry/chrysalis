"""Interaction Record Storage Manager

Provides a unified management interface for interaction history,
decoupled through dependency injection of storage backend.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import MemoryBackend

logger = logging.getLogger(__name__)


class InteractionStore:
    """Interaction Record Storage Manager

    Dependency injects storage backend, providing unified interaction record management interface.
    Business code only needs to interact with InteractionStore, without caring about underlying storage details.
    """

    def __init__(self, backend: MemoryBackend):
        """Initialize storage manager

        Args:
            backend: Storage backend instance
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
        """Add interaction record

        Args:
            user_input: User input
            agent_response: Agent response
            feedback: Feedback score [0, 1]
            metadata: Additional metadata
            embedding: Text vector embedding

        Returns:
            Record ID
        """
        return self.backend.add_interaction(
            user_input, agent_response, feedback, metadata, embedding
        )

    def update_feedback(self, interaction_id: str, feedback: float) -> bool:
        """Update feedback score for an interaction record

        Args:
            interaction_id: Record ID
            feedback: Feedback score [0, 1]

        Returns:
            Whether update was successful
        """
        return self.backend.update_feedback(interaction_id, feedback)

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get the most recent N interaction records"""
        return self.backend.get_recent(n)

    def get_by_id(self, interaction_id: str) -> Optional[Dict]:
        """Get interaction record by ID"""
        return self.backend.get_by_id(interaction_id)

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """Semantic search for similar interactions"""
        return self.backend.search_similar(query_embedding, top_k)

    def get_all_feedbacks(self) -> List[Dict]:
        """Get all interaction records containing feedback"""
        return self.backend.get_all_feedbacks()

    def count(self) -> int:
        """Get total number of interaction records"""
        return self.backend.count()

    def save(self) -> None:
        """Persist storage"""
        self.backend.save()

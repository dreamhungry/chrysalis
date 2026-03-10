"""Storage Backend Abstract Interface

Defines unified CRUD and semantic search interfaces,
supporting seamless switching between different storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryBackend(ABC):
    """Storage Backend Abstract Base Class

    Supports seamless switching from Markdown files to vector databases.
    All concrete backend implementations must follow this interface.
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
        """Add interaction record

        Args:
            user_input: User input text
            agent_response: Agent response text
            feedback: User feedback score [0, 1]
            metadata: Additional metadata
            embedding: Text vector embedding (reserved for vector database use)

        Returns:
            Unique ID of the interaction record
        """
        pass

    @abstractmethod
    def update_feedback(self, interaction_id: str, feedback: float) -> bool:
        """Update feedback for a specific interaction record

        Args:
            interaction_id: Interaction record ID
            feedback: Feedback score [0, 1]

        Returns:
            Whether update was successful
        """
        pass

    @abstractmethod
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get the most recent N interactions

        Args:
            n: Number to return

        Returns:
            List of interaction records, sorted by time descending
        """
        pass

    @abstractmethod
    def get_by_id(self, interaction_id: str) -> Optional[Dict]:
        """Get interaction record by ID

        Args:
            interaction_id: Interaction record ID

        Returns:
            Interaction record dict, or None if not found
        """
        pass

    @abstractmethod
    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """Semantic similarity search

        Args:
            query_embedding: Query vector
            top_k: Return top K most similar records

        Returns:
            List of interaction records, sorted by similarity
        """
        pass

    @abstractmethod
    def get_all_feedbacks(self) -> List[Dict]:
        """Get all interaction records containing feedback

        Returns:
            List of interaction records with feedback field
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of interaction records"""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist data to storage"""
        pass

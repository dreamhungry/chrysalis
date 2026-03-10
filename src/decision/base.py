"""Decision Backend Abstract Interface

Defines unified decision reasoning interface, supporting flexible switching
between LLM, self-trained models, and traditional AI algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class DecisionBackend(ABC):
    """Decision Backend Abstract Base Class

    Supports seamless switching between LLM/self-trained models/traditional AI.
    All concrete backend implementations must follow this interface.
    """

    @abstractmethod
    def generate_response(
        self,
        user_input: str,
        personality_vector: np.ndarray,
        conversation_history: List[Dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate conversation response

        Args:
            user_input: User input text
            personality_vector: Current personality vector
            conversation_history: Conversation history records
            context: Additional context information

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def update_model(self, training_data: List[Dict]) -> bool:
        """Update decision model

        Applicable to trainable backends (e.g., self-trained models).

        Args:
            training_data: Training data (interaction history + feedback)

        Returns:
            Whether update was successful
        """
        pass

    @abstractmethod
    def evaluate_action(
        self, action: str, state: Dict, personality_vector: np.ndarray
    ) -> float:
        """Evaluate utility value of an action

        Applicable to Utility AI backend.

        Args:
            action: Candidate action
            state: Current state
            personality_vector: Personality vector

        Returns:
            Utility value [0, 1]
        """
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Return backend type and configuration information"""
        pass

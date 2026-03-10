"""Feedback-Based Incremental Updater

Converts user feedback (positive/negative) into gradient adjustments for personality vector.
"""

import logging
from typing import Dict, Optional

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class FeedbackUpdater:
    """Feedback-Based Personality Incremental Update

    Converts user feedback values to gradient directions,
    fine-tuning personality vector along user preference direction.
    Supports adaptive learning rate: fast adaptation early, stabilizing later.
    """

    def __init__(
        self,
        base_learning_rate: float = 0.1,
        decay_factor: float = 0.995,
        min_learning_rate: float = 0.01,
    ):
        """Initialize feedback updater

        Args:
            base_learning_rate: Initial learning rate
            decay_factor: Learning rate decay factor (multiplied after each update)
            min_learning_rate: Minimum learning rate bound
        """
        self.base_learning_rate = base_learning_rate
        self.current_learning_rate = base_learning_rate
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate
        self._update_count = 0

    def compute_gradient(
        self, feedback: float, context: Optional[Dict] = None
    ) -> np.ndarray:
        """Compute personality vector gradient corresponding to feedback

        Positive feedback (feedback > 0.5) enhances current trait direction,
        Negative feedback (feedback < 0.5) suppresses current trait direction.

        Args:
            feedback: User feedback value [0, 1]
            context: Interaction context information (includes current personality vector, etc.)

        Returns:
            Gradient vector
        """
        context = context or {}
        current_vector = context.get("personality_vector", np.zeros(10))

        if isinstance(current_vector, list):
            current_vector = np.array(current_vector)

        # Map feedback [0, 1] to [-1, 1]
        direction = (feedback - 0.5) * 2.0

        # Gradient direction: positive feedback enhances current traits, negative suppresses
        if np.linalg.norm(current_vector) > 0.01:
            # Adjust along current vector direction
            gradient = direction * current_vector / (np.linalg.norm(current_vector) + 1e-8)
        else:
            # When current vector is near zero, use random perturbation
            rng = np.random.default_rng()
            gradient = direction * rng.normal(0, 0.1, size=len(current_vector))

        # Add exploration noise
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.02, size=len(gradient))
        gradient += noise

        return gradient

    def update(
        self, trait_vector: TraitVector, feedback: float, context: Optional[Dict] = None
    ) -> None:
        """Execute one feedback update

        Args:
            trait_vector: Personality vector to update
            feedback: User feedback [0, 1]
            context: Interaction context
        """
        context = context or {}
        context["personality_vector"] = trait_vector.vector.copy()

        gradient = self.compute_gradient(feedback, context)
        trait_vector.update(gradient, self.current_learning_rate)

        # Decay learning rate
        self._update_count += 1
        self.current_learning_rate = max(
            self.min_learning_rate,
            self.base_learning_rate * (self.decay_factor ** self._update_count),
        )

        logger.info(
            "Feedback update #%d: feedback=%.2f, lr=%.4f",
            self._update_count,
            feedback,
            self.current_learning_rate,
        )

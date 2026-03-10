"""Reinforcement Learning Updater

Uses simplified policy gradient method to update personality vector based on user feedback signals.
Treats feedback as reward signal and personality vector as policy parameters for optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class RLUpdater:
    """Reinforcement Learning Personality Updater

    Models personality evolution as a policy optimization problem:
    - State: Conversation context
    - Action: Personality vector direction
    - Reward: User feedback
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        gamma: float = 0.95,
        baseline_decay: float = 0.99,
    ):
        """Initialize RL updater

        Args:
            learning_rate: Policy learning rate
            gamma: Discount factor (for weighting historical rewards)
            baseline_decay: Exponential moving average decay rate for reward baseline
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        self._reward_baseline = 0.5  # Initial reward baseline
        self._episode_buffer: List[Tuple[np.ndarray, float]] = []

    def record_step(self, personality_vector: np.ndarray, reward: float) -> None:
        """Record a state-reward pair for one interaction step

        Args:
            personality_vector: Personality vector at that time (snapshot)
            reward: User feedback reward [0, 1]
        """
        self._episode_buffer.append((personality_vector.copy(), reward))

    def compute_returns(self) -> List[float]:
        """Compute discounted returns

        Returns:
            List of discounted returns for each step
        """
        returns = []
        G = 0.0
        for _, reward in reversed(self._episode_buffer):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, trait_vector: TraitVector) -> Optional[float]:
        """Execute policy update based on accumulated experience

        Uses a simplified version of REINFORCE algorithm.

        Args:
            trait_vector: Personality vector to update

        Returns:
            Average return, or None if buffer is empty
        """
        if not self._episode_buffer:
            return None

        returns = self.compute_returns()
        avg_return = np.mean(returns)

        # Update reward baseline
        self._reward_baseline = (
            self.baseline_decay * self._reward_baseline
            + (1 - self.baseline_decay) * avg_return
        )

        # Compute policy gradient
        gradient = np.zeros(trait_vector.dimensions)
        for i, (state_vector, _) in enumerate(self._episode_buffer):
            advantage = returns[i] - self._reward_baseline

            # Simplified policy gradient: use state vector change direction as gradient
            direction = state_vector / (np.linalg.norm(state_vector) + 1e-8)
            gradient += advantage * direction

        # Normalize gradient
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / (np.linalg.norm(gradient) + 1e-8)

        # Apply update
        trait_vector.update(gradient, self.learning_rate)

        # Clear buffer
        step_count = len(self._episode_buffer)
        self._episode_buffer.clear()

        logger.info(
            "RL update applied: %d steps, avg_return=%.3f, baseline=%.3f",
            step_count,
            avg_return,
            self._reward_baseline,
        )
        return float(avg_return)

    @property
    def buffer_size(self) -> int:
        """Current number of steps in buffer"""
        return len(self._episode_buffer)

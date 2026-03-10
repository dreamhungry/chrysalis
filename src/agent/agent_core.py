"""Agent Core - Agent Core Conversation Management

Integrates decision backend, personality vector, memory storage, and learning modules,
providing unified conversation processing interface.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from ..decision.base import DecisionBackend
from ..learning.feedback_updater import FeedbackUpdater
from ..learning.pattern_extractor import PatternExtractor
from ..learning.reflector import Reflector
from ..learning.rl_updater import RLUpdater
from ..memory.interaction_store import InteractionStore
from ..personality.personality_store import PersonalityStore
from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)

# Interval for triggering pattern learning
PATTERN_LEARNING_INTERVAL = 50
# Interval for triggering RL update
RL_UPDATE_INTERVAL = 20
# Interval for triggering reflection
REFLECTION_INTERVAL = 30


class AgentCore:
    """AI Agent Core

    Manages conversation flow, personality-driven response generation,
    and scheduling of multiple learning mechanisms.
    """

    def __init__(
        self,
        decision_backend: DecisionBackend,
        trait_vector: TraitVector,
        interaction_store: InteractionStore,
        personality_store: PersonalityStore,
        feedback_updater: Optional[FeedbackUpdater] = None,
        pattern_extractor: Optional[PatternExtractor] = None,
        rl_updater: Optional[RLUpdater] = None,
        reflector: Optional[Reflector] = None,
    ):
        """Initialize Agent Core

        Args:
            decision_backend: Decision backend
            trait_vector: Personality vector
            interaction_store: Interaction record storage
            personality_store: Personality persistence
            feedback_updater: Feedback updater
            pattern_extractor: Pattern extractor
            rl_updater: RL updater
            reflector: Reflector
        """
        self.decision = decision_backend
        self.personality = trait_vector
        self.memory = interaction_store
        self.personality_store = personality_store

        # Learning modules
        self.feedback_updater = feedback_updater or FeedbackUpdater()
        self.pattern_extractor = pattern_extractor or PatternExtractor()
        self.rl_updater = rl_updater or RLUpdater()
        self.reflector = reflector or Reflector(decision_backend)

        self._interaction_count = 0
        self._last_interaction_id: Optional[str] = None

    def chat(self, user_input: str) -> str:
        """Process user input and generate response

        Args:
            user_input: User input text

        Returns:
            Agent response text
        """
        # 1. Get conversation history
        conversation_history = self.memory.get_recent(10)

        # 2. Call decision backend to generate response
        response = self.decision.generate_response(
            user_input=user_input,
            personality_vector=self.personality.vector,
            conversation_history=conversation_history,
        )

        # 3. Record interaction
        self._last_interaction_id = self.memory.add_interaction(
            user_input, response
        )

        # 4. Update count and trigger periodic learning
        self._interaction_count += 1
        self._trigger_periodic_learning()

        logger.info(
            "Chat completed (interaction #%d, personality norm=%.3f)",
            self._interaction_count,
            np.linalg.norm(self.personality.vector),
        )

        return response

    def provide_feedback(self, feedback: float, interaction_id: Optional[str] = None) -> None:
        """Receive user feedback and trigger learning

        Args:
            feedback: Feedback score [0, 1]
            interaction_id: Associated interaction ID, uses most recent interaction if None
        """
        feedback = max(0.0, min(1.0, feedback))
        target_id = interaction_id or self._last_interaction_id

        # Update feedback in interaction record
        if target_id:
            self.memory.update_feedback(target_id, feedback)

        # 1. Feedback-based incremental update
        self.feedback_updater.update(
            self.personality,
            feedback,
            context={"personality_vector": self.personality.vector.tolist()},
        )

        # 2. Record RL step
        self.rl_updater.record_step(self.personality.vector, feedback)

        # 3. Save personality state
        self.personality_store.save(self.personality)
        self.personality_store.save_snapshot(
            self.personality, event=f"feedback_{feedback:.2f}"
        )

        logger.info(
            "Feedback processed: %.2f, personality updated", feedback
        )

    def get_personality_info(self) -> Dict:
        """Get current personality state information

        Returns:
            Personality information dictionary
        """
        return {
            "vector": self.personality.vector.tolist(),
            "description": self.personality.to_description(),
            "dimension_names": self.personality.dimension_names,
            "traits": {
                name: float(self.personality.vector[i])
                for i, name in enumerate(self.personality.dimension_names)
            },
            "update_count": self.personality._update_count,
            "interaction_count": self._interaction_count,
        }

    def switch_backend(self, new_backend: DecisionBackend) -> None:
        """Dynamically switch decision backend

        Args:
            new_backend: New decision backend instance
        """
        old_info = self.decision.get_backend_info()
        self.decision = new_backend
        self.reflector = Reflector(new_backend)

        logger.info(
            "Decision backend switched: %s -> %s",
            old_info.get("type"),
            new_backend.get_backend_info().get("type"),
        )

    def _trigger_periodic_learning(self) -> None:
        """Trigger periodic learning mechanisms"""
        # Pattern learning
        if self._interaction_count % PATTERN_LEARNING_INTERVAL == 0:
            interactions = self.memory.get_recent(100)
            self.pattern_extractor.update_from_patterns(self.personality, interactions)
            self.personality_store.save(self.personality)
            self.personality_store.save_snapshot(
                self.personality, event="pattern_learning"
            )
            logger.info("Pattern learning triggered at interaction #%d", self._interaction_count)

        # RL update
        if (
            self._interaction_count % RL_UPDATE_INTERVAL == 0
            and self.rl_updater.buffer_size > 0
        ):
            self.rl_updater.update(self.personality)
            self.personality_store.save(self.personality)
            self.personality_store.save_snapshot(
                self.personality, event="rl_update"
            )
            logger.info("RL update triggered at interaction #%d", self._interaction_count)

        # Reflection
        if self._interaction_count % REFLECTION_INTERVAL == 0:
            interactions = self.memory.get_recent(20)
            self.reflector.reflect(self.personality, interactions)
            self.personality_store.save(self.personality)
            self.personality_store.save_snapshot(
                self.personality, event="reflection"
            )
            logger.info("Reflection triggered at interaction #%d", self._interaction_count)

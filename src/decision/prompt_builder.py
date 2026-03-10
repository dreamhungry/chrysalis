"""Prompt Builder

Responsible for converting personality vectors to natural language descriptions
and building complete prompts needed by LLM.
"""

import logging
from typing import Dict, List

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Personality-Driven Prompt Builder

    Converts personality vectors to natural language descriptions
    and assembles system prompts and conversation history.
    """

    def __init__(self, base_system_prompt: str = ""):
        """Initialize builder

        Args:
            base_system_prompt: Base system prompt (personality description will be appended after this)
        """
        self.base_system_prompt = base_system_prompt or (
            "You are Chrysalis, an AI assistant with unique personality traits."
            "Your personality characteristics will gradually evolve through interactions with users."
        )

    def vector_to_text(self, personality_vector: np.ndarray, threshold: float = 0.3) -> str:
        """Convert personality vector to natural language description

        Args:
            personality_vector: Personality vector
            threshold: Significance threshold

        Returns:
            Natural language description of personality traits
        """
        tv = TraitVector(
            dimensions=len(personality_vector),
            initial_values=personality_vector,
        )
        return tv.to_description(threshold)

    def build_system_prompt(self, personality_vector: np.ndarray) -> str:
        """Build system prompt containing personality description

        Args:
            personality_vector: Current personality vector

        Returns:
            Complete system prompt string
        """
        personality_desc = self.vector_to_text(personality_vector)

        system_prompt = (
            f"{self.base_system_prompt}\n\n"
            f"Your current personality traits: {personality_desc}.\n"
            f"Please naturally reflect these traits in conversation, without directly mentioning your personality settings."
        )

        return system_prompt

    def build_messages(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        user_input: str,
        max_history: int = 10,
    ) -> List[Dict[str, str]]:
        """Build LLM message list

        Args:
            system_prompt: System prompt
            conversation_history: Conversation history records
            user_input: Current user input
            max_history: Maximum history turns

        Returns:
            Message list in format [{"role": "...", "content": "..."}]
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Add historical conversation
        recent_history = conversation_history[-max_history:]
        for record in recent_history:
            messages.append(
                {"role": "user", "content": record.get("user_input", "")}
            )
            messages.append(
                {"role": "assistant", "content": record.get("agent_response", "")}
            )

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    def build_prompt_text(
        self,
        personality_vector: np.ndarray,
        conversation_history: List[Dict],
        user_input: str,
    ) -> str:
        """Build plain text format prompt (for non-chat mode LLM)

        Args:
            personality_vector: Personality vector
            conversation_history: Conversation history
            user_input: User input

        Returns:
            Complete prompt text
        """
        system_prompt = self.build_system_prompt(personality_vector)

        parts = [f"[System Instruction]\n{system_prompt}\n"]

        # Add historical conversation
        for record in conversation_history[-5:]:
            parts.append(f"[User]: {record.get('user_input', '')}")
            parts.append(f"[Assistant]: {record.get('agent_response', '')}")

        parts.append(f"[User]: {user_input}")
        parts.append("[Assistant]: ")

        return "\n".join(parts)

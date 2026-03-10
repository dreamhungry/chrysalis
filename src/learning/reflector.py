"""Self-Reflection Module

Uses decision backend (e.g., LLM) to perform reflection analysis on interaction history,
extracting personality adjustment suggestions. Falls back to statistical methods when decision backend is unavailable.
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

from ..decision.base import DecisionBackend
from ..personality.trait_vector import DIMENSION_LABELS_CN, TraitVector

logger = logging.getLogger(__name__)


class Reflector:
    """Self-Reflector

    Analyzes interaction history and generates personality adjustment suggestions.
    Supports both LLM-assisted reflection and statistical analysis modes.
    """

    def __init__(self, decision_backend: Optional[DecisionBackend] = None):
        """Initialize reflector

        Args:
            decision_backend: Decision backend (optional, for LLM-assisted reflection)
        """
        self.decision_backend = decision_backend

    def reflect(
        self,
        trait_vector: TraitVector,
        interactions: List[Dict],
        learning_rate: float = 0.05,
    ) -> Optional[Dict]:
        """Execute self-reflection and update personality vector

        Args:
            trait_vector: Current personality vector
            interactions: Interaction history
            learning_rate: Learning rate

        Returns:
            Reflection result dictionary containing analysis and adjustment information
        """
        if not interactions:
            return None

        # Choose reflection method based on backend type
        if (
            self.decision_backend is not None
            and self.decision_backend.get_backend_info().get("type") == "llm"
        ):
            result = self._llm_reflect(trait_vector, interactions)
        else:
            result = self._statistical_reflect(trait_vector, interactions)

        if result and "adjustment" in result:
            adjustment = np.array(result["adjustment"])
            trait_vector.update(adjustment, learning_rate)
            logger.info("Reflection update applied")

        return result

    def _llm_reflect(
        self, trait_vector: TraitVector, interactions: List[Dict]
    ) -> Dict:
        """Perform self-reflection using LLM"""
        # Build reflection prompt
        reflection_prompt = self._build_reflection_prompt(trait_vector, interactions)

        try:
            response = self.decision_backend.generate_response(
                user_input=reflection_prompt,
                personality_vector=np.zeros(trait_vector.dimensions),  # Don't inject personality during reflection
                conversation_history=[],
            )

            # Try to parse LLM output as adjustment vector
            adjustment = self._parse_reflection_response(
                response, trait_vector.dimensions
            )

            return {
                "method": "llm_reflection",
                "analysis": response,
                "adjustment": adjustment.tolist(),
            }
        except Exception as e:
            logger.warning("LLM reflection failed: %s, falling back to statistical", e)
            return self._statistical_reflect(trait_vector, interactions)

    def _statistical_reflect(
        self, trait_vector: TraitVector, interactions: List[Dict]
    ) -> Dict:
        """Perform reflection using statistical methods"""
        feedbacks = [
            r["feedback"]
            for r in interactions
            if r.get("feedback") is not None
        ]

        adjustment = np.zeros(trait_vector.dimensions)
        analysis = "Statistical reflection analysis:\n"

        if feedbacks:
            avg_feedback = np.mean(feedbacks)
            recent_feedbacks = feedbacks[-10:]
            recent_avg = np.mean(recent_feedbacks) if recent_feedbacks else avg_feedback

            analysis += f"- Overall average feedback: {avg_feedback:.2f}\n"
            analysis += f"- Recent average feedback: {recent_avg:.2f}\n"

            # If recent feedback improved -> reinforce current direction
            if recent_avg > avg_feedback + 0.05:
                adjustment = 0.05 * trait_vector.vector
                analysis += "- Trend: Recent performance improved, reinforcing current personality direction\n"
            # If recent feedback declined -> small rollback
            elif recent_avg < avg_feedback - 0.05:
                adjustment = -0.03 * trait_vector.vector
                analysis += "- Trend: Recent performance declined, slightly rolling back personality direction\n"
            else:
                analysis += "- Trend: Performance stable, maintaining current state\n"

            # Feedback variance analysis
            if np.std(feedbacks) > 0.3:
                analysis += "- Note: High feedback variance, need more data for observation\n"
        else:
            analysis += "- No user feedback data available\n"

        return {
            "method": "statistical_reflection",
            "analysis": analysis,
            "adjustment": adjustment.tolist(),
        }

    def _build_reflection_prompt(
        self, trait_vector: TraitVector, interactions: List[Dict]
    ) -> str:
        """Build LLM reflection prompt"""
        current_desc = trait_vector.to_description()

        # Take recent interactions as samples
        recent = interactions[-5:]
        conversation_samples = ""
        for r in recent:
            conversation_samples += f"User: {r.get('user_input', '')}\n"
            conversation_samples += f"Assistant: {r.get('agent_response', '')}\n"
            if r.get("feedback") is not None:
                conversation_samples += f"Feedback score: {r['feedback']:.2f}\n"
            conversation_samples += "\n"

        feedbacks = [
            r["feedback"]
            for r in interactions
            if r.get("feedback") is not None
        ]
        avg_feedback = np.mean(feedbacks) if feedbacks else 0.5

        prompt = (
            f"You are an AI personality analyst. Please analyze the following conversation history and feedback data, "
            f"provide suggestions for AI assistant's personality trait adjustments.\n\n"
            f"Current personality traits: {current_desc}\n"
            f"Average user feedback: {avg_feedback:.2f}\n\n"
            f"Recent conversation samples:\n{conversation_samples}\n"
            f"Please return personality adjustment suggestions in JSON format:\n"
            f'{{"adjustment": [v0, v1, ..., v9]}}\n'
            f"Where each value is in [-0.2, 0.2] range, "
            f"positive means enhance, negative means reduce."
        )

        return prompt

    def _parse_reflection_response(
        self, response: str, dimensions: int
    ) -> np.ndarray:
        """Parse LLM reflection response into adjustment vector"""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                if "adjustment" in data:
                    adj = np.array(data["adjustment"][:dimensions], dtype=np.float64)
                    # Limit adjustment magnitude
                    adj = np.clip(adj, -0.2, 0.2)
                    # Pad dimensions if needed
                    if len(adj) < dimensions:
                        adj = np.pad(adj, (0, dimensions - len(adj)))
                    return adj
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse reflection response: %s", e)

        return np.zeros(dimensions)

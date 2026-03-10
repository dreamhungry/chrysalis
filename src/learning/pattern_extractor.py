"""Interaction History Pattern Extractor

Statistically analyzes meaningful behavioral patterns from large amounts of interaction records,
and maps them to personality vector adjustment suggestions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class PatternExtractor:
    """Interaction Pattern Extractor

    Extracts conversation style, user preferences, and other patterns through statistical analysis,
    converting them to personality vector adjustment directions.
    """

    def __init__(self, analysis_window: int = 50):
        """Initialize pattern extractor

        Args:
            analysis_window: Analysis window size (use most recent N interactions)
        """
        self.analysis_window = analysis_window

    def extract_patterns(self, interactions: List[Dict]) -> Dict[str, float]:
        """Extract behavioral patterns from interaction history

        Args:
            interactions: List of interaction records

        Returns:
            Pattern feature dictionary, key is feature name, value is feature value
        """
        if not interactions:
            return {}

        recent = interactions[-self.analysis_window :]

        patterns = {}

        # 1. Average response length -> maps to verbosity
        avg_response_len = np.mean(
            [len(r.get("agent_response", "")) for r in recent]
        )
        patterns["avg_response_length"] = avg_response_len

        # 2. Average user input length
        avg_input_len = np.mean(
            [len(r.get("user_input", "")) for r in recent]
        )
        patterns["avg_input_length"] = avg_input_len

        # 3. Feedback statistics
        feedbacks = [
            r["feedback"] for r in recent if r.get("feedback") is not None
        ]
        if feedbacks:
            patterns["avg_feedback"] = np.mean(feedbacks)
            patterns["feedback_std"] = np.std(feedbacks)
            patterns["positive_feedback_ratio"] = sum(
                1 for f in feedbacks if f > 0.6
            ) / len(feedbacks)
        else:
            patterns["avg_feedback"] = 0.5
            patterns["feedback_std"] = 0.0
            patterns["positive_feedback_ratio"] = 0.5

        # 4. Question mark frequency (user questioning tendency)
        question_count = sum(
            1
            for r in recent
            if "?" in r.get("user_input", "") or "？" in r.get("user_input", "")
        )
        patterns["question_ratio"] = question_count / len(recent) if recent else 0

        # 5. Interaction frequency (if timestamps available)
        patterns["interaction_count"] = len(recent)

        logger.debug("Extracted patterns: %s", patterns)
        return patterns

    def patterns_to_adjustment(
        self, patterns: Dict[str, float], current_vector: TraitVector
    ) -> np.ndarray:
        """Convert extracted patterns to personality vector adjustment

        Args:
            patterns: Pattern feature dictionary
            current_vector: Current personality vector

        Returns:
            Adjustment vector
        """
        adjustment = np.zeros(current_vector.dimensions)

        # Adjust friendliness based on user feedback
        avg_feedback = patterns.get("avg_feedback", 0.5)
        if avg_feedback > 0.6:
            adjustment[0] += 0.1  # friendliness +
        elif avg_feedback < 0.4:
            adjustment[0] -= 0.1  # friendliness -

        # Adjust verbosity based on response length preference
        avg_len = patterns.get("avg_response_length", 100)
        if avg_len > 200:
            adjustment[5] += 0.05  # verbosity +
        elif avg_len < 50:
            adjustment[5] -= 0.05  # verbosity -

        # Adjust curiosity/patience based on question frequency
        question_ratio = patterns.get("question_ratio", 0.5)
        if question_ratio > 0.6:
            adjustment[7] += 0.05  # curiosity +
            adjustment[8] += 0.05  # patience +

        # Reinforce overall based on positive feedback ratio
        pos_ratio = patterns.get("positive_feedback_ratio", 0.5)
        if pos_ratio > 0.7:
            # High positive feedback -> maintain current direction
            adjustment += 0.02 * current_vector.vector
        elif pos_ratio < 0.3:
            # Low positive feedback -> reverse adjustment
            adjustment -= 0.02 * current_vector.vector

        return adjustment

    def update_from_patterns(
        self,
        trait_vector: TraitVector,
        interactions: List[Dict],
        learning_rate: float = 0.05,
    ) -> Dict[str, float]:
        """Execute one pattern-based personality update

        Args:
            trait_vector: Personality vector to update
            interactions: Interaction history
            learning_rate: Learning rate

        Returns:
            Extracted pattern features
        """
        patterns = self.extract_patterns(interactions)

        if not patterns:
            return patterns

        adjustment = self.patterns_to_adjustment(patterns, trait_vector)
        trait_vector.update(adjustment, learning_rate)

        logger.info(
            "Pattern-based update applied (window=%d interactions)",
            len(interactions[-self.analysis_window :]),
        )
        return patterns

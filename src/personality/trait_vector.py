"""Personality Trait Vector Module

Models personality traits as a multi-dimensional continuous vector,
supporting quantitative personality state representation and evolution.
Each dimension maps to a specific personality attribute, range [-1, 1].
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default 10-dimensional personality traits and their semantic labels
DEFAULT_DIMENSION_NAMES = [
    "friendliness",   # friendliness: -1=cold → 1=warm
    "formality",      # formality: -1=casual → 1=formal
    "humor",          # humor: -1=serious → 1=humorous
    "creativity",     # creativity: -1=conservative → 1=innovative
    "empathy",        # empathy: -1=rational → 1=emotional
    "verbosity",      # verbosity: -1=concise → 1=verbose
    "confidence",     # confidence: -1=humble → 1=confident
    "curiosity",      # curiosity: -1=indifferent → 1=curious
    "patience",       # patience: -1=impatient → 1=patient
    "assertiveness",  # assertiveness: -1=passive → 1=assertive
]

# Chinese label mapping (for display and LLM prompt construction)
DIMENSION_LABELS_CN = {
    "friendliness": ("友好度", "冷淡", "热情"),
    "formality": ("正式度", "随意", "正式"),
    "humor": ("幽默感", "严肃", "幽默"),
    "creativity": ("创造性", "保守", "创新"),
    "empathy": ("共情力", "理性", "感性"),
    "verbosity": ("表达量", "简洁", "详细"),
    "confidence": ("自信度", "谦逊", "自信"),
    "curiosity": ("好奇心", "淡定", "好奇"),
    "patience": ("耐心度", "急躁", "耐心"),
    "assertiveness": ("主动性", "被动", "主动"),
}


class TraitVector:
    """Personality Trait Vector Representation

    Encapsulates multi-dimensional trait vectors and their semantic labels,
    supporting incremental updates, normalization, and serialization.

    Attributes:
        vector: NumPy vector, each dimension ranges [-1, 1]
        dimension_names: Semantic names for each dimension
    """

    def __init__(
        self,
        dimensions: int = 10,
        dimension_names: Optional[List[str]] = None,
        initial_values: Optional[np.ndarray] = None,
    ):
        """Initialize trait vector

        Args:
            dimensions: Number of vector dimensions
            dimension_names: Semantic names for each dimension, defaults if None
            initial_values: Initial values, zero vector if None
        """
        if dimension_names is not None:
            self.dimension_names = dimension_names
            dimensions = len(dimension_names)
        elif dimensions <= len(DEFAULT_DIMENSION_NAMES):
            self.dimension_names = DEFAULT_DIMENSION_NAMES[:dimensions]
        else:
            self.dimension_names = DEFAULT_DIMENSION_NAMES + [
                f"dim_{i}" for i in range(len(DEFAULT_DIMENSION_NAMES), dimensions)
            ]

        if initial_values is not None:
            if len(initial_values) != dimensions:
                raise ValueError(
                    f"initial_values length ({len(initial_values)}) != dimensions ({dimensions})"
                )
            self.vector = np.clip(np.array(initial_values, dtype=np.float64), -1.0, 1.0)
        else:
            self.vector = np.zeros(dimensions, dtype=np.float64)

        self._update_count = 0

    @property
    def dimensions(self) -> int:
        """Number of vector dimensions"""
        return len(self.vector)

    def update(self, delta: np.ndarray, learning_rate: float = 0.1) -> None:
        """Incrementally update vector

        Args:
            delta: Update direction vector
            learning_rate: Learning rate controlling update magnitude
        """
        if len(delta) != self.dimensions:
            raise ValueError(
                f"delta dimensions ({len(delta)}) != vector dimensions ({self.dimensions})"
            )

        self.vector += learning_rate * np.array(delta, dtype=np.float64)
        self.vector = np.clip(self.vector, -1.0, 1.0)
        self._update_count += 1

        logger.debug(
            "TraitVector updated (count=%d), norm=%.4f",
            self._update_count,
            np.linalg.norm(self.vector),
        )

    def get_trait(self, name: str) -> float:
        """Get trait value for specified dimension

        Args:
            name: Dimension name

        Returns:
            Trait value [-1, 1] for that dimension
        """
        if name not in self.dimension_names:
            raise KeyError(f"Unknown trait dimension: {name}")
        idx = self.dimension_names.index(name)
        return float(self.vector[idx])

    def set_trait(self, name: str, value: float) -> None:
        """Set trait value for specified dimension

        Args:
            name: Dimension name
            value: Trait value, clipped to [-1, 1]
        """
        if name not in self.dimension_names:
            raise KeyError(f"Unknown trait dimension: {name}")
        idx = self.dimension_names.index(name)
        self.vector[idx] = np.clip(value, -1.0, 1.0)

    def distance(self, other: "TraitVector") -> float:
        """Calculate Euclidean distance to another trait vector"""
        return float(np.linalg.norm(self.vector - other.vector))

    def cosine_similarity(self, other: "TraitVector") -> float:
        """Calculate cosine similarity to another trait vector"""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))

    def to_description(self, threshold: float = 0.3) -> str:
        """Convert trait vector to natural language description (Chinese)

        Only describes significant traits with absolute value above threshold.

        Args:
            threshold: Significance threshold, dimensions below this are ignored

        Returns:
            Natural language description of personality traits
        """
        descriptions = []
        for i, name in enumerate(self.dimension_names):
            value = self.vector[i]
            if abs(value) < threshold:
                continue

            if name in DIMENSION_LABELS_CN:
                label, neg_desc, pos_desc = DIMENSION_LABELS_CN[name]
                intensity = abs(value)
                if intensity > 0.7:
                    degree = "非常"
                elif intensity > 0.5:
                    degree = "比较"
                else:
                    degree = "略微"

                desc = pos_desc if value > 0 else neg_desc
                descriptions.append(f"{degree}{desc}")

        if not descriptions:
            return "Neutral personality traits, no distinct preferences"

        return "、".join(descriptions)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "dimensions": self.dimensions,
            "dimension_names": self.dimension_names,
            "values": self.vector.tolist(),
            "update_count": self._update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TraitVector":
        """Deserialize from dictionary"""
        tv = cls(
            dimensions=data.get("dimensions", 10),
            dimension_names=data.get("dimension_names"),
            initial_values=np.array(data.get("values", [])),
        )
        tv._update_count = data.get("update_count", 0)
        return tv

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TraitVector":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def __repr__() -> str:
        traits = ", ".join(
            f"{name}={self.vector[i]:.2f}"
            for i, name in enumerate(self.dimension_names)
        )
        return f"TraitVector({traits})"

    def __str__(self) -> str:
        return self.to_description()

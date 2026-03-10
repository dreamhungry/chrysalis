"""人格特征向量表示模块

将人格特征建模为多维连续向量，支持量化的人格状态表示和演化。
每个维度映射到具体人格属性，取值范围 [-1, 1]。
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# 默认的10维人格特征及其语义标签
DEFAULT_DIMENSION_NAMES = [
    "friendliness",   # 友好度: -1=冷淡 → 1=热情
    "formality",      # 正式度: -1=随意 → 1=正式
    "humor",          # 幽默感: -1=严肃 → 1=幽默
    "creativity",     # 创造性: -1=保守 → 1=创新
    "empathy",        # 共情力: -1=理性 → 1=感性
    "verbosity",      # 表达量: -1=简洁 → 1=详细
    "confidence",     # 自信度: -1=谦逊 → 1=自信
    "curiosity",      # 好奇心: -1=淡定 → 1=好奇
    "patience",       # 耐心度: -1=急躁 → 1=耐心
    "assertiveness",  # 主动性: -1=被动 → 1=主动
]

# 中文标签映射（用于展示和LLM提示构建）
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
    """人格特征向量表示

    封装多维特征向量及其语义标签，支持增量更新、归一化和序列化。

    Attributes:
        vector: NumPy向量，每维取值 [-1, 1]
        dimension_names: 各维度的语义名称列表
    """

    def __init__(
        self,
        dimensions: int = 10,
        dimension_names: Optional[List[str]] = None,
        initial_values: Optional[np.ndarray] = None,
    ):
        """初始化人格向量

        Args:
            dimensions: 向量维度数
            dimension_names: 各维度语义名称，为空则使用默认值
            initial_values: 初始值向量，为空则初始化为零向量
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
        """向量维度数"""
        return len(self.vector)

    def update(self, delta: np.ndarray, learning_rate: float = 0.1) -> None:
        """增量更新向量

        Args:
            delta: 更新方向向量
            learning_rate: 学习率，控制更新幅度
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
        """获取指定维度的特征值

        Args:
            name: 维度名称

        Returns:
            该维度的特征值 [-1, 1]
        """
        if name not in self.dimension_names:
            raise KeyError(f"Unknown trait dimension: {name}")
        idx = self.dimension_names.index(name)
        return float(self.vector[idx])

    def set_trait(self, name: str, value: float) -> None:
        """设置指定维度的特征值

        Args:
            name: 维度名称
            value: 特征值，会被裁剪到 [-1, 1]
        """
        if name not in self.dimension_names:
            raise KeyError(f"Unknown trait dimension: {name}")
        idx = self.dimension_names.index(name)
        self.vector[idx] = np.clip(value, -1.0, 1.0)

    def distance(self, other: "TraitVector") -> float:
        """计算与另一个人格向量的欧几里得距离"""
        return float(np.linalg.norm(self.vector - other.vector))

    def cosine_similarity(self, other: "TraitVector") -> float:
        """计算与另一个人格向量的余弦相似度"""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))

    def to_description(self, threshold: float = 0.3) -> str:
        """将人格向量转换为自然语言描述（中文）

        只描述绝对值超过阈值的显著特征。

        Args:
            threshold: 显著性阈值，绝对值低于此值的维度将被忽略

        Returns:
            人格特征的自然语言描述
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
            return "性格特征中性，无明显偏好"

        return "、".join(descriptions)

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "dimensions": self.dimensions,
            "dimension_names": self.dimension_names,
            "values": self.vector.tolist(),
            "update_count": self._update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TraitVector":
        """从字典反序列化"""
        tv = cls(
            dimensions=data.get("dimensions", 10),
            dimension_names=data.get("dimension_names"),
            initial_values=np.array(data.get("values", [])),
        )
        tv._update_count = data.get("update_count", 0)
        return tv

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TraitVector":
        """从JSON字符串反序列化"""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        traits = ", ".join(
            f"{name}={self.vector[i]:.2f}"
            for i, name in enumerate(self.dimension_names)
        )
        return f"TraitVector({traits})"

    def __str__(self) -> str:
        return self.to_description()

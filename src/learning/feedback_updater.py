"""基于反馈的增量更新器

将用户反馈（正/负）转化为人格向量的梯度调整。
"""

import logging
from typing import Dict, Optional

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class FeedbackUpdater:
    """基于反馈的人格增量更新

    将用户反馈值转换为梯度方向，沿用户偏好方向微调人格向量。
    支持自适应学习率：初期快速适应，后期趋于稳定。
    """

    def __init__(
        self,
        base_learning_rate: float = 0.1,
        decay_factor: float = 0.995,
        min_learning_rate: float = 0.01,
    ):
        """初始化反馈更新器

        Args:
            base_learning_rate: 初始学习率
            decay_factor: 学习率衰减因子（每次更新后乘以此值）
            min_learning_rate: 最小学习率下界
        """
        self.base_learning_rate = base_learning_rate
        self.current_learning_rate = base_learning_rate
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate
        self._update_count = 0

    def compute_gradient(
        self, feedback: float, context: Optional[Dict] = None
    ) -> np.ndarray:
        """计算反馈对应的人格向量梯度

        正反馈（feedback > 0.5）增强当前特征方向，
        负反馈（feedback < 0.5）抑制当前特征方向。

        Args:
            feedback: 用户反馈值 [0, 1]
            context: 交互上下文信息（包含当前人格向量等）

        Returns:
            梯度向量
        """
        context = context or {}
        current_vector = context.get("personality_vector", np.zeros(10))

        if isinstance(current_vector, list):
            current_vector = np.array(current_vector)

        # 将 feedback [0, 1] 映射到 [-1, 1]
        direction = (feedback - 0.5) * 2.0

        # 梯度方向：正反馈增强当前特征，负反馈抑制
        if np.linalg.norm(current_vector) > 0.01:
            # 沿当前向量方向调整
            gradient = direction * current_vector / (np.linalg.norm(current_vector) + 1e-8)
        else:
            # 当前向量接近零时，使用随机微扰
            rng = np.random.default_rng()
            gradient = direction * rng.normal(0, 0.1, size=len(current_vector))

        # 添加探索噪声
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.02, size=len(gradient))
        gradient += noise

        return gradient

    def update(
        self, trait_vector: TraitVector, feedback: float, context: Optional[Dict] = None
    ) -> None:
        """执行一次反馈更新

        Args:
            trait_vector: 要更新的人格向量
            feedback: 用户反馈 [0, 1]
            context: 交互上下文
        """
        context = context or {}
        context["personality_vector"] = trait_vector.vector.copy()

        gradient = self.compute_gradient(feedback, context)
        trait_vector.update(gradient, self.current_learning_rate)

        # 衰减学习率
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

"""强化学习更新器

使用简化的策略梯度方法，基于用户反馈信号更新人格向量。
将反馈作为奖励信号，人格向量作为策略参数进行优化。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class RLUpdater:
    """强化学习人格更新器

    将人格演化建模为策略优化问题：
    - 状态: 对话上下文
    - 动作: 人格向量方向
    - 奖励: 用户反馈
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        gamma: float = 0.95,
        baseline_decay: float = 0.99,
    ):
        """初始化RL更新器

        Args:
            learning_rate: 策略学习率
            gamma: 折扣因子（用于加权历史奖励）
            baseline_decay: 奖励基线的指数移动平均衰减率
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.baseline_decay = baseline_decay
        self._reward_baseline = 0.5  # 初始奖励基线
        self._episode_buffer: List[Tuple[np.ndarray, float]] = []

    def record_step(self, personality_vector: np.ndarray, reward: float) -> None:
        """记录一步交互的状态-奖励对

        Args:
            personality_vector: 当时的人格向量（快照）
            reward: 用户反馈奖励 [0, 1]
        """
        self._episode_buffer.append((personality_vector.copy(), reward))

    def compute_returns(self) -> List[float]:
        """计算折扣回报

        Returns:
            每步的折扣回报列表
        """
        returns = []
        G = 0.0
        for _, reward in reversed(self._episode_buffer):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, trait_vector: TraitVector) -> Optional[float]:
        """基于累积经验执行策略更新

        使用REINFORCE算法的简化版本。

        Args:
            trait_vector: 要更新的人格向量

        Returns:
            平均回报，如果缓冲区为空则返回None
        """
        if not self._episode_buffer:
            return None

        returns = self.compute_returns()
        avg_return = np.mean(returns)

        # 更新奖励基线
        self._reward_baseline = (
            self.baseline_decay * self._reward_baseline
            + (1 - self.baseline_decay) * avg_return
        )

        # 计算策略梯度
        gradient = np.zeros(trait_vector.dimensions)
        for i, (state_vector, _) in enumerate(self._episode_buffer):
            advantage = returns[i] - self._reward_baseline

            # 简化的策略梯度：用状态向量的变化方向作为梯度
            direction = state_vector / (np.linalg.norm(state_vector) + 1e-8)
            gradient += advantage * direction

        # 归一化梯度
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / (np.linalg.norm(gradient) + 1e-8)

        # 应用更新
        trait_vector.update(gradient, self.learning_rate)

        # 清空缓冲区
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
        """当前缓冲区中的步数"""
        return len(self._episode_buffer)

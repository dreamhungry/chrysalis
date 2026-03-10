"""交互历史模式提取器

从大量交互记录中统计分析有意义的行为模式，
并将其映射为人格向量的调整建议。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


class PatternExtractor:
    """交互模式提取器

    通过统计分析提取对话风格、用户偏好等模式，
    将其转换为人格向量调整方向。
    """

    def __init__(self, analysis_window: int = 50):
        """初始化模式提取器

        Args:
            analysis_window: 分析窗口大小（使用最近N条交互）
        """
        self.analysis_window = analysis_window

    def extract_patterns(self, interactions: List[Dict]) -> Dict[str, float]:
        """从交互历史中提取行为模式

        Args:
            interactions: 交互记录列表

        Returns:
            模式特征字典，key为特征名，value为特征值
        """
        if not interactions:
            return {}

        recent = interactions[-self.analysis_window :]

        patterns = {}

        # 1. 平均响应长度 → 映射到 verbosity
        avg_response_len = np.mean(
            [len(r.get("agent_response", "")) for r in recent]
        )
        patterns["avg_response_length"] = avg_response_len

        # 2. 平均用户输入长度
        avg_input_len = np.mean(
            [len(r.get("user_input", "")) for r in recent]
        )
        patterns["avg_input_length"] = avg_input_len

        # 3. 反馈统计
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

        # 4. 问号频率（用户提问倾向）
        question_count = sum(
            1
            for r in recent
            if "?" in r.get("user_input", "") or "？" in r.get("user_input", "")
        )
        patterns["question_ratio"] = question_count / len(recent) if recent else 0

        # 5. 交互频率（如果有时间戳）
        patterns["interaction_count"] = len(recent)

        logger.debug("Extracted patterns: %s", patterns)
        return patterns

    def patterns_to_adjustment(
        self, patterns: Dict[str, float], current_vector: TraitVector
    ) -> np.ndarray:
        """将提取的模式转换为人格向量调整

        Args:
            patterns: 模式特征字典
            current_vector: 当前人格向量

        Returns:
            调整向量
        """
        adjustment = np.zeros(current_vector.dimensions)

        # 根据用户反馈调整友好度
        avg_feedback = patterns.get("avg_feedback", 0.5)
        if avg_feedback > 0.6:
            adjustment[0] += 0.1  # friendliness +
        elif avg_feedback < 0.4:
            adjustment[0] -= 0.1  # friendliness -

        # 根据响应长度偏好调整表达量
        avg_len = patterns.get("avg_response_length", 100)
        if avg_len > 200:
            adjustment[5] += 0.05  # verbosity +
        elif avg_len < 50:
            adjustment[5] -= 0.05  # verbosity -

        # 根据提问频率调整好奇心/耐心
        question_ratio = patterns.get("question_ratio", 0.5)
        if question_ratio > 0.6:
            adjustment[7] += 0.05  # curiosity +
            adjustment[8] += 0.05  # patience +

        # 根据正面反馈比例整体强化
        pos_ratio = patterns.get("positive_feedback_ratio", 0.5)
        if pos_ratio > 0.7:
            # 高正面反馈 → 维持当前方向
            adjustment += 0.02 * current_vector.vector
        elif pos_ratio < 0.3:
            # 低正面反馈 → 反向调整
            adjustment -= 0.02 * current_vector.vector

        return adjustment

    def update_from_patterns(
        self,
        trait_vector: TraitVector,
        interactions: List[Dict],
        learning_rate: float = 0.05,
    ) -> Dict[str, float]:
        """执行一次基于模式的人格更新

        Args:
            trait_vector: 要更新的人格向量
            interactions: 交互历史
            learning_rate: 学习率

        Returns:
            提取的模式特征
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

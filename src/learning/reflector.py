"""自我反思模块

使用决策后端（如LLM）对交互历史进行反思分析，
提取人格调整建议。当决策后端不可用时，使用统计方法替代。
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

from ..decision.base import DecisionBackend
from ..personality.trait_vector import DIMENSION_LABELS_CN, TraitVector

logger = logging.getLogger(__name__)


class Reflector:
    """自我反思器

    分析交互历史，生成人格调整建议。
    支持LLM辅助反思和统计分析两种模式。
    """

    def __init__(self, decision_backend: Optional[DecisionBackend] = None):
        """初始化反思器

        Args:
            decision_backend: 决策后端（可选，用于LLM辅助反思）
        """
        self.decision_backend = decision_backend

    def reflect(
        self,
        trait_vector: TraitVector,
        interactions: List[Dict],
        learning_rate: float = 0.05,
    ) -> Optional[Dict]:
        """执行自我反思并更新人格向量

        Args:
            trait_vector: 当前人格向量
            interactions: 交互历史
            learning_rate: 学习率

        Returns:
            反思结果字典，包含分析和调整信息
        """
        if not interactions:
            return None

        # 根据后端类型选择反思方式
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
        """使用LLM进行自我反思"""
        # 构建反思prompt
        reflection_prompt = self._build_reflection_prompt(trait_vector, interactions)

        try:
            response = self.decision_backend.generate_response(
                user_input=reflection_prompt,
                personality_vector=np.zeros(trait_vector.dimensions),  # 反思时不注入人格
                conversation_history=[],
            )

            # 尝试解析LLM输出为调整向量
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
        """使用统计方法进行反思"""
        feedbacks = [
            r["feedback"]
            for r in interactions
            if r.get("feedback") is not None
        ]

        adjustment = np.zeros(trait_vector.dimensions)
        analysis = "统计反思分析:\n"

        if feedbacks:
            avg_feedback = np.mean(feedbacks)
            recent_feedbacks = feedbacks[-10:]
            recent_avg = np.mean(recent_feedbacks) if recent_feedbacks else avg_feedback

            analysis += f"- 总体平均反馈: {avg_feedback:.2f}\n"
            analysis += f"- 近期平均反馈: {recent_avg:.2f}\n"

            # 如果近期反馈改善 → 强化当前方向
            if recent_avg > avg_feedback + 0.05:
                adjustment = 0.05 * trait_vector.vector
                analysis += "- 趋势: 近期表现改善，强化当前人格方向\n"
            # 如果近期反馈下降 → 小幅回调
            elif recent_avg < avg_feedback - 0.05:
                adjustment = -0.03 * trait_vector.vector
                analysis += "- 趋势: 近期表现下降，小幅回调人格方向\n"
            else:
                analysis += "- 趋势: 表现稳定，维持当前状态\n"

            # 反馈方差分析
            if np.std(feedbacks) > 0.3:
                analysis += "- 注意: 反馈波动较大，需要更多数据观察\n"
        else:
            analysis += "- 暂无用户反馈数据\n"

        return {
            "method": "statistical_reflection",
            "analysis": analysis,
            "adjustment": adjustment.tolist(),
        }

    def _build_reflection_prompt(
        self, trait_vector: TraitVector, interactions: List[Dict]
    ) -> str:
        """构建LLM反思提示"""
        current_desc = trait_vector.to_description()

        # 取最近几条交互作为样本
        recent = interactions[-5:]
        conversation_samples = ""
        for r in recent:
            conversation_samples += f"用户: {r.get('user_input', '')}\n"
            conversation_samples += f"助手: {r.get('agent_response', '')}\n"
            if r.get("feedback") is not None:
                conversation_samples += f"反馈评分: {r['feedback']:.2f}\n"
            conversation_samples += "\n"

        feedbacks = [
            r["feedback"]
            for r in interactions
            if r.get("feedback") is not None
        ]
        avg_feedback = np.mean(feedbacks) if feedbacks else 0.5

        prompt = (
            f"你是一个AI人格分析师。请分析以下对话历史和反馈数据，"
            f"为AI助手的人格特征调整提供建议。\n\n"
            f"当前人格特征: {current_desc}\n"
            f"平均用户反馈: {avg_feedback:.2f}\n\n"
            f"近期对话样本:\n{conversation_samples}\n"
            f"请用JSON格式返回人格调整建议，格式为:\n"
            f'{{"adjustment": [v0, v1, ..., v9]}}\n'
            f"其中每个值在 [-0.2, 0.2] 范围内，"
            f"正数表示增强，负数表示减弱。"
        )

        return prompt

    def _parse_reflection_response(
        self, response: str, dimensions: int
    ) -> np.ndarray:
        """解析LLM反思响应为调整向量"""
        try:
            # 尝试从响应中提取JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                if "adjustment" in data:
                    adj = np.array(data["adjustment"][:dimensions], dtype=np.float64)
                    # 限制调整幅度
                    adj = np.clip(adj, -0.2, 0.2)
                    # 补齐维度
                    if len(adj) < dimensions:
                        adj = np.pad(adj, (0, dimensions - len(adj)))
                    return adj
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse reflection response: %s", e)

        return np.zeros(dimensions)

"""Agent Core - 代理核心对话管理

整合决策后端、人格向量、记忆存储和学习模块，
提供统一的对话处理接口。
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

# 每隔多少次交互触发一次模式学习
PATTERN_LEARNING_INTERVAL = 50
# 每隔多少次交互触发一次RL更新
RL_UPDATE_INTERVAL = 20
# 每隔多少次交互触发一次反思
REFLECTION_INTERVAL = 30


class AgentCore:
    """AI代理核心

    管理对话流程、人格驱动的响应生成和多种学习机制的调度。
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
        """初始化Agent Core

        Args:
            decision_backend: 决策后端
            trait_vector: 人格向量
            interaction_store: 交互记录存储
            personality_store: 人格持久化
            feedback_updater: 反馈更新器
            pattern_extractor: 模式提取器
            rl_updater: RL更新器
            reflector: 反思器
        """
        self.decision = decision_backend
        self.personality = trait_vector
        self.memory = interaction_store
        self.personality_store = personality_store

        # 学习模块
        self.feedback_updater = feedback_updater or FeedbackUpdater()
        self.pattern_extractor = pattern_extractor or PatternExtractor()
        self.rl_updater = rl_updater or RLUpdater()
        self.reflector = reflector or Reflector(decision_backend)

        self._interaction_count = 0
        self._last_interaction_id: Optional[str] = None

    def chat(self, user_input: str) -> str:
        """处理用户输入，生成响应

        Args:
            user_input: 用户输入文本

        Returns:
            Agent响应文本
        """
        # 1. 获取对话历史
        conversation_history = self.memory.get_recent(10)

        # 2. 调用决策后端生成响应
        response = self.decision.generate_response(
            user_input=user_input,
            personality_vector=self.personality.vector,
            conversation_history=conversation_history,
        )

        # 3. 记录交互
        self._last_interaction_id = self.memory.add_interaction(
            user_input, response
        )

        # 4. 更新计数并触发周期性学习
        self._interaction_count += 1
        self._trigger_periodic_learning()

        logger.info(
            "Chat completed (interaction #%d, personality norm=%.3f)",
            self._interaction_count,
            np.linalg.norm(self.personality.vector),
        )

        return response

    def provide_feedback(self, feedback: float, interaction_id: Optional[str] = None) -> None:
        """接收用户反馈，触发学习

        Args:
            feedback: 反馈评分 [0, 1]
            interaction_id: 关联的交互ID，为空则使用最近一次交互
        """
        feedback = max(0.0, min(1.0, feedback))
        target_id = interaction_id or self._last_interaction_id

        # 更新交互记录中的反馈
        if target_id:
            self.memory.update_feedback(target_id, feedback)

        # 1. 基于反馈的增量更新
        self.feedback_updater.update(
            self.personality,
            feedback,
            context={"personality_vector": self.personality.vector.tolist()},
        )

        # 2. 记录RL步骤
        self.rl_updater.record_step(self.personality.vector, feedback)

        # 3. 保存人格状态
        self.personality_store.save(self.personality)
        self.personality_store.save_snapshot(
            self.personality, event=f"feedback_{feedback:.2f}"
        )

        logger.info(
            "Feedback processed: %.2f, personality updated", feedback
        )

    def get_personality_info(self) -> Dict:
        """获取当前人格状态信息

        Returns:
            人格信息字典
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
        """动态切换决策后端

        Args:
            new_backend: 新的决策后端实例
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
        """触发周期性学习机制"""
        # 模式学习
        if self._interaction_count % PATTERN_LEARNING_INTERVAL == 0:
            interactions = self.memory.get_recent(100)
            self.pattern_extractor.update_from_patterns(self.personality, interactions)
            self.personality_store.save(self.personality)
            self.personality_store.save_snapshot(
                self.personality, event="pattern_learning"
            )
            logger.info("Pattern learning triggered at interaction #%d", self._interaction_count)

        # RL更新
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

        # 反思
        if self._interaction_count % REFLECTION_INTERVAL == 0:
            interactions = self.memory.get_recent(20)
            self.reflector.reflect(self.personality, interactions)
            self.personality_store.save(self.personality)
            self.personality_store.save_snapshot(
                self.personality, event="reflection"
            )
            logger.info("Reflection triggered at interaction #%d", self._interaction_count)

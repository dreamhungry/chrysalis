"""系统初始化工厂

根据配置创建并组装所有模块实例，提供统一的系统启动入口。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .agent.agent_core import AgentCore
from .config import Config
from .decision.backends.llm_backend import LLMBackend
from .decision.base import DecisionBackend
from .learning.feedback_updater import FeedbackUpdater
from .learning.pattern_extractor import PatternExtractor
from .learning.reflector import Reflector
from .learning.rl_updater import RLUpdater
from .memory.interaction_store import InteractionStore
from .memory.memory_manager import MemoryManager
from .personality.personality_store import PersonalityStore
from .personality.trait_vector import TraitVector

logger = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    """配置日志系统"""
    log_path = config.get_absolute_path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def create_decision_backend(config: Config) -> DecisionBackend:
    """根据配置创建决策后端"""
    if config.decision_backend == "llm":
        return LLMBackend(
            llm_endpoint=config.llm_endpoint,
            model_name=config.llm_model_name,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
    else:
        raise ValueError(f"Unsupported decision backend: {config.decision_backend}")


def create_agent(config: Optional[Config] = None) -> AgentCore:
    """创建完整的Agent实例

    根据配置组装所有模块。

    Args:
        config: 配置对象，为空则加载默认配置

    Returns:
        组装完成的AgentCore实例
    """
    if config is None:
        config = Config.load()

    setup_logging(config)

    # 1. 创建人格向量和持久化
    personality_path = str(config.get_absolute_path(config.personality_storage_path))
    personality_store = PersonalityStore(personality_path)

    trait_vector = personality_store.load()
    if trait_vector is None:
        trait_vector = TraitVector(dimensions=config.personality_dimensions)
        logger.info("Initialized new personality vector (%d dimensions)", config.personality_dimensions)

    # 2. 创建记忆存储
    memory_path = str(config.get_absolute_path(config.memory_markdown_path))
    interaction_store = MemoryManager.create_store(
        config.memory_backend, storage_path=memory_path
    )

    # 3. 创建决策后端
    decision_backend = create_decision_backend(config)

    # 4. 创建学习模块
    feedback_updater = FeedbackUpdater(
        base_learning_rate=config.personality_learning_rate
    )
    pattern_extractor = PatternExtractor()
    rl_updater = RLUpdater()
    reflector = Reflector(decision_backend)

    # 5. 组装AgentCore
    agent = AgentCore(
        decision_backend=decision_backend,
        trait_vector=trait_vector,
        interaction_store=interaction_store,
        personality_store=personality_store,
        feedback_updater=feedback_updater,
        pattern_extractor=pattern_extractor,
        rl_updater=rl_updater,
        reflector=reflector,
    )

    # 保存初始人格状态
    personality_store.save(trait_vector)

    logger.info(
        "Agent created: backend=%s, personality=%s",
        config.decision_backend,
        trait_vector.to_description(),
    )

    return agent

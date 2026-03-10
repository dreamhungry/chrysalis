"""System Initialization Factory

Creates and assembles all module instances based on configuration, providing unified system startup entry point.
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
    """Configure logging system"""
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
    """Create decision backend based on configuration"""
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
    """Create complete Agent instance

    Assembles all modules based on configuration.

    Args:
        config: Config object, loads default config if None

    Returns:
        Assembled AgentCore instance
    """
    if config is None:
        config = Config.load()

    setup_logging(config)

    # 1. Create personality vector and persistence
    personality_path = str(config.get_absolute_path(config.personality_storage_path))
    personality_store = PersonalityStore(personality_path)

    trait_vector = personality_store.load()
    if trait_vector is None:
        trait_vector = TraitVector(dimensions=config.personality_dimensions)
        logger.info("Initialized new personality vector (%d dimensions)", config.personality_dimensions)

    # 2. Create memory storage
    memory_path = str(config.get_absolute_path(config.memory_markdown_path))
    interaction_store = MemoryManager.create_store(
        config.memory_backend, storage_path=memory_path
    )

    # 3. Create decision backend
    decision_backend = create_decision_backend(config)

    # 4. Create learning modules
    feedback_updater = FeedbackUpdater(
        base_learning_rate=config.personality_learning_rate
    )
    pattern_extractor = PatternExtractor()
    rl_updater = RLUpdater()
    reflector = Reflector(decision_backend)

    # 5. Assemble AgentCore
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

    # Save initial personality state
    personality_store.save(trait_vector)

    logger.info(
        "Agent created: backend=%s, personality=%s",
        config.decision_backend,
        trait_vector.to_description(),
    )

    return agent

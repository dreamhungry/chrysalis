"""人格持久化模块

负责将人格向量状态保存到JSON文件和从文件加载恢复。
同时维护人格向量的历史快照，用于追踪演化轨迹。
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .trait_vector import TraitVector

logger = logging.getLogger(__name__)


class PersonalityStore:
    """人格向量持久化管理器

    提供人格向量的保存、加载和历史快照管理功能。

    Attributes:
        storage_path: 人格向量JSON文件路径
        history_path: 人格演化历史JSON文件路径
    """

    def __init__(self, storage_path: str):
        """初始化持久化管理器

        Args:
            storage_path: JSON文件存储路径
        """
        self.storage_path = Path(storage_path)
        self.history_path = self.storage_path.parent / "evolution_history.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保存储目录存在"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trait_vector: TraitVector) -> None:
        """保存当前人格向量到文件

        Args:
            trait_vector: 要保存的人格向量
        """
        data = {
            "saved_at": datetime.now().isoformat(),
            "trait_vector": trait_vector.to_dict(),
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Personality saved to %s", self.storage_path)

    def load(self) -> Optional[TraitVector]:
        """从文件加载人格向量

        Returns:
            加载的人格向量，文件不存在时返回None
        """
        if not self.storage_path.exists():
            logger.info("No personality file found at %s", self.storage_path)
            return None

        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tv = TraitVector.from_dict(data["trait_vector"])
        logger.info("Personality loaded from %s", self.storage_path)
        return tv

    def save_snapshot(self, trait_vector: TraitVector, event: str = "") -> None:
        """保存人格向量的历史快照

        用于追踪人格演化轨迹。

        Args:
            trait_vector: 当前人格向量状态
            event: 触发快照的事件描述
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "trait_vector": trait_vector.to_dict(),
        }

        # 加载已有历史
        history = self._load_history()
        history.append(snapshot)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        logger.debug("Personality snapshot saved (event: %s)", event)

    def get_evolution_history(self) -> List[dict]:
        """获取人格演化历史

        Returns:
            按时间排序的历史快照列表
        """
        return self._load_history()

    def _load_history(self) -> List[dict]:
        """加载历史文件"""
        if not self.history_path.exists():
            return []

        with open(self.history_path, "r", encoding="utf-8") as f:
            return json.load(f)

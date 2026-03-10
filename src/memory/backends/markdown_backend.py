"""Markdown文件存储后端

将交互历史以Markdown格式追加写入文件，便于人工和LLM直接阅读。
同时在内存中维护结构化列表，支持程序化查询。
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import MemoryBackend

logger = logging.getLogger(__name__)


class MarkdownFileBackend(MemoryBackend):
    """Markdown文件存储实现

    交互记录以追加方式写入Markdown文件，格式便于人工审查和LLM读取。
    启动时解析Markdown文件恢复内存中的结构化数据。

    Markdown格式示例:
        ---
        ## 交互 2026-03-10 15:23:45
        **ID**: `uuid-here`
        **用户**: 你好
        **助手**: 你好！有什么可以帮你的？
        **反馈**: ⭐⭐⭐⭐⭐ (0.95)
    """

    def __init__(self, storage_path: str):
        """初始化Markdown存储后端

        Args:
            storage_path: Markdown文件路径
        """
        self.storage_path = Path(storage_path)
        self.interactions: List[Dict] = []
        self._ensure_file_exists()
        self._load_from_file()

    def _ensure_file_exists(self) -> None:
        """确保Markdown文件和目录存在"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as f:
                f.write("# Chrysalis 交互历史记录\n\n")
                f.write(
                    "> 本文件记录AI代理与用户的所有交互历史，包括对话内容和用户反馈。\n\n"
                )
            logger.info("Created new interaction history file: %s", self.storage_path)

    def add_interaction(
        self,
        user_input: str,
        agent_response: str,
        feedback: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """追加交互记录到Markdown文件"""
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # 构建Markdown格式记录
        md_content = "\n---\n\n"
        md_content += (
            f"## 交互 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        md_content += f"**ID**: `{interaction_id}`\n\n"
        md_content += f"**用户**:\n\n{user_input}\n\n"
        md_content += f"**助手**:\n\n{agent_response}\n\n"

        if feedback is not None:
            feedback = max(0.0, min(1.0, feedback))
            star_count = int(feedback * 5)
            stars = "⭐" * star_count
            md_content += f"**反馈**: {stars} ({feedback:.2f})\n\n"

        if metadata:
            md_content += (
                f"**元数据**: `{json.dumps(metadata, ensure_ascii=False)}`\n\n"
            )

        # 追加写入文件
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(md_content)

        # 同时保存到内存列表
        record = {
            "id": interaction_id,
            "timestamp": timestamp.isoformat(),
            "user_input": user_input,
            "agent_response": agent_response,
            "feedback": feedback,
            "metadata": metadata or {},
            "embedding": embedding,
        }
        self.interactions.append(record)

        logger.debug("Interaction added: %s", interaction_id)
        return interaction_id

    def update_feedback(self, interaction_id: str, feedback: float) -> bool:
        """更新指定交互记录的反馈

        在内存中更新反馈值，并追加一条反馈更新记录到Markdown文件。
        """
        feedback = max(0.0, min(1.0, feedback))

        for record in self.interactions:
            if record["id"] == interaction_id:
                record["feedback"] = feedback

                # 追加反馈更新到文件
                star_count = int(feedback * 5)
                stars = "⭐" * star_count
                md_content = (
                    f"\n> **反馈更新** [{datetime.now().strftime('%H:%M:%S')}] "
                    f"交互 `{interaction_id}`: {stars} ({feedback:.2f})\n\n"
                )
                with open(self.storage_path, "a", encoding="utf-8") as f:
                    f.write(md_content)

                logger.debug(
                    "Feedback updated for %s: %.2f", interaction_id, feedback
                )
                return True

        logger.warning("Interaction not found: %s", interaction_id)
        return False

    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近N条交互"""
        return self.interactions[-n:]

    def get_by_id(self, interaction_id: str) -> Optional[Dict]:
        """根据ID获取交互记录"""
        for record in self.interactions:
            if record["id"] == interaction_id:
                return record
        return None

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """Markdown后端不支持语义搜索，回退到返回最近记录"""
        logger.debug(
            "Markdown backend does not support semantic search, falling back to recent"
        )
        return self.get_recent(top_k)

    def get_all_feedbacks(self) -> List[Dict]:
        """获取所有包含反馈的交互记录"""
        return [r for r in self.interactions if r.get("feedback") is not None]

    def count(self) -> int:
        """获取交互记录总数"""
        return len(self.interactions)

    def save(self) -> None:
        """Markdown采用追加写入模式，无需额外的save操作"""
        pass

    def _load_from_file(self) -> None:
        """从Markdown文件解析加载历史记录到内存"""
        if not self.storage_path.exists():
            return

        with open(self.storage_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 使用正则提取交互块
        # 匹配格式: ## 交互 YYYY-MM-DD HH:MM:SS
        block_pattern = r"---\s*\n\n## 交互 (.+?)\n"
        blocks = list(re.finditer(block_pattern, content))

        for i, match in enumerate(blocks):
            start = match.start()
            end = blocks[i + 1].start() if i + 1 < len(blocks) else len(content)
            block_text = content[start:end]

            timestamp_str = match.group(1).strip()

            # 提取ID
            id_match = re.search(r"\*\*ID\*\*:\s*`(.+?)`", block_text)
            interaction_id = id_match.group(1) if id_match else str(uuid.uuid4())

            # 提取用户输入
            user_match = re.search(
                r"\*\*用户\*\*:\s*\n\n(.*?)(?=\n\n\*\*助手\*\*:)", block_text, re.DOTALL
            )
            user_input = user_match.group(1).strip() if user_match else ""

            # 提取助手响应
            agent_match = re.search(
                r"\*\*助手\*\*:\s*\n\n(.*?)(?=\n\n(?:\*\*反馈\*\*|\*\*元数据\*\*|---|\Z))",
                block_text,
                re.DOTALL,
            )
            agent_response = agent_match.group(1).strip() if agent_match else ""

            # 提取反馈
            feedback_match = re.search(
                r"\*\*反馈\*\*:.*?\((\d+\.\d+)\)", block_text
            )
            feedback = float(feedback_match.group(1)) if feedback_match else None

            # 提取元数据
            metadata_match = re.search(r"\*\*元数据\*\*:\s*`(.+?)`", block_text)
            metadata = {}
            if metadata_match:
                try:
                    metadata = json.loads(metadata_match.group(1))
                except json.JSONDecodeError:
                    pass

            record = {
                "id": interaction_id,
                "timestamp": timestamp_str,
                "user_input": user_input,
                "agent_response": agent_response,
                "feedback": feedback,
                "metadata": metadata,
                "embedding": None,
            }
            self.interactions.append(record)

        logger.info(
            "Loaded %d interactions from %s", len(self.interactions), self.storage_path
        )

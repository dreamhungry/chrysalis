"""Markdown File Storage Backend

Appends interaction history to file in Markdown format for easy reading by humans and LLMs.
Also maintains a structured list in memory for programmatic queries.
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
    """Markdown File Storage Implementation

    Interaction records are appended to Markdown file in a format easy for humans
    to review and LLMs to read. Parses Markdown file on startup to restore
    structured data in memory.

    Markdown format example:
        ---
        ## Interaction 2026-03-10 15:23:45
        **ID**: `uuid-here`
        **User**: Hello
        **Assistant**: Hello! How can I help you?
        **Feedback**: ⭐⭐⭐⭐⭐ (0.95)
    """

    def __init__(self, storage_path: str):
        """Initialize Markdown storage backend

        Args:
            storage_path: Markdown file path
        """
        self.storage_path = Path(storage_path)
        self.interactions: List[Dict] = []
        self._ensure_file_exists()
        self._load_from_file()

    def _ensure_file_exists(self) -> None:
        """Ensure Markdown file and directory exist"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as f:
                f.write("# Chrysalis Interaction History\n\n")
                f.write(
                    "> This file records all interaction history between the AI agent and users, including conversation content and user feedback.\n\n"
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
        """Append interaction record to Markdown file"""
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Build Markdown format record
        md_content = "\n---\n\n"
        md_content += (
            f"## Interaction {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        md_content += f"**ID**: `{interaction_id}`\n\n"
        md_content += f"**User**:\n\n{user_input}\n\n"
        md_content += f"**Assistant**:\n\n{agent_response}\n\n"

        if feedback is not None:
            feedback = max(0.0, min(1.0, feedback))
            star_count = int(feedback * 5)
            stars = "⭐" * star_count
            md_content += f"**Feedback**: {stars} ({feedback:.2f})\n\n"

        if metadata:
            md_content += (
                f"**Metadata**: `{json.dumps(metadata, ensure_ascii=False)}`\n\n"
            )

        # Append to file
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(md_content)

        # Also save to in-memory list
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
        """Update feedback for a specific interaction record

        Updates feedback value in memory and appends a feedback update record to Markdown file.
        """
        feedback = max(0.0, min(1.0, feedback))

        for record in self.interactions:
            if record["id"] == interaction_id:
                record["feedback"] = feedback

                # Append feedback update to file
                star_count = int(feedback * 5)
                stars = "⭐" * star_count
                md_content = (
                    f"\n> **Feedback Update** [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Interaction `{interaction_id}`: {stars} ({feedback:.2f})\n\n"
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
        """Get the most recent N interactions"""
        return self.interactions[-n:]

    def get_by_id(self, interaction_id: str) -> Optional[Dict]:
        """Get interaction record by ID"""
        for record in self.interactions:
            if record["id"] == interaction_id:
                return record
        return None

    def search_similar(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict]:
        """Markdown backend does not support semantic search, falls back to returning recent records"""
        logger.debug(
            "Markdown backend does not support semantic search, falling back to recent"
        )
        return self.get_recent(top_k)

    def get_all_feedbacks(self) -> List[Dict]:
        """Get all interaction records containing feedback"""
        return [r for r in self.interactions if r.get("feedback") is not None]

    def count(self) -> int:
        """Get total number of interaction records"""
        return len(self.interactions)

    def save(self) -> None:
        """Markdown uses append mode, no additional save operation needed"""
        pass

    def _load_from_file(self) -> None:
        """Parse and load history records from Markdown file to memory"""
        if not self.storage_path.exists():
            return

        with open(self.storage_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract interaction blocks using regex
        # Match format: ## Interaction YYYY-MM-DD HH:MM:SS
        block_pattern = r"---\s*\n\n## Interaction (.+?)\n"
        blocks = list(re.finditer(block_pattern, content))

        for i, match in enumerate(blocks):
            start = match.start()
            end = blocks[i + 1].start() if i + 1 < len(blocks) else len(content)
            block_text = content[start:end]

            timestamp_str = match.group(1).strip()

            # Extract ID
            id_match = re.search(r"\*\*ID\*\*:\s*`(.+?)`", block_text)
            interaction_id = id_match.group(1) if id_match else str(uuid.uuid4())

            # Extract user input
            user_match = re.search(
                r"\*\*User\*\*:\s*\n\n(.*?)(?=\n\n\*\*Assistant\*\*:)", block_text, re.DOTALL
            )
            user_input = user_match.group(1).strip() if user_match else ""

            # Extract assistant response
            agent_match = re.search(
                r"\*\*Assistant\*\*:\s*\n\n(.*?)(?=\n\n(?:\*\*Feedback\*\*|\*\*Metadata\*\*|---|\Z))",
                block_text,
                re.DOTALL,
            )
            agent_response = agent_match.group(1).strip() if agent_match else ""

            # Extract feedback
            feedback_match = re.search(
                r"\*\*Feedback\*\*:.*?\((\d+\.\d+)\)", block_text
            )
            feedback = float(feedback_match.group(1)) if feedback_match else None

            # Extract metadata
            metadata_match = re.search(r"\*\*Metadata\*\*:\s*`(.+?)`", block_text)
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

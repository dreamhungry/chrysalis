"""Personality Persistence Module

Responsible for saving personality vector state to JSON file and loading from file.
Also maintains personality vector history snapshots for tracking evolution trajectory.
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
    """Personality Vector Persistence Manager

    Provides functionality for saving, loading, and managing history snapshots
    of personality vectors.

    Attributes:
        storage_path: Path to the personality vector JSON file
        history_path: Path to the personality evolution history JSON file
    """

    def __init__(self, storage_path: str):
        """Initialize persistence manager

        Args:
            storage_path: JSON file storage path
        """
        self.storage_path = Path(storage_path)
        self.history_path = self.storage_path.parent / "evolution_history.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure storage directory exists"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trait_vector: TraitVector) -> None:
        """Save current personality vector to file

        Args:
            trait_vector: Personality vector to save
        """
        data = {
            "saved_at": datetime.now().isoformat(),
            "trait_vector": trait_vector.to_dict(),
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Personality saved to %s", self.storage_path)

    def load(self) -> Optional[TraitVector]:
        """Load personality vector from file

        Returns:
            Loaded personality vector, or None if file doesn't exist
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
        """Save a history snapshot of personality vector

        Used for tracking personality evolution trajectory.

        Args:
            trait_vector: Current personality vector state
            event: Description of the event triggering the snapshot
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "trait_vector": trait_vector.to_dict(),
        }

        # Load existing history
        history = self._load_history()
        history.append(snapshot)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        logger.debug("Personality snapshot saved (event: %s)", event)

    def get_evolution_history(self) -> List[dict]:
        """Get personality evolution history

        Returns:
            List of history snapshots sorted by time
        """
        return self._load_history()

    def _load_history(self) -> List[dict]:
        """Load history file"""
        if not self.history_path.exists():
            return []

        with open(self.history_path, "r", encoding="utf-8") as f:
            return json.load(f)

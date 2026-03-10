"""Configuration Management Class - Load config from environment variables"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration Management Class - Load config from .env file and environment variables"""

    def __init__(self):
        # Load .env file (from project root)
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
        load_dotenv(env_path)

        # Project root directory
        self.project_root = project_root

        # Decision backend configuration
        self.decision_backend = os.getenv("DECISION_BACKEND", "llm")
        self.llm_endpoint = os.getenv(
            "LLM_ENDPOINT", "http://localhost:8000/v1/gentext"
        )
        self.llm_model_name = os.getenv(
            "LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "256"))

        # Memory storage configuration
        self.memory_backend = os.getenv("MEMORY_BACKEND", "markdown")
        self.memory_markdown_path = os.getenv(
            "MEMORY_MARKDOWN_PATH", "data/interactions/history.md"
        )

        # Personality configuration
        self.personality_dimensions = int(os.getenv("PERSONALITY_DIMENSIONS", "10"))
        self.personality_learning_rate = float(
            os.getenv("PERSONALITY_LEARNING_RATE", "0.1")
        )
        self.personality_storage_path = os.getenv(
            "PERSONALITY_STORAGE_PATH", "data/personalities/current.json"
        )

        # System configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_path = os.getenv("LOG_PATH", "logs/chrysalis.log")
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8080"))

    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path"""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root / p

    @classmethod
    def load(cls) -> "Config":
        """Factory method - load config"""
        return cls()


# Global config instance
config = Config.load()

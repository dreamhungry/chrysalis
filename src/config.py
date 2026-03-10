"""配置管理类 - 从环境变量加载配置"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """配置管理类 - 从.env文件和环境变量加载配置"""

    def __init__(self):
        # 加载.env文件 (从项目根目录)
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
        load_dotenv(env_path)

        # 项目根目录
        self.project_root = project_root

        # 决策后端配置
        self.decision_backend = os.getenv("DECISION_BACKEND", "llm")
        self.llm_endpoint = os.getenv(
            "LLM_ENDPOINT", "http://localhost:8000/v1/gentext"
        )
        self.llm_model_name = os.getenv(
            "LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "256"))

        # 记忆存储配置
        self.memory_backend = os.getenv("MEMORY_BACKEND", "markdown")
        self.memory_markdown_path = os.getenv(
            "MEMORY_MARKDOWN_PATH", "data/interactions/history.md"
        )

        # 人格配置
        self.personality_dimensions = int(os.getenv("PERSONALITY_DIMENSIONS", "10"))
        self.personality_learning_rate = float(
            os.getenv("PERSONALITY_LEARNING_RATE", "0.1")
        )
        self.personality_storage_path = os.getenv(
            "PERSONALITY_STORAGE_PATH", "data/personalities/current.json"
        )

        # 系统配置
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_path = os.getenv("LOG_PATH", "logs/chrysalis.log")
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8080"))

    def get_absolute_path(self, relative_path: str) -> Path:
        """将相对路径转为绝对路径"""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root / p

    @classmethod
    def load(cls) -> "Config":
        """工厂方法 - 加载配置"""
        return cls()


# 全局配置实例
config = Config.load()

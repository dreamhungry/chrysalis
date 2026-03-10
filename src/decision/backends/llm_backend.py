"""LLM Decision Backend

Decision backend implementation based on Large Language Models.
Supports multiple LLM providers via LangChain unified interface.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from ..base import DecisionBackend
from ..prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM provider types
    
    All providers are integrated via LangChain for unified interface.
    """
    # OpenAI-compatible providers (via langchain-openai)
    OPENAI = "openai"
    VLLM = "vllm"
    KIMI = "kimi"
    OPENROUTER = "openrouter"
    AIHUBMIX = "aihubmix"
    
    # Custom providers (via dedicated LangChain packages)
    OLLAMA = "ollama"
    GEMINI = "gemini"
    CLAUDE = "claude"
    MINIMAX = "minimax"


class LLMBackend(DecisionBackend):
    """LLM Decision Backend

    Supports multiple LLM providers via LangChain unified interface.
    
    Supported providers:
    - OpenAI-compatible: OpenAI, vLLM, Kimi, OpenRouter, AihubMix
    - Dedicated packages: Ollama, Gemini, Claude, Minimax
    """
    
    # Provider-specific default endpoints
    DEFAULT_ENDPOINTS = {
        LLMProvider.OPENAI: "https://api.openai.com/v1",
        LLMProvider.VLLM: "http://localhost:8000/v1",
        LLMProvider.KIMI: "https://api.moonshot.cn/v1",
        LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
        LLMProvider.AIHUBMIX: "https://aihubmix.com/v1",
        LLMProvider.OLLAMA: "http://localhost:11434",
        LLMProvider.GEMINI: None,  # Gemini doesn't need base_url
        LLMProvider.CLAUDE: "https://api.anthropic.com/v1",
        LLMProvider.MINIMAX: "https://api.minimax.chat/v1",
    }

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 256,
        base_system_prompt: str = "",
        **kwargs,
    ):
        """Initialize LLM backend

        Args:
            model_name: Model name
            provider: Provider type (openai/vllm/kimi/ollama/gemini/claude/etc.)
            base_url: Base URL (auto-detect if None, not needed for Gemini)
            api_key: API key for authentication
            temperature: Generation temperature
            max_tokens: Maximum generation tokens
            base_system_prompt: Base system prompt
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.provider = LLMProvider(provider.lower())
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = PromptBuilder(base_system_prompt)
        self.extra_params = kwargs
        
        # Auto-detect base_url if not provided
        if not base_url:  # Handles None or empty string
            base_url = self.DEFAULT_ENDPOINTS.get(self.provider)
            if base_url:
                logger.info(f"Using default base_url for {self.provider}: {base_url}")
        
        self.base_url = base_url.rstrip("/") if base_url else None
        
        # Create LangChain chat model
        self.chat_model = self._create_chat_model()
        
        logger.info(
            f"Initialized LLM backend: provider={self.provider}, "
            f"model={self.model_name}, base_url={self.base_url}"
        )

    def _create_chat_model(self):
        """Create LangChain chat model based on provider
        
        Returns:
            LangChain ChatModel instance
        """
        try:
            # OpenAI-compatible providers
            if self.provider in [
                LLMProvider.OPENAI,
                LLMProvider.VLLM,
                LLMProvider.KIMI,
                LLMProvider.OPENROUTER,
                LLMProvider.AIHUBMIX,
            ]:
                from langchain_openai import ChatOpenAI
                
                kwargs = {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                    
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                
                # OpenRouter specific headers
                if self.provider == LLMProvider.OPENROUTER:
                    kwargs["default_headers"] = {
                        "HTTP-Referer": self.extra_params.get("site_url", ""),
                        "X-Title": self.extra_params.get("site_name", "Chrysalis"),
                    }
                
                return ChatOpenAI(**kwargs)
            
            # Ollama
            elif self.provider == LLMProvider.OLLAMA:
                from langchain_ollama import ChatOllama
                
                kwargs = {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
                
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                
                return ChatOllama(**kwargs)
            
            # Google Gemini
            elif self.provider == LLMProvider.GEMINI:
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                if not self.api_key:
                    raise ValueError("Gemini requires api_key")
                
                return ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            
            # Anthropic Claude
            elif self.provider == LLMProvider.CLAUDE:
                from langchain_anthropic import ChatAnthropic
                
                if not self.api_key:
                    raise ValueError("Claude requires api_key")
                
                kwargs = {
                    "model": self.model_name,
                    "anthropic_api_key": self.api_key,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                
                return ChatAnthropic(**kwargs)
            
            # Minimax
            elif self.provider == LLMProvider.MINIMAX:
                from langchain_community.chat_models import ChatMinimax
                
                if not self.api_key:
                    raise ValueError("Minimax requires api_key")
                
                kwargs = {
                    "model": self.model_name,
                    "minimax_api_key": self.api_key,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                
                if "group_id" in self.extra_params:
                    kwargs["minimax_group_id"] = self.extra_params["group_id"]
                
                return ChatMinimax(**kwargs)
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except ImportError as e:
            logger.error(f"Failed to import LangChain package for {self.provider}: {e}")
            raise ImportError(
                f"Please install required package for {self.provider}:\n"
                f"  pip install langchain-{self.provider.value}"
            ) from e

    def generate_response(
        self,
        user_input: str,
        personality_vector: np.ndarray,
        conversation_history: List[Dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response using LLM via LangChain"""
        # Build system prompt with personality
        system_prompt = self.prompt_builder.build_system_prompt(personality_vector)
        
        # Build message history
        messages = self._build_messages(
            system_prompt, conversation_history, user_input
        )

        # Call LLM via LangChain
        try:
            response = self.chat_model.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return f"[System Error] Unable to generate response: {e}"

    def _build_messages(
        self,
        system_prompt: str,
        conversation_history: List[Dict],
        user_input: str,
        max_history: int = 10,
    ) -> List:
        """Build LangChain message list
        
        Args:
            system_prompt: System prompt with personality
            conversation_history: Conversation history records
            user_input: Current user input
            max_history: Maximum history turns
            
        Returns:
            List of LangChain Message objects
        """
        messages = [SystemMessage(content=system_prompt)]
        
        # Add recent conversation history
        recent_history = conversation_history[-max_history:]
        for record in recent_history:
            messages.append(HumanMessage(content=record.get("user_input", "")))
            messages.append(
                SystemMessage(content=record.get("agent_response", ""))
            )
        
        # Add current user input
        messages.append(HumanMessage(content=user_input))
        
        return messages

    def update_model(self, training_data: List[Dict]) -> bool:
        """LLM typically does not support online updates"""
        logger.debug("LLM backend does not support online model updates")
        return False

    def evaluate_action(
        self, action: str, state: Dict, personality_vector: np.ndarray
    ) -> float:
        """LLM backend returns maximum utility by default"""
        return 1.0

    def get_backend_info(self) -> Dict[str, Any]:
        """Return backend information"""
        return {
            "type": "llm",
            "provider": self.provider.value,
            "model": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "langchain_version": True,
        }

    def close(self) -> None:
        """Cleanup resources (no-op for LangChain)"""
        pass

    def __del__(self):
        """Destructor (no-op for LangChain)"""
        pass


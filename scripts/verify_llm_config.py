#!/usr/bin/env python
"""Verify LLM configuration and test connection"""

import sys
import os
from pathlib import Path

# Setup project root and Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Must insert parent of src to import as package
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Change to project root for .env loading

# Now import from src package
from src.config import Config
from src.decision.backends.llm_backend import LLMBackend


def main():
    """Verify LLM configuration from .env file"""
    print("=" * 60)
    print("Chrysalis LLM Configuration Verification")
    print("=" * 60)
    
    # Load config
    config = Config.load()
    
    # Display configuration
    print("\n[Configuration]")
    print(f"  Provider:    {config.llm_provider}")
    print(f"  Base URL:    {config.llm_base_url or '(auto-detect)'}")
    print(f"  Model:       {config.llm_model_name}")
    print(f"  API Key:     {'***' if config.llm_api_key else 'Not set'}")
    print(f"  Temperature: {config.llm_temperature}")
    print(f"  Max Tokens:  {config.llm_max_tokens}")
    
    # Create backend
    print("\n[Creating LLM backend...]")
    try:
        backend = LLMBackend(
            model_name=config.llm_model_name,
            provider=config.llm_provider,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        print("OK: Backend created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create backend: {e}")
        return 1
    
    # Test connection
    print("\n[Testing connection...]")
    
    import numpy as np
    
    try:
        # Use public generate_response method instead of private _call_llm_api
        personality_vector = np.array([0.5] * 10)
        response = backend.generate_response(
            user_input="Say 'Hello' in one word.",
            personality_vector=personality_vector,
            conversation_history=[],
        )
        print(f"OK: Connection successful!")
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")
        print("\nTroubleshooting tips:")
        
        if config.llm_provider == "vllm":
            print("  - Make sure vLLM server is running:")
            print("    python -m vllm.entrypoints.openai.api_server --model <model> --port 8000")
        elif config.llm_provider == "ollama":
            print("  - Make sure Ollama is running:")
            print("    ollama serve")
            print("  - Make sure model is pulled:")
            print(f"    ollama pull {config.llm_model_name}")
        elif config.llm_provider in ["openai", "kimi", "claude", "gemini", "minimax"]:
            print("  - Check your API key is valid")
            print("  - Verify base URL")
            print("  - Check network connectivity")
        elif config.llm_provider in ["openrouter", "aihubmix"]:
            print("  - Check your API key is valid")
            print("  - Verify the proxy service is accessible")
        
        backend.close()
        return 1
    
    # Display backend info
    print("\n[Backend Information]")
    info = backend.get_backend_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    backend.close()
    print("\nOK: All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Test LLM backend with different providers"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from decision.backends.llm_backend import LLMBackend, LLMProvider


def test_vllm():
    """Test vLLM provider"""
    print("\n=== Testing vLLM Provider ===")
    backend = LLMBackend(
        llm_endpoint="http://localhost:8000/v1/completions",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        provider="vllm",
        temperature=0.7,
        max_tokens=50,
    )
    
    try:
        response = backend._call_vllm("Say hello in one sentence.")
        print(f"Response: {response}")
        print("✅ vLLM test passed")
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
    finally:
        backend.close()


def test_ollama():
    """Test Ollama provider"""
    print("\n=== Testing Ollama Provider ===")
    backend = LLMBackend(
        llm_endpoint="http://localhost:11434/api/generate",
        model_name="llama2",
        provider="ollama",
        temperature=0.7,
        max_tokens=50,
    )
    
    try:
        response = backend._call_ollama("Say hello in one sentence.")
        print(f"Response: {response}")
        print("✅ Ollama test passed")
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
    finally:
        backend.close()


def test_openai():
    """Test OpenAI provider"""
    print("\n=== Testing OpenAI Provider ===")
    backend = LLMBackend(
        llm_endpoint="https://api.openai.com/v1/chat/completions",
        model_name="gpt-3.5-turbo",
        provider="openai",
        api_key="sk-test",  # Replace with real key
        temperature=0.7,
        max_tokens=50,
    )
    
    try:
        response = backend._call_openai("Say hello in one sentence.")
        print(f"Response: {response}")
        print("✅ OpenAI test passed")
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
    finally:
        backend.close()


def test_generate_response():
    """Test full generate_response workflow"""
    print("\n=== Testing Full Workflow ===")
    backend = LLMBackend(
        llm_endpoint="http://localhost:8000/v1/completions",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        provider="vllm",
    )
    
    personality_vector = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.4, -0.3, 0.2, 0.1, -0.1])
    conversation_history = []
    
    try:
        response = backend.generate_response(
            user_input="Hello, how are you?",
            personality_vector=personality_vector,
            conversation_history=conversation_history,
        )
        print(f"Response: {response}")
        print(f"Backend info: {backend.get_backend_info()}")
        print("✅ Full workflow test passed")
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
    finally:
        backend.close()


if __name__ == "__main__":
    print("Testing LLM Backend with multiple providers")
    print("Note: Make sure corresponding services are running")
    
    # Test individual providers (uncomment as needed)
    test_vllm()
    # test_ollama()
    # test_openai()
    
    # Test full workflow
    test_generate_response()

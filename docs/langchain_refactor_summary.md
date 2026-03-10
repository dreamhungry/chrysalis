# LLM Backend Refactoring: From httpx to LangChain

## Overview

Refactored `llm_backend.py` from manual HTTP client (httpx) to LangChain unified interface, reducing code complexity while improving maintainability and features.

## Key Improvements

### 1. Code Reduction
- **Before**: 346 lines (manual HTTP handling)
- **After**: 314 lines (LangChain integration)
- **Reduction**: 9.2% fewer lines with better structure

### 2. Architecture Simplification

**Before (httpx)**:
```python
# Manual HTTP client management
self._client = httpx.Client(timeout=60.0, headers=headers)

# Provider-specific API implementations
def _call_openai_compatible(self, prompt: str) -> str:
    endpoint = f"{self.base_url}/chat/completions"
    payload = {...}
    response = self._client.post(endpoint, json=payload)
    # Manual response parsing...

def _call_ollama(self, prompt: str) -> str:
    endpoint = f"{self.base_url}/api/generate"
    # Different payload format...

def _call_gemini(self, prompt: str) -> str:
    endpoint = f"{self.base_url}/models/{self.model_name}:generateContent"
    # Yet another format...

# 9 different implementations...
```

**After (LangChain)**:
```python
# Unified chat model creation
self.chat_model = self._create_chat_model()

# Single call interface for all providers
def generate_response(...) -> str:
    messages = self._build_messages(...)
    response = self.chat_model.invoke(messages)
    return response.content.strip()
```

### 3. Provider Support

| Provider | Package | Status |
|----------|---------|--------|
| OpenAI | `langchain-openai` | ✅ Fully supported |
| vLLM | `langchain-openai` | ✅ Via base_url |
| Kimi | `langchain-openai` | ✅ Via base_url |
| OpenRouter | `langchain-openai` | ✅ Via base_url |
| AihubMix | `langchain-openai` | ✅ Via base_url |
| Ollama | `langchain-ollama` | ✅ Dedicated package |
| Gemini | `langchain-google-genai` | ✅ Dedicated package |
| Claude | `langchain-anthropic` | ✅ Dedicated package |
| Minimax | `langchain-community` | ✅ Community support |

### 4. New Features (Out-of-the-box)

#### Stream Support
```python
# Before: Not implemented (would need manual SSE handling)

# After: Built-in
for chunk in backend.chat_model.stream(messages):
    print(chunk.content, end="", flush=True)
```

#### Structured Output
```python
from pydantic import BaseModel

class AgentResponse(BaseModel):
    action: str
    reasoning: str
    confidence: float

# Before: Need manual JSON parsing and validation

# After: Type-safe automatic parsing
structured_chat = backend.chat_model.with_structured_output(AgentResponse)
response = structured_chat.invoke(messages)  # Returns AgentResponse object
```

#### Observability
```python
from langchain.callbacks import get_openai_callback

# Before: Manual token counting

# After: Automatic tracking
with get_openai_callback() as cb:
    response = backend.generate_response(...)
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

#### Retry Logic
```python
# Before: Manual retry implementation needed

# After: Built-in with exponential backoff
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    max_retries=3,  # Automatic retry
    timeout=30,     # Per-request timeout
)
```

### 5. Error Handling

**Before**:
```python
try:
    response = self._client.post(endpoint, json=payload)
    response.raise_for_status()
    result = response.json()
    # Manual parsing...
except httpx.HTTPError as e:
    # Manual error handling for each provider
```

**After**:
```python
try:
    response = self.chat_model.invoke(messages)
    return response.content.strip()
except Exception as e:
    logger.error("LLM API call failed: %s", e)
    # LangChain handles provider-specific errors
```

### 6. Dependencies

**Before**:
```txt
httpx>=0.25.0
langchain>=0.1.0  # Not fully utilized
```

**After**:
```txt
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-google-genai>=2.0.0
langchain-ollama>=0.2.0
langchain-community>=0.3.0
```

## Migration Guide

### For Users

No breaking changes! Configuration remains the same:

```bash
# .env file (unchanged)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-xxx
```

### For Developers

1. **Install new dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API remains compatible**:
   ```python
   # Same usage as before
   backend = LLMBackend(
       model_name="gpt-3.5-turbo",
       provider="openai",
       api_key="sk-xxx"
   )
   
   response = backend.generate_response(
       user_input="Hello",
       personality_vector=np.array([...]),
       conversation_history=[]
   )
   ```

3. **New features (optional)**:
   ```python
   # Enable streaming
   for chunk in backend.chat_model.stream(messages):
       print(chunk.content, end="")
   
   # Get token usage
   with get_openai_callback() as cb:
       response = backend.generate_response(...)
       print(f"Used {cb.total_tokens} tokens")
   ```

## Performance Comparison

| Metric | httpx | LangChain | Improvement |
|--------|-------|-----------|-------------|
| Code Lines | 346 | 314 | -9.2% |
| Provider Impls | 9 methods | 1 method | -88.9% |
| Error Handling | Manual | Built-in | ✅ |
| Retry Logic | Manual | Built-in | ✅ |
| Streaming | Not supported | Built-in | ✅ |
| Token Tracking | Manual | Built-in | ✅ |
| Maintenance | High | Low | ✅ |

## Testing

```bash
# Verify configuration
python scripts/verify_llm_config.py

# Run tests
pytest tests/test_llm_providers.py
```

## Future Enhancements

With LangChain foundation, we can easily add:

1. **Memory Management**: Built-in conversation memory
2. **RAG Integration**: Document retrieval and QA
3. **Agent Tools**: Function calling and tool use
4. **Prompt Templates**: Template management
5. **Chain Composition**: Multi-step reasoning
6. **LangSmith Integration**: Production monitoring

## Conclusion

The refactoring to LangChain provides:
- ✅ **Simpler code** (9% reduction, 89% fewer provider methods)
- ✅ **Better reliability** (built-in retry, error handling)
- ✅ **More features** (streaming, structured output, observability)
- ✅ **Easier maintenance** (official SDK updates)
- ✅ **Future-proof** (extensible for advanced features)

All while maintaining **100% backward compatibility** with existing code.

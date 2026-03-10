# LLM Backend Multi-Provider Support

## Summary

Enhanced `llm_backend.py` to support multiple LLM providers with configuration-based switching.

## Changes

### 1. Core Implementation (`src/decision/backends/llm_backend.py`)

Added support for three providers:

| Provider | Endpoint Pattern | Request Format |
|----------|------------------|----------------|
| **vLLM** | `/v1/completions` | OpenAI-compatible |
| **Ollama** | `/api/generate` | Ollama-specific |
| **OpenAI** | `/v1/chat/completions` | OpenAI chat format |

**Key features:**
- Provider-specific API call methods
- Automatic format adaptation
- API key support (for OpenAI)
- Unified error handling

### 2. Configuration (`src/config.py`, `.env`)

Added new configuration parameters:

```bash
LLM_PROVIDER=vllm          # vllm|ollama|openai
LLM_ENDPOINT=<url>
LLM_MODEL_NAME=<model>
LLM_API_KEY=<key>          # Optional, required for OpenAI
LLM_TEMPERATURE=0.8
LLM_MAX_TOKENS=256
```

### 3. Bootstrap Integration (`src/bootstrap.py`)

Updated `create_decision_backend()` to pass provider parameters.

### 4. Testing & Verification

Created tools:
- `scripts/verify_llm_config.py` - Verify configuration and test connection
- `tests/test_llm_providers.py` - Unit tests for each provider
- `configs/llm_providers.example` - Configuration examples

### 5. Documentation (`README.md`)

Added comprehensive LLM provider configuration section with:
- Configuration table
- Setup instructions for each provider
- Quick start examples

## Usage Examples

### vLLM (Local)

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000

# Configure
LLM_PROVIDER=vllm
LLM_ENDPOINT=http://localhost:8000/v1/completions
LLM_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

### Ollama (Easiest)

```bash
# Start Ollama
ollama serve
ollama pull llama2

# Configure
LLM_PROVIDER=ollama
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL_NAME=llama2
```

### OpenAI (Cloud)

```bash
# Configure
LLM_PROVIDER=openai
LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-xxx
```

## Testing

```bash
# Verify configuration
.venv\Scripts\python scripts\verify_llm_config.py

# Run provider tests (requires services running)
.venv\Scripts\python tests\test_llm_providers.py
```

## Implementation Details

### Provider Detection

```python
class LLMProvider(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"
```

### API Call Routing

```python
def _call_llm_api(self, prompt: str) -> str:
    if self.provider == LLMProvider.VLLM:
        return self._call_vllm(prompt)
    elif self.provider == LLMProvider.OLLAMA:
        return self._call_ollama(prompt)
    elif self.provider == LLMProvider.OPENAI:
        return self._call_openai(prompt)
```

### Response Format Handling

Each provider method handles its specific response format:
- vLLM: `result["choices"][0]["text"]`
- Ollama: `result["response"]`
- OpenAI: `result["choices"][0]["message"]["content"]`

## Design Decisions

1. **Enum-based provider selection**: Type-safe and IDE-friendly
2. **Separate methods per provider**: Clear separation of concerns
3. **Configuration-driven**: Easy to switch providers without code changes
4. **Backward compatible**: Existing vLLM configurations still work
5. **Minimal dependencies**: Uses only `httpx` for HTTP calls

## Future Enhancements

- [ ] Support for streaming responses
- [ ] Custom provider plugin system
- [ ] Provider-specific retry logic
- [ ] Performance metrics per provider
- [ ] Fallback provider chain

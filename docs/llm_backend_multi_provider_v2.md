# LLM Backend Multi-Provider Support - Architecture Update

## Overview

Extended LLM backend to support **9 major providers** with unified interface and auto-configuration.

## Architecture Changes

### Before (v1)

```
┌─────────────────────────────────────┐
│  LLMBackend                         │
├─────────────────────────────────────┤
│  - llm_endpoint (full URL)          │
│  - provider: vllm/ollama/openai     │
│  - Hardcoded endpoint formats       │
└─────────────────────────────────────┘
```

### After (v2)

```
┌─────────────────────────────────────┐
│  LLMBackend                         │
├─────────────────────────────────────┤
│  - base_url (flexible)              │
│  - provider: 9 types                │
│  - Auto-detect default URLs         │
│  - Provider-specific adapters       │
└─────────────────────────────────────┘
```

## Supported Providers

### OpenAI-Compatible (5 providers)

Use standard OpenAI API format - simple integration:

| Provider | Base URL | Auth Header |
|----------|----------|-------------|
| OpenAI | `https://api.openai.com/v1` | Bearer token |
| vLLM | `http://localhost:8000/v1` | Optional |
| Kimi | `https://api.moonshot.cn/v1` | Bearer token |
| OpenRouter | `https://openrouter.ai/api/v1` | Bearer token |
| AihubMix | `https://aihubmix.com/v1` | Bearer token |

**Endpoint**: `{base_url}/chat/completions`

**Request Format**:
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.8,
  "max_tokens": 256
}
```

### Custom Format (4 providers)

Each has unique API format - requires adapter:

#### 1. Ollama
- **Endpoint**: `{base_url}/api/generate`
- **Format**: `{"model": "...", "prompt": "...", "stream": false}`
- **Response**: `{"response": "..."}`

#### 2. Google Gemini
- **Endpoint**: `{base_url}/models/{model}:generateContent?key={api_key}`
- **Format**: `{"contents": [{"parts": [{"text": "..."}]}]}`
- **Response**: `{"candidates": [{"content": {"parts": [{"text": "..."}]}}]}`

#### 3. Anthropic Claude
- **Endpoint**: `{base_url}/messages`
- **Auth**: `x-api-key` header + `anthropic-version`
- **Format**: `{"model": "...", "messages": [...]}`
- **Response**: `{"content": [{"text": "..."}]}`

#### 4. Minimax
- **Endpoint**: `{base_url}/text/chatcompletion_pro?GroupId={group_id}`
- **Format**: OpenAI-like but with `tokens_to_generate`
- **Response**: `{"choices": [...]}` or `{"reply": "..."}`

## Key Design Improvements

### 1. BASE_URL Instead of ENDPOINT

**Before**:
```bash
LLM_ENDPOINT=https://api.openai.com/v1/chat/completions  # Full path
```

**After**:
```bash
LLM_BASE_URL=https://api.openai.com/v1  # Base only
```

**Benefits**:
- Backend determines correct endpoint path
- Easier to switch between providers
- Supports multiple endpoints per provider

### 2. Auto-Detection

```python
DEFAULT_ENDPOINTS = {
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.VLLM: "http://localhost:8000/v1",
    # ... 7 more
}

# Usage
if not base_url:
    base_url = DEFAULT_ENDPOINTS.get(provider)
```

**Benefits**:
- Zero-config for standard setups
- Only need to set provider + API key
- Override available when needed

### 3. Provider Categories

```python
class LLMProvider(str, Enum):
    # OpenAI-compatible
    OPENAI = "openai"
    VLLM = "vllm"
    KIMI = "kimi"
    OPENROUTER = "openrouter"
    AIHUBMIX = "aihubmix"
    
    # Custom format
    OLLAMA = "ollama"
    GEMINI = "gemini"
    CLAUDE = "claude"
    MINIMAX = "minimax"
```

**Benefits**:
- Easy to add new OpenAI-compatible providers
- Clear separation of integration complexity
- Type-safe provider selection

### 4. Flexible Authentication

Different providers, different auth methods:

```python
if provider == CLAUDE:
    headers["x-api-key"] = api_key
    headers["anthropic-version"] = "2023-06-01"
elif provider == GEMINI:
    # Use query param instead
    endpoint += f"?key={api_key}"
else:
    # OpenAI-compatible
    headers["Authorization"] = f"Bearer {api_key}"
```

## Configuration Examples

### Minimal (Auto-detect)

```bash
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-xxx
# LLM_BASE_URL is auto-detected
```

### Custom Endpoint

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=https://your-proxy.com/v1
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-xxx
```

### Local Deployment

```bash
LLM_PROVIDER=vllm
# LLM_BASE_URL auto-detects to http://localhost:8000/v1
LLM_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

## Testing

Verify configuration:
```bash
.venv\Scripts\python scripts\verify_llm_config.py
```

Expected output:
```
[Configuration]
  Provider:    openai
  Base URL:    (auto-detect) or https://...
  Model:       gpt-3.5-turbo
  API Key:     ***
  
[Creating LLM backend...]
OK: Backend created successfully
Using default base_url for openai: https://api.openai.com/v1

[Testing connection...]
OK: Connection successful!  (or appropriate error with troubleshooting tips)
```

## Provider Comparison

| Provider | Setup | Cost | Speed | Use Case |
|----------|-------|------|-------|----------|
| **vLLM** | Local | Free | Fast* | Development, privacy |
| **Ollama** | Local | Free | Medium* | Easy local testing |
| **OpenAI** | Cloud | $$$ | Fast | Production, quality |
| **Claude** | Cloud | $$$$ | Fast | Complex reasoning |
| **Gemini** | Cloud | $ | Fast | Cost-effective |
| **Kimi** | Cloud | $$ | Fast | Chinese, long context |
| **OpenRouter** | Cloud | Varies | Fast | Multi-model access |
| **Minimax** | Cloud | $$ | Fast | Chinese market |
| **AihubMix** | Proxy | $ | Fast | API proxy/caching |

\* *Performance depends on hardware*

## Implementation Checklist

- [x] Add `LLMProvider` enum with 9 providers
- [x] Replace `llm_endpoint` with `llm_base_url`
- [x] Implement auto-detection of default URLs
- [x] Add OpenAI-compatible adapter
- [x] Add Ollama adapter
- [x] Add Gemini adapter
- [x] Add Claude adapter (with custom headers)
- [x] Add Minimax adapter
- [x] Update Config class
- [x] Update bootstrap.py
- [x] Update .env with examples
- [x] Update README documentation
- [x] Create verify_llm_config.py script
- [x] Update llm_providers.example
- [x] Test with multiple providers

## Future Enhancements

- [ ] Streaming response support
- [ ] Retry logic with exponential backoff
- [ ] Provider health check / fallback chain
- [ ] Rate limiting per provider
- [ ] Cost tracking and budgets
- [ ] Async API calls
- [ ] Batch processing optimization
- [ ] Provider-specific system prompt optimization

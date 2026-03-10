# LLM Backend 升级总结

## 问题回答

### Q1: 为什么之前没有 LLM_BASE_URL？

**之前的设计**：使用 `LLM_ENDPOINT` 指定完整的 API 路径
```bash
LLM_ENDPOINT=https://api.openai.com/v1/chat/completions  # 完整路径
```

**问题**：
- 不灵活，每个 provider 的 endpoint 格式不同
- 需要记住完整路径
- 难以支持一个 provider 有多个 endpoint 的情况

**现在的设计**：使用 `LLM_BASE_URL` + 自动路径拼接
```bash
LLM_BASE_URL=https://api.openai.com/v1  # 只需要基础 URL
```

**优势**：
- ✅ 后端自动拼接正确的 endpoint 路径
- ✅ 支持 auto-detect（留空即可）
- ✅ 更符合行业标准实践
- ✅ 易于切换 provider

---

### Q2: 能否支持多个厂商（Gemini、Claude、Kimi、Minimax 等）？

**答案：现在完全可以！**

## 新增支持的厂商

### OpenAI 兼容厂商（5个）
使用标准 OpenAI API 格式，接入简单：

| 厂商 | 配置 | 特点 |
|------|------|------|
| **OpenAI** | `LLM_PROVIDER=openai` | 官方 API |
| **vLLM** | `LLM_PROVIDER=vllm` | 本地部署 |
| **Kimi** | `LLM_PROVIDER=kimi` | 月之暗面，长文本 |
| **OpenRouter** | `LLM_PROVIDER=openrouter` | 多模型聚合 |
| **AihubMix** | `LLM_PROVIDER=aihubmix` | API 中转 |

### 自定义格式厂商（4个）
每个都有独特 API，需要专门适配器：

| 厂商 | 配置 | 特点 |
|------|------|------|
| **Ollama** | `LLM_PROVIDER=ollama` | 本地部署，最简单 |
| **Gemini** | `LLM_PROVIDER=gemini` | Google，性价比高 |
| **Claude** | `LLM_PROVIDER=claude` | Anthropic，推理强 |
| **Minimax** | `LLM_PROVIDER=minimax` | 海螺 AI，中文友好 |

---

## 核心改进

### 1. 配置更灵活

```bash
# 最小配置（自动检测 base_url）
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-xxx

# 自定义 base_url
LLM_PROVIDER=openai
LLM_BASE_URL=https://your-proxy.com/v1
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-xxx
```

### 2. 零配置支持

每个 provider 都有默认 base_url：
```python
DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "vllm": "http://localhost:8000/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "claude": "https://api.anthropic.com/v1",
    # ... 等等
}
```

### 3. 统一调用接口

所有 provider 使用相同的调用方式：
```python
backend = LLMBackend(
    model_name="gpt-3.5-turbo",
    provider="openai",  # 或 kimi/gemini/claude/等
    api_key="sk-xxx"
)
response = backend.generate_response(...)
```

---

## 快速开始

### OpenAI
```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=gpt-3.5-turbo
```

### Kimi（月之暗面）
```bash
LLM_PROVIDER=kimi
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=moonshot-v1-8k
```

### Claude
```bash
LLM_PROVIDER=claude
LLM_API_KEY=sk-ant-xxx
LLM_MODEL_NAME=claude-3-sonnet-20240229
```

### Gemini
```bash
LLM_PROVIDER=gemini
LLM_API_KEY=your-google-api-key
LLM_MODEL_NAME=gemini-pro
```

### vLLM（本地）
```bash
# 启动服务
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct

# 配置
LLM_PROVIDER=vllm
LLM_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

### Ollama（本地，最简单）
```bash
# 启动服务
ollama serve
ollama pull llama2

# 配置
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama2
```

---

## 验证配置

运行验证脚本：
```bash
.venv\Scripts\python scripts\verify_llm_config.py
```

输出示例：
```
[Configuration]
  Provider:    openai
  Base URL:    (auto-detect)
  Model:       gpt-3.5-turbo
  API Key:     ***

[Creating LLM backend...]
OK: Backend created successfully
Using default base_url for openai: https://api.openai.com/v1

[Testing connection...]
OK: Connection successful!
Response: Hello! How can I help you today?

[Backend Information]
  type: llm
  provider: openai
  model: gpt-3.5-turbo
  base_url: https://api.openai.com/v1
  temperature: 0.8
  max_tokens: 256
```

---

## 架构对比

### 之前
```
LLMBackend
├── provider: 3种 (vllm/ollama/openai)
├── llm_endpoint: 完整路径
└── _call_vllm() / _call_ollama() / _call_openai()
```

### 现在
```
LLMBackend
├── provider: 9种
├── llm_base_url: 基础URL + 自动检测
├── 分类处理:
│   ├── OpenAI兼容: 5种 → _call_openai_compatible()
│   └── 自定义格式: 4种 → 专门适配器
└── DEFAULT_ENDPOINTS: 每个provider的默认URL
```

---

## 技术细节

### API 格式差异处理

**OpenAI 兼容**（统一处理）：
```python
endpoint = f"{base_url}/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": temp,
    "max_tokens": tokens
}
```

**Gemini**（特殊格式）：
```python
endpoint = f"{base_url}/models/{model}:generateContent?key={api_key}"
payload = {
    "contents": [{"parts": [{"text": prompt}]}],
    "generationConfig": {...}
}
```

**Claude**（特殊 header）：
```python
headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}
```

---

## 文件清单

### 修改的文件
- ✅ `src/decision/backends/llm_backend.py` - 核心实现
- ✅ `src/config.py` - 配置管理
- ✅ `src/bootstrap.py` - 初始化逻辑
- ✅ `.env` - 配置文件
- ✅ `README.md` - 文档更新

### 新增的文件
- ✅ `configs/llm_providers.example` - 配置示例
- ✅ `scripts/verify_llm_config.py` - 验证脚本
- ✅ `docs/llm_backend_multi_provider_v2.md` - 架构文档

---

## 总结

### 解决的问题
1. ✅ 统一使用 `LLM_BASE_URL`（更标准）
2. ✅ 支持 **9 个主流 LLM 厂商**
3. ✅ 自动检测默认 endpoint
4. ✅ 统一的调用接口
5. ✅ 灵活的配置方式

### 关键特性
- 🚀 **零配置启动**：只需设置 provider 和 API key
- 🔌 **即插即用**：切换厂商只需改配置
- 🛡️ **类型安全**：使用 Enum 定义 provider
- 📚 **完善文档**：包含所有厂商的配置示例
- 🧪 **验证工具**：一键测试配置是否正确

### 下一步建议
- 根据实际需求选择合适的 provider
- 本地开发推荐：vLLM 或 Ollama
- 生产环境推荐：OpenAI、Claude 或 Kimi
- 成本敏感推荐：Gemini 或 AihubMix

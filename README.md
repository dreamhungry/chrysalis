# Chrysalis

**Chrysalis** is a research project exploring how AI agents can **develop evolving personalities through long-term interaction with humans**.

Inspired by the biological chrysalis stage—where transformation occurs—this project investigates how an interactive agent can gradually form **stable behavioral traits and identity through experience**.

The goal is to move beyond static chatbot personalities and build agents that **grow, adapt, and develop behavioral consistency over time**.

---

## Overview

Most AI agents today generate responses based only on short-term context.
As a result, they lack:

* persistent behavioral traits
* long-term adaptation
* consistent identity across interactions

Chrysalis explores a different approach: treating **personality as a learnable and evolving structure** rather than a predefined setting.

Through interaction history, feedback, and memory, the agent gradually develops **stable yet adaptive behavioral tendencies**.

---

## Core Idea

Chrysalis models personality as an evolving internal state shaped by interaction.

```
Human Interaction
        ↓
Interaction Memory
        ↓
Personality Update
        ↓
Personality Representation
        ↓
Agent Behavior
```

Over time, this process allows the agent to form a **consistent behavioral identity**.

---

## Features

### Trait Vector Representation
- 10-dimensional latent personality vector
- Semantic mapping (e.g., dimension 0 = friendliness, dimension 1 = humor)
- JSON-based persistence for programmatic access

### Interaction Memory
- Markdown-based storage for human and LLM readability
- Temporal sequence recording with feedback support
- Abstract backend interface (extensible to vector databases)

### Personality Learning
- **Feedback-based Update**: Incremental adjustment via user ratings
- **Pattern Extraction**: Mining recurring interaction patterns
- **RL Updater**: Reward-driven trait evolution
- **LLM Reflection**: Self-assessment using the agent's own reasoning

### Decision Engine
- **LLM Backend**: Direct LLM reasoning with personality context
- **Custom Model Backend**: Fine-tuned model integration
- **Utility AI Backend**: Score-based action selection
- Personality-driven prompt construction

### Multiple Interfaces
- **CLI**: Interactive command-line chat
- **REST API**: FastAPI-based `/chat` and `/feedback` endpoints
- **Web UI**: Gradio-based conversational interface

---

## Repository Structure

```
chrysalis/
├── src/
│   ├── personality/           # Trait vector & personality store
│   │   ├── trait_vector.py
│   │   └── personality_store.py
│   ├── memory/                 # Interaction memory system
│   │   ├── base.py             # Abstract backend interface
│   │   ├── interaction_store.py
│   │   ├── memory_manager.py
│   │   └── backends/
│   │       ├── markdown_backend.py   # Current implementation
│   │       └── vector_backend.py     # Vector database backend
│   ├── decision/              # Decision engine
│   │   ├── base.py            # Abstract decision interface
│   │   ├── prompt_builder.py
│   │   └── backends/
│   │       ├── llm_backend.py
│   │       ├── custom_model_backend.py
│   │       └── utility_ai_backend.py
│   ├── learning/              # Personality learning
│   │   ├── feedback_updater.py
│   │   ├── pattern_extractor.py
│   │   ├── rl_updater.py
│   │   └── reflector.py
│   ├── agent/                 # Agent core
│   │   └── agent_core.py
│   ├── interfaces/            # User interfaces
│   │   ├── cli.py
│   │   ├── api_service.py
│   │   └── web_ui.py
│   ├── config.py              # Configuration management
│   ├── bootstrap.py           # System initialization
│   └── main.py                # Entry point
├── configs/
│   ├── agent_config.yaml      # Agent configuration
│   └── .env.example           # Environment variables template
├── tests/
│   └── test_integration.py    # Integration tests
├── experiments/               # Research experiments
└── README.md
```

---

## Architecture

Chrysalis models personality as an evolving internal state influenced by interaction history and feedback.

```
                    ┌───────────────┐
                    │   Human User  │
                    └───────┬───────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │   Interface Layer  │
                 │ (CLI/API/Web UI)   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │      Agent Core    │
                 └─────────┬──────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
 ┌──────────────────┐            ┌──────────────────┐
 │  Personality     │            │  Interaction     │
 │  (Trait Vector)  │            │  Memory          │
 └─────────┬────────┘            └─────────┬────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
                ┌────────────────────┐
                │   Learning Module   │
                │ (Feedback/Pattern/ │
                │  RL/Reflection)    │
                └────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy and configure environment variables:

```bash
cp configs/.env.example .env
```

Edit `.env` with your settings:
- `LLM_API_BASE`: LLM service endpoint (default: http://localhost:8000)
- `LLM_MODEL`: Model name
- `MEMORY_BACKEND`: Storage type (markdown/vector)
- `DECISION_BACKEND`: Decision method (llm/custom/utility)

### 3. Run

```bash
# CLI mode
python -m src.main --cli

# API server
python -m src.main --api

# Web UI
python -m src.main --web

# Interactive mode (default)
python -m src.main
```

### 4. Docker

```bash
docker build -t chrysalis .
docker run -p 8000:8000 chrysalis
```

---

## Configuration

### Agent Configuration (`configs/agent_config.yaml`)

```yaml
# LLM Service
llm:
  api_base: "http://localhost:8000"
  model: "qwen2.5"
  temperature: 0.7
  max_tokens: 2048

# Personality Vector
personality:
  dimensions: 10
  init_values: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  dimension_labels:
    - "friendliness"
    - "humor"
    - "formality"
    - "curiosity"
    - "patience"
    - "empathy"
    - "confidence"
    - "creativity"
    - "caution"
    - "optimism"
  storage_path: "data/personality/trait_vector.json"

# Memory
memory:
  backend: "markdown"
  markdown_path: "data/interactions/history.md"
  max_memory_items: 1000

# Decision Engine
decision:
  backend: "llm"
  prompt_template: "default"

# Learning
learning:
  feedback_weight: 0.1
  pattern_threshold: 0.7
  rl_learning_rate: 0.01
  reflection_interval: 10
```

### Environment Variables

The project supports multiple LLM providers. Configure them in `.env` file:

#### LLM Provider Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider type: `vllm`, `ollama`, `openai` | vllm |
| `LLM_ENDPOINT` | LLM API endpoint | http://localhost:8000/v1/completions |
| `LLM_MODEL_NAME` | Model name | Qwen/Qwen2.5-1.5B-Instruct |
| `LLM_API_KEY` | API key (required for OpenAI) | - |
| `LLM_TEMPERATURE` | Generation temperature | 0.8 |
| `LLM_MAX_TOKENS` | Maximum generation tokens | 256 |

#### Supported Providers

**1. vLLM (Recommended for local deployment)**
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --port 8000

# Configure in .env
LLM_PROVIDER=vllm
LLM_ENDPOINT=http://localhost:8000/v1/completions
LLM_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

**2. Ollama (Easiest local setup)**
```bash
# Install and start Ollama
ollama serve
ollama pull llama2

# Configure in .env
LLM_PROVIDER=ollama
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL_NAME=llama2
```

**3. OpenAI API (Cloud service)**
```bash
# Configure in .env
LLM_PROVIDER=openai
LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_API_KEY=sk-your-api-key-here
```

See `configs/llm_providers.example` for more configuration examples.

#### Other Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DECISION_BACKEND` | Decision method | llm |
| `MEMORY_BACKEND` | Storage backend | markdown |
| `MEMORY_MARKDOWN_PATH` | Markdown storage path | data/interactions/history.md |
| `PERSONALITY_PATH` | Personality vector file | data/personalities/current.json |
| `LOG_LEVEL` | Logging level | INFO |

---

## Data Storage

### Personality Vector
- **Format**: JSON
- **Location**: `data/personality/trait_vector.json`
- **Reason**: Programmatic access efficiency

### Interaction History
- **Format**: Markdown (.md)
- **Location**: `data/interactions/history.md`
- **Reason**: Human-readable and LLM-friendly for context injection

---

## Research Direction

This project investigates three key questions:

* Can personality **emerge from interaction history**?
* Can agents maintain **stable yet adaptive behavioral traits**?
* Does evolving personality improve **long-term engagement** in human–AI interaction?

---

## Testing

```bash
# Run integration tests
python -m tests.test_integration
```

---

## License

MIT

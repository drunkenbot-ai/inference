# DrunkenBot Inference Engine (`drunkenbot-inference`)

Standalone, modular inference runtime and UI components for DrunkenBot language models.

## Overview

This package is structured to be decoupled from the core training engine and from desktop GUI frameworks:
- **`inference.core`**: Completely headless inference engine. Executes native PyTorch MicroGPT models and GGUF llama.cpp models, autonomous ReAct agent loops (`python_interpreter`, `web_search`), stop sequence pruning, and benchmark evaluation. Has **zero Qt / PySide6 dependencies**—ideal for CLI utilities, microservices, and web backends.
- **`inference.ui`**: Desktop Qt (PySide6) widgets and tab builders. Includes collapsible chain-of-thought reasoning accordions (`<thought>`), themed tool execution chips, syntax-highlighted Markdown renderer, and chat/benchmark tab managers.

---

## Directory Layout

```
inference/
├── README.md                      # Package overview & documentation
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Headless core dependencies
├── requirements-ui.txt            # Desktop UI dependencies
├── __init__.py                    # Top-level exports
│
├── core/                          # === Headless Inference Core ===
│   ├── __init__.py                # Public core exports
│   ├── types.py                   # Pure Python dataclasses & protocols
│   ├── agent_executor.py          # Autonomous tool router (python, web search)
│   ├── microgpt_chat.py           # Native PyTorch KV-cached chat session & ReAct loop
│   ├── llama_chat.py              # llama.cpp GGUF streamed chat wrapper
│   ├── generation.py              # Checkpoint loader & single prompt generation
│   └── evaluation.py              # Model benchmarking & throughput metrics
│
└── ui/                            # === Desktop Qt UI & Widgets ===
    ├── __init__.py                # Public UI exports
    ├── chat_widgets.py            # Chat bubble widget with collapsible thought toggle
    ├── markdown_renderer.py       # Markdown-to-HTML parser with tool call boxes
    ├── chat_screen.py             # ChatScreenMixin (streaming thread manager)
    ├── chat_tab.py                # build_chat_tab() (Chat Studio Qt tab layout)
    ├── benchmark_screen.py        # BenchmarkScreenMixin
    └── benchmark_tab.py           # build_benchmark_tab()
```

---

## Installation & Usage

### 1. Headless Usage (CLI, Scripts, Web Servers)

Install headless dependencies:
```bash
pip install torch requests beautifulsoup4
```

Using in Python:
```python
from pathlib import Path
from inference.core import load_microgpt_chat_session, stream_microgpt_chat_reply

# Load model checkpoint
session = load_microgpt_chat_session(Path("checkpoints/final_model.pt"), device="cuda")

# Stream reply with autonomous tool calling
for chunk in stream_microgpt_chat_reply(
    session,
    prompt="Calculate 25 * 43 using python",
    system_prompt="You are a helpful assistant with access to tools.",
    temperature=0.7,
    max_tokens=256,
    enable_tools=True,
):
    print(chunk["content"], end="", flush=True)
```

### 2. Desktop Qt UI Usage

Install UI dependencies:
```bash
pip install PySide6
```

Using in PySide6 applications:
```python
from inference.ui import ChatMessageWidget, markdown_to_html
```

### 3. Future Web App Integration (FastAPI / WebSockets / SSE)

Because `inference.core` is decoupled from GUI libraries, building a web server is simple:
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from inference.core import load_microgpt_chat_session, stream_microgpt_chat_reply

app = FastAPI()
session = load_microgpt_chat_session("final_model.pt")

@app.post("/v1/chat/completions")
async def chat_endpoint(prompt: str):
    def event_generator():
        for chunk in stream_microgpt_chat_reply(session, prompt):
            yield f"data: {chunk['content']}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## Submodule Setup

To publish this directory as its own GitHub repository and consume it as a Git submodule:
```bash
cd inference
git init
git remote add origin https://github.com/drunkenbot-ai/inference.git
git add .
git commit -m "feat: initial inference engine and ui components"
git push -u origin main
```
Then in the parent `LLM-IDE` repository:
```bash
git submodule add https://github.com/drunkenbot-ai/inference.git inference
```
No import paths in your application need to be updated.

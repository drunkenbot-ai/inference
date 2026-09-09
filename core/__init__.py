"""Headless inference core package.

Provides pure Python / PyTorch / llama.cpp inference and autonomous agent loops
with ZERO graphical UI dependencies.
"""

from __future__ import annotations

from .types import (
    GenerationOptions,
    InferenceMessage,
    StreamChunk,
)
from .agent_executor import (
    execute_agent_tool,
    execute_python_code,
    execute_web_search,
    parse_tool_calls,
)
from .microgpt_chat import (
    STOP_SEQUENCES,
    MicroGPTChatSession,
    load_microgpt_chat_session,
    stream_microgpt_chat_reply,
)
from .llama_chat import (
    LlamaChatSession,
    generate_chat_reply,
    load_llama_chat_session,
    stream_chat_reply,
)
from .generation import (
    generate_text,
    load_model_from_checkpoint,
)
from .evaluation import (
    DEFAULT_BENCHMARK_PROMPTS,
    BenchmarkResult,
    evaluate_checkpoint,
    normalize_prompts,
)

__all__ = [
    # Types
    "GenerationOptions",
    "InferenceMessage",
    "StreamChunk",
    # Agent Tools
    "execute_agent_tool",
    "execute_python_code",
    "execute_web_search",
    "parse_tool_calls",
    # MicroGPT Native
    "MicroGPTChatSession",
    "load_microgpt_chat_session",
    "stream_microgpt_chat_reply",
    "STOP_SEQUENCES",
    # LLaMA GGUF
    "LlamaChatSession",
    "load_llama_chat_session",
    "generate_chat_reply",
    "stream_chat_reply",
    # Generation
    "load_model_from_checkpoint",
    "generate_text",
    # Evaluation & Benchmark
    "DEFAULT_BENCHMARK_PROMPTS",
    "BenchmarkResult",
    "evaluate_checkpoint",
    "normalize_prompts",
]

"""Data structures and type definitions for DrunkenBot headless inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class InferenceMessage:
    """A single chat message in a conversation.

    Attributes:
        role: Role of the message sender ('system', 'user', 'assistant', 'tool').
        content: Text content of the message.
        tool_calls: Optional list of tool call dictionaries.
        thought: Optional extracted chain-of-thought text.
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    thought: Optional[str] = None


@dataclass
class GenerationOptions:
    """Sampling and control parameters for inference generation.

    Attributes:
        temperature: Softmax sampling temperature.
        top_k: Top-k sampling cutoff.
        max_new_tokens: Maximum tokens to generate.
        stop_sequences: List of string sequences that halt generation.
        enable_tools: Whether the model may autonomously invoke agent tools.
        max_tool_hops: Maximum consecutive tool calls allowed in one response turn.
    """

    temperature: float = 0.7
    top_k: Optional[int] = 50
    max_new_tokens: int = 256
    stop_sequences: tuple[str, ...] = ()
    enable_tools: bool = True
    max_tool_hops: int = 3


@dataclass
class StreamChunk:
    """A streaming text or event chunk emitted during generation.

    Attributes:
        content: Delta text chunk.
        token_count: Cumulative token count generated so far.
        elapsed_seconds: Elapsed wall-clock time in seconds.
        tokens_per_second: Generation speed in tokens per second.
        is_tool_call: True if this chunk signals an autonomous tool invocation.
        is_observation: True if this chunk contains a tool observation.
        is_thought: True if this chunk is part of an internal reasoning block.
    """

    content: str = ""
    token_count: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float = 0.0
    is_tool_call: bool = False
    is_observation: bool = False
    is_thought: bool = False

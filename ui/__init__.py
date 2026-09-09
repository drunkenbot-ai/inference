"""Desktop Qt UI components for inference and benchmarking.

Provides rich chat widgets, thought process collapsible toggles,
Markdown-to-HTML rendering, and tab layout builders for PySide6.
"""

from __future__ import annotations

from .chat_widgets import ChatInputEdit, ChatMessageWidget
from .markdown_renderer import format_agent_artifacts_markdown, markdown_to_html
from .chat_screen import ChatScreenMixin
from .chat_tab import build_chat_tab
from .benchmark_screen import BenchmarkScreenMixin
from .benchmark_tab import build_benchmark_tab

__all__ = [
    "ChatInputEdit",
    "ChatMessageWidget",
    "markdown_to_html",
    "format_agent_artifacts_markdown",
    "ChatScreenMixin",
    "build_chat_tab",
    "BenchmarkScreenMixin",
    "build_benchmark_tab",
]

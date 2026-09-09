from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union  # noqa: F401

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox, QTextBrowser

from inference.core.llama_chat import load_llama_chat_session, stream_chat_reply
from inference.core.microgpt_chat import load_microgpt_chat_session, stream_microgpt_chat_reply
from .chat_widgets import ChatMessageWidget
from .markdown_renderer import markdown_to_html


class ChatScreenMixin:
    """Mixin class providing Chat Studio interactions for LLM-IDE and standalone Qt windows."""

    def _render_chat_markdown(self, markdown_text: str) -> None:
        """Render chat Markdown with highlighted fenced code blocks when possible.

        Args:
            markdown_text: Markdown transcript to render.
        """

        if not hasattr(self, "current_assistant_message") or self.current_assistant_message is None:
            return
        self.current_assistant_message.set_content(markdown_text)

    def _add_chat_message(
        self,
        role: str,
        content: str,
        metrics: str = "",
        resend_prompt: Optional[str] = None,
    ) -> QTextBrowser:
        """Add one chat bubble.

        Args:
            role: Message role, either ``user`` or ``assistant``.
            content: Markdown message content.
            metrics: Optional metric text shown under assistant replies.
            resend_prompt: Prompt to resend from the bubble.

        Returns:
            Text browser used by the bubble.
        """

        should_follow = self._is_chat_near_bottom()
        max_width = max(320, int(self.chat_scroll.viewport().width() * 0.78)) if hasattr(self, "chat_scroll") else 900
        message = ChatMessageWidget(
            role,
            content,
            markdown_to_html,
            self._resend_chat_message,
            metrics=metrics,
            resend_prompt=resend_prompt,
            max_width=max_width,
        )
        self.chat_messages.insertWidget(max(self.chat_messages.count() - 1, 0), message)
        if should_follow:
            message.scroll_later(lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))
        if role == "assistant":
            self.current_assistant_message = message
            self.current_assistant_browser = message.browser
            self.current_assistant_meta = message.meta_label
        return message.browser

    def _is_chat_near_bottom(self) -> bool:
        """Return whether the chat scroll is close enough to follow streaming.

        Returns:
            True when the view should auto-scroll.
        """

        if not hasattr(self, "chat_scroll"):
            return True
        bar = self.chat_scroll.verticalScrollBar()
        return bar.maximum() - bar.value() < 48

    def _clear_chat_messages(self) -> None:
        """Remove all message bubbles."""

        while self.chat_messages.count() > 1:
            item = self.chat_messages.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.current_assistant_message = None
        self.current_assistant_browser = None
        self.current_assistant_meta = None

    def _resend_chat_message(self, prompt: str) -> None:
        """Resend text from a message bubble.

        Args:
            prompt: Prompt text to send.
        """

        self.chat_input.setPlainText(prompt)
        self.send_chat_message()

    def _set_chat_stats(self, elapsed_seconds: float, token_count: int, tokens_per_second: float) -> None:
        """Update live chat generation metrics.

        Args:
            elapsed_seconds: Elapsed generation time.
            token_count: Generated token count.
            tokens_per_second: Approximate token speed.
        """

        text = f"Time: {elapsed_seconds:.2f}s  |  Tokens: {token_count:,}  |  Speed: {tokens_per_second:.2f} tok/s"
        self.chat_stats.setText(text)
        if self.current_assistant_meta is not None:
            self.current_assistant_meta.setText(text)
            self.current_assistant_meta.setVisible(True)

    def _chat_backend_value(self) -> str:
        """Return the selected chat model backend.

        Returns:
            Stable chat backend identifier.
        """

        if not hasattr(self, "chat_model_backend"):
            return "gguf"
        return "microgpt" if self.chat_model_backend.currentText() == "MicroGPT checkpoint" else "gguf"

    def _update_chat_backend_controls(self) -> None:
        """Show controls relevant to the selected chat backend."""

        if not hasattr(self, "chat_model_backend"):
            return
        native = self._chat_backend_value() == "microgpt"
        self.gguf_path_row.setVisible(not native)
        self.microgpt_path_row.setVisible(native)
        self.llama_gpu_layers.setEnabled(not native)
        self.llama_threads.setEnabled(not native)
        self.llama_context.setEnabled(not native)
        if native:
            self._tip(self.load_llm_button, "Load the native MicroGPT checkpoint into memory once for repeated chat messages.")
        else:
            self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")

    def _apply_chat_delta(self, event: dict[str, Any]) -> None:
        """Apply one streamed chat chunk to the rendered conversation.

        Args:
            event: Chat stream progress event.
        """

        self.chat_stream_reply += str(event.get("content", ""))
        should_follow = self._is_chat_near_bottom()
        self._render_chat_markdown(self.chat_stream_reply)
        if should_follow:
            self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())
        self._set_chat_stats(
            float(event.get("elapsed_seconds", 0.0)),
            int(event.get("token_count", 0)),
            float(event.get("tokens_per_second", 0.0)),
        )

    def toggle_llm_model(self) -> None:
        """Load or unload the selected chat model depending on current state."""

        if self.chat_session is not None:
            self.unload_llm_model()
            return
        self.load_llm_model()

    def load_llm_model(self) -> None:
        """Load a selected model backend for chat testing."""

        backend = self._chat_backend_value()
        path_text = self.microgpt_chat_path.text().strip() if backend == "microgpt" else self.gguf_path.text().strip()
        if not path_text:
            required = "MicroGPT model folder or checkpoint" if backend == "microgpt" else "GGUF model file"
            QMessageBox.information(self, "Model required", f"Choose a {required} first.")
            return
        model_path = Path(path_text)
        self.chat_progress.setValue(0)
        self._render_chat_markdown("**Loading model...**")
        self.chat_stats.setText("Loading model...")
        self.project_state.setText("Loading chat model")
        self.chat_status.setText("Chat: loading model")
        loader = load_microgpt_chat_session if backend == "microgpt" else load_llama_chat_session
        args = (
            (model_path, self.device.currentText())
            if backend == "microgpt"
            else (model_path, self.llama_context.value(), self.llama_threads.value(), self.llama_gpu_layers.value())
        )
        self._run_task(
            loader,
            args,
            self._llm_loaded,
            self.chat_event_log,
            self.chat_progress,
            button=self.load_llm_button,
            busy_text="Loading Model",
            task_kind="chat",
        )

    @Slot(object)
    def _llm_loaded(self, session: Any) -> None:
        """Store a loaded GGUF or MicroGPT chat session.

        Args:
            session: Loaded chat session.
        """

        self.chat_session = session
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message(
            "assistant",
            f"Loaded model: `{session.model_path.name}`\n\n{session.runtime_summary}\n\nSend a message to begin.",
        )
        self.chat_progress.setValue(100)
        self.chat_stats.setText(session.runtime_summary)
        self.project_state.setText("Chat model loaded")
        self.chat_status.setText(f"Chat: {session.runtime_summary}")
        self._clear_button_busy("Unload")
        self._tip(self.load_llm_button, "Unload the currently loaded model from memory.")

    def unload_llm_model(self) -> None:
        """Unload the active chat model and clear chat state."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message("assistant", "Model unloaded.\n\nLoad a model to start testing.")
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(0)
        self.chat_stats.setText("Idle")
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: no model loaded")
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()

    def send_chat_message(self) -> None:
        """Send a prompt to the loaded chat model."""

        if self.chat_session is None:
            QMessageBox.information(self, "Load model", "Load a model before sending a message.")
            return
        prompt = self.chat_input.toPlainText().strip()
        if not prompt:
            return
        self.pending_user_message = prompt
        self.chat_input.clear()
        self._add_chat_message("user", prompt, resend_prompt=prompt)
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "_Thinking..._", resend_prompt=prompt)
        self.chat_progress.setRange(0, 0)
        self.chat_stats.setText("Thinking...")
        self.project_state.setText("Generating")
        self.chat_status.setText("Chat: generating reply")
        streamer = stream_microgpt_chat_reply if self._chat_backend_value() == "microgpt" else stream_chat_reply
        self._run_task(
            streamer,
            (
                self.chat_session,
                prompt,
                self.system_prompt.toPlainText(),
                self.chat_max_tokens.value(),
                self.chat_temperature.value(),
                self.chat_top_p.value(),
                self.chat_repeat_penalty.value(),
                self.reasoning_effort.currentText(),
                self.thinking_enabled.isChecked(),
            ),
            self._chat_reply_finished,
            self.chat_event_log,
            self.chat_progress,
            with_progress=True,
            button=self.send_chat_button,
            stop_button=self.stop_chat_button,
            busy_text="Thinking",
        )

    @Slot(object)
    def _chat_reply_finished(self, reply: Any) -> None:
        """Render the model reply.

        Args:
            reply: Assistant reply text and metrics.
        """

        result = reply if isinstance(reply, dict) else {"reply": str(reply)}
        text = str(result.get("reply", "")).strip()
        if text:
            self.chat_stream_reply = text
        else:
            self.chat_stream_reply = self.chat_stream_reply or "_No reply returned._"
        self._render_chat_markdown(self.chat_stream_reply)
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(100)
        self._set_chat_stats(
            float(result.get("elapsed_seconds", 0.0)),
            int(result.get("token_count", 0)),
            float(result.get("tokens_per_second", 0.0)),
        )
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: ready")
        self._clear_button_busy("Send")

    def reset_chat(self) -> None:
        """Clear the chat transcript and model conversation memory."""

        if self.chat_session is not None:
            self.chat_session.reset()
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "Chat reset.")
        self.chat_stats.setText("Idle")
        self.chat_status.setText("Chat: ready")

    def _append_chat_markdown(self, role: str, content: str) -> None:
        """Append one rendered chat message.

        Args:
            role: Display role heading.
            content: Markdown content.
        """

        block = f"### {role}\n{content.strip()}\n"
        self.chat_markdown = f"{self.chat_markdown.rstrip()}\n\n{block}" if self.chat_markdown else block
        self._add_chat_message("user" if role.lower() in {"you", "user"} else "assistant", content)

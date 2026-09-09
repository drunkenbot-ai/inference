from __future__ import annotations

import re
from typing import Callable, Optional
from urllib.parse import unquote

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ChatInputEdit(QTextEdit):
    """Chat input that sends on Enter and inserts newlines with Shift+Enter."""

    sendRequested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle Enter as send.

        Args:
            event: Key event.
        """

        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            self.sendRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class ChatMessageWidget(QWidget):
    """Reusable user/assistant message bubble with outside actions and metrics."""

    def __init__(
        self,
        role: str,
        content: str,
        html_renderer: Callable[[str], str],
        resend_callback: Callable[[str], None],
        metrics: str = "",
        resend_prompt: Optional[str] = None,
        max_width: int = 900,
    ) -> None:
        """Create a chat message widget.

        Args:
            role: Message role.
            content: Markdown message content.
            html_renderer: Callable that converts Markdown to HTML.
            resend_callback: Callable used by the resend action.
            metrics: Optional metrics text.
            resend_prompt: Optional prompt to resend.
            max_width: Maximum bubble width.
        """

        super().__init__()
        self.role = role
        self.content = content
        self.html_renderer = html_renderer
        self.resend_callback = resend_callback
        self.resend_prompt = resend_prompt or content
        self.max_width = max_width

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        self.stack = QWidget()
        stack_layout = QVBoxLayout(self.stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(4)

        self.bubble = QWidget()
        self.bubble.setObjectName("UserBubble" if role == "user" else "AssistantBubble")
        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(12, 10, 12, 10)
        bubble_layout.setSpacing(8)

        self.thought_toggle = QPushButton("💭 Thought Process  ▼")
        self.thought_toggle.setObjectName("ThoughtToggle")
        self.thought_toggle.setCursor(Qt.PointingHandCursor)
        self.thought_toggle.setVisible(False)
        self.thought_toggle.clicked.connect(self._toggle_thought)
        bubble_layout.addWidget(self.thought_toggle)

        self.thought_browser = QTextBrowser()
        self.thought_browser.setObjectName("ThoughtText")
        self.thought_browser.setOpenExternalLinks(False)
        self.thought_browser.setOpenLinks(False)
        self.thought_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thought_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thought_browser.document().setDocumentMargin(0)
        self.thought_browser.setVisible(False)
        bubble_layout.addWidget(self.thought_browser)

        self.browser = QTextBrowser()
        self.browser.setObjectName("BubbleText")
        self.browser.setOpenExternalLinks(False)
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._handle_anchor_clicked)
        self.browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.document().setDocumentMargin(0)
        bubble_layout.addWidget(self.browser)
        stack_layout.addWidget(self.bubble)

        footer = QHBoxLayout()
        footer.setContentsMargins(6, 0, 6, 0)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setObjectName("MessageAction")
        self.copy_button.setFixedWidth(28)
        self.copy_button.setToolTip("Copy this message.")
        self.copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.browser.toPlainText()))
        self.resend_button = QPushButton("Resend")
        self.resend_button.setObjectName("MessageAction")
        self.resend_button.setFixedWidth(28)
        self.resend_button.setToolTip("Send this message again.")
        self.resend_button.clicked.connect(lambda: self.resend_callback(self.resend_prompt))
        self.meta_label = QLabel(metrics)
        self.meta_label.setObjectName("MessageMeta")
        self.meta_label.setVisible(bool(metrics))
        if role == "user":
            footer.addStretch(1)
            footer.addWidget(self.copy_button)
            footer.addWidget(self.resend_button)
        else:
            footer.addWidget(self.copy_button)
            footer.addWidget(self.resend_button)
            footer.addWidget(self.meta_label)
            footer.addStretch(1)
        stack_layout.addLayout(footer)

        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(self.stack)
        else:
            row_layout.addWidget(self.stack)
            row_layout.addStretch(1)

        self.set_content(content)

    def _toggle_thought(self) -> None:
        """Toggle visibility of the internal reasoning trace."""
        is_visible = not self.thought_browser.isVisible()
        self.thought_browser.setVisible(is_visible)
        arrow = "▲" if is_visible else "▼"
        self.thought_toggle.setText(f"💭 Thought Process  {arrow}")
        self._fit_browser()
        self.updateGeometry()

    def set_content(self, content: str) -> None:
        """Update message content, cleanly separating internal reasoning traces.

        Args:
            content: Markdown message content.
        """

        self.content = content
        thought_match = re.search(
            r"<(?:thought|think)>(.*?)(?:<\/(?:thought|think)>|$)",
            content,
            flags=re.DOTALL,
        )
        if thought_match and self.role == "assistant":
            thought_text = thought_match.group(1).strip()
            close_match = re.search(r"<\/(?:thought|think)>", content)
            if close_match:
                reply_text = content[close_match.end() :].strip()
            else:
                reply_text = ""

            if thought_text:
                self.thought_toggle.setVisible(True)
                self.thought_browser.setHtml(self.html_renderer(thought_text))
                # If still streaming the thought before reply begins, keep open
                if not reply_text and not self.thought_browser.isVisible():
                    self.thought_browser.setVisible(True)
                    self.thought_toggle.setText("💭 Thought Process  ▲")
            else:
                self.thought_toggle.setVisible(False)
                self.thought_browser.setVisible(False)

            display_content = (
                reply_text
                if reply_text
                else ("_Thinking in progress..._" if "<thought>" in content or "<think>" in content else content)
            )
        else:
            if hasattr(self, "thought_toggle"):
                self.thought_toggle.setVisible(False)
            if hasattr(self, "thought_browser"):
                self.thought_browser.setVisible(False)
            display_content = content

        self.browser.setHtml(self.html_renderer(display_content))
        self._fit_browser()
        self.updateGeometry()

    def set_metrics(self, metrics: str) -> None:
        """Update metrics text.

        Args:
            metrics: Metrics text.
        """

        self.meta_label.setText(metrics)
        self.meta_label.setVisible(bool(metrics))

    def _fit_browser(self) -> None:
        """Resize message body to content."""

        self.browser.setFont(QFont("Arial", 10))
        self.browser.document().setDocumentMargin(0)
        text = self.browser.toPlainText()
        lines = text.splitlines() or [text]
        metrics = QFontMetrics(self.browser.font())
        longest = max((metrics.horizontalAdvance(line[:180]) for line in lines), default=240)
        width = max(240, min(longest + 58, self.max_width))

        if hasattr(self, "thought_browser") and self.thought_browser.isVisible():
            self.thought_browser.setFont(QFont("Arial", 9))
            self.thought_browser.document().setDocumentMargin(0)
            self.thought_browser.setFixedWidth(width)
            self.thought_browser.document().setTextWidth(width - 28)
            self.thought_browser.document().adjustSize()
            t_height = int(self.thought_browser.document().size().height()) + 16
            t_fitted = max(32, min(t_height, 420))
            self.thought_browser.setFixedHeight(t_fitted)
            if hasattr(self, "thought_toggle"):
                self.thought_toggle.setFixedWidth(width)
        elif hasattr(self, "thought_toggle") and self.thought_toggle.isVisible():
            self.thought_toggle.setFixedWidth(width)

        self.browser.setFixedWidth(width)
        self.browser.document().setTextWidth(width - 28)
        self.browser.document().adjustSize()
        height = int(self.browser.document().size().height()) + 24
        fitted_height = max(36, height)
        self.browser.setMinimumHeight(fitted_height)
        self.browser.setMaximumHeight(fitted_height)
        self.browser.setFixedHeight(fitted_height)
        self.bubble.adjustSize()
        self.stack.adjustSize()
        self.adjustSize()

    def scroll_later(self, scroll_callback: Callable[[], None]) -> None:
        """Schedule a scroll callback after layout settles.

        Args:
            scroll_callback: Callback to run.
        """

        QTimer.singleShot(0, scroll_callback)

    def _handle_anchor_clicked(self, url) -> None:
        """Handle links inside a message.

        Args:
            url: Clicked URL.
        """

        link = url.toString()
        if link.startswith("copycode:"):
            QApplication.clipboard().setText(unquote(link[len("copycode:") :]))
            self.set_content(self.content)

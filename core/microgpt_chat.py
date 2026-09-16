from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.model import MicroGPT
from engine.tokenizer import EOS_TOKEN, load_tokenizer, token_id
from engine.tool_call_data import STANDARD_AGENT_TOOLS
from .agent_executor import execute_agent_tool, parse_tool_calls

STOP_SEQUENCES = (
    "\nUser:",
    "\nSystem:",
    "\nHuman:",
    "\nAssistant:",
    "</tool_calls>",
    "</CALL>",
    "<|endoftext|>",
    "<eos>",
)


class MicroGPTChatSession:
    """Persistent chat session backed by a native MicroGPT checkpoint."""

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        """Load a native MicroGPT checkpoint for repeated prompts.

        Args:
            model_path: Model folder or checkpoint path.
            device: Device selector: auto, cuda, or cpu.

        Raises:
            FileNotFoundError: If checkpoint or tokenizer files are missing.
            ValueError: If the checkpoint is not a MicroGPT checkpoint.
        """

        self.model_path = _resolve_model_checkpoint(model_path)
        self.model_dir = self.model_path.parent
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            # Training checkpoints are stored in a child checkpoints folder,
            # while the tokenizer is copied to the training output directory.
            for cand_dir in (self.model_dir.parent, self.model_dir.parent.parent):
                cand_tok = cand_dir / "tokenizer.json"
                if cand_tok.exists():
                    tokenizer_path = cand_tok
                    break

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                "Tokenizer not found beside checkpoint or its training output folder: "
                f"{self.model_path}. Expected {self.model_dir / 'tokenizer.json'} "
                f"or {self.model_dir.parent / 'tokenizer.json'}."
            )
        requested_device = device.lower().strip()
        self.device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device not in {"cuda", "cpu"}:
            self.device = "cpu"

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config_data = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None

        # Resilient fallback: Check if checkpoint is a legacy raw state dict
        if state_dict is None and isinstance(checkpoint, dict) and any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
            # Try to recover model_config from sibling or parent metadata
            if not isinstance(config_data, dict):
                candidates = [
                    self.model_dir / "training_summary.json",
                    self.model_dir.parent / "training_summary.json",
                    self.model_dir.parent.parent / "training_summary.json",
                    self.model_dir / "project.json",
                    self.model_dir.parent / "project.json",
                    self.model_dir.parent.parent / "project.json",
                ]
                for cand in candidates:
                    if cand.exists():
                        try:
                            with open(cand, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            m_cfg = data.get("model_config")
                            if isinstance(m_cfg, str):
                                m_cfg = json.loads(m_cfg)
                            if isinstance(m_cfg, dict) and ("embedding_size" in m_cfg or "vocab_size" in m_cfg):
                                config_data = m_cfg
                                break
                        except Exception:
                            pass

        if not isinstance(config_data, dict) or not state_dict:
            raise ValueError("Checkpoint must contain model_config and model_state_dict.")
        self.config = ModelConfig(**config_data)
        self.model = MicroGPT(self.config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.eos_id = token_id(self.tokenizer, EOS_TOKEN)
        self._lock = Lock()
        self._messages: list[dict[str, str]] = []

    @property
    def runtime_summary(self) -> str:
        """Return a short runtime summary.

        Returns:
            Runtime summary text.
        """

        return (
            f"Runtime: native MicroGPT on {self.device.upper()} | "
            f"{self.config.layer_count} layers, {self.config.embedding_size} hidden, ctx {self.config.context_length}"
        )

    def reset(self) -> None:
        """Clear conversation history while keeping the model loaded."""

        with self._lock:
            self._messages = []

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        reasoning_effort: str = "Balanced",
        thinking_enabled: bool = True,
        enable_tools: bool = True,
        max_tool_hops: int = 3,
        progress: Optional[Callable[[Any], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        """Stream one assistant reply with autonomous ReAct tool execution.

        Args:
            prompt: User message.
            system_prompt: Optional system instruction.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            repeat_penalty: Penalty for generated token repetition.
            reasoning_effort: Effort mode label.
            thinking_enabled: Whether to add reasoning guidance.
            enable_tools: Whether to parse and execute agent tool calls automatically.
            max_tool_hops: Maximum number of sequential tool round-trips allowed.
            progress: Optional progress callback.
            should_stop: Optional cancellation callback.

        Returns:
            Reply text and timing metrics.
        """

        started_at = perf_counter()
        accumulated_reply = ""
        total_generated_tokens = 0

        with self._lock, torch.no_grad():
            prompt_text = self._render_prompt(
                prompt,
                system_prompt,
                reasoning_effort,
                thinking_enabled,
                max_tokens=max_tokens,
                enable_tools=enable_tools,
            )
            input_ids = self.tokenizer.encode(prompt_text).ids[-self.config.context_length :]
            ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            current_hop = 0
            while current_hop <= max_tool_hops:
                if should_stop and should_stop():
                    break

                generated_ids: list[int] = []
                emitted_text = ""
                stop_hit = None

                for _ in range(max_tokens):
                    if should_stop and should_stop():
                        break
                    idx_cond = ids[:, -self.config.context_length :]
                    logits = self.model(idx_cond)[:, -1, :]
                    logits = self._apply_repeat_penalty(logits, generated_ids, repeat_penalty)
                    next_id = self._sample_next_token(logits, temperature, top_p)
                    if next_id == self.eos_id and generated_ids:
                        break
                    ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.long, device=self.device)), dim=1)
                    generated_ids.append(next_id)
                    total_generated_tokens += 1

                    full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    if full_text.endswith("\ufffd"):
                        continue

                    for s in STOP_SEQUENCES:
                        if s in full_text:
                            stop_hit = s
                            break
                    if stop_hit is not None:
                        break

                    piece = full_text[len(emitted_text) :]
                    if not piece:
                        continue
                    if any(s.startswith(piece) for s in STOP_SEQUENCES):
                        continue
                    emitted_text = full_text
                    elapsed = max(perf_counter() - started_at, 0.001)
                    if progress:
                        progress(
                            {
                                "type": "chat_delta",
                                "content": piece,
                                "elapsed_seconds": elapsed,
                                "token_count": total_generated_tokens,
                                "tokens_per_second": total_generated_tokens / elapsed,
                            }
                        )

                hop_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip() if generated_ids else ""
                for s in STOP_SEQUENCES:
                    if s in hop_text:
                        if s in ("</tool_calls>", "</CALL>"):
                            idx = hop_text.find(s)
                            hop_text = hop_text[: idx + len(s)].strip()
                        else:
                            hop_text = hop_text.split(s)[0].strip()
                hop_text = hop_text.rstrip("\ufffd").strip()

                if accumulated_reply:
                    accumulated_reply = f"{accumulated_reply}\n{hop_text}".strip()
                else:
                    accumulated_reply = hop_text

                # Check if this hop produced tool calls that need execution
                if stop_hit in ("</tool_calls>", "</CALL>") and enable_tools and current_hop < max_tool_hops:
                    calls = parse_tool_calls(hop_text)
                    if calls:
                        current_hop += 1
                        for call in calls:
                            tool_name = call["name"]
                            tool_args = call["arguments"]
                            elapsed = max(perf_counter() - started_at, 0.001)
                            call_banner = f"\n\n⚙️ *Calling tool `{tool_name}`...*\n"
                            accumulated_reply += call_banner
                            if progress:
                                progress({
                                    "type": "chat_delta",
                                    "content": call_banner,
                                    "elapsed_seconds": elapsed,
                                    "token_count": total_generated_tokens,
                                    "tokens_per_second": total_generated_tokens / elapsed,
                                })
                            tool_obs = execute_agent_tool(tool_name, tool_args)
                            obs_block = f"\n<tool_result id=\"{call['id']}\">{tool_obs}</tool_result>\nAssistant:"
                            obs_display = f"📋 *Observation*: `{tool_obs[:140] + ('...' if len(tool_obs) > 140 else '')}`\n\n"
                            accumulated_reply += obs_display
                            if progress:
                                progress({
                                    "type": "chat_delta",
                                    "content": obs_display,
                                    "elapsed_seconds": max(perf_counter() - started_at, 0.001),
                                    "token_count": total_generated_tokens,
                                    "tokens_per_second": total_generated_tokens / max(perf_counter() - started_at, 0.001),
                                })
                            obs_ids = self.tokenizer.encode(obs_block).ids
                            ids = torch.cat((ids, torch.tensor([obs_ids], dtype=torch.long, device=self.device)), dim=1)
                        continue

                # No tool call or execution finished
                break

            reply = accumulated_reply.strip()
            if reply:
                self._messages.append({"role": "user", "content": prompt})
                self._messages.append({"role": "assistant", "content": reply})

        elapsed = max(perf_counter() - started_at, 0.001)
        return {
            "reply": reply,
            "elapsed_seconds": elapsed,
            "token_count": total_generated_tokens,
            "tokens_per_second": total_generated_tokens / elapsed if total_generated_tokens else 0.0,
            "stopped": bool(should_stop and should_stop()),
        }

    def _render_prompt(
        self,
        prompt: str,
        system_prompt: str,
        reasoning_effort: str,
        thinking_enabled: bool,
        max_tokens: int = 512,
        enable_tools: bool = False,
    ) -> str:
        """Render chat history into plain text for MicroGPT with turn-aware history pruning.

        Args:
            prompt: Latest user message.
            system_prompt: Optional system instruction.
            reasoning_effort: Effort mode label.
            thinking_enabled: Whether reasoning guidance is enabled.
            max_tokens: Maximum new tokens reserved for reply.
            enable_tools: Whether to declare available agent tools in the system prompt.

        Returns:
            Prompt text.
        """

        system_text = system_prompt.strip()
        if thinking_enabled and reasoning_effort not in {"None", "none"}:
            effort_text = self._effort_instruction(reasoning_effort)
            if effort_text and effort_text not in system_text:
                system_text = f"{system_text} {effort_text}".strip() if system_text else effort_text
        elif not system_text and not thinking_enabled:
            system_text = self._plain_instruction()

        if enable_tools and "<tools>" not in system_text:
            tools_decl = "Tools:\n<tools>\n" + json.dumps(STANDARD_AGENT_TOOLS, indent=2) + "\n</tools>"
            decl_len = len(self.tokenizer.encode(tools_decl).ids)
            if self.config.context_length > decl_len + max_tokens + 16:
                system_text = f"{system_text}\n\n{tools_decl}".strip() if system_text else tools_decl

        parts = []
        if system_text:
            parts.append(f"System: {system_text}")

        latest_turn = f"User: {prompt}\nAssistant:"

        # Calculate token budget available for conversation history
        prefix_ids = self.tokenizer.encode("\n".join(parts) + ("\n" if parts else "")).ids
        suffix_ids = self.tokenizer.encode("\n" + latest_turn).ids
        overhead = len(prefix_ids) + len(suffix_ids)
        available_budget = max(0, self.config.context_length - max_tokens - overhead - 4)

        # Walk history backwards, taking complete turns while budget allows
        history_parts: list[str] = []
        total_hist_tokens = 0
        for message in reversed(self._messages):
            role = "User" if message["role"] == "user" else "Assistant"
            line = f"{role}: {message['content']}"
            line_tokens = len(self.tokenizer.encode(line + "\n").ids)
            if total_hist_tokens + line_tokens > available_budget:
                break
            history_parts.insert(0, line)
            total_hist_tokens += line_tokens

        parts.extend(history_parts)
        parts.append(latest_turn)
        return "\n".join(parts)

    def _apply_repeat_penalty(self, logits: torch.Tensor, generated_ids: list[int], repeat_penalty: float) -> torch.Tensor:
        """Apply a simple repeat penalty to recently generated tokens."""

        if repeat_penalty <= 1.0 or not generated_ids:
            return logits
        for token in set(generated_ids[-128:]):
            logits[:, token] = logits[:, token] / repeat_penalty
        return logits

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """Sample the next token from logits."""

        temperature = max(float(temperature), 1e-5)
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > max(0.01, min(1.0, float(top_p)))
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _effort_instruction(reasoning_effort: str) -> str:
        """Translate effort label into prompt guidance."""

        if reasoning_effort == "Fast":
            return "Answer concisely. Put code inside fenced Markdown code blocks with language labels."
        if reasoning_effort == "Deep":
            return "Think carefully and provide a detailed answer when useful. Put code inside fenced Markdown code blocks with language labels."
        return "Use balanced reasoning and answer clearly. Put code inside fenced Markdown code blocks with language labels."

    @staticmethod
    def _plain_instruction() -> str:
        """Return direct answer guidance."""

        return "Answer directly. Put code inside fenced Markdown code blocks with language labels."


def _resolve_model_checkpoint(path: Path) -> Path:
    """Resolve a model folder or checkpoint path to a checkpoint file."""

    path = Path(path)
    if path.is_dir():
        final_model = path / "final_model.pt"
        if final_model.exists():
            return final_model
        checkpoints = sorted((path / "checkpoints").glob("checkpoint_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        if checkpoints:
            return checkpoints[0]
    if path.exists() and path.suffix == ".pt":
        return path
    raise FileNotFoundError(f"MicroGPT checkpoint not found: {path}")


def load_microgpt_chat_session(model_path: Path, device: str = "auto") -> MicroGPTChatSession:
    """Load a native MicroGPT chat session.

    Args:
        model_path: Model folder or checkpoint path.
        device: Device selector.

    Returns:
        Loaded MicroGPT chat session.
    """

    return MicroGPTChatSession(model_path, device=device)


def stream_microgpt_chat_reply(
    session: MicroGPTChatSession,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    reasoning_effort: str,
    thinking_enabled: bool = True,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Stream a reply from a native MicroGPT chat session."""

    return session.generate_stream(
        prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        reasoning_effort=reasoning_effort,
        thinking_enabled=thinking_enabled,
        progress=progress,
        should_stop=should_stop,
    )

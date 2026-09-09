from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union  # noqa: F401

from PySide6.QtCore import Slot

from inference.core.evaluation import evaluate_checkpoint, normalize_prompts


class BenchmarkScreenMixin:
    """Mixin providing benchmark test execution and progress reporting."""

    def run_benchmark(self) -> None:
        """Run benchmark prompts against the current trained model."""

        prompts = normalize_prompts(self.benchmark_prompts.toPlainText())
        self.benchmark_log.append(f"Running benchmark with {len(prompts)} prompt(s)...")
        self.benchmark_progress.setValue(0)
        self.project_state.setText("Benchmarking")
        self._run_task(
            evaluate_checkpoint,
            (
                Path(self.model_dir.text()),
                prompts,
                None,
                self.benchmark_tokens.value(),
                self.benchmark_temperature.value(),
                50,
                self.device.currentText(),
                self.benchmark_kv_cache.isChecked(),
            ),
            self._benchmark_finished,
            self.benchmark_log,
            self.benchmark_progress,
            with_progress=True,
            button=self.run_benchmark_button,
            stop_button=self.stop_benchmark_button,
            busy_text="Benchmarking",
        )

    @Slot(object)
    def _benchmark_finished(self, result: Any) -> None:
        """Update UI after benchmark prompts finish.

        Args:
            result: Benchmark result object.
        """

        self.benchmark_progress.setRange(0, 100)
        self.benchmark_progress.setValue(100)
        self.benchmark_log.append(
            f"Benchmark complete: {result.prompt_count} prompt(s), {result.total_seconds:.2f}s, "
            f"{result.total_generated_tokens} generated token(s), {result.tokens_per_second:.2f} tok/s."
        )
        self.benchmark_log.append(f"Benchmark saved: {result.output_path}")
        self.project_state.setText("Benchmark complete")
        self._clear_button_busy("Run Benchmark")

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from engine.config import ModelConfig
from engine.model import MicroGPT
from engine.tokenizer import EOS_TOKEN, load_tokenizer, token_id


def load_model_from_checkpoint(checkpoint_path: Path, device: Optional[str] = None) -> MicroGPT:
    """Load a trained MicroGPT checkpoint.

    Args:
        checkpoint_path: Path to a saved model checkpoint.
        device: Optional device override.

    Returns:
        Loaded model in evaluation mode.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    state_dict = checkpoint["model_state_dict"]

    # Check if the checkpoint contains LoRA adapter weights (e.g. unmerged checkpoint)
    is_lora = (
        checkpoint.get("peft_method") == "lora"
        or any(".lora_a" in k or ".base.weight" in k for k in state_dict.keys())
    )
    if is_lora:
        from engine.model_norm_lora import apply_lora_adapters, merged_lora_state_dict

        lora_cfg = checkpoint.get("lora_config") or {}
        inferred_rank = next(
            (v.shape[0] for k, v in state_dict.items() if ".lora_a" in k and hasattr(v, "shape")),
            8,
        )
        rank = int(lora_cfg.get("rank") or inferred_rank)
        alpha = float(lora_cfg.get("alpha") or (2 * rank))
        dropout = float(lora_cfg.get("dropout") or 0.0)
        target_modules = lora_cfg.get("target_modules") or "all"

        temp_model = MicroGPT(config)
        apply_lora_adapters(temp_model, rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)
        temp_model.load_state_dict(state_dict)
        state_dict = merged_lora_state_dict(temp_model)

    model = MicroGPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_text(
    checkpoint_path: Path,
    tokenizer_path: Path,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device: Optional[str] = None,
    use_kv_cache: bool = True,
) -> str:
    """Generate text from a trained checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        tokenizer_path: Path to tokenizer JSON.
        prompt: Prompt text.
        max_new_tokens: Maximum tokens to sample.
        temperature: Sampling temperature.
        top_k: Optional top-k sampling cutoff.
        device: Optional device override.
        use_kv_cache: Whether to use key/value cache during generation.

    Returns:
        Decoded generated text.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    input_ids = tokenizer.encode(prompt).ids
    eos_id = token_id(tokenizer, EOS_TOKEN)
    while input_ids and input_ids[-1] == eos_id:
        input_ids.pop()
    context = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k, use_kv_cache=use_kv_cache)
    output_ids = generated[0].tolist()
    if eos_id in output_ids[len(input_ids) :]:
        eos_index = output_ids.index(eos_id, len(input_ids))
        output_ids = output_ids[:eos_index]
    return tokenizer.decode(output_ids)

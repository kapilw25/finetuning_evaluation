"""
Shared Generation Utilities for All Evaluations

Provides:
- batch_generate(): Batch generation with configurable batch_size (~8x speedup)
- cleanup_gpu(): Proper GPU memory cleanup (gc.collect + cache clear)
- format_chat_messages(): Format messages using chat template
"""

import gc
import torch
import time
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm


def cleanup_gpu():
    """
    Proper GPU memory cleanup

    Call after unloading models to ensure memory is freed.
    Uses gc.collect() + torch.cuda.empty_cache() like toxicity.py
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    show_progress: bool = True,
    desc: str = "Generating",
    checkpoint_callback: Callable = None,
    checkpoint_interval: int = 100
) -> List[str]:
    """
    Batch generation with configurable batch_size

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of formatted prompts (already with chat template applied)
        batch_size: Number of prompts to process at once (default 8)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        show_progress: Show tqdm progress bar
        desc: Description for progress bar
        checkpoint_callback: Function to call for checkpointing (receives list of responses so far)
        checkpoint_interval: Save checkpoint every N responses

    Returns:
        List of generated response strings
    """
    responses = []
    device = next(model.parameters()).device

    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, total=(len(prompts) + batch_size - 1) // batch_size)

    for i in iterator:
        batch_prompts = prompts[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode responses (skip prompt tokens)
        for j, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][j].shape[0]
            response = tokenizer.decode(
                output[prompt_length:],
                skip_special_tokens=True
            ).strip()
            responses.append(response)

        # Checkpoint callback
        if checkpoint_callback and len(responses) % checkpoint_interval == 0:
            checkpoint_callback(responses)

    return responses


def format_chat_messages(
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    add_generation_prompt: bool = True
) -> List[str]:
    """
    Format multiple message lists using chat template

    Args:
        tokenizer: Tokenizer with chat_template
        messages_list: List of message lists, each like [{"role": "system", ...}, {"role": "user", ...}]
        add_generation_prompt: Whether to add generation prompt

    Returns:
        List of formatted prompt strings
    """
    formatted = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        formatted.append(text)
    return formatted

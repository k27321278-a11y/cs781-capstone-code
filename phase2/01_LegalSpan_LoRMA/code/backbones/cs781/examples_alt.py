"""Batching helpers for encoder/recurrent models with an explicit prompt stream."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Seq2SeqBatch:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    decoder_ids: torch.Tensor
    targets: torch.Tensor
    input_tokens: int
    target_tokens: int


def collate_b(examples, pad_id, bos_id, eos_id, max_seq_len):
    """Create prompt-only encoder inputs and BOS-shifted suffix decoder inputs."""
    if not examples:
        raise ValueError("Empty batch")
    prompts, decoder_inputs, labels = [], [], []
    for example in examples:
        if not example.prefix_ids or not example.suffix_ids:
            raise ValueError("Empty prefix/suffix")
        prompt = [bos_id, *example.prefix_ids]
        decoder = [bos_id, *example.suffix_ids]
        target = [*example.suffix_ids, eos_id]
        if max(len(prompt), len(decoder)) > max_seq_len:
            raise ValueError("Encoder or decoder sequence exceeds context; do not truncate")
        # Preserve the common A/D token-budget boundary, allowing B's second BOS.
        if len(prompt) + len(decoder) > max_seq_len + 1:
            raise ValueError("Combined example exceeds the shared training context")
        prompts.append(prompt)
        decoder_inputs.append(decoder)
        labels.append(target)

    prompt_width = max(map(len, prompts))
    decoder_width = max(map(len, decoder_inputs))
    prompt_ids = torch.full((len(examples), prompt_width), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((len(examples), prompt_width), dtype=torch.bool)
    decoder_ids = torch.full((len(examples), decoder_width), pad_id, dtype=torch.long)
    targets = torch.full_like(decoder_ids, -100)
    for row, (prompt, decoder, target) in enumerate(zip(prompts, decoder_inputs, labels)):
        prompt_ids[row, :len(prompt)] = torch.tensor(prompt)
        prompt_mask[row, :len(prompt)] = True
        decoder_ids[row, :len(decoder)] = torch.tensor(decoder)
        targets[row, :len(target)] = torch.tensor(target)
    return Seq2SeqBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        decoder_ids=decoder_ids,
        targets=targets,
        input_tokens=sum(map(len, prompts)) + sum(map(len, decoder_inputs)),
        target_tokens=sum(map(len, labels)),
    )

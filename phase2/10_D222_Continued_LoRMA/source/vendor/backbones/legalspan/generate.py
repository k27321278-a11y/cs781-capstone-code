"""Cached learned continuation generation; incomplete output is never padded."""
import contextlib
import torch
from cs781.contracts import GenerationIncomplete, PromptTooLong
from .data import role_ids


@torch.inference_mode()
def generate(model, tokenizer, prompt, cfg):
    if not isinstance(prompt, str) or len(prompt.split()) != 512:
        raise ValueError('Exactly 512 prompt words required')
    if cfg.target_words != 128 or cfg.max_new_tokens < 1 or cfg.strategy not in ('greedy', 'nucleus') or cfg.repetition_penalty < 1 or cfg.temperature <= 0 or not 0 < cfg.top_p <= 1:
        raise ValueError('Invalid decoding policy')
    device = next(model.parameters()).device
    encoded = tokenizer.encode(prompt)
    ids = [1] + encoded.ids
    remaining = model.cfg.max_seq_len - len(ids)
    if remaining <= 1:
        raise PromptTooLong('Full prompt exhausts context; no truncation')
    marks = torch.tensor([[0] + role_ids(prompt, encoded)], device=device, dtype=torch.long)
    model.eval(); cache = None; generated = []
    current = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    for _ in range(min(remaining, cfg.max_new_tokens)):
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
        with ctx:
            scores, _, cache = model(current, prompt_length=len(ids), prompt_roles=marks,
                                     cache=cache, use_cache=True)
        values = scores[0, -1].float().clone()
        if not torch.isfinite(values).all(): raise GenerationIncomplete('Nonfinite output distribution')
        if generated and cfg.repetition_penalty != 1:
            index = torch.tensor(sorted(set(generated)), device=device)
            vals = values[index]
            values[index] = torch.where(vals < 0, vals * cfg.repetition_penalty, vals / cfg.repetition_penalty)
        # All compared architectures share fixed 128-word constrained decoding.
        # EOS remains a supervised vocabulary token; it is barred before 128 words.
        values[:4] = -torch.inf
        if cfg.strategy == 'greedy': token = int(values.argmax())
        else:
            logits, order = torch.sort(values / cfg.temperature, descending=True)
            probabilities = logits.softmax(-1)
            logits[probabilities.cumsum(-1) - probabilities > cfg.top_p] = -torch.inf
            token = int(order[torch.multinomial(logits.softmax(-1), 1, generator=generator)])
        generated.append(token)
        text = tokenizer.decode(generated)
        words = text.split()
        if len(words) > 128 or len(words) == 128 and text[-1:].isspace():
            return ' '.join(words[:128])
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    raise GenerationIncomplete('Token/context cap reached before 128 complete words; no filler')

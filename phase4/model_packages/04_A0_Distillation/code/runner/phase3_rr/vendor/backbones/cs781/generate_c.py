"""Cached autoregressive decoding for architecture C."""
import contextlib

import torch

from .contracts import GenerationIncomplete, PromptTooLong, SPECIAL_TOKENS


@torch.inference_mode()
def generate_continuation_c(model, tokenizer, prompt, config):
    if not isinstance(prompt, str) or len(prompt.split()) != 512:
        raise ValueError("Official continuation prompts must contain exactly 512 words")
    if (config.target_words != 128 or config.max_new_tokens < 1
            or config.repetition_penalty < 1
            or config.strategy not in ("greedy", "nucleus")):
        raise ValueError("Invalid continuation decoding config")
    if config.temperature <= 0 or not 0 < config.top_p <= 1:
        raise ValueError("Invalid temperature/top_p")

    device = next(model.parameters()).device
    prompt_ids = [tokenizer.token_to_id("<bos>"), *tokenizer.encode(prompt).ids]
    if len(prompt_ids) > model.cfg.max_prompt_len:
        raise PromptTooLong("Prompt exceeds encoder context; do not truncate")
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_mask = torch.ones_like(prompt_tensor, dtype=torch.bool)
    decoder_ids = torch.tensor([[tokenizer.token_to_id("<bos>")]],
                               dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    banned = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    generated, cache = [], None
    # Decoder positions are separate from encoder positions in approved C design.
    max_steps = min(config.max_new_tokens, model.cfg.max_seq_len - 1)
    model.eval()
    for _ in range(max_steps):
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else contextlib.nullcontext())
        with autocast:
            if cache is None:
                logits, _, cache = model(prompt_tensor, decoder_ids,
                                         prompt_mask=prompt_mask, use_cache=True)
            else:
                logits, _, cache = model(None, decoder_ids, past_kvs=cache, use_cache=True)
        scores = logits[0, -1].float().clone()
        if not torch.isfinite(scores).all():
            raise GenerationIncomplete("Nonfinite generation logits")
        if generated and config.repetition_penalty != 1:
            index = torch.tensor(sorted(set(generated)), device=device)
            values = scores[index]
            scores[index] = torch.where(values < 0, values * config.repetition_penalty,
                                        values / config.repetition_penalty)
        scores[banned] = -torch.inf
        if config.strategy == "greedy":
            token = int(scores.argmax())
        else:
            values, indices = torch.sort(scores / config.temperature, descending=True)
            probabilities = values.softmax(-1)
            exclude = probabilities.cumsum(-1) - probabilities > config.top_p
            values[exclude] = -torch.inf
            token = int(indices[torch.multinomial(values.softmax(-1), 1, generator=generator)])
        generated.append(token)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        words = text.split()
        if len(words) > 128 or (len(words) == 128 and text and text[-1].isspace()):
            return " ".join(words[:128])
        decoder_ids = torch.tensor([[token]], dtype=torch.long, device=device)
    raise GenerationIncomplete("Token/context cap reached before 128 complete words; no filler added")

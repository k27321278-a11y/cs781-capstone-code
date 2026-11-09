import contextlib

import torch

from .contracts import GenerationIncomplete, PromptTooLong, SPECIAL_TOKENS


@torch.inference_mode()
def generate_continuation(model,tokenizer,prompt,config):
    if not isinstance(prompt,str) or len(prompt.split()) != 512:
        raise ValueError("Official continuation prompts must contain exactly 512 words")
    if config.target_words!=128 or config.max_new_tokens<1 or config.repetition_penalty<1 or config.strategy not in ("greedy","nucleus"):
        raise ValueError("Invalid continuation decoding config")
    if config.temperature<=0 or not 0<config.top_p<=1:
        raise ValueError("Invalid temperature/top_p")
    device = next(model.parameters()).device
    ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(prompt).ids
    remaining = model.cfg.max_seq_len-len(ids)
    if remaining <= 1:
        raise PromptTooLong("Prompt exhausts context; increase/re-profile context, do not truncate")
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    generated,cache = [],None
    current = torch.tensor([ids],dtype=torch.long,device=device)
    banned = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    for _ in range(min(config.max_new_tokens,remaining)):
        ctx = torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else contextlib.nullcontext()
        with ctx:
            logits,_,cache = model(current,past_kvs=cache,use_cache=True)
        scores = logits[0,-1].float().clone()
        if not torch.isfinite(scores).all():
            raise GenerationIncomplete("Nonfinite generation logits")
        if generated and config.repetition_penalty!=1:
            index = torch.tensor(sorted(set(generated)),device=device)
            values = scores[index]
            scores[index] = torch.where(values<0,values*config.repetition_penalty,values/config.repetition_penalty)
        scores[banned] = -torch.inf
        if config.strategy=="greedy":
            token = int(scores.argmax())
        else:
            values,index = torch.sort(scores/config.temperature,descending=True)
            probabilities = values.softmax(-1)
            exclude = probabilities.cumsum(-1)-probabilities > config.top_p
            values[exclude] = -torch.inf
            token = int(index[torch.multinomial(values.softmax(-1),1,generator=generator)])
        generated.append(token)
        text = tokenizer.decode(generated,skip_special_tokens=True)
        words = text.split()
        # Wait for the next word boundary; do not return an incomplete BPE fragment as word 128.
        if len(words)>128 or len(words)==128 and text and text[-1].isspace():
            return " ".join(words[:128])
        current = torch.tensor([[token]],device=device,dtype=torch.long)
    raise GenerationIncomplete("Token/context cap reached before 128 complete words; no filler added")

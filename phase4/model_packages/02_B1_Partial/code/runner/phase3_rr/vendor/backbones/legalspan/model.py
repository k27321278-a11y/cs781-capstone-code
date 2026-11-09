"""Manual role/continuity prompt route and learned pointer/vocabulary mixture.

Backbone primitives: our A0 implementation (RMSNorm, RoPE, SwiGLU, causal MHA;
LLaMA 2023). Copy-mixture precedent: See et al. ACL 2017 / CopyNet ACL 2016.
This experiment tests joint role/span conditioning plus internal residual fusion;
it does not claim invention of copying or gating. See docs/ATTRIBUTION.md.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from cs781.contracts import ModelConfig
from cs781.models.a import LegalGPT, RMSNorm, TransformerBlock, RotaryEmbedding


@dataclass(frozen=True)
class SpanConfig(ModelConfig):
    max_seq_len: int = 8192
    route_layers: int = 4
    route_dim: int = 128
    route_mode: str = 'full'
    use_roles: bool = True
    use_continuity: bool = True
    use_fusion: bool = True
    use_copy: bool = True

    def validate(self):
        super().validate()
        if self.route_mode not in ('control', 'generic', 'full') or not 1 <= self.route_layers <= self.n_layers or self.route_dim < 1:
            raise ValueError('Invalid legal-span configuration')


def continuity(history, prompt, query_start):
    """Longest observed suffix of history matching tokens immediately before j.

    Tensor comparisons compute all query/prompt pairs in parallel. No next-label
    input and no recurrence over attention distributions. Supports cached history.
    """
    b, length = history.shape
    m = prompt.shape[1]
    if query_start < 0 or query_start >= length:
        raise ValueError('Invalid observed-history query boundary')
    q = torch.arange(query_start, length, device=history.device)
    j = torch.arange(m, device=history.device)
    match = torch.ones((b, len(q), m), dtype=torch.bool, device=history.device)
    out = torch.zeros_like(match, dtype=torch.float32)
    for k in range(1, 5):
        hi, pi = q - k + 1, j - k
        valid = (hi[:, None] >= 0) & (pi[None, :] >= 0)
        match = match & valid[None] & (history[:, hi.clamp_min(0), None] == prompt[:, None, pi.clamp_min(0)])
        out = torch.where(match, float(k), out)
    return out


def copy_distribution(alpha, prompt_ids, vocab_size):
    result = torch.zeros((*alpha.shape[:2], vocab_size), dtype=torch.float32, device=alpha.device)
    result.scatter_add_(-1, prompt_ids[:, None].expand(-1, alpha.shape[1], -1), alpha.float())
    return result


class PromptRoute(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, r = cfg.d_model, cfg.route_dim
        self.norm = RMSNorm(d)
        self.q, self.k, self.v = [nn.Linear(d, r, bias=False) for _ in range(3)]
        self.out = nn.Linear(r, d, bias=False)
        self.role_q = nn.Linear(d, r, bias=False)
        self.role_embeddings = nn.Embedding(5, r)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.gate = nn.Linear(d + r + 1, 1)

    def forward(self, x, start, prompt_length, roles, eligible, counts, memory=None):
        if self.cfg.route_mode != 'full' or not self.cfg.use_continuity:
            counts = torch.zeros_like(counts)
        normalized = self.norm(x)
        if memory is None:
            prompt = normalized[:, :prompt_length]
            memory = (self.k(prompt), self.v(prompt))
        key, value = memory
        query = normalized[:, start:]
        scores = torch.bmm(self.q(query).float(), key.float().transpose(1, 2)) / math.sqrt(self.cfg.route_dim)
        if self.cfg.route_mode == 'full' and self.cfg.use_roles:
            bias = torch.bmm(self.role_q(query).float(), self.role_embeddings(roles).float().transpose(1, 2)) / math.sqrt(self.cfg.route_dim)
            scores = scores + bias
        if self.cfg.route_mode == 'full' and self.cfg.use_continuity:
            scores = scores + F.softplus(self.gamma.float()) * torch.log1p(counts)
        has_copy = eligible.any(-1)
        # An all-ineligible row gets a finite temporary softmax, then zero mass.
        safe_mask = eligible | (~has_copy)[:, None]
        scores = scores.masked_fill(~safe_mask[:, None], -torch.inf)
        alpha = scores.float().softmax(-1) * eligible[:, None]
        context = torch.bmm(alpha.to(value.dtype), value)
        strength = (alpha * counts).sum(-1, keepdim=True)
        gate = self.gate(torch.cat((query, context.to(query.dtype), strength.to(query.dtype)), -1)).sigmoid()
        update = self.out(context) * gate
        if self.cfg.use_fusion:
            # Earlier prompt states stay unchanged: no future-prompt leakage.
            x = torch.cat((x[:, :start], x[:, start:] + update), 1)
        return x, memory, alpha, context, strength


@dataclass
class SpanCache:
    kvs: list
    memories: list
    prompt_ids: torch.Tensor
    roles: torch.Tensor
    history: torch.Tensor


class LegalSpanGPT(LegalGPT):
    def __init__(self, cfg: SpanConfig):
        # Keep control parameter names and initialization identical to A0.
        super().__init__(cfg)
        if cfg.route_mode != 'control':
            for block in self.blocks[-cfg.route_layers:]:
                block.route = PromptRoute(cfg)
                block.route.apply(self._init_weights)
                nn.init.zeros_(block.route.gate.weight)
                nn.init.constant_(block.route.gate.bias, -2.)
                nn.init.zeros_(block.route.gamma)
            self.copy_gate = nn.Linear(cfg.d_model + cfg.route_dim + 2, 1)
            nn.init.zeros_(self.copy_gate.weight)
            nn.init.constant_(self.copy_gate.bias, -2.)

    def forward(self, input_ids, targets=None, prompt_length=None, prompt_roles=None,
                cache=None, use_cache=False, return_probs=True):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.numel():
            raise ValueError('Legal-span uses one variable-length document per microbatch')
        if cache is None:
            if not isinstance(prompt_length, int) or not 1 <= prompt_length <= input_ids.shape[1]:
                raise ValueError('Explicit observed prompt boundary required')
            prompt_ids, history = input_ids[:, :prompt_length], input_ids
            roles = prompt_roles
            if roles is None or roles.shape != prompt_ids.shape or roles.min() < 0 or roles.max() > 4:
                raise ValueError('Prompt-only role IDs 0..4 required')
            start = prompt_length - 1
            offset = 0
        else:
            if input_ids.shape[1] != 1 or len(cache.kvs) != self.cfg.n_layers:
                raise ValueError('Invalid cached query')
            prompt_ids, roles = cache.prompt_ids, cache.roles
            prompt_length = prompt_ids.shape[1]
            history = torch.cat((cache.history, input_ids), 1)
            start, offset = 0, cache.history.shape[1]
        if offset + input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError('Full prompt/context exceeds capacity; do not truncate')
        eligible = prompt_ids >= 4
        counts = continuity(history, prompt_ids, history.shape[1] - input_ids.shape[1] + start)
        x, kvs, memories = self.tok_embeddings(input_ids), [], []
        alpha = context = strength = None
        for i, block in enumerate(self.blocks):
            past = None if cache is None else cache.kvs[i]
            memory = None if cache is None else cache.memories[i]
            def apply_block(inp, blk=block, past=past, memory=memory):
                output, kv = blk.attn(blk.attn_norm(inp), past, use_cache)
                hidden = inp + output
                a = c = s = None
                if hasattr(blk, 'route'):
                    hidden, mem, a, c, s = blk.route(hidden, start, prompt_length, roles, eligible, counts, memory)
                else:
                    mem = None
                hidden = hidden + blk.ffn(blk.ffn_norm(hidden))
                return hidden, kv, mem, a, c, s
            if self.training and self.gradient_checkpointing and not use_cache and cache is None:
                def checked(inp, fn=apply_block):
                    hidden, _, _, a, c, s = fn(inp)
                    return (hidden, a, c, s) if a is not None else (hidden,)
                result = checkpoint(checked, x, use_reentrant=False)
                x = result[0]
                if len(result) > 1:
                    alpha, context, strength = result[1:]
                kv = mem = None
            else:
                x, kv, mem, a, c, s = apply_block(x)
                if a is not None:
                    alpha, context, strength = a, c, s
            if use_cache:
                kvs.append(kv); memories.append(mem)
        hidden = self.norm(x[:, start:])
        gate_logits = None
        if alpha is not None and self.cfg.use_copy:
            entropy = -(alpha * alpha.clamp_min(1e-30).log()).sum(-1, keepdim=True)
            features = torch.cat((hidden, context.to(hidden.dtype), entropy.to(hidden.dtype), strength.to(hidden.dtype)), -1)
            gate_logits = self.copy_gate(features).float()
        if targets is not None:
            if targets.shape != input_ids.shape or cache is not None or (targets[:, :start] != -100).any() or not (targets[:, start:] != -100).any():
                raise ValueError('Continuation-only aligned targets required')
        if not return_probs:
            if targets is None: raise ValueError('Gold-only loss requires aligned targets')
            labels = targets[:, start:]
            gold = []
            # Checkpoint each output-head chunk: retain O(S*D) activations rather
            # than O(S*V) vocabulary mixtures. This computes the same normalized
            # NLL; it does not replace the copy distribution with an auxiliary loss.
            for lo in range(0, hidden.shape[1], 256):
                target = labels[:, lo:lo+256].reshape(-1)
                def gold_vocab(h, target=target):
                    return -F.cross_entropy(self.lm_head(h).float().reshape(-1, self.cfg.vocab_size),
                                            target, ignore_index=-100, reduction='none')
                h = hidden[:, lo:lo+256]
                gold.append(checkpoint(gold_vocab, h, use_reentrant=False) if self.training else gold_vocab(h))
            log_gold = torch.cat(gold).view_as(labels)
            if gate_logits is not None:
                copy_gold = (alpha * (prompt_ids[:, None] == labels[:, :, None])).sum(-1)
                mix = torch.logaddexp(F.logsigmoid(-gate_logits.squeeze(-1)) + log_gold,
                                     F.logsigmoid(gate_logits.squeeze(-1)) + copy_gold.clamp_min(1e-30).log())
                log_gold = torch.where(eligible.any(-1)[:, None], mix, log_gold)
            loss = -log_gold[labels != -100].mean()
            return None, loss, None
        log_vocab = self.lm_head(hidden).float().log_softmax(-1)
        log_probs = log_vocab
        if gate_logits is not None:
            p_copy = copy_distribution(alpha, prompt_ids, self.cfg.vocab_size)
            log_mixture = torch.logaddexp(F.logsigmoid(-gate_logits) + log_vocab,
                                         F.logsigmoid(gate_logits) + p_copy.clamp_min(1e-30).log())
            log_probs = torch.where(eligible.any(-1)[:, None, None], log_mixture, log_vocab)
        loss = None
        if targets is not None:
            loss = F.nll_loss(log_probs.reshape(-1, self.cfg.vocab_size), targets[:, start:].reshape(-1), ignore_index=-100)
        result_cache = SpanCache(kvs, memories, prompt_ids, roles, history) if use_cache else None
        return log_probs, loss, result_cache

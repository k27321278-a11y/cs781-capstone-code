"""Bidirectional prompt encoder with a causal, cross-attending decoder (family B)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .a import RMSNorm, RotaryEmbedding, SwiGLU


@dataclass(frozen=True)
class EncoderDecoderConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_ff: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    pad_token_id: int = 0

    def validate(self):
        positive = (self.vocab_size, self.max_seq_len, self.n_encoder_layers,
                    self.n_decoder_layers, self.d_model, self.n_heads,
                    self.d_head, self.d_ff)
        if min(positive) < 1 or self.d_model != self.n_heads * self.d_head:
            raise ValueError("Invalid encoder-decoder dimensions")
        if (self.d_head % 2 or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("RoPE needs an even head width and dropout must be in [0,1)")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("RoPE theta must be finite and positive")
        if type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("Padding token is outside the vocabulary")


class EncoderSelfAttention(nn.Module):
    def __init__(self, cfg: EncoderDecoderConfig, rope: RotaryEmbedding):
        super().__init__()
        self.cfg, self.rope = cfg, rope
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, valid_mask):
        batch, length, width = x.shape
        q, k, v = [part.view(batch, length, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
                   for part in self.wqkv(x).chunk(3, dim=-1)]
        q, k = self.rope.apply_rotary(q), self.rope.apply_rotary(k)
        # SDPA boolean masks use True for keys that may participate.
        allowed = valid_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        return self.wo(out.transpose(1, 2).contiguous().view(batch, length, width))


class DecoderSelfAttention(nn.Module):
    def __init__(self, cfg: EncoderDecoderConfig, rope: RotaryEmbedding):
        super().__init__()
        self.cfg, self.rope = cfg, rope
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, past_kv=None, use_cache=False):
        batch, length, width = x.shape
        q, k, v = [part.view(batch, length, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
                   for part in self.wqkv(x).chunk(3, dim=-1)]
        offset = 0 if past_kv is None else past_kv[0].size(-2)
        q, k = self.rope.apply_rotary(q, offset), self.rope.apply_rotary(k, offset)
        if past_kv is not None:
            if length != 1:
                raise ValueError("Cached B decoding accepts one token at a time")
            k = torch.cat((past_kv[0], k), dim=-2)
            v = torch.cat((past_kv[1], v), dim=-2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=past_kv is None,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        out = self.wo(out.transpose(1, 2).contiguous().view(batch, length, width))
        return out, (k, v) if use_cache else None


class PromptCrossAttention(nn.Module):
    def __init__(self, cfg: EncoderDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def project_memory(self, memory):
        batch, length, _ = memory.shape
        shape = (batch, length, self.cfg.n_heads, self.cfg.d_head)
        key = self.wk(memory).view(shape).transpose(1, 2)
        value = self.wv(memory).view(shape).transpose(1, 2)
        return key, value

    def forward(self, x, valid_mask, memory=None, past_kv=None, use_cache=False):
        batch, length, width = x.shape
        query = self.wq(x).view(batch, length, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        if past_kv is None:
            if memory is None:
                raise ValueError("Prompt memory is required when cross-attention has no cache")
            key, value = self.project_memory(memory)
        else:
            key, value = past_kv
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=valid_mask[:, None, None, :],
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        out = self.wo(out.transpose(1, 2).contiguous().view(batch, length, width))
        return out, (key, value) if use_cache else None


class EncoderBlock(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attn = EncoderSelfAttention(cfg, rope)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, valid_mask):
        x = x + self.attn(self.attn_norm(x), valid_mask)
        return x + self.ffn(self.ffn_norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model)
        self.cross_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.self_attn = DecoderSelfAttention(cfg, rope)
        self.cross_attn = PromptCrossAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, memory, prompt_mask, self_past=None, cross_past=None, use_cache=False):
        update, self_cache = self.self_attn(self.self_norm(x), self_past, use_cache)
        x = x + update
        update, cross_cache = self.cross_attn(
            self.cross_norm(x), prompt_mask, memory, cross_past, use_cache,
        )
        x = x + update
        x = x + self.ffn(self.ffn_norm(x))
        return x, (self_cache, cross_cache) if use_cache else None


class EncoderDecoderLegalModel(nn.Module):
    """Encode prompt IDs bidirectionally; predict shifted suffix IDs autoregressively."""

    def __init__(self, cfg: EncoderDecoderConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder_rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_theta)
        self.decoder_rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_theta)
        self.encoder = nn.ModuleList([
            EncoderBlock(cfg, self.encoder_rope) for _ in range(cfg.n_encoder_layers)
        ])
        self.encoder_norm = RMSNorm(cfg.d_model)
        self.decoder = nn.ModuleList([
            DecoderBlock(cfg, self.decoder_rope) for _ in range(cfg.n_decoder_layers)
        ])
        self.decoder_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        encoder_residual_std = 0.02 / math.sqrt(2 * cfg.n_encoder_layers)
        for block in self.encoder:
            nn.init.normal_(block.attn.wo.weight, std=encoder_residual_std)
            nn.init.normal_(block.ffn.w2.weight, std=encoder_residual_std)
        decoder_residual_std = 0.02 / math.sqrt(3 * cfg.n_decoder_layers)
        for block in self.decoder:
            nn.init.normal_(block.self_attn.wo.weight, std=decoder_residual_std)
            nn.init.normal_(block.cross_attn.wo.weight, std=decoder_residual_std)
            nn.init.normal_(block.ffn.w2.weight, std=decoder_residual_std)
        self.lm_head.weight = self.tok_embeddings.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def encode(self, prompt_ids, prompt_mask=None):
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1:
            raise ValueError("Expected nonempty [batch,prompt] token IDs")
        if prompt_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("Prompt exceeds encoder context")
        if prompt_mask is None:
            prompt_mask = prompt_ids.ne(self.cfg.pad_token_id)
        if prompt_mask.shape != prompt_ids.shape or prompt_mask.dtype != torch.bool:
            raise ValueError("Prompt mask must be boolean and match prompt IDs")
        if not prompt_mask.any(dim=1).all():
            raise ValueError("Every prompt must contain at least one valid token")
        x = self.tok_embeddings(prompt_ids)
        for block in self.encoder:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(lambda inp, blk=block: blk(inp, prompt_mask), x, use_reentrant=False)
            else:
                x = block(x, prompt_mask)
        return self.encoder_norm(x), prompt_mask

    def forward(self, prompt_ids, decoder_ids, targets=None, prompt_mask=None,
                past_kvs=None, use_cache=False):
        if decoder_ids.ndim != 2 or decoder_ids.shape[1] < 1:
            raise ValueError("Expected nonempty [batch,decoder] token IDs")
        if decoder_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("Decoder sequence exceeds decoder context")
        if past_kvs is None:
            if prompt_ids is None:
                raise ValueError("Prompt IDs are required before creating a cache")
            memory, prompt_mask = self.encode(prompt_ids, prompt_mask)
            self_pasts = cross_pasts = [None] * len(self.decoder)
        else:
            if not use_cache:
                raise ValueError("Cached decoding must request the updated cache")
            if prompt_ids is not None or prompt_mask is not None:
                raise ValueError("Cached decoding reuses prompt memory; omit prompt IDs/mask")
            if not isinstance(past_kvs, dict) or set(past_kvs) != {"self", "cross", "prompt_mask"}:
                raise ValueError("Malformed encoder-decoder cache")
            self_pasts, cross_pasts = past_kvs["self"], past_kvs["cross"]
            prompt_mask = past_kvs["prompt_mask"]
            if len(self_pasts) != len(self.decoder) or len(cross_pasts) != len(self.decoder):
                raise ValueError("Cache layer count differs from decoder")
            if decoder_ids.shape[0] != prompt_mask.shape[0] or decoder_ids.shape[1] != 1:
                raise ValueError("Cached decoding requires a matching batch and one new token")
            memory = None

        x = self.tok_embeddings(decoder_ids)
        new_self, new_cross = [], []
        for i, block in enumerate(self.decoder):
            if past_kvs is None and self.training and self.cfg.gradient_checkpointing and not use_cache:
                x = checkpoint(lambda inp, mem, blk=block: blk(inp, mem, prompt_mask)[0],
                               x, memory, use_reentrant=False)
                caches = None
            else:
                x, caches = block(
                    x, memory, prompt_mask, self_pasts[i], cross_pasts[i], use_cache,
                )
            if use_cache:
                new_self.append(caches[0])
                new_cross.append(caches[1])

        logits = self.lm_head(self.decoder_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != decoder_ids.shape or not (targets != -100).any():
                raise ValueError("Targets must match decoder IDs and include supervised tokens")
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1), ignore_index=-100,
            )
        cache = None
        if use_cache:
            cache = {"self": new_self, "cross": new_cross, "prompt_mask": prompt_mask}
        return logits, loss, cache

    def count_parameters(self):
        return sum(parameter.numel() for parameter in {id(p): p for p in self.parameters()}.values())

"""Approved C: bidirectional Transformer prompt encoder and recurrent generator.

The GRU never reads attended context or future suffix tokens. Fixed prompt memory
is read after each recurrent output, permitting equivalent parallel teacher
forcing and single-token recurrent inference. All participant weights are new.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .a import RMSNorm, RotaryEmbedding, SwiGLU
from .b import EncoderBlock, PromptCrossAttention


@dataclass(frozen=True)
class RecurrentConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    max_prompt_len: int = 8192
    n_encoder_layers: int = 10
    n_gru_layers: int = 4
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_ff: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    pad_token_id: int = 0

    def validate(self):
        dimensions = (self.vocab_size, self.max_seq_len, self.max_prompt_len,
                      self.n_encoder_layers, self.n_gru_layers, self.d_model,
                      self.n_heads, self.d_head, self.d_ff)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError('C dimensions must be positive integers')
        if self.d_model != self.n_heads * self.d_head or self.d_head % 2:
            raise ValueError('C head dimensions must match width and support RoPE')
        if self.max_prompt_len < self.max_seq_len:
            raise ValueError('C prompt capacity must cover training context')
        if (not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or not math.isfinite(self.rope_theta) or self.rope_theta <= 0):
            raise ValueError('Invalid C dropout or RoPE theta')
        if type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError('C padding ID is outside vocabulary')


class RecurrentLegalModel(nn.Module):
    def __init__(self, cfg: RecurrentConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder_rope = RotaryEmbedding(cfg.d_head, cfg.max_prompt_len, cfg.rope_theta)
        self.encoder = nn.ModuleList([EncoderBlock(cfg, self.encoder_rope)
                                      for _ in range(cfg.n_encoder_layers)])
        self.encoder_norm = RMSNorm(cfg.d_model)
        self.initial_projection = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.gru = nn.GRU(cfg.d_model, cfg.d_model, cfg.n_gru_layers, batch_first=True,
                         dropout=cfg.dropout if cfg.n_gru_layers > 1 else 0.0)
        self.cross_norm = RMSNorm(cfg.d_model)
        self.cross_attn = PromptCrossAttention(cfg)
        self.fusion_norm = RMSNorm(cfg.d_model)
        self.fusion = SwiGLU(cfg)
        self.output_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_encoder_layers)
        for block in self.encoder:
            nn.init.normal_(block.attn.wo.weight, std=residual_std)
            nn.init.normal_(block.ffn.w2.weight, std=residual_std)
        self.lm_head.weight = self.tok_embeddings.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        # nn.GRU retains its standard uniform initialisation for recurrent gates.

    def encode(self, prompt_ids, prompt_mask=None):
        if prompt_ids.ndim != 2 or min(prompt_ids.shape) < 1:
            raise ValueError('Expected nonempty [batch,prompt] IDs')
        if prompt_ids.size(1) > self.cfg.max_prompt_len:
            raise ValueError('Prompt exceeds encoder context; do not truncate')
        if prompt_mask is None:
            prompt_mask = prompt_ids.ne(self.cfg.pad_token_id)
        if prompt_mask.shape != prompt_ids.shape or prompt_mask.dtype != torch.bool:
            raise ValueError('Prompt mask must be boolean and match IDs')
        if prompt_mask.device != prompt_ids.device or not prompt_mask.any(dim=1).all():
            raise ValueError('Every prompt must contain a valid token on the input device')
        x = self.tok_embeddings(prompt_ids)
        for block in self.encoder:
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(lambda inp, blk=block: blk(inp, prompt_mask), x,
                               use_reentrant=False)
            else:
                x = block(x, prompt_mask)
        return self.encoder_norm(x), prompt_mask

    def _validate_cache(self, cache, decoder_ids):
        if not isinstance(cache, dict) or set(cache) != {'hidden', 'cross', 'prompt_mask', 'steps'}:
            raise ValueError('Malformed C recurrent cache')
        hidden, cross, mask, steps = (cache[key] for key in
                                      ('hidden', 'cross', 'prompt_mask', 'steps'))
        batch = decoder_ids.size(0)
        if decoder_ids.size(1) != 1:
            raise ValueError('Cached C decoding requires one token')
        if (not isinstance(hidden, torch.Tensor)
                or hidden.shape != (self.cfg.n_gru_layers, batch, self.cfg.d_model)
                or hidden.dtype != torch.float32 or hidden.device != decoder_ids.device):
            raise ValueError('Invalid C hidden cache shape/device/dtype')
        if (not isinstance(mask, torch.Tensor) or mask.ndim != 2
                or mask.size(0) != batch or mask.dtype != torch.bool
                or mask.device != decoder_ids.device or not mask.any(dim=1).all()):
            raise ValueError('Invalid C prompt cache mask')
        expected = (batch, self.cfg.n_heads, mask.size(1), self.cfg.d_head)
        if (not isinstance(cross, (tuple, list)) or len(cross) != 2
                or any(not isinstance(x, torch.Tensor) or x.shape != expected
                       or x.device != decoder_ids.device for x in cross)
                or cross[0].dtype != cross[1].dtype):
            raise ValueError('Invalid C cross-attention cache')
        if type(steps) is not int or steps < 1:
            raise ValueError('Invalid C cache position')
        if steps + decoder_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError('Cached sequence exceeds decoder context')
        return hidden, cross, mask, steps

    def forward(self, prompt_ids, decoder_ids, targets=None, prompt_mask=None,
                past_kvs=None, use_cache=False):
        if decoder_ids.ndim != 2 or min(decoder_ids.shape) < 1:
            raise ValueError('Expected nonempty [batch,decoder] IDs')
        if decoder_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError('Sequence exceeds decoder context')
        if past_kvs is None:
            if prompt_ids is None or prompt_ids.size(0) != decoder_ids.size(0):
                raise ValueError('Matching prompt IDs required to initialise recurrence')
            memory, prompt_mask = self.encode(prompt_ids, prompt_mask)
            # Padded encoder states never affect recurrent initialisation.
            valid = prompt_mask.unsqueeze(-1).to(memory.dtype)
            mean = (memory * valid).sum(1) / valid.sum(1)
            initial = torch.tanh(self.initial_projection(mean)).float()
            hidden = initial.unsqueeze(0).expand(self.cfg.n_gru_layers, -1, -1).contiguous()
            cross, steps = None, 0
        else:
            if not use_cache or prompt_ids is not None or prompt_mask is not None:
                raise ValueError('Cached C decoding reuses prompt state and requires an updated cache')
            hidden, cross, prompt_mask, steps = self._validate_cache(past_kvs, decoder_ids)
            memory = None

        embeddings = self.tok_embeddings(decoder_ids)
        # Installed cuDNN GRU uses FP32 weights, inputs and hidden state. Attention
        # and FFN remain inside the caller's BF16 autocast context. Casts preserve
        # gradient flow from recurrent initialisation into encoder states.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            recurrent, new_hidden = self.gru(embeddings.float(), hidden.float())
        update, new_cross = self.cross_attn(self.cross_norm(recurrent), prompt_mask,
                                           memory=memory, past_kv=cross, use_cache=use_cache)
        x = recurrent + update
        x = x + self.fusion(self.fusion_norm(x))
        logits = self.lm_head(self.output_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != decoder_ids.shape or not targets.ne(-100).any():
                raise ValueError('C targets must match decoder IDs and include supervised tokens')
            loss = F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1), ignore_index=-100)
        cache = ({'hidden': new_hidden, 'cross': new_cross, 'prompt_mask': prompt_mask,
                  'steps': steps + decoder_ids.size(1)} if use_cache else None)
        return logits, loss, cache

    def count_parameters(self):
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

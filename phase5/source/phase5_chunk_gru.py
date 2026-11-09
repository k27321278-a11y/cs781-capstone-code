"""Six-window B1 encoder with a gated BiGRU and statute-wise attention head."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import phase5_b1_train as base


WINDOWS, TOKENS, CHARS = 6, 128, 650


def pack_six_windows(text, tokenizer):
    if not isinstance(text, str) or not text:
        raise ValueError("Expected nonempty legal text")
    last = max(0, len(text) - CHARS)
    starts = [int(last * i / (WINDOWS - 1)) for i in range(WINDOWS)]
    ids = np.zeros((WINDOWS, TOKENS), dtype=np.int32)
    masks = np.zeros((WINDOWS, TOKENS), dtype=np.bool_)
    for i, start in enumerate(starts):
        tokens = tokenizer.encode(text[start:start + CHARS]).ids[:TOKENS] or [3]
        ids[i, :len(tokens)] = tokens
        masks[i, :len(tokens)] = True
    return ids, masks


class ChunkGRUB1Classifier(base.B1Classifier):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        for block in self.backbone.encoder:
            block.attn.wqkv = block.attn.wqkv.base
            block.attn.wo = block.attn.wo.base
            for name in ("w1", "w2", "w3"):
                setattr(block.ffn, name, getattr(block.ffn, name).base)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone.encoder[-1].parameters():
            parameter.requires_grad_(True)
        width = self.backbone.cfg.d_model
        self.head = nn.Identity()
        self.projection = nn.Linear(2 * width, 384)
        self.chunk_position = nn.Parameter(torch.zeros(WINDOWS, 384))
        self.gru = nn.GRU(384, 256, batch_first=True, bidirectional=True)
        self.label_query = nn.Parameter(torch.randn(base.LABELS, 512) * .02)
        self.label_weight = nn.Parameter(torch.randn(base.LABELS, 512) * .02)
        self.label_bias = nn.Parameter(torch.zeros(base.LABELS))
        self.global_head = nn.Linear(512, base.LABELS)

    def forward(self, ids, masks):
        batch, windows, width = ids.shape
        if windows != WINDOWS or width != TOKENS:
            raise ValueError("Expected six 128-token document windows")
        encoded, _ = self.backbone.encode(ids.reshape(batch * windows, width), masks.reshape(batch * windows, width))
        valid = masks.reshape(batch * windows, width)
        mean = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        maximum = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min).max(1).values
        chunk = torch.cat((mean, maximum), dim=-1).reshape(batch, windows, -1)
        chunk = F.gelu(self.projection(self.dropout(chunk))) + self.chunk_position
        contextual, _ = self.gru(chunk)
        scores = torch.einsum("bwd,ld->bwl", contextual, self.label_query)
        weights = torch.softmax(scores, dim=1)
        evidence = torch.einsum("bwl,bwd->bld", weights, contextual)
        label_logits = (evidence * self.label_weight).sum(-1) + self.label_bias
        return label_logits + .5 * self.global_head(contextual.mean(1))


def tokenize_split_six(name, rows, tokenizer, cache):
    cache = Path(cache)
    ids_path, mask_path = cache / f"{name}_ids.npy", cache / f"{name}_masks.npy"
    if ids_path.exists() and mask_path.exists():
        ids, masks = np.load(ids_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")
        if ids.shape == (len(rows), WINDOWS, TOKENS) and masks.shape == ids.shape:
            return ids, masks
    ids = np.zeros((len(rows), WINDOWS, TOKENS), dtype=np.int32)
    masks = np.zeros_like(ids, dtype=np.bool_)
    for i, row in enumerate(rows):
        ids[i], masks[i] = pack_six_windows(row["text"], tokenizer)
        if i % 2000 == 0:
            print(json.dumps({"six_window_tokenizing": name, "done": i, "total": len(rows)}), flush=True)
    np.save(ids_path, ids)
    np.save(mask_path, masks)
    return ids, masks


def install():
    base.B1Classifier = ChunkGRUB1Classifier
    base.tokenize_split = tokenize_split_six

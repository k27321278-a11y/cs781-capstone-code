"""Four-window B1 classifier with bottleneck adapters and label-wise evidence pooling.

The Phase 1 model weights remain frozen. Four distributed text windows replace
the previous opening/closing pair; each statute receives evidence from the
windows via a smooth max of its per-window logits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import phase5_b1_train as base


WINDOWS = 4
TOKENS = 192
CHARS = 900


def pack_four_windows(text, tokenizer, window_tokens=TOKENS, window_chars=CHARS):
    if not isinstance(text, str) or not text or min(window_tokens, window_chars) < 1:
        raise ValueError("Expected nonempty text and positive window sizes")
    max_start = max(0, len(text) - window_chars)
    starts = [int(max_start * fraction / 3) for fraction in range(4)]
    ids = np.zeros((4, window_tokens), dtype=np.int32)
    masks = np.zeros((4, window_tokens), dtype=np.bool_)
    for index, start in enumerate(starts):
        piece = text[start:start + window_chars]
        tokens = tokenizer.encode(piece).ids[:window_tokens] or [3]
        ids[index, :len(tokens)] = tokens
        masks[index, :len(tokens)] = True
    return ids, masks


class AdapterBlock(nn.Module):
    def __init__(self, original_block, width=768, bottleneck=96):
        super().__init__()
        self.original = original_block
        self.down = nn.Linear(width, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, valid_mask):
        hidden = self.original(x, valid_mask)
        return hidden + self.up(F.gelu(self.down(hidden)))


class MultiWindowB1Classifier(base.B1Classifier):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        for block in self.backbone.encoder:
            block.attn.wqkv = block.attn.wqkv.base
            block.attn.wo = block.attn.wo.base
            for name in ("w1", "w2", "w3"):
                setattr(block.ffn, name, getattr(block.ffn, name).base)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        width = self.backbone.cfg.d_model
        self.backbone.encoder = nn.ModuleList([AdapterBlock(block, width) for block in self.backbone.encoder])
        self.head = nn.Linear(2 * width, base.LABELS)
        self.document_head = nn.Linear(2 * width, base.LABELS)

    def forward(self, ids, masks):
        batch, windows, width = ids.shape
        if windows != WINDOWS or width != TOKENS:
            raise ValueError("Expected four 192-token windows")
        encoded, _ = self.backbone.encode(ids.reshape(batch * windows, width), masks.reshape(batch * windows, width))
        valid = masks.reshape(batch * windows, width)
        mean = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        maximum = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min).max(1).values
        evidence = torch.cat((mean, maximum), dim=-1).reshape(batch, windows, -1)
        evidence = self.dropout(evidence)
        label_window_logits = self.head(evidence)
        label_evidence = torch.logsumexp(label_window_logits, dim=1) - math.log(windows)
        document_logits = self.document_head(evidence.mean(dim=1))
        return .65 * label_evidence + .35 * document_logits


def tokenize_split_four(name, rows, tokenizer, cache):
    cache = Path(cache)
    ids_path = cache / f"{name}_ids.npy"
    mask_path = cache / f"{name}_masks.npy"
    if ids_path.exists() and mask_path.exists():
        ids = np.load(ids_path, mmap_mode="r")
        masks = np.load(mask_path, mmap_mode="r")
        if ids.shape == (len(rows), WINDOWS, TOKENS) and masks.shape == ids.shape:
            return ids, masks
    ids = np.zeros((len(rows), WINDOWS, TOKENS), dtype=np.int32)
    masks = np.zeros_like(ids, dtype=np.bool_)
    for index, row in enumerate(rows):
        ids[index], masks[index] = pack_four_windows(row["text"], tokenizer)
        if index % 2000 == 0:
            print(json.dumps({"four_window_tokenizing": name, "done": index, "total": len(rows)}), flush=True)
    np.save(ids_path, ids)
    np.save(mask_path, masks)
    return ids, masks


def install():
    base.B1Classifier = MultiWindowB1Classifier
    base.tokenize_split = tokenize_split_four

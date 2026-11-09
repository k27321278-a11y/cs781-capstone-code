"""Four-window Phase 1 B1 encoder with a separate gated memory per statute."""
import math

import torch
from torch import nn

import phase5_b1_train as base
import phase5_multiwindow as multi


class StatuteMemoryB1Classifier(multi.MultiWindowB1Classifier):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        width = self.backbone.cfg.d_model
        self.head = nn.Identity()
        self.document_head = nn.Identity()
        self.evidence_proj = nn.Linear(2 * width, 96)
        self.statute_code = nn.Parameter(torch.randn(base.LABELS, 96) * .02)
        self.memory = nn.GRUCell(96, 96)
        self.statute_weight = nn.Parameter(torch.randn(base.LABELS, 96) * .02)
        self.statute_bias = nn.Parameter(torch.zeros(base.LABELS))
        self.global_head = nn.Linear(2 * width, base.LABELS)

    def forward(self, ids, masks):
        batch, windows, width = ids.shape
        if windows != 4 or width != 192:
            raise ValueError("Expected four 192-token windows")
        encoded, _ = self.backbone.encode(ids.reshape(batch * windows, width), masks.reshape(batch * windows, width))
        valid = masks.reshape(batch * windows, width)
        means = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        maxima = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min).max(1).values
        evidence = self.dropout(torch.cat((means, maxima), dim=-1).reshape(batch, windows, -1))
        features = torch.tanh(self.evidence_proj(evidence))
        memory = features.new_zeros(batch * base.LABELS, 96)
        for t in range(windows):
            statute_input = features[:, t, None, :] + self.statute_code[None, :, :]
            memory = self.memory(statute_input.reshape(batch * base.LABELS, 96), memory)
        memory = memory.reshape(batch, base.LABELS, 96)
        label_logits = (memory * self.statute_weight).sum(-1) / math.sqrt(96) + self.statute_bias
        return label_logits + .25 * self.global_head(evidence.mean(1))


def install():
    base.B1Classifier = StatuteMemoryB1Classifier
    base.tokenize_split = multi.tokenize_split_four

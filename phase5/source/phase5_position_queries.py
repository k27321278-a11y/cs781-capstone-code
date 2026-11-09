"""Phase 1 B1 chunks read by position-aware statute queries."""
import math

import torch
from torch import nn

import phase5_b1_train as base
import phase5_multiwindow as multi


class PositionQueryB1Classifier(multi.MultiWindowB1Classifier):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        width = self.backbone.cfg.d_model
        query_width = 128
        self.head = nn.Identity()
        self.document_head = nn.Identity()
        self.key = nn.Linear(2 * width, query_width)
        self.value = nn.Linear(2 * width, query_width)
        self.window_position = nn.Parameter(torch.randn(4, query_width) * .02)
        self.statute_query = nn.Parameter(torch.randn(base.LABELS, query_width) * .02)
        self.statute_weight = nn.Parameter(torch.randn(base.LABELS, query_width) * .02)
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
        keys = self.key(evidence) + self.window_position
        values = self.value(evidence)
        scores = torch.einsum("ld,bwd->blw", self.statute_query, keys) / math.sqrt(keys.shape[-1])
        weights = scores.softmax(-1)
        context = torch.einsum("blw,bwd->bld", weights, values)
        logits = (context * self.statute_weight).sum(-1) / math.sqrt(context.shape[-1]) + self.statute_bias
        return logits + .25 * self.global_head(evidence.mean(1))


def install():
    base.B1Classifier = PositionQueryB1Classifier
    base.tokenize_split = multi.tokenize_split_four

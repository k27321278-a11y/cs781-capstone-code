"""Phase 1 B1 statute queries with a train-only co-occurrence graph."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import phase5_b1_train as base
import phase5_multiwindow as multi


GRAPH = None


def build_fit_graph(input_dir):
    inp = Path(input_dir)
    train = json.loads((inp / "train.json").read_text(encoding="utf-8"))
    manifest = json.loads((inp / "phase5_split_manifest.json").read_text(encoding="utf-8"))
    fit_ids = set(str(key) for key in manifest["fit_ids"])
    supports = np.zeros(base.LABELS, dtype=np.float64)
    pairs = np.zeros((base.LABELS, base.LABELS), dtype=np.float64)
    for row in train:
        if str(row["id"]) not in fit_ids:
            continue
        labels = sorted(set(row["labels"]))
        supports[labels] += 1
        pairs[np.ix_(labels, labels)] += 1
    np.fill_diagonal(pairs, 0)
    strength = pairs / np.sqrt(np.maximum(supports[:, None] * supports[None, :], 1))
    graph = np.zeros_like(strength)
    for label in range(base.LABELS):
        neighbors = np.argsort(strength[label])[-8:]
        graph[label, neighbors] = strength[label, neighbors]
    graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-12)
    return torch.tensor(graph, dtype=torch.float32)


class LabelGraphB1Classifier(multi.MultiWindowB1Classifier):
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
        self.graph_gate = nn.Linear(2 * query_width, query_width)
        self.statute_weight = nn.Parameter(torch.randn(base.LABELS, query_width) * .02)
        self.statute_bias = nn.Parameter(torch.zeros(base.LABELS))
        self.global_head = nn.Linear(2 * width, base.LABELS)
        self.register_buffer("graph", GRAPH.clone())

    def forward(self, ids, masks):
        batch, windows, width = ids.shape
        if windows != 4 or width != 192:
            raise ValueError("Expected four 192-token windows")
        encoded, _ = self.backbone.encode(ids.reshape(batch * windows, width), masks.reshape(batch * windows, width))
        valid = masks.reshape(batch * windows, width)
        means = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        maxima = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min).max(1).values
        evidence = self.dropout(torch.cat((means, maxima), dim=-1).reshape(batch, windows, -1))
        neighbors = self.graph @ self.statute_query
        gate = torch.sigmoid(self.graph_gate(torch.cat((self.statute_query, neighbors), -1)))
        queries = self.statute_query + gate * neighbors
        keys = self.key(evidence) + self.window_position
        values = self.value(evidence)
        scores = torch.einsum("ld,bwd->blw", queries, keys) / math.sqrt(keys.shape[-1])
        weights = scores.softmax(-1)
        context = torch.einsum("blw,bwd->bld", weights, values)
        logits = (context * self.statute_weight).sum(-1) / math.sqrt(context.shape[-1]) + self.statute_bias
        return logits + .25 * self.global_head(evidence.mean(1))


def install(input_dir):
    global GRAPH
    GRAPH = build_fit_graph(input_dir)
    base.B1Classifier = LabelGraphB1Classifier
    base.tokenize_split = multi.tokenize_split_four

from dataclasses import asdict
import json

import torch

from .contracts import Batch, ContinuationExample


def collate_a(examples, pad_id, bos_id, eos_id, max_seq_len):
    if not examples:
        raise ValueError("Empty batch")
    rows, labels = [], []
    for e in examples:
        if not e.prefix_ids or not e.suffix_ids:
            raise ValueError("Empty prefix/suffix")
        ids = [bos_id] + e.prefix_ids + e.suffix_ids + [eos_id]
        if len(ids) - 1 > max_seq_len:
            raise ValueError("Example exceeds context; do not truncate")
        rows.append(ids[:-1])
        targets = ids[1:]
        targets[:len(e.prefix_ids)] = [-100] * len(e.prefix_ids)
        labels.append(targets)
    size = max(map(len, rows))
    x = torch.full((len(rows), size), pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (row, target) in enumerate(zip(rows, labels)):
        x[i, :len(row)] = torch.tensor(row)
        y[i, :len(target)] = torch.tensor(target)
    return Batch(x, y, sum(map(len, rows)), int((y != -100).sum()))


def iter_examples(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            yield ContinuationExample(**json.loads(line))

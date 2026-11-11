"""Small, deterministic utilities for CS781 Phase 5 LSI."""

import numpy as np


def pack_windows(text, tokenizer, window_tokens=384, window_chars=1600):
    """Encode a document's opening and closing spans into two fixed token windows."""
    if not isinstance(text, str) or not text:
        raise ValueError("Expected nonempty text")
    if min(window_tokens, window_chars) < 1:
        raise ValueError("Window sizes must be positive")
    first = tokenizer.encode(text[:window_chars]).ids[:window_tokens]
    last = tokenizer.encode(text[-window_chars:]).ids[-window_tokens:]
    rows = np.zeros((2, window_tokens), dtype=np.int32)
    mask = np.zeros((2, window_tokens), dtype=np.bool_)
    for i, tokens in enumerate((first, last)):
        if not tokens:
            tokens = [3]
        rows[i, :len(tokens)] = tokens
        mask[i, :len(tokens)] = True
    return rows, mask


def f1_scores(y, pred):
    y = np.asarray(y, dtype=np.bool_)
    pred = np.asarray(pred, dtype=np.bool_)
    if y.shape != pred.shape or y.ndim != 2:
        raise ValueError("Expected matching [examples, labels] arrays")
    tp = np.sum(y & pred, axis=0, dtype=np.int64)
    fp = np.sum(~y & pred, axis=0, dtype=np.int64)
    fn = np.sum(y & ~pred, axis=0, dtype=np.int64)
    denominator = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator > 0)
    micro_den = 2 * tp.sum() + fp.sum() + fn.sum()
    return {
        "macro_f1": float(per_class.mean()),
        "micro_f1": float(2 * tp.sum() / micro_den) if micro_den else 0.0,
        "subset_accuracy": float(np.all(y == pred, axis=1).mean()),
        "support": np.sum(y, axis=0, dtype=np.int64).tolist(),
        "per_label_f1": per_class.tolist(),
    }


def choose_thresholds(y, probs):
    """Fit a global F1 threshold, then class thresholds shrunk by support."""
    y = np.asarray(y, dtype=np.bool_)
    probs = np.asarray(probs, dtype=np.float64)
    if y.shape != probs.shape or y.ndim != 2 or not np.isfinite(probs).all():
        raise ValueError("Expected finite matching [examples, labels] arrays")
    grid = np.concatenate([np.arange(.01, .201, .01), np.arange(.225, .901, .025)])
    global_scores = [f1_scores(y, probs >= t)["macro_f1"] for t in grid]
    global_t = float(grid[int(np.argmax(global_scores))])
    thresholds = np.full(y.shape[1], global_t, dtype=np.float32)
    for j in range(y.shape[1]):
        support = int(y[:, j].sum())
        if support < 10:
            continue
        tp = np.sum((probs[:, j, None] >= grid) & y[:, j, None], axis=0)
        pp = np.sum(probs[:, j, None] >= grid, axis=0)
        scores = np.divide(2 * tp, pp + support, out=np.zeros_like(grid), where=(pp + support) > 0)
        local_t = float(grid[int(np.argmax(scores))])
        weight = support / (support + 40)
        thresholds[j] = np.clip(weight * local_t + (1 - weight) * global_t, .01, .9)
    return thresholds

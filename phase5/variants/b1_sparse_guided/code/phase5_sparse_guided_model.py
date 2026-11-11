"""Use fit-only lexical evidence to select B1 windows without labels at inference."""
import json
from pathlib import Path

import numpy as np

import phase5_b1_train as base
import phase5_multiwindow as multi


def install(guided_dir):
    guided_dir = Path(guided_dir)
    starts_by_split = {
        name: json.loads((guided_dir / f"{name}_starts.json").read_text(encoding="utf-8"))
        for name in ("train", "dev", "test")
    }

    def tokenize_split(name, rows, tokenizer, cache):
        cache = Path(cache)
        ids_path, masks_path = cache / f"{name}_ids.npy", cache / f"{name}_masks.npy"
        if ids_path.exists() and masks_path.exists():
            ids, masks = np.load(ids_path, mmap_mode="r"), np.load(masks_path, mmap_mode="r")
            if ids.shape == (len(rows), 4, 192) and masks.shape == ids.shape:
                return ids, masks
        starts_map = starts_by_split[name]
        if len(starts_map) != len(rows):
            raise ValueError("Window plan does not match official split")
        ids = np.zeros((len(rows), 4, 192), dtype=np.int32)
        masks = np.zeros_like(ids, dtype=np.bool_)
        for i, row in enumerate(rows):
            text = row["text"]
            starts = starts_map[str(row["id"])]
            if len(starts) != 4 or any(type(s) is not int or s < 0 or s > max(0, len(text) - 900) for s in starts):
                raise ValueError("Invalid fit-only selected window starts")
            for j, start in enumerate(starts):
                tokens = tokenizer.encode(text[start:start + 900]).ids[:192] or [3]
                ids[i, j, :len(tokens)] = tokens
                masks[i, j, :len(tokens)] = True
            if i % 2000 == 0:
                print(json.dumps({"guided_tokenizing": name, "done": i, "total": len(rows)}), flush=True)
        np.save(ids_path, ids)
        np.save(masks_path, masks)
        return ids, masks

    base.B1Classifier = multi.MultiWindowB1Classifier
    base.tokenize_split = tokenize_split

"""Choose four B1 text windows using fit-only sparse lexical evidence."""
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "phase5_data"
OUT = ROOT / "work" / "phase5_sparse_guided"
CHARS = 900


def select_batch(rows, vectorizer, feature_scores):
    candidates, owners, starts_by_doc = [], [], []
    for i, row in enumerate(rows):
        text = row["text"]
        end = max(0, len(text) - CHARS)
        starts = sorted(set(int(end * j / 11) for j in range(1, 11)))
        starts_by_doc.append((end, starts))
        for start in starts:
            owners.append((i, start))
            candidates.append(text[start:start + CHARS])
    scores = np.asarray(vectorizer.transform(candidates) @ feature_scores).reshape(-1)
    ranked = [[] for _ in rows]
    for (i, start), score in zip(owners, scores):
        ranked[i].append((float(score), start))
    selected = []
    for (end, starts), values in zip(starts_by_doc, ranked):
        middle = []
        for _, start in sorted(values, reverse=True):
            if min(abs(start - x) for x in [0, end, *middle]) >= CHARS // 2:
                middle.append(start)
            if len(middle) == 2:
                break
        for start in [int(end / 3), int(2 * end / 3)]:
            if len(middle) < 2 and start not in middle:
                middle.append(start)
        while len(middle) < 2:
            middle.append(0)
        selected.append([0, *sorted(middle[:2]), end])
    return selected


def main():
    OUT.mkdir(exist_ok=True)
    with (ROOT / "work" / "agent_sparse" / "logistic_model.pkl").open("rb") as stream:
        artifact = pickle.load(stream)
    vectorizer = artifact["vectorizer"]
    weights = np.stack([est.coef_[0] for est in artifact["model"].estimators_]).astype(np.float32)
    feature_scores = np.maximum(weights.max(axis=0), 0)
    for split in ("train", "dev", "test"):
        rows = json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))
        selected = {}
        for offset in range(0, len(rows), 500):
            batch = rows[offset:offset + 500]
            for row, starts in zip(batch, select_batch(batch, vectorizer, feature_scores)):
                selected[str(row["id"])] = starts
            if offset % 5000 == 0:
                print(split, offset, len(rows), flush=True)
        assert len(selected) == len(rows)
        (OUT / f"{split}_starts.json").write_text(json.dumps(selected, separators=(",", ":")), encoding="utf-8")
        print(split, "complete", len(selected), flush=True)


if __name__ == "__main__":
    main()

"""Create official-data-only sparse probabilities for a B1/lexical hybrid."""
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "phase5_data"
OUT = ROOT / "work" / "phase5_hybrid"


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "outputs" / "phase5_split_manifest.json").read_text(encoding="utf-8"))
    train = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in train}
    selected = {
        "calibration": [index[str(key)] for key in manifest["calibration_ids"]],
        "dev": dev,
        "test": test,
    }
    with (ROOT / "work" / "agent_sparse" / "logistic_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    for name, rows in selected.items():
        texts = [row["text"] for row in rows]
        fresh = model["model"].predict_proba(model["vectorizer"].transform(texts)).astype(np.float32)
        if model["candidate"] == "blend_75":
            prior = model["baseline"].predict_proba(model["baseline_vectorizer"].transform(texts)).astype(np.float32)
            probs = .75 * fresh + .25 * prior
        else:
            raise ValueError("Unexpected sparse model selection")
        assert probs.shape == (len(rows), 100) and np.isfinite(probs).all()
        np.save(OUT / f"sparse_{name}_probabilities.npy", probs)
        (OUT / f"{name}_ids.json").write_text(json.dumps([str(row["id"]) for row in rows]))
        print(name, probs.shape, float(probs.mean()), flush=True)


if __name__ == "__main__":
    main()

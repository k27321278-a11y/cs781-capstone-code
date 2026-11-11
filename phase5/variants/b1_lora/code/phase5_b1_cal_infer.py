"""Infer train-calibration probabilities from the finished Phase 1-adapted B1 LoRA."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import phase5_b1_train as b1
from cs781.own_bpe import OwnBPE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(input_dir, output_dir, adapter_file):
    inp, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = inp / "b1_latest_model.pt"
    adapter_path = Path(adapter_file)
    assert digest(checkpoint) == "111296fb36a23dac1a11c07ede927cfc89dc86d51253e441d59afb82bd6924eb"
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    assert adapter["phase1_checkpoint_sha256"] == digest(checkpoint)
    manifest = json.loads((inp / "phase5_split_manifest.json").read_text(encoding="utf-8"))
    train = json.loads((inp / "train.json").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in train}
    rows = [index[str(key)] for key in manifest["calibration_ids"]]
    tokenizer = OwnBPE(json.loads((inp / "tokenizer.json").read_text(encoding="utf-8"))["merges"])
    (out / "token_cache").mkdir(exist_ok=True)
    ids, masks = b1.tokenize_split("hybrid_calibration", rows, tokenizer, out / "token_cache")
    labels = b1.label_matrix(rows)
    loader = DataLoader(b1.Rows(ids, masks, labels, range(len(rows))), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    model = b1.B1Classifier(checkpoint)
    own = dict(model.named_parameters())
    for name, value in adapter["trainable_state"].items():
        own[name].data.copy_(value)
    model.cuda()
    probabilities = b1.predict(model, loader, "cuda")
    metrics = b1.f1_scores(labels, probabilities >= adapter["thresholds"])
    np.save(out / "b1_calibration_probabilities.npy", probabilities)
    (out / "calibration_ids.json").write_text(json.dumps([str(row["id"]) for row in rows]))
    report = {"source_adapter_sha256": digest(adapter_path), "phase1_checkpoint_sha256": digest(checkpoint), "rows": len(rows), "recomputed_calibration_macro_f1": metrics["macro_f1"], "saved_calibration_macro_f1": adapter["calibration_metrics"]["macro_f1"]}
    (out / "b1_calibration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])

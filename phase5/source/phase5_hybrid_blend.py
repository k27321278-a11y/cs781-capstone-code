"""Calibrate and export an official-train-only B1/sparse Phase 5 hybrid."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from phase5_core import choose_thresholds, f1_scores


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "phase5_data"
WORK = ROOT / "work" / "phase5_hybrid"
OUTPUT = ROOT / "outputs"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def labels(rows):
    result = np.zeros((len(rows), 100), dtype=np.uint8)
    for i, row in enumerate(rows):
        result[i, row["labels"]] = 1
    return result


def main():
    manifest = json.loads((OUTPUT / "phase5_split_manifest.json").read_text(encoding="utf-8"))
    train = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
    lookup = {str(row["id"]): row for row in train}
    calibration = [lookup[str(key)] for key in manifest["calibration_ids"]]
    ids = json.loads((WORK / "calibration_ids.json").read_text())
    assert ids == json.loads((WORK / "sparse_calibration_ids.json").read_text())
    assert ids == [str(row["id"]) for row in calibration]
    for name, rows in (("dev", dev), ("test", test)):
        assert json.loads((WORK / f"{name}_ids.json").read_text()) == [str(row["id"]) for row in rows]
    assert all("labels" not in row for row in test)
    b1 = {
        "calibration": np.load(WORK / "b1_calibration_probabilities.npy"),
        "dev": np.load(ROOT / "work" / "phase5_retrieved_lora" / "dev_probabilities.npy"),
        "test": np.load(ROOT / "work" / "phase5_retrieved_lora" / "test_probabilities.npy"),
    }
    sparse = {name: np.load(WORK / f"sparse_{name}_probabilities.npy") for name in b1}
    for name in b1:
        assert b1[name].shape == sparse[name].shape == (len({"calibration": calibration, "dev": dev, "test": test}[name]), 100)
        assert np.isfinite(b1[name]).all() and np.isfinite(sparse[name]).all()
    ycal = labels(calibration)
    trials = []
    for sparse_weight in (0., .25, .5, .75, .9):
        probs = sparse_weight * sparse["calibration"] + (1 - sparse_weight) * b1["calibration"]
        thresholds = choose_thresholds(ycal, probs)
        metrics = f1_scores(ycal, probs >= thresholds)
        trials.append((metrics["macro_f1"], sparse_weight, thresholds, metrics))
        print("calibration", sparse_weight, metrics["macro_f1"], flush=True)
    score, sparse_weight, thresholds, cal_metrics = max(trials, key=lambda x: x[0])
    dev_probs = sparse_weight * sparse["dev"] + (1 - sparse_weight) * b1["dev"]
    dev_metrics = f1_scores(labels(dev), dev_probs >= thresholds)
    test_probs = sparse_weight * sparse["test"] + (1 - sparse_weight) * b1["test"]
    predictions = [{"id": str(row["id"]), "pred": np.flatnonzero(prob >= thresholds).astype(int).tolist()} for row, prob in zip(test, test_probs)]
    zip_path = OUTPUT / "phase5_b1_sparse_hybrid_prediction.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions, separators=(",", ":")))
    report = {
        "method": "Phase 1 B1 LoRA encoder adaptation plus official-train-only full-document sparse classifier; fixed global probability blend",
        "phase1_adaptation": True,
        "selection": "Global blend weight and per-label thresholds selected exclusively on grouped official-train calibration",
        "source_b1_adapter_sha256": sha256(OUTPUT / "phase5_b1_lora_best_adapter.pt"),
        "source_sparse_model_sha256": sha256(ROOT / "work" / "agent_sparse" / "logistic_model.pkl"),
        "sparse_weight": sparse_weight,
        "b1_weight": 1 - sparse_weight,
        "calibration_macro_f1": cal_metrics["macro_f1"],
        "official_dev_macro_f1": dev_metrics["macro_f1"],
        "official_dev_micro_f1": dev_metrics["micro_f1"],
        "calibration_candidates": [{"sparse_weight": w, "macro_f1": s} for s, w, _, _ in trials],
        "fit_count": len(manifest["fit_ids"]),
        "calibration_count": len(calibration),
        "dev_count": len(dev),
        "test_count": len(test),
        "submission_sha256": sha256(zip_path),
        "codabench_score": None,
    }
    (OUTPUT / "phase5_b1_sparse_hybrid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_sparse_weight": sparse_weight, "calibration_macro_f1": score, "official_dev_macro_f1": dev_metrics["macro_f1"], "zip": str(zip_path)}), flush=True)


if __name__ == "__main__":
    main()

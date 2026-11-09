"""Select hybrid blend/threshold regularization by calibration-only CV.

The official dev labels are loaded only after CV has selected the recipe. The
new report and ZIP are separate from the existing fallback artifacts.
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
from phase5_core import f1_scores  # noqa: E402

DATA = ROOT / "work" / "phase5_data"
WORK = ROOT / "work" / "phase5_hybrid"
OUTPUT = ROOT / "outputs"
THRESHOLD_GRID = np.concatenate(
    [np.arange(0.01, 0.201, 0.01), np.arange(0.225, 0.901, 0.025)]
)
BLEND_WEIGHTS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PRIOR_STRENGTHS = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1e9)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_labels(rows):
    y = np.zeros((len(rows), 100), dtype=np.uint8)
    for i, row in enumerate(rows):
        y[i, row["labels"]] = 1
    return y


def fit_threshold_parts(y, probs):
    y = np.asarray(y, dtype=bool)
    probs = np.asarray(probs, dtype=np.float64)
    predictions = probs[:, :, None] >= THRESHOLD_GRID[None, None, :]
    tp = np.sum(predictions & y[:, :, None], axis=0, dtype=np.int64)
    predicted = np.sum(predictions, axis=0, dtype=np.int64)
    support = np.sum(y, axis=0, dtype=np.int64)
    denom = predicted + support[:, None]
    scores = np.divide(
        2 * tp, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0
    )
    global_t = float(THRESHOLD_GRID[int(np.argmax(scores.mean(axis=0)))])
    local_idx = np.argmax(scores, axis=1)
    local_t = THRESHOLD_GRID[local_idx].astype(np.float64)
    return global_t, local_t, support


def thresholds_from_parts(global_t, local_t, support, prior_strength):
    if prior_strength >= 1e8:
        return np.full(support.shape, global_t, dtype=np.float32)
    thresholds = np.full(support.shape, global_t, dtype=np.float64)
    eligible = support >= 10
    if prior_strength == 0:
        weight = np.ones_like(support, dtype=np.float64)
    else:
        weight = support / (support + prior_strength)
    thresholds[eligible] = (
        weight[eligible] * local_t[eligible]
        + (1.0 - weight[eligible]) * global_t
    )
    return np.clip(thresholds, 0.01, 0.9).astype(np.float32)


def main():
    manifest = json.loads(
        (OUTPUT / "phase5_split_manifest.json").read_text(encoding="utf-8")
    )
    train = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    train_lookup = {str(row["id"]): row for row in train}
    calibration_ids = json.loads((WORK / "calibration_ids.json").read_text())
    if calibration_ids != manifest["calibration_ids"]:
        raise ValueError("Calibration ID manifest mismatch")
    if calibration_ids != json.loads((WORK / "sparse_calibration_ids.json").read_text()):
        raise ValueError("B1 and sparse calibration IDs differ")
    calibration = [train_lookup[str(key)] for key in calibration_ids]
    y_cal = make_labels(calibration)
    b1_cal = np.load(WORK / "b1_calibration_probabilities.npy")
    sparse_cal = np.load(WORK / "sparse_calibration_probabilities.npy")
    expected = (len(calibration), 100)
    if b1_cal.shape != expected or sparse_cal.shape != expected:
        raise ValueError("Unexpected calibration probability shape")

    folds = list(KFold(n_splits=5, shuffle=True, random_state=781).split(y_cal))
    scores = {(w, k): [] for w in BLEND_WEIGHTS for k in PRIOR_STRENGTHS}
    for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
        y_fit, y_valid = y_cal[train_idx], y_cal[valid_idx]
        for weight in BLEND_WEIGHTS:
            p_fit = weight * sparse_cal[train_idx] + (1.0 - weight) * b1_cal[train_idx]
            p_valid = weight * sparse_cal[valid_idx] + (1.0 - weight) * b1_cal[valid_idx]
            global_t, local_t, support = fit_threshold_parts(y_fit, p_fit)
            for prior_strength in PRIOR_STRENGTHS:
                thresholds = thresholds_from_parts(
                    global_t, local_t, support, prior_strength
                )
                score = f1_scores(y_valid, p_valid >= thresholds)["macro_f1"]
                scores[(weight, prior_strength)].append(score)
        print(f"completed calibration fold {fold_no}/{len(folds)}", flush=True)

    cv_table = []
    for (weight, prior_strength), fold_scores in scores.items():
        cv_table.append(
            {
                "sparse_weight": weight,
                "b1_weight": 1.0 - weight,
                "threshold_prior_strength": prior_strength,
                "cv_macro_f1_mean": float(np.mean(fold_scores)),
                "cv_macro_f1_std": float(np.std(fold_scores)),
            }
        )
    best = max(
        cv_table,
        key=lambda row: (row["cv_macro_f1_mean"], row["threshold_prior_strength"]),
    )
    sparse_weight = best["sparse_weight"]
    prior_strength = best["threshold_prior_strength"]
    cal_probs = sparse_weight * sparse_cal + (1.0 - sparse_weight) * b1_cal
    global_t, local_t, support = fit_threshold_parts(y_cal, cal_probs)
    thresholds = thresholds_from_parts(global_t, local_t, support, prior_strength)
    cal_metrics = f1_scores(y_cal, cal_probs >= thresholds)

    # Only now open official dev labels for one final, calibration-selected check.
    dev = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
    if json.loads((WORK / "dev_ids.json").read_text()) != [str(x["id"]) for x in dev]:
        raise ValueError("Dev ID alignment mismatch")
    if json.loads((WORK / "test_ids.json").read_text()) != [str(x["id"]) for x in test]:
        raise ValueError("Test ID alignment mismatch")
    if any("labels" in row for row in test):
        raise ValueError("Unexpected labels in test input")

    b1_dev = np.load(ROOT / "work" / "phase5_retrieved_lora" / "dev_probabilities.npy")
    b1_test = np.load(ROOT / "work" / "phase5_retrieved_lora" / "test_probabilities.npy")
    sparse_dev = np.load(WORK / "sparse_dev_probabilities.npy")
    sparse_test = np.load(WORK / "sparse_test_probabilities.npy")
    dev_probs = sparse_weight * sparse_dev + (1.0 - sparse_weight) * b1_dev
    test_probs = sparse_weight * sparse_test + (1.0 - sparse_weight) * b1_test
    dev_metrics = f1_scores(make_labels(dev), dev_probs >= thresholds)
    predictions = [
        {"id": str(row["id"]), "pred": np.flatnonzero(prob >= thresholds).astype(int).tolist()}
        for row, prob in zip(test, test_probs)
    ]
    zip_path = OUTPUT / "phase5_b1_sparse_hybrid_cv_prediction.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions, separators=(",", ":")))

    report = {
        "method": "Phase 1 B1 LoRA plus official-train-only full-text sparse classifier with calibration-CV-selected fixed blend and support-shrunk per-label thresholds",
        "phase1_adaptation": True,
        "selection": "Blend weight and threshold prior strength selected by five-fold cross-validation entirely within the official-train calibration split; final thresholds refit on all calibration rows",
        "phase1_checkpoint_sha256": "111296fb36a23dac1a11c07ede927cfc89dc86d51253e441d59afb82bd6924eb",
        "source_b1_adapter_sha256": sha256(OUTPUT / "phase5_b1_lora_best_adapter.pt"),
        "source_sparse_model_sha256": sha256(ROOT / "work" / "agent_sparse" / "logistic_model.pkl"),
        "sparse_weight": sparse_weight,
        "b1_weight": 1.0 - sparse_weight,
        "threshold_prior_strength": prior_strength,
        "calibration_macro_f1": cal_metrics["macro_f1"],
        "calibration_micro_f1": cal_metrics["micro_f1"],
        "selected_cv_macro_f1": best["cv_macro_f1_mean"],
        "selected_cv_macro_f1_std": best["cv_macro_f1_std"],
        "cv_folds": 5,
        "cv_grid": cv_table,
        "official_dev_macro_f1": dev_metrics["macro_f1"],
        "official_dev_micro_f1": dev_metrics["micro_f1"],
        "fit_count": len(manifest["fit_ids"]),
        "calibration_count": len(calibration),
        "dev_count": len(dev),
        "test_count": len(test),
        "submission_sha256": sha256(zip_path),
        "codabench_score": None,
        "course_method_eligibility": "Unverified: confirm the capstone method list accepts a Phase 1-adapted B1 plus full-text sparse hybrid.",
    }
    report_path = OUTPUT / "phase5_b1_sparse_hybrid_cv_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_sparse_weight": sparse_weight,
                "threshold_prior_strength": prior_strength,
                "calibration_macro_f1": cal_metrics["macro_f1"],
                "cv_macro_f1": best["cv_macro_f1_mean"],
                "official_dev_macro_f1": dev_metrics["macro_f1"],
                "zip_sha256": report["submission_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

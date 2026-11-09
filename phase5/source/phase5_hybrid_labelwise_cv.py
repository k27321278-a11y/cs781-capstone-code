"""Calibration-CV selection of per-label B1/sparse blend weights."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from phase5_hybrid_calibration_cv import (
    PRIOR_STRENGTHS,
    THRESHOLD_GRID,
    fit_threshold_parts,
    sha256,
    thresholds_from_parts,
)
from phase5_core import f1_scores

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "phase5_data"
WORK = ROOT / "work" / "phase5_hybrid"
OUTPUT = ROOT / "outputs"
GLOBAL_SPARSE_WEIGHT = 0.90
LOCAL_WEIGHT_GRID = (0.50, 0.70, 0.80, 0.90, 0.95)
WEIGHT_PRIORS = (0.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)


def make_labels(rows):
    y = np.zeros((len(rows), 100), dtype=np.uint8)
    for i, row in enumerate(rows):
        y[i, row["labels"]] = 1
    return y


def fit_local_weights(y, b1_probs, sparse_probs):
    support = np.sum(y, axis=0, dtype=np.int64)
    weights = np.full(y.shape[1], GLOBAL_SPARSE_WEIGHT, dtype=np.float64)
    for j in range(y.shape[1]):
        if support[j] < 10:
            continue
        best_score = -1.0
        best_weight = GLOBAL_SPARSE_WEIGHT
        for weight in LOCAL_WEIGHT_GRID:
            p = weight * sparse_probs[:, j] + (1.0 - weight) * b1_probs[:, j]
            pred = p[:, None] >= THRESHOLD_GRID[None, :]
            tp = np.sum(pred & y[:, j, None], axis=0, dtype=np.int64)
            predicted = np.sum(pred, axis=0, dtype=np.int64)
            scores = np.divide(
                2 * tp,
                predicted + support[j],
                out=np.zeros_like(THRESHOLD_GRID, dtype=np.float64),
                where=(predicted + support[j]) > 0,
            )
            score = float(np.max(scores))
            if (score > best_score + 1e-12) or (
                abs(score - best_score) <= 1e-12
                and abs(weight - GLOBAL_SPARSE_WEIGHT)
                < abs(best_weight - GLOBAL_SPARSE_WEIGHT)
            ):
                best_score = score
                best_weight = weight
        weights[j] = best_weight
    return weights, support


def shrink_weights(local_weights, support, prior):
    if prior == 0:
        alpha = np.ones_like(support, dtype=np.float64)
    else:
        alpha = support / (support + prior)
    shrunk = GLOBAL_SPARSE_WEIGHT + alpha * (local_weights - GLOBAL_SPARSE_WEIGHT)
    return np.clip(shrunk, 0.50, 0.95)


def blend(b1_probs, sparse_probs, label_weights):
    return label_weights[None, :] * sparse_probs + (1.0 - label_weights[None, :]) * b1_probs


def main():
    manifest = json.loads((OUTPUT / "phase5_split_manifest.json").read_text(encoding="utf-8"))
    train = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    train_by_id = {str(row["id"]): row for row in train}
    calibration_ids = json.loads((WORK / "calibration_ids.json").read_text())
    if calibration_ids != manifest["calibration_ids"]:
        raise ValueError("Calibration ID manifest mismatch")
    if calibration_ids != json.loads((WORK / "sparse_calibration_ids.json").read_text()):
        raise ValueError("B1/sparse calibration IDs differ")
    calibration = [train_by_id[str(row_id)] for row_id in calibration_ids]
    y_cal = make_labels(calibration)
    b1_cal = np.load(WORK / "b1_calibration_probabilities.npy")
    sparse_cal = np.load(WORK / "sparse_calibration_probabilities.npy")
    expected = (len(calibration), 100)
    if b1_cal.shape != expected or sparse_cal.shape != expected:
        raise ValueError("Unexpected calibration probability shape")

    fold_rows = []
    split_iter = KFold(n_splits=5, shuffle=True, random_state=781).split(y_cal)
    for fold_no, (fit_idx, valid_idx) in enumerate(split_iter, start=1):
        y_fit, y_valid = y_cal[fit_idx], y_cal[valid_idx]
        local_weights, support = fit_local_weights(
            y_fit, b1_cal[fit_idx], sparse_cal[fit_idx]
        )
        for weight_prior in WEIGHT_PRIORS:
            weights = shrink_weights(local_weights, support, weight_prior)
            p_fit = blend(b1_cal[fit_idx], sparse_cal[fit_idx], weights)
            p_valid = blend(b1_cal[valid_idx], sparse_cal[valid_idx], weights)
            global_t, local_t, threshold_support = fit_threshold_parts(y_fit, p_fit)
            for threshold_prior in PRIOR_STRENGTHS:
                thresholds = thresholds_from_parts(
                    global_t, local_t, threshold_support, threshold_prior
                )
                score = f1_scores(y_valid, p_valid >= thresholds)["macro_f1"]
                fold_rows.append(
                    {
                        "fold": fold_no,
                        "weight_prior": weight_prior,
                        "threshold_prior": threshold_prior,
                        "macro_f1": score,
                    }
                )
        print(f"completed calibration fold {fold_no}/5", flush=True)

    cv_grid = []
    for weight_prior in WEIGHT_PRIORS:
        for threshold_prior in PRIOR_STRENGTHS:
            vals = [
                row["macro_f1"]
                for row in fold_rows
                if row["weight_prior"] == weight_prior
                and row["threshold_prior"] == threshold_prior
            ]
            cv_grid.append(
                {
                    "weight_prior": weight_prior,
                    "threshold_prior": threshold_prior,
                    "cv_macro_f1_mean": float(np.mean(vals)),
                    "cv_macro_f1_std": float(np.std(vals)),
                }
            )
    best_score = max(row["cv_macro_f1_mean"] for row in cv_grid)
    near_best = [row for row in cv_grid if row["cv_macro_f1_mean"] >= best_score - 0.001]
    best = max(near_best, key=lambda row: (row["weight_prior"], row["threshold_prior"]))

    local_weights, support = fit_local_weights(y_cal, b1_cal, sparse_cal)
    final_weights = shrink_weights(local_weights, support, best["weight_prior"])
    cal_probs = blend(b1_cal, sparse_cal, final_weights)
    global_t, local_t, threshold_support = fit_threshold_parts(y_cal, cal_probs)
    thresholds = thresholds_from_parts(global_t, local_t, threshold_support, best["threshold_prior"])
    cal_metrics = f1_scores(y_cal, cal_probs >= thresholds)

    # Open official dev only after calibration-CV has fixed the recipe.
    dev = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
    if json.loads((WORK / "dev_ids.json").read_text()) != [str(row["id"]) for row in dev]:
        raise ValueError("Dev ID alignment mismatch")
    if json.loads((WORK / "test_ids.json").read_text()) != [str(row["id"]) for row in test]:
        raise ValueError("Test ID alignment mismatch")
    if any("labels" in row for row in test):
        raise ValueError("Unexpected labels in test inputs")

    b1_dev = np.load(ROOT / "work" / "phase5_retrieved_lora" / "dev_probabilities.npy")
    b1_test = np.load(ROOT / "work" / "phase5_retrieved_lora" / "test_probabilities.npy")
    sparse_dev = np.load(WORK / "sparse_dev_probabilities.npy")
    sparse_test = np.load(WORK / "sparse_test_probabilities.npy")
    dev_probs = blend(b1_dev, sparse_dev, final_weights)
    test_probs = blend(b1_test, sparse_test, final_weights)
    dev_metrics = f1_scores(make_labels(dev), dev_probs >= thresholds)
    predictions = [
        {"id": str(row["id"]), "pred": np.flatnonzero(prob >= thresholds).astype(int).tolist()}
        for row, prob in zip(test, test_probs)
    ]
    zip_path = OUTPUT / "phase5_b1_sparse_hybrid_labelwise_prediction.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions, separators=(",", ":")))
    report = {
        "method": "Phase 1 B1 LoRA plus official-train-only sparse classifier with calibration-CV-selected per-label blend weights and support-shrunk thresholds",
        "phase1_adaptation": True,
        "selection": "Per-label sparse weights and threshold shrinkage selected using five-fold CV within official-train calibration only; final weights and thresholds refit on all calibration rows",
        "phase1_checkpoint_sha256": "111296fb36a23dac1a11c07ede927cfc89dc86d51253e441d59afb82bd6924eb",
        "source_b1_adapter_sha256": sha256(OUTPUT / "phase5_b1_lora_best_adapter.pt"),
        "source_sparse_model_sha256": sha256(ROOT / "work" / "agent_sparse" / "logistic_model.pkl"),
        "global_sparse_weight": GLOBAL_SPARSE_WEIGHT,
        "weight_prior_strength": best["weight_prior"],
        "threshold_prior_strength": best["threshold_prior"],
        "selected_cv_macro_f1": best["cv_macro_f1_mean"],
        "selected_cv_macro_f1_std": best["cv_macro_f1_std"],
        "cv_folds": 5,
        "cv_grid": cv_grid,
        "calibration_macro_f1": cal_metrics["macro_f1"],
        "official_dev_macro_f1": dev_metrics["macro_f1"],
        "official_dev_micro_f1": dev_metrics["micro_f1"],
        "calibration_count": len(calibration),
        "dev_count": len(dev),
        "test_count": len(test),
        "per_label_sparse_weights": final_weights.tolist(),
        "submission_sha256": sha256(zip_path),
        "codabench_score": None,
        "course_method_eligibility": "Unverified: confirm the capstone method list accepts a Phase 1-adapted B1 plus full-text sparse hybrid.",
    }
    (OUTPUT / "phase5_b1_sparse_hybrid_labelwise_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "weight_prior": best["weight_prior"],
                "threshold_prior": best["threshold_prior"],
                "selected_cv_macro_f1": best["cv_macro_f1_mean"],
                "calibration_macro_f1": cal_metrics["macro_f1"],
                "official_dev_macro_f1": dev_metrics["macro_f1"],
                "zip_sha256": report["submission_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

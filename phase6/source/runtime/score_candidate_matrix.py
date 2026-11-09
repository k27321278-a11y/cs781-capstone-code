#!/usr/bin/env python3
"""Score completed Phase 6 held-out generations on identical public splits.

This reads public train/test inputs only. Test documents are used solely to
reproduce the official train/test duplicate exclusion; no test references are
loaded. The main comparison uses identical first 16 validation and first 32
confirmation IDs for every full-budget run. The short AdaLoRA run is reported
separately at its actual 8/8 sample size.
"""
import argparse
import hashlib
import json
from pathlib import Path

from rouge_score import rouge_scorer
import bert_score
import torch


EXPERIMENTS = {
    "legalspan_lora": ("legalspan_lora", "main"),
    "b1_lorma_plus": ("b1_lorma", "main"),
    "c1_houlsby": ("c1_houlsby", "main"),
    "c1_houlsby_continuation": ("c1_continue", "main"),
    "d222_prefix": ("d222_prefix", "main"),
    "b1_adalora_short": ("b1_adalora", "short"),
    "b1_hierarchical_evidence_fusion": (None, "main"),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def doc_hash(text):
    import re
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().casefold().encode()).hexdigest()


def normalize(data, training, drop_empty=False):
    rows = []
    items = data.items() if isinstance(data, dict) else ((None, row) for row in data)
    for key, value in items:
        row = dict(value) if isinstance(value, dict) else {"document": value}
        row["id"] = str(row.get("id", key))
        if training and (not isinstance(row.get("summary"), str) or not row["summary"].strip()):
            if drop_empty:
                continue
            raise ValueError(f"Missing training summary for {row['id']}")
        rows.append(row)
    return rows


def split(rows):
    out = {"fit": [], "validation": [], "confirmation": []}
    for row in rows:
        bucket = int(doc_hash(row["document"])[:12], 16) % 100
        name = "fit" if bucket < 90 else "validation" if bucket < 98 else "confirmation"
        out[name].append(row)
    return out


def expected_ids(rows, name, count):
    ordered = sorted(rows[name], key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
    if len(ordered) < count:
        raise ValueError(f"Only {len(ordered)} rows available for {name}, need {count}")
    return [row["id"] for row in ordered[:count]]


def prediction_map(path, wanted_ids):
    rows = load(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list predictions in {path}")
    mapping = {}
    order = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"Malformed prediction row in {path}")
        if row["id"] in mapping:
            raise ValueError(f"Duplicate prediction ID {row['id']} in {path}")
        text = row.get("summary")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty prediction for {row['id']} in {path}")
        mapping[row["id"]] = text.strip()
        order.append(row["id"])
    if order[:len(wanted_ids)] != wanted_ids:
        raise ValueError(f"Prediction IDs/order differ from deterministic held-out subset: {path}")
    return mapping


def rouge1_f1(scorer, reference, prediction):
    return scorer.score(reference, prediction)["rouge1"].fmeasure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outputs", required=True, help="leaderboard_models directory")
    ap.add_argument("--hif-output", required=True, help="HIF kernel output folder")
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    raw_train = load(args.train)
    train = normalize(raw_train, True, drop_empty=True)
    test = normalize(load(args.test), False)
    test_hashes = {doc_hash(row["document"]) for row in test}
    eligible = [row for row in train if doc_hash(row["document"]) not in test_hashes]
    splits = split(eligible)

    hif_report = load(Path(args.hif_output) / "report.json")
    experiments = dict(EXPERIMENTS)
    experiments["b1_hierarchical_evidence_fusion"] = (str(Path(args.hif_output)), "main")

    # HIF's verified split uses 16 validation and 32 confirmation examples.
    # The full-run comparison is intentionally matched to that exact subset.
    main_counts = {"validation": 16, "confirmation": 32}
    short_counts = {"validation": 8, "confirmation": 8}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    records = []
    bert_inputs, bert_keys = [], []
    references = {}

    for name, (directory, group) in experiments.items():
        folder = Path(args.hif_output) if group == "main" and name == "b1_hierarchical_evidence_fusion" else Path(args.outputs) / directory
        report = hif_report if name == "b1_hierarchical_evidence_fusion" else load(next(folder.rglob("report.json")))
        selected_counts = main_counts if group == "main" else short_counts
        for split_name, count in selected_counts.items():
            ids = expected_ids(splits, split_name, count)
            pred_name = "best_validation_predictions.json" if split_name == "validation" else "confirmation_predictions.json"
            pred_file = next(folder.rglob(pred_name))
            mapping = prediction_map(pred_file, ids)
            truth = {row["id"]: row["summary"] for row in splits[split_name]}
            refs = [truth[item] for item in ids]
            preds = [mapping[item] for item in ids]
            rouge1 = sum(rouge1_f1(scorer, ref, pred) for ref, pred in zip(refs, preds)) / count
            rouge_l = sum(
                scorer.score(ref, pred)["rougeL"].fmeasure
                for ref, pred in zip(refs, preds)
            ) / count
            key = f"{name}:{split_name}"
            references[key] = refs
            bert_inputs.extend(preds)
            bert_keys.extend([key] * count)
            records.append({
                "model": name,
                "comparison_group": group,
                "split": split_name,
                "count": count,
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "rouge1_f1": rouge1,
                "rougeL_f1": rouge_l,
                "mean_generated_words": sum(len(p.split()) for p in preds) / count,
                "selected_epoch": report.get("selected_epoch"),
                "codabench_score": None,
            })

    # Matching order in references is the same as in bert_inputs.
    flat_refs = []
    cursor = {}
    for key in bert_keys:
        i = cursor.get(key, 0)
        flat_refs.append(references[key][i])
        cursor[key] = i + 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = bert_score.score(
        bert_inputs, flat_refs, lang="en", batch_size=8, verbose=False, device=device
    )
    group_scores = {}
    for key, value in zip(bert_keys, f1.tolist()):
        group_scores.setdefault(key, []).append(float(value))
    for row in records:
        values = group_scores[f"{row['model']}:{row['split']}"]
        row["bertscore_f1"] = sum(values) / len(values)

    result = {
        "protocol": {
            "source": "public Phase 6 training summaries only",
            "validation_n": 16,
            "confirmation_n": 32,
            "short_run_n": 8,
            "test_references_used": False,
            "test_documents_used_only_for_train_duplicate_exclusion": True,
            "bertscore_device": device,
            "comparison_note": "Main models use the same deterministic IDs. AdaLoRA is a separate 8/8 short-run diagnostic and should not be ranked against the larger samples.",
        },
        "results": records,
    }
    out = Path(args.result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(out), "rows": len(records), "device": device}))


if __name__ == "__main__":
    main()

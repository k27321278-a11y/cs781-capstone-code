#!/usr/bin/env python3
"""Add one completed model to the matched public Phase 6 score matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import bert_score
from rouge_score import rouge_scorer
import torch

from score_candidate_matrix import (
    doc_hash,
    expected_ids,
    load,
    normalize,
    prediction_map,
    split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", choices=["main", "short"], default="main")
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    train = normalize(load(args.train), True, drop_empty=True)
    test = normalize(load(args.test), False)
    test_hashes = {doc_hash(row["document"]) for row in test}
    eligible = [row for row in train if doc_hash(row["document"]) not in test_hashes]
    splits = split(eligible)
    report = load(next(args.folder.rglob("report.json")))
    counts = {"validation": 16, "confirmation": 32} if args.group == "main" else {
        "validation": 8,
        "confirmation": 8,
    }

    records = []
    predictions_flat = []
    references_flat = []
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    for split_name, count in counts.items():
        ids = expected_ids(splits, split_name, count)
        pred_name = "best_validation_predictions.json" if split_name == "validation" else "confirmation_predictions.json"
        path = next(args.folder.rglob(pred_name))
        mapping = prediction_map(path, ids)
        truth = {row["id"]: row["summary"] for row in splits[split_name]}
        refs = [truth[item] for item in ids]
        preds = [mapping[item] for item in ids]
        id_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        records.append({
            "model": args.model,
            "comparison_group": args.group,
            "split": split_name,
            "count": count,
            "ids_sha256": id_hash,
            "rouge1_f1": sum(rouge.score(r, p)["rouge1"].fmeasure for r, p in zip(refs, preds)) / count,
            "rougeL_f1": sum(rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)) / count,
            "mean_generated_words": sum(len(p.split()) for p in preds) / count,
            "selected_epoch": report.get("selected_epoch"),
            "codabench_score": report.get("codabench_score"),
        })
        predictions_flat.extend(preds)
        references_flat.extend(refs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, scores = bert_score.score(
        predictions_flat,
        references_flat,
        lang="en",
        batch_size=8,
        verbose=False,
        device=device,
    )
    cursor = 0
    for record in records:
        n = record["count"]
        record["bertscore_f1"] = float(scores[cursor:cursor + n].mean())
        cursor += n

    prior = load(args.matrix) if args.matrix.exists() else {
        "protocol": {
            "source": "public Phase 6 training summaries only",
            "validation_n": 16,
            "confirmation_n": 32,
            "short_run_n": 8,
            "test_references_used": False,
            "test_documents_used_only_for_train_duplicate_exclusion": True,
        },
        "results": [],
    }
    for record in records:
        peer = next((row for row in prior["results"] if row["model"] == "legalspan_lora" and row["split"] == record["split"]), None)
        if peer and peer["ids_sha256"] != record["ids_sha256"]:
            raise ValueError("Candidate split IDs do not match the common comparison IDs")
    prior["results"] = [row for row in prior["results"] if row["model"] != args.model] + records
    prior["protocol"]["a0_sequence_kd_added"] = args.model == "a0_sequence_kd"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "device": device, "rows": records, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

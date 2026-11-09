#!/usr/bin/env python3
"""Verify sequence-KD teacher summaries are hashed and fitting-only."""

import argparse
import hashlib
import json
from pathlib import Path

from phase6_data import document_hash, normalize_rows, split_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    train = normalize_rows(json.loads(args.train.read_text(encoding="utf-8-sig")), True, drop_empty_summaries=True)
    test = normalize_rows(json.loads(args.test.read_text(encoding="utf-8-sig")), False)
    test_hashes = {document_hash(row["document"]) for row in test}
    eligible = [row for row in train if document_hash(row["document"]) not in test_hashes]
    fit_ids = {row["id"] for row in split_rows(eligible)["fit"]}

    teacher_data = json.loads(args.teacher.read_text(encoding="utf-8-sig"))
    if isinstance(teacher_data, list):
        ids = [str(row["id"]) for row in teacher_data]
        nonempty = all(isinstance(row.get("summary"), str) and row["summary"].strip() for row in teacher_data)
    elif isinstance(teacher_data, dict):
        ids = [str(key) for key in teacher_data]
        nonempty = all(isinstance(value, str) and value.strip() for value in teacher_data.values())
    else:
        raise ValueError("Teacher predictions must be a list or ID-to-summary object")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    teacher_hash = sha256(args.teacher)
    checks = {
        "teacher_target_count": len(ids),
        "unique_ids": len(ids) == len(set(ids)),
        "all_ids_in_fit_split": set(ids) <= fit_ids,
        "no_test_or_heldout_ids": not (set(ids) - fit_ids),
        "nonempty_targets": nonempty,
        "manifest_fit_only": manifest.get("fit_only") is True,
        "manifest_count_matches": manifest.get("document_count") == len(ids),
        "teacher_file_sha256_matches": manifest.get("output_sha256") == teacher_hash,
        "teacher_parent_sha256": manifest.get("teacher_parent_sha256"),
        "teacher_adapter_sha256": manifest.get("teacher_adapter_sha256"),
        "teacher_selected_epoch": manifest.get("teacher_selected_epoch"),
    }
    passed = all(value for key, value in checks.items() if isinstance(value, bool))
    result = {"passed": passed, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

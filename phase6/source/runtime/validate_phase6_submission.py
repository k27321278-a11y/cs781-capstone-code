#!/usr/bin/env python3
"""Verify a Phase 6 prediction ZIP against the official test ID file."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from phase6_data import normalize_rows, validate_predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_predictions(value):
    if isinstance(value, dict):
        rows = [{"id": str(key), "summary": summary} for key, summary in value.items()]
    elif isinstance(value, list):
        rows = value
    else:
        raise ValueError("prediction.json must be a list or ID-to-summary object")
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError("Malformed prediction record")
        row["id"] = str(row["id"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--expected-parent-sha")
    parser.add_argument("--expected-tokenizer-sha")
    args = parser.parse_args()

    test_rows = normalize_rows(json.loads(args.test.read_text(encoding="utf-8-sig")), False)
    expected_ids = [row["id"] for row in test_rows]
    if len(expected_ids) != args.expected_count:
        raise ValueError(f"Official test file has {len(expected_ids)} IDs, expected {args.expected_count}")

    with zipfile.ZipFile(args.zip) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC failed for {bad_member}")
        names = archive.namelist()
        if names != ["prediction.json"]:
            raise ValueError(f"Expected only root-level prediction.json, found {names}")
        predictions = as_predictions(json.loads(archive.read("prediction.json")))
    validate_predictions(predictions, expected_ids)

    archive_sha = sha256(args.zip)
    report = json.loads(args.report.read_text(encoding="utf-8-sig")) if args.report else {}
    report_sha = report.get("zip_sha256")
    if report_sha and report_sha != archive_sha:
        raise ValueError("ZIP SHA256 differs from the run report")
    report_test_count = report.get("test_count")
    if report_test_count is not None and report_test_count != args.expected_count:
        raise ValueError("Report test_count differs from official test count")

    parent = report.get("phase1_parent", {})
    parent_sha = parent.get("phase1_checkpoint_sha256")
    tokenizer_sha = parent.get("tokenizer_sha256")
    if args.expected_parent_sha and parent_sha != args.expected_parent_sha:
        raise ValueError("Phase 1 parent SHA256 does not match the expected checkpoint")
    if args.expected_tokenizer_sha and tokenizer_sha != args.expected_tokenizer_sha:
        raise ValueError("Tokenizer SHA256 does not match the expected tokenizer")
    result = {
        "zip": str(args.zip),
        "zip_sha256": archive_sha,
        "zip_crc": "passed",
        "members": names,
        "prediction_count": len(predictions),
        "exact_test_ids": True,
        "nonempty_summaries": True,
        "report_zip_sha256_matches": report_sha == archive_sha if report_sha else None,
        "phase1_checkpoint_sha256": parent_sha,
        "tokenizer_sha256": tokenizer_sha,
        "selected_epoch": report.get("selected_epoch"),
        "codabench_score": report.get("codabench_score"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from .io import sha256_file, write_json_atomic


SCORER_ALIAS = "roberta-large"
SCORER_REPOSITORY = "FacebookAI/roberta-large"
SCORER_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"


def parse_metrics(output):
    metrics = {}
    for label, key in (("BERTScore F1", "bertscore_f1"), ("ROUGE-L F1", "rouge_l_f1"), ("BLEU-4", "bleu4")):
        match = re.search(rf"^{re.escape(label)}:\s+(\S+)", output, re.MULTILINE)
        if not match or not math.isfinite(float(match.group(1))):
            raise ValueError("Missing or nonfinite official metric")
        metrics[key] = float(match.group(1))
    return metrics


def load_scorer_provenance(model_cache):
    cache = Path(model_cache).resolve()
    record_path = cache.parent / "bert_score_model_provenance.json"
    if not cache.is_dir() or not record_path.is_file():
        raise FileNotFoundError("Pinned BERTScore cache and provenance record are required")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Invalid pinned BERTScore provenance record") from None

    expected_snapshot = cache / "models--roberta-large" / "snapshots" / SCORER_REVISION
    ref = cache / "models--roberta-large" / "refs" / "main"
    if (
        record.get("requested_id") != SCORER_ALIAS
        or record.get("resolved_id") != SCORER_REPOSITORY
        or record.get("revision") != SCORER_REVISION
        or record.get("offline_config_tokenizer_resolution") != "PASS"
        or Path(record.get("snapshot", "")).resolve() != expected_snapshot.resolve()
        or not expected_snapshot.is_dir()
        or not ref.is_file()
        or ref.read_text(encoding="utf-8").strip() != SCORER_REVISION
    ):
        raise ValueError("BERTScore cache is not pinned to the approved evaluator model revision")

    files = {item.get("path"): item for item in record.get("files", []) if isinstance(item, dict)}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors"}
    if not required.issubset(files):
        raise ValueError("Pinned scorer provenance omits required model or tokenizer files")
    for name in required:
        item = files[name]
        path = expected_snapshot / name
        if not path.is_file() or path.stat().st_size != item.get("bytes") or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            raise ValueError("Pinned scorer asset is absent or does not match its provenance record")
    record["file_hashes"] = {name: files[name]["sha256"] for name in required}
    return record


def score_predictions(reference_json, prediction_json, evaluator_path, model_cache):
    cache = Path(model_cache).resolve()
    provenance = load_scorer_provenance(cache)
    env = os.environ.copy()
    env.update({
        "HF_HUB_CACHE": str(cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    process = subprocess.run(
        [sys.executable, str(evaluator_path), "--truth", str(reference_json), "--prediction", str(prediction_json)],
        capture_output=True,
        text=True,
        env=env,
    )
    if process.returncode:
        raise RuntimeError(f"Published evaluator failed (exit {process.returncode}); check dependencies, pinned model cache and IDs")
    metrics = parse_metrics(process.stdout)
    versions = {}
    for package in ("bert-score", "transformers", "torch", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "metrics": metrics,
        "evaluator_sha256": sha256_file(evaluator_path),
        "prediction_sha256": sha256_file(prediction_json),
        "reference_sha256": sha256_file(reference_json),
        "versions": versions,
        "raw_stdout": process.stdout,
        "bert_score_settings": {
            "lang": "en",
            "batch_size": 8,
            "idf": False,
            "rescale_with_baseline": False,
            "model": SCORER_ALIAS,
            "resolved_model": SCORER_REPOSITORY,
            "revision": SCORER_REVISION,
            "offline": True,
            "model_files": {name: provenance["file_hashes"][name]
                            for name in ("model.safetensors", "config.json", "tokenizer.json", "vocab.json", "merges.txt")},
        },
    }
    write_json_atomic(Path(prediction_json).with_suffix(".metrics.json"), report)
    return metrics

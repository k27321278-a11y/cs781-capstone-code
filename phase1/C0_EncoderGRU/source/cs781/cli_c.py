"""Train, evaluate and package architecture C without changing A's active CLI."""
import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

import torch

from .contracts import GenerationConfig, TrainConfig
from .evaluate import score_predictions
from .generate_c import generate_continuation_c
from .io import sha256_file, write_json_atomic
from .models.c import RecurrentConfig, RecurrentLegalModel
from .prepare import load_prepared
from .cli import (
    _download_official_test,
    _official_dataset_revision,
)
from .submission import (
    extract_official_test_prompts,
    load_json_strict,
    validate_predictions,
    write_submission,
)
from .tokenizer import load_tokenizer
from .train_c import _code_digest, train_c
from .prediction_progress import PredictionProgress
from .checkpoints import run_lock


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs/c.json"
EVALUATOR = Path(__file__).parents[1] / "vendor/phase1_evaluate.py"


def _checkpoint_run_directory(path):
    checkpoint = Path(path).resolve()
    latest = checkpoint.parent.name == "checkpoints" and checkpoint.name == "latest.pt"
    candidate = checkpoint.parent.name == "candidates" and re.fullmatch(r"(step_\d{7,}|milestone_\d+)\.pt", checkpoint.name)
    if not checkpoint.is_file() or not (latest or candidate):
        raise ValueError("Use C latest.pt or a recorded step/milestone candidate")
    run_dir = checkpoint.parent.parent
    metadata = load_json_strict(run_dir / "run.json")
    if not isinstance(metadata, dict):
        raise ValueError("C run provenance must be an object")
    return checkpoint, run_dir, metadata


def _generate_records(output, prompts, model, tokenizer, decode, fingerprint):
    progress = PredictionProgress(output.with_suffix(".progress.json"), prompts, fingerprint)
    pending = list(progress.pending())
    for index, key in enumerate(pending, start=1):
        try:
            progress.record(key, generate_continuation_c(model, tokenizer, prompts[key], decode))
        except (RuntimeError, ValueError) as exc:
            progress.failure(key, type(exc).__name__ + ": " + str(exc))
        if index % 10 == 0 or index == len(pending):
            print(json.dumps({"phase": "C generation", "completed": len(progress.predictions),
                              "total": len(prompts), "failures": len(progress.failures)}), flush=True)
    return progress.complete()

def _config(path):
    value = load_json_strict(path)
    if not isinstance(value, dict) or not all(k in value for k in ("model", "train", "decoding")):
        raise ValueError("C config must define model, train and decoding sections")
    return value


def _model_from_checkpoint(checkpoint_path, data, device_name="cuda"):
    checkpoint, run_dir, run_metadata = _checkpoint_run_directory(checkpoint_path)
    payload, loaded_hash = _load_checkpoint_snapshot(checkpoint)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("C checkpoint payload or provenance is malformed")
    metadata = payload["metadata"]
    manifest = load_json_strict(data.root / "manifest.json")
    if metadata != run_metadata or metadata.get("architecture") != "C":
        raise ValueError("Checkpoint is not a matching architecture-C project run")
    if (metadata.get("synthetic_smoke") is not False or manifest.get("synthetic") is not False
            or metadata.get("data_manifest_sha256") != data.manifest_sha256
            or metadata.get("tokenizer_sha256") != data.tokenizer_sha256):
        raise ValueError("C checkpoint is synthetic or has mismatched official data/tokenizer provenance")
    if (re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("code_sha256", ""))) is None
            or metadata["code_sha256"] != _code_digest()):
        raise ValueError("C checkpoint source code differs from the code used for training")
    count = metadata.get("unique_parameters")
    if type(count) is not int or not 50_000_000 <= count <= 300_000_000:
        raise ValueError("C checkpoint parameter count is outside the foundation range")
    try:
        cfg = RecurrentConfig(**metadata["model_config"])
        cfg.validate()
    except (KeyError, TypeError, ValueError):
        raise ValueError("C checkpoint model configuration is invalid") from None
    if cfg.max_seq_len != manifest.get("max_seq_len"):
        raise ValueError("C checkpoint context differs from prepared-data context")
    if _official_dataset_revision(manifest) is None:
        raise ValueError("C run is not tied to the pinned official corpus")
    if not isinstance(payload.get("model"), dict) or not isinstance(payload.get("state"), dict):
        raise ValueError("C checkpoint is missing model weights or training state")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for C inference but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError("Requested C CUDA device is outside the visible-device set")
    model = RecurrentLegalModel(cfg).to(device)
    if model.count_parameters() != count:
        raise ValueError("Instantiated C parameter count differs from checkpoint provenance")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    payload["_loaded_checkpoint_sha256"] = loaded_hash
    return model, payload, run_dir, metadata, device


def _load_checkpoint_snapshot(path):
    # The trainer atomically replaces latest.pt. Keeping one descriptor open
    # binds both weights and their digest to the same inode even during a save.
    with Path(path).open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=False)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return payload, digest.hexdigest()


def _finalize_official_export(output, predictions, prompts, report):
    output = Path(output)
    validate_predictions(predictions, prompts)
    if output.exists():
        with ZipFile(output) as archive:
            if archive.namelist() != ["prediction.json"]:
                raise ValueError("The existing ZIP has invalid root entries")
            records = json.loads(archive.read("prediction.json"))
        expected = {key: {"text": value.strip()} for key, value in predictions.items()}
        if records != expected:
            raise ValueError("The existing ZIP differs from completed prediction progress")
        zip_hash = sha256_file(output)
    else:
        zip_hash = write_submission(predictions, prompts, output)
    result = {**report, "prediction_zip_sha256": zip_hash}
    context = output.with_suffix(".context.json")
    if context.exists() and load_json_strict(context) != result:
        raise ValueError("Existing submission context has different provenance")
    write_json_atomic(context, result)
    return result


def _require_complete_training(checkpoint, payload, run_dir, metadata):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != (run_dir / "checkpoints/latest.pt").resolve():
        raise ValueError("Official prediction requires the completed run's latest checkpoint")
    if metadata.get("synthetic_smoke") is not False:
        raise ValueError("Official prediction requires substantive, non-synthetic training")
    train_config = metadata.get("train_config")
    state = payload.get("state")
    if not isinstance(train_config, dict) or not isinstance(state, dict):
        raise ValueError("C checkpoint is missing its training budget or state")
    target_tokens = train_config.get("target_tokens")
    completed_tokens = state.get("target_tokens")
    if (type(target_tokens) is not int or target_tokens < 1
            or type(completed_tokens) is not int or completed_tokens < target_tokens):
        raise ValueError("C has not reached its declared supervised-target training budget")
    status = load_json_strict(run_dir / "training_status.json")
    current_hash = sha256_file(checkpoint)
    if payload.get("_loaded_checkpoint_sha256", current_hash) != current_hash:
        raise ValueError("Completed checkpoint changed after weights were loaded")
    if (not isinstance(status, dict) or status.get("architecture") != "C"
            or status.get("status") != "complete"
            or status.get("target_tokens") != target_tokens
            or status.get("state") != state
            or status.get("code_sha256") != metadata.get("code_sha256")
            or status.get("checkpoint_sha256") != current_hash):
        raise ValueError("C completion record does not match the checkpoint and declared budget")


def _decode_config(config, policy):
    choices = config["decoding"]
    if not isinstance(choices, list) or not 0 <= policy < len(choices):
        raise ValueError("Selected C decoding policy is missing")
    try:
        decode = GenerationConfig(**choices[policy])
    except (TypeError, ValueError):
        raise ValueError("Selected C decoding policy is invalid") from None
    if decode.target_words != 128:
        raise ValueError("Official continuations must contain exactly 128 words")
    return decode


def _evaluate(args):
    data = load_prepared(args.data)
    model, payload, _, metadata, device = _model_from_checkpoint(
        args.checkpoint, data, args.device,
    )
    tokenizer = load_tokenizer(data.root / "tokenizer.json")
    prompts = load_json_strict(data.root / f"{args.split}.prompts.json")
    refs = load_json_strict(data.root / f"{args.split}.references.json")
    if not isinstance(prompts, dict) or not isinstance(refs, dict) or set(prompts) != set(refs):
        raise ValueError("C evaluation prompts and references must have the same IDs")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Evaluation limit must be positive")
    keys = list(prompts)[:args.limit] if args.limit else list(prompts)
    if not keys:
        raise ValueError("No evaluation IDs")
    output = Path(args.output)
    decode = _decode_config(_config(args.config), args.policy)
    declared_prompts = {key: prompts[key]["text"] for key in keys}
    fingerprint = {"checkpoint": payload["_loaded_checkpoint_sha256"], "tokenizer": data.tokenizer_sha256,
                   "code": metadata["code_sha256"], "policy": asdict(decode), "split": args.split}
    predictions = _generate_records(output, declared_prompts, model, tokenizer, decode, fingerprint)
    write_json_atomic(output, {key: {"text": value} for key, value in predictions.items()})
    reference_path = output.with_suffix(".references.json")
    write_json_atomic(reference_path, {key: refs[key] for key in keys})
    metrics = score_predictions(reference_path, output, EVALUATOR, args.scorer_cache)
    per_source = {}  # Aggregate published metrics; source scoring can be requested separately.
    repetitions = {key: 1 - len(set(value.split())) / 128 for key, value in predictions.items()}
    report = {
        "architecture": "C", "checkpoint_sha256": payload["_loaded_checkpoint_sha256"],
        "checkpoint_state": payload["state"], "code_sha256": metadata["code_sha256"],
        "split": args.split, "limited_sample": bool(args.limit),
        "synthetic_smoke": metadata["synthetic_smoke"],
        "decoding": asdict(decode), "metrics": metrics, "per_source": per_source,
        "repeated_word_fraction": repetitions, "declared_ids": keys,
    }
    write_json_atomic(output.with_suffix(".context.json"), report)
    print(json.dumps(report, allow_nan=False))


def _predict_official_test(args):
    output = Path(args.output)
    if output.suffix.lower() != ".zip":
        raise ValueError("Official predictions must be packaged to a .zip output")
    predictions_path = output.with_name(output.stem + ".predictions.json")

    data = load_prepared(args.data)
    manifest = load_json_strict(data.root / "manifest.json")
    model, payload, run_dir, metadata, device = _model_from_checkpoint(
        args.checkpoint, data, args.device,
    )
    _require_complete_training(args.checkpoint, payload, run_dir, metadata)
    revision = _official_dataset_revision(manifest)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data.root.parent / "hf_hub"
    test_path = _download_official_test(revision, cache_dir)
    prompts = extract_official_test_prompts(load_json_strict(test_path))
    tokenizer = load_tokenizer(data.root / "tokenizer.json")
    decode = _decode_config(_config(args.config), args.policy)

    fingerprint = {"checkpoint": payload["_loaded_checkpoint_sha256"], "tokenizer": data.tokenizer_sha256,
                   "code": metadata["code_sha256"], "policy": asdict(decode),
                   "test": sha256_file(test_path), "dataset_revision": revision}
    predictions = _generate_records(output, prompts, model, tokenizer, decode, fingerprint)

    validate_predictions(predictions, prompts)
    write_json_atomic(predictions_path, {key: {"text": value}
                                         for key, value in predictions.items()})
    report = {
        "architecture": "C", "checkpoint_sha256": payload["_loaded_checkpoint_sha256"],
        "checkpoint_state": payload["state"], "checkpoint_run_dir": str(run_dir),
        "code_sha256": metadata["code_sha256"],
        "data_manifest_sha256": data.manifest_sha256,
        "tokenizer_sha256": data.tokenizer_sha256,
        "dataset_revision": revision, "test_sha256": sha256_file(test_path),
        "generation_config": asdict(decode), "synthetic_smoke": False,
        "prediction_count": len(predictions),
    }
    report = _finalize_official_export(output, predictions, prompts, report)
    print(json.dumps(report, allow_nan=False))


def _train_or_benchmark(args, benchmark=False):
    config = _config(args.config)
    model_cfg = RecurrentConfig(**config["model"])
    train_values = dict(config["train"])
    configured_budget = train_values.pop("target_tokens", None)
    target_tokens = args.target_tokens if args.target_tokens is not None else configured_budget
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("A positive supervised-target budget must be configured explicitly")
    train_cfg = TrainConfig(target_tokens=target_tokens, **train_values)
    if args.device:
        train_cfg = replace(train_cfg, device=args.device)
    if benchmark:
        if (Path(args.run_dir) / "metrics.jsonl").exists():
            raise FileExistsError("C benchmark run directory must be fresh")
        train_cfg = replace(train_cfg, target_tokens=10**15, warmup_ratio=0,
                            log_every=1, save_every=125)
    state = train_c(
        load_prepared(args.data), model_cfg, train_cfg, Path(args.run_dir), args.resume,
        stop_after_steps=125 if benchmark else args.stop_after_steps,
        smoke_test=args.smoke_test,
    )
    if benchmark:
        rows = [json.loads(line) for line in
                (Path(args.run_dir) / "metrics.jsonl").read_text().splitlines()]
        if len(rows) < 125:
            raise ValueError("C benchmark requires 125 logged steps")
        first, last = rows[24], rows[124]
        seconds = (last["gpu_hours_this_session"] - first["gpu_hours_this_session"]) * 3600
        if seconds <= 0:
            raise ValueError("C benchmark requires CUDA timing")
        from .io import write_json_atomic
        write_json_atomic(Path(args.run_dir) / "benchmark.json", {
            "warmup_steps": 25, "measured_steps": 100, "seconds": seconds,
            "input_tokens_per_second": (last["input_tokens"] - first["input_tokens"]) / seconds,
            "target_tokens_per_second": (last["target_tokens"] - first["target_tokens"]) / seconds,
            "peak_memory_bytes": last["peak_memory_bytes"], "state": asdict(state),
        })
    print(json.dumps(asdict(state), allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="CS781 architecture C tools")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--target-tokens", type=int)
        command.add_argument("--config", default=str(DEFAULT_CONFIG))
        command.add_argument("--device")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--smoke-test", action="store_true")
        command.add_argument("--stop-after-steps", type=int)
    evaluate = sub.add_parser("evaluate")
    for field in ("data", "checkpoint", "output", "scorer-cache"):
        evaluate.add_argument("--" + field, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--config", default=str(DEFAULT_CONFIG))
    evaluate.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    evaluate.add_argument("--policy", type=int, default=0)
    evaluate.add_argument("--limit", type=int)
    official = sub.add_parser("predict-official-test")
    for field in ("data", "checkpoint", "output"):
        official.add_argument("--" + field, required=True)
    official.add_argument("--device", default="cuda")
    official.add_argument("--config", default=str(DEFAULT_CONFIG))
    official.add_argument("--policy", type=int, default=0)
    official.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    try:
        if args.command in ("train", "benchmark"):
            _train_or_benchmark(args, args.command == "benchmark")
        elif args.command in ("evaluate", "predict-official-test"):
            with run_lock(Path(args.output).with_suffix(".generation-lock")):
                if args.command == "evaluate":
                    _evaluate(args)
                else:
                    _predict_official_test(args)
        return 0
    except (RuntimeError, ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        print(f"C {args.command} failed ({type(exc).__name__}): {exc}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

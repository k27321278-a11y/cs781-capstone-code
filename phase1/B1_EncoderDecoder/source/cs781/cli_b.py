"""Train, evaluate and package architecture B without changing A's active CLI."""
import argparse
import json
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .contracts import GenerationConfig, TrainConfig
from .evaluate import score_predictions
from .generate_b import generate_continuation_b
from .io import sha256_file, write_json_atomic
from .models.b import EncoderDecoderConfig, EncoderDecoderLegalModel
from .prepare import load_prepared
from .cli import (
    _checkpoint_run_directory,
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
from .train_b import _code_digest, train_b


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs/b.json"
EVALUATOR = Path(__file__).parents[1] / "vendor/phase1_evaluate.py"
PINNED_OFFICIAL_TEST_SHA256 = "614f348b3dcbb2a9f43706cd239438386dc396246cd5cc685a5ccb6eb06b7218"


def _config(path):
    value = load_json_strict(path)
    if not isinstance(value, dict) or not all(k in value for k in ("model", "train", "decoding")):
        raise ValueError("B config must define model, train and decoding sections")
    return value


def _model_from_checkpoint(checkpoint_path, data, device_name="cuda"):
    checkpoint, run_dir, run_metadata = _checkpoint_run_directory(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("B checkpoint payload or provenance is malformed")
    metadata = payload["metadata"]
    manifest = load_json_strict(data.root / "manifest.json")
    if metadata != run_metadata or metadata.get("architecture") != "B":
        raise ValueError("Checkpoint is not a matching architecture-B project run")
    if (metadata.get("synthetic_smoke") is not False or manifest.get("synthetic") is not False
            or metadata.get("data_manifest_sha256") != data.manifest_sha256
            or metadata.get("tokenizer_sha256") != data.tokenizer_sha256):
        raise ValueError("B checkpoint is synthetic or has mismatched official data/tokenizer provenance")
    if (re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("code_sha256", ""))) is None
            or metadata["code_sha256"] != _code_digest()):
        raise ValueError("B checkpoint source code differs from the code used for training")
    count = metadata.get("unique_parameters")
    if type(count) is not int or not 50_000_000 <= count <= 300_000_000:
        raise ValueError("B checkpoint parameter count is outside the foundation range")
    try:
        cfg = EncoderDecoderConfig(**metadata["model_config"])
        cfg.validate()
    except (KeyError, TypeError, ValueError):
        raise ValueError("B checkpoint model configuration is invalid") from None
    if cfg.max_seq_len != manifest.get("max_seq_len"):
        raise ValueError("B checkpoint context differs from prepared-data context")
    if _official_dataset_revision(manifest) is None:
        raise ValueError("B run is not tied to the pinned official corpus")
    if not isinstance(payload.get("model"), dict) or not isinstance(payload.get("state"), dict):
        raise ValueError("B checkpoint is missing model weights or training state")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for B inference but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError("Requested B CUDA device is outside the visible-device set")
    model = EncoderDecoderLegalModel(cfg).to(device)
    if model.count_parameters() != count:
        raise ValueError("Instantiated B parameter count differs from checkpoint provenance")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload, run_dir, metadata, device


def _require_complete_training(checkpoint, payload, run_dir, metadata):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint != (run_dir / "checkpoints/latest.pt").resolve():
        raise ValueError("Official prediction requires the completed run's latest checkpoint")
    if metadata.get("synthetic_smoke") is not False:
        raise ValueError("Official prediction requires substantive, non-synthetic training")
    train_config = metadata.get("train_config")
    state = payload.get("state")
    if not isinstance(train_config, dict) or not isinstance(state, dict):
        raise ValueError("B checkpoint is missing its training budget or state")
    target_tokens = train_config.get("target_tokens")
    completed_tokens = state.get("target_tokens")
    if (type(target_tokens) is not int or target_tokens < 1
            or type(completed_tokens) is not int or completed_tokens < target_tokens):
        raise ValueError("B has not reached its declared supervised-target training budget")
    status = load_json_strict(run_dir / "training_status.json")
    if (not isinstance(status, dict) or status.get("architecture") != "B"
            or status.get("status") != "complete"
            or status.get("target_tokens") != target_tokens
            or status.get("state") != state
            or status.get("code_sha256") != metadata.get("code_sha256")
            or status.get("checkpoint_sha256") != sha256_file(checkpoint)):
        raise ValueError("B completion record does not match the checkpoint and declared budget")


def _decode_config(config, policy):
    choices = config["decoding"]
    if not isinstance(choices, list) or not 0 <= policy < len(choices):
        raise ValueError("Selected B decoding policy is missing")
    try:
        decode = GenerationConfig(**choices[policy])
    except (TypeError, ValueError):
        raise ValueError("Selected B decoding policy is invalid") from None
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
        raise ValueError("B evaluation prompts and references must have the same IDs")
    keys = list(prompts)[:args.limit] if args.limit else list(prompts)
    if not keys:
        raise ValueError("No evaluation IDs")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    decode = _decode_config(_config(args.config), args.policy)
    predictions, failures = {}, {}
    for key in keys:
        try:
            predictions[key] = generate_continuation_b(
                model, tokenizer, prompts[key]["text"], decode,
            )
        except (RuntimeError, ValueError) as exc:
            failures[key] = type(exc).__name__
    write_json_atomic(output.with_suffix(".failures.json"), failures)
    if failures:
        raise ValueError(f"B generation failed for {len(failures)} evaluation IDs")
    validate_predictions(predictions, {key: prompts[key]["text"] for key in keys})
    write_json_atomic(output, {key: {"text": value} for key, value in predictions.items()})
    reference_path = output.with_suffix(".references.json")
    write_json_atomic(reference_path, {key: refs[key] for key in keys})
    metrics = score_predictions(reference_path, output, EVALUATOR, args.scorer_cache)
    sources = {}
    for key in keys:
        sources.setdefault(key.split(":", 1)[0], []).append(key)
    per_source = {}
    for source, source_ids in sources.items():
        source_output = output.with_name(output.stem + "." + source + ".json")
        source_reference = source_output.with_suffix(".references.json")
        write_json_atomic(source_output, {key: {"text": predictions[key]} for key in source_ids})
        write_json_atomic(source_reference, {key: refs[key] for key in source_ids})
        per_source[source] = score_predictions(
            source_reference, source_output, EVALUATOR, args.scorer_cache,
        )
    repetitions = {key: 1 - len(set(value.split())) / 128 for key, value in predictions.items()}
    report = {
        "architecture": "B", "checkpoint_sha256": sha256_file(args.checkpoint),
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
    context_path = output.with_name(output.stem + ".context.json")
    failures_path = output.with_name(output.stem + ".failures.json")
    for path in (output, predictions_path, context_path, failures_path):
        if path.exists():
            raise FileExistsError(path)

    data = load_prepared(args.data)
    manifest = load_json_strict(data.root / "manifest.json")
    model, payload, run_dir, metadata, device = _model_from_checkpoint(
        args.checkpoint, data, args.device,
    )
    _require_complete_training(args.checkpoint, payload, run_dir, metadata)
    revision = _official_dataset_revision(manifest)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data.root.parent / "hf_hub"
    test_path = Path(args.test_file) if args.test_file else _download_official_test(revision, cache_dir)
    if not test_path.is_file() or sha256_file(test_path) != PINNED_OFFICIAL_TEST_SHA256:
        raise ValueError("Official test file does not match the pinned course revision")
    prompts = extract_official_test_prompts(load_json_strict(test_path))
    tokenizer = load_tokenizer(data.root / "tokenizer.json")
    decode = _decode_config(_config(args.config), args.policy)

    predictions, failures = {}, {}
    for index, (key, prompt) in enumerate(prompts.items(), start=1):
        try:
            predictions[key] = generate_continuation_b(model, tokenizer, prompt, decode)
        except (RuntimeError, ValueError) as exc:
            failures[key] = type(exc).__name__
        if index % 25 == 0 or index == len(prompts):
            print(json.dumps({"phase": "B official-test generation",
                              "completed": index, "total": len(prompts),
                              "failures": len(failures)}), flush=True)
    if failures:
        write_json_atomic(failures_path, {
            "dataset_revision": revision, "test_sha256": sha256_file(test_path),
            "failures": failures,
        })
        raise ValueError(f"B generation failed for {len(failures)} official IDs; no partial ZIP created")

    validate_predictions(predictions, prompts)
    write_json_atomic(predictions_path, {key: {"text": value}
                                         for key, value in predictions.items()})
    zip_sha = write_submission(predictions, prompts, output)
    report = {
        "architecture": "B", "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_state": payload["state"], "checkpoint_run_dir": str(run_dir),
        "code_sha256": metadata["code_sha256"],
        "data_manifest_sha256": data.manifest_sha256,
        "tokenizer_sha256": data.tokenizer_sha256,
        "dataset_revision": revision, "test_sha256": sha256_file(test_path),
        "generation_config": asdict(decode), "synthetic_smoke": False,
        "prediction_count": len(predictions), "prediction_zip_sha256": zip_sha,
    }
    write_json_atomic(context_path, report)
    print(json.dumps(report, allow_nan=False))


def _train_or_benchmark(args, benchmark=False):
    config = _config(args.config)
    model_cfg = EncoderDecoderConfig(**config["model"])
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
            raise FileExistsError("B benchmark run directory must be fresh")
        train_cfg = replace(train_cfg, target_tokens=10**15, warmup_ratio=0,
                            log_every=1, save_every=125)
    state = train_b(
        load_prepared(args.data), model_cfg, train_cfg, Path(args.run_dir), args.resume,
        stop_after_steps=125 if benchmark else args.stop_after_steps,
        smoke_test=args.smoke_test,
    )
    if benchmark:
        rows = [json.loads(line) for line in
                (Path(args.run_dir) / "metrics.jsonl").read_text().splitlines()]
        if len(rows) < 125:
            raise ValueError("B benchmark requires 125 logged steps")
        first, last = rows[24], rows[124]
        seconds = (last["gpu_hours_this_session"] - first["gpu_hours_this_session"]) * 3600
        if seconds <= 0:
            raise ValueError("B benchmark requires CUDA timing")
        from .io import write_json_atomic
        write_json_atomic(Path(args.run_dir) / "benchmark.json", {
            "warmup_steps": 25, "measured_steps": 100, "seconds": seconds,
            "input_tokens_per_second": (last["input_tokens"] - first["input_tokens"]) / seconds,
            "target_tokens_per_second": (last["target_tokens"] - first["target_tokens"]) / seconds,
            "peak_memory_bytes": last["peak_memory_bytes"], "state": asdict(state),
        })
    print(json.dumps(asdict(state), allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="CS781 architecture B tools")
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
    official.add_argument("--test-file")
    args = parser.parse_args(argv)
    try:
        if args.command in ("train", "benchmark"):
            _train_or_benchmark(args, args.command == "benchmark")
        elif args.command == "evaluate":
            _evaluate(args)
        elif args.command == "predict-official-test":
            _predict_official_test(args)
        return 0
    except (RuntimeError, ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        print(f"B {args.command} failed ({type(exc).__name__}); inspect completed manifests and run artifacts",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Token-budgeted from-scratch trainer for architecture C."""
import contextlib
import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .checkpoints import restore_checkpoint, run_lock, save_checkpoint
from .contracts import TrainState
from .examples_alt import collate_b
from .io import sha256_file, write_json_atomic
from .models.c import RecurrentConfig, RecurrentLegalModel
from .prepare import load_prepared
from .tokenizer import load_tokenizer
from .train import IndexedExamples


def _code_digest():
    root = Path(__file__).parent
    names = (
        "train_c.py", "examples_alt.py", "models/c.py", "models/b.py", "models/a.py",
        "generate_c.py", "cli_c.py", "prediction_progress.py", "cli.py", "evaluate.py",
        "submission.py", "contracts.py", "checkpoints.py", "train.py", "examples.py",
        "prepare.py", "tokenizer.py", "own_bpe.py", "io.py",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _save_candidate(path, model, metadata, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as handle:
            torch.save({"model": model.state_dict(), "metadata": metadata,
                        "state": asdict(state)}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_c(data, model_cfg, train_cfg, run_dir, resume=False, *, stop_after_steps=None, smoke_test=False,
            candidate_every=2000, milestone_tokens=(209914685,)):
    if type(candidate_every) is not int or candidate_every < 1 or any(type(v) is not int or v < 1 for v in milestone_tokens):
        raise ValueError("Positive C candidate interval and milestones required")
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError("Positive stopping step required")
    if min(train_cfg.target_tokens, train_cfg.micro_batch_size, train_cfg.grad_accum_steps,
           train_cfg.save_every, train_cfg.log_every) < 1:
        raise ValueError("Positive training budgets, batches and intervals required")
    if (not math.isfinite(train_cfg.learning_rate) or train_cfg.learning_rate <= 0
            or not math.isfinite(train_cfg.min_lr_ratio)
            or not 0 <= train_cfg.min_lr_ratio <= 1
            or not math.isfinite(train_cfg.warmup_ratio)
            or not 0 <= train_cfg.warmup_ratio < 1
            or not math.isfinite(train_cfg.weight_decay) or train_cfg.weight_decay < 0
            or not 0 < train_cfg.beta1 < 1 or not 0 < train_cfg.beta2 < 1
            or not math.isfinite(train_cfg.eps) or train_cfg.eps <= 0
            or not math.isfinite(train_cfg.grad_clip) or train_cfg.grad_clip <= 0):
        raise ValueError("Invalid C optimizer or learning-rate configuration")
    model_cfg.validate()
    data = load_prepared(data.root if hasattr(data, "root") else data)
    manifest = json.loads((data.root / "manifest.json").read_text())
    if manifest["synthetic"] and not smoke_test:
        raise ValueError("Synthetic data is for explicitly labeled smoke tests only")
    if model_cfg.max_seq_len != manifest["max_seq_len"]:
        raise ValueError("Model and prepared training context differ")

    tokenizer = load_tokenizer(data.root / "tokenizer.json")
    ids = {name: tokenizer.token_to_id(token) for name, token in
           (("pad_id", "<pad>"), ("bos_id", "<bos>"), ("eos_id", "<eos>"))}
    if (any(type(value) is not int or value < 0 for value in ids.values())
            or len(set(ids.values())) != len(ids)
            or tokenizer.get_vocab_size() > model_cfg.vocab_size):
        raise ValueError("Tokenizer special IDs or vocabulary do not fit model configuration")
    if model_cfg.pad_token_id != ids["pad_id"]:
        model_cfg = replace(model_cfg, pad_token_id=ids["pad_id"])
        model_cfg.validate()

    run_dir = Path(run_dir)
    with run_lock(run_dir):
        checkpoint_path = run_dir / "checkpoints/latest.pt"
        if not resume and checkpoint_path.exists():
            raise FileExistsError("Existing checkpoint: explicitly resume or use a new run")
        if resume and not checkpoint_path.is_file():
            raise FileNotFoundError("Requested resume without a checkpoint")

        random.seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)
        torch.manual_seed(train_cfg.seed)
        device = torch.device(train_cfg.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for C but is unavailable")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError("Requested C CUDA device is outside the visible-device set")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(train_cfg.seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = RecurrentLegalModel(model_cfg).to(device)
        parameter_count = model.count_parameters()
        if not smoke_test and not 50_000_000 <= parameter_count <= 300_000_000:
            raise ValueError("Substantive foundation must have 50M–300M unique parameters")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps,
            weight_decay=train_cfg.weight_decay,
        )
        metadata = {
            "architecture": "C",
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "data_manifest_sha256": data.manifest_sha256,
            "tokenizer_sha256": data.tokenizer_sha256,
            "code_sha256": _code_digest(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "synthetic_smoke": bool(smoke_test),
            "unique_parameters": parameter_count,
            "snapshot_schedule": {"candidate_every": candidate_every, "milestone_tokens": list(milestone_tokens)},
            "recurrent_precision": "float32",
        }
        state = TrainState()
        if resume:
            state, sampler = restore_checkpoint(checkpoint_path, model, optimizer, metadata)
            if sampler["cursor"] != state.example_cursor:
                raise ValueError("Sampler/checkpoint cursor mismatch")
        write_json_atomic(run_dir / "run.json", metadata)
        write_json_atomic(run_dir / "training_status.json", {
            "architecture": "C", "status": "running",
            "target_tokens": train_cfg.target_tokens,
            "state": asdict(state), "code_sha256": metadata["code_sha256"],
        })

        dataset = IndexedExamples(data.root)
        if len(dataset.offsets) == 0:
            raise ValueError("Prepared training split has no examples")
        began = time.monotonic()
        initial_target = state.target_tokens
        model.train()
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
            while state.target_tokens < train_cfg.target_tokens and (
                stop_after_steps is None or state.step < stop_after_steps
            ):
                batches, step_input, step_target = [], 0, 0
                cursor = state.example_cursor
                source_counts = {}
                for _ in range(train_cfg.grad_accum_steps):
                    examples = [dataset.at(cursor + i, train_cfg.seed)
                                for i in range(train_cfg.micro_batch_size)]
                    cursor += len(examples)
                    for example in examples:
                        source_counts[example.source] = source_counts.get(example.source, 0) + 1
                    batch = collate_b(examples, **ids, max_seq_len=model_cfg.max_seq_len)
                    batches.append(batch)
                    step_input += batch.input_tokens
                    step_target += batch.target_tokens

                fraction = state.target_tokens / train_cfg.target_tokens
                if fraction < train_cfg.warmup_ratio:
                    scale = max(1 / train_cfg.target_tokens,
                                fraction / max(train_cfg.warmup_ratio, 1e-12))
                else:
                    progress = (fraction - train_cfg.warmup_ratio) / max(
                        1 - train_cfg.warmup_ratio, 1e-12,
                    )
                    scale = train_cfg.min_lr_ratio + (1 - train_cfg.min_lr_ratio) * 0.5 * (
                        1 + math.cos(math.pi * min(1, progress))
                    )
                for group in optimizer.param_groups:
                    group["lr"] = train_cfg.learning_rate * scale
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for batch in batches:
                    prompt_ids = batch.prompt_ids.to(device)
                    prompt_mask = batch.prompt_mask.to(device)
                    decoder_ids = batch.decoder_ids.to(device)
                    targets = batch.targets.to(device)
                    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                                if device.type == "cuda" else contextlib.nullcontext())
                    with autocast:
                        _, loss, _ = model(prompt_ids, decoder_ids, targets, prompt_mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite C loss; previous checkpoint preserved")
                    (loss * batch.target_tokens / step_target).backward()
                    step_loss += loss.item() * batch.target_tokens / step_target

                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip, error_if_nonfinite=True,
                )
                optimizer.step()
                previous_target = state.target_tokens
                state = replace(
                    state, step=state.step + 1,
                    input_tokens=state.input_tokens + step_input,
                    target_tokens=state.target_tokens + step_target,
                    example_cursor=cursor,
                )
                if state.step % train_cfg.log_every == 0:
                    elapsed = time.monotonic() - began
                    row = {
                        **asdict(state), "loss": step_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": float(norm),
                        "gpu_hours_this_session": elapsed / 3600 if device.type == "cuda" else 0,
                        "target_tokens_per_second":
                            (state.target_tokens - initial_target) / elapsed,
                        "peak_memory_bytes":
                            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        "source_examples_this_step": source_counts,
                    }
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    write_json_atomic(run_dir / "training_status.json", {
                        "architecture": "C", "status": "running",
                        "target_tokens": train_cfg.target_tokens, "state": asdict(state),
                        "code_sha256": metadata["code_sha256"],
                        "target_tokens_per_second": row["target_tokens_per_second"],
                    })
                if state.step % candidate_every == 0:
                    _save_candidate(run_dir / f"candidates/step_{state.step:07d}.pt", model, metadata, state)
                for milestone in milestone_tokens:
                    if previous_target < milestone <= state.target_tokens:
                        _save_candidate(run_dir / f"candidates/milestone_{milestone}.pt", model, metadata, state)
                # Publish due snapshots before advancing the resumable checkpoint.
                # An interruption then replays this step instead of skipping them.
                if state.step % train_cfg.save_every == 0:
                    save_checkpoint(checkpoint_path, model, optimizer, state,
                                    {"cursor": cursor}, metadata)
            save_checkpoint(checkpoint_path, model, optimizer, state,
                            {"cursor": state.example_cursor}, metadata)
        completed = state.target_tokens >= train_cfg.target_tokens
        write_json_atomic(run_dir / "training_status.json", {
            "architecture": "C",
            "status": ("complete" if completed and not smoke_test else
                       "smoke_complete" if completed else "incomplete"),
            "target_tokens": train_cfg.target_tokens,
            "state": asdict(state), "code_sha256": metadata["code_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        return state

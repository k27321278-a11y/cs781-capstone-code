"""Task-aligned continuation of our own from-scratch architecture B."""
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
from .models.b import EncoderDecoderConfig, EncoderDecoderLegalModel
from .prepare import load_prepared
from .tokenizer import load_tokenizer
from .train import IndexedExamples
from .b_upgrade import initialization, upgrade_digest


def _code_digest():
    return upgrade_digest()


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


def train_b_improved(data, model_cfg, train_cfg, run_dir, parent_checkpoint, original_data, resume=False, *, stop_after_steps=None, smoke_test=False):
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
        raise ValueError("Invalid B optimizer or learning-rate configuration")
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
                raise RuntimeError("CUDA was requested for B but is unavailable")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError("Requested B CUDA device is outside the visible-device set")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(train_cfg.seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = EncoderDecoderLegalModel(model_cfg).to(device)
        parameter_count = model.count_parameters()
        if not smoke_test and not 50_000_000 <= parameter_count <= 300_000_000:
            raise ValueError("Substantive foundation must have 50Mâ€“300M unique parameters")
        initial_weights, lineage = initialization(parent_checkpoint, original_data, model_cfg)
        if not resume:
            model.load_state_dict(initial_weights, strict=True)
        del initial_weights
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps,
            weight_decay=train_cfg.weight_decay,
        )
        metadata = {
            "architecture": "B",
            "variant": "B_taskaligned",
            "initialization": lineage,
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "data_manifest_sha256": data.manifest_sha256,
            "tokenizer_sha256": data.tokenizer_sha256,
            "code_sha256": _code_digest(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "synthetic_smoke": bool(smoke_test),
            "unique_parameters": parameter_count,
        }
        state = TrainState()
        if resume:
            state, sampler = restore_checkpoint(checkpoint_path, model, optimizer, metadata)
            if sampler["cursor"] != state.example_cursor:
                raise ValueError("Sampler/checkpoint cursor mismatch")
        write_json_atomic(run_dir / "run.json", metadata)
        write_json_atomic(run_dir / "training_status.json", {
            "architecture": "B", "status": "running",
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
                        raise FloatingPointError("Nonfinite B loss; previous checkpoint preserved")
                    (loss * batch.target_tokens / step_target).backward()
                    step_loss += loss.item() * batch.target_tokens / step_target

                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip, error_if_nonfinite=True,
                )
                optimizer.step()
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
                        "architecture": "B", "status": "running", "target_tokens": train_cfg.target_tokens,
                        "state": asdict(state), "code_sha256": metadata["code_sha256"],
                    })
                milestone = train_cfg.target_tokens // 2
                crossed = state.target_tokens - step_target < milestone <= state.target_tokens
                if crossed:
                    _save_candidate(run_dir / f"candidates/milestone_{milestone}.pt", model, metadata, state)
                if state.step % train_cfg.save_every == 0 or crossed:
                    save_checkpoint(checkpoint_path, model, optimizer, state,
                                    {"cursor": cursor}, metadata)
            save_checkpoint(checkpoint_path, model, optimizer, state,
                            {"cursor": state.example_cursor}, metadata)
        completed = state.target_tokens >= train_cfg.target_tokens
        write_json_atomic(run_dir / "training_status.json", {
            "architecture": "B",
            "status": ("complete" if completed and not smoke_test else
                       "smoke_complete" if completed else "incomplete"),
            "target_tokens": train_cfg.target_tokens,
            "state": asdict(state), "code_sha256": metadata["code_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        return state

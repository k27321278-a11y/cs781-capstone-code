# C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, train, evaluate and export C before proceeding to D and E.

**Architecture:** Ten-layer bidirectional encoder, four-layer GRU, fixed prompt cross-attention and SwiGLU fusion; tied embedding/output. Reuse data and scoring contracts.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, existing private dependency overlays, RTX 4080.

**Spec:** `docs/superpowers/specs/2026-10-03-c-design.md`

## Global Constraints

- Official four sources only; no pretrained participant weights or external teacher.
- 512 prompt words and exactly 128 generated words; never truncate/fill.
- 50M–300M unique parameters; preserve A/B and protected stopped jobs.
- Training context 2048, prompt capacity 8192; finite GPU backward before launch.
- Initial budget 419829370 supervised target tokens; comparable milestone 209914685.

## Review Focus

- Recurrent sequence versus cached single-step equivalence and hidden-state dtype.
- Long prompts, padded prompt batches, completely masked prompts.
- Future suffix leakage and ignored labels.
- Atomic checkpoint/resume cursor/RNG and budget provenance.
- Interrupted generation, missing scorer cache, partial/duplicate IDs.

### Task 1: C model

Files: `cs781/models/c.py`, `tests/test_arch_c.py`.
Interface: `RecurrentLegalModel.forward(prompt_ids, decoder_ids, targets=None, prompt_mask=None, past_kvs=None, use_cache=False) -> (logits, loss, cache)`; config `RecurrentConfig`.

- [ ] Add failing tests for absent implementation, causality, pad invariance, cache equivalence, gradient flow, loss masking and parameter range.
- [ ] Observe failure remotely; implement approved graph; pass CPU suite and actual-GPU backward/cache check.

### Task 2: C trainer and generation

Files: `cs781/train_c.py`, `cs781/generate_c.py`, `configs/c.json`, `tests/test_train_c.py`.
Consumes Task 1 model and established batch/checkpoint contracts. Produces `train_c(...) -> TrainState`, `generate_continuation_c(...) -> str` and provenance records.

- [ ] Add failing tests for deterministic resume, completed-budget record and sparse snapshots; implement trainer and verify.
- [ ] Reuse B's proven decoding contract with C's prompt capacity; test long prompt behavior and no fabricated output.
- [ ] Run finite real-data 125-step GPU benchmark; save throughput/memory and kernel evidence.

### Task 3: CLI, inference progress and scoring

Files: `cs781/cli_c.py`, `cs781/prediction_progress.py`, `tests/test_prediction_progress.py`, `scripts/run_c_gpu0.sh`.
Consumes C training metadata/model and generation function. Produces resumable dev/confirmation/official predictions and validated ZIP.

- [ ] Add failing tests for atomic progress resume and mismatched prompt/checkpoint/policy rejection; implement and pass.
- [ ] Add train, benchmark, evaluate and predict-official-test commands with complete provenance gates; verify actual small end-to-end flow.
- [ ] Request one independent whole-change review and resolve important findings with regression tests.
- [ ] Launch substantive C detached after checking no duplicate; verify live PID, GPU assignment and increasing metrics.

### Task 4: C measured deliverables, then D/E

- [ ] Verify C budget/checkpoint completion; compare four dev policies, choose measured best; evaluate confirmation.
- [ ] Generate/validate all official IDs and transfer ZIP, source and results report.
- [ ] Continue full objective with D, then E, preserving each distinct graph and measuring their results against B and the 0.84 target.

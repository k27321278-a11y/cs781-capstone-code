# Reproduction & TA Audit Checklist — Phase 1 A0

This checklist is a recordkeeping instrument. **No box is pre-checked on behalf of the student.** It distinguishes static code audit, tiny synthetic test, actual full-size training and external grading.

## A. Code/version provenance

- [ ] The public GitHub repository URL is recorded in the final course report.
- [ ] All A0 original implementation and launcher files are committed and the Git commit SHA is recorded.
- [ ] `sha256sum code/architecture_a_legalgpt.py` matches `docs/SOURCE_MANIFEST.md`.
- [ ] Python, torch, CUDA, GPU driver and tokenizer/datasets/hub library versions are recorded from the actual run.
- [ ] Hugging Face dataset exact revision SHA for all source streams is recorded; no implicit mutable `main` change.
- [ ] Actual run `run_config.json` is archived **before** eval/submit commands overwrite it; saved checkpoint configs are captured.
- [ ] A0 model parameters measured from actual constructed checkpoint match expected nominal 110,119,680 or actual-vocab adjustment.

## B. Tokenizer and dataset

- [ ] Tokenizer trained only on train-split documents, not hidden test or held-out validation.
- [ ] Fitted `tokenizer/tokenizer.json` and `tokenizer_meta.json` are archived with checksums.
- [ ] Actual `vocab_size_actual`, number of BPE merges, data character count and documents seen are recorded.
- [ ] `data/data_meta.json` records actual 200M / 5M cache token counts and source distribution.
- [ ] `train_tokens.bin` and `val_tokens.bin` checksums recorded; zero-length and insufficient-length caches ruled out.
- [ ] Document identity collisions and corpus duplications, if any, are disclosed.
- [ ] Sample encodings round-trip by `tokenizer.decode(tokenizer.encode(text).ids)` to acceptable text.

## C. Code correctness

- [x] **Local tiny built-in source `--mode selftest`** passed during this documentation audit (2-layer toy forward, finite CE, KV cache grows from 10 to 11); original source unchanged.
- [ ] Nominal full model `LegalGPT(ModelConfig()).count_parameters()` run on target environment.
- [ ] Shapes verified for all attention Q/K/V, RoPE, RMSNorm, SwiGLU and output logits with target precision.
- [ ] Causal mask prevents future-token leakage, including 2,048-token windows.
- [ ] Cross-entropy includes shifted labels as specified and is averaged correctly over accumulation steps.
- [ ] DDP (if used) global token count and gradient synchronization flags validated.
- [ ] Selected SDPA kernel recorded using appropriate profiler, not inferred from API name.

## D. Actual training evidence

- [ ] Training log from all steps archived.
- [ ] Successful optimizer-step count and processed-token count recorded.
- [ ] Train NLL/perplexity progression plotted from logs, with axes and checkpoints.
- [ ] Validation NLL/perplexity progression plotted, indicating selected best step.
- [ ] `latest.pt`/`best.pt` actual files present and sha256 checksums archived.
- [ ] No claim that the full 200M token budget was completed unless logs/metadata prove it.
- [ ] Measured CUDA peak memory and training runtime recorded if reported.

## E. Held-out generation and official test

- [ ] `data/val_continuations.json` has intended number of 512/128-word windows.
- [ ] Evaluation demonstrates 128 whitespace-word outputs for all evaluated samples.
- [ ] Default greedy, repetition penalty, and optional top-k/nucleus configurations clearly distinguished.
- [ ] `evaluation/metrics_greedy.json` and generated text/reference outputs archived if actually evaluated.
- [ ] BERTScore is claimed only with an authoritative official scorer result or a documented permitted local evaluation.
- [ ] Codabench `submission.zip` contains only `prediction.json`; all test IDs present and every result exactly 128 words.
- [ ] Actual Codabench submission ID, timestamp and leaderboard scores preserved if available.

## F. Final course submission compatibility

- [ ] Only the **best-performing** Phase 1 model's code is selected for official final submission (A0 comparison evidence required if alternatives exist).
- [ ] `README.md` explains all phases and includes this A0 source-exact guide.
- [ ] GitHub repository is public, contains reproducible code, figures and method docs, and its URL appears in final report.
- [ ] Final report uses the supplied LaTeX template, exports PDF, and includes an accurate LLM usage declaration.
- [ ] Final Hello IITK ZIP has `code/` and `report/` folders directly at root without an outer wrapping folder.
- [ ] ZIP filename matches the exact staff-prescribed pattern.
- [ ] Upload to Hello IITK is confirmed; merely creating a local ZIP does not submit it.

## G. Distinguish external artifacts

**Codabench output:** `submission.zip` containing `prediction.json` only, used to score generated continuations.

**Hello IITK report/code output:** a different ZIP whose root contains `code/` and `report/`, used for final course marking.

**Public GitHub repository:** a separate hosted version-controlled record; a local ZIP or local Git commit does not establish publication.

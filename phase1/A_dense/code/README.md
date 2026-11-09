# CS781 Phase 1: shared pipeline and A

Four-family design: ../docs/superpowers/specs/2026-10-02-cs781-phase1-design.md. This release implements **A**, not B/C/D. No leaderboard improvement is claimed before substantive training and held-out official evaluation.

A is a from-scratch 110,119,680-parameter causal Transformer: 12×768, 12 heads, RoPE, pre-RMSNorm, SwiGLU2048, tied 32768 embeddings. Flash forward/backward, prefill and cached decode were observed in strict-backend tests on RTX4080/Torch2.5.1+cu121. The tests are not a trained foundation.

## Paths and isolation

Code: `/home/stud1/cs781_phase1_v2`. New artifacts: `/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2`. This alternate HDD parent was verified owned/writable by stud1; no old FeelWM/A0 artifact is edited. Preserve old tokenizer/logs; provenance is insufficient for reuse with the new cluster split, so a new common BPE is fitted.

Server interpreter remains the existing A virtualenv; private dependency overlays `.test-deps` and `.runtime` belong only to this release. Do not install/upgrade Torch globally. Authentication uses the user's cached HF login; never paste, read or print token files. Local relay and reverse tunnel must stay alive while downloading; their loss aborts preparation, never changes sources or starts training.

## Explicit workflow

From the code directory, use `PYTHONPATH=.runtime:.test-deps:.` with the server interpreter. Commands are shown as `python` for brevity; set `PYTHON_BIN` to the existing interpreter in scripts.

1. Prepare in a new named artifact root (example):
   ```bash
   python -u -m cs781.cli prepare --root /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/official_100m_chars_per_source --revision 89edba6a44d6bb9593fd13b6895b5cd3b06eb82e --max-chars-per-source 100000000
   ```
   Ceiling applies per source, keeping whole final documents; actual selected counts/characters and revision are recorded. It is a subset, not the full 19.2GB corpus. Dataset source/revision metadata was inspected on 3 October IST. The current official test file is `Phase_1_Pretraining/Test/test.json`; do not use it for BPE, training or mixture selection.
2. `python -m cs781.cli validate-data --data <root>/prepared` must succeed. Manifest, cache/index hashes and nonempty four-source train/dev/confirmation data gate training. Exact/SimHash duplicate-cluster split precedes BPE; BPE character budgets are source-balanced and training-only. Near-duplicate detection is heuristic and needs an audited sample.
3. Benchmark using a **new** run directory: `python -m cs781.cli benchmark --data <root>/prepared --run-dir <artifact>/A_benchmark --target-tokens 1`. It records 100 measured optimizer steps after 25 startup steps; no substantive training cap is implied. Inspect peak VRAM, target/input throughput and context coverage before setting A–D budgets.
4. After live process/GPU inspection, launch detached with an explicit supervised-suffix-token budget:
   ```bash
   nohup bash scripts/run_a.sh <root>/prepared <artifact>/A_main 0 <measured_target_budget> > <artifact>/A_main.log 2>&1 < /dev/null &
   ```
   Monitor `tail -f <artifact>/A_main/metrics.jsonl`. Resume with the identical config/data and append `--resume`; first verify no surviving trainer. The Linux run lock rejects duplicates. Checkpoints restore optimizer, RNG and sampling cursor at completed accumulation boundaries. Input tokens and supervised suffix+EOS tokens are separate counters.
5. Evaluate saved candidates with `python -m cs781.cli evaluate --data <root>/prepared --checkpoint <candidate.pt> --output <new-prediction.json> --policy 0`. Compare policies0/1/2 and candidates on dev by official BERTScore. Keep confirmation for one check after selection; report per-source metrics, repetition and all generation failures. Limited integration samples are labeled, never filtered after seeing failed IDs.
6. Generate official-test predictions using the selected model/policy and the same generation contract, then validate/package via `python -m cs781.cli submission --predictions <prediction.json> --prompts <id-text-prompt.json> --output <new-submission.zip>`. Current CLI evaluation targets held-out references; a test-prediction command will be added with the verified official-test schema before actual submission. Upload only substantive trained-model outputs, never synthetic smoke results. ZIP contains only root `prediction.json`; preserve string IDs including leading zeros. Record evaluated submission status/ID/scores, not just upload success.

Initial context2048 is not permission to truncate 512-word prompts. Token-length report and generation failures may require a larger RoPE/context limit plus re-profiled memory/Flash tests. Generation waits for a complete word boundary before trimming to128; cap failures raise errors, never add filler.

## Verification and attribution

`python -m pytest tests -m 'not gpu and not integration' -q`; GPU: `python -m pytest tests/test_flash.py -m gpu -q`. Official evaluator is preserved byte-for-byte in vendor/ with source/hash. BERT scoring encoder is for the authorized metric only, never participant weights. Record its package versions and downloaded model snapshot separately.

Citations: Transformer (1706.03762), FlashAttention (2205.14135), RoPE (2104.09864), SwiGLU (2002.05202), RMSNorm (1910.07467), BERTScore (1904.09675). Proposed B/C/D papers and novelty hypotheses are in the approved spec. LLM declaration: architecture/code assistance was provided by Codex under the instructor permission reported by the user; the student must understand and verify the implementation and results.

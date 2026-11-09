# SDD ledger — plan: docs/superpowers/plans/2026-10-03-c-implementation.md

Baseline: isolated source snapshot from `/home/stud1/cs781_phase1_v2` to `/home/stud1/cs781_phase1_cde_v3`; 73 passed, 1 deselected (CPU suite). Live 2026-10-03 16:52 UTC: both GPUs idle; GPU0 14050 MiB free, GPU1 11731 MiB free; HDD 78 GiB free. No C/D/E implementation or trainers yet.

Ruling: execute inline under user's explicit design/code/train continuation approval and previously approved architecture specification — avoid asking for the same workflow authorisation again — refinements are recorded in the written C spec.

Ruling: use the already isolated source directories without creating another worktree — original remote snapshot has no Git repository — preserve source backups and local change history.

Ruling: maintain this plain ledger instead of invoking a Bash workspace helper from Windows — avoids the known shell/path incompatibility — same task/evidence records are retained.

Pre-flight: Task 1 forward/cache interface matches trainer/generator consumers in Task 2. Task 2 metadata/state feed Task 3 checkpoint and completion gates. Task 3 validated predictions feed Task 4 scoring/export. No interface conflicts found.

B diagnostics: exact archived ZIP `b_phase1_5pass_20261003_submission_policy0_20261003T062709Z.zip` contains root `prediction.json`. 1000 records, 956 unique texts, mean 27.829 distinct words/128. Existing scorer model snapshot and provenance are present. Online Codabench worker cache failure is separate from our local scoring cache.

Task 1: complete — 10 C tests pass, including actual CUDA backward, future-suffix perturbation, padded-summary invariance, cached recurrence and tiny overfit. Default model has 117805824 unique parameters.

Task 2: training/generation implementation verified — deterministic split/resume weights agree with uninterrupted CPU run; changed budget is rejected by CheckpointMismatch; synthetic completion is labelled smoke_complete; candidates are sparse. Initial test expected the wrong exception superclass; corrected to the established CheckpointMismatch contract. No production resume defect observed.

Task 3: CLI and saved-progress implementation verified — interruptions retain completed records and reject mixed prompt/checkpoint/policy provenance; partial export is blocked. Candidate regex regression observed (2 failing tests), corrected, and rerun. Full remote suite: 92 passed in 7.37s.

Default GPU profile on physical GPU1: prompt 702 tokens, decoder 1347 tokens, finite loss/backward, 1352362496 peak allocated bytes; BF16 cached logits match full recurrence (maximum absolute difference 0.015625, tolerance 0.025). cuDNN GRU and memory-efficient SDPA dispatched. FlashAttention did not dispatch for this masked C profile; do not claim it did. A's prior actual Flash evidence remains separate.

Real-data C benchmark launched physical GPU0, Python PID 626553. Live step59, 1039162 supervised tokens, finite loss6.766, ~7154 target tokens/s, peak2876354560bytes. This is a sizing benchmark, not substantive completed training or a leaderboard result.

Independent whole-change reviewer dispatched once; findings pending. No per-task agent fan-out.

Review outcome: three Important issues reproduced with failing tests, then fixed. Loaded checkpoint and SHA256 now use the same open descriptor; official ZIP finalization resumes after verifying existing ZIP content; due candidate/milestone snapshots publish before the resumable latest checkpoint. Added permanent CUDA/BF16 cache comparison. Final full remote suite: 96 passed in 7.94s. No remaining Important/Critical reviewer finding.

Benchmark terminal evidence: Python PID626553 absent, benchmark.json exists, 125 steps and 2223990 supervised tokens. 100 measured steps after25warmup took246.445s; throughput7202.516target tokens/s and14373.646input tokens/s; peak2876354560bytes. Fresh ten-pass target419829370 projects16.19hours at this measured rate, excluding checkpoint/evaluation overhead. Benchmark source preserved at `/home/stud1/cs781_c_benchmark_20261003_source` before review fixes.

End-to-end engineering gate: benchmark checkpoint generated two dev continuations of exactly128words without failures, and the unchanged evaluator successfully scored them using the pinned offline cache. Saved `c_benchmark_dev_smoke_20261003.metrics.json` and context record. This tiny, minimally trained check is not a model-quality comparison or a leaderboard result.

Protected-process check: all seven original protected PIDs are still T/Tl. GPU0 returned to14050MiBfree; GPU1returned to11731MiBfree. No A/B trainer, no duplicate C trainer. Ready for substantive C launch.

Substantive launch verified: trainer PID627627, physical GPU0, fresh weights, declared419829370target tokens, run `/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/c_phase1_10pass_20261003`. Training source digest `cd79bc27fa72c97af27fb73771a97e8281f694a3291fa8ecdc8df8d435be6020` is frozen. Live 17:42UTC: step410, 7366045target tokens, loss4.9493, measured6852target tokens/s; atomic latest1.4GiB exists. All training code edits ceased before this run launched.

Completion orchestration added outside the frozen training digest. It waits for exact PID627627, requires the completed ten-pass parent, compares the five-pass milestone and final checkpoint across four predeclared policies on full dev, freezes the selected policy/checkpoint, evaluates confirmation, then generates all official IDs and validated ZIP. Selected earlier own checkpoint can be exported only after parent completion and >=209914685trainedtargets. Reports disclose both selected and final parent states. This is checkpoint selection, not a claim that an earlier candidate trained ten passes.

Completion delta review found two issues: mutable selection across retries and a five-pass parent accepted by the selected-candidate gate. Both reproduced and fixed; full suite102passed in7.96s, including an actual restart orchestration regression and negative latest/candidate five-pass gates. Test script import uses an explicit file path because an unrelated user-site `scripts` package shadowed the namespace; application execution uses the direct script file.

Finalizer PID628745 verified alive, phase waiting_for_training, exact target trainer627627. Controller log `/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/c_finalizer_20261003.log`. It does not restart/kill jobs. Final artifact target is `CS781_Model_C_Codabench_Submission_20261003.zip`; it does not exist yet.

Original shared chat now readable through browser at https://chatgpt.com/share/6abfcb02-5b5c-83e8-90b8-0db40f76352b. Read its requirements, architecture reasoning, experimental protocol and final RTX-A workflow. Its early B/C naming predates the explicit approved handoff (B encoder-decoder, C GRU, D hierarchical, E PrefixLM). Current user's C→D→E order and later approved graphs are authoritative. Original chat reiterates from-scratch50–300M, official-only training,512/128, >=4differentfoundations, measuredcheckpoint/decodingselection and preservation for laterphases.

Ruling: compare C's five-pass milestone against B's trained-target budget, and disclose C's ten-pass cosine schedule — same consumed token count does not make the learning-rate schedules identical — avoid attributing any eventual gain exclusively to architecture.

Remaining full goal: C training and measured scores/export are pending; D and E remain unimplemented and follow C. The >=0.84 objective remains unproven. No goal completion claim.

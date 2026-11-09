# A legal-span implementation plan and ledger

Design: `outputs/CS781_A_Novelty_Proposal_20261004.md` (approved by the current explicit execution goal).
The new goal authorizes implementation, meaningful verification, remote deployment, substantive training and automatic ZIP preparation. No additional design approval is pending.

Isolation: separate local `work/cs781_a_legalspan_20261004`, remote `/home/stud1/cs781_a_legalspan_v1`, HDD `.../cs781_phase1_v2/a_legalspan_v1`. Original A/B/C sources, tokenizer IDs, checkpoints and seven paused processes are protected.

1. Own BPE (`legalspan/bpe.py`, `tests/test_own_bpe.py`): reversible Unicode-class segmentation, weighted incremental pair learner, deterministic tie breaking, ranked encoding, byte offsets, atomic save, train-only provenance. Compare independent tiny reference, overlaps and Unicode round trips. Start train-only CPU fitting after these pass.
2. Data (`legalspan/data.py`, tests): retain existing catalog split and all held-out IDs; reencode with own BPE. Deterministic equal mix of exact 512/128 word windows and conditional random token windows; prompt roles derived solely from observed text. Manifest, byte offsets, context checks, source accounting, no cross-document windows.
3. Model (`legalspan/model.py`, tests): inherited manually coded RMS/RoPE/SwiGLU dense backbone; upper four blocks have prompt-only 128-wide attention route, role bias, longest 1–4 token continuity bias, learned residual gate and normalized pointer/vocabulary mixture. Parallel teacher-forced route; independent cached generation. Verify causality, boundary, cache parity, repeated-ID scatter, finite gradients, no eligible copies and disabled-route control.
4. Training/generation (`legalspan/train.py`, `generate.py`, tests): fresh initialization, frozen code/data/tokenizer metadata, atomic resume, finite checks, target-token accounting, bounded snapshot retention, strict 128-word continuation generation. Actual GPU1 long-context/Flash benchmark and small train→generate→published scorer→ZIP verification.
5. Orchestration (`legalspan/pipeline.py`, `scripts/run_a_legalspan.sh`, tests): exclusive supervisor; matched pilot control/generic/full; substantive matched control and full candidate each 209,914,685 targets; dev checkpoint/policy selection, frozen confirmation, official 1,000-ID generation and automatic strict ZIP. Persist status/failure/provenance; no fabricated output. Final independent review before substantive launch.
6. Completion: observed completed runs/checkpoints, controlled measurements/ablations, actual valid ZIP and local/server hashes. Goal stays active while real jobs run. No guaranteed leaderboard gain; score claims require measured evidence.

## Ledger

- 2026-10-04 00:56 UTC: fresh SSH check; GPU1 11,731 MiB free and 0% utilization; C PID627627 running on GPU0; all seven protected jobs stopped. HDD 71 GiB free.
- Copied reusable source to a separate directory. The enclosing host repository is C:/.git; do not modify that unrelated repository. Use the separate source snapshot and hashes instead of a host-wide worktree.
- Tokenizer tests are the first implementation gate; CPU fit can run while model code is developed.

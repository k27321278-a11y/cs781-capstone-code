# Final review resolution and verified launch gates

The independent raw report is retained in FINAL_REVIEW.md. Implementation fixes were reproduced with failing tests and then verified on the server. No second reviewer was dispatched.

- Generic/removal modes now zero continuity evidence before every gate; gate-sensitive regression passes.
- Random-token roles now use only observed prefix bytes, including bisected cue/UTF8 handling. A separate prepared_v2 cache preserves every ID/token/window and recomputes roles. The original prepared cache is evidence only and rejected by the current loader.
- Due candidates are atomically published before advancing latest.pt. An interruption injected after the real latest save verifies the candidate is already present.
- All three matched pilots receive full-dev screening before substantive training. A predeclared repetition screen runs separately on control when >25% of outputs exceed10% repeated4grams. Pilots remain engineering-only; subsequent policies are shared across models.
- Route softmax explicitly accumulates float32. Actual CUDA/BF16 normalized mixture and multiple-token cache parity pass for full and generic. CPU cache probabilities and reduced-memory loss/gradients match independently verified full paths.
- All1000 actual pinned public prompts roundtrip with own BPE:1052–1457 tokens including BOS, plus2048 generation reserve fits8192. No truncated or missing IDs.
- Current full suite:128 passed in10.78s. Full112095887-parameter8192-token GPU1 optimizer-step profile captures Flash forward/backward; allocated peak2277625856bytes.
- Corrected-source synthetic engineering pipeline trains real random weights, generates actual128words, runs the published pinned scorer and creates/validates a ZIP. It is not a quality or leaderboard result and cannot be exported as an official foundation.

Training source and inference source digests are frozen in experiment/run metadata. Changes require a new versioned run. All seven older stopped processes and active C on GPU0 remain protected. Substantive training, full held-out comparisons and the actual1000-ID deliverable still need to finish; the active goal is not complete.

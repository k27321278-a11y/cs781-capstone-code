# D222 execution ledger

- User approved the architecture proposal, requested a score-focused cross-check, then approved the revised plan for direct implementation.
- Ruling: new efficient model starts from random weights; it is not declared a leaderboard improvement before held-out metrics. Costs if wrong: additional pretraining without outperforming the smaller prior models.
- Ruling: Wikipedia is disabled in the default run because the pinned FAQ and live document conflict. Costs if wrong: omitted generic-data ablation; official-data run remains clearly identified.
- Ruling: pretraining initially selects checkpoints by fixed held-out LM loss; official BERTScore/ROUGE/BLEU selection runs after time-sensitive NER exports. Costs if wrong: the NER base may not be the best continuation-scoring checkpoint; it is labeled preliminary CE selection.
- Ruling: no new unit-test suite added. The user-approved runtime gates check causal/cache/parameter/gradient and exact export invariants before launch, separately from quality evaluation.
- Earlier Phase 2 four-model workflow completed and all four ZIPs/complete archives verified locally. Its heartbeat is paused; new D work is isolated.
- Implemented sparse GQA decoder, verified token-cache loader, atomic resumable LM training, held-out loss/candidate selection, original-character NER/CRF/boundary head, training-only rare-label sampling, bounded NER candidate promotion, both export contracts, GPU1 guard and downloader.
- Notebook: 17 cells; JSON and all Python code-cell syntax parsed. Runtime preflight on GPU1 is underway. No substantive D training launched yet.
- Independent code review requested before launch. Any reported score is a measured local result; private Codabench scores are not invented.

- Independent review found five material issues; fixed all before launch: per-candidate NER budgets/export allowance, fingerprint checks on completed-run and generation shortcuts, explicit finalization deadline checks/post-deadline benchmark flag, fully covered overlength LM validation, and current preflight lineage binding.
- Ruling: learned CRF DP uses CPU with differentiable GPU copies to reduce sequential kernel overhead; backbone/head and optimizer stay on the selected GPU. Cost if wrong: CPU throughput becomes limiting; actual 1600-character runtime is recorded.
- Ruling: an authorized Phase 1 local benchmark can continue after the Phase 2 deadline; NER delivery is prioritized and no closed-phase upload occurs. Cost if wrong: additional GPU hours for a benchmark that may not beat the earlier model.

- Corrected runtime preflight passed: exact 222,079,872 parameters; causal/cache checks (max cached error 1.192e-7); finite real gradients; efficient masked and Flash SDPA kernels observed; CRF 1600-character finite-gradient path measured 0.133 seconds; one dev document has 5954 tokens and is now fully covered rather than rejected.
- Launched detached GPU1 queue PID 153218; first observed real updates step 30 / 358,957 input and supervised target tokens, loss 4.9256, finite gradient norm 8.3822. Calibration 9224.7 tokens/s, peak allocated memory 4,958,018,048 bytes; effective target retained at 300M supervised tokens. This is progress, not completion or accuracy evidence.
- Local hidden downloader PID 10192 verified alive with no startup error. Selected tokens and final metrics remain pending.

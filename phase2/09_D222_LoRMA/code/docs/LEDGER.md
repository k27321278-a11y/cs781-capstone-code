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

# Phase 7: prior case retrieval

`source/phase7_run.py` trains/evaluates the lexical and neural retrieval variants; `source/phase7_export.py` creates corrected submission exports. `evidence/Phase7_PCR_all_models_report.json` is a local run record, not a Codabench result. The final corrected method excludes the query itself and emits seven unique candidate IDs per test query.

The corrected model ensemble and corrected full model tied at **0.4064 Codabench micro-F1@7** on 7 November 2026. Corrected lexical fallback scored 0.4022, partial 0.4028, KD 0.4034, and LoRMA 0.3969. The earlier selected ensemble's local 0.4158 on 118 dev queries predates the self-match correction; do not cite it as a corrected local score.

The repository does not include large checkpoints or course documents. Original run and export records are stored in the named local Phase 7 work folder.

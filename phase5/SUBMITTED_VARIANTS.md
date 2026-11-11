# Phase 5: Legal Statute Identification — scored submission map

The table records **distinct scored variants**, not duplicate uploads. The score column is finished Codabench macro-F1 from the supplied screenshots. Original training code, an archived runner notebook where available, evidence, a later illustrated notebook, a detailed README, a technical reference, and PDF/LaTeX methodology notes live in each linked folder.

| Variant | Kind | Codabench macro-F1 | Finished submission ID(s) | Exact local ZIP |
|---|---|---:|---|---|
| [B1 LoRA statute classifier](variants/b1_lora/README.md) | trained model | 0.2554 | 1016271 | present |
| [B1 LoRMA+ statute classifier](variants/b1_lorma/README.md) | trained model | 0.2542 | 1016282 | present |
| [B1 plus full-document sparse hybrid](variants/b1_sparse_hybrid/README.md) | two-branch prediction system | 0.3949 | 1016346, 1016350 | present |
| [B1 sparse hybrid with calibration CV](variants/b1_sparse_hybrid_cv/README.md) | calibration variant | 0.3944 | 1016805 | present |
| [B1 sparse-guided passage selection](variants/b1_sparse_guided/README.md) | trained model | 0.2417 | 1016593 | present |
| [B1 statute-wise GRU memory](variants/b1_statute_gru/README.md) | trained model | 0.2238 | 1016621 | present |
| [B1 position-aware statute queries](variants/b1_position_queries/README.md) | trained model | 0.2254 | 1016786 | present |

The neural branches, full-text sparse hybrid, and calibration-CV export are distinct experiments. The hybrid's course-method eligibility is not asserted by the saved report.

Original checkpoint weights and official course data remain in their separate archives and are not copied into this public repository. A saved local dev or calibration value is never labelled a Codabench score. Each package records original source SHA-256 and makes the reconstructed nature of its explanatory notebook explicit.

[View the supplied Codabench screenshot evidence](evidence/README.md).

[Read the architecture and experiment progression](ARCHITECTURE_AND_EXPERIMENTS.md).

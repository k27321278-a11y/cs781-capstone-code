# Phase 7: Prior Case Retrieval — scored submission map

The table records **distinct scored variants**, not duplicate uploads. The score column is finished Codabench micro-F1@7 from the supplied screenshots. Original training code, an archived runner notebook where available, evidence, a later illustrated notebook, a detailed README, a technical reference, and PDF/LaTeX methodology notes live in each linked folder.

| Variant | Kind | Codabench micro-F1@7 | Finished submission ID(s) | Exact local ZIP |
|---|---|---:|---|---|
| [Initial lexical fallback](variants/lexical_fallback_initial/README.md) | retrieval baseline | 0.3680 | 1039286 | missing locally |
| [Lexical ranking excluding query itself](variants/lexical_excluded_self/README.md) | retrieval correction | 0.4017 | 1039896 | present |
| [Corrected lexical fallback](variants/lexical_fallback_corrected/README.md) | retrieval correction | 0.4022 | 1039899, 1039901 | present |
| [Full fine-tuned neural retriever](variants/full_corrected/README.md) | trained retriever | 0.4064 | 1039892 | present |
| [Partial fine-tuned neural retriever](variants/partial_corrected/README.md) | trained retriever | 0.4028 | 1039890 | present |
| [LoRMA neural retriever](variants/lorma_corrected/README.md) | trained retriever | 0.3969 | 1039891 | present |
| [Distilled retrieval student](variants/kd_corrected/README.md) | trained retriever | 0.4034 | 1039888 | present |
| [Reselected neural-lexical ensemble](variants/ensemble_reselected/README.md) | ensemble export | 0.4017 | 1039894 | present |
| [Corrected neural-lexical model ensemble](variants/model_ensemble_corrected/README.md) | ensemble export | 0.4064 | 1039898, 1039902 | present |

Full, partial, LoRMA, and KD are model-training routes; lexical and ensemble entries are ranking/export systems. The initial lexical ZIP was not found locally. Corrected exports enforce self-exclusion.

Original checkpoint weights and official course data remain in their separate archives and are not copied into this public repository. A saved local dev or calibration value is never labelled a Codabench score. Each package records original source SHA-256 and makes the reconstructed nature of its explanatory notebook explicit.

[View the supplied Codabench screenshot evidence](evidence/README.md).

[Read the architecture and experiment progression](ARCHITECTURE_AND_EXPERIMENTS.md).

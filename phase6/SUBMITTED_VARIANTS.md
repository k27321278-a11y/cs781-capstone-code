# Phase 6: Abstractive Legal Summarization — scored submission map

The table records **distinct scored variants**, not duplicate uploads. The score column is finished Codabench BERTScore F1 from the supplied screenshots. Original training code, an archived runner notebook where available, evidence, a later illustrated notebook, a detailed README, a technical reference, and PDF/LaTeX methodology notes live in each linked folder.

| Variant | Kind | Codabench BERTScore F1 | Finished submission ID(s) | Exact local ZIP |
|---|---|---:|---|---|
| [B1 passage fusion with LoRMA+](variants/b1_fusion/README.md) | trained summarizer | 0.7674 | 1030742, 1030744, 1030983 | present |
| [B1 fusion with 300-word output cap](variants/b1_fusion_length/README.md) | decoding variant | 0.7783 | 1030821 | present |
| [C1 Houlsby bottleneck summarizer](variants/c1_houlsby/README.md) | trained summarizer | 0.7700 | 1030778, 1030779, 1030987 | present |
| [C1 Houlsby with 400-word output cap](variants/c1_houlsby_length/README.md) | decoding variant | 0.7750 | 1030822 | present |
| [B1 AdaLoRA summarizer](variants/b1_adalora/README.md) | trained summarizer | 0.7656 | 1030985 | present |
| [C1 Houlsby continued checkpoint](variants/c1_continue/README.md) | continued training variant | 0.7671 | 1030988 | present |
| [D222 prefix-tuned summarizer](variants/d222_prefix/README.md) | trained summarizer | 0.7710 | 1031005 | present |
| [LegalSpan LoRA summarizer](variants/legalspan_lora/README.md) | trained summarizer | 0.7847 | 1031076, 1031077, 1031078 | present |

The 300/400-word caps are decoding variants of existing models, and C1 continuation is a later checkpoint of C1 Houlsby. They are not new architectures.

Original checkpoint weights and official course data remain in their separate archives and are not copied into this public repository. A saved local dev or calibration value is never labelled a Codabench score. Each package records original source SHA-256 and makes the reconstructed nature of its explanatory notebook explicit.

[View the supplied Codabench screenshot evidence](evidence/README.md).

[Read the architecture and experiment progression](ARCHITECTURE_AND_EXPERIMENTS.md).

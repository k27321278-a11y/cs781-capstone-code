# Phase 6: abstractive legal summarization

The available source under `source/` and notebooks under `notebooks/` cover LegalSpan LoRA/coverage, B1 LoRMA, B1 AdaLoRA, B1 hierarchical fusion, C1 Houlsby, D222 prefix tuning, and A0 sequence-level distillation. The notebook source is separate from the historical Kaggle execution logs and checkpoints.

The highest finished Codabench submission visible on 9 November 2026 is LegalSpan LoRA, **0.7847 BERTScore F1** (`phase6_legalspan_lora_prediction.zip`, 3 November). B1 fusion with length control reached 0.7783; C1 length candidate reached 0.7750; D222 prefix reached 0.7710. Local HIF probes on 16 examples are not evidence of a higher hidden score. The main report explains the architecture and why semantic-overlap metrics do not guarantee legal correctness.

Reproduction needs the selected in-house Phase 1 checkpoint, tokenizer, course datasets, exact configs, and saved adaptation weights. They are not duplicated here.

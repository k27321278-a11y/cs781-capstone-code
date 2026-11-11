# CS781 Phase 1–7: Source-led methodology audit

**Audit date: 12 November 2026.** Repository: https://github.com/k27321278-a11y/cs781-capstone-code

## Executive finding

The public repository contains **at least 61 curated, indexed model/variant/export packages** in the audited package indexes. This is not necessarily a census of every unindexed experiment or notebook version in the source tree. These are not 61 independent deep-network architectures: stacks, consensus, calibration and output-only variants are counted as variants. Among them, **57 have methodology PDFs**, four Phase 3 prepared variants have **no methodology PDF**, and **only the historical A0 reference is currently demonstrated at the unusually detailed A0 walkthrough standard**. The other existing PDFs are useful but source inspections of representative packages reveal much shorter documentation. An additional source notebook that is not indexed as a scored/package variant is outside this 61-target count. Every variant should be individually reviewed before a claim of A0-level completeness.

The original pretrained checkpoints and official datasets are not in this public repository. The 7 Nov 2026 Discord screenshot explicitly requests a public code repository, detailed README, model/method/execution documentation, final report code ZIP, and LLM declaration; it does **not** expressly require all original checkpoint weights in the public repository. The whole private Discord history is not accessible for exhaustive confirmation.

| Phase | Indexed variants | Existing methodology PDFs | No methodology PDF |
|---:|---:|---:|---:|
| 1 | 8 | 8 | 0 |
| 2 | 12 | 12 | 0 |
| 3 | 4 | 0 | 4 |
| 4 | 13 | 13 | 0 |
| 5 | 7 | 7 | 0 |
| 6 | 8 | 8 | 0 |
| 7 | 9 | 9 | 0 |
| **Total** | **61** | **57** | **4** |

## Per-variant inventory

**Status codes:** `A0-depth reference` = historical A0 source-audited detailed walkthrough; `shorter pre-existing` = a methodology PDF exists, but depth not certified as matching A0; `missing, prepared` = no completed/scored Phase 3 model or methodology PDF in repository at audit time.

| Phase | Variant | Existing report/location | Assessment |
|---:|---|---|---|
| 1 | `A_dense (A0)` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/A_dense/historical_A0_methodology/report/A0_Methodology_Report.pdf) | **A0-depth reference** |
| 1 | `A0_LegalSpan` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/A0_LegalSpan/report/A0_LegalSpan_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 1 | `B0_EncoderDecoder` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/B0_EncoderDecoder/report/B0_EncoderDecoder_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 1 | `B_Improved` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/B_Improved/report/B_Improved_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 1 | `B1_EncoderDecoder` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/B1_EncoderDecoder/report/B1_EncoderDecoder_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 1 | `C0_EncoderGRU` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/C0_EncoderGRU/report/C0_EncoderGRU_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 1 | `C1_EncoderGRU` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase1/C1_EncoderGRU/report/C1_EncoderGRU_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 1 | `D222_LocalGlobal (extension)` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/extensions/D222_LocalGlobal/report/D222_LocalGlobal_Methodology_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `01_LegalSpan_LoRMA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/01_LegalSpan_LoRMA/report/01_LegalSpan_LoRMA_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `02_B1_LoRA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/02_B1_LoRA/report/02_B1_LoRA_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `03_C1_Full` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/03_C1_Full/report/03_C1_Full_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `04_A0_KD` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/04_A0_KD/report/04_A0_KD_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `05_Consensus_2of4` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/05_Consensus_2of4/report/05_Consensus_2of4_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `06_LegalSpan_HeaderCRF` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/06_LegalSpan_HeaderCRF/report/06_LegalSpan_HeaderCRF_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `07_LegalSpan_RoleSpan` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/07_LegalSpan_RoleSpan/report/07_LegalSpan_RoleSpan_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `08_D222_Full` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/08_D222_Full/report/08_D222_Full_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `09_D222_LoRMA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/09_D222_LoRMA/report/09_D222_LoRMA_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `10_D222_Continued_LoRMA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/10_D222_Continued_LoRMA/report/10_D222_Continued_LoRMA_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `11_D222_100M_LoRMA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/11_D222_100M_LoRMA/report/11_D222_100M_LoRMA_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 2 | `12_D222_200M_LoRMA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase2/12_D222_200M_LoRMA/report/12_D222_200M_LoRMA_Technical_Report.pdf) | Existing; deeper line-by-line verification required |
| 3 | `01_D222_LoRMA` | No existing report | Prepared only, no verified run/score |
| 3 | `02_B1_LoRA` | No existing report | Prepared only, no verified run/score |
| 3 | `03_C1_IA3` | No existing report | Prepared only, no verified run/score |
| 3 | `04_A0_Distillation` | No existing report | Prepared only, no verified run/score |
| 4 | `01_D222_LoRMA` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/01_D222_LoRMA/report/) | Existing; deeper line-by-line verification required |
| 4 | `02_B1_Partial` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/02_B1_Partial/report/) | Existing; deeper line-by-line verification required |
| 4 | `03_C1_Full` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/03_C1_Full/report/) | Existing; deeper line-by-line verification required |
| 4 | `04_A0_Distillation` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/04_A0_Distillation/report/) | Existing; deeper line-by-line verification required |
| 4 | `05_D222_Adapters` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/05_D222_Adapters/report/) | Existing; deeper line-by-line verification required |
| 4 | `06_C1_BitFit` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/06_C1_BitFit/report/) | Existing; deeper line-by-line verification required |
| 4 | `07_B1_Reference` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/07_B1_Reference/report/) | Existing; deeper line-by-line verification required |
| 4 | `08_B1_Hierarchy` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/08_B1_Hierarchy/report/) | Existing; deeper line-by-line verification required |
| 4 | `09_B1_GRUOnly` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/09_B1_GRUOnly/report/) | Existing; deeper line-by-line verification required |
| 4 | `10_B1_TransformerOnly` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/10_B1_TransformerOnly/report/) | Existing; deeper line-by-line verification required |
| 4 | `11_B1_Stack4` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/11_B1_Stack4/report/) | Existing; deeper line-by-line verification required |
| 4 | `12_B1_Stack6` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/12_B1_Stack6/report/) | Existing; deeper line-by-line verification required |
| 4 | `13_B1_Stack6_CVCal` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase4/model_packages/13_B1_Stack6_CVCal/report/) | Existing; deeper line-by-line verification required |
| 5 | `b1_lora` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_lora/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 5 | `b1_lorma` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_lorma/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 5 | `b1_position_queries` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_position_queries/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 5 | `b1_sparse_guided` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_sparse_guided/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 5 | `b1_sparse_hybrid` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_sparse_hybrid/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 5 | `b1_sparse_hybrid_cv` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_sparse_hybrid_cv/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 5 | `b1_statute_gru` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase5/variants/b1_statute_gru/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `b1_adalora` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/b1_adalora/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `b1_fusion` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/b1_fusion/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `b1_fusion_length` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/b1_fusion_length/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `c1_continue` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/c1_continue/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `c1_houlsby` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/c1_houlsby/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `c1_houlsby_length` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/c1_houlsby_length/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `d222_prefix` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/d222_prefix/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 6 | `legalspan_lora` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase6/variants/legalspan_lora/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `ensemble_reselected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/ensemble_reselected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `full_corrected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/full_corrected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `kd_corrected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/kd_corrected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `lexical_excluded_self` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/lexical_excluded_self/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `lexical_fallback_corrected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/lexical_fallback_corrected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `lexical_fallback_initial` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/lexical_fallback_initial/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `lorma_corrected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/lorma_corrected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `model_ensemble_corrected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/model_ensemble_corrected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |
| 7 | `partial_corrected` | [existing source/report](https://github.com/k27321278-a11y/cs781-capstone-code/blob/main/phase7/variants/partial_corrected/report/METHODOLOGY_REPORT.pdf) | Existing; deeper line-by-line verification required |

## Standard for a truly A0-level methodology report

1. Trace every concrete input and its provenance, split, sampling, packing, normalization and tokenizer configuration, with the original code and line numbers.
2. Give mathematical equations with all symbols defined, every relevant tensor dimension, masking convention, weight tying and parameter count measured or derived from source.
3. Map the forward and backward pass, optimizer/precision/accumulation schedule, exact checkpoint selection and reproducible execution commands.
4. Explain the actual output decoder, thresholds, submission JSON and validity checks. Distinguish local from official scores.
5. Include honest observed failure modes, train/validation plots only if available, memory accounting (analytical vs measured), alternatives and limitations.
6. Preserve original source unmodified and record source hashes; do not invent checkpoints or completed experiments.
7. Present clear standalone diagrams and explanatory prose so a TA can understand and independently reimplement the exact method.

## Access and provenance boundaries

- Public GitHub **read** access and source tree checked. A **write operation failed with HTTP 403** `Resource not accessible by integration`. Thus this document/package is local and **has not been pushed to GitHub**.
- The source `phase3_rr` was recovered in the supplied finalized code ZIP; original Phase 3 notebook configurations were inspected on public GitHub. All four were prepared/unscored.
- Variant-specific original checkpoint tensors were not available for inspection; theoretical architecture and preserved scripts are distinct from executed-run verification.
- The additional Phase 3 reports delivered with this inventory are **new local documentation drafts**, not measured model results.
- No original implementation file was modified.

## How to publish without changing model code

After ensuring your GitHub connection has **Contents: Read and write** permission, copy the new methodology files into the matching `phase3/` paths in a normal Git commit, or upload via the GitHub web UI. The failed write attempt here did not change `main`.

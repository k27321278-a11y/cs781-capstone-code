# CS781 Phase 2 source-audited architecture packages

This collection follows the detailed A0 explanation style: architecture diagram, line-level code map, technical reference, companion notebook, report, run evidence, and original prediction export. The companions and reports were assembled after the original training runs and contain no fabricated executed outputs.

**Status vocabulary:** ‘export’ means a locally verified prediction ZIP, not proof of Codabench upload or hidden score. The last two D222 checkpoint studies finished after the original Phase 2 deadline.

| No. | Export / method | Local held-out strict macro-F1 | Timing | Package |
|---:|---|---:|---|---|
| 01 | LegalSpan with LoRMA+ | 0.411641 | Pre-deadline export | `CS781_Phase2_01_LegalSpan_LoRMA_Reproducibility.zip` |
| 02 | B1 encoder-decoder with LoRA | 0.386275 | Pre-deadline export | `CS781_Phase2_02_B1_LoRA_Reproducibility.zip` |
| 03 | C1 recurrent hybrid, full tuning | 0.357207 | Pre-deadline export | `CS781_Phase2_03_C1_Full_Reproducibility.zip` |
| 04 | A0 distilled from B1 NER | 0.409579 | Pre-deadline export | `CS781_Phase2_04_A0_KD_Reproducibility.zip` |
| 05 | Exact-span two-of-four consensus | 0.463067 | Pre-deadline export; rule selected on validation | `CS781_Phase2_05_Consensus_2of4_Reproducibility.zip` |
| 06 | LegalSpan header attention and CRF | 0.411516 | Pre-deadline export | `CS781_Phase2_06_LegalSpan_HeaderCRF_Reproducibility.zip` |
| 07 | LegalSpan coarse spans and legal roles | 0.408721 | Pre-deadline export | `CS781_Phase2_07_LegalSpan_RoleSpan_Reproducibility.zip` |
| 08 | D222 with full NER tuning | 0.400950 | Pre-deadline export | `CS781_Phase2_08_D222_Full_Reproducibility.zip` |
| 09 | D222 with LoRMA NER tuning | 0.427679 | Pre-deadline export | `CS781_Phase2_09_D222_LoRMA_Reproducibility.zip` |
| 10 | Continued D222 pretraining + LoRMA | 0.436864 | Pre-deadline export | `CS781_Phase2_10_D222_Continued_LoRMA_Reproducibility.zip` |
| 11 | D222 100M checkpoint + LoRMA | 0.420074 | Completed after original Phase 2 deadline | `CS781_Phase2_11_D222_100M_LoRMA_Reproducibility.zip` |
| 12 | D222 200M checkpoint + LoRMA | 0.455429 | Completed after original Phase 2 deadline | `CS781_Phase2_12_D222_200M_LoRMA_Reproducibility.zip` |

## Architecture count

There are 12 distinct prediction exports: 11 trained model variants and one span-voting ensemble. The D222 full/LoRMA/continued/100M/200M runs share the same D222 block architecture but differ in adaptation method or pretrained checkpoint. Thus the count of unique neural backbone families here is **five**: LegalSpan, B1, C1, A0, and D222. The header/CRF and role-span modules are distinct Phase 2 heads on LegalSpan.

## Original source and weight archives

The complete archives under `../phase2_20261011` and `../extension_archives` contain the selected large weights, tokenizer bytes, logs, and run provenance. These compact study packages intentionally include source/config/record snapshots and exact prediction ZIPs, not multi-gigabyte weights or official dataset copies.

## GitHub boundary

The private GitHub repository contains code only. These explanatory documents, reports, evidence copies, and prediction ZIPs are local packages and are not pushed there.

# CS781 Phase 1 all-model methodology packages

These packages combine original source and archived logs/metrics with later, explicitly labeled explanation notebooks and reports. Original October 6 complete archives contain no .ipynb files. The A0 historical methodology package separately contains a Colab code notebook, preserved under `A_dense/historical_A0_methodology/`. D222 has a separate original code notebook and is post-deadline.

| Lineage | Evidence | Original small artifacts | Local metric files | Package |
|---|---|---:|---:|---|
| A0 dense causal Transformer | 0.8304 (historical Codabench screenshot) | 6 | 0 | `CS781_Phase1_A_dense_Reproducibility.zip` |
| Improved A / LegalSpan | 0.8341 (historical Codabench screenshot) | 54 | 21 | `CS781_Phase1_A0_LegalSpan_Reproducibility.zip` |
| B0 encoder-decoder Transformer | 0.7972 (historical Codabench screenshot) | 7 | 0 | `CS781_Phase1_B0_EncoderDecoder_Reproducibility.zip` |
| Improved B continuation | 0.8244 (historical Codabench screenshot) | 45 | 8 | `CS781_Phase1_B_Improved_Reproducibility.zip` |
| B1 own-BPE encoder-decoder | No independently verified Codabench score | 46 | 7 | `CS781_Phase1_B1_EncoderDecoder_Reproducibility.zip` |
| C prompt encoder + GRU generator | 0.8237 (historical Codabench screenshot) | 47 | 7 | `CS781_Phase1_C0_EncoderGRU_Reproducibility.zip` |
| C1 own-BPE encoder + GRU | No independently verified Codabench score | 5 | 0 | `CS781_Phase1_C1_EncoderGRU_Reproducibility.zip` |
| D222 local/global grouped-query decoder | Local confirmation BERTScore F1 0.863190; post-deadline, no Codabench score | 10 | 7 | `CS781_Phase1_D222_LocalGlobal_Reproducibility.zip` |

The large original complete archives with selected checkpoint weights and official data remain separately preserved locally. This GitHub-ready collection includes code and compact evidence but does not duplicate official corpus data or multi-gigabyte weights.

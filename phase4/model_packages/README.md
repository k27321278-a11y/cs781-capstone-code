# CS781 Phase 4 CJPE — source-audited architecture packages

This follows the Phase 1 A0 and Phase 2 package format: source, run evidence, README, technical reference, code map, architecture diagram, illustrated companion notebook, editable LaTeX/Markdown/PDF methodology report, and a verified prediction ZIP. Documentation was assembled after the actual Kaggle runs. Companion notebooks have no fabricated execution outputs.

**Counting:** 13 prediction/method exports in this collection: six initial Kaggle adaptation jobs, four B1 study heads, three later B1 stacks. The six initial jobs and four B1 heads are trained variants; the stacks are prediction-level ensembles, not new neural backbones. B1 v5/v6/v7 are continuation segments of the same two head experiments, not separate architectures. Prepared but unlaunched depth/FFN options and additional model notebooks are retained in the source inventory but are not scored runs.

| Package | Method | Official-dev macro-F1 | Codabench screenshot | Download |
|---|---:|---:|---:|---|
| [01_D222_LoRMA](01_D222_LoRMA/README.md) | D222 + LoRMA | 0.449472 | 0.4650 | [ZIP](../../packages/phase4/CS781_Phase4_01_D222_LoRMA_Reproducibility.zip) |
| [02_B1_Partial](02_B1_Partial/README.md) | B1 partial fine-tuning | 0.594530 | 0.5895 | [ZIP](../../packages/phase4/CS781_Phase4_02_B1_Partial_Reproducibility.zip) |
| [03_C1_Full](03_C1_Full/README.md) | C1 full fine-tuning | 0.484079 | 0.4994 | [ZIP](../../packages/phase4/CS781_Phase4_03_C1_Full_Reproducibility.zip) |
| [04_A0_Distillation](04_A0_Distillation/README.md) | A0 knowledge distillation | 0.563179 | 0.5691 | [ZIP](../../packages/phase4/CS781_Phase4_04_A0_Distillation_Reproducibility.zip) |
| [05_D222_Adapters](05_D222_Adapters/README.md) | D222 bottleneck adapters | 0.530606 | not evidenced | [ZIP](../../packages/phase4/CS781_Phase4_05_D222_Adapters_Reproducibility.zip) |
| [06_C1_BitFit](06_C1_BitFit/README.md) | C1 BitFit | 0.535560 | not evidenced | [ZIP](../../packages/phase4/CS781_Phase4_06_C1_BitFit_Reproducibility.zip) |
| [07_B1_Reference](07_B1_Reference/README.md) | B1 full-data reference | 0.604727 | 0.6145 | [ZIP](../../packages/phase4/CS781_Phase4_07_B1_Reference_Reproducibility.zip) |
| [08_B1_Hierarchy](08_B1_Hierarchy/README.md) | B1 document hierarchy | 0.534473 | not evidenced | [ZIP](../../packages/phase4/CS781_Phase4_08_B1_Hierarchy_Reproducibility.zip) |
| [09_B1_GRUOnly](09_B1_GRUOnly/README.md) | B1 window GRU-only | 0.619677 | not evidenced | [ZIP](../../packages/phase4/CS781_Phase4_09_B1_GRUOnly_Reproducibility.zip) |
| [10_B1_TransformerOnly](10_B1_TransformerOnly/README.md) | B1 window Transformer-only | 0.674622 | 0.6641 | [ZIP](../../packages/phase4/CS781_Phase4_10_B1_TransformerOnly_Reproducibility.zip) |
| [11_B1_Stack4](11_B1_Stack4/README.md) | Four-model B1 logistic stack | 0.693956 | 0.6723 | [ZIP](../../packages/phase4/CS781_Phase4_11_B1_Stack4_Reproducibility.zip) |
| [12_B1_Stack6](12_B1_Stack6/README.md) | Six-model B1 logistic stack | 0.703217 | 0.6677 | [ZIP](../../packages/phase4/CS781_Phase4_12_B1_Stack6_Reproducibility.zip) |
| [13_B1_Stack6_CVCal](13_B1_Stack6_CVCal/README.md) | Six-model B1 stack with calibrated threshold | 0.701977 | 0.6651 | [ZIP](../../packages/phase4/CS781_Phase4_13_B1_Stack6_CVCal_Reproducibility.zip) |

## Score evidence boundaries

Official dev is the 994-case labeled set. The Codabench column is transcribed from the user-supplied finished-submissions screenshot, retained in `evidence/CODABENCH_20261026_USER_SCREENSHOT.png`. It was not retrieved from Codabench's API. The screenshot truncates the GRU-only filename, so this collection does not attribute its 0.6223 row to a specific v5/v6/v7 ZIP. The best score visible there is 0.6723 for the four-model stack; the same stack's official-dev score is 0.693956. Scores are not interchangeable.

Several initial v3 Kaggle exports were completed after the original nominal run deadline; each original terminal/config record retains timing. The B1 v5-v7 head runs paused under runtime budgets and are not presented as converged. The stack comparison on official dev is exploratory/post-selection.

## Code, weights, and data

The model source and notebooks also live under `../shared_runner/`, `../individual_model_notebooks/`, `../b1_study/`, and `../ensembles/`. These packages duplicate compact source to make each ZIP reviewable. Multi-gigabyte checkpoints and official course datasets are retained in original local archives, not in this private GitHub repository. The prediction ZIP inside each package is distinct from its report/code ZIP.

## Submission guidance

The course's final Hello IITK ZIP is a separate deliverable, with only best-performing code for each phase plus the report, while the private GitHub repository is the broader record of all model code and experiments. The private repository must be made public only after a fresh user instruction. These model packages are documentation and reproducibility bundles, not an automatically submitted final Hello IITK ZIP.

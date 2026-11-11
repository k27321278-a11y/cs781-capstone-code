# B1 sparse-guided passage selection

**Phase 5 · trained model · Codabench macro-F1 0.2417**
Scored submission ID(s): 1016593. Duplicate uploads of the same export are one variant, not separate architectures.

## The question behind this run

Try to make the limited neural context see passages that a sparse statute signal judges useful. This changes evidence selection inside a single neural model rather than blending two final prediction streams.

## What the code does

A fit-only TF-IDF logistic salience model selects passage starts without dev or test labels. B1 encodes the selected windows; label-wise smooth-max pooling turns window evidence into statute probabilities.

![Processing path](figures/pipeline.svg)

## Evidence and result

The archived local result is **0.2464 official-dev macro-F1**. The visible finished Codabench result is **0.2417** on the macro-F1 scale. These are separate evaluations; neither is an estimate of the other. The local official-dev macro-F1 is about 0.2464 and Codabench is 0.2417, below the plain LoRA baseline. The selected passages may have overemphasized lexical cues or disrupted broader context; this is a hypothesis, not a measured ablation.

The logged run does not isolate the passage selector from the changed pooling head, so causal attribution is limited.

## Files

- [`code/phase5_sparse_guided_model.py`](code/phase5_sparse_guided_model.py) — original file, SHA-256 `0734b56f556c486cbe7f741c6bfd198d399c34396b7bc55b605c3064a7da2071`
- [`code/phase5_sparse_guided_starts.py`](code/phase5_sparse_guided_starts.py) — original file, SHA-256 `ebe4269cabef2e4d588470631846a5191f108ae826c346eefe8ce574943739d1`
- [`code/phase5_multiwindow.py`](code/phase5_multiwindow.py) — original file, SHA-256 `86ac644db5ab5f19b2da1772deae2ebcc8fbe25b67cb599b30f765fa53a990da`
- [`code/phase5_b1_train.py`](code/phase5_b1_train.py) — original file, SHA-256 `645cb2e65313ff49aa6ae574aeab9b015be2365bf427b5e03e3672bc94e20996`
- [`code/original_runner.ipynb`](code/original_runner.ipynb) — archived launcher/source notebook; saved outputs are not assumed.
- [`notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb`](notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb) — explanatory notebook made after the run; no saved training outputs.
- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) — source symbols, method mechanics, provenance, and interpretation.
- [`report/METHODOLOGY_REPORT.tex`](report/METHODOLOGY_REPORT.tex) and [`report/METHODOLOGY_REPORT.pdf`](report/METHODOLOGY_REPORT.pdf) — editable and typeset methodology note.
- [`evidence/submitted_prediction.zip`](evidence/submitted_prediction.zip) — original-named export `phase5_b1_sparse_guided_prediction.zip`, 13019 predictions, CRC checked.
- [`evidence/PROVENANCE.json`](evidence/PROVENANCE.json) records original-file hashes and metric provenance.

## Reproduction boundary

The original source code is preserved byte for byte. It requires the authorized course dataset, the parent Phase 1 checkpoint and tokenizer, installed dependencies, and the appropriate GPU/runtime to retrain. Read the source entry point and original notebook before adapting paths: the historical Kaggle mounts are not local defaults. The illustrated notebook reads archived evidence; running it does not retrain the model. The prediction ZIP, where present, can be inspected without model weights.

## What we can and cannot conclude

The local official-dev macro-F1 is about 0.2464 and Codabench is 0.2417, below the plain LoRA baseline. The selected passages may have overemphasized lexical cues or disrupted broader context; this is a hypothesis, not a measured ablation. The logged run does not isolate the passage selector from the changed pooling head, so causal attribution is limited. The source and logs establish this particular run and export; this comparison does not isolate every changed variable or imply statistical significance.

Documentation was assembled after the runs with AI assistance and manually checked against the archived source, report, and displayed submissions. It is not a claim that the original Kaggle notebook had these explanatory cells or saved outputs.

# B1 AdaLoRA summarizer

**Phase 6 · trained summarizer · Codabench BERTScore F1 0.7656**
Scored submission ID(s): 1030985. Duplicate uploads of the same export are one variant, not separate architectures.

## The question behind this run

Try allocating low-rank adaptation capacity where the summarizer's B1 projections most need it. The dynamic rank budget is a parameter-efficiency experiment; it does not change the task's 100 test IDs or the summary output schema.

## What the code does

AdaLoRA wraps linear layers with trainable low-rank factors and adjusts rank allocation during optimizer updates. The B1 passage-fusion path supplies long-document context.

![Processing path](figures/pipeline.svg)

## Evidence and result

The archived local result is **-1.6305 validation negative NLL; BERTScore not measured in this report**. The visible finished Codabench result is **0.7656** on the BERTScore F1 scale. These are separate evaluations; neither is an estimate of the other. Codabench BERTScore F1 was 0.7656. The archived report selected epoch 1, so this result is particularly sensitive to training duration and checkpoint selection.

The saved evidence does not support a claim that adaptive rank allocation beat the fixed B1 fusion system.

## Files

- [`code/b1_adalora.py`](code/b1_adalora.py) — original file, SHA-256 `768c00167623ba837641ebdcd0eebb0c2ec0210d67ad90c642c6c918215458c8`
- [`code/b1_fusion.py`](code/b1_fusion.py) — original file, SHA-256 `58c9ee6bddcca2fd62d285ebaf6fb70524405cda5de90c451ea46984ddb26aa5`
- [`code/train_phase6.py`](code/train_phase6.py) — original file, SHA-256 `4d1b2f0c303e140e659136f8712c2337cfd095004a643c49e8b0a03c4fb84af8`
- [`code/original_runner.ipynb`](code/original_runner.ipynb) — archived launcher/source notebook; saved outputs are not assumed.
- [`notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb`](notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb) — explanatory notebook made after the run; no saved training outputs.
- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) — source symbols, method mechanics, provenance, and interpretation.
- [`report/METHODOLOGY_REPORT.tex`](report/METHODOLOGY_REPORT.tex) and [`report/METHODOLOGY_REPORT.pdf`](report/METHODOLOGY_REPORT.pdf) — editable and typeset methodology note.
- [`evidence/submitted_prediction.zip`](evidence/submitted_prediction.zip) — original-named export `phase6_b1_adalora_prediction.zip`, 100 predictions, CRC checked.
- [`evidence/PROVENANCE.json`](evidence/PROVENANCE.json) records original-file hashes and metric provenance.

## Reproduction boundary

The original source code is preserved byte for byte. It requires the authorized course dataset, the parent Phase 1 checkpoint and tokenizer, installed dependencies, and the appropriate GPU/runtime to retrain. Read the source entry point and original notebook before adapting paths: the historical Kaggle mounts are not local defaults. The illustrated notebook reads archived evidence; running it does not retrain the model. The prediction ZIP, where present, can be inspected without model weights.

## What we can and cannot conclude

Codabench BERTScore F1 was 0.7656. The archived report selected epoch 1, so this result is particularly sensitive to training duration and checkpoint selection. The saved evidence does not support a claim that adaptive rank allocation beat the fixed B1 fusion system. The source and logs establish this particular run and export; this comparison does not isolate every changed variable or imply statistical significance.

Documentation was assembled after the runs with AI assistance and manually checked against the archived source, report, and displayed submissions. It is not a claim that the original Kaggle notebook had these explanatory cells or saved outputs.

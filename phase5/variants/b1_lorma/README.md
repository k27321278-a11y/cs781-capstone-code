# B1 LoRMA+ statute classifier

**Phase 5 · trained model · Codabench macro-F1 0.2542**
Scored submission ID(s): 1016282. Duplicate uploads of the same export are one variant, not separate architectures.

## The question behind this run

Test whether a multiplicative low-rank adaptation changes the inherited B1 representation more effectively than additive LoRA. Only the adaptation mechanism changes relative to B1 LoRA; that makes the pair useful for testing parameter-efficient adaptation rather than changing the task pipeline all at once.

## What the code does

LoRMA+ wraps a linear output with trainable low-rank multiplicative factors while preserving the pretrained Phase 1 weights. The same multi-label calibration and evaluation route is retained for a meaningful comparison.

![Processing path](figures/pipeline.svg)

## Evidence and result

The archived local result is **0.2517 official-dev macro-F1**. The visible finished Codabench result is **0.2542** on the macro-F1 scale. These are separate evaluations; neither is an estimate of the other. The observed Codabench score, 0.2542, is slightly below LoRA's 0.2554. The difference is small and cannot establish a general ranking of the methods.

A low-rank output correction cannot recover evidence that the window selection never passes to the encoder.

## Files

- [`code/phase5_b1_train.py`](code/phase5_b1_train.py) — original file, SHA-256 `645cb2e65313ff49aa6ae574aeab9b015be2365bf427b5e03e3672bc94e20996`
- [`code/phase5_lorma.py`](code/phase5_lorma.py) — original file, SHA-256 `d991d6da627a21f8f178dedff98c090eba364f55ba0b15db51467c384f7f1ba1`
- [`code/phase5_core.py`](code/phase5_core.py) — original file, SHA-256 `5ea4fb38802cabb47692e80ae21b873afdb0caa67cee7fbbec59fc44db575009`
- [`code/original_runner.ipynb`](code/original_runner.ipynb) — archived launcher/source notebook; saved outputs are not assumed.
- [`notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb`](notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb) — explanatory notebook made after the run; no saved training outputs.
- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) — source symbols, method mechanics, provenance, and interpretation.
- [`report/METHODOLOGY_REPORT.tex`](report/METHODOLOGY_REPORT.tex) and [`report/METHODOLOGY_REPORT.pdf`](report/METHODOLOGY_REPORT.pdf) — editable and typeset methodology note.
- [`evidence/submitted_prediction.zip`](evidence/submitted_prediction.zip) — original-named export `phase5_b1_lorma_prediction.zip`, 13019 predictions, CRC checked.
- [`evidence/PROVENANCE.json`](evidence/PROVENANCE.json) records original-file hashes and metric provenance.

## Reproduction boundary

The original source code is preserved byte for byte. It requires the authorized course dataset, the parent Phase 1 checkpoint and tokenizer, installed dependencies, and the appropriate GPU/runtime to retrain. Read the source entry point and original notebook before adapting paths: the historical Kaggle mounts are not local defaults. The illustrated notebook reads archived evidence; running it does not retrain the model. The prediction ZIP, where present, can be inspected without model weights.

## What we can and cannot conclude

The observed Codabench score, 0.2542, is slightly below LoRA's 0.2554. The difference is small and cannot establish a general ranking of the methods. A low-rank output correction cannot recover evidence that the window selection never passes to the encoder. The source and logs establish this particular run and export; this comparison does not isolate every changed variable or imply statistical significance.

Documentation was assembled after the runs with AI assistance and manually checked against the archived source, report, and displayed submissions. It is not a claim that the original Kaggle notebook had these explanatory cells or saved outputs.

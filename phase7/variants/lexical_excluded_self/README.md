# Lexical ranking excluding query itself

**Phase 7 · retrieval correction · Codabench micro-F1@7 0.4017**
Scored submission ID(s): 1039896. Duplicate uploads of the same export are one variant, not separate architectures.

## The question behind this run

A prior case retrieval system must never return its own query as precedent. This isolates the validity correction from the neural model so its score can be compared with the uncorrected lexical fallback.

## What the code does

The export removes self IDs before ranking and fills the seven slots with eligible lexical candidates from each query's pool.

![Processing path](figures/pipeline.svg)

## Evidence and result

The archived local result is **not independently reported after correction**. The visible finished Codabench result is **0.4017** on the micro-F1@7 scale. These are separate evaluations; neither is an estimate of the other. Codabench micro-F1@7 rose to 0.4017 from the earlier 0.3680 lexical submission. The comparison also changes the export rule, so it should not be labelled a model-training gain.

The score difference cannot isolate every other version change without the missing original ZIP.

## Files

- [`code/phase7_export.py`](code/phase7_export.py) — original file, SHA-256 `8080cd75b1600dbd2fb7ef2be574bff3e06b18dfbaaeaa34678ecb0ad9d2f9ea`
- [`code/phase7_run.py`](code/phase7_run.py) — original file, SHA-256 `849a42dc8b5f179ce01197419c9cd2429bd324e4d3d60a949b863caa9b70191e`
- No original runner notebook was located for this exact export. The illustrated notebook is a later reconstruction.
- [`notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb`](notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb) — explanatory notebook made after the run; no saved training outputs.
- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) — source symbols, method mechanics, provenance, and interpretation.
- [`report/METHODOLOGY_REPORT.tex`](report/METHODOLOGY_REPORT.tex) and [`report/METHODOLOGY_REPORT.pdf`](report/METHODOLOGY_REPORT.pdf) — editable and typeset methodology note.
- [`evidence/submitted_prediction.zip`](evidence/submitted_prediction.zip) — original-named export `Phase7_PCR_lexical_excluded_self.zip`, 237 predictions, CRC checked.
- [`evidence/PROVENANCE.json`](evidence/PROVENANCE.json) records original-file hashes and metric provenance.

## Reproduction boundary

The original source code is preserved byte for byte. It requires the authorized course dataset, the parent Phase 1 checkpoint and tokenizer, installed dependencies, and the appropriate GPU/runtime to retrain. Read the source entry point and original notebook before adapting paths: the historical Kaggle mounts are not local defaults. The illustrated notebook reads archived evidence; running it does not retrain the model. The prediction ZIP, where present, can be inspected without model weights.

## What we can and cannot conclude

Codabench micro-F1@7 rose to 0.4017 from the earlier 0.3680 lexical submission. The comparison also changes the export rule, so it should not be labelled a model-training gain. The score difference cannot isolate every other version change without the missing original ZIP. The source and logs establish this particular run and export; this comparison does not isolate every changed variable or imply statistical significance.

Documentation was assembled after the runs with AI assistance and manually checked against the archived source, report, and displayed submissions. It is not a claim that the original Kaggle notebook had these explanatory cells or saved outputs.

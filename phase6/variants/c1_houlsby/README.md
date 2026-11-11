# C1 Houlsby bottleneck summarizer

**Phase 6 · trained summarizer · Codabench BERTScore F1 0.7700**
Scored submission ID(s): 1030778, 1030779, 1030987. Duplicate uploads of the same export are one variant, not separate architectures.

## The question behind this run

Reuse the Phase 1 C1 legal encoder while changing a small bottleneck module for summarization. This tests a different Phase 1 lineage and adaptation family against B1 fusion under the same summarization contract.

## What the code does

Houlsby-style adapters insert down-project, nonlinear transform, up-project residual modules in the encoder. Recurrent readout tuning and the generation head map case evidence into the summary task.

![Processing path](figures/pipeline.svg)

## Evidence and result

The archived local result is **-1.9414 validation negative NLL; BERTScore not measured in this report**. The visible finished Codabench result is **0.7700** on the BERTScore F1 scale. These are separate evaluations; neither is an estimate of the other. The Codabench BERTScore F1 was 0.7700. The run report selects epoch 3 using negative summary NLL; that is a training selection metric, not the hidden score.

Different backbones and adaptation methods change together, so a single score cannot isolate the benefit of Houlsby adapters.

## Files

- [`code/c1_houlsby.py`](code/c1_houlsby.py) — original file, SHA-256 `aac5188ee4f3c29e861938c2ddba05edbbcb9c32d1e02f79fd99d4b6d3aeab37`
- [`code/phase6_data.py`](code/phase6_data.py) — original file, SHA-256 `7b539dd1679e593c85058aa14d31f04ad92b85ff7c9d9e9fa647d48aa3a40b4f`
- [`code/train_phase6.py`](code/train_phase6.py) — original file, SHA-256 `4d1b2f0c303e140e659136f8712c2337cfd095004a643c49e8b0a03c4fb84af8`
- [`code/original_runner.ipynb`](code/original_runner.ipynb) — archived launcher/source notebook; saved outputs are not assumed.
- [`notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb`](notebooks/ILLUSTRATED_RECONSTRUCTION.ipynb) — explanatory notebook made after the run; no saved training outputs.
- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md) — source symbols, method mechanics, provenance, and interpretation.
- [`report/METHODOLOGY_REPORT.tex`](report/METHODOLOGY_REPORT.tex) and [`report/METHODOLOGY_REPORT.pdf`](report/METHODOLOGY_REPORT.pdf) — editable and typeset methodology note.
- [`evidence/submitted_prediction.zip`](evidence/submitted_prediction.zip) — original-named export `phase6_c1_houlsby_prediction.zip`, 100 predictions, CRC checked.
- [`evidence/PROVENANCE.json`](evidence/PROVENANCE.json) records original-file hashes and metric provenance.

## Reproduction boundary

The original source code is preserved byte for byte. It requires the authorized course dataset, the parent Phase 1 checkpoint and tokenizer, installed dependencies, and the appropriate GPU/runtime to retrain. Read the source entry point and original notebook before adapting paths: the historical Kaggle mounts are not local defaults. The illustrated notebook reads archived evidence; running it does not retrain the model. The prediction ZIP, where present, can be inspected without model weights.

## What we can and cannot conclude

The Codabench BERTScore F1 was 0.7700. The run report selects epoch 3 using negative summary NLL; that is a training selection metric, not the hidden score. Different backbones and adaptation methods change together, so a single score cannot isolate the benefit of Houlsby adapters. The source and logs establish this particular run and export; this comparison does not isolate every changed variable or imply statistical significance.

Documentation was assembled after the runs with AI assistance and manually checked against the archived source, report, and displayed submissions. It is not a claim that the original Kaggle notebook had these explanatory cells or saved outputs.

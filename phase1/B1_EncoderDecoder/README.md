# B1 own-BPE encoder-decoder

B1 retains the B0 six-encoder/six-decoder graph, but uses the handwritten byte-level BPE and a separate scratch/continuation pretraining lineage. This isolates tokenizer and training changes without claiming a different block graph.

**Evidence:** No independently verified Codabench score. Original complete archive: `CS781_B1_OwnBPE_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# B1 own-BPE encoder-decoder

**Family:** B: same encoder-decoder graph as B0  
**Parameters:** 124.3 million  
**Original archive:** `CS781_B1_OwnBPE_Complete_20261006.zip`  
**Result evidence:** No independently verified Codabench score in the preserved evidence.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

B1 uses B0's byte-identical six-encoder/six-decoder Transformer graph, including causal decoder self-attention and prompt cross-attention. It is a scratch/continuation lineage using its own BPE vocabulary and training record, not a fourth architecture family.

## Original source and notebook

- Model: [`source/cs781/models/b.py`](source/cs781/models/b.py)
- Run configuration: [`source/configs/b_ownbpe.json`](source/configs/b_ownbpe.json)
- Training and export entry points: `scripts/run_b1_ownbpe_gpu0.sh; cs781/train_b.py; cs781/train_b_improved.py; cs781/cli_b.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_B1_EncoderDecoder_Code.ipynb](Phase1_B1_EncoderDecoder_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- This archive includes the handwritten `cs781/own_bpe.py` tokenizer implementation.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


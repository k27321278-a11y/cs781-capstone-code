# C1 own-BPE encoder + GRU

The model uses the C0 prompt-encoder/GRU computation graph with an independently trained handwritten BPE tokenizer and a separate scratch-training lineage. The architecture graph is unchanged from C0.

**Evidence:** No independently verified Codabench score. Original complete archive: `CS781_C1_OwnBPE_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# C1 own-BPE encoder plus GRU

**Family:** C: same Transformer-encoder/GRU graph as C  
**Parameters:** 117.8 million  
**Original archive:** `CS781_C1_OwnBPE_Complete_20261006.zip`  
**Result evidence:** No independently verified Codabench score in the preserved evidence.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

C1 is a separate scratch-training lineage using the same byte-identical C model graph: ten-block bidirectional prompt encoder, four-layer GRU generator, prompt cross-attention, SwiGLU fusion, and tied output embeddings. Its own BPE vocabulary changes the training lineage but does not create a new architecture family.

## Original source and notebook

- Model: [`source/cs781/models/c.py`](source/cs781/models/c.py)
- Run configuration: [`source/configs/c1_ownbpe.json`](source/configs/c1_ownbpe.json)
- Training and export entry points: `scripts/run_c1_ownbpe_gpu1.sh; cs781/train_c.py; cs781/cli_c.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_C1_EncoderGRU_Code.ipynb](Phase1_C1_EncoderGRU_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- This archive includes the handwritten `cs781/own_bpe.py` tokenizer implementation.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


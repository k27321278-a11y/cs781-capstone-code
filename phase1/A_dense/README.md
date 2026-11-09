# A0 dense causal Transformer

A 12-block decoder-only Transformer predicts the next subword token from a causal context. Width 768, 12 attention heads of 64, 2048-wide SwiGLU feed-forward layers, RMSNorm, rotary positions, and tied input/output embeddings make the 110.1M-parameter graph.

**Evidence:** 0.8304 (historical Codabench screenshot). Original complete archive: `CS781_A0_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# A0 dense causal Transformer

**Family:** A: decoder-only Transformer  
**Parameters:** 110.1 million  
**Original archive:** `CS781_A0_Complete_20261006.zip`  
**Result evidence:** 0.8304 historical Codabench screenshot score.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

A 12-block causal decoder with 768-dimensional hidden states, 12 attention heads of width 64, 2,048-dimensional SwiGLU feed-forward layers, rotary positions, RMSNorm, and tied input/output token embeddings. Its 32,768-token vocabulary and 2,048-token context are configured in `configs/a.json`. It predicts the next token autoregressively.

## Original source and notebook

- Model: [`source/cs781/models/a.py`](source/cs781/models/a.py)
- Run configuration: [`source/configs/a.json`](source/configs/a.json)
- Training and export entry points: `scripts/run_a.sh; cs781/train.py; cs781/cli.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_A_dense_Code.ipynb](Phase1_A_dense_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- The archived tokenizer was trained on the course data with a library BPE merge trainer; this is distinct from a handwritten BPE implementation.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


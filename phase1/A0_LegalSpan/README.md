# Improved A / LegalSpan

The dense 12-block causal backbone gains a 128-dimensional prompt route in its upper four blocks. Role and continuity cues bias access to prompt states; a learned gate mixes vocabulary generation with prompt copying. The route respects causal boundaries.

**Evidence:** 0.8341 (historical Codabench screenshot). Original complete archive: `CS781_Improved_A_LegalSpan_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# Improved A / LegalSpan

**Family:** A: decoder-only Transformer with a distinct routed/copy graph  
**Parameters:** 112.1 million  
**Original archive:** `CS781_Improved_A_LegalSpan_Complete_20261006.zip`  
**Result evidence:** 0.8341 historical Codabench screenshot score; highest evidenced October 6 Phase 1 submission.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

It retains A0's 12-block, width-768 causal Transformer and adds a prompt route in the upper four blocks. Each route projects prompt keys/values and generation queries into 128 dimensions, biases attention with prompt-only role features and continuity counts, then gates the route update into the block. A separate learned gate mixes vocabulary generation with copy probabilities over eligible prompt tokens. The configured context is 8,192 tokens and the vocabulary is 32,768 tokens. The prompt boundary and role IDs are explicit inputs; the implementation prevents future-prompt leakage.

## Original source and notebook

- Model: [`source/legalspan/model.py (inherits cs781/models/a.py)`](source/legalspan/model.py)
- Run configuration: [`source/configs/a_legalspan.json`](source/configs/a_legalspan.json)
- Training and export entry points: `scripts/run_a_legalspan.sh; legalspan/train.py; legalspan/pipeline.py; legalspan/generate.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_A0_LegalSpan_Code.ipynb](Phase1_A0_LegalSpan_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- This archive includes `legalspan/bpe.py`, a handwritten BPE learner and codec.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


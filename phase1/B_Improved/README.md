# Improved B continuation

The B0 encoder-decoder computation graph stays the same. The run continues training from B0 with task-aligned windows and a changed selection/export policy, so this is a training-lineage experiment rather than a new neural architecture.

**Evidence:** 0.8244 (historical Codabench screenshot). Original complete archive: `CS781_Improved_B_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# Improved B continuation

**Family:** B: same encoder-decoder graph as B0  
**Parameters:** 124.3 million  
**Original archive:** `CS781_Improved_B_Complete_20261006.zip`  
**Result evidence:** 0.8244 historical Codabench screenshot score.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

The model definition is byte-identical to B0: six bidirectional prompt-encoder blocks and six causal cross-attending decoder blocks, width 768, 12 heads, and 2,048-dimensional feed-forward layers. Its improvement came from continued training on task-aligned windows and changed training/export settings, not a new architecture family.

## Original source and notebook

- Model: [`source/cs781/models/b.py`](source/cs781/models/b.py)
- Run configuration: [`source/configs/b_improved.json`](source/configs/b_improved.json)
- Training and export entry points: `cs781/train_b_improved.py; scripts/improve_b.py; scripts/finalize_b_after_training.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_B_Improved_Code.ipynb](Phase1_B_Improved_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- This continuation inherited the library-trained BPE tokenizer from B0.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


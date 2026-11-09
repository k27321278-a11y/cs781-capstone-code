# B0 encoder-decoder Transformer

Six bidirectional prompt-encoder blocks build source memory. Six causal decoder blocks attend both to earlier generated tokens and to that source memory before SwiGLU. The model has width 768 and approximately 124.3M parameters.

**Evidence:** 0.7972 (historical Codabench screenshot). Original complete archive: `CS781_B0_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# B0 Transformer encoder-decoder

**Family:** B: encoder-decoder Transformer  
**Parameters:** 124.3 million  
**Original archive:** `CS781_B0_Complete_20261006.zip`  
**Result evidence:** 0.7972 historical Codabench screenshot score.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

A six-block bidirectional prompt encoder feeds a separate six-block causal decoder. Each decoder block has causal self-attention, cross-attention to the prompt encoder output, and a SwiGLU feed-forward layer. Hidden width is 768, with 12 attention heads of width 64, 2,048-dimensional feed-forward layers, rotary positions and RMSNorm. The configuration uses a 32,768-token vocabulary and 2,048-token maximum sequence length.

## Original source and notebook

- Model: [`source/cs781/models/b.py`](source/cs781/models/b.py)
- Run configuration: [`source/configs/b.json`](source/configs/b.json)
- Training and export entry points: `scripts/run_b_gpu1.sh; cs781/train_b.py; cs781/cli_b.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_B0_EncoderDecoder_Code.ipynb](Phase1_B0_EncoderDecoder_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- The archived tokenizer used a library BPE merge trainer.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


# C prompt encoder + GRU generator

Ten bidirectional Transformer blocks encode the prompt. A four-layer GRU decoder, initialized from mean prompt memory, generates in FP32 and cross-attends to encoded context; SwiGLU fusion and tied embeddings produce token logits.

**Evidence:** 0.8237 (historical Codabench screenshot). Original complete archive: `CS781_C_Complete_20261006.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# C prompt encoder plus GRU generator

**Family:** C: Transformer encoder plus recurrent generator  
**Parameters:** 117.8 million  
**Original archive:** `CS781_C_Complete_20261006.zip`  
**Result evidence:** 0.8237 historical Codabench screenshot score.  
**Timing:** This lineage is preserved in an October 6 Phase 1 archive.

## Exact computation graph

A ten-block bidirectional Transformer encodes the prompt. Its masked mean output initializes a four-layer, width-768 GRU. At each generated position, the recurrent output cross-attends to the encoded prompt, then passes through a SwiGLU fusion and tied-vocabulary output head. The encoder uses 12 attention heads of width 64, a maximum prompt length of 8,192 tokens, and a 2,048-token decoder limit. The GRU runs in FP32 in the archived implementation.

## Original source and notebook

- Model: [`source/cs781/models/c.py`](source/cs781/models/c.py)
- Run configuration: [`source/configs/c.json`](source/configs/c.json)
- Training and export entry points: `scripts/run_c_gpu0.sh; cs781/train_c.py; cs781/cli_c.py` under `source/`.
- No original `.ipynb` is in this lineage archive. Its original code is the Python model, training, tokenizer and export modules under `source/`; do not relabel a new wrapper notebook as an original training notebook.
- Review notebook assembled from the recovered source: [Phase1_C0_EncoderGRU_Code.ipynb](Phase1_C0_EncoderGRU_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- The archived tokenizer used a library BPE merge trainer.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


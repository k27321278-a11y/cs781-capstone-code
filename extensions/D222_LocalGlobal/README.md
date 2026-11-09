# D222 local/global grouped-query decoder

A 24-block, width-896 causal decoder uses 14 query heads over two shared key/value heads. Every fourth block attends globally and intervening blocks use 512-token local attention. The 222M-scale run was built after the original Phase 1 deadline.

**Evidence:** Local confirmation BERTScore F1 0.863190; post-deadline, no Codabench score. Original complete archive: `CS781_D222_Benchmark_Complete_20261012.zip`. Original October 6 archive has no executed notebook; the notebooks here are later source-based companions. D222 is a post-deadline extension with its own code notebook.

## Package contents

- `code/`: recovered original Python code, config, scripts and available original code notebooks.
- `notebooks/`: illustrated explanation and code-review notebook.
- `docs/TECHNICAL_REFERENCE.md` and `docs/CODE_MAP.md`: detailed mechanics and line map.
- `figures/architecture.svg`: architecture path.
- `report/`: PDF, Markdown and editable LaTeX methodology sections.
- `evidence/`: extracted original logs/metrics/manifest, with SHA-256 index.

## Original archive summary

# D222 local/global decoder

**Family:** D: grouped-query local/global causal decoder  
**Parameters:** 222.1 million (222,079,872)  
**Original archive:** `CS781_D222_Source_and_Notebook_20261011.zip`  
**Result evidence:** Phase 1 local benchmark dev BERTScore F1 0.863767; no Phase 1 Codabench score.  
**Timing:** D222 was built on October 11, after the October 6 Phase 1 deadline. Its Phase 1 evaluation was a post-deadline local benchmark, not a Phase 1 Codabench submission.

## Exact computation graph

A 24-block causal decoder with width 896, 14 query heads and two key/value heads (grouped-query attention), head width 64 and 2,304-dimensional SwiGLU feed-forward layers. Most blocks use a 512-token local causal attention window; every fourth block uses global causal attention (six global blocks). It uses RMSNorm, rotary positions, tied embeddings, a 32,768-token vocabulary and a 2,048-token context.

## Original source and notebook

- Model: [`source/architecture_d/legal_d/model.py`](source/architecture_d/legal_d/model.py)
- Run configuration: [`source/architecture_d/configs/d222.json`](source/architecture_d/configs/d222.json)
- Training and export entry points: `architecture_d/CS781_D222_Pretrain_and_NER.ipynb; architecture_d/scripts/run_gpu1.sh; architecture_d/legal_d/pretrain.py` under `source/`.
- The original archived notebook is `source/architecture_d/CS781_D222_Pretrain_and_NER.ipynb`.
- Review notebook assembled from the recovered source: [Phase1_D222_LocalGlobal_Code.ipynb](Phase1_D222_LocalGlobal_Code.ipynb). It contains the original model implementation as code cells and links to the original training/export modules. This is a new companion notebook, not an archived original.
- It uses the own-BPE tokenizer from the later prepared dataset; see its archived config and source.

## Provenance and execution

These are recovered original Python modules/configurations/launch scripts. The linked notebook was assembled later for review and is not an original executed training notebook. The old launch paths refer to the former GPU workstation; obtain the official dataset, install the declared dependencies, and adapt paths to run in a new environment. Large weights, private data, logs and prediction ZIPs are retained in the student archive, not this GitHub repository. See the repository root for result limitations and LLM disclosure.


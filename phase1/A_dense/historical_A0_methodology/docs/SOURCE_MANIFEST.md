# Original source and documentation provenance

The original Phase 1 A0 implementation was extracted from supplied source ZIPs; `code/architecture_a_legalgpt.py` was copied without edits. No training checkpoint, tokenizer-fit file or result log came with the source archives.

| Included original code file | SHA-256 | Bytes |
|---|---|---:|
| `code/Architecture_A_LegalGPT_Colab.ipynb` | `393382e6d55a88cf6e585125587ecf8b15b30f28dfcbcc8626b21ab1311363f1` | 5,396 |
| `code/architecture_a_legalgpt.py` | `df76aa7b365d7415d8358e878c8e0ce3be5a0ba5c601b2bc4a7b694ffd4ac754` | 54,994 |
| `code/experiment_ledger.csv` | `2b2acdf64f9bb4780ddec5e83f1b656407cb520aa654321510a0354411608858` | 364 |
| `code/requirements.txt` | `c6a366bd81528161cdbda3a4ff7e68413cf2a9287ba4379ebdf034daffa22a06` | 226 |
| `code/rtx/README_RTX.md` | `3c7c937784b50d393eaa0035550a39a9cc291d030ce4f09afe76883b635f8d34` | 3,231 |
| `code/rtx/architecture_a_legalgpt.py` | `df76aa7b365d7415d8358e878c8e0ce3be5a0ba5c601b2bc4a7b694ffd4ac754` | 54,994 |
| `code/rtx/eval_A0.sh` | `a06c0e9fb245448d1224e1f456f1ce2f954d25c03f1769a70cb6ed8eda5ab4c9` | 354 |
| `code/rtx/experiment_ledger.csv` | `2b2acdf64f9bb4780ddec5e83f1b656407cb520aa654321510a0354411608858` | 364 |
| `code/rtx/prepare_A0.sh` | `ed7851068a4384efe022395ae8ec30949dabfbcee8e2b6fbbc7246ccb21a7234` | 583 |
| `code/rtx/requirements.txt` | `8fb7fb7c37dd0fb3f88c13af3981d0477f19064c12ae9c1494e2b620bf199b0d` | 121 |
| `code/rtx/rtx_preflight.sh` | `d5e5914cdaad9442dc551bdb511ff8302e5db22640e60e5049fbb0abb84a4417` | 648 |
| `code/rtx/select_gpu.sh` | `3705f6b97a5042d587b36b9bd7645a956b92f494b715e64abf0a4430ad484551` | 1,427 |
| `code/rtx/setup_env.sh` | `bcf3f7515b5c2347541c27a7682320febde4b1898a7eec1558f9257ba68f198d` | 747 |
| `code/rtx/submit_A0.sh` | `5445b440738cdb6dd52ec11962b6453d342496f1c1747ccdd96a06ce5fb0df8d` | 471 |
| `code/rtx/train_A0.sh` | `908f0d987ff570e7e354c9d97187893da632c6a3fb7cf348763a67bbcdc7ae99` | 790 |
| `code/rtx/watch_A0.sh` | `2ba2ca72d8bfb7b5bc104b51abe2390024a5113799287bada4134e1796877308` | 287 |

**Audit actions performed:**

- Parsed original Python source with Python AST and indexed top-level classes/functions by exact file line number.
- Executed original `python architecture_a_legalgpt.py --mode selftest`: **PASS**, tiny model 98,624 parameters.
- Inspected Phase 1 original Colab launcher and RTX scripts; the two copies of `architecture_a_legalgpt.py` have the same SHA-256.
- Inspected supplied CS781 Module 2.2 and 2.3 lecture files for normalization, MHA, RoPE and attention algorithm framing.
- Did **not** load the official large corpus, train the 110M model, produce BPE merges, run full inference or obtain Codabench score.

Earlier similarly named `CS781_A0_KrishnaKarthikeya_GN_250564.ipynb` is an unrelated text-classification/Word2Vec/MLP assignment and is intentionally excluded.

# Code map

The paths below point to source recovered from the original complete archive. They are not new training implementations.

## `code/lner/models.py`
- Line 20: `AdaptedLinear`
- Line 21: `__init__`
- Line 37: `forward`
- Line 42: `inject_adapters`
- Line 46: `visit`
- Line 59: `CharacterNER`
- Line 60: `__init__`
- Line 68: `representations`
- Line 76: `run`
- Line 88: `run`
- Line 103: `forward`
- Line 112: `build_model`
- Line 144: `tensor_row`

## `code/lner/train.py`
- Line 23: `save_torch`
- Line 31: `implementation_hashes`
- Line 38: `autocast`
- Line 43: `prepare`
- Line 73: `document_logits`
- Line 97: `decode`
- Line 101: `evaluate`
- Line 113: `teacher_cache`
- Line 153: `run_stage`
- Line 196: `snapshot`
- Line 275: `preflight`
- Line 297: `main`

## `code/backbones/cs781/models/a.py`
- Line 12: `RMSNorm`
- Line 13: `__init__`
- Line 18: `forward`
- Line 23: `RotaryEmbedding`
- Line 24: `__init__`
- Line 31: `apply_rotary`
- Line 41: `CausalMHA`
- Line 42: `__init__`
- Line 48: `forward`
- Line 64: `SwiGLU`
- Line 65: `__init__`
- Line 71: `forward`
- Line 75: `TransformerBlock`
- Line 76: `__init__`
- Line 81: `forward`
- Line 87: `LegalGPT`
- Line 88: `__init__`
- Line 106: `_init_weights`

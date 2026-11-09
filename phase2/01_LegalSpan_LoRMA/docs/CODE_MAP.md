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

## `code/lner/data.py`
- Line 11: `sha256`
- Line 19: `atomic_json`
- Line 30: `byte_offsets_to_chars`
- Line 43: `bio_targets`
- Line 58: `decode_bio`
- Line 71: `strict_score`
- Line 96: `validate_predictions`
- Line 114: `export_zip`
- Line 128: `load_dataset`
- Line 153: `OffsetTokenizer`
- Line 154: `__init__`
- Line 164: `encode`
- Line 190: `prepare_windows`
- Line 214: `constrained_decode`

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

## `code/backbones/legalspan/model.py`
- Line 21: `SpanConfig`
- Line 31: `validate`
- Line 37: `continuity`
- Line 59: `copy_distribution`
- Line 65: `PromptRoute`
- Line 66: `__init__`
- Line 78: `forward`
- Line 109: `SpanCache`
- Line 117: `LegalSpanGPT`
- Line 118: `__init__`
- Line 132: `forward`
- Line 161: `apply_block`
- Line 172: `checked`
- Line 204: `gold_vocab`

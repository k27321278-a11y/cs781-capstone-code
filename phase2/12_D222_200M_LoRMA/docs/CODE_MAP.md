# Code map

The paths below point to source recovered from the original complete archive. They are not new training implementations.

## `code/legal_d/model.py`
- Line 14: `DConfig`
- Line 29: `validate`
- Line 35: `parameter_formula`
- Line 39: `RMSNorm`
- Line 40: `__init__`
- Line 42: `forward`
- Line 45: `rotary`
- Line 52: `GQAAttention`
- Line 53: `__init__`
- Line 59: `attend`
- Line 87: `forward`
- Line 107: `Block`
- Line 108: `__init__`
- Line 113: `forward`
- Line 117: `EfficientDecoder`
- Line 118: `__init__`
- Line 127: `initialize`
- Line 129: `features`

## `code/legal_d/ner.py`
- Line 9: `LinearCRF`
- Line 10: `__init__`
- Line 18: `scores`
- Line 20: `nll`
- Line 35: `DNER`
- Line 36: `__init__`
- Line 41: `emissions`
- Line 48: `forward`
- Line 50: `loss`
- Line 62: `load_ner`

## `code/legal_d/ner_train.py`
- Line 14: `document_logits`
- Line 30: `prepare_ner`
- Line 47: `decode_crf`
- Line 62: `evaluate`
- Line 71: `train_candidate`
- Line 101: `snapshot`
- Line 168: `run_ner`

## `code/scripts/milestone_ner.py`
- Line 21: `main`

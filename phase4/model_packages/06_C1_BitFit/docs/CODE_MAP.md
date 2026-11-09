# Code map

All listed sources are copied from the preserved Phase 4 repository snapshot. The companion notebook is explanatory and was not the training notebook.

## `runner/phase3_rr/data.py`

`acquire_data` (line 8), `documents` (line 34), `load_official` (line 53), `Tokenizer` (line 67), `Tokenizer.__init__` (line 68), `Tokenizer.encode` (line 93), `prepare_examples` (line 95), `validate_predictions` (line 115), `export` (line 120)

## `runner/phase3_rr/models.py`

`LowRank` (line 17), `LowRank.__init__` (line 18), `LowRank.forward` (line 25), `IA3` (line 29), `IA3.__init__` (line 30), `IA3.forward` (line 37), `adapt` (line 43), `load_backbone` (line 58), `RRClassifier` (line 79), `RRClassifier.__init__` (line 80), `RRClassifier.representations` (line 83), `RRClassifier.forward` (line 104), `make_model` (line 110), `restore_selected` (line 118), `tensors` (line 141)

## `runner/phase3_rr/train.py`

`autocast` (line 13), `logits_for` (line 16), `score` (line 25), `prepare` (line 37), `train_one` (line 55), `run` (line 171), `distill` (line 177), `package_records` (line 211)

## `runner/phase3_rr/vendor/backbones/cs781/train.py`

`IndexedExamples` (line 22), `IndexedExamples.__init__` (line 23), `IndexedExamples.at` (line 28), `_code_digest` (line 39), `train_a` (line 48)

## `runner/phase3_rr/vendor/backbones/legalspan/data.py`

`role_ids` (line 22), `observed_role_ids` (line 43), `prepare` (line 54), `load_data` (line 143), `rebuild_roles` (line 155), `MixedExamples` (line 191), `MixedExamples.__init__` (line 192), `MixedExamples.at` (line 198)

## `runner/phase3_rr/vendor/backbones/legalspan/train.py`

`code_digest` (line 22), `batch` (line 32), `atomic_model` (line 41), `train` (line 50)

## `runner/phase4_cjpe/data.py`

`acquire` (line 12), `stream_rows` (line 45), `normal` (line 67), `index_data` (line 89), `choose_ids` (line 135), `chunks` (line 153), `cache_tokens` (line 170), `TokenStore` (line 199), `TokenStore.__init__` (line 200), `TokenStore.row` (line 201), `TokenStore.rows` (line 205), `TokenStore.close` (line 207)

## `runner/phase4_cjpe/export.py`

`autocast` (line 9), `predict` (line 12), `probabilities` (line 22), `score` (line 25), `select_threshold` (line 33), `write_predictions` (line 48), `validate_zip` (line 54), `export_result` (line 66), `package_records` (line 92)

## `runner/phase4_cjpe/models.py`

`OutputAdapter` (line 13), `OutputAdapter.__init__` (line 14), `OutputAdapter.forward` (line 19), `configure` (line 22), `CaseModel` (line 47), `CaseModel.__init__` (line 48), `CaseModel.token_features` (line 52), `CaseModel.chunk_feature` (line 53), `CaseModel.forward` (line 57), `EvidenceCaseModel` (line 72), `EvidenceCaseModel.__init__` (line 74), `EvidenceCaseModel.chunk_feature` (line 87), `EvidenceCaseModel.forward` (line 94), `make_model` (line 111), `restore` (line 121)

## `runner/phase4_cjpe/online_teacher.py`

`CachedTeacher` (line 9), `CachedTeacher.__init__` (line 10), `CachedTeacher.__call__` (line 27), `CachedTeacher.flush` (line 36)

## `runner/phase4_cjpe/train.py`

`deadline_timestamp` (line 14), `snapshot` (line 17), `train_one` (line 23), `cache_namespace` (line 176), `prepare` (line 181), `run` (line 198), `distill` (line 203), `distill_online` (line 235), `worker` (line 253)

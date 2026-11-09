import importlib.util

import pytest


def cls():
    assert importlib.util.find_spec('cs781.prediction_progress') is not None, 'Inference progress is missing'
    from cs781.prediction_progress import PredictionProgress
    return PredictionProgress


def test_saved_records_resume_and_reject_mixed_provenance(tmp_path):
    Progress = cls()
    prompts = {'1': 'prompt '*512, '2': 'second '*512}
    fingerprint = {'checkpoint': 'a'*64, 'policy': 'b'*64}
    path = tmp_path/'progress.json'
    p = Progress(path, prompts, fingerprint)
    p.record('1', 'continuation '*128)
    resumed = Progress(path, prompts, fingerprint)
    assert list(resumed.pending()) == ['2']
    assert resumed.predictions['1'].split() == ['continuation']*128
    with pytest.raises(ValueError, match='provenance'):
        Progress(path, prompts, {'checkpoint': 'c'*64, 'policy': 'b'*64})
    with pytest.raises(ValueError, match='provenance'):
        Progress(path, {**prompts, '2': 'changed '*512}, fingerprint)
    with pytest.raises(ValueError):
        resumed.record('2', 'incomplete')
    resumed.record('2', 'different '*128)
    assert resumed.complete() == resumed.predictions


def test_failure_does_not_lose_other_predictions_or_allow_partial_export(tmp_path):
    p = cls()(tmp_path/'p.json', {'1': 'p '*512, '2': 'q '*512}, {'policy': 'one'})
    p.record('1', 'words '*128)
    p.failure('2', 'GenerationIncomplete: token cap')
    with pytest.raises(ValueError, match='incomplete'):
        p.complete()
    again = cls()(tmp_path/'p.json', p.prompts, {'policy': 'one'})
    assert list(again.pending()) == ['2']
    again.record('2', 'new '*128)
    assert not again.failures
    assert len(again.complete()) == 2

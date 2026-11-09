import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('name', ['step_0002000.pt', 'milestone_209914685.pt'])
def test_c_candidate_names_have_valid_run_provenance(tmp_path, name):
    from cs781.cli_c import _checkpoint_run_directory
    from cs781.io import write_json_atomic
    run = tmp_path/'run'
    checkpoint = run/'candidates'/name
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'owned fixture')
    write_json_atomic(run/'run.json', {'architecture': 'C'})
    assert _checkpoint_run_directory(checkpoint)[1] == run


def test_c_official_gate_requires_matching_completed_checkpoint(tmp_path):
    from cs781.cli_c import _require_complete_training
    from cs781.io import sha256_file, write_json_atomic
    run = tmp_path/'run'
    checkpoint = run/'checkpoints/latest.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'owned fixture')
    metadata = {'synthetic_smoke': False, 'train_config': {'target_tokens': 100},
                'code_sha256': 'a'*64}
    with pytest.raises(ValueError, match='has not reached'):
        _require_complete_training(checkpoint, {'state': {'target_tokens': 99}}, run, metadata)
    state = {'step': 2, 'target_tokens': 101}
    record = {'architecture': 'C', 'status': 'complete', 'target_tokens': 100,
              'state': state, 'code_sha256': metadata['code_sha256'],
              'checkpoint_sha256': sha256_file(checkpoint)}
    write_json_atomic(run/'training_status.json', record)
    _require_complete_training(checkpoint, {'state': state}, run, metadata)
    record['checkpoint_sha256'] = 'f'*64
    write_json_atomic(run/'training_status.json', record)
    with pytest.raises(ValueError, match='completion record'):
        _require_complete_training(checkpoint, {'state': state}, run, metadata)


def test_loaded_checkpoint_hash_tracks_open_descriptor_when_latest_is_replaced(tmp_path, monkeypatch):
    import cs781.cli_c as module
    import torch
    from cs781.io import sha256_file
    assert hasattr(module, '_load_checkpoint_snapshot'), 'Stable loaded-weight hashing is missing'
    path, replacement = tmp_path/'latest.pt', tmp_path/'replacement.pt'
    torch.save({'state': {'step': 1}}, path)
    torch.save({'state': {'step': 2}}, replacement)
    expected_hash = sha256_file(path)
    original = torch.load
    def load_then_replace(handle, **kwargs):
        payload = original(handle, **kwargs)
        replacement.replace(path)
        return payload
    monkeypatch.setattr(torch, 'load', load_then_replace)
    payload, observed_hash = module._load_checkpoint_snapshot(path)
    assert payload['state']['step'] == 1
    assert observed_hash == expected_hash and observed_hash != sha256_file(path)


def test_official_export_can_finish_context_after_zip_was_written(tmp_path, monkeypatch):
    import cs781.cli_c as module
    from cs781.io import sha256_file
    assert hasattr(module, '_finalize_official_export'), 'Resumable ZIP finalization is missing'
    output = tmp_path/'submission.zip'
    prompts, predictions = {'1': 'prompt '*512}, {'1': 'continuation '*128}
    report = {'architecture': 'C', 'checkpoint_sha256': 'a'*64}
    original = module.write_json_atomic
    def interrupt(path, value):
        raise OSError('injected report interruption')
    monkeypatch.setattr(module, 'write_json_atomic', interrupt)
    with pytest.raises(OSError, match='injected'):
        module._finalize_official_export(output, predictions, prompts, report)
    assert output.is_file()
    expected = sha256_file(output)
    monkeypatch.setattr(module, 'write_json_atomic', original)
    result = module._finalize_official_export(output, predictions, prompts, report)
    assert result['prediction_zip_sha256'] == expected == sha256_file(output)
    assert json.loads(output.with_suffix('.context.json').read_text()) == result
    with pytest.raises(ValueError, match='existing ZIP'):
        module._finalize_official_export(output, {'1': 'different '*128}, prompts, report)

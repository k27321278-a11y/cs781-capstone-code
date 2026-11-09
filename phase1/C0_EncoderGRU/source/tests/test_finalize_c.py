import importlib.util

import pytest


def api():
    from pathlib import Path
    path = Path(__file__).parents[1]/'scripts/finalize_c.py'
    assert path.is_file(), 'C completion pipeline is missing'
    spec = importlib.util.spec_from_file_location('cs781_finalize_c_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_uses_complete_dev_reports_and_not_confirmation():
    module = api()
    reports = [
        {'checkpoint': 'one', 'policy': 0, 'report': {'split': 'dev', 'limited_sample': False,
         'declared_ids': ['a', 'b'], 'metrics': {'bertscore_f1': 0.8, 'rouge_l_f1': 0.2, 'bleu4': 0.02}}},
        {'checkpoint': 'two', 'policy': 2, 'report': {'split': 'dev', 'limited_sample': False,
         'declared_ids': ['a', 'b'], 'metrics': {'bertscore_f1': 0.83, 'rouge_l_f1': 0.21, 'bleu4': 0.03}}},
    ]
    assert module.select_dev(reports, {'a', 'b'})['checkpoint'] == 'two'
    reports[1]['report']['split'] = 'confirmation'
    with pytest.raises(ValueError, match='full dev'):
        module.select_dev(reports, {'a', 'b'})


def test_selected_candidate_needs_completed_parent_and_substantive_matched_budget(tmp_path):
    module = api()
    from cs781.io import sha256_file, write_json_atomic
    run = tmp_path/'run'
    latest, candidate = run/'checkpoints/latest.pt', run/'candidates/milestone_209914685.pt'
    latest.parent.mkdir(parents=True)
    candidate.parent.mkdir()
    latest.write_bytes(b'owned latest')
    candidate.write_bytes(b'owned candidate')
    metadata = {'synthetic_smoke': False, 'architecture': 'C',
                'train_config': {'target_tokens': 419829370}, 'code_sha256': 'a'*64}
    payload = {'state': {'step': 100, 'target_tokens': 209914685},
               '_loaded_checkpoint_sha256': sha256_file(candidate)}
    status = {'architecture': 'C', 'status': 'running', 'target_tokens': 419829370,
              'state': {'step': 200, 'target_tokens': 419829370},
              'code_sha256': 'a'*64, 'checkpoint_sha256': sha256_file(latest)}
    write_json_atomic(run/'training_status.json', status)
    with pytest.raises(ValueError, match='completed parent'):
        module.require_selected_candidate(candidate, payload, run, metadata)
    status['status'] = 'complete'
    write_json_atomic(run/'training_status.json', status)
    assert module.require_selected_candidate(candidate, payload, run, metadata) == status
    payload['state']['target_tokens'] = 100
    with pytest.raises(ValueError, match='matched budget'):
        module.require_selected_candidate(candidate, payload, run, metadata)


@pytest.mark.parametrize('use_candidate', [False, True])
def test_selected_export_rejects_completed_five_pass_parent(tmp_path, use_candidate):
    module = api()
    from cs781.io import sha256_file, write_json_atomic
    run = tmp_path/'run'
    latest = run/'checkpoints/latest.pt'
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b'owned latest')
    candidate = run/'candidates/milestone_209914685.pt'
    candidate.parent.mkdir()
    candidate.write_bytes(b'owned candidate')
    metadata = {'architecture': 'C', 'synthetic_smoke': False,
                'train_config': {'target_tokens': 209914685}, 'code_sha256': 'a'*64}
    state = {'step': 100, 'target_tokens': 209914685}
    write_json_atomic(run/'training_status.json', {'architecture': 'C', 'status': 'complete',
                      'target_tokens': 209914685, 'state': state, 'code_sha256': 'a'*64,
                      'checkpoint_sha256': sha256_file(latest)})
    with pytest.raises(ValueError, match='ten-pass'):
        module.require_selected_candidate(candidate if use_candidate else latest,
                                          {'state': state}, run, metadata)


def test_committed_selection_is_frozen_across_retries(tmp_path):
    module = api()
    assert hasattr(module, 'commit_selection'), 'Committed selection can still be overwritten'
    from cs781.io import sha256_file
    path = tmp_path/'selection.json'
    first = {'config_sha256': 'a'*64, 'data_manifest_sha256': 'b'*64,
             'selected': {'checkpoint': 'one', 'policy': 0, 'report': {'split': 'dev',
             'limited_sample': False, 'declared_ids': ['a'],
             'metrics': {'bertscore_f1': 0.8, 'rouge_l_f1': 0.2, 'bleu4': 0.02}}}}
    assert module.commit_selection(path, first, {'a'}) == first
    original_hash = sha256_file(path)
    changed = {**first, 'selected': {**first['selected'], 'checkpoint': 'two'}}
    assert module.commit_selection(path, changed, {'a'}) == first
    assert sha256_file(path) == original_hash


def test_restart_after_partial_export_skips_reselection_even_if_failed_policy_recovers(tmp_path, monkeypatch):
    module = api()
    from cs781.io import sha256_file, write_json_atomic
    from types import SimpleNamespace
    run, data = tmp_path/'run', tmp_path/'data'
    latest, milestone = run/'checkpoints/latest.pt', run/'candidates/milestone_209914685.pt'
    latest.parent.mkdir(parents=True)
    milestone.parent.mkdir()
    latest.write_bytes(b'owned final')
    milestone.write_bytes(b'owned milestone')
    data.mkdir()
    write_json_atomic(data/'dev.prompts.json', {'a': {'text': 'prompt '*512}})
    write_json_atomic(run/'training_status.json', {'status': 'complete', 'architecture': 'C',
                      'target_tokens': 419829370, 'state': {'target_tokens': 419829370},
                      'code_sha256': 'c'*64, 'checkpoint_sha256': sha256_file(latest)})
    write_json_atomic(run/'run.json', {'architecture': 'C', 'synthetic_smoke': False,
                      'train_config': {'target_tokens': 419829370}, 'code_sha256': 'c'*64})
    config = tmp_path/'config.json'
    write_json_atomic(config, {'model': {}, 'train': {}, 'decoding': [{}, {}]})
    monkeypatch.setattr(module, 'trainer_alive', lambda *args: False)
    monkeypatch.setattr(module, 'load_prepared', lambda *args: SimpleNamespace(root=data, manifest_sha256='b'*64))
    calls, phase = [], {'restart': False, 'selection_hash': None}
    def evaluate(data, checkpoint, config, policy, split, output, cache):
        calls.append((checkpoint.name, policy, split))
        if split == 'dev' and checkpoint == milestone and policy == 1:
            if not phase['restart']:
                raise RuntimeError('first attempt failed')
            score = 0.99  # This recovered policy must never reopen committed selection.
        else:
            score = 0.83 if checkpoint == latest and policy == 0 else 0.8
        return {'split': split, 'limited_sample': False, 'declared_ids': ['a'],
                'metrics': {'bertscore_f1': score, 'rouge_l_f1': 0.2, 'bleu4': 0.02}}
    monkeypatch.setattr(module, 'run_evaluation', evaluate)
    def export(data, checkpoint, config, policy, output, selection):
        digest = sha256_file(selection)
        if not phase['restart']:
            phase['selection_hash'] = digest
            raise RuntimeError('interrupted after official progress')
        assert digest == phase['selection_hash']
        assert checkpoint == latest and policy == 0
        return {'prediction_count': 1000}
    monkeypatch.setattr(module, 'export_selected', export)
    argv = ['--training-pid', '1', '--run-dir', str(run), '--data', str(data),
            '--config', str(config), '--scorer-cache', str(tmp_path/'cache')]
    assert module.main(argv) == 2
    calls.clear()
    phase['restart'] = True
    assert module.main(argv) == 0
    assert calls == [('latest.pt', 0, 'dev'), ('latest.pt', 0, 'confirmation')]

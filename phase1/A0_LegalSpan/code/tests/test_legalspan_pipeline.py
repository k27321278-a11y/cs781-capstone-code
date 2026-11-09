"""Catch selection on incomplete IDs, overclaims, mutable selection and bad ZIPs."""
import importlib.util
import pytest
from pathlib import Path


def api():
    assert importlib.util.find_spec('legalspan.pipeline') is not None, 'Automatic A evaluation/export not implemented'
    from legalspan.pipeline import choose, commit_selection, package
    return choose, commit_selection, package


def test_selection_requires_exact_dev_coverage_and_is_immutable(tmp_path):
    choose, commit, _ = api()
    expected = {'one', 'two'}
    candidates = [{'ids': ['one'], 'metrics': {'bertscore_f1': .99}, 'policy': 0},
                  {'ids': ['one', 'two'], 'metrics': {'bertscore_f1': .8}, 'policy': 1},
                  {'ids': ['one', 'two'], 'metrics': {'bertscore_f1': .82}, 'policy': 2}]
    assert choose(candidates, expected)['policy'] == 2
    path = tmp_path / 'selection.json'; commit(path, {'policy': 2})
    assert commit(path, {'policy': 2}) == {'policy': 2}
    with pytest.raises(ValueError, match='immutable'): commit(path, {'policy': 1})


def test_strict_zip_can_be_resumed_but_never_accepts_filler_or_missing_ids(tmp_path):
    _, _, package = api()
    prompts = {'1': 'prompt '*512, '2': 'prompt '*512}
    predictions = {'1': 'real '*128, '2': 'generated '*128}
    path = tmp_path / 'submission.zip'
    digest = package(predictions, prompts, path)
    assert package(predictions, prompts, path) == digest
    with pytest.raises(ValueError): package({'1': 'real '*127}, prompts, tmp_path / 'bad.zip')
    with pytest.raises(ValueError): package({'1': 'prompt '*128, '2': 'generated '*128}, prompts, tmp_path / 'echo.zip')


def test_substantive_gate_requires_all_three_complete_pilot_dev_reports():
    api()
    import legalspan.pipeline as module
    assert hasattr(module, 'screening_gate'), 'Approved pilot/control-dev screening gate missing'
    expected = {'1', '2'}
    def report(ids):
        return {'ids': ids, 'metrics': {'bertscore_f1': .7}, 'diagnostics': {'per_id': {k:{'repeated_4gram_fraction': .2} for k in ids}}}
    with pytest.raises(ValueError): module.screening_gate({'full':report(['1','2'])}, expected)
    reports = {k:report(['1','2']) for k in ('full','control','generic')}
    assert module.screening_gate(reports, expected)['screen_control_repetition_intervention']


def test_context_preflight_includes_every_public_id_and_no_prompt_truncation():
    api()
    import legalspan.pipeline as module
    from legalspan.bpe import OwnBPE
    assert hasattr(module, 'context_preflight'), 'Complete public-context preflight missing'
    tok=OwnBPE.train(['word '*512],280)
    prompts={str(i):'word '*512 for i in range(1000)}
    report=module.context_preflight(prompts,tok,8192,2048)
    assert report['count']==1000
    with pytest.raises(ValueError, match='context'): module.context_preflight(prompts,tok,64,2048)

import importlib.util
import json
from dataclasses import replace

import pytest
import torch


def api():
    assert importlib.util.find_spec('cs781.train_c') is not None, 'C trainer is not implemented'
    from cs781.train_c import train_c
    return train_c


def setup_data(catalog, tmp_path):
    from cs781.tokenizer import fit_tokenizer
    from cs781.prepare import prepare_data
    from cs781.models.c import RecurrentConfig
    path = tmp_path / 'tokenizer.json'
    fit_tokenizer(catalog, path, 512)
    data = prepare_data(catalog, path, tmp_path/'data', 781, max_seq_len=128)
    cfg = RecurrentConfig(vocab_size=512, max_seq_len=128, max_prompt_len=256,
                          n_encoder_layers=1, n_gru_layers=1, d_model=16,
                          n_heads=2, d_head=8, d_ff=32, gradient_checkpointing=True)
    return data, cfg


def test_c_resume_matches_full_run_and_candidates_are_sparse(catalog, tmp_path):
    train = api()
    from cs781.contracts import TrainConfig
    data, model_cfg = setup_data(catalog, tmp_path)
    cfg = TrainConfig(target_tokens=10000, device='cpu', grad_accum_steps=2,
                      save_every=1, log_every=1)
    expected = train(data, model_cfg, cfg, tmp_path/'full', stop_after_steps=6,
                     smoke_test=True, candidate_every=3, milestone_tokens=())
    train(data, model_cfg, cfg, tmp_path/'split', stop_after_steps=3,
          smoke_test=True, candidate_every=3, milestone_tokens=())
    actual = train(data, model_cfg, cfg, tmp_path/'split', resume=True,
                   stop_after_steps=6, smoke_test=True, candidate_every=3, milestone_tokens=())
    assert expected == actual
    a = torch.load(tmp_path/'full/checkpoints/latest.pt', weights_only=False)['model']
    b = torch.load(tmp_path/'split/checkpoints/latest.pt', weights_only=False)['model']
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=1e-7, rtol=1e-6)
    assert len(list((tmp_path/'full/candidates').glob('*.pt'))) == 2
    record = json.loads((tmp_path/'split/training_status.json').read_text())
    assert record['status'] == 'incomplete' and record['state']['step'] == 6
    from cs781.contracts import CheckpointMismatch
    with pytest.raises(CheckpointMismatch):
        train(data, model_cfg, replace(cfg, target_tokens=20000), tmp_path/'split',
              resume=True, smoke_test=True, candidate_every=3, milestone_tokens=())


def test_c_smoke_completion_and_milestone_are_labelled(catalog, tmp_path):
    train = api()
    from cs781.contracts import TrainConfig
    data, model_cfg = setup_data(catalog, tmp_path)
    cfg = TrainConfig(target_tokens=1, device='cpu', grad_accum_steps=1, save_every=10)
    state = train(data, model_cfg, cfg, tmp_path/'run', smoke_test=True,
                  candidate_every=2000, milestone_tokens=(1,))
    assert state.target_tokens >= 1
    record = json.loads((tmp_path/'run/training_status.json').read_text())
    assert record['architecture'] == 'C' and record['status'] == 'smoke_complete'
    assert len(list((tmp_path/'run/candidates').glob('*milestone*.pt'))) == 1
    with pytest.raises(ValueError, match='Synthetic'):
        train(data, model_cfg, cfg, tmp_path/'not-smoke')


def test_c_generation_respects_word_boundary_and_long_prompt_capacity():
    assert importlib.util.find_spec('cs781.generate_c') is not None, 'C generation is not implemented'
    from cs781.generate_c import generate_continuation_c
    from cs781.contracts import GenerationConfig, GenerationIncomplete, PromptTooLong
    from test_generate import Pieces
    from types import SimpleNamespace
    class Scripted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.cfg = SimpleNamespace(max_seq_len=512, max_prompt_len=1024)
            self.step = 0
        def forward(self, prompt, decoder, **kwargs):
            logits = torch.full((1, 1, 8), -100.)
            logits[:, :, 2] = 1000
            logits[:, :, 4+self.step%3] = 100
            self.step += 1
            return logits, None, {'steps': self.step}
    assert generate_continuation_c(Scripted(), Pieces(), 'prompt '*512,
                                   GenerationConfig(max_new_tokens=400)).split() == ['continuation']*128
    with pytest.raises(GenerationIncomplete):
        generate_continuation_c(Scripted(), Pieces(), 'prompt '*512, GenerationConfig(max_new_tokens=10))
    model = Scripted()
    model.cfg.max_prompt_len = 500
    with pytest.raises(PromptTooLong):
        generate_continuation_c(model, Pieces(), 'prompt '*512, GenerationConfig())


def test_candidate_and_milestone_survive_interruption_at_checkpoint_boundary(catalog, tmp_path, monkeypatch):
    train = api()
    import cs781.train_c as module
    from cs781.contracts import TrainConfig
    from cs781.train import IndexedExamples
    data, model_cfg = setup_data(catalog, tmp_path)
    dataset = IndexedExamples(data.root)
    milestone = sum(len(dataset.at(i, 781).suffix_ids)+1 for i in range(2))
    cfg = TrainConfig(target_tokens=10000, device='cpu', grad_accum_steps=1,
                      save_every=1, log_every=1)
    original = module._save_candidate
    def interrupt(path, model, metadata, state):
        if state.step == 2:
            raise RuntimeError('injected candidate interruption')
        return original(path, model, metadata, state)
    monkeypatch.setattr(module, '_save_candidate', interrupt)
    with pytest.raises(RuntimeError, match='injected'):
        train(data, model_cfg, cfg, tmp_path/'run', stop_after_steps=2, smoke_test=True,
              candidate_every=2, milestone_tokens=(milestone,))
    monkeypatch.setattr(module, '_save_candidate', original)
    state = train(data, model_cfg, cfg, tmp_path/'run', resume=True, stop_after_steps=2,
                  smoke_test=True, candidate_every=2, milestone_tokens=(milestone,))
    for name in ('step_0000002.pt', f'milestone_{milestone}.pt'):
        path = tmp_path/'run/candidates'/name
        assert path.exists(), 'Required snapshot was skipped after interrupted save'
        assert torch.load(path, weights_only=False)['state']['step'] == state.step == 2

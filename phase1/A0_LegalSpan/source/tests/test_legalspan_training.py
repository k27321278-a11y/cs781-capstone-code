"""Catch non-resumable optimizer/sampler state and wrong target accounting."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import torch
import pytest
from test_legalspan_data import fixture
from legalspan.data import prepare, MixedExamples
from legalspan.model import SpanConfig
from legalspan.bpe import OwnBPE
from cs781.contracts import TrainConfig


def api():
    assert importlib.util.find_spec('legalspan.train') is not None, 'New A trainer not implemented'
    from legalspan.train import train, batch
    return train, batch


def test_mixed_sampler_alternates_and_batch_boundary_is_exact(tmp_path):
    _, batch = api()
    db, parent, tok = fixture(tmp_path); root = tmp_path / 'data'
    prepare(db, parent, tok, root)
    sample = MixedExamples(root)
    assert [sample.at(i)['kind'] for i in range(6)] == ['word', 'token'] * 3
    x, y, roles, m = batch(sample.at(0), 'cpu')
    assert m == roles.shape[1]
    assert (y[:, :m-1] == -100).all()
    assert (y[:, m-1:] != -100).all()
    assert y[0, -1] == 2


def test_resume_matches_uninterrupted_and_rejects_changed_graph(tmp_path):
    train, _ = api()
    db, parent, tok = fixture(tmp_path); root = tmp_path / 'data'
    prepare(db, parent, tok, root)
    model = SpanConfig(vocab_size=OwnBPE.load(tok).get_vocab_size(), max_seq_len=8192, n_layers=2, d_model=16,
                       n_heads=2, d_head=8, d_ff=32, route_dim=8, route_layers=1,
                       gradient_checkpointing=True)
    cfg = TrainConfig(target_tokens=100000, grad_accum_steps=2, device='cpu', save_every=1, log_every=1)
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    train(root, model, cfg, whole, smoke=True, stop_after_steps=2)
    train(root, model, cfg, split, smoke=True, stop_after_steps=1)
    train(root, model, cfg, split, smoke=True, resume=True, stop_after_steps=2)
    a = torch.load(whole / 'checkpoints/latest.pt', weights_only=False)
    b = torch.load(split / 'checkpoints/latest.pt', weights_only=False)
    assert a['state'] == b['state']
    for key in a['model']: torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    with pytest.raises(Exception, match='metadata mismatch'):
        train(root, replace(model, use_roles=False), cfg, split, smoke=True, resume=True, stop_after_steps=3)


def test_due_selection_snapshot_exists_before_resume_checkpoint_advance(tmp_path, monkeypatch):
    train, _ = api()
    import legalspan.train as module
    db, parent, tok = fixture(tmp_path); root = tmp_path / 'data'; prepare(db, parent, tok, root)
    sample = MixedExamples(root)
    n = sum(len(sample.at(i)['suffix_ids'])+1 for i in range(2)); sample.handle.close()
    cfg = TrainConfig(target_tokens=2*n+1, grad_accum_steps=2, device='cpu', save_every=1, log_every=1)
    model = SpanConfig(vocab_size=OwnBPE.load(tok).get_vocab_size(), n_layers=2, d_model=16,
                       n_heads=2, d_head=8, d_ff=32, route_dim=8, route_layers=1)
    run = tmp_path / 'interrupted'; original_save = module.save_checkpoint
    def interrupt_after_actual_save(*args, **kwargs):
        original_save(*args, **kwargs)
        raise RuntimeError('injected interruption after latest publication')
    monkeypatch.setattr(module, 'save_checkpoint', interrupt_after_actual_save)
    with pytest.raises(RuntimeError, match='injected interruption'):
        train(root, model, cfg, run, smoke=True, stop_after_steps=1)
    assert (run / f'candidates/targets_{n}.pt').is_file()

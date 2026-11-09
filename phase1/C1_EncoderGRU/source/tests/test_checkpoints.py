import importlib
import random

import pytest
import torch


def api():
    try:
        return importlib.import_module("cs781.checkpoints")
    except ImportError as exc:
        pytest.fail(f"Checkpoint implementation missing: {exc}")


def test_checkpoint_rejects_different_data_or_tokenizer(tmp_path):
    c = api()
    from cs781.contracts import TrainState, CheckpointMismatch
    model = torch.nn.Linear(2,2)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path/"latest.pt"
    c.save_checkpoint(path,model,opt,TrainState(step=3),{"cursor":7},{"data":"one"})
    with pytest.raises(CheckpointMismatch):
        c.restore_checkpoint(path,model,opt,{"data":"two"})
    state,sampler = c.restore_checkpoint(path,model,opt,{"data":"one"})
    assert state.step == 3 and sampler["cursor"] == 7


def test_interrupted_checkpoint_preserves_previous(tmp_path,monkeypatch):
    c = api()
    from cs781.contracts import TrainState
    model = torch.nn.Linear(2,2)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path/"latest.pt"
    c.save_checkpoint(path,model,opt,TrainState(),{}, {})
    before = path.read_bytes()
    def interrupted(*args,**kwargs):
        raise OSError("interrupted write")
    monkeypatch.setattr(c.os,"replace",interrupted)
    with pytest.raises(OSError):
        c.save_checkpoint(path,model,opt,TrainState(step=1),{}, {})
    assert path.read_bytes() == before


def test_duplicate_run_lock_is_rejected(tmp_path):
    c = api()
    from cs781.contracts import RunAlreadyActive
    with c.run_lock(tmp_path):
        with pytest.raises(RunAlreadyActive):
            with c.run_lock(tmp_path):
                pass

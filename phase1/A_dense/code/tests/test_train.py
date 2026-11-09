import importlib
from dataclasses import replace

import pytest
import torch


def api():
    try:
        return importlib.import_module("cs781.train")
    except ImportError as exc:
        pytest.fail(f"Training implementation missing: {exc}")


def prepared(catalog,tmp_path):
    from cs781.tokenizer import fit_tokenizer
    from cs781.prepare import prepare_data
    path = tmp_path/"tokenizer.json"
    fit_tokenizer(catalog,path,512)
    return prepare_data(catalog,path,tmp_path/"data",781,max_seq_len=128)


def test_resume_matches_uninterrupted_training(catalog,tmp_path,tiny_config):
    t = api()
    from cs781.contracts import TrainConfig
    data = prepared(catalog,tmp_path)
    cfg = TrainConfig(target_tokens=10000,device="cpu",grad_accum_steps=2,save_every=1,log_every=1)
    expected = t.train_a(data,tiny_config,cfg,tmp_path/"full",False,stop_after_steps=6,smoke_test=True)
    t.train_a(data,tiny_config,cfg,tmp_path/"split",False,stop_after_steps=3,smoke_test=True)
    actual = t.train_a(data,tiny_config,cfg,tmp_path/"split",True,stop_after_steps=6,smoke_test=True)
    assert actual == expected
    a = torch.load(tmp_path/"full/checkpoints/latest.pt",weights_only=False)["model"]
    b = torch.load(tmp_path/"split/checkpoints/latest.pt",weights_only=False)["model"]
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=1e-7,rtol=1e-6)
    assert actual.input_tokens > actual.target_tokens > 0


def test_synthetic_training_requires_explicit_smoke_flag(catalog,tmp_path,tiny_config):
    t = api()
    from cs781.contracts import TrainConfig
    data = prepared(catalog,tmp_path)
    with pytest.raises(ValueError):
        t.train_a(data,tiny_config,TrainConfig(target_tokens=100,device="cpu"),tmp_path/"run",False)


def test_new_run_does_not_overwrite_existing_checkpoint(catalog,tmp_path,tiny_config):
    t = api()
    from cs781.contracts import TrainConfig
    data = prepared(catalog,tmp_path)
    run = tmp_path/"run"
    run.joinpath("checkpoints").mkdir(parents=True)
    sentinel = run/"checkpoints/latest.pt"
    sentinel.write_text("old checkpoint")
    with pytest.raises(FileExistsError):
        t.train_a(data,tiny_config,TrainConfig(target_tokens=100,device="cpu"),run,False,smoke_test=True)
    assert sentinel.read_text() == "old checkpoint"
